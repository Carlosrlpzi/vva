"""Command-line entry point: the strict machine boundary.

``vva-contract run --workspace DIR``   read ONE request JSON from stdin, write
                                       ONE report JSON to stdout, exit 0.
``vva-contract schemas --output DIR``  export JSON schemas of all contracts.
``vva-contract validate --contract NAME --file F``  re-check a saved report.

On failure, stdout carries ``{"error": {"code", "message", "exit_code"}}`` and
the process exits with the code from ``errors.py``. Nothing else is ever
written to stdout, so a consumer can parse it without heuristics.

The report is validated twice: once when the task builds it (Pydantic
model construction) and again here by round-tripping through JSON. The second
pass guarantees that what is *serialized* satisfies the contract, not just
the in-memory object (for example, a float that does not survive JSON).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import BaseModel, ValidationError

from vva_contracts.contracts import INPUT_FORMATS, REPORTS
from vva_contracts.contracts.requests import VVARequest
from vva_contracts.errors import ContractViolationError, InputError, VVAError
from vva_contracts.tasks import HANDLERS

MAX_STDIN_BYTES = 1 << 20  # a request is small; anything bigger is a mistake


def _emit_error(error: VVAError) -> int:
    payload = {"error": {"code": error.code, "message": str(error), "exit_code": error.exit_code}}
    sys.stdout.write(json.dumps(payload) + "\n")
    return error.exit_code


def _validation_message(exc: ValidationError) -> str:
    # include_input=False: never echo request values back (they may be paths
    # or other data the operator considers private).
    errs = exc.errors(include_url=False, include_input=False)
    return "; ".join(f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}" for e in errs[:5])


def run_request(raw: str, workspace: Path) -> BaseModel:
    """Parse, dispatch and double-validate one request. Raises VVAError."""
    try:
        request = VVARequest.model_validate_json(raw).root
    except ValidationError as exc:
        raise InputError(f"invalid request: {_validation_message(exc)}") from exc

    handler = HANDLERS[request.task]
    try:
        report = handler(request, workspace)
        # Second validation pass on the serialized form (see module docstring).
        return type(report).model_validate_json(report.model_dump_json())
    except ValidationError as exc:
        # A report that fails its own invariants is a bug in this package,
        # never something to "fix up" downstream.
        raise ContractViolationError(f"report failed its contract: {_validation_message(exc)}") from exc


def _cmd_run(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    if not workspace.is_dir():
        return _emit_error(InputError("workspace is not a directory"))
    raw = sys.stdin.read(MAX_STDIN_BYTES + 1)
    if len(raw) > MAX_STDIN_BYTES:
        return _emit_error(InputError("request larger than 1 MiB"))
    try:
        report = run_request(raw, workspace)
    except VVAError as exc:
        return _emit_error(exc)
    sys.stdout.write(report.model_dump_json() + "\n")
    return 0


def _cmd_schemas(args: argparse.Namespace) -> int:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for name, model in {**REPORTS, **INPUT_FORMATS}.items():
        schema = model.model_json_schema()
        (output / f"{name}.schema.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    model = REPORTS.get(args.contract)
    if model is None:
        return _emit_error(InputError(f"unknown contract {args.contract!r}; choose from {sorted(REPORTS)}"))
    try:
        model.model_validate_json(Path(args.file).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        return _emit_error(InputError(f"cannot read {args.file}: {exc.__class__.__name__}"))
    except ValidationError as exc:
        return _emit_error(ContractViolationError(_validation_message(exc)))
    # Structure and invariants hold. This does NOT prove the numbers were
    # measured: provenance comes from the inputs' fingerprints, not from here.
    sys.stdout.write(json.dumps({"valid": True, "contract": args.contract}) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vva-contract", description="Verifiable audit tasks for the VVA project")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="execute one request read from stdin")
    run.add_argument("--workspace", required=True)
    run.set_defaults(func=_cmd_run)

    schemas = sub.add_parser("schemas", help="export JSON schemas")
    schemas.add_argument("--output", required=True)
    schemas.set_defaults(func=_cmd_schemas)

    validate = sub.add_parser("validate", help="re-check a saved report")
    validate.add_argument("--contract", required=True)
    validate.add_argument("--file", required=True)
    validate.set_defaults(func=_cmd_validate)

    args = parser.parse_args(argv)
    try:
        code: int = args.func(args)
    except VVAError as exc:
        code = _emit_error(exc)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
