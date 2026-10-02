"""Command-line boundary: ``vva run``, ``vva validate`` and ``vva schemas``.

Why this module exists
----------------------
This is the *machine* acceptance boundary. Whatever the agent writes in chat,
automation should consume only what this CLI prints:

* ``vva run``: request JSON on stdin -> exactly one report JSON on stdout,
  exit 0. Any failure prints one error JSON and a non-zero exit code.
* ``vva validate``: re-check an existing report; with ``--workspace`` also
  recompute every input fingerprint (provenance, not just shape).
* ``vva schemas``: export JSON Schemas for inspection. They are generated from
  the Pydantic models, which remain the source of truth: a JSON Schema cannot
  express invariants like "precision = tp / (tp + fp)".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import BaseModel, ValidationError

from vva_contracts.contracts import logs
from vva_contracts.contracts.base import StrictModel
from vva_contracts.contracts.requests import REQUEST_ADAPTER
from vva_contracts.errors import ContractViolation, InvalidInput, VVAError, redact
from vva_contracts.fingerprint import sha256_path
from vva_contracts.paths import resolve_in_workspace
from vva_contracts.registry import TASKS

# A request is a few hundred bytes; 1 MiB is a generous ceiling that still
# stops a runaway producer from making the tool buffer unbounded input.
MAX_REQUEST_BYTES = 1 << 20


def _summarise(exc: ValidationError) -> str:
    """First few Pydantic errors as 'field.path: message' (bounded length)."""
    parts = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:5]]
    return "; ".join(parts)


def run(workspace: Path, raw: str) -> StrictModel:
    """Validate a request, dispatch it, and re-validate the report via JSON."""
    try:
        request = REQUEST_ADAPTER.validate_json(raw)
    except ValidationError as exc:
        raise ContractViolation(f"invalid request: {_summarise(exc)}") from exc
    spec = TASKS[request.task]
    report = spec.runner(request, workspace)
    # Round trip through JSON: the bytes we print must themselves pass the
    # strict contract, not just the in-memory object we built.
    try:
        return spec.report.model_validate_json(report.model_dump_json())
    except ValidationError as exc:
        raise ContractViolation(f"report failed its own contract: {_summarise(exc)}") from exc


def validate(task: str, file: Path, workspace: Path | None) -> StrictModel:
    """Validate a saved report; optionally verify its input fingerprints."""
    if task not in TASKS:
        raise ContractViolation(f"unknown task {task!r}; expected one of {sorted(TASKS)}")
    try:
        report = TASKS[task].report.model_validate_json(file.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise ContractViolation(f"report invalid: {_summarise(exc)}") from exc
    if workspace is not None:
        meta = getattr(report, "meta", None)
        for fp in getattr(meta, "inputs", []):
            actual = sha256_path(resolve_in_workspace(workspace, fp.path))
            if actual != fp.sha256:
                raise ContractViolation(f"provenance mismatch: {fp.role} ({fp.path}) changed since the report")
    return report


def export_schemas(output: Path) -> list[str]:
    """Write one JSON Schema per request, report and log-row model."""
    output.mkdir(parents=True, exist_ok=True)
    models: dict[str, type[BaseModel]] = {spec.report.__name__: spec.report for spec in TASKS.values()}
    for row in (logs.PredictionRow, logs.FrameRow, logs.EventRow, logs.TimingRow, logs.SystemRow):
        models[row.__name__] = row
    written = []
    for name, model in sorted(models.items()):
        path = output / f"{name}.schema.json"
        path.write_text(json.dumps(model.model_json_schema(), indent=2) + "\n", encoding="utf-8")
        written.append(path.name)
    request_path = output / "VVARequest.schema.json"
    request_path.write_text(json.dumps(REQUEST_ADAPTER.json_schema(), indent=2) + "\n", encoding="utf-8")
    return [*written, request_path.name]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vva", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="read a request on stdin, print one report")
    p_run.add_argument("--workspace", type=Path, default=Path.cwd())
    p_val = sub.add_parser("validate", help="validate a saved report")
    p_val.add_argument("--task", required=True)
    p_val.add_argument("--file", type=Path, required=True)
    p_val.add_argument("--workspace", type=Path, default=None, help="also verify input fingerprints")
    p_sch = sub.add_parser("schemas", help="export JSON Schemas")
    p_sch.add_argument("--output", type=Path, required=True)
    return parser


def _emit(document: object) -> None:
    """Print exactly one JSON document on one line."""
    sys.stdout.write(json.dumps(document, separators=(",", ":"), ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            raw = sys.stdin.read(MAX_REQUEST_BYTES + 1)
            if len(raw) > MAX_REQUEST_BYTES:
                raise InvalidInput("request exceeds 1 MiB")
            sys.stdout.write(run(args.workspace, raw).model_dump_json() + "\n")
        elif args.command == "validate":
            report = validate(args.task, args.file, args.workspace)
            _emit(
                {
                    "valid": True,
                    "contract_id": getattr(report, "contract_id", None),
                    "provenance_checked": args.workspace is not None,
                }
            )
        else:
            _emit({"written": export_schemas(args.output)})
    except VVAError as exc:
        _emit(exc.to_dict())
        return exc.exit_code
    except Exception as exc:
        _emit(
            {
                "error": {
                    "code": "internal_error",
                    "exit_code": 1,
                    "message": redact(f"{type(exc).__name__}: {exc}")[:500],
                }
            }
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
