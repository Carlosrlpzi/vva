"""File reading and hashing shared by all tasks.

Two rules make reports traceable:
1. Every input file is fingerprinted (SHA-256 + size) into the report.
2. Every JSONL line is validated individually and errors cite ``file:line``,
   so a malformed log points at the exact record to fix.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from vva_contracts.contracts.base import InputFingerprint
from vva_contracts.errors import InputError
from vva_contracts.workspace import display_path

M = TypeVar("M", bound=BaseModel)
_CHUNK_BYTES = 1 << 20  # 1 MiB reads keep memory flat for multi-GB logs


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(workspace: Path, role: str, path: Path) -> InputFingerprint:
    return InputFingerprint(
        role=role,
        path=display_path(workspace, path),
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def manifest_digest(workspace: Path, files: Iterable[Path]) -> tuple[str, int]:
    """Hash of a file set: SHA-256 over sorted ``relative_path<TAB>sha256`` lines.

    Sorting makes the digest independent of directory listing order, so the
    same dataset produces the same digest on the laptop and on the Pi.

    Returns:
        ``(hex_digest, total_bytes)``.
    """
    lines: list[str] = []
    total = 0
    for path in files:
        lines.append(f"{display_path(workspace, path)}\t{sha256_file(path)}")
        total += path.stat().st_size
    digest = hashlib.sha256("\n".join(sorted(lines)).encode("utf-8")).hexdigest()
    return digest, total


def _first_error(exc: ValidationError) -> str:
    err = exc.errors(include_url=False, include_input=False)[0]
    location = ".".join(str(part) for part in err["loc"]) or "<root>"
    return f"{location}: {err['msg']}"


def iter_jsonl(path: Path, model: type[M], *, shown_as: str) -> Iterator[tuple[int, M]]:
    """Yield ``(line_number, record)``; blank lines are skipped.

    Raises:
        InputError: on the first invalid line, citing ``shown_as:line``.
    """
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                yield lineno, model.model_validate_json(line)
            except ValidationError as exc:
                # include_input=False above: never echo the offending record,
                # which may contain data the operator did not mean to share.
                raise InputError(f"{shown_as}:{lineno}: {_first_error(exc)}") from exc


def read_json(path: Path, model: type[M], *, shown_as: str) -> M:
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise InputError(f"{shown_as}: {_first_error(exc)}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputError(f"{shown_as}: not valid UTF-8 JSON") from exc
