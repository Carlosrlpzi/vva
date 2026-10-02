"""Read JSONL files into row models, failing fast with line numbers.

Why this module exists
----------------------
Logs from a camera pipeline get truncated (power cut mid-write), duplicated
or hand-edited. Skipping bad lines would silently change the measured
numbers, so every malformed line stops the run (exit code 3) and the error
names the file and line. Fixing the input is the operator's decision.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import TypeVar

from pydantic import ValidationError

from vva_contracts.contracts.base import StrictModel
from vva_contracts.errors import ContractViolation, InvalidInput

RowT = TypeVar("RowT", bound=StrictModel)


def read_rows(path: Path, model: type[RowT]) -> Iterator[tuple[int, RowT]]:
    """Yield ``(line_number, row)`` for every non-blank line of ``path``.

    Lines are validated with ``model_validate_json`` so the strict JSON rules
    apply (a quoted number is a string, not a float).
    """
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue  # blank lines carry no data (trailing newline, etc.)
            try:
                yield line_no, model.model_validate_json(raw)
            except ValidationError as exc:
                first = exc.errors()[0]
                where = ".".join(str(p) for p in first["loc"]) or "<line>"
                raise InvalidInput(f"{path.name}:{line_no}: {where}: {first['msg']}") from exc
            except ContractViolation as exc:
                # Semantic row checks (e.g. unordered timestamps) are input
                # errors here: the *file* is wrong, not the request.
                raise InvalidInput(f"{path.name}:{line_no}: {exc}") from exc
            except json.JSONDecodeError as exc:  # pragma: no cover - pydantic wraps this
                raise InvalidInput(f"{path.name}:{line_no}: invalid JSON") from exc


def read_all(path: Path, model: type[RowT]) -> list[tuple[int, RowT]]:
    """Materialise ``read_rows``; raise if the file holds no rows at all."""
    rows = list(read_rows(path, model))
    if not rows:
        raise InvalidInput(f"{path.name}: file contains no rows")
    return rows
