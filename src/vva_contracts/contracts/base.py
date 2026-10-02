"""Shared building blocks for every request and report contract.

Why this module exists
----------------------
Three rules apply to every contract and are enforced here once:

1. ``extra="forbid"``: an unknown field is an error, never silently dropped.
   A typo like ``min_confidance`` must fail loudly, not fall back to a default.
2. ``strict=True``: no type coercion. ``"0.6"`` is not accepted for a float,
   ``"true"`` is not a boolean. Models write JSON; coercion would hide their
   mistakes.
3. ``frozen=True``: a validated report cannot be mutated afterwards, so what
   was validated is exactly what gets serialised.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from vva_contracts.errors import ContractViolation

SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"

Sha256 = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
RelPath = Annotated[str, StringConstraints(min_length=1, max_length=4096)]
Ratio = Annotated[float, Field(ge=0.0, le=1.0)]
NonNegFloat = Annotated[float, Field(ge=0.0)]
NonNegInt = Annotated[int, Field(ge=0)]
Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.-]{1,64}$")]


class StrictModel(BaseModel):
    """Base model: forbid extras, no coercion, immutable after validation."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class InputFingerprint(StrictModel):
    """Provenance of one input: which file, in which role, with which bytes."""

    role: Identifier
    path: RelPath
    sha256: Sha256


class ReportMeta(StrictModel):
    """Header present in every report (who produced it, from what, when)."""

    tool_version: str
    generated_at: datetime
    # Empty only for live sources (an RTSP stream has no bytes to hash); every
    # file-based task records at least one fingerprint.
    inputs: list[InputFingerprint]


def close(a: float, b: float) -> bool:
    """Float equality for invariant checks.

    Values are recomputed from the same integers with the same formulas, so
    they should match to the last bit; the tolerance only absorbs summation
    order differences (e.g. mean of a list computed in two places).
    """
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def require(condition: bool, message: str) -> None:
    """Raise a ContractViolation (exit code 2) when an invariant fails.

    Used inside model validators. ContractViolation is not a ValueError, so
    Pydantic lets it propagate unchanged and the message names the exact
    invariant that failed (schema errors arrive as ValidationError instead;
    the CLI maps both to exit code 2).
    """
    if not condition:
        raise ContractViolation(message)
