"""Shared building blocks for every request and report contract.

Design rules applied to all contracts
-------------------------------------
* ``extra="forbid"``: an unknown field is an error, never silently ignored.
  A typo such as ``min_confidance`` must fail, not fall back to a default.
* ``strict=True``: no coercion. ``"0.6"`` is not a float and ``1`` is not a
  bool. A model that writes the wrong type gets an input error.
* ``allow_inf_nan=False``: NaN/inf cannot appear in a report. A metric that is
  undefined (e.g. precision with zero predictions) is ``null`` instead, and the
  contract states when ``null`` is allowed.
* ``frozen=True``: a validated report cannot be mutated afterwards, so what was
  validated is exactly what gets serialized.

Each report also re-derives its own verdict and flags from its evidence fields
(see the ``model_validator`` methods in each contract module). A report whose
verdict does not follow from its numbers is rejected, so nobody, human or
model, can hand-edit a FAIL into a PASS and still pass validation.
"""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"

# Absolute tolerance used when a validator recomputes a float. Reports are
# produced by the same float64 arithmetic, so 1e-9 only absorbs the error of
# summing in a different order; it is far below any meaningful metric change.
FLOAT_TOLERANCE = 1e-9

Sha256 = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
CameraId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")]
ClassName = Annotated[str, StringConstraints(min_length=1, max_length=64)]
Verdict = Literal["PASS", "WARN", "FAIL"]


class StrictModel(BaseModel):
    """Base class for every contract in this package (see module docstring)."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class InputFingerprint(StrictModel):
    """Which exact bytes produced a report: the audit trail of every input.

    If someone later asks "was this mAP computed on the current predictions
    file?", the answer is a hash comparison, not a recollection.
    """

    role: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    path: Annotated[str, StringConstraints(min_length=1, max_length=4096)]
    sha256: Sha256
    size_bytes: int = Field(ge=0)


def is_close(actual: float, expected: float) -> bool:
    """Float equality used by every recomputation check in the contracts."""
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=FLOAT_TOLERANCE)


def require_close(name: str, actual: float | None, expected: float | None) -> None:
    """Raise ValueError (turned into a ValidationError by Pydantic) on mismatch.

    ``None`` must match ``None`` exactly: an undefined metric reported as a
    number, or a defined one reported as null, are both contract violations.
    """
    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError(f"{name}: expected {expected!r}, report says {actual!r}")
        return
    if not is_close(actual, expected):
        raise ValueError(f"{name}: recomputed {expected!r}, report says {actual!r}")


def require_ordered(name: str, values: list[float]) -> None:
    """Quantiles must be non-decreasing (p50 <= p90 <= p99 <= max)."""
    for lower, upper in pairwise(values):
        if lower > upper + FLOAT_TOLERANCE:
            raise ValueError(f"{name}: values must be non-decreasing, got {values!r}")


def safe_ratio(numerator: float, denominator: float) -> float | None:
    """``numerator / denominator`` or ``None`` when the ratio is undefined."""
    return None if denominator == 0 else numerator / denominator
