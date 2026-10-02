"""Small, explicitly-defined statistics helpers.

Why this module exists
----------------------
"p95 latency" is ambiguous: numpy alone offers nine percentile methods that
give different answers on small samples. Contracts must state *which*
definition produced a number, so every percentile in this package comes from
``percentile`` below, which fixes the method to ``linear`` (numpy's default,
Hyndman & Fan type 7) and records it in the reports.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from typing import Literal

import numpy as np

PERCENTILE_METHOD: Literal["linear"] = "linear"


def percentile(values: Sequence[float], q: float) -> float:
    """q-th percentile (0..100) with the documented interpolation method."""
    return float(np.percentile(np.asarray(values, dtype=np.float64), q, method=PERCENTILE_METHOD))


def safe_ratio(numerator: float, denominator: float) -> float:
    """Numerator / denominator, defined as 0.0 when the denominator is 0.

    The convention matters for validators: precision with no predictions is
    reported as 0.0 (not NaN, which JSON cannot represent), and the same
    function is used when the validator recomputes it, so they always agree.
    """
    return numerator / denominator if denominator else 0.0


def parse_rational(text: str) -> float | None:
    """Parse ffprobe rationals like '30000/1001' into a float (29.97...).

    ffprobe reports frame rates as exact fractions; '0/0' means "unknown"
    and is returned as None rather than raising a ZeroDivisionError.
    """
    try:
        value = Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None
    return float(value) if value > 0 else None
