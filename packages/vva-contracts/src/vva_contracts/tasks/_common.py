"""Helpers shared by the task implementations."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from vva_contracts.workspace import resolve_in_workspace


def tool_version() -> str:
    """Installed package version, recorded in every report for traceability."""
    try:
        return f"vva-contracts {version('vva-contracts')}"
    except PackageNotFoundError:  # running from a source checkout without install
        return "vva-contracts (uninstalled source)"


def quantile(values: ArrayLike, q: float) -> float:
    """Linear-interpolation quantile (numpy default, Hyndman-Fan type 7)."""
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def resolve_data_yaml(workspace: Path, root: Path, data_yaml: str | None) -> Path:
    """Explicit ``data_yaml`` if given, else ``<dataset_root>/data.yaml``."""
    if data_yaml is not None:
        return resolve_in_workspace(workspace, data_yaml, kind="file")
    default = root / "data.yaml"
    relative = default.relative_to(workspace).as_posix()
    return resolve_in_workspace(workspace, relative, kind="file")
