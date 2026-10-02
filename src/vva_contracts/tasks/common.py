"""Helpers shared by every task runner.

Why this module exists
----------------------
Every report starts with the same provenance header (tool version, UTC time,
fingerprinted inputs). Building it in one place guarantees that ``vva
validate --workspace`` can recompute the fingerprints of *any* report.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from vva_contracts import __version__
from vva_contracts.contracts.base import InputFingerprint, ReportMeta
from vva_contracts.fingerprint import sha256_path
from vva_contracts.paths import to_workspace_relative


def build_meta(workspace: Path, inputs: list[tuple[str, Path]]) -> ReportMeta:
    """Fingerprint each (role, absolute_path) input and stamp the report."""
    fingerprints = [
        InputFingerprint(role=role, path=to_workspace_relative(workspace, path), sha256=sha256_path(path))
        for role, path in inputs
    ]
    return ReportMeta(tool_version=__version__, generated_at=datetime.now(UTC), inputs=fingerprints)
