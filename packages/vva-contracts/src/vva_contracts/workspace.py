"""Workspace path policy: every file a task touches must live inside the project.

Why this module exists
----------------------
Request paths are chosen by a language model. Without a policy, a request such
as ``{"frames": "../../home/carlos/.ssh/id_ed25519"}`` would make a task read
(and hash, and partially echo in error messages) an arbitrary file. The rules:

1. Request paths must be *relative* to the workspace (OpenCode's session dir).
2. After resolving symlinks, the real path must still be inside the workspace.
   ``Path.resolve()`` follows every symlink, so a link that points outside is
   caught even if its own location is inside.
3. Files discovered while walking a directory (dataset images, label files)
   go through the same check, because a dataset folder can contain symlinks.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal

from vva_contracts.errors import InputError


def resolve_in_workspace(
    workspace: Path,
    relative: str,
    *,
    kind: Literal["file", "dir"],
) -> Path:
    """Resolve a request path and enforce the workspace policy.

    Args:
        workspace: Absolute, already-resolved workspace directory.
        relative: Path exactly as written in the request.
        kind: Whether the path must be an existing regular file or directory.

    Returns:
        The resolved absolute path (symlinks followed).

    Raises:
        InputError: absolute path, escape outside the workspace, or missing
            file/directory of the expected kind.
    """
    # Reject absolute paths in both POSIX and Windows syntax. Checking only the
    # host flavour would let "C:\\x" through on Linux as a weird relative name.
    if PurePosixPath(relative).is_absolute() or PureWindowsPath(relative).is_absolute():
        raise InputError(f"path must be workspace-relative, got an absolute path: {relative!r}")

    resolved = (workspace / relative).resolve()
    ensure_inside(workspace, resolved, shown_as=relative)

    if kind == "file" and not resolved.is_file():
        raise InputError(f"expected an existing file: {relative!r}")
    if kind == "dir" and not resolved.is_dir():
        raise InputError(f"expected an existing directory: {relative!r}")
    return resolved


def ensure_inside(workspace: Path, candidate: Path, *, shown_as: str | None = None) -> None:
    """Raise InputError if ``candidate`` (after resolving) is outside ``workspace``."""
    real = candidate.resolve()
    if not real.is_relative_to(workspace):
        # Never print the resolved target: it may reveal where a symlink points.
        label = shown_as if shown_as is not None else candidate.name
        raise InputError(f"path escapes the workspace: {label!r}")


def display_path(workspace: Path, path: Path) -> str:
    """Workspace-relative POSIX string used in reports (stable across machines)."""
    return path.resolve().relative_to(workspace).as_posix()
