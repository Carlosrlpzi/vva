"""Workspace path policy.

Why this module exists
----------------------
Request paths are chosen by a language model. The model must be able to name
files *inside* the project, and nothing else: no absolute paths, no ``..``
escapes, and no symlink that points outside the workspace. Enforcing this in
one function means every task gets the same rule and tests cover it once.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath

from vva_contracts.errors import InvalidInput, PolicyBlocked


def resolve_in_workspace(workspace: Path, relative: str, *, must_exist: bool = True) -> Path:
    """Resolve ``relative`` against ``workspace`` or raise.

    Args:
        workspace: The project directory. It is resolved once here so the
            containment test below compares canonical paths.
        relative: A workspace-relative path as written in the request.
        must_exist: When True a missing target is an input error (exit 3).

    Returns:
        The canonical absolute path, guaranteed to lie inside ``workspace``.

    Raises:
        PolicyBlocked: absolute path, ``..`` component, or symlink escape.
        InvalidInput: target missing while ``must_exist`` is True.

    """
    # Reject absolute paths in both POSIX and Windows syntax: the request may
    # have been written on a different OS than the one executing it.
    if PurePosixPath(relative).is_absolute() or PureWindowsPath(relative).is_absolute():
        raise PolicyBlocked(f"absolute paths are not allowed: {relative!r}")
    # A literal '..' is rejected even if it would resolve back inside: the rule
    # stays simple to audit ("no parent references at all").
    if ".." in PurePosixPath(relative.replace("\\", "/")).parts:
        raise PolicyBlocked(f"parent-directory references are not allowed: {relative!r}")

    root = workspace.resolve(strict=True)
    # resolve() follows symlinks, so the containment check below also catches
    # a symlink inside the workspace that points outside of it.
    candidate = (root / relative).resolve(strict=False)
    if not candidate.is_relative_to(root):
        raise PolicyBlocked(f"path resolves outside the workspace: {relative!r}")
    if must_exist and not candidate.exists():
        raise InvalidInput(f"path does not exist: {relative!r}")
    return candidate


def to_workspace_relative(workspace: Path, absolute: Path) -> str:
    """Return ``absolute`` as a POSIX path relative to ``workspace``.

    Reports store relative paths so they stay valid when the repository is
    cloned elsewhere (the laptop and the Raspberry Pi use different roots).
    """
    return absolute.resolve().relative_to(workspace.resolve()).as_posix()
