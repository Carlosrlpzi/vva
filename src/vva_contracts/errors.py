"""Typed errors and their process exit codes.

Why this module exists
----------------------
The OpenCode bridge (and any automation) must be able to tell *why* a run did
not produce an accepted artifact without parsing free text. Each failure class
therefore maps to one stable exit code, and the CLI prints a small JSON error
document instead of a traceback.

Exit-code table (also documented in README.md):

    0  accepted output contract
    2  request or output contract violation (schema or semantic invariant)
    3  missing or invalid input data (file absent, malformed JSONL line, ...)
    4  execution policy blocked (path escape, env var outside the allowlist)
    6  external subprocess timed out (e.g. ffprobe on a dead RTSP stream)
"""

from __future__ import annotations

import re

# Credentials embedded in URLs look like scheme://user:password@host.
# The regex keeps the scheme so the message stays useful, but drops the secret.
_URL_CREDENTIALS = re.compile(r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*://)[^/@\s]+@")


def redact(text: str) -> str:
    """Remove ``user:password@`` fragments from any URL inside ``text``.

    Every message that can reach stdout passes through here, because an
    ffprobe error typically echoes the full RTSP URL, password included.
    """
    return _URL_CREDENTIALS.sub(r"\g<scheme>***@", text)


class VVAError(Exception):
    """Base class: carries a machine-readable code and an exit status."""

    exit_code: int = 1
    code: str = "internal_error"

    def __init__(self, message: str) -> None:
        # Redact at construction time so no later code path can leak the
        # original message by accident (e.g. through repr() or logging).
        super().__init__(redact(message))

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-serialisable error document printed by the CLI."""
        return {"error": {"code": self.code, "exit_code": self.exit_code, "message": str(self)}}


class ContractViolation(VVAError):
    """The request or the produced report broke a schema or an invariant."""

    exit_code = 2
    code = "contract_violation"


class InvalidInput(VVAError):
    """Input data is missing or malformed; the caller must fix the input."""

    exit_code = 3
    code = "invalid_input"


class PolicyBlocked(VVAError):
    """The request is well-formed but not allowed (e.g. path outside workspace)."""

    exit_code = 4
    code = "policy_blocked"


class SubprocessTimeout(VVAError):
    """An external tool did not finish in time; no artifact is accepted."""

    exit_code = 6
    code = "subprocess_timeout"
