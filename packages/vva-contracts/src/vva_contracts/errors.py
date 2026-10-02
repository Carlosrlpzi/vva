"""Typed failures and the process exit code each one maps to.

Why this module exists
----------------------
The OpenCode bridge (``.opencode/lib/vva_bridge.ts``) treats the process exit
code as the *only* signal of success. A report is accepted only when the CLI
exits with 0. Every other outcome must be distinguishable by the operator
without parsing free text, so each failure class owns a fixed exit code that
mirrors the original ``mlcode`` adapter:

    0  accepted output contract
    1  unexpected internal error (a bug in this package)
    2  output contract violation (the report contradicted its own invariants)
    3  missing or invalid input (bad path, malformed JSONL, unknown class...)
    4  execution policy blocked (an operator gate is not set)
    6  subprocess timeout (ffprobe did not finish in time)

Exit code 5 (provider failure) is intentionally unused: no task in this
package calls an LLM. Every number in a report is computed locally.
"""

from __future__ import annotations

from typing import ClassVar


class VVAError(Exception):
    """Base class. Subclasses only override the two class-level constants."""

    exit_code: ClassVar[int] = 1
    code: ClassVar[str] = "internal_error"


class ContractViolationError(VVAError):
    """A produced report failed its own Pydantic invariants.

    This should never happen in correct code: the task computed numbers that
    the contract, recomputing them independently, does not agree with.
    """

    exit_code: ClassVar[int] = 2
    code: ClassVar[str] = "contract_violation"


class InputError(VVAError):
    """The request or one of the files it points to is invalid."""

    exit_code: ClassVar[int] = 3
    code: ClassVar[str] = "invalid_input"


class PolicyError(VVAError):
    """The task needs an operator opt-in (environment gate) that is not set."""

    exit_code: ClassVar[int] = 4
    code: ClassVar[str] = "policy_blocked"


class TaskTimeoutError(VVAError):
    """An external process (ffprobe) exceeded its time budget."""

    exit_code: ClassVar[int] = 6
    code: ClassVar[str] = "timeout"
