"""Outcome-driven execution model for agent turns.

Replaces implicit while-loop exit conditions with explicit outcome states
that drive recovery, followup, and termination decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

RunOutcomeKind = Enum(
    "RunOutcomeKind",
    [
        "SUCCESS",  # Turn completed successfully
        "NEEDS_FOLLOWUP",  # More work remains (e.g., backlog)
        "RETRYABLE_ERROR",  # Transient failure (can retry)
        "FATAL_ERROR",  # Unrecoverable failure
        "NO_REPLY",  # No response generated
    ],
)


@dataclass
class RunOutcome:
    """Result of a single agent turn execution."""

    kind: RunOutcomeKind
    payload: str | None = None
    reason: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def is_terminal(self) -> bool:
        """Return True if this outcome should terminate the run loop."""
        return self.kind in {
            RunOutcomeKind.SUCCESS,
            RunOutcomeKind.FATAL_ERROR,
            RunOutcomeKind.NO_REPLY,
        }

    def is_recoverable(self) -> bool:
        """Return True if this outcome can be recovered with retries."""
        return self.kind == RunOutcomeKind.RETRYABLE_ERROR

    def requires_followup(self) -> bool:
        """Return True if this outcome indicates more work is pending."""
        return self.kind == RunOutcomeKind.NEEDS_FOLLOWUP


@dataclass
class RecoveryAction:
    """Action to take in response to a recoverable outcome."""

    should_retry: bool = False
    retry_with_compaction: bool = False
    fallback_message: str | None = None

    @staticmethod
    def retry_with_compacted_context() -> "RecoveryAction":
        return RecoveryAction(should_retry=True, retry_with_compaction=True)

    @staticmethod
    def retry_same() -> "RecoveryAction":
        return RecoveryAction(should_retry=True, retry_with_compaction=False)

    @staticmethod
    def stop_with_message(message: str) -> "RecoveryAction":
        return RecoveryAction(should_retry=False, fallback_message=message)

    @staticmethod
    def stop_no_action() -> "RecoveryAction":
        return RecoveryAction(should_retry=False, fallback_message=None)
