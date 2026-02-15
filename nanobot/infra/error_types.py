"""Runtime error typing for retries and recovery decisions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

ErrorKind = Literal[
    "transient_http",
    "rate_limit",
    "timeout",
    "context_overflow",
    "session_ordering",
    "fatal",
]


@dataclass(frozen=True)
class ErrorInfo:
    """Normalized runtime error metadata."""

    kind: ErrorKind
    message: str
    retry_after_seconds: float | None = None


class NanobotRuntimeError(RuntimeError):
    """Runtime error carrying normalized error metadata."""

    def __init__(
        self,
        kind: ErrorKind,
        message: str,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.retry_after_seconds = retry_after_seconds


def _extract_retry_after_seconds(error_text: str) -> float | None:
    """Best-effort parse retry delay from error text."""
    match = re.search(r"retry[-_\s]?after[^0-9]*(\d+(?:\.\d+)?)", error_text, re.IGNORECASE)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(token in text for token in keywords)


def classify_error(error: BaseException | str) -> ErrorInfo:
    """
    Classify an exception/string into a normalized runtime error kind.

    The classifier is intentionally simple and keyword-based so it works
    across different provider/tool backends.
    """

    if isinstance(error, NanobotRuntimeError):
        return ErrorInfo(
            kind=error.kind,
            message=str(error),
            retry_after_seconds=error.retry_after_seconds,
        )

    text = str(error)
    lowered = text.lower()
    retry_after = _extract_retry_after_seconds(text)

    if isinstance(error, TimeoutError) or _contains_any(
        lowered,
        ("timeout", "timed out", "deadline exceeded"),
    ):
        return ErrorInfo(kind="timeout", message=text, retry_after_seconds=retry_after)

    if _contains_any(
        lowered,
        ("rate limit", "rate_limit", "too many requests", " 429", "(429)", "status 429"),
    ):
        return ErrorInfo(kind="rate_limit", message=text, retry_after_seconds=retry_after)

    if _contains_any(
        lowered,
        (
            "context length",
            "context window",
            "token limit",
            "too many tokens",
            "maximum context",
        ),
    ):
        return ErrorInfo(kind="context_overflow", message=text, retry_after_seconds=retry_after)

    if _contains_any(
        lowered,
        (
            "session ordering",
            "out of order",
            "concurrent session",
            "active session",
        ),
    ):
        return ErrorInfo(kind="session_ordering", message=text, retry_after_seconds=retry_after)

    if _contains_any(
        lowered,
        (
            "connection reset",
            "temporarily unavailable",
            "service unavailable",
            "bad gateway",
            "gateway timeout",
            "network",
            "connection aborted",
            "connection error",
            "server error",
            "status 500",
            "status 502",
            "status 503",
            "status 504",
            "econnreset",
        ),
    ):
        return ErrorInfo(kind="transient_http", message=text, retry_after_seconds=retry_after)

    return ErrorInfo(kind="fatal", message=text, retry_after_seconds=retry_after)


def is_retryable_error_kind(kind: ErrorKind) -> bool:
    """Return whether this error kind should be retried by default."""
    return kind in {"transient_http", "rate_limit", "timeout"}
