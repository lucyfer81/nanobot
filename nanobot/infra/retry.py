"""Unified async retry helpers."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Awaitable, Callable, TypeVar

from nanobot.infra.error_types import ErrorInfo, classify_error, is_retryable_error_kind

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """Retry policy with exponential backoff and optional jitter."""

    max_attempts: int = 3
    base_delay_ms: int = 200
    max_delay_ms: int = 2_000
    backoff_multiplier: float = 2.0
    jitter_ratio: float = 0.15

    def compute_delay_seconds(
        self,
        attempt: int,
        retry_after_seconds: float | None = None,
        rng: Callable[[float, float], float] | None = None,
    ) -> float:
        """
        Compute sleep delay before the next attempt.

        Args:
            attempt: Current attempt index (1-based).
            retry_after_seconds: Optional provider-specified delay override.
            rng: Optional random function for tests.
        """

        if retry_after_seconds is not None and retry_after_seconds > 0:
            return retry_after_seconds

        capped_attempt = max(1, attempt)
        raw_delay_ms = self.base_delay_ms * (self.backoff_multiplier ** (capped_attempt - 1))
        delay_ms = min(float(self.max_delay_ms), raw_delay_ms)

        if self.jitter_ratio <= 0:
            return max(0.0, delay_ms / 1000.0)

        spread = delay_ms * self.jitter_ratio
        sampler = rng or random.uniform
        jittered_ms = sampler(delay_ms - spread, delay_ms + spread)
        return max(0.0, jittered_ms / 1000.0)


RetryCallback = Callable[[int, Exception, ErrorInfo, float], Awaitable[None] | None]


async def run_with_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy | None = None,
    classify: Callable[[Exception], ErrorInfo] | None = None,
    on_retry: RetryCallback | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """
    Run an async operation with retry.

    Retries only for retryable error kinds: transient_http/rate_limit/timeout.
    """

    resolved_policy = policy or RetryPolicy()
    classifier = classify or classify_error
    attempt = 1

    while True:
        try:
            return await operation()
        except Exception as exc:
            info = classifier(exc)
            can_retry = (
                is_retryable_error_kind(info.kind)
                and attempt < max(1, resolved_policy.max_attempts)
            )
            if not can_retry:
                raise

            delay_seconds = resolved_policy.compute_delay_seconds(
                attempt=attempt,
                retry_after_seconds=info.retry_after_seconds,
            )
            if on_retry:
                maybe_awaitable = on_retry(attempt, exc, info, delay_seconds)
                if asyncio.iscoroutine(maybe_awaitable):
                    await maybe_awaitable
            await sleep(delay_seconds)
            attempt += 1
