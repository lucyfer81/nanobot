import pytest

from nanobot.infra.error_types import NanobotRuntimeError
from nanobot.infra.retry import RetryPolicy, run_with_retry


@pytest.mark.asyncio
async def test_run_with_retry_retries_transient_then_succeeds() -> None:
    attempts = 0
    delays: list[float] = []

    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise NanobotRuntimeError(kind="transient_http", message="connection reset")
        return "ok"

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    result = await run_with_retry(
        operation,
        policy=RetryPolicy(max_attempts=4, base_delay_ms=100, jitter_ratio=0.0),
        sleep=fake_sleep,
    )

    assert result == "ok"
    assert attempts == 3
    assert delays == [0.1, 0.2]


@pytest.mark.asyncio
async def test_run_with_retry_honors_retry_after_override() -> None:
    attempts = 0
    delays: list[float] = []

    async def operation() -> None:
        nonlocal attempts
        attempts += 1
        raise NanobotRuntimeError(
            kind="rate_limit",
            message="429 retry_after=1.5",
            retry_after_seconds=1.5,
        )

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    with pytest.raises(NanobotRuntimeError):
        await run_with_retry(
            operation,
            policy=RetryPolicy(max_attempts=2, base_delay_ms=100, jitter_ratio=0.0),
            sleep=fake_sleep,
        )

    assert attempts == 2
    assert delays == [1.5]


@pytest.mark.asyncio
async def test_run_with_retry_does_not_retry_fatal() -> None:
    attempts = 0

    async def operation() -> None:
        nonlocal attempts
        attempts += 1
        raise NanobotRuntimeError(kind="fatal", message="boom")

    with pytest.raises(NanobotRuntimeError):
        await run_with_retry(operation, policy=RetryPolicy(max_attempts=5))

    assert attempts == 1
