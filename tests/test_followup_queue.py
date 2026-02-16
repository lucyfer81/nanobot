import pytest

from nanobot.agent.followup_queue import FollowupQueue
from nanobot.bus.events import InboundMessage


def _msg(content: str, message_id: str | None = None) -> InboundMessage:
    metadata = {"message_id": message_id} if message_id else {}
    return InboundMessage(
        channel="telegram",
        sender_id="u1",
        chat_id="chat-1",
        content=content,
        metadata=metadata,
    )


@pytest.mark.asyncio
async def test_followup_queue_claims_idle_session_and_queues_later_messages() -> None:
    queue = FollowupQueue(debounce_ms=0)
    key = "telegram:chat-1"

    first_is_followup = await queue.enqueue_if_active(key, _msg("first", "m1"))
    second_is_followup = await queue.enqueue_if_active(key, _msg("second", "m2"))

    assert first_is_followup is False
    assert second_is_followup is True
    assert await queue.pending_count(key) == 1

    next_msg = await queue.pop_next(key)
    assert next_msg is not None
    assert next_msg.content == "second"
    assert await queue.pending_count(key) == 0


@pytest.mark.asyncio
async def test_followup_queue_dedupes_by_message_id_when_available() -> None:
    queue = FollowupQueue(debounce_ms=0)
    key = "telegram:chat-1"

    await queue.enqueue_if_active(key, _msg("first", "m1"))  # claim active
    queued_1 = await queue.enqueue_if_active(key, _msg("followup-a", "m2"))
    queued_2 = await queue.enqueue_if_active(key, _msg("followup-duplicate", "m2"))

    assert queued_1 is True
    assert queued_2 is True
    assert await queue.pending_count(key) == 1


@pytest.mark.asyncio
async def test_deactivate_if_idle_keeps_active_when_pending_exists() -> None:
    queue = FollowupQueue(debounce_ms=0)
    key = "telegram:chat-1"

    await queue.enqueue_if_active(key, _msg("first", "m1"))
    assert await queue.deactivate_if_idle(key) is True
    assert await queue.is_active(key) is False

    await queue.enqueue_if_active(key, _msg("first-again", "m2"))
    await queue.enqueue_if_active(key, _msg("followup", "m3"))

    assert await queue.deactivate_if_idle(key) is False
    assert await queue.is_active(key) is True

    _ = await queue.pop_next(key)
    assert await queue.deactivate_if_idle(key) is True
    assert await queue.is_active(key) is False
