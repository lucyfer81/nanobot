import asyncio

import pytest

from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus


def _msg(channel: str, content: str, chat_id: str = "chat-1") -> InboundMessage:
    return InboundMessage(
        channel=channel,
        sender_id="user-1",
        chat_id=chat_id,
        content=content,
    )


@pytest.mark.asyncio
async def test_system_lane_has_higher_priority_than_main() -> None:
    bus = MessageBus()

    await bus.publish_inbound(_msg("telegram", "main-1"))
    await bus.publish_inbound(_msg("system", "system-1", chat_id="telegram:chat-1"))

    first = await bus.consume_inbound()
    second = await bus.consume_inbound()

    assert first.channel == "system"
    assert first.content == "system-1"
    assert second.channel == "telegram"
    assert second.content == "main-1"


@pytest.mark.asyncio
async def test_drop_policy_old_discards_oldest_when_lane_full() -> None:
    bus = MessageBus(
        lane_caps={"main": 2, "system": 2, "cron": 2},
        drop_policy="old",
    )

    await bus.publish_inbound(_msg("telegram", "A"))
    await bus.publish_inbound(_msg("telegram", "B"))
    await bus.publish_inbound(_msg("telegram", "C"))

    first = await bus.consume_inbound()
    second = await bus.consume_inbound()

    assert [first.content, second.content] == ["B", "C"]


@pytest.mark.asyncio
async def test_drop_policy_new_rejects_latest_message_when_lane_full() -> None:
    bus = MessageBus(
        lane_caps={"main": 2, "system": 2, "cron": 2},
        drop_policy="new",
    )

    accepted1 = await bus.publish_inbound(_msg("telegram", "A"))
    accepted2 = await bus.publish_inbound(_msg("telegram", "B"))
    accepted3 = await bus.publish_inbound(_msg("telegram", "C"))

    first = await bus.consume_inbound()
    second = await bus.consume_inbound()

    assert accepted1 is True
    assert accepted2 is True
    assert accepted3 is False
    assert [first.content, second.content] == ["A", "B"]


@pytest.mark.asyncio
async def test_wait_for_active_tasks_times_out_and_then_succeeds() -> None:
    bus = MessageBus()
    bus.mark_task_started()

    timed_out = await bus.wait_for_active_tasks(timeout_ms=10)
    assert timed_out is False

    waiter = asyncio.create_task(bus.wait_for_active_tasks(timeout_ms=200))
    await asyncio.sleep(0.01)
    bus.mark_task_done()

    assert await waiter is True
