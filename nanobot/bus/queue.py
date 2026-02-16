"""Async message queue for decoupled channel-agent communication."""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Literal

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage

LaneName = Literal["main", "system", "cron"]
DropPolicy = Literal["old", "new", "summarize"]

DEFAULT_LANE_CAPS: dict[LaneName, int] = {
    "main": 200,
    "system": 100,
    "cron": 100,
}

DEFAULT_LANE_PRIORITY: tuple[LaneName, ...] = ("system", "cron", "main")


class MessageBus:
    """
    Async message bus that decouples chat channels from the agent core.

    Inbound messages are separated into lanes with explicit queue caps.
    """

    def __init__(
        self,
        lane_caps: dict[LaneName, int] | None = None,
        drop_policy: DropPolicy = "old",
    ):
        self._lane_caps: dict[LaneName, int] = {
            lane: max(1, int((lane_caps or {}).get(lane, default_cap)))
            for lane, default_cap in DEFAULT_LANE_CAPS.items()
        }
        self._lane_priority = DEFAULT_LANE_PRIORITY
        if drop_policy not in {"old", "new", "summarize"}:
            raise ValueError(f"Unsupported drop policy: {drop_policy}")
        self._drop_policy: DropPolicy = drop_policy

        self._inbound_lanes: dict[LaneName, asyncio.Queue[InboundMessage]] = {
            lane: asyncio.Queue() for lane in DEFAULT_LANE_CAPS
        }
        self._inbound_event = asyncio.Event()
        self._lane_summarized_drops: dict[LaneName, int] = {
            lane: 0 for lane in DEFAULT_LANE_CAPS
        }

        self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue()
        self._outbound_subscribers: dict[str, list[Callable[[OutboundMessage], Awaitable[None]]]] = {}
        self._running = False

        self._active_tasks = 0
        self._active_tasks_event = asyncio.Event()
        self._active_tasks_event.set()

    @staticmethod
    def _infer_lane(msg: InboundMessage) -> LaneName:
        if msg.channel == "system":
            return "system"
        if msg.channel == "cron":
            return "cron"
        return "main"

    async def publish_inbound(self, msg: InboundMessage, lane: LaneName | None = None) -> bool:
        """Publish a message from a channel to the agent."""
        resolved_lane = lane or self._infer_lane(msg)
        queue = self._inbound_lanes[resolved_lane]
        cap = self._lane_caps[resolved_lane]
        message_to_enqueue = msg

        if queue.qsize() >= cap:
            if self._drop_policy == "new":
                logger.warning(
                    "Inbound queue full: lane={} cap={} drop_policy=new",
                    resolved_lane,
                    cap,
                )
                return False

            dropped = queue.get_nowait()
            if self._drop_policy == "summarize":
                self._lane_summarized_drops[resolved_lane] += 1
                dropped_count = self._lane_summarized_drops[resolved_lane]
                message_to_enqueue = InboundMessage(
                    channel=msg.channel,
                    sender_id=msg.sender_id,
                    chat_id=msg.chat_id,
                    content=(
                        f"[queue backpressure: summarized {dropped_count} "
                        f"message(s) in lane '{resolved_lane}']\n{msg.content}"
                    ),
                    timestamp=msg.timestamp,
                    media=msg.media,
                    metadata={**msg.metadata, "_queue_summarized": True, "_queue_drop_count": dropped_count},
                )
                logger.warning(
                    "Inbound queue full: lane={} cap={} drop_policy=summarize dropped={} chat={}",
                    resolved_lane,
                    cap,
                    dropped.chat_id,
                    msg.chat_id,
                )
            else:
                logger.warning(
                    "Inbound queue full: lane={} cap={} drop_policy=old dropped={} chat={}",
                    resolved_lane,
                    cap,
                    dropped.chat_id,
                    msg.chat_id,
                )
        elif self._drop_policy == "summarize":
            self._lane_summarized_drops[resolved_lane] = 0

        await queue.put(message_to_enqueue)
        self._inbound_event.set()
        return True

    def _try_consume_inbound_nowait(self) -> InboundMessage | None:
        for lane in self._lane_priority:
            queue = self._inbound_lanes[lane]
            if queue.empty():
                continue
            try:
                return queue.get_nowait()
            except asyncio.QueueEmpty:
                continue
        return None

    async def consume_inbound(self) -> InboundMessage:
        """Consume the next inbound message (blocks until available)."""
        while True:
            msg = self._try_consume_inbound_nowait()
            if msg is not None:
                return msg

            self._inbound_event.clear()
            msg = self._try_consume_inbound_nowait()
            if msg is not None:
                return msg
            await self._inbound_event.wait()

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        """Publish a response from the agent to channels."""
        await self.outbound.put(msg)

    async def consume_outbound(self) -> OutboundMessage:
        """Consume the next outbound message (blocks until available)."""
        return await self.outbound.get()

    def subscribe_outbound(
        self,
        channel: str,
        callback: Callable[[OutboundMessage], Awaitable[None]],
    ) -> None:
        """Subscribe to outbound messages for a specific channel."""
        if channel not in self._outbound_subscribers:
            self._outbound_subscribers[channel] = []
        self._outbound_subscribers[channel].append(callback)

    async def dispatch_outbound(self) -> None:
        """
        Dispatch outbound messages to subscribed channels.
        Run this as a background task.
        """
        self._running = True
        while self._running:
            try:
                msg = await asyncio.wait_for(self.outbound.get(), timeout=1.0)
                subscribers = self._outbound_subscribers.get(msg.channel, [])
                for callback in subscribers:
                    try:
                        await callback(msg)
                    except Exception as e:
                        logger.error(f"Error dispatching to {msg.channel}: {e}")
            except asyncio.TimeoutError:
                continue

    def mark_task_started(self) -> None:
        """Track a new active agent task."""
        self._active_tasks += 1
        self._active_tasks_event.clear()

    def mark_task_done(self) -> None:
        """Mark an active agent task as complete."""
        self._active_tasks = max(0, self._active_tasks - 1)
        if self._active_tasks == 0:
            self._active_tasks_event.set()

    async def wait_for_active_tasks(self, timeout_ms: int) -> bool:
        """Wait for active task count to reach zero within timeout."""
        if self._active_tasks == 0:
            return True
        timeout_seconds = max(0, timeout_ms) / 1000.0
        try:
            await asyncio.wait_for(self._active_tasks_event.wait(), timeout=timeout_seconds)
            return True
        except asyncio.TimeoutError:
            return False

    def stop(self) -> None:
        """Stop the dispatcher loop."""
        self._running = False

    def lane_size(self, lane: LaneName) -> int:
        """Number of pending inbound messages in one lane."""
        return self._inbound_lanes[lane].qsize()

    @property
    def inbound_size(self) -> int:
        """Number of pending inbound messages."""
        return sum(queue.qsize() for queue in self._inbound_lanes.values())

    @property
    def outbound_size(self) -> int:
        """Number of pending outbound messages."""
        return self.outbound.qsize()

    @property
    def active_task_count(self) -> int:
        """Number of active tasks being processed by the agent."""
        return self._active_tasks
