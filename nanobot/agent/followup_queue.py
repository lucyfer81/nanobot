"""Per-session followup queue for active-session serialization."""

from __future__ import annotations

import asyncio
from collections import deque

from loguru import logger

from nanobot.bus.events import InboundMessage


class FollowupQueue:
    """
    Queue followup messages while a session is active.

    A session can have at most one active worker. New messages for that
    session are queued and drained after a debounce delay.
    """

    def __init__(self, debounce_ms: int = 800, max_pending_per_session: int = 200):
        self.debounce_ms = max(0, debounce_ms)
        self.max_pending_per_session = max(1, max_pending_per_session)

        self._active_sessions: set[str] = set()
        self._pending: dict[str, deque[InboundMessage]] = {}
        self._pending_ids: dict[str, set[str]] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _extract_message_id(msg: InboundMessage) -> str | None:
        if not msg.metadata:
            return None
        for key in ("message_id", "msg_id", "id", "event_id", "update_id"):
            value = msg.metadata.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return None

    async def enqueue_if_active(self, session_key: str, msg: InboundMessage) -> bool:
        """
        Queue message only if the session is already active.

        Returns:
            True:  message queued as followup (caller should not start new run).
            False: session was idle and is now claimed by caller.
        """

        message_id = self._extract_message_id(msg)
        async with self._lock:
            if session_key not in self._active_sessions:
                self._active_sessions.add(session_key)
                return False

            if message_id:
                seen_ids = self._pending_ids.setdefault(session_key, set())
                if message_id in seen_ids:
                    return True

            queue = self._pending.setdefault(session_key, deque())
            if len(queue) >= self.max_pending_per_session:
                dropped = queue.popleft()
                dropped_id = self._extract_message_id(dropped)
                if dropped_id:
                    self._pending_ids.setdefault(session_key, set()).discard(dropped_id)
                logger.warning(
                    "Followup queue full for session {}. Dropped oldest pending message.",
                    session_key,
                )

            queue.append(msg)
            if message_id:
                self._pending_ids.setdefault(session_key, set()).add(message_id)
            return True

    async def pop_next(self, session_key: str) -> InboundMessage | None:
        """Pop the next followup message after debounce."""
        async with self._lock:
            has_pending = bool(self._pending.get(session_key))
        if not has_pending:
            return None

        if self.debounce_ms > 0:
            await asyncio.sleep(self.debounce_ms / 1000.0)

        async with self._lock:
            queue = self._pending.get(session_key)
            if not queue:
                return None

            msg = queue.popleft()
            message_id = self._extract_message_id(msg)
            if message_id:
                self._pending_ids.setdefault(session_key, set()).discard(message_id)
            if not queue:
                self._pending.pop(session_key, None)
                if session_key in self._pending_ids and not self._pending_ids[session_key]:
                    self._pending_ids.pop(session_key, None)
            return msg

    async def deactivate_if_idle(self, session_key: str) -> bool:
        """Release active session marker only when there is no pending followup."""
        async with self._lock:
            if self._pending.get(session_key):
                return False
            self._active_sessions.discard(session_key)
            self._pending.pop(session_key, None)
            self._pending_ids.pop(session_key, None)
            return True

    async def pending_count(self, session_key: str) -> int:
        """Return pending followup count for one session."""
        async with self._lock:
            queue = self._pending.get(session_key)
            return len(queue) if queue else 0

    async def is_active(self, session_key: str) -> bool:
        """Return whether the session is currently active."""
        async with self._lock:
            return session_key in self._active_sessions
