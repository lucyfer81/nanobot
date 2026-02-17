"""Automatic memory flush before context compaction pressure."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

import json_repair
from loguru import logger

from nanobot.agent.memory import (
    MemoryStore,
    guess_tldr,
    normalize_memory_type,
    normalize_tags,
)
from nanobot.providers.base import LLMProvider
from nanobot.session.manager import Session


class MemoryFlushTrigger:
    """Trigger a silent memory-saving turn when context is near compaction."""

    def __init__(
        self,
        workspace: Path,
        *,
        context_window_tokens: int = 96_000,
        enabled: bool = True,
        reserve_floor: int = 20_000,
        soft_threshold: int = 4_000,
        max_recent_messages: int = 24,
        max_message_chars: int = 400,
    ) -> None:
        self.workspace = workspace
        self.context_window_tokens = max(1, context_window_tokens)
        self.enabled = enabled
        self.reserve_floor = max(0, reserve_floor)
        self.soft_threshold = max(0, soft_threshold)
        self.max_recent_messages = max(1, max_recent_messages)
        self.max_message_chars = max(40, max_message_chars)

    def should_flush(self, current_tokens: int) -> bool:
        """Return True when token usage reaches the configured trigger point."""
        if not self.enabled:
            return False
        trigger_point = max(
            1,
            self.context_window_tokens - self.reserve_floor - self.soft_threshold,
        )
        return current_tokens >= trigger_point

    async def trigger_flush(
        self,
        *,
        provider: LLMProvider,
        model: str,
        session: Session,
        memory_store: MemoryStore,
        pending_user_message: str | None = None,
    ) -> bool:
        """
        Run one silent memory-saving turn and write results to memory files.

        Returns:
            True when at least one memory file was updated.
        """
        conversation = self._render_recent_conversation(
            session=session,
            pending_user_message=pending_user_message,
        )
        if not conversation:
            return False

        today = datetime.now().strftime("%Y-%m-%d")
        system_prompt = (
            "You extract durable memories from conversations. "
            "Return only NO_REPLY or valid JSON."
        )
        user_prompt = (
            "Session is nearing context compaction. Review this recent conversation and "
            "extract durable notes.\n\n"
            "Conversation:\n"
            f"{conversation}\n\n"
            "Return either:\n"
            "1) NO_REPLY (if nothing durable should be saved), OR\n"
            "2) JSON object with keys:\n"
            '- "memory_entry": object with type/tags/tldr/details\n'
            '- "daily_note": object with type/tags/tldr/details\n\n'
            "Rules:\n"
            "- type must be one of: decision, note, bug, idea, config.\n"
            "- Keep memory_entry focused on cross-session value.\n"
            "- Keep daily_note specific to today's progress.\n"
            "- Do not include markdown fences.\n"
        )

        try:
            response = await provider.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=model,
                max_tokens=600,
                temperature=0.1,
            )
        except Exception as exc:
            logger.warning("Memory flush: silent turn failed: {}", exc)
            return False

        content = (response.content or "").strip()
        parsed = self._parse_flush_payload(content)
        if parsed is None:
            return False

        memory_entry, daily_note = parsed
        wrote = False
        if memory_entry:
            memory_store.append_long_term_entry(
                memory_entry.get("details", ""),
                memory_type=str(memory_entry.get("type") or "note"),
                tags=memory_entry.get("tags"),
                tldr=str(memory_entry.get("tldr") or ""),
            )
            wrote = True
        if daily_note:
            memory_store.append_daily_note(
                today,
                str(daily_note.get("details") or ""),
                memory_type=str(daily_note.get("type") or "note"),
                tags=daily_note.get("tags"),
                tldr=str(daily_note.get("tldr") or ""),
            )
            wrote = True
        return wrote

    def _render_recent_conversation(
        self,
        *,
        session: Session,
        pending_user_message: str | None,
    ) -> str:
        """Render recent session messages into compact plain text for flush prompt."""
        lines: list[str] = []
        for msg in session.messages[-self.max_recent_messages :]:
            role = str(msg.get("role") or "unknown").upper()
            content = self._stringify(msg.get("content"))
            if not content:
                continue
            ts = str(msg.get("timestamp") or "")[:16]
            prefix = f"[{ts}] " if ts else ""
            lines.append(f"{prefix}{role}: {content[:self.max_message_chars]}")

        pending = (pending_user_message or "").strip()
        if pending:
            lines.append(f"USER: {pending[:self.max_message_chars]}")

        return "\n".join(lines)

    @staticmethod
    def _stringify(content: Any) -> str:
        """Stringify arbitrary message payloads into one compact line."""
        if content is None:
            return ""
        if isinstance(content, str):
            return " ".join(content.split())
        try:
            serialized = json.dumps(content, ensure_ascii=False, default=str)
        except Exception:
            serialized = str(content)
        return " ".join(serialized.split())

    def _parse_flush_payload(
        self,
        raw_text: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None] | None:
        """Parse NO_REPLY or JSON payload from the silent turn output."""
        if not raw_text:
            return None
        text = self._strip_markdown_fence(raw_text.strip())
        if not text:
            return None
        if text.lower().startswith("no_reply"):
            return None

        try:
            payload = json_repair.loads(text)
        except Exception:
            payload = None

        if isinstance(payload, dict):
            memory_entry = self._coerce_structured_entry(
                payload.get("memory_entry")
                or payload.get("memory")
                or payload.get("long_term"),
                default_type="decision",
            )
            daily_note = self._coerce_structured_entry(
                payload.get("daily_note")
                or payload.get("history_entry")
                or payload.get("note"),
                default_type="note",
            )
            if memory_entry or daily_note:
                return memory_entry, daily_note
            return None

        # Fallback: if JSON parsing failed but model returned plain text, keep it.
        fallback = self._coerce_structured_entry(text, default_type="note")
        return fallback, None

    @staticmethod
    def _strip_markdown_fence(text: str) -> str:
        """Strip outer markdown code fences if present."""
        if not text.startswith("```"):
            return text
        body = text.split("\n", 1)
        if len(body) == 2:
            text = body[1]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    @staticmethod
    def _coerce_text(value: Any) -> str:
        """Normalize model payload fields into strings."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            parts = [str(item).strip() for item in value if str(item).strip()]
            return "\n".join(parts).strip()
        return str(value).strip()

    def _coerce_structured_entry(self, value: Any, *, default_type: str) -> dict[str, Any] | None:
        """Normalize legacy/string/dict values into a structured memory entry."""
        if value is None:
            return None

        if isinstance(value, (str, list)):
            details = self._coerce_text(value)
            if not details:
                return None
            return {
                "type": normalize_memory_type(default_type),
                "tags": [],
                "tldr": guess_tldr(details),
                "details": details,
            }

        if not isinstance(value, dict):
            details = self._coerce_text(value)
            if not details:
                return None
            return {
                "type": normalize_memory_type(default_type),
                "tags": [],
                "tldr": guess_tldr(details),
                "details": details,
            }

        entry_type = normalize_memory_type(
            self._coerce_text(value.get("type")) or default_type,
        )
        tags = normalize_tags(value.get("tags"))
        tldr = self._coerce_text(value.get("tldr") or value.get("tl;dr") or value.get("summary"))
        details = self._coerce_text(
            value.get("details")
            or value.get("content")
            or value.get("memory_entry")
            or value.get("note")
            or value.get("daily_note")
        )

        if not details and tldr:
            details = tldr
        if not tldr and details:
            tldr = guess_tldr(details)
        if not details:
            return None

        return {
            "type": entry_type,
            "tags": tags,
            "tldr": tldr,
            "details": details,
        }
