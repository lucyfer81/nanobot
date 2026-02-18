"""Context builder for assembling agent prompts."""

from __future__ import annotations

import base64
import json
import mimetypes
import platform
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Sequence

from loguru import logger

from nanobot.agent.memory import MemoryStore
from nanobot.agent.memory_retrieval import MemoryRetriever
from nanobot.agent.skills import SkillsLoader

DEFAULT_EXPLICIT_ANCHORS = [
    "之前",
    "上次",
    "记得",
    "回顾",
    "复盘",
    "我们聊过",
    "你说过",
    "我说过",
    "按之前",
    "照旧",
    "沿用",
    "查一下记录",
    "as before",
    "you said",
    "i said",
]
DEFAULT_IMPLICIT_ANCHORS = [
    "具体日期",
    "端口",
    "参数",
    "命令",
    "决策原因",
    "不要冲突",
    "日期",
    "配置",
    "constraint",
    "consistency",
]
DEFAULT_UNCERTAIN_MARKERS = [
    "可能",
    "大概",
    "我猜",
    "不确定",
    "记不清",
    "maybe",
    "not sure",
    "i guess",
]


class RetrievalTier(IntEnum):
    """Memory retrieval tier."""

    OFF = 0
    LIGHT = 1
    HEAVY = 2


@dataclass
class RetrievalRunResult:
    """One retrieval run result consumed by ContextBuilder and AgentLoop."""

    tier: RetrievalTier
    probe_score: float = 0.0
    hits: list[dict[str, Any]] = field(default_factory=list)
    injected_context: str = ""


def _contains_any(text: str, anchors: Sequence[str]) -> bool:
    lower = text.lower()
    for anchor in anchors:
        probe = anchor.strip()
        if not probe:
            continue
        if probe.lower() in lower:
            return True
    return False


def choose_retrieval_tier(
    user_message: str,
    recent_turns: list[str],
    *,
    explicit_anchors: Sequence[str] | None = None,
    implicit_anchors: Sequence[str] | None = None,
) -> RetrievalTier:
    """Choose retrieval tier using explicit/implicit recall signals."""
    explicit = list(explicit_anchors or DEFAULT_EXPLICIT_ANCHORS)
    implicit = list(implicit_anchors or DEFAULT_IMPLICIT_ANCHORS)
    haystack = "\n".join([user_message, *recent_turns[-3:]])
    if _contains_any(haystack, explicit):
        return RetrievalTier.HEAVY
    if _contains_any(haystack, implicit):
        return RetrievalTier.LIGHT
    return RetrievalTier.OFF


def maybe_upgrade_after_draft(
    draft: str,
    current_tier: RetrievalTier,
    *,
    uncertain_markers: Sequence[str] | None = None,
) -> RetrievalTier:
    """Upgrade retrieval tier when the draft answer indicates uncertainty."""
    if current_tier >= RetrievalTier.HEAVY:
        return current_tier

    markers = list(uncertain_markers or DEFAULT_UNCERTAIN_MARKERS)
    text = (draft or "").strip()
    if text and _contains_any(text, markers):
        return RetrievalTier(current_tier + 1)
    return current_tier


class MemoryRetrievalRouter:
    """Three-tier retrieval router with OFF-tier probe support."""

    def __init__(
        self,
        retriever: MemoryRetriever | None,
        *,
        enabled: bool = True,
        enable_probe_on_off: bool = True,
        light_top_k: int = 3,
        heavy_top_k: int = 8,
        heavy_max_snippet_chars: int = 360,
        inject_budget_chars: int = 1800,
        explicit_anchors: Sequence[str] | None = None,
        implicit_anchors: Sequence[str] | None = None,
        uncertain_markers: Sequence[str] | None = None,
    ) -> None:
        self.retriever = retriever
        self.enabled = enabled and retriever is not None
        self.enable_probe_on_off = enable_probe_on_off
        self.light_top_k = max(1, light_top_k)
        self.heavy_top_k = max(1, heavy_top_k)
        self.heavy_max_snippet_chars = max(120, heavy_max_snippet_chars)
        self.inject_budget_chars = max(200, inject_budget_chars)
        self.explicit_anchors = list(explicit_anchors or DEFAULT_EXPLICIT_ANCHORS)
        self.implicit_anchors = list(implicit_anchors or DEFAULT_IMPLICIT_ANCHORS)
        self.uncertain_markers = list(uncertain_markers or DEFAULT_UNCERTAIN_MARKERS)

    def run(
        self,
        *,
        query: str,
        recent_turns: list[str],
        tier: RetrievalTier | None = None,
    ) -> RetrievalRunResult:
        """Run retrieval for one user query and return optional injected context."""
        if not self.enabled or self.retriever is None:
            return RetrievalRunResult(tier=RetrievalTier.OFF)

        selected_tier = (
            choose_retrieval_tier(
                query,
                recent_turns,
                explicit_anchors=self.explicit_anchors,
                implicit_anchors=self.implicit_anchors,
            )
            if tier is None
            else tier
        )
        probe_score = 0.0
        if self.enable_probe_on_off:
            try:
                probe_score = max(0.0, float(self.retriever.probe(query)))
            except Exception as exc:
                logger.debug("Memory probe failed: {}", exc)

        if selected_tier == RetrievalTier.OFF:
            return RetrievalRunResult(
                tier=selected_tier,
                probe_score=probe_score,
                hits=[],
                injected_context="",
            )

        try:
            if selected_tier == RetrievalTier.LIGHT:
                hits = self.retriever.search_light(query, top_k=self.light_top_k)
            else:
                hits = self.retriever.search_heavy(
                    query,
                    top_k=self.heavy_top_k,
                    max_snippet_chars=self.heavy_max_snippet_chars,
                )
        except Exception as exc:
            logger.warning("Memory retrieval failed, continue without injection: {}", exc)
            return RetrievalRunResult(
                tier=selected_tier,
                probe_score=probe_score,
                hits=[],
                injected_context="",
            )

        injected = self._format_injected_context(hits, selected_tier)
        return RetrievalRunResult(
            tier=selected_tier,
            probe_score=probe_score,
            hits=hits,
            injected_context=injected,
        )

    def maybe_upgrade_after_draft(self, draft: str, current_tier: RetrievalTier) -> RetrievalTier:
        """Upgrade tier when draft answer appears uncertain."""
        return maybe_upgrade_after_draft(
            draft,
            current_tier,
            uncertain_markers=self.uncertain_markers,
        )

    def _format_injected_context(
        self,
        rows: list[dict[str, Any]],
        tier: RetrievalTier,
    ) -> str:
        if not rows:
            return ""
        title = (
            "## Memory Retrieval (HEAVY)\n"
            "Historical citations (file | date | snippet). Keep answers consistent with these records:\n"
            if tier == RetrievalTier.HEAVY
            else "## Memory Retrieval (LIGHT)\nRelevant historical notes:\n"
        )
        remaining = self.inject_budget_chars - len(title)
        if remaining <= 0:
            return ""

        lines: list[str] = []
        for row in rows:
            source = f"{row.get('file', '')} | {row.get('date', '')}".strip(" |")
            tldr = str(row.get("tldr") or row.get("title") or "").strip()
            snippet = str(row.get("snippet") or "").strip()

            if tier == RetrievalTier.HEAVY:
                line = f"- {source}\n  tl;dr: {tldr}\n  snippet: {snippet}"
            else:
                line = f"- {source}: {tldr}"

            size = len(line) + 1
            if size > remaining:
                break
            lines.append(line)
            remaining -= size

        if not lines:
            return ""
        return f"{title}{chr(10).join(lines)}"


def _cfg_get(config: Any, name: str, default: Any) -> Any:
    if config is None:
        return default
    return getattr(config, name, default)


class ContextBuilder:
    """
    Builds the context (system prompt + messages) for the agent.

    Assembles bootstrap files, memory, skills, and conversation history
    into a coherent prompt for the LLM.
    """

    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md", "IDENTITY.md"]
    COMPACTION_MARKER = "[context compacted]"

    def __init__(self, workspace: Path, memory_search_config: Any | None = None):
        self.workspace = workspace
        self.memory = MemoryStore(workspace)
        self.skills = SkillsLoader(workspace)
        self.memory_search_config = memory_search_config
        self.memory_search_enabled = bool(_cfg_get(memory_search_config, "enabled", True))
        self._last_retrieval = RetrievalRunResult(tier=RetrievalTier.OFF)
        self._retrieval_router = self._build_retrieval_router(memory_search_config)

    def _build_retrieval_router(self, cfg: Any | None) -> MemoryRetrievalRouter:
        if not self.memory_search_enabled:
            return MemoryRetrievalRouter(retriever=None, enabled=False)

        backend = str(_cfg_get(cfg, "backend", "builtin_fts"))
        tiers_cfg = _cfg_get(cfg, "tiers", None)
        routing_cfg = _cfg_get(cfg, "routing", None)

        retriever: MemoryRetriever | None = None
        try:
            retriever = MemoryRetriever(self.workspace, backend=backend)
        except Exception as exc:
            logger.warning("Memory retriever unavailable (fallback to no retrieval): {}", exc)
            self.memory_search_enabled = False
            return MemoryRetrievalRouter(retriever=None, enabled=False)

        return MemoryRetrievalRouter(
            retriever=retriever,
            enabled=True,
            enable_probe_on_off=bool(_cfg_get(tiers_cfg, "enable_probe_on_off", True)),
            light_top_k=int(_cfg_get(tiers_cfg, "light_top_k", 3)),
            heavy_top_k=int(_cfg_get(tiers_cfg, "heavy_top_k", 8)),
            heavy_max_snippet_chars=int(_cfg_get(tiers_cfg, "heavy_max_snippet_chars", 360)),
            inject_budget_chars=int(_cfg_get(tiers_cfg, "inject_budget_chars", 1800)),
            explicit_anchors=list(_cfg_get(routing_cfg, "explicit_anchors", DEFAULT_EXPLICIT_ANCHORS)),
            uncertain_markers=list(
                _cfg_get(routing_cfg, "uncertain_markers", DEFAULT_UNCERTAIN_MARKERS)
            ),
        )

    def get_last_retrieval_result(self) -> RetrievalRunResult:
        """Return retrieval metadata for the latest build_messages() call."""
        return self._last_retrieval

    def maybe_upgrade_retrieval_tier(
        self,
        draft: str,
        current_tier: RetrievalTier,
    ) -> RetrievalTier:
        """Check whether uncertain draft should trigger retrieval tier upgrade."""
        return self._retrieval_router.maybe_upgrade_after_draft(draft, current_tier)

    def build_system_prompt(
        self,
        skill_names: list[str] | None = None,
        *,
        memory_injection: str = "",
    ) -> str:
        """
        Build the system prompt from bootstrap files, memory, and skills.

        Args:
            skill_names: Optional list of skills to include.
            memory_injection: Optional retrieval snippets to inject.

        Returns:
            Complete system prompt.
        """
        parts = []

        # Core identity
        parts.append(self._get_identity())

        # Bootstrap files
        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)

        # Memory context
        if memory_injection:
            parts.append(f"# Memory\n\n{memory_injection}")
        elif not self.memory_search_enabled:
            memory = self.memory.get_memory_context()
            if memory:
                parts.append(f"# Memory\n\n{memory}")

        # Skills - progressive loading
        # 1. Always-loaded skills: include full content
        always_skills = self.skills.get_always_skills()
        if always_skills:
            always_content = self.skills.load_skills_for_context(always_skills)
            if always_content:
                parts.append(f"# Active Skills\n\n{always_content}")

        # 2. Available skills: only show summary (agent uses read_file to load)
        skills_summary = self.skills.build_skills_summary()
        if skills_summary:
            parts.append(
                f"""# Skills

The following skills extend your capabilities. To use a skill, read its SKILL.md file using the read_file tool.
Skills with available="false" need dependencies installed first - you can try installing them with apt/brew.

{skills_summary}"""
            )

        return "\n\n---\n\n".join(parts)

    def _get_identity(self) -> str:
        """Get the core identity section."""
        from datetime import datetime
        import time as _time

        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = _time.strftime("%Z") or "UTC"
        workspace_path = str(self.workspace.expanduser().resolve())
        system = platform.system()
        runtime = (
            f"{'macOS' if system == 'Darwin' else system} "
            f"{platform.machine()}, Python {platform.python_version()}"
        )

        return f"""# nanobot 🐈

You are nanobot, a helpful AI assistant. You have access to tools that allow you to:
- Read, write, and edit files
- Execute shell commands
- Search the web and fetch web pages
- Send messages to users on chat channels
- Spawn subagents for complex background tasks

## Current Time
{now} ({tz})

## Runtime
{runtime}

## Workspace
Your workspace is at: {workspace_path}
- Long-term memory: {workspace_path}/memory/MEMORY.md
- History log: {workspace_path}/memory/HISTORY.md (grep-searchable)
- Custom skills: {workspace_path}/skills/{{skill-name}}/SKILL.md

IMPORTANT: When responding to direct questions or conversations, reply directly with your text response.
Only use the 'message' tool when you need to send a message to a specific chat channel (like WhatsApp).
For normal conversation, just respond with text - do not call the message tool.

Always be helpful, accurate, and concise. When using tools, think step by step: what you know, what you need, and why you chose this tool.
When remembering something important, write to {workspace_path}/memory/MEMORY.md
To recall past events, use retrieval records and cite source file/date when relevant."""

    def _load_bootstrap_files(self) -> str:
        """Load all bootstrap files from workspace."""
        parts = []

        for filename in self.BOOTSTRAP_FILES:
            file_path = self.workspace / filename
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                parts.append(f"## {filename}\n\n{content}")

        return "\n\n".join(parts) if parts else ""

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        forced_retrieval_tier: RetrievalTier | None = None,
    ) -> list[dict[str, Any]]:
        """
        Build the complete message list for an LLM call.

        Args:
            history: Previous conversation messages.
            current_message: The new user message.
            skill_names: Optional skills to include.
            media: Optional list of local file paths for images/media.
            channel: Current channel (telegram, feishu, etc.).
            chat_id: Current chat/user ID.
            forced_retrieval_tier: Optional override for retrieval tier.

        Returns:
            List of messages including system prompt.
        """
        messages: list[dict[str, Any]] = []
        retrieval = self._run_retrieval(
            query=current_message,
            history=history,
            forced_tier=forced_retrieval_tier,
        )

        # System prompt
        system_prompt = self.build_system_prompt(
            skill_names,
            memory_injection=retrieval.injected_context,
        )
        if channel and chat_id:
            system_prompt += f"\n\n## Current Session\nChannel: {channel}\nChat ID: {chat_id}"
        messages.append({"role": "system", "content": system_prompt})

        # History
        messages.extend(history)

        # Current message (with optional image attachments)
        user_content = self._build_user_content(current_message, media)
        messages.append({"role": "user", "content": user_content})

        return messages

    def _run_retrieval(
        self,
        *,
        query: str,
        history: list[dict[str, Any]],
        forced_tier: RetrievalTier | None,
    ) -> RetrievalRunResult:
        if not self.memory_search_enabled:
            self._last_retrieval = RetrievalRunResult(tier=RetrievalTier.OFF)
            return self._last_retrieval

        recent_turns = self._extract_recent_turns(history)
        result = self._retrieval_router.run(
            query=query,
            recent_turns=recent_turns,
            tier=forced_tier,
        )
        self._last_retrieval = result
        return result

    @staticmethod
    def _extract_recent_turns(history: list[dict[str, Any]], *, max_turns: int = 6) -> list[str]:
        turns: list[str] = []
        for msg in history[-max_turns:]:
            content = msg.get("content")
            if isinstance(content, str):
                turns.append(content)
            else:
                try:
                    turns.append(json.dumps(content, ensure_ascii=False, default=str))
                except Exception:
                    turns.append(str(content))
        return turns

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """Build user message content with optional base64-encoded images."""
        if not media:
            return text

        images = []
        for path in media:
            p = Path(path)
            mime, _ = mimetypes.guess_type(path)
            if not p.is_file() or not mime or not mime.startswith("image/"):
                continue
            b64 = base64.b64encode(p.read_bytes()).decode()
            images.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

        if not images:
            return text
        return images + [{"type": "text", "text": text}]

    def add_tool_result(
        self,
        messages: list[dict[str, Any]],
        tool_call_id: str,
        tool_name: str,
        result: str,
        truncation_notice: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Add a tool result to the message list.

        Args:
            messages: Current message list.
            tool_call_id: ID of the tool call.
            tool_name: Name of the tool.
            result: Tool execution result.
            truncation_notice: Optional truncation marker to prepend.

        Returns:
            Updated message list.
        """
        content = result
        if truncation_notice:
            content = f"{truncation_notice}\n\n{result}"

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": content,
            }
        )
        return messages

    def inject_compaction_summary(
        self,
        messages: list[dict[str, Any]],
        summary: str,
        compacted_count: int,
    ) -> list[dict[str, Any]]:
        """
        Inject a compacted summary block near the front of message history.

        The summary replaces older dropped messages so the model keeps coarse
        continuity without carrying the full token cost.
        """
        summary_text = summary.strip() or "Older conversation content was compacted."
        block = {
            "role": "system",
            "content": (
                f"{self.COMPACTION_MARKER} {compacted_count} older messages were summarized.\n"
                f"{summary_text}"
            ),
        }

        cleaned = [
            msg
            for msg in messages
            if not (
                msg.get("role") == "system"
                and isinstance(msg.get("content"), str)
                and msg["content"].startswith(self.COMPACTION_MARKER)
            )
        ]
        insert_at = 1 if cleaned and cleaned[0].get("role") == "system" else 0
        return [*cleaned[:insert_at], block, *cleaned[insert_at:]]

    def add_assistant_message(
        self,
        messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Add an assistant message to the message list.

        Args:
            messages: Current message list.
            content: Message content.
            tool_calls: Optional tool calls.
            reasoning_content: Thinking output (Kimi, DeepSeek-R1, etc.).

        Returns:
            Updated message list.
        """
        msg: dict[str, Any] = {"role": "assistant"}

        if content:
            msg["content"] = content

        if tool_calls:
            msg["tool_calls"] = tool_calls

        # Thinking models reject history without this
        if reasoning_content:
            msg["reasoning_content"] = reasoning_content

        messages.append(msg)
        return messages
