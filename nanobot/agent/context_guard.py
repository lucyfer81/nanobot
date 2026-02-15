"""Utilities for context pressure checks before provider calls."""

from __future__ import annotations

import json
import math
from typing import Any


DEFAULT_CONTEXT_LIMIT = 8192
_CHARS_PER_TOKEN = 4

# Conservative model context hints used only when no explicit limit is configured.
_MODEL_CONTEXT_HINTS: list[tuple[str, int]] = [
    ("claude", 200_000),
    ("gpt-5", 128_000),
    ("gpt-4", 128_000),
    ("o1", 128_000),
    ("o3", 128_000),
    ("gemini", 1_000_000),
    ("deepseek", 128_000),
    ("qwen", 128_000),
    ("moonshot", 128_000),
    ("kimi", 128_000),
    ("minimax", 128_000),
    ("llama", 128_000),
    ("mistral", 32_000),
]


def estimate_message_tokens(messages: list[dict[str, Any]]) -> int:
    """Rough token estimate for message payloads using a chars/token heuristic."""
    if not messages:
        return 0

    total = 0
    for msg in messages:
        total += 4  # Base message framing overhead.
        total += _estimate_value_tokens(msg.get("content", ""))
        total += _estimate_value_tokens(msg.get("tool_calls"))
        total += _estimate_value_tokens(msg.get("reasoning_content"))
        total += _estimate_value_tokens(msg.get("name"))
        total += _estimate_value_tokens(msg.get("tool_call_id"))

    return total + 2  # Reply priming overhead.


def compute_context_limit(model: str, configured_max_tokens: int | None) -> int:
    """Compute usable context limit from explicit config or model hint."""
    if configured_max_tokens is not None and configured_max_tokens > 0:
        return configured_max_tokens

    model_lower = (model or "").lower()
    for keyword, limit in _MODEL_CONTEXT_HINTS:
        if keyword in model_lower:
            return limit

    return DEFAULT_CONTEXT_LIMIT


def evaluate_context_pressure(
    used: int,
    limit: int,
    warn_ratio: float,
    block_ratio: float,
) -> str:
    """Return context pressure level: ok/warn/block."""
    if limit <= 0:
        return "block"

    warn = max(0.0, min(1.0, warn_ratio))
    block = max(warn, min(1.0, block_ratio))

    if used >= int(limit * block):
        return "block"
    if used >= int(limit * warn):
        return "warn"
    return "ok"


def compute_tool_result_limit(limit: int, ratio: float, max_chars: int) -> int:
    """Compute hard char cap for a single tool result."""
    if max_chars <= 0:
        return 1

    ratio = max(0.0, ratio)
    if ratio == 0:
        return max_chars

    ratio_cap = int(max(0, limit) * ratio * _CHARS_PER_TOKEN)
    if ratio_cap <= 0:
        return 1

    return max(1, min(max_chars, ratio_cap))


def _estimate_value_tokens(value: Any) -> int:
    """Estimate token usage for arbitrary message fields."""
    if value is None:
        return 0
    if isinstance(value, str):
        if not value:
            return 0
        return max(1, math.ceil(len(value) / _CHARS_PER_TOKEN))
    if isinstance(value, (int, float, bool)):
        return 1

    try:
        serialized = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        serialized = str(value)

    if not serialized:
        return 0
    return max(1, math.ceil(len(serialized) / _CHARS_PER_TOKEN))
