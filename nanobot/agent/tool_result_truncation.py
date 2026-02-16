"""Tool result truncation helpers for context safety."""

from __future__ import annotations


_TRUNCATION_MARKER = "\n\n[... tool output truncated ...]\n\n"


def truncate_tool_result(
    result: str,
    hard_limit: int,
    keep_head: int = 4000,
    keep_tail: int = 2000,
) -> tuple[str, bool]:
    """
    Truncate oversized tool output to a hard char limit.

    Prefers preserving head+tail with a clear marker in the middle.
    """
    text = result if isinstance(result, str) else str(result)

    if hard_limit <= 0:
        return "", bool(text)
    if len(text) <= hard_limit:
        return text, False

    if hard_limit <= len(_TRUNCATION_MARKER) + 1:
        return text[:hard_limit], True

    keep_head = max(0, keep_head)
    keep_tail = max(0, keep_tail)
    content_budget = hard_limit - len(_TRUNCATION_MARKER)

    if keep_head == 0 and keep_tail == 0:
        return text[:hard_limit], True

    head_budget = min(keep_head, content_budget)
    tail_budget = min(keep_tail, max(0, content_budget - head_budget))
    remainder = content_budget - head_budget - tail_budget

    if remainder > 0:
        # Fill remaining budget into the head first for better readability.
        head_budget += remainder

    head = text[:head_budget] if head_budget > 0 else ""
    tail = text[-tail_budget:] if tail_budget > 0 else ""

    truncated = f"{head}{_TRUNCATION_MARKER}{tail}"
    if len(truncated) > hard_limit:
        truncated = truncated[:hard_limit]
    return truncated, True


def format_truncation_notice(original_len: int, kept_len: int) -> str:
    """Format a user-visible truncation notice line."""
    dropped = max(0, original_len - kept_len)
    return (
        f"[tool output truncated: kept {kept_len} of {original_len} chars, "
        f"dropped {dropped} chars]"
    )
