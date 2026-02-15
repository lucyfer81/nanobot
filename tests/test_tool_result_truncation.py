from nanobot.agent.tool_result_truncation import format_truncation_notice, truncate_tool_result


def test_truncate_tool_result_keeps_short_text() -> None:
    text = "short output"
    kept, was_truncated = truncate_tool_result(text, hard_limit=100)
    assert kept == text
    assert was_truncated is False


def test_truncate_tool_result_adds_marker_and_respects_limit() -> None:
    text = "A" * 500
    kept, was_truncated = truncate_tool_result(text, hard_limit=120, keep_head=60, keep_tail=30)
    assert was_truncated is True
    assert len(kept) <= 120
    assert "tool output truncated" in kept


def test_truncate_tool_result_handles_tiny_limit() -> None:
    text = "B" * 500
    kept, was_truncated = truncate_tool_result(text, hard_limit=5)
    assert was_truncated is True
    assert kept == "BBBBB"


def test_format_truncation_notice() -> None:
    notice = format_truncation_notice(original_len=500, kept_len=120)
    assert "kept 120 of 500 chars" in notice
    assert "dropped 380 chars" in notice
