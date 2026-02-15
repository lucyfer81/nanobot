from nanobot.agent.context_guard import (
    compute_context_limit,
    compute_tool_result_limit,
    estimate_message_tokens,
    evaluate_context_pressure,
)


def test_estimate_message_tokens_grows_with_payload() -> None:
    small = [{"role": "user", "content": "hi"}]
    large = [{"role": "user", "content": "x" * 2000}]
    assert estimate_message_tokens(large) > estimate_message_tokens(small)


def test_compute_context_limit_prefers_configured_value() -> None:
    assert compute_context_limit("anthropic/claude-opus-4-5", 8192) == 8192


def test_compute_context_limit_uses_model_hint_without_config() -> None:
    assert compute_context_limit("anthropic/claude-opus-4-5", None) == 200_000


def test_evaluate_context_pressure_states() -> None:
    assert evaluate_context_pressure(used=70, limit=100, warn_ratio=0.8, block_ratio=0.9) == "ok"
    assert evaluate_context_pressure(used=82, limit=100, warn_ratio=0.8, block_ratio=0.9) == "warn"
    assert evaluate_context_pressure(used=95, limit=100, warn_ratio=0.8, block_ratio=0.9) == "block"


def test_compute_tool_result_limit_applies_ratio_and_max_chars() -> None:
    # ratio cap: 1000 * 0.1 * 4 = 400
    assert compute_tool_result_limit(limit=1000, ratio=0.1, max_chars=12000) == 400
    # hard cap by max_chars
    assert compute_tool_result_limit(limit=100000, ratio=0.5, max_chars=12000) == 12000
