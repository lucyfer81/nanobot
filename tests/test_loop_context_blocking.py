import copy
from pathlib import Path
from typing import Any

import pytest

from nanobot.agent.context import RetrievalRunResult, RetrievalTier
from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.base import Tool
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class StubProvider(LLMProvider):
    def __init__(self, responses: list[LLMResponse]):
        super().__init__(api_key=None, api_base=None)
        self._responses = responses
        self.calls = 0
        self.captured_messages: list[list[dict[str, Any]]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        self.calls += 1
        self.captured_messages.append(copy.deepcopy(messages))
        if self.calls <= len(self._responses):
            return self._responses[self.calls - 1]
        return self._responses[-1]

    def get_default_model(self) -> str:
        return "test/model"


class HugeTool(Tool):
    @property
    def name(self) -> str:
        return "huge_tool"

    @property
    def description(self) -> str:
        return "returns very large content"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        return "Z" * 500


class ToolLoopThenDirectProvider(LLMProvider):
    def __init__(self) -> None:
        super().__init__(api_key=None, api_base=None)
        self.calls = 0
        self.calls_with_tools = 0
        self.calls_without_tools = 0

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        self.calls += 1
        if tools is None:
            self.calls_without_tools += 1
            return LLMResponse(content="这是最终直答：飞书 channel 可在 config 中开启并填写 app_id/app_secret。")

        self.calls_with_tools += 1
        return LLMResponse(
            content="检查配置",
            tool_calls=[
                ToolCallRequest(
                    id=f"loop-{self.calls}",
                    name="huge_tool",
                    arguments={},
                )
            ],
        )

    def get_default_model(self) -> str:
        return "test/model"


class ForcedAnswerCaptureProvider(LLMProvider):
    def __init__(self) -> None:
        super().__init__(api_key=None, api_base=None)
        self.captured_messages: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        self.captured_messages = messages
        return LLMResponse(content="这是最终回答。")

    def get_default_model(self) -> str:
        return "test/model"


@pytest.mark.asyncio
async def test_run_agent_loop_compacts_then_calls_provider_on_block(tmp_path: Path) -> None:
    provider = StubProvider([LLMResponse(content="recovered")])
    agent = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        max_tokens=256,
        context_window_tokens=1000,
        context_guard_warn_ratio=0.8,
        context_guard_block_ratio=0.9,
        context_reserve_tokens=0,
    )

    initial_messages: list[dict[str, Any]] = [{"role": "system", "content": "system"}]
    for i in range(20):
        role = "user" if i % 2 == 0 else "assistant"
        initial_messages.append({"role": role, "content": f"msg{i} " + ("X" * 1600)})
    final_content, tools_used = await agent._run_agent_loop(initial_messages, session_key="test:block")

    assert provider.calls == 1
    assert tools_used == []
    assert final_content == "recovered"
    compacted_messages = provider.captured_messages[0]
    assert any(
        m.get("role") == "system"
        and isinstance(m.get("content"), str)
        and m["content"].startswith("[context compacted]")
        for m in compacted_messages
    )


@pytest.mark.asyncio
async def test_tool_result_is_truncated_before_next_provider_call(tmp_path: Path) -> None:
    provider = StubProvider(
        [
            LLMResponse(
                content="run tool",
                tool_calls=[
                    ToolCallRequest(
                        id="call-1",
                        name="huge_tool",
                        arguments={},
                    )
                ],
            ),
            LLMResponse(content="done"),
        ]
    )
    agent = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        max_tokens=4000,
        context_reserve_tokens=0,
        tool_result_max_chars=120,
        tool_result_max_ratio=1.0,
        tool_result_truncation_notice=True,
    )
    agent.tools.register(HugeTool())

    initial_messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "hello"},
    ]
    final_content, tools_used = await agent._run_agent_loop(initial_messages, session_key="test:truncate")

    assert final_content == "done"
    assert tools_used == ["huge_tool"]
    assert provider.calls == 2

    second_call_messages = provider.captured_messages[1]
    tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1

    tool_content = tool_messages[0]["content"]
    assert tool_content.startswith("[tool output truncated:")
    assert "tool output truncated" in tool_content
    _, truncated_payload = tool_content.split("\n\n", 1)
    assert len(truncated_payload) <= 120


class RetryContextStub:
    def __init__(self, retrieval_result: RetrievalRunResult, upgraded_tier: RetrievalTier):
        self._retrieval_result = retrieval_result
        self._upgraded_tier = upgraded_tier
        self.build_messages_called = 0

    def get_last_retrieval_result(self) -> RetrievalRunResult:
        return self._retrieval_result

    def maybe_upgrade_retrieval_tier(
        self,
        draft: str,
        current_tier: RetrievalTier,
    ) -> RetrievalTier:
        return self._upgraded_tier

    def build_messages(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.build_messages_called += 1
        return [
            {"role": "system", "content": "system"},
            {"role": "user", "content": kwargs.get("current_message", "")},
        ]


@pytest.mark.asyncio
async def test_uncertain_retry_skips_when_off_tier_probe_is_zero(tmp_path: Path) -> None:
    provider = StubProvider([LLMResponse(content="ok")])
    agent = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
    )
    session = agent.sessions.get_or_create("cli:probe-zero")

    stub = RetryContextStub(
        retrieval_result=RetrievalRunResult(
            tier=RetrievalTier.OFF,
            probe_score=0.0,
            hits=[],
            injected_context="",
        ),
        upgraded_tier=RetrievalTier.LIGHT,
    )
    agent.context = stub  # type: ignore[assignment]

    original = "我不确定，可能需要再确认。"
    content, tools = await agent._maybe_retry_with_upgraded_retrieval(
        session=session,
        session_key="cli:probe-zero",
        current_message="如何配置飞书？",
        channel="cli",
        chat_id="probe-zero",
        history_budget_tokens=500,
        media=None,
        final_content=original,
        tools_used=[],
    )

    assert content == original
    assert tools == []
    assert stub.build_messages_called == 0


@pytest.mark.asyncio
async def test_uncertain_retry_keeps_original_when_retry_returns_internal_fallback(tmp_path: Path) -> None:
    provider = StubProvider([LLMResponse(content="ok")])
    agent = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
    )
    session = agent.sessions.get_or_create("cli:fallback-preserve")

    stub = RetryContextStub(
        retrieval_result=RetrievalRunResult(
            tier=RetrievalTier.LIGHT,
            probe_score=0.6,
            hits=[{"file": "memory/MEMORY.md"}],
            injected_context="light",
        ),
        upgraded_tier=RetrievalTier.HEAVY,
    )
    agent.context = stub  # type: ignore[assignment]

    async def _fake_run_agent_loop(initial_messages: list[dict], session_key: str | None = None):
        return (
            "我已执行多轮操作但未能在限制内生成完整的响应。这可能是因为任务较复杂。",
            [],
        )

    agent._run_agent_loop = _fake_run_agent_loop  # type: ignore[assignment]

    original = "我不确定，你可以先打开配置文件。"
    content, tools = await agent._maybe_retry_with_upgraded_retrieval(
        session=session,
        session_key="cli:fallback-preserve",
        current_message="如何添加飞书channel？",
        channel="cli",
        chat_id="fallback-preserve",
        history_budget_tokens=500,
        media=None,
        final_content=original,
        tools_used=[],
    )

    assert content == original
    assert tools == []
    assert stub.build_messages_called == 1


@pytest.mark.asyncio
async def test_no_reply_triggers_forced_final_answer_without_tools(tmp_path: Path) -> None:
    provider = ToolLoopThenDirectProvider()
    agent = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        max_turns_per_request=2,
    )
    agent.tools.register(HugeTool())

    initial_messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "我如何添加飞书 channel？"},
    ]
    final_content, tools_used = await agent._run_agent_loop(
        initial_messages,
        session_key="test:forced-final",
    )

    assert final_content.startswith("这是最终直答")
    assert len(tools_used) == 2
    assert provider.calls_with_tools == 2
    assert provider.calls_without_tools == 1


@pytest.mark.asyncio
async def test_forced_final_answer_restates_latest_user_question(tmp_path: Path) -> None:
    provider = ForcedAnswerCaptureProvider()
    agent = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
    )

    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "第一问"},
        {"role": "assistant", "content": "先看配置"},
        {"role": "tool", "name": "read_file", "tool_call_id": "1", "content": "content"},
        {"role": "user", "content": "我如何给你添加飞书的channel？"},
    ]
    answer = await agent._attempt_forced_final_answer(messages, session_key="test:forced-shape")

    assert answer == "这是最终回答。"
    assert len(provider.captured_messages) == 2
    assert provider.captured_messages[0]["role"] == "system"
    assert provider.captured_messages[1]["role"] == "user"
    assert "我如何给你添加飞书的channel？" in provider.captured_messages[1]["content"]


def test_bad_forced_answer_detection() -> None:
    assert AgentLoop._is_bad_forced_answer(
        "Do not apologize for not being able to finish.Just give the best answer you can with what you have."
    )
    assert not AgentLoop._is_bad_forced_answer("你可以在 config.json 里开启 feishu 并填写 app_id/app_secret。")
