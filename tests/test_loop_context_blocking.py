import copy
from pathlib import Path
from typing import Any

import pytest

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


@pytest.mark.asyncio
async def test_run_agent_loop_blocks_before_provider_call(tmp_path: Path) -> None:
    provider = StubProvider([LLMResponse(content="should not be called")])
    agent = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        max_tokens=100,
        context_guard_warn_ratio=0.8,
        context_guard_block_ratio=0.9,
        context_reserve_tokens=0,
    )

    # ~750 tokens by heuristic, above block threshold (90 tokens).
    initial_messages = [{"role": "system", "content": "X" * 3000}]
    final_content, tools_used = await agent._run_agent_loop(initial_messages, session_key="test:block")

    assert provider.calls == 0
    assert tools_used == []
    assert final_content is not None
    assert "Context window is near capacity" in final_content


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
