from typing import Any

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.config.loader import convert_keys, convert_to_camel


class SampleTool(Tool):
    @property
    def name(self) -> str:
        return "sample"

    @property
    def description(self) -> str:
        return "sample tool"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 2},
                "count": {"type": "integer", "minimum": 1, "maximum": 10},
                "mode": {"type": "string", "enum": ["fast", "full"]},
                "meta": {
                    "type": "object",
                    "properties": {
                        "tag": {"type": "string"},
                        "flags": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["tag"],
                },
            },
            "required": ["query", "count"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return "ok"


def test_validate_params_missing_required() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "hi"})
    assert "missing required count" in "; ".join(errors)


def test_validate_params_type_and_range() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "hi", "count": 0})
    assert any("count must be >= 1" in e for e in errors)

    errors = tool.validate_params({"query": "hi", "count": "2"})
    assert any("count should be integer" in e for e in errors)


def test_validate_params_enum_and_min_length() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "h", "count": 2, "mode": "slow"})
    assert any("query must be at least 2 chars" in e for e in errors)
    assert any("mode must be one of" in e for e in errors)


def test_validate_params_nested_object_and_array() -> None:
    tool = SampleTool()
    errors = tool.validate_params(
        {
            "query": "hi",
            "count": 2,
            "meta": {"flags": [1, "ok"]},
        }
    )
    assert any("missing required meta.tag" in e for e in errors)
    assert any("meta.flags[0] should be string" in e for e in errors)


def test_validate_params_ignores_unknown_fields() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "hi", "count": 2, "extra": "x"})
    assert errors == []


async def test_registry_returns_validation_error() -> None:
    reg = ToolRegistry()
    reg.register(SampleTool())
    result = await reg.execute("sample", {"query": "hi"})
    assert "Invalid parameters" in result


def test_convert_keys_preserves_mcp_server_and_env_keys() -> None:
    raw = {
        "tools": {
            "mcpServers": {
                "MyServer": {
                    "command": "npx",
                    "env": {
                        "OPENAI_API_KEY": "secret",
                        "MixedCaseKey": "value",
                    },
                }
            },
            "restrictToWorkspace": True,
        }
    }

    converted = convert_keys(raw)

    assert "mcp_servers" in converted["tools"]
    assert "MyServer" in converted["tools"]["mcp_servers"]
    assert converted["tools"]["mcp_servers"]["MyServer"]["env"]["OPENAI_API_KEY"] == "secret"
    assert converted["tools"]["mcp_servers"]["MyServer"]["env"]["MixedCaseKey"] == "value"
    assert "restrict_to_workspace" in converted["tools"]


def test_convert_to_camel_preserves_mcp_server_and_env_keys() -> None:
    snake = {
        "tools": {
            "mcp_servers": {
                "MyServer": {
                    "command": "npx",
                    "env": {
                        "OPENAI_API_KEY": "secret",
                        "MixedCaseKey": "value",
                    },
                }
            },
            "restrict_to_workspace": False,
        }
    }

    converted = convert_to_camel(snake)

    assert "mcpServers" in converted["tools"]
    assert "MyServer" in converted["tools"]["mcpServers"]
    assert converted["tools"]["mcpServers"]["MyServer"]["env"]["OPENAI_API_KEY"] == "secret"
    assert converted["tools"]["mcpServers"]["MyServer"]["env"]["MixedCaseKey"] == "value"
    assert "restrictToWorkspace" in converted["tools"]


def test_round_trip_keeps_mcp_env_keys_unchanged() -> None:
    raw = {
        "tools": {
            "mcpServers": {
                "Server-A": {
                    "command": "uvx",
                    "env": {
                        "OPENAI_API_KEY": "x",
                        "AZURE_OPENAI_API_KEY": "y",
                    },
                }
            }
        }
    }

    round_tripped = convert_to_camel(convert_keys(raw))
    env = round_tripped["tools"]["mcpServers"]["Server-A"]["env"]

    assert env == raw["tools"]["mcpServers"]["Server-A"]["env"]
