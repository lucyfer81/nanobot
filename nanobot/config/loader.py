"""Configuration loading utilities."""

import json
import os
from pathlib import Path
from typing import Any

from nanobot.config.schema import Config


def get_config_path() -> Path:
    """Get the default configuration file path."""
    return Path.home() / ".nanobot" / "config.json"


def get_data_dir() -> Path:
    """Get the nanobot data directory."""
    from nanobot.utils.helpers import get_data_path
    return get_data_path()


def load_config(config_path: Path | None = None) -> Config:
    """
    Load configuration from file or create default.
    
    Args:
        config_path: Optional path to config file. Uses default if not provided.
    
    Returns:
        Loaded configuration object.
    """
    path = config_path or get_config_path()
    
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
            data = _migrate_config(data)
            config = Config.model_validate(convert_keys(data))
            _apply_env_overrides(config)
            return config
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Failed to load config from {path}: {e}")
            print("Using default configuration.")

    config = Config()
    _apply_env_overrides(config)
    return config


def save_config(config: Config, config_path: Path | None = None) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration to save.
        config_path: Optional path to save to. Uses default if not provided.
    """
    path = config_path or get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to camelCase format
    data = config.model_dump()
    data = convert_to_camel(data)
    
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _migrate_config(data: dict) -> dict:
    """Migrate old config formats to current."""
    # Move tools.exec.restrictToWorkspace → tools.restrictToWorkspace
    tools = data.get("tools", {})
    exec_cfg = tools.get("exec", {})
    if "restrictToWorkspace" in exec_cfg and "restrictToWorkspace" not in tools:
        tools["restrictToWorkspace"] = exec_cfg.pop("restrictToWorkspace")
    return data


def _load_dotenv(path: Path) -> dict[str, str]:
    """Load simple KEY=VALUE pairs from .env file.

    This parser is intentionally minimal and supports the common patterns used
    in local development. Existing process env vars should be preferred when
    both are present.
    """
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values


def _collect_env() -> dict[str, str]:
    """Collect env vars with optional .env fallback from current directory."""
    env = dict(os.environ)
    dotenv_vars = _load_dotenv(Path.cwd() / ".env")
    for key, value in dotenv_vars.items():
        env.setdefault(key, value)
    return env


def _apply_env_overrides(config: Config) -> None:
    """Apply non-file overrides that help local CLI workflows.

    Supported overrides:
    - SILICONFLOW_API_KEY     -> providers.vllm.api_key
    - SILICONFLOW_BASE_URL    -> providers.vllm.api_base
    - SILICONFLOW_API_BASE    -> providers.vllm.api_base
    - SILICONFLOW_MODEL_MAIN  -> agents.defaults.model
    - SILICONFLOW_MODEL       -> agents.defaults.model
    - BRAVE_API_KEY           -> tools.web.search.api_key
    - BRAVE_SEARCH_API_KEY    -> tools.web.search.api_key
    """
    env = _collect_env()

    siliconflow_api_key = env.get("SILICONFLOW_API_KEY", "").strip()
    if siliconflow_api_key:
        config.providers.vllm.api_key = siliconflow_api_key

    siliconflow_base_url = (
        env.get("SILICONFLOW_BASE_URL")
        or env.get("SILICONFLOW_API_BASE")
        or ""
    ).strip()
    if siliconflow_base_url:
        config.providers.vllm.api_base = siliconflow_base_url

    siliconflow_model = (
        env.get("SILICONFLOW_MODEL_MAIN")
        or env.get("SILICONFLOW_MODEL")
        or ""
    ).strip()
    if siliconflow_model:
        config.agents.defaults.model = siliconflow_model

    brave_api_key = (
        env.get("BRAVE_API_KEY")
        or env.get("BRAVE_SEARCH_API_KEY")
        or ""
    ).strip()
    if brave_api_key:
        config.tools.web.search.api_key = brave_api_key


def _is_mcp_server_map_path(path: tuple[str, ...]) -> bool:
    return len(path) == 2 and path[0] == "tools" and path[1] in {"mcp_servers", "mcpServers"}


def _is_mcp_env_path(path: tuple[str, ...]) -> bool:
    return (
        len(path) == 4
        and path[0] == "tools"
        and path[1] in {"mcp_servers", "mcpServers"}
        and path[3] == "env"
    )


def convert_keys(data: Any, _path: tuple[str, ...] = ()) -> Any:
    """Convert camelCase keys to snake_case for Pydantic."""
    if isinstance(data, dict):
        converted: dict[str, Any] = {}
        keep_keys = _is_mcp_server_map_path(_path) or _is_mcp_env_path(_path)
        for k, v in data.items():
            key = k if keep_keys else camel_to_snake(k)
            converted[key] = convert_keys(v, _path + (key,))
        return converted
    if isinstance(data, list):
        return [convert_keys(item, _path) for item in data]
    return data


def convert_to_camel(data: Any, _path: tuple[str, ...] = ()) -> Any:
    """Convert snake_case keys to camelCase."""
    if isinstance(data, dict):
        converted: dict[str, Any] = {}
        keep_keys = _is_mcp_server_map_path(_path) or _is_mcp_env_path(_path)
        for k, v in data.items():
            key = k if keep_keys else snake_to_camel(k)
            converted[key] = convert_to_camel(v, _path + (key,))
        return converted
    if isinstance(data, list):
        return [convert_to_camel(item, _path) for item in data]
    return data


def camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case."""
    result = []
    for i, char in enumerate(name):
        if char.isupper() and i > 0:
            result.append("_")
        result.append(char.lower())
    return "".join(result)


def snake_to_camel(name: str) -> str:
    """Convert snake_case to camelCase."""
    components = name.split("_")
    return components[0] + "".join(x.title() for x in components[1:])
