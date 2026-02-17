#!/usr/bin/env python3
"""
Summarize skill 调用接口
供 Nanobot agent 使用
"""

import os
import subprocess
import json
from pathlib import Path
from typing import Optional

def load_env():
    """从项目 .env 文件加载环境变量"""
    project_root = Path(__file__).parent.parent.parent.parent.parent  # 多一层 parent
    env_file = project_root / ".env"

    env_vars = {}
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key] = value

    # 构建环境变量
    env = os.environ.copy()
    for key, value in env_vars.items():
        if key == "SILICONFLOW_API_KEY":
            env["OPENAI_API_KEY"] = value
        elif key == "SILICONFLOW_BASE_URL":
            env["OPENAI_BASE_URL"] = value.rstrip('/')

    return env

def summarize_url(
    url: str,
    length: str = "medium",
    model: Optional[str] = None,
    json_output: bool = False
) -> dict | str:
    """
    调用 summarize CLI 摘要 URL

    Args:
        url: 要摘要的 URL
        length: 摘要长度 (short|medium|long|xl|xxl)
        model: 模型 ID (可选，默认使用配置文件中的模型)
        json_output: 是否返回 JSON 格式

    Returns:
        如果 json_output=True: 返回 dict
        否则: 返回字符串摘要
    """
    env = load_env()

    cmd = ["summarize", url, "--length", length]

    if model:
        cmd.extend(["--model", model])

    if json_output:
        cmd.append("--json")

    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=120
    )

    if result.returncode != 0:
        error_msg = result.stderr.strip()
        raise Exception(f"Summarize failed: {error_msg}")

    if json_output:
        return json.loads(result.stdout)
    else:
        return result.stdout.strip()

def summarize_file(
    file_path: str,
    length: str = "medium",
    model: Optional[str] = None
) -> str:
    """
    摘要本地文件

    Args:
        file_path: 文件路径
        length: 摘要长度
        model: 模型 ID

    Returns:
        摘要字符串
    """
    env = load_env()

    cmd = ["summarize", file_path, "--length", length]

    if model:
        cmd.extend(["--model", model])

    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=120
    )

    if result.returncode != 0:
        error_msg = result.stderr.strip()
        raise Exception(f"Summarize failed: {error_msg}")

    return result.stdout.strip()

# CLI 接口，供命令行直接调用
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize skill 调用接口")
    parser.add_argument("url", help="要摘要的 URL 或文件路径")
    parser.add_argument("--length", default="medium", help="摘要长度")
    parser.add_argument("--model", help="模型 ID")
    parser.add_argument("--json", action="store_true", help="JSON 输出")

    args = parser.parse_args()

    try:
        result = summarize_url(
            args.url,
            length=args.length,
            model=args.model,
            json_output=args.json
        )
        print(result)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        exit(1)
