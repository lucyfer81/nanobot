#!/usr/bin/env python3
"""
Step-3.5-Flash 适配器
处理推理模型的特殊响应格式
"""

import os
import subprocess
import json
from pathlib import Path

def load_env():
    """加载 .env 文件"""
    project_root = Path("/home/ubuntu/PyProjects/nanobot")
    env_file = project_root / ".env"

    env_vars = {}
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key] = value

    return env_vars

def summarize_with_step_flash(url: str, length: str = "medium") -> str:
    """
    使用 Step-3.5-Flash 摘要 URL

    注意：这个函数绕过 summarize CLI，直接调用 API
    """
    env_vars = load_env()
    api_key = env_vars.get("SILICONFLOW_API_KEY")
    base_url = env_vars.get("SILICONFLOW_BASE_URL")

    # 1. 先提取内容（使用 summarize 的 --extract 模式）
    # 或者直接用 curl 获取网页内容
    # 这里简化处理，假设已经获取了内容

    # 2. 调用 Step-3.5-Flash API
    prompt = f"请用{length}长度总结这个网页的内容：{url}"

    payload = {
        "model": "stepfun-ai/Step-3.5-Flash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2000
    }

    curl_cmd = [
        "curl", "-s", "-X", "POST",
        f"{base_url.rstrip('/')}/chat/completions",
        "-H", f"Authorization: Bearer {api_key}",
        "-H", "Content-Type: application/json",
        "-d", json.dumps(payload)
    ]

    result = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=120)
    response = json.loads(result.stdout)

    # 3. 提取 reasoning_content（而不是 content）
    reasoning = response['choices'][0]['message'].get('reasoning_content', '')
    content = response['choices'][0]['message'].get('content', '')

    # 优先使用 reasoning，如果没有则用 content
    summary = reasoning or content

    if not summary:
        raise Exception("模型返回空响应")

    return summary

# 测试
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python step_flash_adapter.py <URL>")
        sys.exit(1)

    url = sys.argv[1]
    length = sys.argv[2] if len(sys.argv) > 2 else "medium"

    try:
        summary = summarize_with_step_flash(url, length)
        print("摘要:")
        print("=" * 70)
        print(summary)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)
