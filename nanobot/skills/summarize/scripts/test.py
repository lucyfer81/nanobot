#!/usr/bin/env python3
"""Summarize skill 测试脚本 - 验证配置是否正常"""

import os
import subprocess
import sys
from pathlib import Path

# 加载项目 .env 文件
project_root = Path(__file__).parent.parent.parent.parent.parent  # 多一层 parent
env_file = project_root / ".env"

# 读取环境变量
env_vars = {}
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                env_vars[key] = value

# 设置环境变量
for key, value in env_vars.items():
    if key.startswith("SILICONFLOW_"):
        if key == "SILICONFLOW_API_KEY":
            os.environ["OPENAI_API_KEY"] = value
        elif key == "SILICONFLOW_BASE_URL":
            os.environ["OPENAI_BASE_URL"] = value.rstrip('/')

def test_summarize():
    """测试 summarize 是否正常工作"""
    try:
        result = subprocess.run(
            ["summarize", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        print(f"✅ Summarize CLI 已安装: {result.stdout.strip()}")
        return True
    except Exception as e:
        print(f"❌ Summarize CLI 不可用: {e}")
        return False

def test_api():
    """测试 API 配置"""
    try:
        result = subprocess.run(
            ["summarize", "https://www.baidu.com", "--length", "short"],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            print("✅ SiliconFlow API 配置正常")
            return True
        else:
            print(f"❌ API 配置有问题: {result.stderr[:200]}")
            return False
    except Exception as e:
        print(f"❌ API 测试失败: {e}")
        return False

if __name__ == "__main__":
    print("Summarize Skill 配置测试")
    print("=" * 60)
    print(f"项目目录: {project_root}")
    print(f"环境文件: {env_file}")
    print()

    cli_ok = test_summarize()
    api_ok = test_api() if cli_ok else False

    print()
    if cli_ok and api_ok:
        print("🎉 Summarize skill 配置完成，可以正常使用！")
        sys.exit(0)
    else:
        print("⚠️  请检查配置")
        sys.exit(1)
