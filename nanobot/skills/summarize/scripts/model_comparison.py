#!/usr/bin/env python3
"""
Summarize 模型对比测试
对比 DeepSeek-V3.2 和 Qwen3-30B-A3B 的性能
"""

import os
import subprocess
import time
from pathlib import Path
from dotenv import load_dotenv

def load_env():
    """加载环境变量"""
    project_root = Path("/home/ubuntu/PyProjects/nanobot/.env")
    load_dotenv(project_root)

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = os.getenv("SILICONFLOW_API_KEY")
    env["OPENAI_BASE_URL"] = os.getenv("SILICONFLOW_BASE_URL", "").rstrip('/')

    return env

def test_model(model_id: str, url: str, length: str = "medium") -> dict:
    """测试单个模型"""
    env = load_env()

    start = time.time()
    result = subprocess.run(
        ["summarize", url, "--model", model_id, "--length", length],
        env=env,
        capture_output=True,
        text=True,
        timeout=120
    )
    elapsed = time.time() - start

    if result.returncode == 0:
        # 提取 token 信息
        tokens = "N/A"
        for line in result.stdout.split('\n'):
            if "token" in line.lower():
                tokens = line.strip()
                break

        return {
            "success": True,
            "time": elapsed,
            "summary": result.stdout.strip(),
            "tokens": tokens
        }
    else:
        return {
            "success": False,
            "error": result.stderr
        }

def compare_models():
    """对比测试多个模型"""
    test_url = "https://www.python.org"
    length = "medium"

    models = [
        {
            "name": "DeepSeek-V3.2",
            "id": "openai/deepseek-ai/DeepSeek-V3.2",
            "desc": "MoE 架构，推理能力强"
        },
        {
            "name": "Qwen3-30B-A3B",
            "id": "openai/Qwen/Qwen3-30B-A3B-Instruct-2507",
            "desc": "30B 参数，速度快"
        }
    ]

    print("=" * 80)
    print("Summarize 模型性能对比测试")
    print("=" * 80)
    print(f"测试 URL: {test_url}")
    print(f"摘要长度: {length}")
    print()

    results = []

    for model in models:
        print(f"🧪 测试: {model['name']} ({model['desc']})")
        print("-" * 80)

        result = test_model(model['id'], test_url, length)
        results.append({**model, **result})

        if result['success']:
            print(f"✅ 成功")
            print(f"⏱️  耗时: {result['time']:.1f}秒")
            if result.get('tokens'):
                print(f"📊 {result['tokens']}")
            print(f"📄 摘要预览:")
            # 只显示前 300 字符
            summary_preview = result['summary'][:300].split('\n')
            for line in summary_preview[:5]:
                print(f"   {line}")
            if len(result['summary']) > 300:
                print(f"   ...")
        else:
            print(f"❌ 失败: {result.get('error', 'Unknown error')[:200]}")

        print()

    # 总结
    print("=" * 80)
    print("测试总结")
    print("=" * 80)
    print()

    successful = [r for r in results if r['success']]

    if successful:
        print(f"{'模型':<20} {'状态':<8} {'耗时':<10} {'推荐度'}")
        print("-" * 80)

        for r in successful:
            status = "✅ 可用"
            time_str = f"{r['time']:.1f}s"

            # 根据速度给推荐度
            if r['time'] < 10:
                recommend = "⭐⭐⭐⭐⭐ 极速推荐"
            elif r['time'] < 15:
                recommend = "⭐⭐⭐⭐ 推荐"
            else:
                recommend = "⭐⭐⭐ 可用"

            print(f"{r['name']:<20} {status:<8} {time_str:<10} {recommend}")

        print()

        # 推荐
        fastest = min(successful, key=lambda x: x['time'])
        print(f"🏆 最快模型: {fastest['name']} ({fastest['time']:.1f}秒)")
        print()
        print("💡 使用建议:")
        print("  • 追求速度 →", fastest['name'])
        print("  • 追求性价比 → 需查看 SiliconFlow 定价")
        print("  • 追求质量 → 两个模型质量都很好")

    else:
        print("❌ 所有模型测试失败")

if __name__ == "__main__":
    compare_models()
