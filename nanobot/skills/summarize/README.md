# Summarize Skill 配置文档

## ✅ 配置状态

**状态**: 已配置完成
**模型**: deepseek-ai/DeepSeek-V3.2 (SiliconFlow)
**API**: SiliconFlow (https://api.siliconflow.cn/v1)

## 📋 配置文件位置

### 1. Summarize 配置
```
~/.summarize/config.json
```

内容：
```json
{
  "model": {
    "id": "openai/deepseek-ai/DeepSeek-V3.2"
  },
  "openai": {
    "baseUrl": "https://api.siliconflow.cn/v1",
    "useChatCompletions": true
  },
  "cache": {
    "enabled": true,
    "maxMb": 500,
    "ttlDays": 30
  },
  "output": {
    "language": "zh-CN"
  }
}
```

### 2. API 密钥配置
```
/home/ubuntu/PyProjects/nanobot/.env
```

内容：
```
SILICONFLOW_API_KEY=sk-fszlxkcmrpvxplcpbjdsricmwwpdgsjnojcamgswmxrnepda
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1/
```

## 🚀 使用方法

### 在 Python 中调用

```python
from nanobot.skills.summarize.scripts.invoke import summarize_url

# 基本使用
summary = summarize_url("https://example.com/article")
print(summary)

# 指定长度
summary = summarize_url("https://example.com", length="short")

# JSON 输出
result = summarize_url("https://example.com", json_output=True)
print(result["summary"])
print(result["metrics"])
```

### 命令行直接调用

```bash
# 基本使用
summarize "https://example.com/article"

# 指定长度
summarize "https://example.com" --length short

# YouTube 视频
summarize "https://www.youtube.com/watch?v=xxx" --youtube auto

# 本地文件
summarize "/path/to/file.pdf" --length medium
```

### 在 Agent Loop 中使用

```python
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
project_root = Path("/home/ubuntu/PyProjects/nanobot")
load_dotenv(project_root / ".env")

# 调用 summarize
result = subprocess.run(
    ["summarize", url, "--length", "medium"],
    capture_output=True,
    text=True,
    timeout=120
)

summary = result.stdout
```

## 🎯 使用场景

| 场景 | 命令示例 |
|-----|---------|
| 网页摘要 | `summarize "https://example.com/article"` |
| YouTube 摘要 | `summarize "https://youtu.be/..." --youtube auto` |
| PDF 文件 | `summarize "/path/to/file.pdf"` |
| 长文章摘要 | `summarize "URL" --length long` |
| 提取内容（不摘要） | `summarize "URL" --extract` |
| JSON 输出 | `summarize "URL" --json` |

## ⚙️ 配置选项

### 摘要长度
- `short` - 简短摘要（约 1500 字符）
- `medium` - 中等摘要（约 3000 字符）- 默认
- `long` - 长篇摘要（约 6000 字符）
- `xl` - 超长摘要（约 12000 字符）
- `xxl` - 完整摘要（约 20000 字符）

### YouTube 模式
- `--youtube auto` - 自动选择最佳方式（推荐）
- `--youtube off` - 禁用 YouTube 特殊处理
- `--youtube always` - 强制使用 YouTube 模式

### 输出格式
- `--json` - JSON 格式输出（包含 metrics）
- `--extract` - 仅提取内容，不摘要
- `--plain` - 纯文本输出（无 Markdown）

## 🧪 测试

运行测试脚本：
```bash
python3 nanobot/skills/summarize/scripts/test.py
```

期望输出：
```
✅ Summarize CLI 已安装: 0.11.1
✅ SiliconFlow API 配置正常
🎉 Summarize skill 配置完成，可以正常使用！
```

## 🔍 故障排查

### 问题 1: "Missing OPENAI_API_KEY"
**原因**: 环境变量未设置
**解决**: 确保在调用前加载了 .env 文件

```python
from dotenv import load_dotenv
load_dotenv("/home/ubuntu/PyProjects/nanobot/.env")
```

### 问题 2: "LLM returned an empty summary"
**原因**: 模型 ID 错误或 API 响应格式问题
**解决**: 使用正确的模型 ID，如 `deepseek-ai/DeepSeek-V3.2`

### 问题 3: "summarize command not found"
**原因**: CLI 未安装
**解决**:
```bash
npm install -g @steipete/summarize
```

## 📊 性能

| 任务 | 预计时间 | Token 使用 |
|-----|---------|-----------|
| 短网页摘要 | 5-10s | ~2K tokens |
| 长文章摘要 | 15-30s | ~5K-10K tokens |
| YouTube (有字幕) | 10-20s | ~3K-8K tokens |
| YouTube (无字幕) | 60-120s | ~8K-15K tokens + 转录 |

## 💡 提示

1. **缓存**: Summarize 会缓存结果，相同 URL 不会重复调用 API
2. **成本**: DeepSeek-V3.2 在 SiliconFlow 上价格较低，约 ¥0.01-0.02/次摘要
3. **速度**: 首次调用会下载 LiteLLM 模型目录，后续调用更快
4. **语言**: 配置已设置为中文输出 (`"language": "zh-CN"`)

## 🔗 相关链接

- Summarize 官网: https://summarize.sh
- GitHub: https://github.com/steipete/summarize
- SiliconFlow: https://api.siliconflow.cn
