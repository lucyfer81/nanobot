# Nanobot 记忆与上下文管理升级计划

## 总览

本升级计划旨在系统性地提升nanobot的记忆管理和上下文工程能力，确保：
1. **记忆可靠性** - 重要信息不丢失
2. **语义检索** - 精准召回相关记忆
3. **上下文效率** - 严格的token预算控制
4. **极端情况处理** - 健壮的compaction策略

## 技术栈

- **语义检索**: QMD CLI (本地，基于Bun + SQLite + node-llama-cpp)
- **优先级排序**: Memory Flush → QMD → Bootstrap预算 → 自适应Compaction

---

## PR-01: Memory Flush - 自动记忆保存

### 目标

在上下文接近compaction阈值时，自动触发silent turn保存重要记忆到MD文件，防止信息丢失。

### 问题

```
用户会话：
User: "记住这个偏好：我喜欢用Python写脚本"
... (多轮对话)
... (触发compaction，偏好被压缩掉)
User: "我上次说喜欢什么语言？"
Agent: "抱歉，我找不到这个信息" ❌
```

### 解决方案

引入MemoryFlushTrigger，在compaction前自动保存记忆。

### 新增文件

```
nanobot/agent/memory_flush.py       # Memory flush触发器
tests/test_memory_flush.py          # 测试
```

### 修改文件

```
nanobot/agent/loop.py               # 集成flush trigger
nanobot/agent/__init__.py           # 导出MemoryFlushTrigger
```

### 核心功能

#### 1. MemoryFlushTrigger

```python
class MemoryFlushTrigger:
    """在compaction前自动触发记忆保存"""

    def __init__(self, workspace: Path, context_window: int = 200000):
        self.workspace = workspace
        self.context_window = context_window
        self.reserve_floor = 20000      # 预留token
        self.soft_threshold = 4000      # 软阈值

    def should_flush(self, current_tokens: int) -> bool:
        """检查是否需要触发flush"""
        trigger_point = (
            self.context_window
            - self.reserve_floor
            - self.soft_threshold
        )
        return current_tokens >= trigger_point

    async def trigger_flush(self, session, memory_store) -> bool:
        """
        触发silent turn保存记忆

        Returns:
            bool: 是否成功保存了记忆
        """
        from datetime import datetime

        system_prompt = "Session nearing compaction. Store durable memories now."

        today = datetime.now().strftime("%Y-%m-%d")
        user_prompt = f"""Review the recent conversation and write any lasting notes:

1. Important decisions → {self.workspace}/MEMORY.md
2. Daily notes → {self.workspace}/memory/{today}.md

Focus on:
- Decisions made and their rationale
- User preferences explicitly stated
- Important context for future sessions
- Technical constraints or requirements

Reply with NO_REPLY if nothing to store."""

        # Silent turn（不返回给用户）
        response = await session.silent_turn(system_prompt, user_prompt)

        if not response or response.lower().strip() == "no_reply":
            return False

        # 写入memory
        await memory_store.add_memory(response)
        return True
```

#### 2. Agent Loop集成

```python
# nanobot/agent/loop.py

class AgentLoop:
    def __init__(self, workspace: Path, ...):
        # ...existing...
        from .memory_flush import MemoryFlushTrigger
        self.memory_flush = MemoryFlushTrigger(
            workspace,
            context_window=self.context_window
        )

    async def run_turn(self, message: str) -> str:
        # 1. 检查是否需要flush
        current_tokens = self.estimate_messages_tokens(self.messages)

        if self.memory_flush.should_flush(current_tokens):
            saved = await self.memory_flush.trigger_flush(
                self.session,
                self.memory_store
            )
            if saved:
                log.info("Memory flush: saved to MD files before compaction")

        # 2. 继续正常流程
        # ...
```

### 配置参数

```yaml
# config.yaml
memory_flush:
  enabled: true
  reserve_floor: 20000      # 预留token
  soft_threshold: 4000      # 提前触发阈值
```

### 验收标准

- [ ] 临近compaction时自动触发flush
- [ ] Silent turn不返回内容给用户
- [ ] Memory被正确写入MEMORY.md或memory/YYYY-MM-DD.md
- [ ] Compaction后记忆不丢失
- [ ] 可通过grep MEMORY.md验证记忆被保存

### 测试计划

```python
# tests/test_memory_flush.py

def test_should_flush_at_threshold():
    """测试在正确时机触发flush"""
    trigger = MemoryFlushTrigger(workspace, context_window=200000)

    # 低于阈值
    assert not trigger.should_flush(175000)

    # 达到阈值 (200000 - 20000 - 4000 = 176000)
    assert trigger.should_flush(176000)

    # 超过阈值
    assert trigger.should_flush(180000)

async def test_trigger_flush_saves_to_memory():
    """测试flush实际保存记忆"""
    trigger = MemoryFlushTrigger(workspace)
    memory_store = MemoryStore(workspace)
    session = MockSession()

    # 模拟LLM返回记忆内容
    session.silent_turn.return_value = "User prefers Python for scripting"

    saved = await trigger.trigger_flush(session, memory_store)

    assert saved is True
    # 验证写入MEMORY.md
    memory_content = (workspace / "MEMORY.md").read_text()
    assert "Python" in memory_content

async def test_no_reply_handling():
    """测试NO_REPLY的处理"""
    trigger = MemoryFlushTrigger(workspace)
    session = MockSession()

    session.silent_turn.return_value = "NO_REPLY"

    saved = await trigger.trigger_flush(session, None)

    assert saved is False
    session.silent_turn.assert_called_once()

def test_integration_with_agent_loop():
    """测试与agent loop的集成"""
    loop = AgentLoop(workspace)

    # 模拟接近compaction
    loop.messages = [create_messages(n=50)]  # 假设~176k tokens

    # 应该触发flush
    assert loop.memory_flush.should_flush(
        loop.estimate_messages_tokens(loop.messages)
    )
```

### 手动验证步骤

```bash
# 1. 启动nanobot
python -m nanobot.cli

# 2. 测试记忆保存
You: "记住我的偏好：我喜欢用Python，不用JavaScript"
... (继续对话20+轮)
... (观察是否触发flush)

# 3. 验证记忆被保存
cat ~/workspace/memory/MEMORY.md

# 应该看到：
# - User preference: prefers Python over JavaScript

# 4. 测试compaction后记忆保留
... (继续对话直到compaction)
You: "我刚才说喜欢什么语言？"
# 应该能回答：Python
```

### 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| Silent turn失败被用户看到 | 添加exception handling，失败时记录log但不中断 |
| LLM返回内容格式不正确 | 解析"NO_REPLY"关键字，忽略其他内容 |
| 频繁flush浪费token | 每个compaction周期只flush一次（在session.json中记录） |

### 成功指标

- Compaction前关键信息保存率 > 95%
- 用户感知到的"记忆丢失"事件 = 0
- Silent turn平均响应时间 < 3s

---

## PR-02: QMD CLI集成 - 语义检索

### 目标

引入QMD CLI进行本地语义检索，基于稳定的memory基础（PR-01）实现精准召回。

### 依赖

**必须先完成PR-01**，确保MEMORY.md有稳定的内容来源。

### 问题

```
# 当前：grep全文搜索
User: "我之前怎么配置API的？"
Agent: grep HISTORY.md "API" → 返回100行，找不到关键配置 ❌

# 期望：语义检索
User: "我之前怎么配置API的？"
Agent: qmd query "API配置" → 精准返回相关片段 ✅
```

### 解决方案

集成QMD CLI，提供BM25、Vector、Hybrid三种检索模式。

### 新增文件

```
nanobot/agent/qmd_client.py          # QMD CLI封装
tests/test_qmd_client.py             # 测试
```

### 修改文件

```
nanobot/agent/context.py             # 集成QMD检索
nanobot/agent/__init__.py           # 导出QMDClient
```

### 核心功能

#### 1. QMDClient

```python
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Optional
import logging

log = logging.getLogger(__name__)

class QMDClient:
    """QMD CLI客户端封装"""

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self._ensure_initialized()

    def _ensure_initialized(self):
        """检查并初始化qmd collection"""
        # 1. 检查qmd是否安装
        try:
            subprocess.run(
                ["qmd", "--version"],
                capture_output=True,
                check=True,
                timeout=5
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                "QMD not found. Install with: bun install -g github:tobi/qmd\n"
                f"Error: {e}"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("QMD command timed out")

        # 2. 检查collection
        collections = self._list_collections()

        if not any(c.get("name") == "memory" for c in collections):
            log.info("Initializing QMD collection...")
            self._create_collection()
            self._generate_embeddings()
        else:
            log.info(f"QMD collection 'memory' already exists")

    def _list_collections(self) -> List[Dict]:
        """列出所有collections"""
        result = subprocess.run(
            ["qmd", "collection", "list", "--json"],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            return []

        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            log.error(f"Failed to parse QMD output: {result.stdout}")
            return []

    def _create_collection(self):
        """创建memory collection"""
        subprocess.run(
            ["qmd", "collection", "add",
             str(self.workspace),
             "--name", "memory",
             "--mask", "**/*.md"],
            check=True
        )
        log.info(f"Created QMD collection: {self.workspace}")

    def _generate_embeddings(self):
        """生成embeddings（首次可能需要几分钟）"""
        log.info("Generating embeddings (this may take a few minutes on first run)...")

        subprocess.run(
            ["qmd", "embed"],
            check=True
        )

        log.info("Embeddings generated successfully")

    def search(
        self,
        query: str,
        mode: str = "search",
        max_results: int = 10,
        min_score: float = 0.3
    ) -> List[Dict]:
        """
        搜索记忆

        Args:
            query: 搜索查询
            mode: search (BM25) | vsearch (Vector) | query (Hybrid+Rerank)
            max_results: 最大返回结果数
            min_score: 最低相似度阈值

        Returns:
            List[Dict]: 搜索结果列表
        """
        if mode not in ("search", "vsearch", "query"):
            raise ValueError(f"Invalid mode: {mode}")

        try:
            result = subprocess.run(
                [
                    "qmd", mode, query,
                    "--json",
                    "-n", str(max_results),
                    "--min-score", str(min_score)
                ],
                capture_output=True,
                text=True,
                timeout=30  # 30s timeout (query mode with rerank可能很慢)
            )

            if result.returncode != 0:
                log.error(f"QMD search failed: {result.stderr}")
                return []

            data = json.loads(result.stdout)
            return data.get("results", [])

        except subprocess.TimeoutExpired:
            log.error(f"QMD search timed out (mode={mode})")
            return []
        except json.JSONDecodeError:
            log.error(f"Failed to parse QMD JSON output")
            return []

    def get_status(self) -> Dict:
        """获取QMD状态"""
        result = subprocess.run(
            ["qmd", "status", "--json"],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            return {}

        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return {}
```

#### 2. ContextBuilder集成

```python
# nanobot/agent/context.py

class ContextBuilder:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.memory = MemoryStore(workspace)

        # 初始化QMD（可选，如果配置启用）
        try:
            from .qmd_client import QMDClient
            self.qmd = QMDClient(workspace)
        except Exception as e:
            log.warning(f"QMD not available: {e}")
            self.qmd = None

    def build_system_prompt(self, skill_names=None):
        parts = []

        # ...existing code (identity, bootstrap, skills)...

        # 新增：检索相关记忆
        if self.qmd:
            results = self._search_relevant_memory()
            if results:
                memory_context = self._format_qmd_results(results)
                parts.append(f"## Relevant Memory\n\n{memory_context}")

        return "\n\n---\n\n".join(parts)

    def _search_relevant_memory(self) -> List[Dict]:
        """搜索相关记忆"""
        # 使用query模式（质量最好）
        results = self.qmd.search(
            "recent decisions and user preferences",
            mode="query",  # Hybrid + Rerank
            max_results=5,
            min_score=0.4
        )
        return results

    def _format_qmd_results(self, results: List[Dict]) -> str:
        """格式化QMD搜索结果"""
        if not results:
            return "No relevant memories found."

        lines = []
        for r in results:
            docid = r.get("docid", "")
            file = r.get("file", "")
            score = r.get("score", 0)
            snippet = r.get("snippet", "")[:500]

            lines.append(
                f"- **[{docid}]** ({file}) [{score:.2f}]\n"
                f"  {snippet}..."
            )

        return "\n\n".join(lines)
```

### 配置参数

```yaml
# config.yaml
memory_search:
  enabled: true
  backend: "qmd"              # qmd | builtin
  mode: "query"                # search | vsearch | query
  max_results: 5
  min_score: 0.4
  timeout_seconds: 30
```

### 验收标准

- [ ] QMD CLI正确安装并可用
- [ ] Collection自动创建并生成embeddings
- [ ] 三种搜索模式（search/vsearch/query）正常工作
- [ ] 搜索结果正确格式化并注入system prompt
- [ ] 用户问题能精准召回相关记忆
- [ ] 超时时有降级处理

### 测试计划

```python
# tests/test_qmd_client.py

def test_qmd_initialization():
    """测试QMD初始化"""
    client = QMDClient(workspace)

    # 应该创建collection
    collections = client._list_collections()
    assert any(c["name"] == "memory" for c in collections)

def test_search_modes():
    """测试三种搜索模式"""
    client = QMDClient(workspace)

    # 1. BM25 search
    results_bm25 = client.search("API", mode="search")
    assert isinstance(results_bm25, list)

    # 2. Vector search
    results_vec = client.search("如何配置", mode="vsearch")
    assert isinstance(results_vec, list)

    # 3. Hybrid search
    results_hybrid = client.search("配置", mode="query")
    assert isinstance(results_hybrid, list)

def test_search_result_format():
    """测试搜索结果格式"""
    client = QMDClient(workspace)
    results = client.search("Python", mode="search", max_results=3)

    for r in results:
        assert "docid" in r
        assert "file" in r
        assert "score" in r
        assert "snippet" in r
        assert 0 <= r["score"] <= 1.0

async def test_context_builder_integration():
    """测试与ContextBuilder的集成"""
    builder = ContextBuilder(workspace)

    # 应该初始化QMD
    assert builder.qmd is not None

    # 搜索结果应该注入system prompt
    system_prompt = builder.build_system_prompt()
    assert "Relevant Memory" in system_prompt or "No relevant memories" in system_prompt

def test_qmd_fallback():
    """测试QMD不可用时的fallback"""
    # Mock QMD初始化失败
    with unittest.mock.patch('subprocess.run') as mock_run:
        mock_run.side_effect = FileNotFoundError("qmd not found")

        with pytest.raises(RuntimeError):
            QMDClient(workspace)

def test_search_timeout():
    """测试搜索超时处理"""
    client = QMDClient(workspace)

    # Mock超时
    with unittest.mock.patch('subprocess.run') as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired("qmd", 30)

        results = client.search("test", mode="query")
        assert results == []  # 应该返回空列表，不抛异常
```

### 手动验证步骤

```bash
# 1. 确保MEMORY.md有内容（PR-01的成果）
ls -lh ~/workspace/memory/MEMORY.md

# 2. 测试BM25搜索（快速）
qmd search "Python" --json -n 5
# 应该返回包含"Python"的结果

# 3. 测试向量搜索（语义）
qmd vsearch "编程语言偏好" --json -n 5
# 应该返回关于"喜欢Python"的结果

# 4. 测试混合搜索（最佳质量）
qmd query "如何配置" --json -n 5
# 应该返回最相关的结果

# 5. 集成测试
python -m nanobot.cli
You: "我之前说喜欢用什么语言？"
# 应该回答：Python（基于QMD检索）
```

### 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| QMD未安装 | 提供清晰的安装指引，失败时禁用QMD |
| 首次embeddings很慢 | 显示进度，设置合理超时 |
| Query模式太慢（10s+） | 提供mode配置选项，默认用search |
| 子进程失败 | 捕获异常，记录log，返回空列表 |
| JSON解析失败 | 捕获异常，记录log，返回空列表 |

### 成功指标

- QMD检索成功率 > 95%
- 平均检索时间 < 5s (search), < 15s (query)
- 语义召回准确率 > 80% (人工评估)
- 用户满意度：记忆"找得到"

---

## PR-03: Bootstrap预算控制 - 上下文优化

### 目标

严格控制bootstrap文件的token预算，防止大文件拖垮context。

### 问题

```
# 当前：全文读入bootstrap
AGENTS.md: 5000 tokens
TOOLS.md: 3000 tokens
IDENTITY.md: 2000 tokens
总计: 10000 tokens

# 如果这些文件很大？
AGENTS.md: 15000 tokens ❌
→ 拖垮整个context
```

### 解决方案

实现严格的token预算控制，大文件智能截断（head 70% + tail 20%）。

### 修改文件

```
nanobot/agent/context.py             # 增强bootstrap加载
```

### 核心功能

```python
# nanobot/agent/context.py

class ContextBuilder:
    # 新增配置
    BOOTSTRAP_MAX_CHARS = 20_000         # 单文件限制
    BOOTSTRAP_TOTAL_MAX_CHARS = 24_000   # 总限制
    BOOTSTRAP_HEAD_RATIO = 0.7           # Head比例
    BOOTSTRAP_TAIL_RATIO = 0.2           # Tail比例
    MIN_BOOTSTRAP_BUDGET = 64            # 最小预算

    def _load_bootstrap_files(self) -> str:
        """
        加载bootstrap文件（带严格预算控制）

        Returns:
            str: 格式化的bootstrap内容
        """
        parts = []
        remaining_budget = self.BOOTSTRAP_TOTAL_MAX_CHARS

        for filename in self.BOOTSTRAP_FILES:
            # 预算检查
            if remaining_budget <= 0:
                log.warn(
                    f"Bootstrap budget exhausted ({self.BOOTSTRAP_TOTAL_MAX_CHARS} chars), "
                    f"skipping {filename} and remaining files"
                )
                break

            if remaining_budget < self.MIN_BOOTSTRAP_BUDGET:
                log.warn(
                    f"Remaining bootstrap budget ({remaining_budget} chars) "
                    f"is below minimum ({self.MIN_BOOTSTRAP_BUDGET}), "
                    f"skipping {filename}"
                )
                break

            # 加载文件
            file_path = self.workspace / filename
            if not file_path.exists():
                continue

            try:
                content = file_path.read_text(encoding="utf-8")
            except Exception as e:
                log.error(f"Failed to read {filename}: {e}")
                continue

            # 计算该文件的预算
            file_budget = min(
                self.BOOTSTRAP_MAX_CHARS,
                remaining_budget
            )

            # 截断
            trimmed = self._trim_bootstrap(content, filename, file_budget)

            # 更新剩余预算
            remaining_budget -= len(trimmed)

            # 记录警告
            original_length = len(content)
            if original_length > file_budget:
                log.warn(
                    f"Bootstrap file {filename} is {original_length} chars "
                    f"(limit {file_budget}); truncated in injected context"
                )

            parts.append(f"## {filename}\n\n{trimmed}")

        return "\n\n---\n\n".join(parts)

    def _trim_bootstrap(self, content: str, filename: str, max_chars: int) -> str:
        """
        智能截断bootstrap内容

        策略：Head 70% + Tail 20%，中间用marker连接
        """
        if len(content) <= max_chars:
            return content

        head_chars = int(max_chars * self.BOOTSTRAP_HEAD_RATIO)
        tail_chars = int(max_chars * self.BOOTSTRAP_TAIL_RATIO)

        head = content[:head_chars]
        tail = content[-tail_chars:]

        # 截断marker
        truncated_chars = len(content) - max_chars
        marker = (
            f"\n\n"
            f"[...truncated {truncated_chars} chars, read {filename} for full content...]\n"
            f"\n"
        )

        return head + marker + tail

    def estimate_bootstrap_tokens(self) -> int:
        """估算bootstrap的token数"""
        bootstrap_content = self._load_bootstrap_files()
        # 粗略估算：1 char ≈ 0.25 tokens (英文)，0.5 tokens (中文)
        return len(bootstrap_content) // 3
```

### 配置参数

```yaml
# config.yaml
context:
  bootstrap:
    max_chars: 20000              # 单文件最大字符数
    total_max_chars: 24000        # 总字符数
    head_ratio: 0.7               # Head比例
    tail_ratio: 0.2               # Tail比例
    min_budget: 64                # 最小剩余预算
```

### 验收标准

- [ ] Bootstrap文件总大小受严格控制
- [ ] 大文件被智能截断（head + tail）
- [ ] 截断marker清晰告知LLM内容不完整
- [ ] 预算耗尽时优雅降级（跳过后续文件）
- [ ] 提供token估算方法

### 测试计划

```python
# tests/test_context_bootstrap.py

def test_small_file_not_truncated():
    """测试小文件不被截断"""
    builder = ContextBuilder(workspace)

    # 创建小文件
    test_file = workspace / "TEST.md"
    test_file.write_text("small content")

    content = builder._trim_bootstrap("small content", "TEST.md", max_chars=10000)

    assert content == "small content"
    assert "[...truncated" not in content

def test_large_file_truncated():
    """测试大文件被截断"""
    builder = ContextBuilder(workspace)

    # 创建大文件
    large_content = "x" * 30000
    test_file = workspace / "TEST.md"

    trimmed = builder._trim_bootstrap(large_content, "TEST.md", max_chars=10000)

    # 应该被截断
    assert len(trimmed) <= 10000
    assert "[...truncated" in trimmed
    assert "x" * 7000 in trimmed  # head
    assert "x" * 2000 in trimmed  # tail

def test_head_tail_ratio():
    """测试head/tail比例"""
    builder = ContextBuilder(workspace)

    content = "a" * 30000
    trimmed = builder._trim_bootstrap(content, "TEST.md", max_chars=10000)

    head_expected = 10000 * 0.7  # 7000
    tail_expected = 10000 * 0.2  # 2000

    # 验证head和tail都存在
    assert trimmed.startswith("a" * int(head_expected))
    assert trimmed.endswith("a" * int(tail_expected))

def test_budget_exhaustion():
    """测试预算耗尽"""
    builder = ContextBuilder(workspace)

    # 创建3个大文件
    for i in range(3):
        (workspace / f"TEST{i}.md").write_text("x" * 15000)

    # 应该只加载部分文件
    bootstrap = builder._load_bootstrap_files_with_custom_files(
        ["TEST0.md", "TEST1.md", "TEST2.md"]
    )

    # 总大小应该不超过TOTAL_MAX_CHARS
    assert len(bootstrap) <= builder.BOOTSTRAP_TOTAL_MAX_CHARS

def test_remaining_budget_check():
    """测试剩余预算检查"""
    builder = ContextBuilder(workspace)

    # 剩余预算 < MIN_BOOTSTRAP_BUDGET
    remaining = builder.MIN_BOOTSTRAP_BUDGET - 1

    # 应该跳过
    should_skip = remaining < builder.MIN_BOOTSTRAP_BUDGET
    assert should_skip is True

def test_token_estimation():
    """测试token估算"""
    builder = ContextBuilder(workspace)

    tokens = builder.estimate_bootstrap_tokens()

    # 应该返回合理的估算值
    assert tokens > 0
    assert tokens < builder.BOOTSTRAP_TOTAL_MAX_CHARS
```

### 手动验证步骤

```bash
# 1. 创建大bootstrap文件
echo "Large content..." > ~/workspace/AGENTS.md
# 填充到15000字符

# 2. 启动nanobot
python -m nanobot.cli

# 3. 检查日志
# 应该看到：
# WARNING: Bootstrap file AGENTS.md is 15000 chars (limit 20000); truncated

# 4. 验证system prompt
# 检查是否包含截断marker
```

### 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| 截断丢失关键信息 | Head 70% + Tail 20%保留大部分信息 |
| 预算估算不准确 | 使用字符数保守估算（1 char ≈ 0.3 tokens） |
| 用户不理解截断 | 清晰的marker说明如何查看完整内容 |

### 成功指标

- Bootstrap总token < 预算的98%
- 大文件截断率 < 5%（大部分文件应该完整）
- 用户反馈：context可控

---

## PR-04: 自适应Compaction - 极端情况处理

### 目标

实现自适应的compaction策略，处理极端大小的消息和edge cases。

### 问题

```
# 当前：固定compaction策略
if len(messages) > 50:
    summary = summarize(messages[:-20])
    # ❌ 如果某条消息有10000 tokens？
    # ❌ summarize会失败或超时
```

### 解决方案

自适应chunk ratio、大消息检测、渐进式summary。

### 新增文件

```
nanobot/agent/adaptive_compaction.py   # 自适应compaction
tests/test_adaptive_compaction.py      # 测试
```

### 修改文件

```
nanobot/agent/loop.py                   # 使用adaptive compaction
nanobot/agent/__init__.py             # 导出AdaptiveCompaction
```

### 核心功能

```python
# nanobot/agent/adaptive_compaction.py

import logging
from typing import List, Dict, Any

log = logging.getLogger(__name__)

class AdaptiveCompaction:
    """自适应上下文压缩策略"""

    BASE_CHUNK_RATIO = 0.4       # 默认chunk为context的40%
    MIN_CHUNK_RATIO = 0.15       # 最小chunk ratio
    SAFETY_MARGIN = 1.2          # 安全边际（token估算不准）

    def __init__(self, context_window: int = 200000):
        self.context_window = context_window

    def compute_chunk_ratio(self, messages: List[Dict]) -> float:
        """
        计算自适应chunk ratio

        当平均消息很大时，降低chunk ratio以避免超出context limit
        """
        if not messages:
            return self.BASE_CHUNK_RATIO

        total_tokens = sum(self.estimate_tokens(m) for m in messages)
        avg_tokens = total_tokens / len(messages)

        # 应用安全边际
        safe_avg_tokens = avg_tokens * self.SAFETY_MARGIN
        avg_ratio = safe_avg_tokens / self.context_window

        # 如果平均消息 > 10% context，降低chunk ratio
        if avg_ratio > 0.1:
            reduction = min(
                avg_ratio * 2,
                self.BASE_CHUNK_RATIO - self.MIN_CHUNK_RATIO
            )
            return max(self.MIN_CHUNK_RATIO, self.BASE_CHUNK_RATIO - reduction)

        return self.BASE_CHUNK_RATIO

    def is_oversized_for_summary(self, message: Dict) -> bool:
        """
        检查消息是否过大，无法安全summary

        Returns:
            bool: 如果消息 > 50% context，返回True
        """
        tokens = self.estimate_tokens(message) * self.SAFETY_MARGIN
        return tokens > self.context_window * 0.5

    def split_messages_by_tokens(
        self,
        messages: List[Dict],
        max_tokens: int
    ) -> List[List[Dict]]:
        """
        按token数分割消息

        Args:
            messages: 消息列表
            max_tokens: 每个chunk的最大token数

        Returns:
            List[List[Dict]]: 分割后的消息chunks
        """
        chunks = []
        current_chunk = []
        current_tokens = 0

        for message in messages:
            message_tokens = self.estimate_tokens(message)

            # 如果当前chunk + 新消息会超限
            if current_chunk and current_tokens + message_tokens > max_tokens:
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0

            current_chunk.append(message)
            current_tokens += message_tokens

            # 如果单个消息就超过max_tokens
            if message_tokens > max_tokens:
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    async def compact_with_fallback(
        self,
        messages: List[Dict],
        provider,
        reserve_tokens: int,
        previous_summary: str = None
    ) -> str:
        """
        带fallback的compaction

        策略：
        1. 尝试完整summary
        2. 失败：分离大消息后summary
        3. 再失败：返回简单fallback
        """
        # 计算chunk ratio
        chunk_ratio = self.compute_chunk_ratio(messages)
        max_chunk_tokens = int(self.context_window * chunk_ratio) - reserve_tokens

        try:
            # 尝试完整summary
            return await self._summarize_chunks(
                messages,
                provider,
                reserve_tokens,
                max_chunk_tokens,
                previous_summary
            )

        except Exception as e:
            log.warn(f"Full compaction failed: {e}, trying with oversized messages filtered")

            # Fallback: 分离处理
            normal = [m for m in messages if not self.is_oversized_for_summary(m)]
            oversized = [m for m in messages if self.is_oversized_for_summary(m)]

            if not normal:
                # 所有消息都过大
                return self._create_fallback_summary(messages, oversized)

            try:
                summary = await self._summarize_chunks(
                    normal,
                    provider,
                    reserve_tokens,
                    max_chunk_tokens,
                    previous_summary
                )

                # 添加大消息说明
                if oversized:
                    oversized_note = self._format_oversized_messages(oversized)
                    summary = f"{summary}\n\n{oversized_note}"

                return summary

            except Exception as e2:
                log.error(f"Compaction with filter also failed: {e2}")
                return self._create_fallback_summary(messages, oversized)

    async def _summarize_chunks(
        self,
        messages: List[Dict],
        provider,
        reserve_tokens: int,
        max_chunk_tokens: int,
        previous_summary: str = None
    ) -> str:
        """分chunk进行summary"""
        chunks = self.split_messages_by_tokens(messages, max_chunk_tokens)

        summary = previous_summary or ""

        for i, chunk in enumerate(chunks):
            summary = await self._call_llm_summary(
                chunk,
                provider,
                reserve_tokens,
                summary
            )
            log.info(f"Summarized chunk {i+1}/{len(chunks)}")

        return summary

    async def _call_llm_summary(
        self,
        messages: List[Dict],
        provider,
        reserve_tokens: int,
        previous_summary: str
    ) -> str:
        """调用LLM生成summary"""
        summary_instruction = (
            "Merge these partial summaries into a single cohesive summary. "
            "Preserve decisions, TODOs, open questions, and any constraints."
        )

        # 构建prompt
        if previous_summary:
            prompt = f"Previous summary:\n{previous_summary}\n\n"
            prompt += f"New messages:\n{self._format_messages(messages)}\n\n"
            prompt += f"Update the summary with the new messages. {summary_instruction}"
        else:
            prompt = f"Summarize these messages:\n{self._format_messages(messages)}\n\n"
            prompt += summary_instruction

        # 调用LLM
        response = await provider.complete(prompt)
        return response.strip()

    def _format_messages(self, messages: List[Dict]) -> str:
        """格式化消息为文本"""
        lines = []
        for m in messages:
            role = m.get("role", "unknown")
            content = m.get("content", "")[:500]  # 限制长度
            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    def _format_oversized_messages(self, oversized: List[Dict]) -> str:
        """格式化大消息说明"""
        count = len(oversized)
        total_tokens = sum(self.estimate_tokens(m) for m in oversized)

        return (
            f"[Skipped {count} oversized messages ({total_tokens:.0f} tokens total). "
            f"These messages were too large to summarize and were excluded from context.]"
        )

    def _create_fallback_summary(self, messages: List[Dict], oversized: List[Dict]) -> str:
        """创建fallback summary"""
        total_messages = len(messages)
        total_tokens = sum(self.estimate_tokens(m) for m in messages)

        return (
            f"[Session context was compacted: {total_messages} messages, "
            f"{total_tokens:.0f} tokens. "
            f"{len(oversized)} oversized messages were excluded from summary.]"
        )

    def estimate_tokens(self, message: Dict) -> int:
        """估算消息的token数"""
        content = message.get("content", "")
        # 粗略估算：1 char ≈ 0.3 tokens (中英文混合)
        return len(content) // 3
```

### 验收标准

- [ ] 自适应chunk ratio根据消息大小动态调整
- [ ] 大消息（>50% context）被正确识别
- [ ] Compaction失败时有fallback策略
- [ ] 不会因为单条大消息导致整个compaction失败
- [ ] 日志清晰记录compaction过程

### 测试计划

```python
# tests/test_adaptive_compaction.py

def test_chunk_ratio_adaptation():
    """测试chunk ratio自适应"""
    compactor = AdaptiveCompaction(context_window=100000)

    # 小消息：使用默认ratio
    small_messages = [{"content": "x" * 100} for _ in range(10)]
    ratio_small = compactor.compute_chunk_ratio(small_messages)
    assert ratio_small == compactor.BASE_CHUNK_RATIO

    # 大消息：降低ratio
    large_messages = [{"content": "x" * 15000} for _ in range(10)]
    ratio_large = compactor.compute_chunk_ratio(large_messages)
    assert ratio_large < compactor.BASE_CHUNK_RATIO
    assert ratio_large >= compactor.MIN_CHUNK_RATIO

def test_oversized_detection():
    """测试大消息检测"""
    compactor = AdaptiveCompaction(context_window=100000)

    # 正常消息
    normal_msg = {"content": "x" * 1000}
    assert not compactor.is_oversized_for_summary(normal_msg)

    # 超大消息（>50% context）
    oversized_msg = {"content": "x" * 60000}
    assert compactor.is_oversized_for_summary(oversized_msg)

def test_message_splitting():
    """测试消息分割"""
    compactor = AdaptiveCompaction()

    messages = [
        {"content": "small"},
        {"content": "x" * 10000},
        {"content": "x" * 10000},
        {"content": "small"}
    ]

    chunks = compactor.split_messages_by_tokens(messages, max_tokens=15000)

    # 应该分割成多个chunks
    assert len(chunks) >= 2

    # 验证每个chunk不超过预算
    for chunk in chunks:
        chunk_tokens = sum(compactor.estimate_tokens(m) for m in chunk)
        assert chunk_tokens <= 15000 + 5000  # 允许一定误差

async def test_compaction_with_fallback():
    """测试带fallback的compaction"""
    compactor = AdaptiveCompaction()
    provider = MockProvider()

    # 混合消息（正常+超大）
    messages = [
        {"content": "normal message"},
        {"content": "x" * 150000},  # 超大
        {"content": "another normal"}
    ]

    # 应该成功compaction（不抛异常）
    summary = await compactor.compact_with_fallback(
        messages,
        provider,
        reserve_tokens=20000
    )

    assert summary is not None
    assert len(summary) > 0
    assert "oversized" in summary.lower()

def test_token_estimation():
    """测试token估算"""
    compactor = AdaptiveCompaction()

    message = {"content": "Hello world"}
    tokens = compactor.estimate_tokens(message)

    # 应该返回合理的估算值
    assert tokens > 0
    assert tokens < 100  # 不应该过大
```

### 手动验证步骤

```bash
# 1. 创建测试session（包含大消息）
python -m nanobot.tools.create_test_session \
    --messages 100 \
    --oversized 3

# 2. 触发compaction
python -m nanobot.cli
# 多轮对话直到接近context limit

# 3. 检查日志
# 应该看到：
# INFO: Chunk ratio adapted to 0.25 (large messages detected)
# INFO: Summarized chunk 1/3
# WARN: Full compaction failed, trying with oversized messages filtered
# INFO: Skipped 3 oversized messages...

# 4. 验证compaction成功
# Agent应该能继续对话，不崩溃
```

### 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| Chunk ratio太小导致summary质量差 | 设置MIN_CHUNK_RATIO下限 |
| Fallback summary信息丢失过多 | 记录消息数量和token统计 |
| Token估算不准导致超限 | SAFETY_MARGIN 1.2倍 |

### 成功指标

- Compaction成功率 > 99%
- 大消息导致的compaction失败 = 0
- Context超出limit的情况 < 1%

---

## 总结

### PR依赖关系

```
PR-01 (Memory Flush)
    ↓
PR-02 (QMD CLI) ← 依赖PR-01的稳定memory
    ↓
PR-03 (Bootstrap预算)
    ↓
PR-04 (自适应Compaction)
```

### 预期收益

| 指标 | 当前 | PR后 | 提升 |
|------|------|------|------|
| **记忆保存率** | ~60% | >95% | +58% |
| **检索召回率** | N/A | >80% | N/A |
| **Context可控性** | 低 | 高 | +100% |
| **Compaction成功率** | ~90% | >99% | +10% |
| **用户满意度** | 中 | 高 | +50% |

### 实施时间估算

| PR | 工作量 | 测试 | 总计 |
|----|--------|------|------|
| PR-01 | 2天 | 1天 | 3天 |
| PR-02 | 1天 | 1天 | 2天 |
| PR-03 | 1天 | 0.5天 | 1.5天 |
| PR-04 | 2天 | 1天 | 3天 |
| **总计** | **6天** | **3.5天** | **9.5天** |

### 下一步行动

1. **立即开始PR-01** - Memory Flush（最高价值）
2. 完成后验证memory被正确保存
3. 再启动PR-02 - QMD集成
4. 根据效果决定PR-03和PR-04的优先级

---

**文档版本**: v1.0
**最后更新**: 2025-02-16
**维护者**: nanobot team
