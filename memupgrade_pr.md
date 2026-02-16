# Nanobot 记忆与上下文管理升级计划

## 总览

本升级计划旨在系统性地提升nanobot的记忆管理和上下文工程能力，确保：
1. **记忆可靠性** - 重要信息不丢失
2. **语义检索** - 精准召回相关记忆
3. **上下文效率** - 严格的token预算控制
4. **极端情况处理** - 健壮的compaction策略

## 技术栈

- **语义检索**: Built-in Lite检索 (SQLite FTS5 + grep fallback)
- **优先级排序**: Memory Flush → Lite检索 → Bootstrap预算 → 自适应Compaction

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

## PR-02: Lite检索与结构化记忆 - 三档开关

### 目标

在不引入QMD/Bun/embedding依赖的前提下，显著提升记忆召回质量，并把“触发式检索漏掉历史信息”的风险降到可接受范围。

### 依赖

**必须先完成PR-01**，确保记忆写入链路稳定（flush + consolidation）。

### 问题

```
# 当前：两种极端都不好
1) 全量grep：噪声高，常返回大量无关行
2) 纯触发检索：如果误判为“非回忆问题”，会漏检

User: "按我之前约束（Cloudflare免费计划 + sqlite）再给方案"
Agent: 未触发检索，直接回答 → 可能与历史决策冲突 ❌
```

### 解决方案

采用 **PR-02 Lite**，由三部分组成：

1. **统一记忆格式**（写入侧提质）
2. **内置轻量检索**（SQLite FTS5 + grep fallback）
3. **三档检索路由**（0档/1档/2档 + 失败回退）

### 新增文件

```
nanobot/agent/memory_retrieval.py      # Lite检索器（FTS + 排序 + 压缩）
tests/test_memory_retrieval.py         # 检索与路由测试
```

### 修改文件

```
nanobot/agent/context.py               # 三档检索路由 + 结果注入
nanobot/agent/memory.py                # 结构化写入 + 索引更新接口
nanobot/agent/memory_flush.py          # flush输出结构化字段
nanobot/agent/loop.py                  # consolidate输出结构化字段
nanobot/config/schema.py               # Lite检索配置
```

### 核心功能

#### 1. 统一记忆文件格式（最小规范）

每条记忆使用固定头部字段，便于低成本检索：

```markdown
### 2026-02-16 14:20
type: decision
tags: cloudflare, sqlite, deploy
tl;dr: 保持免费方案，后端继续使用sqlite并避免额外托管组件

details:
- 决策原因：部署成本最低，维护简单
- 约束条件：Cloudflare Free Plan，轻量部署
```

字段约束：
- `type`: `decision | note | bug | idea | config`
- `tl;dr`: 一句话结论（建议10-30字）
- `tags`: 逗号分隔（可选但强烈建议）

#### 2. Lite检索器（无外部依赖）

```python
# nanobot/agent/memory_retrieval.py

class MemoryRetriever:
    """Built-in retriever: SQLite FTS5 first, grep fallback."""

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.db = workspace / "memory" / "index.db"
        self._ensure_index()

    def search_light(self, query: str, top_k: int = 3) -> list[dict]:
        """
        1档轻检索：仅搜 tl;dr + tags + title
        成本低，适合“可能是回忆问题”但不要求精确引用
        """
        ...

    def search_heavy(self, query: str, top_k: int = 8) -> list[dict]:
        """
        2档重检索：full text chunk + BM25 + 时间衰减 + 去重压缩
        用于精确回忆、参数/日期/命令、一致性约束
        """
        ...

    def probe(self, query: str) -> float:
        """
        0档探测：始终做一次超轻探测（top-1 score）
        仅用于判断“是否可能需要升级检索”，默认不注入上下文
        """
        ...
```

排序建议：
- `final_score = bm25_score * 0.75 + recency_boost * 0.25`
- `recency_boost` 基于条目时间衰减（最近条目略优先）

#### 3. 三档检索路由（0/1/2）

```python
# nanobot/agent/context.py

class RetrievalTier(IntEnum):
    OFF = 0    # 不注入，仅做probe
    LIGHT = 1  # 轻检索（tl;dr/tags/title）
    HEAVY = 2  # 重检索（chunk + 排序 + 压缩 + 引用）


def choose_retrieval_tier(user_message: str, recent_turns: list[str]) -> RetrievalTier:
    explicit_signals = [
        "之前", "上次", "记得", "回顾", "复盘", "我们聊过",
        "你说过", "我说过", "按之前", "照旧", "沿用", "查一下记录"
    ]
    implicit_signals = [
        "具体日期", "端口", "参数", "命令", "决策原因", "不要冲突"
    ]
    # 显式信号命中 -> HEAVY
    # 仅隐式信号命中 -> LIGHT
    # 都未命中 -> OFF（但执行probe）
    ...


def maybe_upgrade_after_draft(draft: str, current_tier: RetrievalTier) -> RetrievalTier:
    uncertain_markers = ["可能", "大概", "我猜", "不确定", "记不清"]
    # 如果回答显著不确定，则升级 0->1 或 1->2 重检索再答
    ...
```

关键点：
- **0档不是“什么都不做”**：执行一次超轻 `probe`，默认不注入上下文。
- **失败回退**：当回答表现出不确定性，自动升级检索档位并重答。
- **重检索必须引用来源**：返回片段附 `file + date + snippet`，降低“记错”风险。

### 配置参数

```yaml
# config.yaml
memory_search:
  enabled: true
  backend: "builtin_fts"           # builtin_fts | grep_only

  tiers:
    enable_probe_on_off: true      # 0档也做超轻探测
    light_top_k: 3                 # 1档
    heavy_top_k: 8                 # 2档
    heavy_max_snippet_chars: 360
    inject_budget_chars: 1800      # 注入系统提示词的总预算

  routing:
    explicit_anchors:
      - "之前"
      - "上次"
      - "记得"
      - "你说过"
      - "我说过"
      - "按之前"
      - "查一下记录"
    uncertain_markers:
      - "可能"
      - "大概"
      - "我猜"
      - "不确定"
      - "记不清"

  timeouts:
    probe_ms: 80
    light_ms: 180
    heavy_ms: 450
```

### 验收标准

- [ ] 不依赖QMD/Bun/node-llama-cpp，开箱可用
- [ ] 记忆条目满足最小结构：`type/tags/tl;dr/details`
- [ ] 三档路由工作正常（0档/1档/2档）
- [ ] 0档默认不注入上下文，但会执行probe
- [ ] 不确定回答可触发自动升级检索并重答
- [ ] 重检索结果包含来源信息（文件/时间/片段）
- [ ] FTS不可用时自动降级到grep，不中断主流程

### 测试计划

```python
# tests/test_memory_retrieval.py

def test_structured_memory_entry_format():
    """写入的记忆条目包含type/tags/tl;dr字段"""
    entry = build_memory_entry(
        type="decision",
        tags=["cloudflare", "sqlite"],
        tldr="继续使用sqlite，保持免费部署方案",
        details="..."
    )
    assert "type:" in entry
    assert "tags:" in entry
    assert "tl;dr:" in entry


def test_tier_routing_explicit_signal():
    """显式回忆词命中应走2档"""
    tier = choose_retrieval_tier("按之前约定的配置继续", [])
    assert tier == RetrievalTier.HEAVY


def test_tier_routing_implicit_signal():
    """需要参数/日期/命令但无显式词时走1档"""
    tier = choose_retrieval_tier("给我上次那个端口和命令", [])
    assert tier in (RetrievalTier.LIGHT, RetrievalTier.HEAVY)


def test_off_tier_still_probes():
    """0档不注入，但会执行probe"""
    result = router.run(query="解释一下这个概念", tier=RetrievalTier.OFF)
    assert result.probe_score >= 0
    assert result.injected_context == ""


def test_uncertain_answer_triggers_upgrade():
    """回答不确定时自动升级档位"""
    upgraded = maybe_upgrade_after_draft("我不确定，可能是...", RetrievalTier.LIGHT)
    assert upgraded == RetrievalTier.HEAVY


def test_heavy_search_returns_citations():
    """2档结果必须可追溯"""
    rows = retriever.search_heavy("为什么不用embedding", top_k=5)
    assert rows
    assert all("file" in r and "snippet" in r for r in rows)


def test_grep_fallback_when_fts_unavailable():
    """FTS不可用时自动回退grep"""
    retriever = MemoryRetriever(workspace, force_disable_fts=True)
    rows = retriever.search_light("sqlite")
    assert isinstance(rows, list)
```

### 手动验证步骤

```bash
# 1. 确认记忆文件存在并带结构化字段
grep -n "tl;dr:" memory/MEMORY.md | head
grep -n "type:" memory/MEMORY.md | head

# 2. 0档问题（默认不注入）
python -m nanobot.cli
You: "解释一下为什么要做上下文压缩"
# 预期：回答正常，日志显示probe执行，但未注入Memory片段

# 3. 1档问题（轻检索）
You: "我以前有没有写过关于部署约束的笔记？"
# 预期：返回少量tl;dr命中结果

# 4. 2档问题（重检索）
You: "我上次为什么决定不用embedding？"
# 预期：回答带来源片段（文件/日期/摘要），且与历史一致

# 5. 失败回退
You: "那个配置细节你记得吗？给精确参数"
# 预期：若首答不确定，系统自动升级检索并重答
```

### 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| 写入格式漂移导致检索质量下降 | 在flush/consolidate阶段强制模板化写入 |
| 0档漏检回忆类问题 | 0档保留probe + 不确定回答自动升级 |
| 重检索注入过多污染上下文 | 设定严格注入预算（字符/条数上限） |
| FTS环境差异导致不可用 | 自动降级grep并打日志，不阻塞会话 |

### 成功指标

- Lite检索可用率 > 99%（含fallback）
- 1档平均检索时间 < 200ms，2档 < 500ms
- 回忆类问题首次命中率 > 80%，升级后二次命中率 > 92%
- 用户反馈：历史决策/配置“更一致、更可追溯”

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
PR-02 (Lite检索 + 结构化记忆) ← 依赖PR-01的稳定memory
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
| PR-02 | 1天 | 0.5天 | 1.5天 |
| PR-03 | 1天 | 0.5天 | 1.5天 |
| PR-04 | 2天 | 1天 | 3天 |
| **总计** | **6天** | **3天** | **9天** |

### 下一步行动

1. **立即开始PR-01** - Memory Flush（最高价值）
2. 完成后验证memory被正确保存
3. 再启动PR-02 - Lite检索与结构化记忆
4. 根据效果决定PR-03和PR-04的优先级

---

**文档版本**: v1.1
**最后更新**: 2026-02-16
**维护者**: nanobot team
