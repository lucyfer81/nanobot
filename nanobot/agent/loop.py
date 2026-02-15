"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.agent.context import ContextBuilder
from nanobot.agent.context_guard import (
    compute_context_limit,
    compute_tool_result_limit,
    estimate_message_tokens,
    evaluate_context_pressure,
    prune_messages_by_tokens,
)
from nanobot.agent.tool_result_truncation import format_truncation_notice, truncate_tool_result
from nanobot.agent.followup_queue import FollowupQueue
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.memory import MemoryStore
from nanobot.agent.subagent import SubagentManager
from nanobot.infra.error_types import NanobotRuntimeError, classify_error
from nanobot.infra.retry import RetryPolicy, run_with_retry
from nanobot.agent.outcome import RunOutcome, RunOutcomeKind, RecoveryAction
from nanobot.session.manager import Session, SessionManager

if TYPE_CHECKING:
    from nanobot.config.schema import ExecToolConfig
    from nanobot.cron.service import CronService


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 20,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        context_window_tokens: int = 96000,
        memory_window: int = 50,
        context_guard_warn_ratio: float = 0.80,
        context_guard_block_ratio: float = 0.90,
        context_reserve_tokens: int = 2000,
        history_budget_ratio: float = 0.60,
        tool_result_max_chars: int = 12000,
        tool_result_max_ratio: float = 0.10,
        tool_result_truncation_notice: bool = True,
        retry_max_attempts: int = 3,
        retry_base_delay_ms: int = 250,
        retry_max_delay_ms: int = 4000,
        retry_jitter_ratio: float = 0.15,
        followup_debounce_ms: int = 800,
        brave_api_key: str | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        # PR-04: Outcome-driven execution limits
        max_turns_per_request: int = 12,
        max_recovery_attempts: int = 4,
        max_transient_retries: int = 1,
        max_context_recoveries: int = 2,
        enable_model_fallback: bool = False,
    ):
        from nanobot.config.schema import ExecToolConfig as RuntimeExecToolConfig

        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.context_window_tokens = context_window_tokens
        self.memory_window = memory_window
        self.context_guard_warn_ratio = context_guard_warn_ratio
        self.context_guard_block_ratio = context_guard_block_ratio
        self.context_reserve_tokens = context_reserve_tokens
        self.history_budget_ratio = max(0.0, min(1.0, history_budget_ratio))
        self.tool_result_max_chars = tool_result_max_chars
        self.tool_result_max_ratio = tool_result_max_ratio
        self.tool_result_truncation_notice = tool_result_truncation_notice
        self.followup_debounce_ms = max(0, followup_debounce_ms)
        self.retry_policy = RetryPolicy(
            max_attempts=max(1, retry_max_attempts),
            base_delay_ms=max(1, retry_base_delay_ms),
            max_delay_ms=max(1, retry_max_delay_ms),
            jitter_ratio=max(0.0, retry_jitter_ratio),
        )
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or RuntimeExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace

        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            brave_api_key=brave_api_key,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )
        self.followup_queue = FollowupQueue(debounce_ms=self.followup_debounce_ms)
        self.context_recovery_max_retries = max_context_recoveries
        self.context_compaction_keep_tail = 8
        # PR-04: Outcome-driven execution limits
        self.max_turns_per_request = max(1, max_turns_per_request)
        self.max_recovery_attempts = max(1, max_recovery_attempts)
        self.max_transient_retries = max(0, max_transient_retries)
        self.max_context_recoveries = max(0, max_context_recoveries)
        self.enable_model_fallback = enable_model_fallback

        self._running = False
        self._session_tasks: set[asyncio.Task[None]] = set()
        self._register_default_tools()
    
    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        # File tools (restrict to workspace if configured)
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        self.tools.register(ReadFileTool(allowed_dir=allowed_dir))
        self.tools.register(WriteFileTool(allowed_dir=allowed_dir))
        self.tools.register(EditFileTool(allowed_dir=allowed_dir))
        self.tools.register(ListDirTool(allowed_dir=allowed_dir))
        
        # Shell tool
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
        ))
        
        # Web tools
        self.tools.register(WebSearchTool(api_key=self.brave_api_key))
        self.tools.register(WebFetchTool())
        
        # Message tool
        message_tool = MessageTool(send_callback=self.bus.publish_outbound)
        self.tools.register(message_tool)
        
        # Spawn tool (for subagents)
        spawn_tool = SpawnTool(manager=self.subagents)
        self.tools.register(spawn_tool)
        
        # Cron tool (for scheduling)
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))
    
    def _set_tool_context(self, channel: str, chat_id: str) -> None:
        """Update context for all tools that need routing info."""
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.set_context(channel, chat_id)

        if spawn_tool := self.tools.get("spawn"):
            if isinstance(spawn_tool, SpawnTool):
                spawn_tool.set_context(channel, chat_id)

        if cron_tool := self.tools.get("cron"):
            if isinstance(cron_tool, CronTool):
                cron_tool.set_context(channel, chat_id)

    @staticmethod
    def _runtime_session_key(msg: InboundMessage) -> str:
        """Derive a stable session key for active-session serialization."""
        if msg.channel == "system" and ":" in msg.chat_id:
            return msg.chat_id
        return msg.session_key

    @staticmethod
    def _provider_error_from_response(response) -> NanobotRuntimeError | None:
        """Convert provider error-shaped response into a typed runtime error."""
        content = (response.content or "").strip()
        if response.finish_reason == "error" or content.lower().startswith("error calling llm:"):
            info = classify_error(content or "Error calling LLM")
            return NanobotRuntimeError(
                kind=info.kind,
                message=content or "Error calling LLM",
                retry_after_seconds=info.retry_after_seconds,
            )
        return None

    @staticmethod
    def _stringify_message_content(content: object) -> str:
        """Render arbitrary message content into compact plain text."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        try:
            return json.dumps(content, ensure_ascii=False, default=str)
        except Exception:
            return str(content)

    def _build_compaction_summary(self, messages: list[dict]) -> str:
        """Build a compact summary from dropped context messages."""
        if not messages:
            return ""

        snippets: list[str] = []
        for msg in messages[-12:]:
            role = str(msg.get("role") or "unknown")
            content = self._stringify_message_content(msg.get("content"))
            content = " ".join(content.split())
            if len(content) > 180:
                content = f"{content[:177]}..."
            if role == "tool":
                tool_name = msg.get("name") or "tool"
                snippets.append(f"- {role}:{tool_name}: {content}")
            else:
                snippets.append(f"- {role}: {content}")

        if not snippets:
            return ""
        return "Compacted context summary:\n" + "\n".join(snippets)

    def _compact_context_messages(self, messages: list[dict]) -> tuple[list[dict] | None, int]:
        """Compact older messages into one synthetic summary block."""
        if not messages:
            return None, 0

        has_system = messages[0].get("role") == "system"
        system_message = messages[0] if has_system else None
        body = messages[1:] if has_system else messages
        if len(body) < 3:
            return None, 0

        keep_tail = min(max(2, self.context_compaction_keep_tail), len(body) - 1)
        if keep_tail <= 0:
            return None, 0

        dropped = body[:-keep_tail]
        kept = body[-keep_tail:]
        summary = self._build_compaction_summary(dropped)
        if not summary:
            return None, 0

        rebuilt = [system_message] if system_message else []
        rebuilt.extend(kept)
        rebuilt = self.context.inject_compaction_summary(
            rebuilt,
            summary=summary,
            compacted_count=len(dropped),
        )
        return rebuilt, len(dropped)

    def _compute_available_context_limit(self) -> int:
        """Compute usable token limit after reserve."""
        context_limit = compute_context_limit(self.model, self.context_window_tokens)
        return max(1, context_limit - self.context_reserve_tokens)

    def _compute_history_budget_tokens(self) -> int:
        """Compute history token budget from available context and configured ratio."""
        available_limit = self._compute_available_context_limit()
        ratio = self.history_budget_ratio if self.history_budget_ratio > 0 else 1.0
        return max(1, int(available_limit * ratio))

    def _apply_runtime_token_pruning(
        self,
        messages: list[dict],
        available_limit: int,
    ) -> tuple[list[dict], int]:
        """
        Prune runtime message list to fit budget.

        Keeps the first system prompt separate so pruning focuses on conversation
        history/tool outputs, prioritizing old tool messages first.
        """
        if not messages:
            return messages, 0

        if messages[0].get("role") != "system":
            return prune_messages_by_tokens(
                messages,
                budget_tokens=available_limit,
                min_messages=3,
                prioritize_tool_messages=True,
            )

        system_message = messages[0]
        system_tokens = estimate_message_tokens([system_message])
        remaining_budget = max(1, available_limit - system_tokens)
        pruned_tail, removed = prune_messages_by_tokens(
            messages[1:],
            budget_tokens=remaining_budget,
            min_messages=3,
            prioritize_tool_messages=True,
        )
        return [system_message, *pruned_tail], removed

    async def _chat_with_retry(
        self,
        *,
        messages: list[dict],
        session_key: str | None = None,
    ):
        """Call provider chat with unified retry behavior."""

        async def _operation():
            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            if typed_error := self._provider_error_from_response(response):
                raise typed_error
            return response

        async def _on_retry(
            attempt: int,
            exc: Exception,
            info,
            delay_seconds: float,
        ) -> None:
            logger.warning(
                "Retrying provider call: session={} kind={} attempt={}/{} next_delay={:.2f}s error={}",
                session_key or "unknown",
                info.kind,
                attempt + 1,
                self.retry_policy.max_attempts,
                delay_seconds,
                str(exc)[:200],
            )

        return await run_with_retry(
            _operation,
            policy=self.retry_policy,
            classify=classify_error,
            on_retry=_on_retry,
        )

    def _apply_recovery(
        self,
        outcome: RunOutcome,
        context_retries: int,
        transient_retries: int,
        messages: list[dict],
    ) -> tuple[RecoveryAction, int, int]:
        """
        Decide recovery action based on outcome and retry budgets.

        Returns:
            Tuple of (RecoveryAction, updated_context_retries, updated_transient_retries).
        """
        if outcome.is_terminal():
            return RecoveryAction.stop_no_action(), context_retries, transient_retries

        if outcome.kind == RunOutcomeKind.RETRYABLE_ERROR:
            error_kind = outcome.diagnostics.get("error_kind")
            if error_kind == "context_overflow":
                if context_retries < self.max_context_recoveries:
                    return RecoveryAction.retry_with_compacted_context(), context_retries + 1, transient_retries
            elif error_kind in {"transient_http", "rate_limit", "timeout"}:
                if transient_retries < self.max_transient_retries:
                    return RecoveryAction.retry_same(), context_retries, transient_retries + 1

            # Retry budget exhausted
            return RecoveryAction.stop_with_message(outcome.payload or "Recovery budget exhausted."), context_retries, transient_retries

        # Should not reach here for other outcome kinds
        return RecoveryAction.stop_no_action(), context_retries, transient_retries

    async def _run_turn(
        self,
        messages: list[dict],
        session_key: str | None = None,
        turn_number: int = 1,
        tools_used: list[str] | None = None,
    ) -> tuple[RunOutcome, list[dict]]:
        """
        Execute a single agent turn and return outcome with updated messages.

        Returns:
            Tuple of (RunOutcome, updated_messages).
        """
        if tools_used is None:
            tools_used = []

        available_limit = self._compute_available_context_limit()
        messages, pruned_count = self._apply_runtime_token_pruning(messages, available_limit)
        used_tokens = estimate_message_tokens(messages)
        pressure = evaluate_context_pressure(
            used=used_tokens,
            limit=available_limit,
            warn_ratio=self.context_guard_warn_ratio,
            block_ratio=self.context_guard_block_ratio,
        )

        log_ctx = {
            "session": session_key or "unknown",
            "model": self.model,
            "turn": turn_number,
            "used": used_tokens,
            "limit": available_limit,
        }

        if pruned_count > 0:
            logger.bind(**log_ctx).warning("Token pruning removed {} message(s) before provider call", pruned_count)
        if pressure == "warn":
            logger.bind(**log_ctx).warning("Context pressure warning")
        elif pressure == "block":
            return RunOutcome(
                kind=RunOutcomeKind.RETRYABLE_ERROR,
                payload=(
                    "Context window is near capacity and automatic compaction could not recover. "
                    "Please send a shorter follow-up or start a new session with /new."
                ),
                reason="Context pressure block",
                diagnostics={"error_kind": "context_overflow"},
            ), messages

        try:
            response = await self._chat_with_retry(
                messages=messages,
                session_key=session_key,
            )
        except Exception as exc:
            info = classify_error(exc)
            logger.warning(
                "Provider call failed after retries: session={} kind={} error={}",
                session_key or "unknown",
                info.kind,
                info.message[:300],
            )
            return RunOutcome(
                kind=RunOutcomeKind.RETRYABLE_ERROR if info.kind in {"context_overflow", "transient_http", "rate_limit", "timeout"} else RunOutcomeKind.FATAL_ERROR,
                payload=None,
                reason=f"Provider error: {info.kind}",
                diagnostics={"error_kind": info.kind, "error_message": info.message[:300]},
            ), messages

        if response.has_tool_calls:
            tool_call_dicts = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments)
                    }
                }
                for tc in response.tool_calls
            ]
            messages = self.context.add_assistant_message(
                messages, response.content, tool_call_dicts,
                reasoning_content=response.reasoning_content,
            )

            for tool_call in response.tool_calls:
                tools_used.append(tool_call.name)
                args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                logger.info(f"Tool call: {tool_call.name}({args_str[:200]})")
                result = await self.tools.execute(tool_call.name, tool_call.arguments)
                result_text = result if isinstance(result, str) else str(result)
                hard_limit = compute_tool_result_limit(
                    limit=available_limit,
                    ratio=self.tool_result_max_ratio,
                    max_chars=self.tool_result_max_chars,
                )
                truncated_result, was_truncated = truncate_tool_result(
                    result_text,
                    hard_limit=hard_limit,
                )
                truncation_notice = None
                if was_truncated and self.tool_result_truncation_notice:
                    truncation_notice = format_truncation_notice(
                        original_len=len(result_text),
                        kept_len=len(truncated_result),
                    )
                messages = self.context.add_tool_result(
                    messages,
                    tool_call.id,
                    tool_call.name,
                    truncated_result,
                    truncation_notice=truncation_notice,
                )

            # Tool calls executed successfully - continue to next turn
            return RunOutcome(
                kind=RunOutcomeKind.NEEDS_FOLLOWUP,
                payload=None,
                reason="Tool calls executed, awaiting next iteration",
            ), messages
        else:
            # No tool calls - terminal success
            return RunOutcome(
                kind=RunOutcomeKind.SUCCESS,
                payload=response.content,
                reason="LLM response without tool calls",
            ), messages

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        session_key: str | None = None,
    ) -> tuple[str | None, list[str]]:
        """
        Run the agent iteration loop using outcome-driven state machine.

        Args:
            initial_messages: Starting messages for the LLM conversation.
            session_key: Session key for context pressure logging.

        Returns:
            Tuple of (final_content, list_of_tools_used).
        """
        messages = initial_messages
        turn_number = 0
        tools_used: list[str] = []
        context_retries = 0
        transient_retries = 0
        final_outcome: RunOutcome | None = None

        while turn_number < self.max_turns_per_request:
            turn_number += 1

            outcome, messages = await self._run_turn(
                messages=messages,
                session_key=session_key,
                turn_number=turn_number,
                tools_used=tools_used,
            )

            recovery, context_retries, transient_retries = self._apply_recovery(
                outcome=outcome,
                context_retries=context_retries,
                transient_retries=transient_retries,
                messages=messages,
            )

            if outcome.is_terminal():
                final_outcome = outcome
                break

            if recovery.should_retry:
                if recovery.retry_with_compaction:
                    compacted = self._compact_context_messages(messages)
                    if compacted[0] is not None:
                        messages = compacted[0]
                        logger.info(f"Turn {turn_number}: retrying with compacted context")
                    else:
                        # Compaction failed, stop
                        final_outcome = RunOutcome(
                            kind=RunOutcomeKind.FATAL_ERROR,
                            payload="Context compaction failed",
                            reason="Could not compact messages",
                        )
                        break
                else:
                    logger.info(f"Turn {turn_number}: retrying without compaction")
                continue

            if recovery.fallback_message:
                final_outcome = RunOutcome(
                    kind=RunOutcomeKind.FATAL_ERROR,
                    payload=recovery.fallback_message,
                    reason="Recovery action requested stop",
                )
                break

            # Should not reach here if logic is correct
            final_outcome = outcome
            break

        if final_outcome is None:
            final_outcome = RunOutcome(
                kind=RunOutcomeKind.NO_REPLY,
                payload=None,
                reason="Max turns reached without terminal outcome",
            )

        # Extract final content from outcome
        if final_outcome.kind == RunOutcomeKind.SUCCESS:
            return final_outcome.payload, tools_used
        elif final_outcome.payload:
            return final_outcome.payload, tools_used
        else:
            return None, tools_used


    def _on_session_task_done(self, task: asyncio.Task[None]) -> None:
        """Handle completion of one session worker task."""
        self._session_tasks.discard(task)
        try:
            task.result()
        except Exception as exc:
            logger.error("Session worker crashed: {}", exc)

    def _start_session_worker(self, first_msg: InboundMessage, flow_session_key: str) -> None:
        """Spawn one worker to process and drain a single session."""
        task = asyncio.create_task(self._process_session_flow(first_msg, flow_session_key))
        self._session_tasks.add(task)
        task.add_done_callback(self._on_session_task_done)

    async def _process_session_flow(self, first_msg: InboundMessage, flow_session_key: str) -> None:
        """Process one message, then drain queued followups for the same session."""
        self.bus.mark_task_started()
        current_msg: InboundMessage | None = first_msg
        try:
            while current_msg is not None:
                try:
                    response = await self._process_message(current_msg)
                    if response:
                        await self.bus.publish_outbound(response)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=current_msg.channel,
                        chat_id=current_msg.chat_id,
                        content=f"Sorry, I encountered an error: {str(e)}"
                    ))

                while True:
                    next_msg = await self.followup_queue.pop_next(flow_session_key)
                    if next_msg is not None:
                        current_msg = next_msg
                        break
                    if await self.followup_queue.deactivate_if_idle(flow_session_key):
                        current_msg = None
                        break
        finally:
            await self.followup_queue.deactivate_if_idle(flow_session_key)
            self.bus.mark_task_done()

    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus."""
        self._running = True
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(
                    self.bus.consume_inbound(),
                    timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            flow_session_key = self._runtime_session_key(msg)
            queued = await self.followup_queue.enqueue_if_active(flow_session_key, msg)
            if queued:
                logger.debug("Queued followup for active session {}", flow_session_key)
                continue

            self._start_session_worker(msg, flow_session_key)

        await self.bus.wait_for_active_tasks(timeout_ms=5_000)
        if self._session_tasks:
            _, pending = await asyncio.wait(self._session_tasks, timeout=5.0)
            if pending:
                logger.warning("Stopping with {} unfinished session task(s)", len(pending))
    
    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")
    
    async def _process_message(self, msg: InboundMessage, session_key: str | None = None) -> OutboundMessage | None:
        """
        Process a single inbound message.
        
        Args:
            msg: The inbound message to process.
            session_key: Override session key (used by process_direct).
        
        Returns:
            The response message, or None if no response needed.
        """
        # System messages route back via chat_id ("channel:chat_id")
        if msg.channel == "system":
            return await self._process_system_message(msg)
        
        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info(f"Processing message from {msg.channel}:{msg.sender_id}: {preview}")
        
        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)
        
        # Handle slash commands
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            # Capture messages before clearing (avoid race condition with background task)
            messages_to_archive = session.messages.copy()
            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)

            async def _consolidate_and_cleanup():
                temp_session = Session(key=session.key)
                temp_session.messages = messages_to_archive
                await self._consolidate_memory(temp_session, archive_all=True)

            asyncio.create_task(_consolidate_and_cleanup())
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started. Memory consolidation in progress.")
        if cmd == "/help":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="🐈 nanobot commands:\n/new — Start a new conversation\n/help — Show available commands")
        
        if len(session.messages) > self.memory_window:
            asyncio.create_task(self._consolidate_memory(session))

        self._set_tool_context(msg.channel, msg.chat_id)
        history_budget_tokens = self._compute_history_budget_tokens()
        initial_messages = self.context.build_messages(
            history=session.get_history(
                max_messages=self.memory_window,
                max_tokens=history_budget_tokens,
            ),
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
        )
        final_content, tools_used = await self._run_agent_loop(initial_messages, session_key=key)

        if final_content is None:
            final_content = "I've completed processing but have no response to give."
        
        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info(f"Response to {msg.channel}:{msg.sender_id}: {preview}")
        
        session.add_message("user", msg.content)
        session.add_message("assistant", final_content,
                            tools_used=tools_used if tools_used else None)
        self.sessions.save(session)
        
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=msg.metadata or {},  # Pass through for channel-specific needs (e.g. Slack thread_ts)
        )
    
    async def _process_system_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a system message (e.g., subagent announce).
        
        The chat_id field contains "original_channel:original_chat_id" to route
        the response back to the correct destination.
        """
        logger.info(f"Processing system message from {msg.sender_id}")
        
        # Parse origin from chat_id (format: "channel:chat_id")
        if ":" in msg.chat_id:
            parts = msg.chat_id.split(":", 1)
            origin_channel = parts[0]
            origin_chat_id = parts[1]
        else:
            # Fallback
            origin_channel = "cli"
            origin_chat_id = msg.chat_id
        
        session_key = f"{origin_channel}:{origin_chat_id}"
        session = self.sessions.get_or_create(session_key)
        self._set_tool_context(origin_channel, origin_chat_id)
        history_budget_tokens = self._compute_history_budget_tokens()
        initial_messages = self.context.build_messages(
            history=session.get_history(
                max_messages=self.memory_window,
                max_tokens=history_budget_tokens,
            ),
            current_message=msg.content,
            channel=origin_channel,
            chat_id=origin_chat_id,
        )
        final_content, _ = await self._run_agent_loop(initial_messages, session_key=session_key)

        if final_content is None:
            final_content = "Background task completed."
        
        session.add_message("user", f"[System: {msg.sender_id}] {msg.content}")
        session.add_message("assistant", final_content)
        self.sessions.save(session)
        
        return OutboundMessage(
            channel=origin_channel,
            chat_id=origin_chat_id,
            content=final_content
        )
    
    async def _consolidate_memory(self, session, archive_all: bool = False) -> None:
        """Consolidate old messages into MEMORY.md + HISTORY.md.

        Args:
            archive_all: If True, clear all messages and reset session (for /new command).
                       If False, only write to files without modifying session.
        """
        memory = MemoryStore(self.workspace)

        if archive_all:
            old_messages = session.messages
            keep_count = 0
            logger.info(f"Memory consolidation (archive_all): {len(session.messages)} total messages archived")
        else:
            keep_count = self.memory_window // 2
            if len(session.messages) <= keep_count:
                logger.debug(f"Session {session.key}: No consolidation needed (messages={len(session.messages)}, keep={keep_count})")
                return

            messages_to_process = len(session.messages) - session.last_consolidated
            if messages_to_process <= 0:
                logger.debug(f"Session {session.key}: No new messages to consolidate (last_consolidated={session.last_consolidated}, total={len(session.messages)})")
                return

            old_messages = session.messages[session.last_consolidated:-keep_count]
            if not old_messages:
                return
            logger.info(f"Memory consolidation started: {len(session.messages)} total, {len(old_messages)} new to consolidate, {keep_count} keep")

        lines = []
        for m in old_messages:
            if not m.get("content"):
                continue
            tools = f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
            lines.append(f"[{m.get('timestamp', '?')[:16]}] {m['role'].upper()}{tools}: {m['content']}")
        conversation = "\n".join(lines)
        current_memory = memory.read_long_term()

        prompt = f"""You are a memory consolidation agent. Process this conversation and return a JSON object with exactly two keys:

1. "history_entry": A paragraph (2-5 sentences) summarizing the key events/decisions/topics. Start with a timestamp like [YYYY-MM-DD HH:MM]. Include enough detail to be useful when found by grep search later.

2. "memory_update": The updated long-term memory content. Add any new facts: user location, preferences, personal info, habits, project context, technical decisions, tools/services used. If nothing new, return the existing content unchanged.

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{conversation}

Respond with ONLY valid JSON, no markdown fences."""

        try:
            response = await self.provider.chat(
                messages=[
                    {"role": "system", "content": "You are a memory consolidation agent. Respond only with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
            )
            text = (response.content or "").strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json.loads(text)

            if entry := result.get("history_entry"):
                memory.append_history(entry)
            if update := result.get("memory_update"):
                if update != current_memory:
                    memory.write_long_term(update)

            if archive_all:
                session.last_consolidated = 0
            else:
                session.last_consolidated = len(session.messages) - keep_count
            logger.info(f"Memory consolidation done: {len(session.messages)} messages, last_consolidated={session.last_consolidated}")
        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
    ) -> str:
        """
        Process a message directly (for CLI or cron usage).
        
        Args:
            content: The message content.
            session_key: Session identifier (overrides channel:chat_id for session lookup).
            channel: Source channel (for tool context routing).
            chat_id: Source chat ID (for tool context routing).
        
        Returns:
            The agent's response.
        """
        msg = InboundMessage(
            channel=channel,
            sender_id="user",
            chat_id=chat_id,
            content=content
        )
        
        response = await self._process_message(msg, session_key=session_key)
        return response.content if response else ""
