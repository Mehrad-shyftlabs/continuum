"""
Agent Runner - Executes agents with full observability.

The main entry point for running agents, handling tool calls,
handoffs, and conversation loops.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from orchestrator.agent.base import BaseAgent
from orchestrator.agent.config import RunnerConfig
from orchestrator.agent.exceptions import (
    AgentError,
    AgentExecutionError,
)
from orchestrator.agent.execution.executor import Executor
from orchestrator.agent.execution.handoff_executor import HandoffExecutor
from orchestrator.agent.execution.message_builder import MessageBuilder
from orchestrator.agent.execution.stream_executor import StreamExecutor
from orchestrator.agent.execution.tool_handler import ToolHandler
from orchestrator.agent.handoff.manager import HandoffManager
from orchestrator.agent.persistence.state import (
    RunStateManager,
)
from orchestrator.agent.services.context_service import ContextService
from orchestrator.agent.services.memory_service import MemoryService
from orchestrator.agent.services.session_service import SessionService
from orchestrator.agent.services.tool_service import ToolService
from orchestrator.agent.types import (
    AgentEvent,
    AgentResponse,
    EventType,
    ResponseStatus,
    RunContext,
    RunState,
    RunStatus,
    generate_run_id,
)
from orchestrator.agent.utils.context_utils import create_run_context
from orchestrator.agent.utils.message_utils import message_to_dict
from orchestrator.agent.utils.validation_utils import validate_input
from orchestrator.core.container import Container, get_container
from orchestrator.logging import get_logger
from orchestrator.observability.decorators import observe
from orchestrator.observability.error_reporter import report_error
from orchestrator.observability.metrics import get_metrics_collector
from orchestrator.observability.trace_context import (
    clear_trace_context,
    get_current_trace_client,
    get_current_trace_id,
    set_trace_context,
    truncate_data,
)

if TYPE_CHECKING:
    from orchestrator.llm import LLMClient
    from orchestrator.llm.types import ChatMessage
    from orchestrator.memory import MemoryClient
    from orchestrator.observability import TracingManager
    from orchestrator.observability.metrics import MetricsCollector
    from orchestrator.session import SessionClient
    from orchestrator.tools import ToolExecutor

logger = get_logger(__name__)


class AgentRunner:
    """
    Executes agents with full observability.

    The AgentRunner is the primary interface for running agents. It handles:
    - LLM calls with automatic retry and fallback
    - Tool execution
    - Agent handoffs
    - Memory retrieval and storage
    - Session management
    - Full Langfuse tracing
    - State persistence

    Example:
        ```python
        from orchestrator.agent import BaseAgent, AgentRunner

        agent = BaseAgent(
            name="support-agent",
            instructions="You are a helpful assistant.",
            tools=[search_tool],
        )

        runner = AgentRunner()

        # Run agent
        response = await runner.run(
            agent,
            "Hello, I need help!",
            user_id="user-123",
        )

        print(response.content)

        # Stream response
        async for event in runner.run_stream(agent, "Tell me a story"):
            if event.type == EventType.CONTENT_DELTA:
                print(event.data["content"], end="")
        ```
    """

    def __init__(
        self,
        container: Container | None = None,
        llm_client: LLMClient | None = None,
        memory_client: MemoryClient | None = None,
        session_client: SessionClient | None = None,
        tool_executor: ToolExecutor | None = None,
        tracing_manager: TracingManager | None = None,
        state_manager: RunStateManager | None = None,
        config: RunnerConfig | None = None,
        agent_registry: dict[str, BaseAgent] | None = None,
    ):
        """
        Initialize the agent runner.

        Uses Container (DI) by default for client management. Clients can be
        explicitly provided to override container defaults (useful for testing).

        Args:
            container: Dependency injection container (uses get_container() if None)
            llm_client: LLM client for model calls (overrides container)
            memory_client: Memory client for long-term memory (overrides container)
            session_client: Session client for conversation history (overrides container)
            tool_executor: Tool executor for MCP tools
            tracing_manager: Tracing manager for observability (overrides container)
            state_manager: State manager for run persistence
            config: Runner configuration
            agent_registry: Dictionary mapping agent names to agents
        """
        # Use Container (DI) by default
        self._container = container or get_container()

        # Use provided clients or get from container
        self._llm_client = llm_client or self._container.llm_client
        self._memory_client = memory_client or self._container.memory_client
        self._session_client = session_client or self._container.session_client
        self._tool_executor = tool_executor or self._container.tool_executor
        # Tracing manager is not in container, use provided or None
        self._tracing_manager = tracing_manager
        self._state_manager = state_manager
        self._config = config or RunnerConfig()
        self._agent_registry = agent_registry or {}

        # Handoff manager
        self._handoff_manager = HandoffManager(
            llm_client=self._llm_client,
            tracing_manager=self._tracing_manager,
        )

        # Initialize services
        self._context_service = ContextService(
            state_manager=self._state_manager,
            config=self._config,
        )
        self._memory_service = MemoryService(
            memory_client=self._memory_client,
            session_client=self._session_client,
        )
        self._session_service = SessionService(
            session_client=self._session_client,
        )
        self._tool_service = ToolService(
            tool_executor=self._tool_executor,
            config=self._config,
        )

        # Initialize execution components
        # Create handoff executor first (without executor reference)
        self._handoff_executor = HandoffExecutor(
            handoff_manager=self._handoff_manager,
            agent_registry=self._agent_registry,
        )

        self._tool_handler = ToolHandler(
            tool_service=self._tool_service,
        )

        # Create executor (will set itself as handoff executor's executor)
        self._executor = Executor(
            llm_client=self._llm_client,
            tool_handler=self._tool_handler,
            handoff_executor=self._handoff_executor,
        )

        # Now set executor reference in handoff executor for recursive execution
        self._handoff_executor._executor = self._executor

        # Register agents in handoff executor's registry
        for agent_obj in self._agent_registry.values():
            self._handoff_executor.register_agent(agent_obj)

        self._stream_executor = StreamExecutor(
            llm_client=self._llm_client,
        )

        self._message_builder = MessageBuilder(
            memory_service=self._memory_service,
            session_service=self._session_service,
        )

    def register_agent(self, agent: BaseAgent) -> None:
        """
        Register an agent for handoffs.

        Args:
            agent: Agent to register
        """
        self._agent_registry[agent.name] = agent
        # Also register in handoff executor
        if self._handoff_executor:
            self._handoff_executor.register_agent(agent)

    def get_agent(self, name: str) -> BaseAgent | None:
        """
        Get a registered agent by name.

        Args:
            name: Agent name

        Returns:
            Agent or None if not found
        """
        return self._agent_registry.get(name)

    @property
    def llm_client(self) -> LLMClient:
        """Get LLM client from container."""
        return self._llm_client

    @property
    def memory_client(self) -> MemoryClient | None:
        """Get memory client from container."""
        return self._memory_client

    @property
    def session_client(self) -> SessionClient | None:
        """Get session client from container."""
        return self._session_client

    @property
    def state_manager(self) -> RunStateManager:
        """Get state manager."""
        return self._context_service.state_manager

    async def _prepare_run(
        self,
        agent: BaseAgent,
        input: str | list[dict[str, Any]] | list[ChatMessage],
        session_id: str | None = None,
        user_id: str | None = None,
        context: RunContext | None = None,
        max_turns: int | None = None,
        trace_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> tuple[RunContext, RunState, int, Any]:
        """
        Prepare for agent run - shared setup for run() and run_stream().

        Returns:
            Tuple of (context, run_state, initial_message_count, tool_context_state)
        """
        # Register agent if not already registered
        if agent.name not in self._agent_registry:
            self.register_agent(agent)

        # Create or use existing context
        if context is None:
            context = create_run_context(
                session_id=session_id,
                user_id=user_id,
                trace_id=trace_id,
                max_turns=max_turns or agent.config.max_turns,
                metadata=metadata or {},
                tags=tags or [],
            )

        # Validate input against schema if defined
        if agent.input_schema is not None:
            validation_result = await validate_input(agent, input, context)
            if validation_result is not None:
                # Validation failed - return None to indicate error
                # Caller should handle the validation_result
                return None, None, 0, None

        # Initialize run state using context service
        run_state = await self._context_service.create_run_state(agent, context)

        # Build input preview for trace
        input_preview = input if isinstance(input, str) else str(input)[:500]

        # Start trace - CRITICAL: This sets global trace context
        await self._trace_run_start(agent, context, run_state, input_preview)

        # Load tool context state from session
        tool_context_state = None
        if context.session_id and self.session_client:
            tool_context_state = await self._session_service.load_tool_context_state(
                session_id=context.session_id,
                trace_id=context.trace_id,
            )

            # Log what we loaded
            if not tool_context_state.is_empty():
                all_namespaces = tool_context_state.get_all_namespaces()
                for namespace in all_namespaces:
                    mcp_session_id = tool_context_state.get(namespace, "session_id")
                    if mcp_session_id:
                        logger.info(
                            f"📥 Loaded MCP session_id from tool context: {mcp_session_id[:8]}... "
                            f"(namespace={namespace})"
                        )
                        break

            # Set context state on tool executors
            if agent.tool_executor and hasattr(agent.tool_executor, "context_state"):
                agent.tool_executor.context_state = tool_context_state
                logger.debug("Loaded tool context state into agent's tool executor")

            if self._tool_executor and hasattr(self._tool_executor, "context_state"):
                self._tool_executor.context_state = tool_context_state
                logger.debug("Loaded tool context state into global tool executor")

        # CLEAR run artifacts at start of each run
        if agent.tool_executor and hasattr(agent.tool_executor, "clear_run_artifacts"):
            agent.tool_executor.clear_run_artifacts(run_id=context.run_id)
            logger.debug(f"Cleared run artifacts for run_id={context.run_id}")
        if self._tool_executor and hasattr(self._tool_executor, "clear_run_artifacts"):
            self._tool_executor.clear_run_artifacts(run_id=context.run_id)

        # Run agent lifecycle hook
        if agent.on_start:
            agent.on_start(agent, {"context": context, "input": input})

        # Prepare messages
        messages = await self._message_builder.prepare_messages(
            agent,
            input,
            context,
            tool_context_state=tool_context_state,
        )
        run_state.messages = [message_to_dict(m) for m in messages]

        # Track message count before execution
        initial_message_count = len(messages)

        return context, run_state, initial_message_count, tool_context_state

    async def run(
        self,
        agent: BaseAgent,
        input: str | list[dict[str, Any]] | list[ChatMessage],
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        context: RunContext | None = None,
        max_turns: int | None = None,
        trace_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> AgentResponse:
        """
        Run an agent to completion.

        This is the main method for executing an agent. It handles the full
        conversation loop including tool calls and handoffs.

        Args:
            agent: Agent to run
            input: User input (string or messages)
            session_id: Session ID for conversation history
            user_id: User ID for memory scoping
            context: Existing run context (for nested calls)
            max_turns: Maximum conversation turns
            trace_id: Trace ID for Langfuse
            metadata: Additional metadata
            tags: Tags for tracing

        Returns:
            AgentResponse with the result

        Example:
            ```python
            response = await runner.run(
                agent,
                "What's the weather in NYC?",
                user_id="user-123",
                session_id="session-456",
            )
            print(response.content)
            ```
        """
        start_time = time.time()
        metrics = get_metrics_collector()

        # Shared setup
        setup_result = await self._prepare_run(
            agent, input, session_id, user_id, context, max_turns, trace_id, metadata, tags
        )
        if setup_result[0] is None:
            # Validation failed - return the error response
            # (validation_result is stored in context by validate_input)
            return AgentResponse(
                content="Input validation failed",
                agent_name=agent.name,
                status=ResponseStatus.ERROR,
                error="Input validation failed",
            )

        context, run_state, initial_message_count, tool_context_state = setup_result

        try:
            # Messages are already prepared in _prepare_run and stored in run_state
            # Reconstruct messages from run_state for execution
            messages = (
                [message_to_dict(m) for m in run_state.messages] if run_state.messages else []
            )

            # Main execution loop
            response = await self._executor.execute_loop(
                agent=agent,
                messages=messages,
                context=context,
                run_state=run_state,
            )

            # Run agent lifecycle hook
            if agent.on_end:
                agent.on_end(agent, {"context": context, "response": response})

            # Finalize response
            response.run_id = context.run_id
            response.latency_ms = int((time.time() - start_time) * 1000)
            response.trace_id = context.trace_id
            response.agents_used = list(set(run_state.agent_stack))
            response.handoff_chain = [h.get("to_agent", "") for h in run_state.handoff_chain]

            # Attach run artifacts (full MCP responses - widgets, structured data, etc.)
            # These are per-run and were cleared at start
            run_artifacts_dict = None
            if agent.tool_executor and hasattr(agent.tool_executor, "run_artifacts"):
                run_artifacts = agent.tool_executor.run_artifacts
                if not run_artifacts.is_empty():
                    run_artifacts_dict = run_artifacts.to_dict()
            elif self._tool_executor and hasattr(self._tool_executor, "run_artifacts"):
                run_artifacts = self._tool_executor.run_artifacts
                if not run_artifacts.is_empty():
                    run_artifacts_dict = run_artifacts.to_dict()

            if run_artifacts_dict:
                response.run_artifacts = run_artifacts_dict
                logger.debug(
                    f"Attached {len(run_artifacts_dict.get('tool_artifacts', []))} artifacts to response"
                )

            # Update state
            run_state.status = RunStatus.COMPLETED
            await self._context_service.save_run_state(run_state)

            # Check if MCP tools captured a new session_id
            # IMPORTANT: We track TWO session_ids:
            # - original_session_id: OUR session in Redis (for saving messages/context)
            # - mcp_session_id: External API session (for LLM awareness and tool injection)
            original_session_id = context.session_id  # Keep original for saving
            mcp_session_id = None
            updated_context_state = None

            # Try both tool executors - agent's and runner's
            if agent.tool_executor and hasattr(agent.tool_executor, "context_state"):
                updated_context_state = agent.tool_executor.context_state
                logger.debug("Using agent.tool_executor.context_state")
            elif self._tool_executor and hasattr(self._tool_executor, "context_state"):
                updated_context_state = self._tool_executor.context_state
                logger.debug("Using self._tool_executor.context_state")

            # Find MCP session_id from context_state
            if updated_context_state:
                all_namespaces = updated_context_state.get_all_namespaces()
                logger.debug(
                    f"🔍 Checking {len(all_namespaces)} namespaces for MCP session_id: {all_namespaces}"
                )

                for namespace in all_namespaces:
                    captured_session_id = updated_context_state.get(namespace, "session_id")
                    if captured_session_id:
                        mcp_session_id = captured_session_id
                        if mcp_session_id != original_session_id:
                            logger.debug(
                                f"🔄 Found MCP session_id (namespace={namespace}): {mcp_session_id[:8]}... "
                                f"(our session: {original_session_id[:8] if original_session_id else 'None'}...)"
                            )
                        break
            else:
                logger.debug("⚠️ No context_state found on tool executors")

            # Save to OUR session (original_session_id), NOT the MCP session
            # This ensures context is persisted and loaded on next request
            if agent.config.log_to_session and original_session_id and self.session_client:
                # Save messages to OUR session
                await self._session_service.save_messages(
                    agent=agent,  # Pass agent for memory config access
                    messages=response.messages or [],
                    initial_count=initial_message_count,
                    session_id=original_session_id,  # Use OUR session, not MCP
                    trace_id=context.trace_id,
                    tool_execution_summary=context.metadata.get("tool_execution_summary"),
                    run_id=context.run_id,  # Pass run_id for RUN-scoped memory isolation
                )

                # Save tool context state (including MCP session_id) to OUR session
                # This allows next request to know about the MCP session
                if updated_context_state:
                    await self._session_service.save_tool_context_state(
                        session_id=original_session_id,  # Use OUR session, not MCP
                        context_state=updated_context_state,
                        trace_id=context.trace_id,
                    )
                    logger.debug(f"💾 Saved tool context to session {original_session_id[:8]}...")

            # Record E2E latency
            e2e_latency_ms = (time.time() - start_time) * 1000
            metrics.record_latency(
                "agent_run_e2e",
                e2e_latency_ms,
                metadata={"agent_name": agent.name, "run_id": context.run_id},
            )

            # Record token usage if available
            if response.usage:
                metrics.track_tokens(
                    "agent_run",
                    prompt_tokens=response.usage.prompt_tokens or 0,
                    completion_tokens=response.usage.completion_tokens or 0,
                    model=agent.model,
                )

            # Report metrics to trace
            await self._report_metrics_to_trace(context, metrics)

            # End trace
            await self._trace_run_end(agent, context, response)

            return response

        except Exception as e:
            # Track error in metrics
            metrics.track_error(
                "agent_run", e, metadata={"agent_name": agent.name, "run_id": context.run_id}
            )

            # Handle error
            run_state.status = RunStatus.FAILED
            run_state.metadata["error"] = str(e)

            await self._context_service.save_run_state(run_state)

            # Run error hook
            if agent.on_error:
                agent.on_error(agent, e, {"context": context})

            # Report metrics even on error
            await self._report_metrics_to_trace(context, metrics)

            # Trace error with full context
            await self._trace_run_error(agent, context, e, run_state)

            # Re-raise as AgentError
            if isinstance(e, AgentError):
                raise
            raise AgentExecutionError(
                str(e),
                agent_name=agent.name,
                run_id=context.run_id,
                trace_id=context.trace_id,
                original_error=e,
            ) from e

    async def run_stream(
        self,
        agent: BaseAgent,
        input: str | list[dict[str, Any]],
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        max_turns: int | None = None,
        trace_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """
        Run an agent with streaming output.

        Yields events as the agent executes, including content deltas
        for real-time display.

        Args:
            agent: Agent to run
            input: User input
            session_id: Session ID
            user_id: User ID
            max_turns: Maximum turns
            trace_id: Trace ID
            metadata: Additional metadata

        Yields:
            AgentEvent for each step

        Example:
            ```python
            async for event in runner.run_stream(agent, "Tell me a story"):
                if event.type == EventType.CONTENT_DELTA:
                    print(event.data["content"], end="", flush=True)
                elif event.type == EventType.TOOL_CALL_START:
                    print(f"\\nCalling tool: {event.data['tool_name']}")
            ```
        """
        start_time = time.time()
        metrics = get_metrics_collector()

        # Shared setup
        setup_result = await self._prepare_run(
            agent, input, session_id, user_id, None, max_turns, trace_id, metadata, None
        )
        if setup_result[0] is None:
            # Validation failed, emit error event
            temp_context = RunContext(
                run_id=generate_run_id(),
                session_id=session_id,
                user_id=user_id,
                trace_id=trace_id,
                max_turns=max_turns or agent.config.max_turns,
                metadata=metadata or {},
            )
            yield AgentEvent(
                type=EventType.RUN_ERROR,
                agent_name=agent.name,
                run_id=temp_context.run_id,
                data={"error": "Input validation failed", "error_type": "ValidationError"},
                trace_id=temp_context.trace_id,
            )
            return

        context, run_state, initial_message_count, tool_context_state = setup_result

        # Emit run start
        yield AgentEvent(
            type=EventType.RUN_START,
            agent_name=agent.name,
            run_id=context.run_id,
            data={"input": input if isinstance(input, str) else "[messages]"},
            trace_id=context.trace_id,
        )

        try:
            # Messages are already prepared in _prepare_run
            messages = (
                [message_to_dict(m) for m in run_state.messages] if run_state.messages else []
            )

            # Emit agent start
            yield AgentEvent(
                type=EventType.AGENT_START,
                agent_name=agent.name,
                run_id=context.run_id,
                trace_id=context.trace_id,
            )

            turn = 0
            while turn < context.max_turns:
                turn += 1

                # Get tools
                tools = agent.get_tools_for_llm()

                # Stream LLM response
                content_parts = []
                tool_calls = []

                # NOTE: session_id intentionally omitted - AgentRunner manages the message loop
                # See Executor.execute_loop() for detailed explanation
                # Trace context is automatically captured from contextvars via @observe decorator
                async for chunk in self.llm_client.chat_stream(
                    messages=messages,
                    tools=tools if tools else None,
                    trace_metadata={"session_id": session_id} if session_id else None,
                ):
                    if chunk.content:
                        content_parts.append(chunk.content)
                        yield AgentEvent(
                            type=EventType.CONTENT_DELTA,
                            agent_name=agent.name,
                            run_id=context.run_id,
                            data={"content": chunk.content},
                            trace_id=context.trace_id,
                        )

                    if chunk.tool_calls:
                        tool_calls = chunk.tool_calls

                content = "".join(content_parts)

                # Emit content complete
                if content:
                    yield AgentEvent(
                        type=EventType.CONTENT_COMPLETE,
                        agent_name=agent.name,
                        run_id=context.run_id,
                        data={"content": content},
                        trace_id=context.trace_id,
                    )

                # Handle tool calls
                if tool_calls:
                    # Add assistant message
                    messages.append(
                        {
                            "role": "assistant",
                            "content": content or None,
                            "tool_calls": [
                                tc.to_dict() if hasattr(tc, "to_dict") else tc for tc in tool_calls
                            ],
                        }
                    )

                    for tc in tool_calls:
                        tool_name = (
                            tc.function.name
                            if hasattr(tc, "function")
                            else tc.get("function", {}).get("name", "")
                        )
                        tool_call_id = tc.id if hasattr(tc, "id") else tc.get("id", "")

                        # Check for handoff
                        is_handoff, target = agent.is_handoff_tool_call(tool_name)
                        if is_handoff and target:
                            yield AgentEvent(
                                type=EventType.HANDOFF_START,
                                agent_name=agent.name,
                                run_id=context.run_id,
                                data={"target": target},
                                trace_id=context.trace_id,
                            )
                            # Add tool result for the handoff (simplified - streaming doesn't fully support handoffs)
                            # This ensures the message sequence is valid (tool_calls followed by tool results)
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "content": f"Handoff to {target} initiated. Note: Full handoff support requires non-streaming mode.",
                                }
                            )
                            yield AgentEvent(
                                type=EventType.HANDOFF_END,
                                agent_name=agent.name,
                                run_id=context.run_id,
                                data={
                                    "target": target,
                                    "note": "Streaming mode has limited handoff support",
                                },
                                trace_id=context.trace_id,
                            )
                            continue

                        # Execute tool
                        yield AgentEvent(
                            type=EventType.TOOL_CALL_START,
                            agent_name=agent.name,
                            run_id=context.run_id,
                            data={"tool_name": tool_name},
                            trace_id=context.trace_id,
                        )

                        try:
                            result, exec_metadata = await self._tool_service.execute_tool_call(
                                agent, tc, context
                            )
                            messages.append(result)

                            yield AgentEvent(
                                type=EventType.TOOL_CALL_END,
                                agent_name=agent.name,
                                run_id=context.run_id,
                                data={
                                    "tool_name": tool_name,
                                    "result": result.get("content", "")[:500],
                                },
                                trace_id=context.trace_id,
                            )
                        except Exception as e:
                            yield AgentEvent(
                                type=EventType.TOOL_CALL_ERROR,
                                agent_name=agent.name,
                                run_id=context.run_id,
                                data={"tool_name": tool_name, "error": str(e)},
                                trace_id=context.trace_id,
                            )

                    # Continue loop for next turn
                    continue

                # No tool calls, we're done
                break

            # Update state (same as run())
            run_state.status = RunStatus.COMPLETED
            await self._context_service.save_run_state(run_state)

            # Run agent lifecycle hook (same as run())
            if agent.on_end:
                # Create a minimal response for the hook
                response = AgentResponse(
                    content=content,
                    run_id=context.run_id,
                    agent_name=agent.name,
                    status=ResponseStatus.SUCCESS,
                    trace_id=context.trace_id,
                )
                agent.on_end(agent, {"context": context, "response": response})

            # Check if MCP tools captured a new session_id (same as run())
            # IMPORTANT: We track TWO session_ids:
            # - original_session_id: OUR session in Redis (for saving messages/context)
            # - mcp_session_id: External API session (for LLM awareness and tool injection)
            original_session_id = context.session_id  # Keep original for saving
            mcp_session_id = None
            updated_context_state = None

            # Try both tool executors - agent's and runner's (same as run())
            if agent.tool_executor and hasattr(agent.tool_executor, "context_state"):
                updated_context_state = agent.tool_executor.context_state
                logger.debug("Using agent.tool_executor.context_state")
            elif self._tool_executor and hasattr(self._tool_executor, "context_state"):
                updated_context_state = self._tool_executor.context_state
                logger.debug("Using self._tool_executor.context_state")

            # Find MCP session_id from context_state (same as run())
            if updated_context_state:
                all_namespaces = updated_context_state.get_all_namespaces()
                logger.debug(
                    f"🔍 Checking {len(all_namespaces)} namespaces for MCP session_id: {all_namespaces}"
                )

                for namespace in all_namespaces:
                    captured_session_id = updated_context_state.get(namespace, "session_id")
                    if captured_session_id:
                        mcp_session_id = captured_session_id
                        if mcp_session_id != original_session_id:
                            logger.debug(
                                f"🔄 Found MCP session_id (namespace={namespace}): {mcp_session_id[:8]}... "
                                f"(our session: {original_session_id[:8] if original_session_id else 'None'}...)"
                            )
                        break
            else:
                logger.debug("⚠️ No context_state found on tool executors")

            # Use updated_context_state if available, otherwise use the one we loaded
            final_context_state = (
                updated_context_state if updated_context_state else tool_context_state
            )

            # Save messages to session (same as run() method)
            # This ensures memories are stored even in streaming mode
            if agent.config.log_to_session and original_session_id and self.session_client:
                # Save messages to session
                # Note: We need to reconstruct the final messages list
                # The messages list already contains all conversation messages
                try:
                    await self._session_service.save_messages(
                        agent=agent,  # Pass agent for memory config access
                        messages=messages,  # Use the messages list from the loop
                        initial_count=initial_message_count,
                        session_id=original_session_id,  # Use OUR session, not MCP
                        trace_id=context.trace_id,
                        tool_execution_summary=context.metadata.get("tool_execution_summary"),
                        run_id=context.run_id,
                    )
                    logger.debug(
                        f"💾 Saved messages to session {original_session_id[:8]}... (streaming mode)"
                    )
                except Exception as e:
                    logger.warning(f"Failed to save messages to session in streaming mode: {e}")

            # Save tool context state (including MCP session_id) to OUR session (same as run())
            # This allows next request to know about the MCP session
            if final_context_state and not final_context_state.is_empty() and original_session_id:
                try:
                    await self._session_service.save_tool_context_state(
                        session_id=original_session_id,  # Use OUR session, not MCP
                        context_state=final_context_state,
                        trace_id=context.trace_id,
                    )
                    logger.debug(
                        f"💾 Saved tool context to session {original_session_id[:8]}... (streaming mode)"
                    )
                except Exception as e:
                    logger.warning(f"Failed to save tool context in streaming mode: {e}")

            # Record E2E latency (same as run())
            e2e_latency_ms = (time.time() - start_time) * 1000
            metrics.record_latency(
                "agent_run_stream_e2e",
                e2e_latency_ms,
                metadata={"agent_name": agent.name, "run_id": context.run_id},
            )

            # Report metrics to trace (same as run())
            await self._report_metrics_to_trace(context, metrics)

            # End trace (same as run())
            response = AgentResponse(
                content=content,
                run_id=context.run_id,
                agent_name=agent.name,
                status=ResponseStatus.SUCCESS,
                trace_id=context.trace_id,
            )
            await self._trace_run_end(agent, context, response)

            # Emit agent end
            yield AgentEvent(
                type=EventType.AGENT_END,
                agent_name=agent.name,
                run_id=context.run_id,
                data={"turn_count": turn},
                trace_id=context.trace_id,
            )

            # Emit run end
            yield AgentEvent(
                type=EventType.RUN_END,
                agent_name=agent.name,
                run_id=context.run_id,
                data={"content": content, "turn_count": turn},
                trace_id=context.trace_id,
            )

        except Exception as e:
            # Track error in metrics (same as run())
            metrics.track_error(
                "agent_run_stream", e, metadata={"agent_name": agent.name, "run_id": context.run_id}
            )

            # Handle error (same as run())
            run_state.status = RunStatus.FAILED
            run_state.metadata["error"] = str(e)

            await self._context_service.save_run_state(run_state)

            # Run error hook (same as run())
            if agent.on_error:
                agent.on_error(agent, e, {"context": context})

            # Report metrics even on error (same as run())
            await self._report_metrics_to_trace(context, metrics)

            # Trace error with full context (same as run())
            await self._trace_run_error(agent, context, e, run_state)

            # Emit error event
            yield AgentEvent(
                type=EventType.RUN_ERROR,
                agent_name=agent.name,
                run_id=context.run_id,
                data={"error": str(e), "error_type": type(e).__name__},
                trace_id=context.trace_id,
            )

            # Re-raise as AgentError (same as run())
            if isinstance(e, AgentError):
                raise
            raise AgentExecutionError(
                str(e),
                agent_name=agent.name,
                run_id=context.run_id,
                trace_id=context.trace_id,
                original_error=e,
            ) from e

    # Tracing methods - kept as they're still used by the runner

    # NOTE: NOT decorated with @observe because this method CREATES the trace.
    # Decorating it would try to create a span before the trace exists.
    async def _trace_run_start(
        self,
        agent: BaseAgent,
        context: RunContext,
        run_state: RunState,
        input_preview: str = "",
    ) -> None:
        """
        Create trace and set trace context for the run.

        CRITICAL: This method creates the trace and sets the global trace context
        so all child operations (session, memory, tools) can create spans under this trace.
        """
        try:
            # CRITICAL: Always check for existing trace context first
            # This ensures handoffs and nested agent calls use the same trace
            existing_trace_id = get_current_trace_id()
            if existing_trace_id:
                # Use existing trace context (from parent agent or initial query)
                context.trace_id = existing_trace_id
                # Get trace client from context if available
                trace_client = get_current_trace_client()
                if trace_client:
                    context._langfuse_trace = trace_client
                logger.debug(f"Using existing trace context: {existing_trace_id}")
            elif not context.trace_id:
                # Create new trace using TracingManager (returns Trace wrapper with consistent interface)
                from orchestrator.observability import TracingManager

                tracing_manager = TracingManager()
                trace = tracing_manager.create_trace(
                    name=f"agent-run-{agent.name}",
                    user_id=context.user_id,
                    session_id=context.session_id,
                    input=truncate_data({"query": input_preview[:500]}),
                    metadata={
                        "run_id": context.run_id,
                        "agent_name": agent.name,
                        "model": agent.model,
                        "max_turns": context.max_turns,
                    },
                    tags=context.tags + agent.tags,
                )
                if trace:
                    context.trace_id = trace.id
                    context._langfuse_trace = trace.langfuse_trace

            # CRITICAL: Set trace context globally so all child operations can find it
            set_trace_context(
                trace_id=context.trace_id,
                trace_client=getattr(context, "_langfuse_trace", None),
                user_id=context.user_id,
                session_id=context.session_id,
                agent_name=agent.name,
                run_id=context.run_id,
            )

            logger.debug(f"Trace context set: trace_id={context.trace_id}")

        except Exception as e:
            logger.warning(f"Failed to set trace context: {e}")

    # NOTE: Decorated with @observe - trace exists at this point
    @observe(name="trace_run_end", capture_output=False)
    async def _trace_run_end(
        self,
        agent: BaseAgent,
        context: RunContext,
        response: AgentResponse,
    ) -> None:
        """Update trace with final output and clear trace context."""
        try:
            # Update trace with final output if available
            trace = getattr(context, "_langfuse_trace", None)
            if trace:
                try:
                    trace.update(
                        output=truncate_data(
                            {
                                "response": response.content[:1000] if response.content else None,
                                "status": response.status.value,
                            }
                        ),
                        metadata={
                            "run_id": context.run_id,
                            "agent_name": agent.name,
                            "status": response.status.value,
                            "turn_count": response.turn_count,
                            "latency_ms": response.latency_ms,
                            "agents_used": response.agents_used,
                            "handoff_chain": response.handoff_chain,
                            "usage": response.usage.to_dict() if response.usage else {},
                        },
                    )
                except Exception as e:
                    logger.debug(f"Failed to update trace: {e}")

            # Clear the global trace context
            clear_trace_context()
            logger.debug(f"Trace context cleared for trace_id={context.trace_id}")

        except Exception as e:
            logger.warning(f"Failed to trace run end: {e}")
            # Always try to clear context
            try:
                clear_trace_context()
            except Exception:
                pass

    # NOTE: Decorated with @observe - trace exists at this point
    @observe(name="trace_run_error", capture_output=False)
    async def _trace_run_error(
        self,
        agent: BaseAgent,
        context: RunContext,
        error: Exception,
        run_state: RunState | None = None,
    ) -> None:
        """Report error to trace and clear trace context."""
        try:
            # Build comprehensive error context
            error_metadata = {
                "run_id": context.run_id,
                "agent_name": agent.name,
                "error_type": type(error).__name__,
                "error_message": str(error)[:500],
                "session_id": context.session_id,
                "user_id": context.user_id,
            }

            # Add run state context if available
            if run_state:
                error_metadata.update(
                    {
                        "current_turn": run_state.turn_count,
                        "agent_stack": run_state.agent_stack,
                        "handoff_chain": [h.get("to_agent") for h in run_state.handoff_chain],
                        "status": run_state.status.value if run_state.status else "unknown",
                    }
                )

            # Add available tools context
            if agent.tools:
                error_metadata["available_tools"] = [t.name for t in agent.tools[:10]]

            # Report error (this will be captured by @observe decorator as well)
            report_error(
                error,
                context="agent_run",
                trace_id=context.trace_id,
                user_id=context.user_id,
                session_id=context.session_id,
                metadata=error_metadata,
            )

            # Update trace with error status if available
            trace = getattr(context, "_langfuse_trace", None)
            if trace:
                try:
                    trace.update(
                        output={"error": str(error)[:500]},
                        level="ERROR",
                        status_message=str(error)[:200],
                    )
                except Exception:
                    pass

            # Clear the global trace context
            clear_trace_context()

        except Exception as e:
            logger.warning(f"Failed to trace run error: {e}")
            try:
                clear_trace_context()
            except Exception:
                pass

    @observe(name="report_metrics", capture_output=False)
    async def _report_metrics_to_trace(
        self,
        context: RunContext,
        metrics: MetricsCollector,
    ) -> None:
        """
        Report collected metrics to the Langfuse trace.

        Args:
            context: Run context with trace info
            metrics: MetricsCollector with collected metrics
        """
        try:
            trace = getattr(context, "_langfuse_trace", None)
            if trace:
                metrics.report_to_trace(trace)
                logger.debug(f"Metrics reported to trace: {context.trace_id}")
        except Exception as e:
            logger.warning(f"Failed to report metrics to trace: {e}")
