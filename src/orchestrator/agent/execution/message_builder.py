"""
Message Builder - Prepares messages for agent execution.

Extracted from AgentRunner to provide clean separation of concerns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from orchestrator.agent.interfaces.handler_interface import IMessageBuilder
from orchestrator.logging import get_logger
from orchestrator.observability.decorators import observe

if TYPE_CHECKING:
    from orchestrator.agent.base import BaseAgent
    from orchestrator.agent.services.memory_service import MemoryService
    from orchestrator.agent.services.session_service import SessionService
    from orchestrator.agent.types import RunContext
    from orchestrator.tools.types import ToolContextState

logger = get_logger(__name__)


class MessageBuilder(IMessageBuilder):
    """
    Builder for preparing messages for agent execution.

    Handles:
    - System prompt injection
    - Tool context injection
    - Memory retrieval and injection
    - Session history loading
    - Context compression
    """

    def __init__(
        self,
        memory_service: MemoryService | None = None,
        session_service: SessionService | None = None,
    ):
        """
        Initialize message builder.

        Args:
            memory_service: Memory service for retrieving memories
            session_service: Session service for loading history
        """
        self._memory_service = memory_service
        self._session_service = session_service

    @observe(name="prepare_messages", capture_output=True)
    async def prepare_messages(
        self,
        agent: BaseAgent,
        input: str | list[dict[str, Any]] | list[Any],
        context: RunContext,
        tool_context_state: ToolContextState | None = None,
    ) -> list[dict[str, Any]]:
        """
        Prepare messages for agent execution.

        Args:
            agent: Agent to prepare messages for
            input: User input (string or messages)
            context: Run context
            tool_context_state: Optional tool context state

        Returns:
            Prepared message list
        """
        messages = []

        # Log agent memory config at start
        if hasattr(agent, "memory_config") and agent.memory_config:
            logger.info(
                f"🔍 AGENT MEMORY CONFIG: "
                f"search_memories={agent.memory_config.search_memories}, "
                f"store_memories={agent.memory_config.store_memories}, "
                f"search_scope={agent.memory_config.search_scope}, "
                f"store_scope={agent.memory_config.store_scope}, "
                f"search_limit={agent.memory_config.search_limit}"
            )
        else:
            logger.warning("⚠️ Agent has no memory_config!")

        # Add system prompt
        if agent.system_prompt:
            messages.append({"role": "system", "content": agent.system_prompt})

        # Inject tool context into system prompt for LLM awareness
        if tool_context_state and not tool_context_state.is_empty():
            context_prompt = self._inject_tool_context_to_prompt(tool_context_state)
            if context_prompt:
                messages.append({"role": "system", "content": context_prompt})
                logger.info(
                    "📋 Injected tool context into system prompt (existing session_id found)"
                )

        # Retrieve memories if enabled
        if agent.memory_config.search_memories and self._memory_service:
            try:
                query = input if isinstance(input, str) else str(input)
                memories = await self._memory_service.retrieve_memories(agent, query, context)

                if memories:
                    # Add memories as context
                    memory_content = "Relevant information from memory:\n"
                    for m in memories:
                        memory_content += f"- {m.get('memory', str(m))}\n"

                    logger.info(f"💾 Injecting {len(memories)} memories into LLM context")
                    logger.debug(f"💾 Memory context content:\n{memory_content}")

                    messages.append({"role": "system", "content": memory_content})
            except Exception as e:
                logger.warning(f"❌ Failed to retrieve memories: {e}", exc_info=True)

        # Load session history if available
        if context.session_id and self._session_service:
            try:
                history = await self._session_service.get_conversation_history(
                    context.session_id, limit=50
                )
                messages.extend(history)
            except Exception as e:
                logger.warning(f"Failed to load session history: {e}")

        # Add user input
        if isinstance(input, str):
            messages.append({"role": "user", "content": input})
        elif isinstance(input, list):
            for item in input:
                messages.append(self._message_to_dict(item))

        # Apply context management (proactive compression) if enabled
        try:
            from orchestrator.llm.context_management import (
                ContextManagementConfig,
                get_progressive_context_manager,
            )

            # Get agent-specific config or use global defaults
            context_config = None
            if agent.config and agent.config.context_management:
                context_config = agent.config.context_management
            else:
                context_config = ContextManagementConfig()

            if context_config.enabled:
                context_manager = get_progressive_context_manager(config=context_config)
                messages, compression_result = await context_manager.compress_if_needed(
                    messages=messages,
                    model=agent.model,
                    config=context_config,
                )

                if compression_result.was_compressed:
                    logger.info(
                        f"Agent {agent.name}: Context compressed proactively - "
                        f"{compression_result.original_token_count} → {compression_result.compressed_token_count} tokens "
                        f"({compression_result.compression_ratio:.1%} ratio, strategy: {compression_result.strategy_used})"
                    )
        except Exception as e:
            logger.warning(
                f"Context management failed for agent {agent.name}, continuing without compression: {e}"
            )

        return messages

    def _inject_tool_context_to_prompt(
        self,
        context_state: ToolContextState,
    ) -> str | None:
        """
        Generate system prompt injection for tool context awareness.

        Args:
            context_state: Tool context state with captured variables

        Returns:
            Context string to inject into system prompt, or None if empty
        """
        if context_state.is_empty():
            return None

        base_context = context_state.to_prompt_context()

        # Check if we have a session_id - if so, tell LLM not to create a new one
        has_session_id = False
        for namespace in context_state.get_all_namespaces():
            if context_state.get(namespace, "session_id"):
                has_session_id = True
                break

        if has_session_id:
            return (
                f"{base_context}\n\n"
                "IMPORTANT: A session already exists. Do NOT call create_session again. "
                "Use the existing session_id for all tool calls that require it."
            )

        return base_context

    def _message_to_dict(self, message: Any) -> dict[str, Any]:
        """Convert a message to dictionary format."""
        if isinstance(message, dict):
            return message
        if hasattr(message, "to_dict"):
            return message.to_dict()
        if hasattr(message, "model_dump"):
            return message.model_dump()
        return {"role": "user", "content": str(message)}
