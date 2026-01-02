"""
Memory Service - Handles memory integration for agents.

Extracted from AgentRunner to provide clean separation of concerns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from orchestrator.agent.interfaces.service_interface import IMemoryService
from orchestrator.logging import get_logger
from orchestrator.observability.decorators import observe

if TYPE_CHECKING:
    from orchestrator.agent.base import BaseAgent
    from orchestrator.agent.types import RunContext

logger = get_logger(__name__)


class MemoryService(IMemoryService):
    """
    Service for memory integration.

    Handles retrieving and storing memories for agents.
    """

    def __init__(
        self,
        memory_client: Any | None = None,
        session_client: Any | None = None,
    ):
        """
        Initialize memory service.

        Args:
            memory_client: Memory client instance
            session_client: Session client for metadata access
        """
        self._memory_client = memory_client
        self._session_client = session_client

    @property
    def memory_client(self) -> Any | None:
        """Get memory client."""
        return self._memory_client

    @observe(name="retrieve_memories", capture_output=True)
    async def retrieve_memories(
        self,
        agent: BaseAgent,
        query: str,
        context: RunContext,
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant memories for the agent.

        Args:
            agent: Agent requesting memories
            query: Search query
            context: Run context

        Returns:
            List of memory dictionaries
        """
        if not agent.memory_config.search_memories or not self._memory_client:
            logger.debug(
                f"💾 Skipping memory search: search_memories={agent.memory_config.search_memories}, "
                f"memory_client={'available' if self._memory_client else 'not available'}"
            )
            return []

        try:
            # Determine search scope based on agent config
            search_scope = agent.memory_config.search_scope.value

            # Get memory client's isolation level to determine required identifiers
            memory_isolation = self._memory_client.config.memory_isolation

            # Mode-aware identifier selection
            user_id_for_memory = context.user_id if memory_isolation == "user" else None

            # CRITICAL: For agent isolation mode, get agent_id from session metadata if available.
            agent_id_for_memory = None
            if memory_isolation == "agent":
                # Try to get agent_id from session metadata first (most accurate)
                if context.session_id and self._session_client and self._session_client.is_enabled:
                    try:
                        session_metadata = await self._session_client.get_session_metadata(
                            context.session_id
                        )
                        if session_metadata and session_metadata.agent_id:
                            agent_id_for_memory = session_metadata.agent_id
                            logger.debug(
                                f"🔍 Using agent_id from session metadata: {agent_id_for_memory} "
                                f"(session_id={context.session_id[:8]}...)"
                            )
                        else:
                            logger.warning(
                                f"⚠️ Session {context.session_id[:8]}... exists but has no agent_id in metadata. "
                                f"Falling back to agent.name={agent.name}. This may cause memory isolation issues."
                            )
                            agent_id_for_memory = agent.name
                    except Exception as e:
                        logger.debug(f"Could not get session metadata for agent_id: {e}")
                        agent_id_for_memory = agent.name
                else:
                    # No session_id or session client not available - use agent.name
                    agent_id_for_memory = agent.name

                # Log final agent_id being used
                if agent_id_for_memory != agent.name:
                    logger.debug(
                        f"🔍 Agent isolation mode: Using agent_id={agent_id_for_memory} "
                        f"(agent.name={agent.name}, may differ when switching agents)"
                    )

            run_id_for_memory = context.run_id if memory_isolation == "run" else None

            # CRITICAL: Filter by session_id/run_id in metadata when agent scope is RUN
            filters = None
            if search_scope == "run":
                # Only filter by session_id when memory isolation is "run"
                if memory_isolation == "run":
                    # Use session_id for filtering (stored in metadata as run_id during storage)
                    filter_run_id = context.session_id or context.run_id
                    if filter_run_id:
                        # Use dict for native Qdrant metadata filtering
                        filters = {"run_id": filter_run_id}
                        logger.debug(
                            f"🔍 Applying RUN-scope filter: session_id={filter_run_id[:8]}... "
                            f"(memory_isolation={memory_isolation}, session-level isolation enabled)"
                        )
                else:
                    # USER/AGENT/SHARED modes: No session filter - memories shared across sessions
                    logger.debug(
                        f"🔍 RUN scope with {memory_isolation} mode: No session filter "
                        f"(memories shared across sessions for {memory_isolation} isolation)"
                    )

            # Log memory search parameters at INFO level
            logger.info(
                f"🔍 MEMORY SEARCH: query='{query[:100]}...', "
                f"scope={search_scope}, isolation={memory_isolation}, "
                f"user_id={user_id_for_memory[:8] if user_id_for_memory else 'none'}, "
                f"agent_id={agent_id_for_memory[:8] if agent_id_for_memory else 'none'}, "
                f"run_id={run_id_for_memory[:8] if run_id_for_memory else 'none'}, "
                f"session_id={context.session_id[:8] if context.session_id else 'none'}, "
                f"filters={filters}"
            )

            memories = await self._memory_client.search(
                query=query,
                user_id=user_id_for_memory,
                agent_id=agent_id_for_memory,
                run_id=run_id_for_memory,
                limit=agent.memory_config.search_limit,
                filters=filters,  # Native Qdrant metadata filters
            )

            # Log search results at INFO level
            logger.info(
                f"💾 MEMORY SEARCH RESULT: found {len(memories.results)} memories "
                f"(total_results={memories.total_results if hasattr(memories, 'total_results') else 'N/A'})"
            )

            if not memories.results:
                logger.warning(
                    f"⚠️ NO MEMORIES FOUND for query='{query[:100]}...' "
                    f"(isolation={memory_isolation}, user_id={user_id_for_memory[:8] if user_id_for_memory else 'none'}, "
                    f"agent_id={agent_id_for_memory[:8] if agent_id_for_memory else 'none'}, "
                    f"run_id={run_id_for_memory[:8] if run_id_for_memory else 'none'}, "
                    f"filters={filters})"
                )

            if memories.results:
                context.retrieved_memories = [m.to_dict() for m in memories.results]

                # Log memory search summary at INFO level
                logger.info(
                    f"💾 Memory search: scope={search_scope}, isolation={memory_isolation}, "
                    f"user_id={context.user_id[:8] if context.user_id else 'none'}, "
                    f"agent_id={agent_id_for_memory[:8] if agent_id_for_memory else 'N/A'}, "
                    f"run_id={context.run_id[:8] if context.run_id else 'none'}, "
                    f"session_id={context.session_id[:8] if context.session_id else 'none'}, "
                    f"found={len(memories.results)} memories, filter_applied={filters is not None}"
                )

                # Log each memory with its metadata to verify isolation (INFO level)
                for idx, m in enumerate(memories.results, 1):
                    memory_metadata = getattr(m, "metadata", {}) or {}
                    memory_run_id = (
                        m.run_id
                        or memory_metadata.get("run_id")
                        or memory_metadata.get("session_id")
                        or "unknown"
                    )
                    memory_session_id = memory_metadata.get("session_id") or "unknown"
                    memory_user_id = m.user_id or memory_metadata.get("_user_id") or "unknown"

                    # Verify session_id/run_id match for RUN scope
                    expected_id = context.session_id or context.run_id
                    run_id_match = (
                        "✓"
                        if (
                            expected_id
                            and memory_run_id != "unknown"
                            and (memory_run_id == expected_id or memory_session_id == expected_id)
                        )
                        else "✗"
                    )

                    logger.info(
                        f"📝 Memory #{idx}: '{m.memory[:100]}...' "
                        f"(score={m.score:.3f}, run_id={run_id_match} {memory_run_id[:8] if memory_run_id != 'unknown' else 'unknown'}, "
                        f"session_id={memory_session_id[:8] if memory_session_id != 'unknown' else 'unknown'}, "
                        f"user_id={memory_user_id[:8] if memory_user_id != 'unknown' else 'unknown'}, "
                        f"expected_id={expected_id[:8] if expected_id else 'none'})"
                    )

                return context.retrieved_memories

            return []

        except Exception as e:
            logger.warning(f"❌ Failed to retrieve memories: {e}", exc_info=True)
            return []

    @observe(name="store_memories", capture_output=False)
    async def store_memories(
        self,
        agent: BaseAgent,
        messages: list[dict[str, Any]],
        context: RunContext,
    ) -> None:
        """
        Store memories from conversation.

        Note: Memory storage is handled by the session service when saving messages.
        This method is kept for interface compatibility but delegates to session.

        Args:
            agent: Agent storing memories
            messages: Conversation messages
            context: Run context
        """
        # Memory storage is handled by SessionService.save_messages()
        # This method exists for interface compatibility
        logger.debug("Memory storage is handled by session service during message save")
