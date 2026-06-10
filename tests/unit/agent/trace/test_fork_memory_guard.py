"""
Fork memory-write suppression: a forked (what-if) run must never persist facts
to long-term memory, even with store_memories=True. Enforced via the
disable_memory_writes flag (set by fork) honored in SessionService.save_messages.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from continuum.agent.base import BaseAgent
from continuum.agent.config import AgentConfig, AgentMemoryConfig
from continuum.agent.services.session_service import SessionService


def _agent_storing() -> BaseAgent:
    return BaseAgent(
        name="a",
        instructions="x",
        config=AgentConfig(),
        memory_config=AgentMemoryConfig(store_memories=True),
    )


async def _run(disable_memory: bool) -> bool:
    """Returns the store_in_memory value add_message was called with."""
    client = SimpleNamespace(is_enabled=True, add_message=AsyncMock())
    svc = SessionService(session_client=client)
    await svc.save_messages(
        agent=_agent_storing(),
        messages=[{"role": "assistant", "content": "ORD-1004 approved"}],
        user_message_index=0,
        session_id="sess-1",
        disable_memory=disable_memory,
    )
    assert client.add_message.await_count == 1
    return client.add_message.await_args.kwargs["store_in_memory"]


async def test_disable_memory_suppresses_store() -> None:
    # Normal run with store_memories=True → stores to memory.
    assert await _run(disable_memory=False) is True
    # Fork (disable_memory=True) → memory write suppressed despite store_memories=True.
    assert await _run(disable_memory=True) is False
