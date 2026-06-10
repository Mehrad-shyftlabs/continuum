"""
End-to-end test for fork + diff — real LLM, real Redis, full runner.

Runs a real agent with checkpointing on, forks the run from its first step with
an instruction override, and diffs the two runs. Nothing is faked.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.e2e

from tests.e2e.conftest import skip_if_no_api_key as _skip_if_no_api_key
from tests.e2e.conftest import skip_on_api_error as _skip_on_api_error


@pytest.fixture
def trace_checkpoint_enabled(monkeypatch):
    from continuum.config import settings

    monkeypatch.setattr(settings, "decision_trace_enabled", True)
    monkeypatch.setattr(settings, "decision_trace_detail", "full")
    monkeypatch.setattr(settings, "decision_trace_store", "redis")
    monkeypatch.setattr(settings, "decision_trace_checkpoint", True)

    from continuum.agent.trace import config as trace_config

    trace_config.get_trace_store.cache_clear()
    yield
    trace_config.get_trace_store.cache_clear()


class TestForkE2E:
    @_skip_on_api_error
    async def test_fork_with_override_changes_outcome(self, trace_checkpoint_enabled):
        _skip_if_no_api_key()

        from continuum.agent.base import BaseAgent
        from continuum.agent.config import AgentConfig, AgentMemoryConfig
        from continuum.agent.runner import AgentRunner
        from continuum.agent.trace import diff_traces
        from continuum.agent.trace.config import get_trace_store
        from continuum.agent.types import RunContext

        agent = BaseAgent(
            name="fork-e2e-agent",
            instructions="You are a helpful assistant. Be extremely concise.",
            memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
            config=AgentConfig(log_to_session=False),
        )
        runner = AgentRunner()
        runner.register_agent(agent)

        # Original run.
        ctx = RunContext(run_id="e2e-fork-parent")
        original = await runner.run(agent, "Reply with one word: hello.", context=ctx)
        assert original.decision_trace is not None
        first_step = original.decision_trace["steps"][0]["step_id"]

        # Fork from the first step, overriding the instruction.
        forked = await runner.fork(
            "e2e-fork-parent",
            first_step,
            override={"system": "You are a pirate. Reply in one word of pirate speak."},
            agent=agent,
        )
        assert forked.decision_trace is not None
        assert forked.decision_trace["parent_run_id"] == "e2e-fork-parent"
        assert forked.decision_trace["forked_from_step"] == first_step

        # The two runs are persisted independently; diff them.
        store = get_trace_store()
        a = await store.get("e2e-fork-parent")
        b = await store.get(forked.decision_trace["run_id"])
        assert a is not None and b is not None
        d = diff_traces(a, b)
        # The override changed behavior, so the final answers should differ.
        assert d["final_response"]["before"] == original.content
        assert d["final_response"]["after"] == forked.content

        await store.delete("e2e-fork-parent")
        await store.delete(forked.decision_trace["run_id"])
