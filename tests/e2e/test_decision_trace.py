"""
End-to-end test for decision trace — real LLM, real Redis, full runner.

Enables DECISION_TRACE_ENABLED, runs a real agent through AgentRunner, and asserts
the trace is both attached to the response (req. 4) and persisted to real Redis by
run_id (req. 2/5). Nothing is faked here.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.e2e

from tests.e2e.conftest import skip_if_no_api_key as _skip_if_no_api_key
from tests.e2e.conftest import skip_on_api_error as _skip_on_api_error


@pytest.fixture
def trace_enabled(monkeypatch):
    """Turn the feature on against the real Redis backend for this test."""
    from continuum.config import settings

    monkeypatch.setattr(settings, "decision_trace_enabled", True)
    monkeypatch.setattr(settings, "decision_trace_detail", "full")
    monkeypatch.setattr(settings, "decision_trace_store", "redis")

    from continuum.agent.trace import config as trace_config

    trace_config.get_trace_store.cache_clear()
    yield
    trace_config.get_trace_store.cache_clear()


class TestDecisionTraceE2E:
    @_skip_on_api_error
    async def test_trace_attached_and_persisted(self, trace_enabled):
        """A real run attaches a decision trace and persists it to real Redis."""
        _skip_if_no_api_key()

        from continuum.agent.base import BaseAgent
        from continuum.agent.config import AgentConfig, AgentMemoryConfig
        from continuum.agent.runner import AgentRunner
        from continuum.agent.trace.config import get_trace_store
        from continuum.agent.types import RunContext

        agent = BaseAgent(
            name="trace-e2e-agent",
            instructions="You are a helpful assistant. Be extremely concise.",
            memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
            config=AgentConfig(log_to_session=False),
        )
        runner = AgentRunner()
        context = RunContext(run_id="e2e-trace-run")

        response = await runner.run(
            agent, "What is 2+2? Reply with just the number.", context=context
        )

        # (4) attached to the response metadata
        assert response.decision_trace is not None
        trace = response.decision_trace
        assert trace["run_id"] == "e2e-trace-run"
        assert trace["final_response"] == response.content
        # at least one real LLM decision was captured
        kinds = [s["kind"] for s in trace["steps"]]
        assert "llm_call" in kinds
        assert trace["metrics"]["total_tokens"] > 0  # real token usage

        # (2/5) persisted to real Redis and readable back by run_id
        store = get_trace_store()
        persisted = await store.get("e2e-trace-run")
        assert persisted is not None
        assert persisted.final_response == response.content
        assert persisted.metrics()["total_tokens"] == trace["metrics"]["total_tokens"]

        await store.delete("e2e-trace-run")  # cleanup
