"""
End-to-end test for MULTI-AGENT decision trace — real LLM, real Redis.

Covers the two capabilities added for handoff runs:
  * Capture: a handoff run produces ONE trace spanning every agent (the handed-off
    agent's own steps are recorded, not just the HANDOFF marker).
  * Fork: forking a step that belongs to a handed-off (non-root) agent resumes
    THAT agent — not the root — with the handoff stack restored.

Terminal handoffs (return_to_parent=False) only, matching current fork support.
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


def _build_runner():
    from continuum.agent.base import BaseAgent
    from continuum.agent.config import AgentConfig, AgentMemoryConfig
    from continuum.agent.runner import AgentRunner
    from continuum.agent.types import Handoff

    _mem = AgentMemoryConfig(search_memories=False, store_memories=False)
    _cfg = AgentConfig(log_to_session=False, session_history_turns=0)

    specialist = BaseAgent(
        name="specialist",
        instructions="You are a haiku specialist. Reply with a short haiku about the topic.",
        memory_config=_mem,
        config=_cfg,
    )
    triage = BaseAgent(
        name="triage",
        instructions=(
            "You route requests. Immediately hand off to 'specialist' for every "
            "request. Do not answer yourself."
        ),
        handoffs=[
            Handoff(
                target_agent="specialist",
                description="Hand off all requests to the specialist.",
                return_to_parent=False,
            )
        ],
        memory_config=_mem,
        config=_cfg,
    )
    runner = AgentRunner()
    runner.register_agent(triage)
    runner.register_agent(specialist)
    return runner, triage


class TestMultiAgentTraceE2E:
    @_skip_on_api_error
    async def test_handoff_capture_and_non_root_fork(self, trace_checkpoint_enabled):
        _skip_if_no_api_key()

        from continuum.agent.trace.config import get_trace_store
        from continuum.agent.types import RunContext

        runner, triage = _build_runner()
        store = get_trace_store()

        # ---- run the handoff chain ----
        ctx = RunContext(run_id="e2e-multi-parent")
        await runner.run(triage, "Write about the ocean.", context=ctx)

        parent = await store.get("e2e-multi-parent")
        assert parent is not None
        agents = {s.agent_name for s in parent.steps}
        # Capture: BOTH agents' steps are in the one trace (not just triage).
        assert "triage" in agents
        assert "specialist" in agents, f"handed-off agent not captured; got {agents}"

        # The specialist's steps carry the restored handoff stack.
        spec_steps = [s for s in parent.steps if s.agent_name == "specialist"]
        assert spec_steps, "no specialist steps captured"
        assert spec_steps[0].agent_stack[-1] == "specialist"
        assert "triage" in spec_steps[0].agent_stack

        # ---- fork a NON-ROOT step (specialist) ----
        spec_step_id = spec_steps[0].step_id
        forked = await runner.fork(
            "e2e-multi-parent",
            spec_step_id,
            override={"system": "Reply with exactly one word: OVERRIDDEN."},
            label="non-root fork",
        )
        child = await store.get(forked.run_id)
        assert child is not None
        # Fork resumed the specialist only — triage did NOT re-run.
        child_agents = {s.agent_name for s in child.steps}
        assert child_agents == {"specialist"}, (
            f"expected specialist-only resume, got {child_agents}"
        )
        assert child.parent_run_id == "e2e-multi-parent"
        assert child.forked_from_step == spec_step_id

        await store.delete("e2e-multi-parent")
        await store.delete(forked.run_id)
