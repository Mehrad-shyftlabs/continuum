"""
End-to-end test for WORKFLOW decision trace — real LLM, real Redis.

Phase 3 foundation: a workflow orchestrator (SequentialAgent) now owns one
decision trace spanning all its sub-agents — created rooted at the workflow,
persisted, and attached. Before this, workflow runs persisted no trace at all.
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


class TestWorkflowTraceE2E:
    @_skip_on_api_error
    async def test_sequential_workflow_persists_one_trace(self, trace_checkpoint_enabled):
        _skip_if_no_api_key()

        from continuum.agent.base import BaseAgent
        from continuum.agent.config import AgentConfig, AgentMemoryConfig
        from continuum.agent.runner import AgentRunner
        from continuum.agent.trace.config import get_trace_store
        from continuum.agent.types import RunContext
        from continuum.agent.workflow.sequential import SequentialAgent

        _mem = AgentMemoryConfig(search_memories=False, store_memories=False)
        _cfg = AgentConfig(log_to_session=False, session_history_turns=0)

        researcher = BaseAgent(
            name="researcher",
            instructions="Give two terse bullet facts about the topic.",
            memory_config=_mem,
            config=_cfg,
        )
        writer = BaseAgent(
            name="writer",
            instructions="Turn the facts into a one-sentence summary.",
            memory_config=_mem,
            config=_cfg,
        )
        pipeline = SequentialAgent(name="research-pipeline", agents=[researcher, writer])

        runner = AgentRunner()
        runner.register_agent(researcher)
        runner.register_agent(writer)

        ctx = RunContext(run_id="e2e-wf-seq")
        resp = await pipeline.execute("the moon", runner, ctx)
        assert resp.content

        # The workflow now persists ONE trace, rooted at the workflow, with both
        # sub-agents' steps in it.
        trace = await get_trace_store().get("e2e-wf-seq")
        assert trace is not None, "workflow run persisted no trace"
        assert trace.root_agent == "research-pipeline"
        agents = {s.agent_name for s in trace.steps}
        assert "researcher" in agents and "writer" in agents, f"missing sub-agent steps: {agents}"

        await get_trace_store().delete("e2e-wf-seq")

    @_skip_on_api_error
    async def test_sequential_fork_resumes_from_stage(self, trace_checkpoint_enabled):
        """Forking a Sequential run resumes from the forked step's STAGE — earlier
        stages are not re-run; the resumed stage onward re-executes."""
        _skip_if_no_api_key()

        from continuum.agent.base import BaseAgent
        from continuum.agent.config import AgentConfig, AgentMemoryConfig
        from continuum.agent.runner import AgentRunner
        from continuum.agent.trace.config import get_trace_store
        from continuum.agent.types import RunContext
        from continuum.agent.workflow.sequential import SequentialAgent

        _mem = AgentMemoryConfig(search_memories=False, store_memories=False)
        _cfg = AgentConfig(log_to_session=False, session_history_turns=0)

        researcher = BaseAgent(
            name="researcher",
            instructions="List two facts about the topic.",
            memory_config=_mem,
            config=_cfg,
        )
        writer = BaseAgent(
            name="writer",
            instructions="Write one sentence from the facts.",
            memory_config=_mem,
            config=_cfg,
        )
        editor = BaseAgent(
            name="editor",
            instructions="Tighten the sentence to <12 words.",
            memory_config=_mem,
            config=_cfg,
        )
        pipeline = SequentialAgent(name="rw-pipeline", agents=[researcher, writer, editor])
        runner = AgentRunner()

        ctx = RunContext(run_id="e2e-wf-seq-fork")
        await pipeline.execute("volcanoes", runner, ctx)
        store = get_trace_store()
        parent = await store.get("e2e-wf-seq-fork")
        assert parent is not None

        # Fork at the writer stage (stage 1), replacing its input.
        writer_step = next(s for s in parent.steps if s.agent_name == "writer")
        forked = await runner.fork(
            "e2e-wf-seq-fork",
            writer_step.step_id,
            agent=pipeline,
            override={"replace_last_user": "Facts: the sky is green; rivers flow uphill."},
            label="resume at writer",
        )
        child = await store.get(forked.run_id)
        assert child is not None
        assert child.parent_run_id == "e2e-wf-seq-fork"
        child_agents = {s.agent_name for s in child.steps}
        # Resumed from the writer stage: researcher (stage 0) did NOT re-run.
        assert "researcher" not in child_agents, f"earlier stage re-ran: {child_agents}"
        assert "writer" in child_agents and "editor" in child_agents, child_agents

        await store.delete("e2e-wf-seq-fork")
        await store.delete(forked.run_id)

    @_skip_on_api_error
    async def test_router_capture_and_reroute_fork(self, trace_checkpoint_enabled):
        """A Router run records its routing decision + the chosen specialist's
        steps; forking with override={'route': X} re-routes to a different one."""
        _skip_if_no_api_key()

        from continuum.agent.base import BaseAgent
        from continuum.agent.config import AgentConfig, AgentMemoryConfig, RouterConfig
        from continuum.agent.runner import AgentRunner
        from continuum.agent.trace.config import get_trace_store
        from continuum.agent.types import Route, RunContext
        from continuum.agent.workflow.router import RouterAgent

        _mem = AgentMemoryConfig(search_memories=False, store_memories=False)
        _cfg = AgentConfig(log_to_session=False, session_history_turns=0)

        billing = BaseAgent(
            name="billing-agent",
            instructions="Answer the billing question in one line.",
            memory_config=_mem,
            config=_cfg,
        )
        technical = BaseAgent(
            name="technical-agent",
            instructions="Answer the technical question in one line.",
            memory_config=_mem,
            config=_cfg,
        )
        router = RouterAgent(
            name="triage",
            routes=[
                Route(
                    agent_name="billing-agent", description="billing refunds", condition="refund"
                ),
                Route(agent_name="technical-agent", description="bugs errors", condition="bug"),
            ],
            router_config=RouterConfig(routing_strategy="rule_based"),
        )
        runner = AgentRunner()
        runner.register_agent(billing)
        runner.register_agent(technical)

        ctx = RunContext(run_id="e2e-wf-router")
        await router.execute("I need a refund please", runner, ctx)
        store = get_trace_store()
        parent = await store.get("e2e-wf-router")
        assert parent is not None
        assert parent.root_agent == "triage"
        kinds = {s.kind.value for s in parent.steps}
        assert "routing" in kinds, f"no routing step recorded: {kinds}"
        assert "billing-agent" in {s.agent_name for s in parent.steps}

        route_step = next(s for s in parent.steps if s.kind.value == "routing")
        forked = await runner.fork(
            "e2e-wf-router",
            route_step.step_id,
            agent=router,
            override={"route": "technical-agent"},
            label="re-route to technical",
        )
        child = await store.get(forked.run_id)
        assert child is not None
        assert child.parent_run_id == "e2e-wf-router"
        child_agents = {s.agent_name for s in child.steps}
        assert "technical-agent" in child_agents, f"re-route failed: {child_agents}"
        assert "billing-agent" not in child_agents, f"old route re-ran: {child_agents}"

        await store.delete("e2e-wf-router")
        await store.delete(forked.run_id)
