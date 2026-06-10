"""
End-to-end fork tests for Phase 5/6 orchestrators — real LLM, real Redis.

Validates the whole path (record → persist → fork → resume) against a live model
for one LINEAR orchestrator (LoopAgent, Phase 5) and one CONCURRENT orchestrator
(ParallelAgent, Phase 6 ordered capture), beyond the stubbed unit tests.
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


class TestPhase56ForkE2E:
    @_skip_on_api_error
    async def test_loop_fork_resumes_from_iteration(self, trace_checkpoint_enabled):
        """A LoopAgent records one stage per iteration; forking an iteration
        resumes from that iteration (earlier iterations do not re-run) and the
        new run links back to the parent."""
        _skip_if_no_api_key()

        from continuum.agent.base import BaseAgent
        from continuum.agent.config import AgentConfig, AgentMemoryConfig
        from continuum.agent.runner import AgentRunner
        from continuum.agent.trace.config import get_trace_store
        from continuum.agent.trace.types import StepKind
        from continuum.agent.types import (
            RunContext,
            TerminationConfig,
            TerminationType,
        )
        from continuum.agent.workflow.loop import LoopAgent

        _mem = AgentMemoryConfig(search_memories=False, store_memories=False)
        _cfg = AgentConfig(log_to_session=False, session_history_turns=0)

        refiner = BaseAgent(
            name="refiner",
            instructions="Rewrite the given line to be slightly shorter. Output only the line.",
            memory_config=_mem,
            config=_cfg,
        )
        # Runs exactly 2 iterations (pattern never matches → max_iterations).
        loop = LoopAgent(
            name="refiner-loop",
            agent=refiner,
            termination=TerminationConfig(
                type=TerminationType.OUTPUT_MATCH, pattern="ZZZ_NEVER", max_iterations=2
            ),
        )
        runner = AgentRunner()
        runner.register_agent(loop)

        ctx = RunContext(run_id="e2e-loop-fork")
        await loop.execute("The quick brown fox jumps over the lazy dog.", runner, ctx)

        store = get_trace_store()
        parent = await store.get("e2e-loop-fork")
        assert parent is not None, "loop run persisted no trace"
        assert parent.root_agent == "refiner-loop"
        markers = [s for s in parent.steps if s.kind == StepKind.WORKFLOW_STEP]
        # Two iterations → two stage markers (0, 1).
        assert [m.decision["stage"] for m in markers] == [0, 1], markers

        # Fork at iteration 2 (stage 1): resume from there with an edited input.
        stage1_step = next(
            s
            for s in parent.steps
            if s.kind != StepKind.WORKFLOW_STEP
            and s.messages_snapshot
            and parent.steps.index(s) > parent.steps.index(markers[1])
        )
        forked = await runner.fork(
            "e2e-loop-fork",
            stage1_step.step_id,
            agent=loop,
            override={"replace_last_user": "A very very very long sentence to compress now."},
            label="resume at iteration 2",
        )
        child = await store.get(forked.run_id)
        assert child is not None
        assert child.parent_run_id == "e2e-loop-fork"
        assert child.forked_from_step == stage1_step.step_id
        # Resumed from stage 1: the child's first marker is stage 1, not 0.
        child_markers = [s for s in child.steps if s.kind == StepKind.WORKFLOW_STEP]
        assert child_markers and child_markers[0].decision["stage"] == 1, child_markers
        assert child.final_response

        await store.delete("e2e-loop-fork")
        await store.delete(forked.run_id)

    @_skip_on_api_error
    async def test_parallel_ordered_capture_and_branch_fork(self, trace_checkpoint_enabled):
        """A ParallelAgent records each branch as a deterministic stage (ordered
        capture) plus a merge stage; forking one branch re-runs only that branch
        and re-merges with the other branch's cached output."""
        _skip_if_no_api_key()

        from continuum.agent.base import BaseAgent
        from continuum.agent.config import AgentConfig, AgentMemoryConfig, ParallelConfig
        from continuum.agent.runner import AgentRunner
        from continuum.agent.trace.config import get_trace_store
        from continuum.agent.trace.types import StepKind
        from continuum.agent.types import MergeStrategy, RunContext
        from continuum.agent.workflow.parallel import ParallelAgent

        _mem = AgentMemoryConfig(search_memories=False, store_memories=False)
        _cfg = AgentConfig(log_to_session=False, session_history_turns=0)

        haiku = BaseAgent(
            name="haiku-writer",
            instructions="Write a one-line haiku about the topic.",
            memory_config=_mem,
            config=_cfg,
        )
        fact = BaseAgent(
            name="fact-writer",
            instructions="State one terse fact about the topic.",
            memory_config=_mem,
            config=_cfg,
        )
        parallel = ParallelAgent(
            name="parallel-writers",
            agents=[haiku, fact],
            parallel_config=ParallelConfig(merge_strategy=MergeStrategy.CONCATENATE),
        )
        runner = AgentRunner()
        runner.register_agent(parallel)

        ctx = RunContext(run_id="e2e-parallel-fork")
        await parallel.execute("the ocean", runner, ctx)

        store = get_trace_store()
        parent = await store.get("e2e-parallel-fork")
        assert parent is not None, "parallel run persisted no trace"
        assert parent.root_agent == "parallel-writers"
        markers = [s for s in parent.steps if s.kind == StepKind.WORKFLOW_STEP]
        # Ordered capture: branch 0, branch 1, then the merge stage (2).
        assert [m.decision["stage"] for m in markers] == [0, 1, 2], markers
        assert markers[2].decision["label"] == "merge"
        parent_agents = {s.agent_name for s in parent.steps}
        assert "haiku-writer" in parent_agents and "fact-writer" in parent_agents

        # Fork branch 0 (haiku-writer): re-run only that branch.
        branch0_step = next(
            s for s in parent.steps if s.agent_name == "haiku-writer" and s.messages_snapshot
        )
        forked = await runner.fork(
            "e2e-parallel-fork",
            branch0_step.step_id,
            agent=parallel,
            override={"replace_last_user": "the desert"},
            label="re-run haiku branch on the desert",
        )
        child = await store.get(forked.run_id)
        assert child is not None
        assert child.parent_run_id == "e2e-parallel-fork"
        child_agents = {s.agent_name for s in child.steps}
        # Only the forked branch re-ran; the sibling branch did NOT.
        assert "haiku-writer" in child_agents, child_agents
        assert "fact-writer" not in child_agents, f"sibling branch re-ran: {child_agents}"
        # The re-merge still produced a final answer (forked branch + cached sibling).
        assert child.final_response

        await store.delete("e2e-parallel-fork")
        await store.delete(forked.run_id)
