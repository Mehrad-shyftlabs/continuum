"""
Phase 6 — ParallelAgent ordered capture + Forkable.

No LLM/network: a fake runner records into each branch's per-branch recorder and
returns canned content; the resume test hand-builds a parent trace with two branch
segments and stubs the re-run branch. We assert the merged trace is deterministic,
contiguous and stage-indexed, and that a fork re-runs ONLY the forked branch,
recovers the sibling's cached output, applies the override, and stamps lineage.
"""

from __future__ import annotations

from continuum.agent.base import BaseAgent
from continuum.agent.config import ParallelConfig
from continuum.agent.interfaces.forkable import Forkable
from continuum.agent.trace.recorder import TraceRecorder
from continuum.agent.types import (
    AgentResponse,
    MergeStrategy,
    ResponseStatus,
)
from continuum.agent.utils.context_utils import create_run_context
from continuum.agent.workflow._forkable import segment_by_markers
from continuum.agent.workflow.parallel import ParallelAgent


def _make_parallel(strategy: MergeStrategy = MergeStrategy.CONCATENATE) -> ParallelAgent:
    return ParallelAgent(
        name="parallel-search",
        agents=[
            BaseAgent(name="web", instructions="Search the web."),
            BaseAgent(name="db", instructions="Search the database."),
        ],
        parallel_config=ParallelConfig(merge_strategy=strategy),
    )


class _FakeRunner:
    """A runner whose ``run`` records into the per-branch recorder on the context
    and returns canned content keyed by agent name."""

    def __init__(self, outputs: dict[str, str]) -> None:
        self.outputs = outputs
        self.persisted: dict[str, object] = {}

    def ensure_recorder(self, context, root_agent, query: str = "") -> bool:
        if context.recorder is not None:
            return False
        context.recorder = TraceRecorder(context.run_id, root_agent, query, checkpoint=True)
        return True

    async def persist_decision_trace(self, context, result) -> None:
        self.persisted["trace"] = context.recorder.trace

    async def save_turn(self, **kwargs) -> None:  # pragma: no cover - not exercised
        pass

    async def run(self, *, agent, input, context):
        content = self.outputs[agent.name]
        # Record into the branch's own recorder so absorb has steps to merge.
        if context.recorder is not None:
            parent = context.recorder.record_llm_call(
                agent.name, 1, output=f"{agent.name} llm", agent_stack=[agent.name]
            )
            context.recorder.record_tool_call(
                agent.name,
                1,
                "t",
                {"q": input},
                content,
                parent_id=parent,
                agent_stack=[agent.name],
            )
        return AgentResponse(content=content, agent_name=agent.name, status=ResponseStatus.SUCCESS)


def test_parallel_satisfies_forkable() -> None:
    assert isinstance(_make_parallel(), Forkable)


async def test_parallel_ordered_capture_is_deterministic() -> None:
    parallel = _make_parallel()
    runner = _FakeRunner({"web": "web output", "db": "db output"})
    ctx = create_run_context(max_turns=5)

    result = await parallel.execute("find X", runner, ctx)

    assert result.status == ResponseStatus.SUCCESS
    assert result.run_id == ctx.run_id

    trace = runner.persisted["trace"]
    steps = trace.steps

    # One WORKFLOW_STEP marker per branch (web=stage 0, db=stage 1) plus the
    # merge/synthesis stage (stage 2), in order.
    from continuum.agent.trace.types import StepKind

    markers = [s for s in steps if s.kind == StepKind.WORKFLOW_STEP]
    assert len(markers) == 3
    assert [m.decision["stage"] for m in markers] == [0, 1, 2]
    assert [m.decision["label"] for m in markers] == ["web", "db", "merge"]

    # Step ids are unique after renumbering.
    ids = [s.step_id for s in steps]
    assert len(ids) == len(set(ids))

    # Branch steps are contiguous and stage-indexed; web before db, then merge.
    step_stage, stage_first = segment_by_markers(trace)
    branch_stages = [step_stage[s.step_id] for s in steps if s.kind != StepKind.WORKFLOW_STEP]
    assert branch_stages == [0, 0, 1, 1, 2]  # two branches (2 steps each) + merge step
    assert set(stage_first) == {0, 1, 2}

    # Orchestrator prepended to every absorbed branch step's stack.
    for s in steps:
        if s.kind != StepKind.WORKFLOW_STEP:
            assert s.agent_stack[0] == "parallel-search"


def _parent_parallel_trace() -> tuple[object, list[str]]:
    """Hand-build a parent trace with two branch segments (web=0, db=1)."""
    rec = TraceRecorder("par-parent", root_agent="parallel-search", checkpoint=True)
    step_ids: list[str] = []
    for stage, (agent, out) in enumerate([("web", "web cached"), ("db", "db cached")]):
        rec.record_workflow_step(
            "parallel-search", stage=stage, label=agent, agent_stack=["parallel-search"]
        )
        sid = rec.record_llm_call(
            agent,
            1,
            output=out,
            agent_stack=["parallel-search", agent],
            messages_snapshot=[{"role": "user", "content": f"{agent} input"}],
        )
        step_ids.append(sid)
    return rec.build_trace(final_response="merged"), step_ids


async def test_parallel_resume_reruns_only_forked_branch() -> None:
    parent, step_ids = _parent_parallel_trace()
    parallel = _make_parallel()
    runner = _FakeRunner({"web": "RERUN web", "db": "db SHOULD NOT RUN"})
    ctx = create_run_context(max_turns=5)

    # Fork branch 0 (web) with an override.
    result = await parallel.resume_from(
        parent_trace=parent,
        from_step=step_ids[0],
        override={"replace_last_user": "edited web input"},
        runner=runner,
        context=ctx,
    )

    # Only the web branch re-ran; db's output came from the parent trace's cache.
    assert "RERUN web" in result.content
    assert "db cached" in result.content
    assert "db SHOULD NOT RUN" not in result.content
    assert result.run_id == ctx.run_id

    # Lineage stamped back to the parent on the forked branch (stage 0).
    assert ctx.recorder.trace.parent_run_id == "par-parent"
    assert ctx.recorder.trace.forked_from_step == step_ids[0]
    assert ctx.recorder.trace.edit["stage"] == 0
    assert "trace" in runner.persisted

    # The re-run branch's input was the override-applied recovered input.
    from continuum.agent.trace.types import StepKind

    tool_steps = [s for s in ctx.recorder.trace.steps if s.kind == StepKind.TOOL_CALL]
    assert any(s.input.get("args", {}).get("q") == "edited web input" for s in tool_steps)
