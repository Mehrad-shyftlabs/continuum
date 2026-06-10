"""
Phase 6 — ScatterAgent ordered capture + Forkable.

No LLM/network: a fake runner records into the per-branch isolated recorders so
we can assert (a) concurrent branches land as deterministic, contiguous,
stage-indexed segments with one marker per branch plus the gather stage, and
(b) resume_from re-runs ONLY the forked branch, recovers siblings from the parent
trace, applies the override, and stamps lineage.
"""

from __future__ import annotations

from continuum.agent.base import BaseAgent
from continuum.agent.interfaces.forkable import Forkable
from continuum.agent.trace.recorder import TraceRecorder
from continuum.agent.trace.types import StepKind
from continuum.agent.types import AgentResponse, MergeStrategy, ResponseStatus
from continuum.agent.utils.context_utils import create_run_context
from continuum.agent.workflow._forkable import segment_by_markers
from continuum.agent.workflow.scatter import ScatterAgent, ScatterConfig


def _make_scatter() -> ScatterAgent:
    return ScatterAgent(
        name="market-scatter",
        agents=[
            BaseAgent(name="financials", instructions="x"),
            BaseAgent(name="competitors", instructions="x"),
            BaseAgent(name="news", instructions="x"),
        ],
        # explicit slices -> no LLM splitting; CONCATENATE -> no LLM merge.
        input_slices=["slice fin", "slice comp", "slice news"],
        scatter_config=ScatterConfig(merge_strategy=MergeStrategy.CONCATENATE),
    )


class FakeRunner:
    """Records one llm step per branch into whatever recorder the branch context
    carries, so absorb has real steps to merge."""

    def __init__(self) -> None:
        self.runs: list[str] = []

    def ensure_recorder(self, context, root_agent, query=""):
        context.recorder = TraceRecorder(context.run_id, root_agent, query, checkpoint=True)
        return True

    async def run(self, *, agent, input, context):
        self.runs.append(agent.name)
        if context.recorder is not None:
            context.recorder.record_llm_call(
                agent.name,
                1,
                output=f"{agent.name} out",
                agent_stack=[agent.name],
                messages_snapshot=[{"role": "user", "content": input}],
            )
        return AgentResponse(
            content=f"{agent.name} out", agent_name=agent.name, status=ResponseStatus.SUCCESS
        )

    async def save_turn(self, **kwargs):  # pragma: no cover - no session in tests
        pass

    async def persist_decision_trace(self, context, result):
        self.persisted = context.recorder.trace


def test_scatter_satisfies_forkable() -> None:
    assert isinstance(_make_scatter(), Forkable)


async def test_scatter_ordered_capture_one_marker_per_branch() -> None:
    scatter = _make_scatter()
    runner = FakeRunner()
    ctx = create_run_context(max_turns=5)

    result = await scatter.execute("Analyse market", runner, ctx, llm_client=None)

    assert result.status == ResponseStatus.SUCCESS
    assert sorted(runner.runs) == ["competitors", "financials", "news"]

    trace = runner.persisted
    step_stage, stage_first = segment_by_markers(trace)

    # 3 branch markers + 1 gather marker.
    markers = [s for s in trace.steps if s.kind == StepKind.WORKFLOW_STEP]
    assert len(markers) == 4
    marker_stages = [s.decision["stage"] for s in markers]
    # Deterministic branch order (0,1,2) then the gather stage (3).
    assert marker_stages == [0, 1, 2, 3]

    # Branch steps fall into contiguous stages 0..2, gather LLM step into stage 3.
    branch_stages = {step_stage[s.step_id] for s in trace.steps if s.kind != StepKind.WORKFLOW_STEP}
    assert branch_stages == {0, 1, 2, 3}
    assert set(stage_first) == {0, 1, 2, 3}

    # Each absorbed branch step has the orchestrator prepended to its stack.
    for s in trace.steps:
        if s.kind != StepKind.WORKFLOW_STEP and step_stage[s.step_id] < 3:
            assert s.agent_stack[0] == "market-scatter"

    # All ids unique after renumbering.
    ids = [s.step_id for s in trace.steps]
    assert len(ids) == len(set(ids))


def _parent_scatter_trace() -> tuple[object, list[str]]:
    """Hand-build a parent trace mirroring what execute() would record:
    3 branch segments (stages 0..2) + a gather stage (3)."""
    rec = TraceRecorder("scatter-parent", root_agent="market-scatter", checkpoint=True)
    branch_step_ids: list[str] = []
    for i, name in enumerate(["financials", "competitors", "news"]):
        rec.record_workflow_step(
            "market-scatter", stage=i, label=name, agent_stack=["market-scatter"]
        )
        sid = rec.record_llm_call(
            name,
            1,
            output=f"{name} cached",
            agent_stack=["market-scatter", name],
            messages_snapshot=[{"role": "user", "content": f"slice {name}"}],
        )
        branch_step_ids.append(sid)
    rec.record_workflow_step(
        "market-scatter", stage=3, label="gather", agent_stack=["market-scatter"]
    )
    rec.record(
        StepKind.LLM_CALL, "market-scatter", agent_stack=["market-scatter"], output="merged cached"
    )
    return rec.build_trace(final_response="merged cached"), branch_step_ids


async def test_scatter_resume_reruns_only_forked_branch() -> None:
    parent, branch_step_ids = _parent_scatter_trace()
    scatter = _make_scatter()
    runner = FakeRunner()
    ctx = create_run_context(max_turns=5)

    # Fork the competitors branch (stage 1).
    result = await scatter.resume_from(
        parent_trace=parent,
        from_step=branch_step_ids[1],
        override={"replace_last_user": "edited competitors slice"},
        runner=runner,
        context=ctx,
    )

    # Only the forked branch was actually re-run.
    assert runner.runs == ["competitors"]

    # Siblings recovered from the parent trace; forked branch replaced with fresh run.
    assert "financials cached" in result.content
    assert "news cached" in result.content
    assert "competitors out" in result.content  # fresh output, not "competitors cached"

    # Override flowed into the re-run input via the branch's recorded snapshot.
    forked_steps = [
        s for s in ctx.recorder.trace.steps if s.agent_name == "competitors" and s.messages_snapshot
    ]
    assert forked_steps
    assert forked_steps[-1].messages_snapshot[-1]["content"] == "edited competitors slice"

    # Lineage stamped back to the parent.
    assert ctx.recorder.trace.parent_run_id == "scatter-parent"
    assert ctx.recorder.trace.forked_from_step == branch_step_ids[1]
    assert ctx.recorder.trace.edit["stage"] == 1
    assert result.run_id == ctx.run_id


async def test_scatter_resume_unknown_step_raises() -> None:
    parent, _ = _parent_scatter_trace()
    scatter = _make_scatter()

    class _R:
        def ensure_recorder(self, *a, **k):
            return False

    import pytest

    with pytest.raises(ValueError, match="not found"):
        await scatter.resume_from(
            parent_trace=parent,
            from_step="nope",
            override=None,
            runner=_R(),
            context=create_run_context(max_turns=5),
        )
