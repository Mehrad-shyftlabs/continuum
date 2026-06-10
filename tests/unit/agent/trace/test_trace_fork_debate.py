"""
Phase 6 — DebateAgent is Forkable and does ordered capture.

No LLM/network. A fake runner records a single LLM step into whatever recorder
lives on the (branch) context it is handed, so we can assert:

* ordered capture — the debaters (pro=stage 0, con=stage 1) and the judge
  (stage 2) are absorbed in deterministic stage order, with contiguous,
  stage-indexed, segmentable markers, regardless of concurrent interleaving;
* Forkable — ``resume_from`` re-runs from the right stage with the override
  applied to the recovered input, lineage stamped, and ValueError on unknown
  steps.

Stage scheme under test: ``stage = round_index * num_debaters + debater_index``
with one round of two debaters → pro=0, con=1; judge = num_rounds*num_debaters = 2.
"""

from __future__ import annotations

import pytest

from continuum.agent.base import BaseAgent
from continuum.agent.interfaces.forkable import Forkable
from continuum.agent.trace.recorder import TraceRecorder
from continuum.agent.trace.types import StepKind
from continuum.agent.types import AgentResponse, ResponseStatus, TokenUsage
from continuum.agent.utils.context_utils import create_run_context
from continuum.agent.workflow._forkable import segment_by_markers
from continuum.agent.workflow.debate import DebateAgent


def _make_debate() -> DebateAgent:
    return DebateAgent(
        name="arch-debate",
        pro_agent=BaseAgent(name="arch-debate-pro", instructions="Argue FOR."),
        con_agent=BaseAgent(name="arch-debate-con", instructions="Argue AGAINST."),
        judge_agent=BaseAgent(name="arch-debate-judge", instructions="Synthesise."),
    )


class FakeRunner:
    """Runner stub: records one LLM step into the given context's recorder and
    returns a deterministic response. ``ensure_recorder`` installs a checkpoint
    recorder on the shared context (so the orchestrator owns the trace)."""

    def __init__(self) -> None:
        self.persisted: dict = {}

    def ensure_recorder(self, context, root_agent, query="") -> bool:
        if context.recorder is not None:
            return False
        context.recorder = TraceRecorder(context.run_id, root_agent, query, checkpoint=True)
        return True

    async def persist_decision_trace(self, context, result) -> None:
        self.persisted["trace"] = context.recorder.trace

    async def save_turn(self, **kwargs) -> None:
        pass

    async def run(self, *, agent, input, context):
        if context.recorder is not None:
            context.recorder.record_llm_call(
                agent.name,
                1,
                output=f"{agent.name} says: {input[:40]}",
                agent_stack=[agent.name],
                messages_snapshot=[{"role": "user", "content": input}],
            )
        return AgentResponse(
            content=f"{agent.name} output",
            agent_name=agent.name,
            status=ResponseStatus.SUCCESS,
            usage=TokenUsage(),
        )


def test_debate_satisfies_forkable() -> None:
    assert isinstance(_make_debate(), Forkable)


async def test_debate_ordered_capture_is_stage_indexed() -> None:
    debate = _make_debate()
    runner = FakeRunner()
    ctx = create_run_context(max_turns=5)

    result = await debate.execute("microservices or monolith?", runner, ctx)
    assert result.run_id == ctx.run_id
    trace = runner.persisted["trace"]

    # Three absorbed segments → three WORKFLOW_STEP markers, stages 0, 1, 2.
    markers = [s for s in trace.steps if s.kind == StepKind.WORKFLOW_STEP]
    assert [m.decision["stage"] for m in markers] == [0, 1, 2]
    # In documented order: pro (round 0), con (round 0), then judge.
    assert markers[0].decision["label"].startswith("arch-debate-pro")
    assert markers[1].decision["label"].startswith("arch-debate-con")
    assert markers[2].decision["label"] == "arch-debate-judge"

    # Orchestrator is prepended to every absorbed branch step's stack.
    for s in trace.steps:
        if s.kind != StepKind.WORKFLOW_STEP:
            assert s.agent_stack[0] == "arch-debate"

    # All step ids unique after renumbering.
    ids = [s.step_id for s in trace.steps]
    assert len(ids) == len(set(ids))

    # Segmentation recovers exactly the three contiguous stages.
    step_stage, stage_first = segment_by_markers(trace)
    stages = {step_stage[s.step_id] for s in trace.steps if s.kind != StepKind.WORKFLOW_STEP}
    assert stages == {0, 1, 2}
    assert set(stage_first) == {0, 1, 2}


def _parent_debate_trace() -> tuple[object, dict[int, str]]:
    """Hand-built parent trace: pro=stage 0, con=stage 1, judge=stage 2."""
    rec = TraceRecorder("debate-parent", root_agent="arch-debate", checkpoint=True)
    step_by_stage: dict[int, str] = {}
    specs = [
        (0, "arch-debate-pro", "round-input pro"),
        (1, "arch-debate-con", "round-input con"),
        (2, "arch-debate-judge", "assembled judge prompt"),
    ]
    for stage, agent, user_msg in specs:
        rec.record_workflow_step(
            "arch-debate", stage=stage, label=agent, agent_stack=["arch-debate"]
        )
        sid = rec.record_llm_call(
            agent,
            1,
            output=f"{agent} output",
            agent_stack=["arch-debate", agent],
            messages_snapshot=[{"role": "user", "content": user_msg}],
        )
        step_by_stage[stage] = sid
    return rec.build_trace(final_response="arch-debate-judge output"), step_by_stage


async def test_debate_resume_from_debater_reruns_round(monkeypatch) -> None:
    parent, step_by_stage = _parent_debate_trace()
    debate = _make_debate()
    runner = FakeRunner()

    captured: dict = {}

    async def fake_run_debaters(self, input_text, runner, context):
        captured["debaters_input"] = input_text
        return "pro out", "con out", TokenUsage()

    async def fake_run_judge(self, input_text, pro, con, runner, context):
        captured["judge_topic"] = input_text
        captured["judge_pro"] = pro
        captured["judge_con"] = con
        return "synthesis", TokenUsage()

    monkeypatch.setattr(DebateAgent, "_run_debaters", fake_run_debaters)
    monkeypatch.setattr(DebateAgent, "_run_judge", fake_run_judge)

    ctx = create_run_context(max_turns=5)
    resp = await debate.resume_from(
        parent_trace=parent,
        from_step=step_by_stage[0],  # a step inside the pro debater (stage 0)
        override={"replace_last_user": "edited round input"},
        runner=runner,
        context=ctx,
    )

    # Round-level resume: both debaters re-run with the overridden round input,
    # then the judge runs on that same topic.
    assert captured["debaters_input"] == "edited round input"
    assert captured["judge_topic"] == "edited round input"
    assert resp.run_id == ctx.run_id

    # Lineage stamped back to the parent at stage 0.
    assert ctx.recorder.trace.parent_run_id == "debate-parent"
    assert ctx.recorder.trace.forked_from_step == step_by_stage[0]
    assert ctx.recorder.trace.edit["stage"] == 0
    assert "trace" in runner.persisted


async def test_debate_resume_from_judge_reuses_arguments(monkeypatch) -> None:
    parent, step_by_stage = _parent_debate_trace()
    debate = _make_debate()
    runner = FakeRunner()

    captured: dict = {}

    async def fake_run_debaters(self, input_text, runner, context):
        captured["debaters_ran"] = True
        return "should not run", "should not run", TokenUsage()

    async def fake_invoke_judge(self, judge_input, pro, con, runner, context):
        captured["judge_input"] = judge_input
        captured["pro"] = pro
        captured["con"] = con
        return "synthesis", TokenUsage()

    monkeypatch.setattr(DebateAgent, "_run_debaters", fake_run_debaters)
    monkeypatch.setattr(DebateAgent, "_invoke_judge", fake_invoke_judge)

    ctx = create_run_context(max_turns=5)
    resp = await debate.resume_from(
        parent_trace=parent,
        from_step=step_by_stage[2],  # the judge stage
        override={"replace_last_user": "edited judge prompt"},
        runner=runner,
        context=ctx,
    )

    # Judge-level resume: debaters NOT re-run; arguments recovered from the
    # parent trace; override applied to the judge's recovered input.
    assert "debaters_ran" not in captured
    assert captured["judge_input"] == "edited judge prompt"
    assert captured["pro"] == "arch-debate-pro output"
    assert captured["con"] == "arch-debate-con output"
    assert resp.run_id == ctx.run_id
    assert ctx.recorder.trace.edit["stage"] == 2


async def test_debate_resume_from_unknown_step_raises() -> None:
    parent, _ = _parent_debate_trace()
    debate = _make_debate()

    class NoopRunner:
        def ensure_recorder(self, *a, **k) -> bool:
            return False

    with pytest.raises(ValueError, match="not found"):
        await debate.resume_from(
            parent_trace=parent,
            from_step="does-not-exist",
            override=None,
            runner=NoopRunner(),
            context=create_run_context(max_turns=5),
        )
