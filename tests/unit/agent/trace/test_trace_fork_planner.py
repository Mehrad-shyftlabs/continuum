"""
Phase 5 — PlannerAgent is Forkable: resume_from re-runs from the right stage.

Stage model: the planning call is stage 0; each executed plan step is stage
1, 2, 3…. No LLM/network here — a recorded parent trace is hand-built (its
stage-0 marker embeds the plan, mirroring what ``execute`` records), ``_drive``
is stubbed to capture where the resume begins, and we assert:

* a mid-plan step resumes at the right stage with the parent's plan re-used,
* the override is applied to the recovered input,
* the lineage stamp links back to the parent,
* forking the plan (stage 0) re-plans via ``execute``,
* an unknown step raises ValueError.
"""

from __future__ import annotations

import pytest

from continuum.agent.base import BaseAgent
from continuum.agent.interfaces.forkable import Forkable
from continuum.agent.trace.recorder import TraceRecorder
from continuum.agent.trace.types import StepKind
from continuum.agent.types import AgentResponse, ResponseStatus
from continuum.agent.utils.context_utils import create_run_context
from continuum.agent.workflow.planner import PlannerAgent


def _make_planner() -> PlannerAgent:
    return PlannerAgent(
        name="research-planner",
        agent=BaseAgent(name="worker", instructions="Do the step."),
    )


def _plan(n: int) -> list[dict[str, str]]:
    return [{"step_id": str(k + 1), "instruction": f"step {k + 1}"} for k in range(n)]


def _parent_trace_with_plan(n: int) -> tuple[object, list[str], list[dict[str, str]]]:
    """Build a parent trace: stage-0 plan marker + one LLM call per executed step.

    Returns the trace, the LLM-call step ids (one per stage 1..n), and the plan.
    """
    plan = _plan(n)
    rec = TraceRecorder("planner-parent", root_agent="research-planner", checkpoint=True)
    # Stage 0: the planning call, with the plan embedded (as execute() records).
    rec.record(
        StepKind.WORKFLOW_STEP,
        "research-planner",
        agent_stack=["research-planner"],
        decision={"stage": 0, "label": "plan", "plan": plan},
    )
    step_ids: list[str] = []
    for k in range(n):
        rec.record_workflow_step(
            "research-planner", stage=k + 1, label="worker", agent_stack=["research-planner"]
        )
        sid = rec.record_llm_call(
            "worker",
            1,
            output=f"out {k + 1}",
            agent_stack=["research-planner", "worker"],
            messages_snapshot=[{"role": "user", "content": f"input {k + 1}"}],
        )
        step_ids.append(sid)
    return rec.build_trace(final_response=f"out {n}"), step_ids, plan


class _FakeRunner:
    def __init__(self) -> None:
        self.persisted: dict = {}

    def ensure_recorder(self, context, root_agent, query=""):
        # Mirror the real runner: no-op when a recorder already exists.
        if context.recorder is not None:
            return False
        context.recorder = TraceRecorder(context.run_id, root_agent, query, checkpoint=True)
        return True

    async def persist_decision_trace(self, context, result):
        self.persisted["trace"] = context.recorder.trace


def test_planner_satisfies_forkable() -> None:
    assert isinstance(_make_planner(), Forkable)


async def test_planner_resume_from_mid_plan_step(monkeypatch) -> None:
    parent, step_ids, plan = _parent_trace_with_plan(3)
    planner = _make_planner()

    captured: dict = {}

    async def fake_drive(
        self,
        plan_steps,
        current_input,
        runner,
        context,
        *,
        workflow_span,
        start_stage,
        goal,
        llm_client,
        initial_usage=None,
    ):  # noqa: E501
        captured["plan_steps"] = plan_steps
        captured["current_input"] = current_input
        captured["start_stage"] = start_stage
        captured["goal"] = goal
        r = AgentResponse(content="ok", agent_name=self.name, status=ResponseStatus.SUCCESS)
        r.run_id = context.run_id
        return r

    monkeypatch.setattr(PlannerAgent, "_drive", fake_drive)

    runner = _FakeRunner()
    ctx = create_run_context(max_turns=5)
    resp = await planner.resume_from(
        parent_trace=parent,
        from_step=step_ids[1],  # a step inside stage 2 (the 2nd executed step)
        override={"replace_last_user": "edited input"},
        runner=runner,
        context=ctx,
    )

    # Resumes at the stage that owned the forked step (1-based: step idx 1 → stage 2).
    assert captured["start_stage"] == 2
    # The parent's plan is RE-USED, not re-made.
    assert captured["plan_steps"] == plan
    # Override applied to the recovered input.
    assert captured["current_input"] == "edited input"
    assert captured["goal"] == parent.user_query
    assert resp.run_id == ctx.run_id

    # Lineage stamped back to the parent.
    assert ctx.recorder.trace.parent_run_id == "planner-parent"
    assert ctx.recorder.trace.forked_from_step == step_ids[1]
    assert ctx.recorder.trace.edit["stage"] == 2
    # Forked trace stays self-describing: it re-records the stage-0 plan marker.
    assert PlannerAgent._recover_plan(ctx.recorder.trace) == plan
    assert "trace" in runner.persisted


async def test_planner_resume_from_plan_replans(monkeypatch) -> None:
    """Forking stage 0 (the plan) re-plans by delegating to execute()."""
    parent, _step_ids, _plan = _parent_trace_with_plan(2)
    planner = _make_planner()

    captured: dict = {}

    async def fake_execute(self, input_text, runner, context):
        captured["input_text"] = input_text
        r = AgentResponse(content="replanned", agent_name=self.name, status=ResponseStatus.SUCCESS)
        r.run_id = context.run_id
        return r

    monkeypatch.setattr(PlannerAgent, "execute", fake_execute)

    runner = _FakeRunner()
    ctx = create_run_context(max_turns=5)

    # The stage-0 marker has no messages_snapshot, so the recovered goal falls
    # back to the parent's user_query.
    resp = await planner.resume_from(
        parent_trace=parent,
        from_step=parent.steps[0].step_id,  # the stage-0 plan marker
        override=None,
        runner=runner,
        context=ctx,
    )

    assert captured["input_text"] == parent.user_query
    assert resp.run_id == ctx.run_id
    assert ctx.recorder.trace.parent_run_id == "planner-parent"
    assert ctx.recorder.trace.edit["stage"] == 0
    # resume_from owns persistence for the stage-0 (re-plan) fork.
    assert "trace" in runner.persisted


async def test_planner_resume_from_unknown_step_raises() -> None:
    parent, _step_ids, _plan = _parent_trace_with_plan(2)
    planner = _make_planner()

    class NoopRunner:
        def ensure_recorder(self, *a, **k):
            return False

    with pytest.raises(ValueError, match="not found"):
        await planner.resume_from(
            parent_trace=parent,
            from_step="does-not-exist",
            override=None,
            runner=NoopRunner(),
            context=create_run_context(max_turns=5),
        )


async def test_planner_resume_mid_plan_without_embedded_plan_raises() -> None:
    """A trace whose stage-0 marker carries no plan can't resume a mid-plan step."""
    planner = _make_planner()
    rec = TraceRecorder("legacy-parent", root_agent="research-planner", checkpoint=True)
    # Stage-0 marker WITHOUT an embedded plan (legacy trace).
    rec.record_workflow_step("research-planner", stage=0, label="plan")
    rec.record_workflow_step("research-planner", stage=1, label="worker")
    sid = rec.record_llm_call(
        "worker",
        1,
        output="out 1",
        messages_snapshot=[{"role": "user", "content": "input 1"}],
    )
    parent = rec.build_trace(final_response="out 1")

    class NoopRunner:
        def ensure_recorder(self, *a, **k):
            return False

    with pytest.raises(ValueError, match="no recoverable plan"):
        await planner.resume_from(
            parent_trace=parent,
            from_step=sid,
            override=None,
            runner=NoopRunner(),
            context=create_run_context(max_turns=5),
        )
