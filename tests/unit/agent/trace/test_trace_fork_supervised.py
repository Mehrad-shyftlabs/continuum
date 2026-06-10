"""
Phase 5 — SupervisedSequentialAgent is Forkable: resume_from re-runs from the
right pipeline stage.

No LLM/network: a recorded parent trace is hand-built, ``_drive`` is stubbed to
capture where the resume begins, and we assert the stage mapping, the
override-applied input, and the lineage stamp.
"""

from __future__ import annotations

import pytest

from continuum.agent.base import BaseAgent
from continuum.agent.interfaces.forkable import Forkable
from continuum.agent.trace.recorder import TraceRecorder
from continuum.agent.types import AgentResponse, ResponseStatus
from continuum.agent.utils.context_utils import create_run_context
from continuum.agent.workflow.supervised import SupervisedSequentialAgent


def _make_supervised() -> SupervisedSequentialAgent:
    return SupervisedSequentialAgent(
        name="supervised-pipeline",
        agents=[
            BaseAgent(name="researcher", instructions="Research the topic."),
            BaseAgent(name="analyst", instructions="Analyze the findings."),
            BaseAgent(name="writer", instructions="Write the summary."),
        ],
    )


def _parent_trace_with_stages(n: int) -> tuple[object, list[str]]:
    rec = TraceRecorder("supervised-parent", root_agent="supervised-pipeline", checkpoint=True)
    labels = ["researcher", "analyst", "writer"]
    step_ids: list[str] = []
    for i in range(n):
        label = labels[i]
        rec.record_workflow_step(
            "supervised-pipeline", stage=i, label=label, agent_stack=["supervised-pipeline"]
        )
        sid = rec.record_llm_call(
            label,
            1,
            output=f"stage {i} output",
            agent_stack=["supervised-pipeline", label],
            messages_snapshot=[{"role": "user", "content": f"input {i}"}],
        )
        step_ids.append(sid)
    return rec.build_trace(final_response=f"stage {n - 1} output"), step_ids


def test_supervised_satisfies_forkable() -> None:
    assert isinstance(_make_supervised(), Forkable)


async def test_supervised_resume_from_starts_at_stage(monkeypatch) -> None:
    parent, step_ids = _parent_trace_with_stages(3)
    pipeline = _make_supervised()

    captured: dict = {}

    async def fake_drive(self, current_input, runner, context, *, start_stage, original_input):
        captured["start_stage"] = start_stage
        captured["current_input"] = current_input
        captured["original_input"] = original_input
        r = AgentResponse(content="ok", agent_name=self.name, status=ResponseStatus.SUCCESS)
        r.run_id = context.run_id
        return r

    monkeypatch.setattr(SupervisedSequentialAgent, "_drive", fake_drive)

    persisted: dict = {}

    class FakeRunner:
        def ensure_recorder(self, context, root_agent, query=""):
            context.recorder = TraceRecorder(context.run_id, root_agent, query, checkpoint=True)
            return True

        async def persist_decision_trace(self, context, result):
            persisted["trace"] = context.recorder.trace

    ctx = create_run_context(max_turns=5)
    resp = await pipeline.resume_from(
        parent_trace=parent,
        from_step=step_ids[1],  # a step inside pipeline stage index 1
        override={"replace_last_user": "edited input"},
        runner=FakeRunner(),
        context=ctx,
    )

    # Resumes at the stage that owned the forked step.
    assert captured["start_stage"] == 1
    # Override applied to the recovered input.
    assert captured["current_input"] == "edited input"
    assert captured["original_input"] == parent.user_query
    assert resp.run_id == ctx.run_id

    # Lineage stamped back to the parent.
    assert ctx.recorder.trace.parent_run_id == "supervised-parent"
    assert ctx.recorder.trace.forked_from_step == step_ids[1]
    assert ctx.recorder.trace.edit["stage"] == 1
    assert "trace" in persisted


async def test_supervised_resume_from_unknown_step_raises() -> None:
    parent, _ = _parent_trace_with_stages(2)
    pipeline = _make_supervised()

    class FakeRunner:
        def ensure_recorder(self, *a, **k):
            return False

    with pytest.raises(ValueError, match="not found"):
        await pipeline.resume_from(
            parent_trace=parent,
            from_step="does-not-exist",
            override=None,
            runner=FakeRunner(),
            context=create_run_context(max_turns=5),
        )
