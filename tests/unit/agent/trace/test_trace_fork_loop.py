"""
Phase 5 — LoopAgent is Forkable: resume_from re-runs from the right iteration.

No LLM/network: a recorded parent trace is hand-built, ``_drive`` is stubbed to
capture where the resume begins, and we assert the iteration mapping, the
override-applied input, and the lineage stamp.
"""

from __future__ import annotations

import pytest

from continuum.agent.base import BaseAgent
from continuum.agent.interfaces.forkable import Forkable
from continuum.agent.trace.recorder import TraceRecorder
from continuum.agent.types import (
    AgentResponse,
    ResponseStatus,
    TerminationConfig,
    TerminationType,
)
from continuum.agent.utils.context_utils import create_run_context
from continuum.agent.workflow.loop import LoopAgent


def _make_loop() -> LoopAgent:
    return LoopAgent(
        name="refiner-loop",
        agent=BaseAgent(name="refiner", instructions="Improve the text."),
        termination=TerminationConfig(
            type=TerminationType.OUTPUT_MATCH, pattern="STOP", max_iterations=5
        ),
    )


def _parent_trace_with_iterations(n: int) -> tuple[object, list[str]]:
    rec = TraceRecorder("loop-parent", root_agent="refiner-loop", checkpoint=True)
    step_ids: list[str] = []
    for i in range(n):
        rec.record_workflow_step(
            "refiner-loop", stage=i, label="refiner", agent_stack=["refiner-loop"]
        )
        sid = rec.record_llm_call(
            "refiner",
            1,
            output=f"draft {i}",
            agent_stack=["refiner-loop", "refiner"],
            messages_snapshot=[{"role": "user", "content": f"input {i}"}],
        )
        step_ids.append(sid)
    return rec.build_trace(final_response=f"draft {n - 1}"), step_ids


def test_loop_satisfies_forkable() -> None:
    assert isinstance(_make_loop(), Forkable)


async def test_loop_resume_from_starts_at_iteration(monkeypatch) -> None:
    parent, step_ids = _parent_trace_with_iterations(3)
    loop = _make_loop()

    captured: dict = {}

    async def fake_drive(
        self, current_input, runner, context, *, start_iteration, original_input, llm_client
    ):  # noqa: E501
        captured["start_iteration"] = start_iteration
        captured["current_input"] = current_input
        captured["original_input"] = original_input
        r = AgentResponse(content="ok", agent_name=self.name, status=ResponseStatus.SUCCESS)
        r.run_id = context.run_id
        return r

    monkeypatch.setattr(LoopAgent, "_drive", fake_drive)

    persisted: dict = {}

    class FakeRunner:
        def ensure_recorder(self, context, root_agent, query=""):
            context.recorder = TraceRecorder(context.run_id, root_agent, query, checkpoint=True)
            return True

        async def persist_decision_trace(self, context, result):
            persisted["trace"] = context.recorder.trace

    ctx = create_run_context(max_turns=5)
    resp = await loop.resume_from(
        parent_trace=parent,
        from_step=step_ids[1],  # a step inside iteration index 1
        override={"replace_last_user": "edited input"},
        runner=FakeRunner(),
        context=ctx,
    )

    # Resumes at the iteration that owned the forked step.
    assert captured["start_iteration"] == 1
    # Override applied to the recovered input.
    assert captured["current_input"] == "edited input"
    assert captured["original_input"] == parent.user_query
    assert resp.run_id == ctx.run_id

    # Lineage stamped back to the parent.
    assert ctx.recorder.trace.parent_run_id == "loop-parent"
    assert ctx.recorder.trace.forked_from_step == step_ids[1]
    assert ctx.recorder.trace.edit["stage"] == 1
    assert "trace" in persisted


async def test_loop_resume_from_unknown_step_raises() -> None:
    parent, _ = _parent_trace_with_iterations(2)
    loop = _make_loop()

    class FakeRunner:
        def ensure_recorder(self, *a, **k):
            return False

    with pytest.raises(ValueError, match="not found"):
        await loop.resume_from(
            parent_trace=parent,
            from_step="does-not-exist",
            override=None,
            runner=FakeRunner(),
            context=create_run_context(max_turns=5),
        )
