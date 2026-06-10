"""
Phase 5 — ReflectionAgent is Forkable: resume_from re-runs from the right attempt.

No LLM/network: a recorded parent trace is hand-built, ``_drive`` is stubbed to
capture where the resume begins, and we assert the attempt mapping, the
override-applied input, and the lineage stamp.
"""

from __future__ import annotations

import pytest

from continuum.agent.base import BaseAgent
from continuum.agent.config import ReflectionConfig
from continuum.agent.interfaces.forkable import Forkable
from continuum.agent.trace.recorder import TraceRecorder
from continuum.agent.types import AgentResponse, ResponseStatus
from continuum.agent.utils.context_utils import create_run_context
from continuum.agent.workflow.reflection import ReflectionAgent


def _make_reflection() -> ReflectionAgent:
    return ReflectionAgent(
        name="reflection-agent",
        agent=BaseAgent(name="writer", instructions="Write concise summaries."),
        reflection_config=ReflectionConfig(max_reflections=3),
    )


def _parent_trace_with_attempts(n: int) -> tuple[object, list[str]]:
    rec = TraceRecorder("reflection-parent", root_agent="reflection-agent", checkpoint=True)
    step_ids: list[str] = []
    for i in range(n):
        rec.record_workflow_step(
            "reflection-agent", stage=i, label="writer", agent_stack=["reflection-agent"]
        )
        sid = rec.record_llm_call(
            "writer",
            1,
            output=f"draft {i}",
            agent_stack=["reflection-agent", "writer"],
            messages_snapshot=[{"role": "user", "content": f"input {i}"}],
        )
        step_ids.append(sid)
    return rec.build_trace(final_response=f"draft {n - 1}"), step_ids


def test_reflection_satisfies_forkable() -> None:
    assert isinstance(_make_reflection(), Forkable)


async def test_reflection_resume_from_starts_at_attempt(monkeypatch) -> None:
    parent, step_ids = _parent_trace_with_attempts(3)
    reflection = _make_reflection()

    captured: dict = {}

    async def fake_drive(
        self, current_input, runner, context, *, start_attempt, original_input, llm_client
    ):  # noqa: E501
        captured["start_attempt"] = start_attempt
        captured["current_input"] = current_input
        captured["original_input"] = original_input
        r = AgentResponse(content="ok", agent_name=self.name, status=ResponseStatus.SUCCESS)
        r.run_id = context.run_id
        return r

    monkeypatch.setattr(ReflectionAgent, "_drive", fake_drive)

    persisted: dict = {}

    class FakeRunner:
        def ensure_recorder(self, context, root_agent, query=""):
            context.recorder = TraceRecorder(context.run_id, root_agent, query, checkpoint=True)
            return True

        async def persist_decision_trace(self, context, result):
            persisted["trace"] = context.recorder.trace

    ctx = create_run_context(max_turns=5)
    resp = await reflection.resume_from(
        parent_trace=parent,
        from_step=step_ids[1],  # a step inside attempt index 1
        override={"replace_last_user": "edited input"},
        runner=FakeRunner(),
        context=ctx,
    )

    # Resumes at the attempt that owned the forked step.
    assert captured["start_attempt"] == 1
    # Override applied to the recovered input.
    assert captured["current_input"] == "edited input"
    assert captured["original_input"] == parent.user_query
    assert resp.run_id == ctx.run_id

    # Lineage stamped back to the parent.
    assert ctx.recorder.trace.parent_run_id == "reflection-parent"
    assert ctx.recorder.trace.forked_from_step == step_ids[1]
    assert ctx.recorder.trace.edit["stage"] == 1
    assert "trace" in persisted


async def test_reflection_resume_from_unknown_step_raises() -> None:
    parent, _ = _parent_trace_with_attempts(2)
    reflection = _make_reflection()

    class FakeRunner:
        def ensure_recorder(self, *a, **k):
            return False

    with pytest.raises(ValueError, match="not found"):
        await reflection.resume_from(
            parent_trace=parent,
            from_step="does-not-exist",
            override=None,
            runner=FakeRunner(),
            context=create_run_context(max_turns=5),
        )
