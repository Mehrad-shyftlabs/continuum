"""Unit tests for the decision-trace data model (Slice 1)."""

from __future__ import annotations

import json

from continuum.agent.trace import (
    SCHEMA_VERSION,
    DecisionStep,
    DecisionTrace,
    StepKind,
    TraceDetail,
)


def _sample_trace() -> DecisionTrace:
    trace = DecisionTrace(
        run_id="run_a1b2c3", root_agent="support_agent", user_query="Is my order delayed?"
    )
    s1 = trace.add(
        DecisionStep(
            step_id="s1",
            kind=StepKind.REASONING,
            agent_name="support_agent",
            turn=1,
            decision="call lookup_order",
            rationale="I don't know the order state yet",
        )
    )
    trace.add(
        DecisionStep(
            step_id="s2",
            kind=StepKind.TOOL_CALL,
            agent_name="support_agent",
            turn=1,
            parent_id=s1.step_id,
            input={"tool": "lookup_order", "args": {"order_id": 123}},
            output={"status": "shipped", "eta": "tomorrow"},
            latency_ms=1200,
        )
    )
    trace.add(
        DecisionStep(
            step_id="s3",
            kind=StepKind.LLM_CALL,
            agent_name="support_agent",
            turn=2,
            parent_id="s2",
            output="No, your order ships tomorrow as planned.",
            prompt_tokens=210,
            completion_tokens=30,
            total_tokens=240,
            latency_ms=1000,
        )
    )
    trace.final_response = "No, your order ships tomorrow as planned."
    return trace


class TestDecisionStep:
    def test_roundtrips_exactly(self) -> None:
        step = DecisionStep(
            step_id="s2",
            kind=StepKind.TOOL_CALL,
            agent_name="a",
            turn=1,
            input={"tool": "x"},
            decision={"call": "x"},
            output={"ok": True},
            total_tokens=5,
            span_id="span-123",
        )
        restored = DecisionStep.from_dict(step.to_dict())
        assert restored == step

    def test_to_dict_is_json_serializable(self) -> None:
        step = DecisionStep(step_id="s1", kind=StepKind.LLM_CALL, agent_name="a")
        # Must not raise — kind/datetime are encoded as primitives.
        json.dumps(step.to_dict())

    def test_kind_serialized_as_value(self) -> None:
        step = DecisionStep(step_id="s1", kind=StepKind.HANDOFF, agent_name="a")
        assert step.to_dict()["kind"] == "handoff"


class TestDecisionTrace:
    def test_roundtrips_exactly(self) -> None:
        trace = _sample_trace()
        restored = DecisionTrace.from_dict(json.loads(json.dumps(trace.to_dict())))
        assert restored.run_id == trace.run_id
        assert restored.root_agent == trace.root_agent
        assert [s.step_id for s in restored.steps] == [s.step_id for s in trace.steps]
        assert [s.kind for s in restored.steps] == [s.kind for s in trace.steps]

    def test_schema_version_present(self) -> None:
        assert _sample_trace().to_dict()["schema_version"] == SCHEMA_VERSION

    def test_tree_parent_ids_resolve(self) -> None:
        trace = _sample_trace()
        ids = {s.step_id for s in trace.steps}
        for step in trace.steps:
            if step.parent_id is not None:
                assert step.parent_id in ids, f"dangling parent_id on {step.step_id}"

    def test_metrics_aggregate(self) -> None:
        m = _sample_trace().metrics()
        assert m["step_count"] == 3
        assert m["turn_count"] == 2
        assert m["total_tokens"] == 240
        assert m["total_latency_ms"] == 2200
        assert m["error_count"] == 0
        assert m["agents"] == ["support_agent"]

    def test_error_count(self) -> None:
        trace = _sample_trace()
        trace.add(
            DecisionStep(
                step_id="s4",
                kind=StepKind.TOOL_CALL,
                agent_name="support_agent",
                status="error",
                error="boom",
            )
        )
        assert trace.metrics()["error_count"] == 1


class TestTraceDetail:
    def test_full_is_default(self) -> None:
        trace = _sample_trace()
        assert trace.to_dict() == trace.to_dict(TraceDetail.FULL)
