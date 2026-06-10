"""Unit tests for TraceRecorder (Slice 3). Replays a scripted run — no LLM."""

from __future__ import annotations

from continuum.agent.trace import StepKind
from continuum.agent.trace.recorder import TraceRecorder


def test_records_scripted_order_query_run() -> None:
    """Replay the 'Is my order delayed?' decision sequence and assert the trace."""
    rec = TraceRecorder(
        run_id="run_a1b2c3", root_agent="support_agent", user_query="Is my order delayed?"
    )

    think = rec.record_reasoning("support_agent", 1, "I'll look up the order first.")
    tool = rec.record_tool_call(
        "support_agent",
        1,
        "lookup_order",
        {"order_id": 123},
        {"status": "shipped", "eta": "tomorrow"},
        parent_id=think,
        latency_ms=1200,
    )
    rec.record_llm_call(
        "support_agent",
        2,
        output="No, ships tomorrow as planned.",
        parent_id=tool,
        total_tokens=240,
        latency_ms=1000,
    )

    trace = rec.build_trace(final_response="No, ships tomorrow as planned.")

    kinds = [s.kind for s in trace.steps]
    assert kinds == [StepKind.REASONING, StepKind.TOOL_CALL, StepKind.LLM_CALL]
    # ids auto-increment and nest correctly
    assert [s.step_id for s in trace.steps] == ["s1", "s2", "s3"]
    assert trace.steps[1].parent_id == "s1"
    assert trace.steps[2].parent_id == "s2"
    assert trace.final_response == "No, ships tomorrow as planned."
    assert trace.metrics()["total_tokens"] == 240


def test_handoff_appends_to_chain() -> None:
    rec = TraceRecorder("r", "triage")
    rec.record_handoff("triage", "billing", turn=1, reason="user asked about an invoice")
    trace = rec.build_trace()
    assert trace.handoff_chain == ["billing"]
    assert trace.steps[0].kind is StepKind.HANDOFF
    assert trace.steps[0].decision == {"handoff_to": "billing", "return_to_parent": False}


def test_span_id_stamped_from_context(monkeypatch) -> None:
    monkeypatch.setattr("continuum.agent.trace.recorder.get_current_span_id", lambda: "span-xyz")
    rec = TraceRecorder("r", "a")
    rec.record_llm_call("a", 1, output="hi")
    assert rec.trace.steps[0].span_id == "span-xyz"


def test_error_step_recorded() -> None:
    rec = TraceRecorder("r", "a")
    rec.record_tool_call("a", 1, "bad_tool", {}, None, status="error", error="boom")
    trace = rec.build_trace(status="error")
    assert trace.status == "error"
    assert trace.metrics()["error_count"] == 1


def test_last_step_id_helper() -> None:
    rec = TraceRecorder("r", "a")
    assert rec.last_step_id() is None
    sid = rec.record_llm_call("a", 1)
    assert rec.last_step_id() == sid


def test_new_step_kinds_and_recorders() -> None:
    """MEMORY_WRITE / GUARDRAIL / WORKFLOW_STEP kinds exist and their recorder
    helpers produce steps of the right kind that round-trip."""
    from continuum.agent.trace.types import DecisionTrace

    rec = TraceRecorder(run_id="run_k", root_agent="wf")

    mw = rec.record_memory_write("agent-a", ["user likes dark mode"])
    gr = rec.record_guardrail("agent-a", "pii_scanner", blocked=False, modified=True)
    ws = rec.record_workflow_step("pipeline", stage=2, label="assessor")

    by_id = {s.step_id: s for s in rec.trace.steps}
    assert by_id[mw].kind is StepKind.MEMORY_WRITE
    assert by_id[gr].kind is StepKind.GUARDRAIL
    assert by_id[gr].decision["modified"] is True
    assert by_id[ws].kind is StepKind.WORKFLOW_STEP
    assert by_id[ws].decision == {"stage": 2, "label": "assessor"}

    # round-trip through dict (persistence) preserves the new kinds
    restored = DecisionTrace.from_dict(rec.trace.to_dict())
    kinds = {s.step_id: s.kind for s in restored.steps}
    assert kinds[mw] is StepKind.MEMORY_WRITE
    assert kinds[gr] is StepKind.GUARDRAIL
    assert kinds[ws] is StepKind.WORKFLOW_STEP
