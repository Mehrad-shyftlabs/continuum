"""Slice 4 — RunFinalizer builds, persists, and attaches the decision trace."""

from __future__ import annotations

from unittest.mock import MagicMock

from continuum.agent.execution.run_finalizer import RunFinalizer
from continuum.agent.trace.recorder import TraceRecorder
from continuum.agent.trace.store import InMemoryTraceStore
from continuum.agent.trace.types import TraceDetail
from continuum.agent.types import AgentResponse, ResponseStatus, RunContext


def _finalizer() -> RunFinalizer:
    return RunFinalizer(
        session_service=MagicMock(),
        context_service=MagicMock(),
        lifecycle=MagicMock(),
    )


def _context_with_recorder() -> RunContext:
    ctx = RunContext(run_id="run_z")
    rec = TraceRecorder("run_z", "a", "hello?")
    rec.record_llm_call("a", 1, output="hi there", total_tokens=12)
    ctx.recorder = rec
    return ctx


def _patch_store_and_detail(monkeypatch, detail: TraceDetail) -> InMemoryTraceStore:
    store = InMemoryTraceStore()
    monkeypatch.setattr("continuum.agent.trace.config.get_trace_store", lambda: store)
    monkeypatch.setattr("continuum.agent.trace.config.trace_detail", lambda: detail)
    return store


async def test_persists_and_attaches_full(monkeypatch) -> None:
    store = _patch_store_and_detail(monkeypatch, TraceDetail.FULL)
    ctx = _context_with_recorder()
    resp = AgentResponse(content="hi there", status=ResponseStatus.SUCCESS)

    await _finalizer()._finalize_decision_trace(ctx, resp)

    # attached to the response
    assert resp.decision_trace is not None
    assert resp.decision_trace["run_id"] == "run_z"
    assert resp.decision_trace["final_response"] == "hi there"
    assert resp.decision_trace["metrics"]["total_tokens"] == 12
    # persisted by run_id
    persisted = await store.get("run_z")
    assert persisted is not None
    assert persisted.final_response == "hi there"


async def test_off_detail_persists_but_attaches_nothing(monkeypatch) -> None:
    store = _patch_store_and_detail(monkeypatch, TraceDetail.OFF)
    ctx = _context_with_recorder()
    resp = AgentResponse(content="hi there", status=ResponseStatus.SUCCESS)

    await _finalizer()._finalize_decision_trace(ctx, resp)

    assert resp.decision_trace is None  # nothing in the response
    assert await store.get("run_z") is not None  # but still persisted


async def test_no_recorder_does_nothing(monkeypatch) -> None:
    _patch_store_and_detail(monkeypatch, TraceDetail.FULL)
    ctx = RunContext(run_id="run_none")  # no recorder
    resp = AgentResponse(content="x", status=ResponseStatus.SUCCESS)

    await _finalizer()._finalize_decision_trace(ctx, resp)
    assert resp.decision_trace is None


async def test_suppressed_subagent_skips(monkeypatch) -> None:
    """Workflow/handoff sub-agents share the recorder but must not finalize."""
    store = _patch_store_and_detail(monkeypatch, TraceDetail.FULL)
    ctx = _context_with_recorder()
    ctx.suppress_session_log = True
    resp = AgentResponse(content="hi", status=ResponseStatus.SUCCESS)

    await _finalizer()._finalize_decision_trace(ctx, resp)
    assert resp.decision_trace is None
    assert await store.get("run_z") is None


async def test_store_failure_is_swallowed(monkeypatch) -> None:
    class Boom:
        async def save(self, *a, **k):
            raise RuntimeError("down")

    monkeypatch.setattr("continuum.agent.trace.config.get_trace_store", lambda: Boom())
    monkeypatch.setattr("continuum.agent.trace.config.trace_detail", lambda: TraceDetail.FULL)
    ctx = _context_with_recorder()
    resp = AgentResponse(content="hi", status=ResponseStatus.SUCCESS)

    # Must not raise.
    await _finalizer()._finalize_decision_trace(ctx, resp)
