"""
End-to-end: streaming runs capture the decision trace (S1) — real LLM, real Redis.

Before S1, ``run_stream`` persisted only an empty trace shell. Now the streaming
loop shares the same capture helpers as the non-streaming executor, so a streamed
run produces the SAME trace (and is equally forkable) as a non-streamed one.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.e2e

from tests.e2e.conftest import skip_if_no_api_key as _skip_if_no_api_key
from tests.e2e.conftest import skip_on_api_error as _skip_on_api_error


@pytest.fixture
def trace_checkpoint_enabled(monkeypatch):
    from continuum.config import settings

    monkeypatch.setattr(settings, "decision_trace_enabled", True)
    monkeypatch.setattr(settings, "decision_trace_detail", "full")
    monkeypatch.setattr(settings, "decision_trace_store", "redis")
    monkeypatch.setattr(settings, "decision_trace_checkpoint", True)

    from continuum.agent.trace import config as trace_config

    trace_config.get_trace_store.cache_clear()
    yield
    trace_config.get_trace_store.cache_clear()


def _agent():
    from continuum.agent.base import BaseAgent
    from continuum.agent.config import AgentConfig, AgentMemoryConfig

    return BaseAgent(
        name="answerer",
        instructions="Answer in one short sentence.",
        memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
        config=AgentConfig(log_to_session=False, session_history_turns=0),
    )


class TestStreamingTraceE2E:
    @_skip_on_api_error
    async def test_streaming_records_and_matches_nonstreaming(self, trace_checkpoint_enabled):
        """A streamed run records the same step kinds as the equivalent
        non-streamed run, and its LLM step carries a checkpoint (forkable)."""
        _skip_if_no_api_key()

        from continuum.agent.runner import AgentRunner
        from continuum.agent.trace.config import get_trace_store
        from continuum.agent.trace.types import StepKind
        from continuum.agent.types import EventType

        runner = AgentRunner()
        store = get_trace_store()
        prompt = "What is the capital of France?"

        # Non-streaming baseline.
        resp = await runner.run(_agent(), prompt)
        base = await store.get(resp.run_id)
        assert base is not None
        base_kinds = [s.kind.value for s in base.steps]

        # Streaming run — consume all events, capture the run_id.
        stream_run_id = None
        async for ev in runner.run_stream(_agent(), prompt):
            stream_run_id = ev.run_id
            if ev.type == EventType.RUN_END:
                break
        assert stream_run_id is not None

        streamed = await store.get(stream_run_id)
        assert streamed is not None, "streaming run persisted no trace (S1 regression)"
        streamed_kinds = [s.kind.value for s in streamed.steps]

        # Same shape: a single LLM_CALL step (no tools) in both.
        assert streamed_kinds == base_kinds, (streamed_kinds, base_kinds)
        assert streamed_kinds == ["llm_call"]

        # The streamed LLM step carries a checkpoint → forkable.
        llm_step = next(s for s in streamed.steps if s.kind == StepKind.LLM_CALL)
        assert llm_step.messages_snapshot, "streamed step has no checkpoint → not forkable"

        await store.delete(resp.run_id)
        await store.delete(stream_run_id)

    @_skip_on_api_error
    async def test_decision_step_events_streamed_live(self, trace_checkpoint_enabled):
        """S2: each recorded step is emitted as a DECISION_STEP event during the
        stream, carrying the step's kind and a forkable flag."""
        _skip_if_no_api_key()

        from continuum.agent.runner import AgentRunner
        from continuum.agent.trace.config import get_trace_store
        from continuum.agent.types import EventType

        runner = AgentRunner()
        run_id = None
        decision_events = []
        async for ev in runner.run_stream(_agent(), "What is 2 + 2?"):
            run_id = ev.run_id
            if ev.type == EventType.DECISION_STEP:
                decision_events.append(ev.data)
            if ev.type == EventType.RUN_END:
                break

        # At least the final LLM decision step was streamed live, and it advertises
        # itself as forkable (carries a checkpoint).
        assert decision_events, "no DECISION_STEP events emitted during the stream"
        llm_events = [d for d in decision_events if d["kind"] == "llm_call"]
        assert llm_events, [d["kind"] for d in decision_events]
        assert llm_events[-1]["forkable"] is True

        if run_id:
            await get_trace_store().delete(run_id)

    @_skip_on_api_error
    async def test_streamed_run_is_forkable(self, trace_checkpoint_enabled):
        """A streamed run can be rewound with runner.fork, exactly like a
        non-streamed one."""
        _skip_if_no_api_key()

        from continuum.agent.runner import AgentRunner
        from continuum.agent.trace.config import get_trace_store
        from continuum.agent.types import EventType

        agent = _agent()
        runner = AgentRunner()
        runner.register_agent(agent)
        store = get_trace_store()

        stream_run_id = None
        async for ev in runner.run_stream(agent, "Name a primary color."):
            stream_run_id = ev.run_id
            if ev.type == EventType.RUN_END:
                break
        assert stream_run_id is not None
        parent = await store.get(stream_run_id)
        assert parent is not None and parent.steps

        step = parent.steps[0]
        forked = await runner.fork(
            stream_run_id,
            step.step_id,
            override={"replace_last_user": "Name a primary color other than red."},
            label="fork a streamed run",
        )
        child = await store.get(forked.run_id)
        assert child is not None
        assert child.parent_run_id == stream_run_id
        assert forked.content

        await store.delete(stream_run_id)
        await store.delete(forked.run_id)
