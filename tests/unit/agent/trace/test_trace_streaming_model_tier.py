"""
The smart-layer (model_tier) streaming router path now captures a decision trace
too — a ROUTING step + an LLM step, each with a checkpoint (forkable) — not just
the normal streaming loop. Mocks the smart-layer streamer so no gateway is needed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from continuum.agent.base import BaseAgent
from continuum.agent.config import AgentConfig, AgentMemoryConfig, RouterConfig
from continuum.agent.runner import AgentRunner
from continuum.agent.trace.types import StepKind
from continuum.agent.types import EventType, Route


@pytest.fixture
def trace_on(monkeypatch):
    from continuum.agent.trace import config as trace_config
    from continuum.config import settings

    monkeypatch.setattr(settings, "decision_trace_enabled", True)
    monkeypatch.setattr(settings, "decision_trace_detail", "full")
    monkeypatch.setattr(settings, "decision_trace_store", "memory")
    monkeypatch.setattr(settings, "decision_trace_checkpoint", True)
    trace_config.get_trace_store.cache_clear()
    yield
    trace_config.get_trace_store.cache_clear()


async def test_model_tier_streaming_captures_trace(trace_on, monkeypatch) -> None:
    import continuum.agent.runner as runner_mod

    # Force the smart-layer streaming branch and stub its streamer + tier parsing.
    monkeypatch.setattr(runner_mod.app_settings, "smart_layer_enabled", True, raising=False)
    monkeypatch.setattr(runner_mod, "parse_product_tier", lambda _t: None)

    async def fake_stream(agent, llm_client, *, user_text, ctx):
        yield SimpleNamespace(kind="routing", routing={"tier": "cheap", "model": "m"}, text=None)
        yield SimpleNamespace(kind="content_delta", routing=None, text="Paris.")

    monkeypatch.setattr(runner_mod, "stream_model_tier_turn", fake_stream)

    _mem = AgentMemoryConfig(search_memories=False, store_memories=False)
    _cfg = AgentConfig(log_to_session=False, session_history_turns=0)
    specialist = BaseAgent(name="geo", instructions="Answer.", memory_config=_mem, config=_cfg)
    router = runner_mod.RouterAgent(
        name="tier-router",
        routes=[Route(agent_name="geo", description="geography", condition="geo")],
        router_config=RouterConfig(routing_strategy="model_tier"),
        memory_config=_mem,
        config=_cfg,
    )

    runner = AgentRunner()
    runner.register_agent(specialist)

    run_id = None
    decision_events = []
    async for ev in runner.run_stream(router, "Capital of France?"):
        run_id = ev.run_id
        if ev.type == EventType.DECISION_STEP:
            decision_events.append(ev.data)
        if ev.type == EventType.RUN_END:
            break
    assert run_id is not None

    # S2: the routing + answer steps are streamed live as DECISION_STEP events.
    live_kinds = [d["kind"] for d in decision_events]
    assert "routing" in live_kinds and "llm_call" in live_kinds, live_kinds

    from continuum.agent.trace.config import get_trace_store

    trace = await get_trace_store().get(run_id)
    assert trace is not None, "model_tier streaming run persisted no trace"
    kinds = [s.kind for s in trace.steps]
    assert StepKind.ROUTING in kinds, kinds
    assert StepKind.LLM_CALL in kinds, kinds
    # Both carry the checkpoint that makes the run forkable.
    routing_step = next(s for s in trace.steps if s.kind == StepKind.ROUTING)
    assert routing_step.messages_snapshot, "routing step has no checkpoint → not forkable"
    assert routing_step.decision.get("tier") == "cheap"

    await get_trace_store().delete(run_id)
