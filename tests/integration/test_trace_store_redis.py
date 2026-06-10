"""
Integration tests for RedisTraceStore — real session Redis (port 6380).

These exercise the *real connection path* (no injected client): the store builds
its own ``redis.asyncio`` client from settings, exactly as it does in production.
The unit tests in tests/unit/agent/trace cover only the infra-free backends.
"""

from __future__ import annotations

import pytest

from continuum.agent.trace import DecisionStep, DecisionTrace, StepKind
from continuum.agent.trace.store import RedisTraceStore

pytestmark = [pytest.mark.integration, pytest.mark.redis]


@pytest.fixture
async def store():
    """Real RedisTraceStore against the live session Redis, isolated by prefix."""
    s = RedisTraceStore(prefix="test:trace:integration", ttl_seconds=60)
    try:
        await s._redis.ping()
    except Exception:
        pytest.skip("Session Redis not reachable on configured host/port")
    yield s
    # Clean up only this test's keys (never a global flush).
    async for key in s._redis.scan_iter(match="test:trace:integration:*"):
        await s._redis.delete(key)
    await s._redis.aclose()


def _trace(run_id: str) -> DecisionTrace:
    t = DecisionTrace(run_id=run_id, root_agent="support_agent", user_query="Is my order delayed?")
    think = t.add(
        DecisionStep(
            step_id="s1",
            kind=StepKind.REASONING,
            agent_name="support_agent",
            turn=1,
            rationale="check the order first",
        )
    )
    t.add(
        DecisionStep(
            step_id="s2",
            kind=StepKind.TOOL_CALL,
            agent_name="support_agent",
            turn=1,
            parent_id=think.step_id,
            input={"tool": "lookup_order", "args": {"order_id": 123}},
            output={"status": "shipped"},
            latency_ms=1200,
        )
    )
    t.add(
        DecisionStep(
            step_id="s3",
            kind=StepKind.LLM_CALL,
            agent_name="support_agent",
            turn=2,
            parent_id="s2",
            output="No, ships tomorrow.",
            total_tokens=240,
        )
    )
    t.final_response = "No, ships tomorrow."
    return t


class TestRedisTraceStoreIntegration:
    async def test_save_then_get_roundtrips_through_real_redis(self, store, test_id) -> None:
        run_id = f"run-{test_id}"
        await store.save(_trace(run_id))

        got = await store.get(run_id)
        assert got is not None
        assert got.run_id == run_id
        assert got.final_response == "No, ships tomorrow."
        assert [s.kind for s in got.steps] == [
            StepKind.REASONING,
            StepKind.TOOL_CALL,
            StepKind.LLM_CALL,
        ]
        assert got.steps[1].input == {"tool": "lookup_order", "args": {"order_id": 123}}
        assert got.metrics()["total_tokens"] == 240

    async def test_ttl_applied_on_real_key(self, store, test_id) -> None:
        run_id = f"run-ttl-{test_id}"
        await store.save(_trace(run_id))
        ttl = await store._redis.ttl(f"test:trace:integration:{run_id}")
        assert 0 < ttl <= 60

    async def test_get_unknown_returns_none(self, store, test_id) -> None:
        assert await store.get(f"missing-{test_id}") is None

    async def test_delete_removes_key(self, store, test_id) -> None:
        run_id = f"run-del-{test_id}"
        await store.save(_trace(run_id))
        assert await store.delete(run_id) is True
        assert await store.get(run_id) is None
        assert await store.delete(run_id) is False
