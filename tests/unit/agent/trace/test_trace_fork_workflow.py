"""
Phase 3 — runner.fork delegates to a Forkable workflow orchestrator.

Verifies the protocol foundation with a fake orchestrator (no LLM, no network):
when the forked run's resume target implements Forkable, fork() hands off to its
resume_from() instead of running the built-in single-agent resume.
"""

from __future__ import annotations

from types import SimpleNamespace

from orchestrator.agent.interfaces.forkable import Forkable
from orchestrator.agent.runner import AgentRunner
from orchestrator.agent.trace.types import DecisionStep, DecisionTrace, StepKind
from orchestrator.agent.types import AgentResponse, ResponseStatus


class _FakeForkable:
    """Minimal Forkable orchestrator — structurally satisfies the protocol."""

    def __init__(self) -> None:
        self.name = "fake-pipeline"
        self.config = SimpleNamespace(max_turns=5)
        self.calls: list[dict] = []

    async def resume_from(self, *, parent_trace, from_step, override, runner, context):
        self.calls.append({"from_step": from_step, "override": override})
        return AgentResponse(
            content="resumed-by-orchestrator",
            agent_name=self.name,
            status=ResponseStatus.SUCCESS,
        )


def test_fake_orchestrator_satisfies_protocol() -> None:
    assert isinstance(_FakeForkable(), Forkable)


async def test_fork_delegates_to_forkable(monkeypatch) -> None:
    # In-memory trace store so we can seed a parent run without Redis.
    from orchestrator.agent.trace import config as trace_config
    from orchestrator.config import settings

    monkeypatch.setattr(settings, "decision_trace_store", "memory")
    monkeypatch.setattr(settings, "decision_trace_enabled", True)
    trace_config.get_trace_store.cache_clear()
    store = trace_config.get_trace_store()

    parent = DecisionTrace(run_id="wf-parent", root_agent="fake-pipeline")
    parent.steps.append(DecisionStep(step_id="s1", kind=StepKind.LLM_CALL, agent_name="stage-1"))
    await store.save(parent)

    runner = AgentRunner()
    fake = _FakeForkable()

    resp = await runner.fork("wf-parent", "s1", agent=fake, override={"system": "x"})

    assert resp.content == "resumed-by-orchestrator"
    assert fake.calls == [{"from_step": "s1", "override": {"system": "x"}}]

    trace_config.get_trace_store.cache_clear()
