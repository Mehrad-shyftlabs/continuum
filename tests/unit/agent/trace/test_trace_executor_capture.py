"""
Slice 4 — capture wired into the real Executor loop, driven by a mocked LLM.

Exercises the actual recorder calls in executor.py (no tokens spent, no network).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from orchestrator.agent.base import BaseAgent
from orchestrator.agent.config import AgentConfig
from orchestrator.agent.execution.executor import Executor
from orchestrator.agent.trace import StepKind
from orchestrator.agent.trace.recorder import TraceRecorder
from orchestrator.agent.types import RunContext, RunState


def _usage(p: int = 10, c: int = 5) -> SimpleNamespace:
    return SimpleNamespace(prompt_tokens=p, completion_tokens=c, total_tokens=p + c)


def _llm_response(content: str = "", tool_calls=None) -> SimpleNamespace:
    return SimpleNamespace(
        content=content, tool_calls=tool_calls, usage=_usage(), model="gpt-4o-mini"
    )


class _FakeFn:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, id: str, name: str, arguments: str) -> None:
        self.id = id
        self.function = _FakeFn(name, arguments)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "function": {"name": self.function.name, "arguments": self.function.arguments},
        }


def _agent() -> BaseAgent:
    return BaseAgent(name="support_agent", instructions="help", config=AgentConfig())


def _context() -> RunContext:
    ctx = RunContext(run_id="run_x", session_id=None)
    ctx.recorder = TraceRecorder("run_x", "support_agent", "Is my order delayed?")
    return ctx


def _run_state() -> RunState:
    return RunState(run_id="run_x")


async def test_captures_single_llm_final_answer() -> None:
    llm = SimpleNamespace(chat=AsyncMock(return_value=_llm_response("No, ships tomorrow.")))
    ex = Executor(llm_client=llm)
    ctx = _context()

    resp = await ex.execute_loop(_agent(), [{"role": "user", "content": "q"}], ctx, _run_state())

    assert resp.content == "No, ships tomorrow."
    steps = ctx.recorder.trace.steps
    assert len(steps) == 1
    assert steps[0].kind is StepKind.LLM_CALL
    assert steps[0].decision == "final_answer"
    assert steps[0].total_tokens == 15


async def test_checkpoint_captures_messages_for_fork() -> None:
    """With checkpoint on, the LLM step stores the messages sent (fork resume point)."""
    llm = SimpleNamespace(chat=AsyncMock(return_value=_llm_response("hi")))
    ex = Executor(llm_client=llm)
    ctx = RunContext(run_id="run_cp")
    ctx.recorder = TraceRecorder("run_cp", "support_agent", "q", checkpoint=True)

    await ex.execute_loop(_agent(), [{"role": "user", "content": "q"}], ctx, _run_state())

    snap = ctx.recorder.trace.steps[0].messages_snapshot
    assert snap is not None
    assert any(m.get("role") == "user" and m.get("content") == "q" for m in snap)


async def test_no_checkpoint_leaves_snapshot_empty() -> None:
    llm = SimpleNamespace(chat=AsyncMock(return_value=_llm_response("hi")))
    ex = Executor(llm_client=llm)
    ctx = _context()  # recorder has checkpoint=False

    await ex.execute_loop(_agent(), [{"role": "user", "content": "q"}], ctx, _run_state())
    assert ctx.recorder.trace.steps[0].messages_snapshot is None


async def test_captures_tool_call_then_answer() -> None:
    tc = _FakeToolCall("c1", "lookup_order", '{"order_id": 123}')
    llm = SimpleNamespace(
        chat=AsyncMock(
            side_effect=[
                _llm_response("", tool_calls=[tc]),  # turn 1: call tool
                _llm_response("No, ships tomorrow."),  # turn 2: final answer
            ]
        )
    )
    tool_handler = SimpleNamespace(
        execute_tools_batch=AsyncMock(
            return_value=[
                {"role": "tool", "tool_call_id": "c1", "content": '{"status": "shipped"}'}
            ]
        )
    )
    ex = Executor(llm_client=llm, tool_handler=tool_handler)
    ctx = _context()

    await ex.execute_loop(_agent(), [{"role": "user", "content": "q"}], ctx, _run_state())

    kinds = [s.kind for s in ctx.recorder.trace.steps]
    assert kinds == [StepKind.LLM_CALL, StepKind.TOOL_CALL, StepKind.LLM_CALL]
    tool_step = ctx.recorder.trace.steps[1]
    assert tool_step.input == {"tool": "lookup_order", "args": {"order_id": 123}}
    assert tool_step.output == '{"status": "shipped"}'
    # tool step nests under turn-1's LLM decision
    assert tool_step.parent_id == ctx.recorder.trace.steps[0].step_id


async def test_checkpoint_snapshot_excludes_turns_own_assistant() -> None:
    """Regression: a step's checkpoint is the messages SENT that turn, captured
    BEFORE the turn's own assistant output is appended. Otherwise fork() would
    replay an already-finished conversation. With a tool turn then a final answer,
    the final LLM step's snapshot ends with the tool result, never the answer."""
    tc = _FakeToolCall("c1", "lookup_order", '{"order_id": 123}')
    llm = SimpleNamespace(
        chat=AsyncMock(
            side_effect=[
                _llm_response("", tool_calls=[tc]),
                _llm_response("Final answer."),
            ]
        )
    )
    tool_handler = SimpleNamespace(
        execute_tools_batch=AsyncMock(
            return_value=[
                {"role": "tool", "tool_call_id": "c1", "content": '{"status": "shipped"}'}
            ]
        )
    )
    ex = Executor(llm_client=llm, tool_handler=tool_handler)
    ctx = RunContext(run_id="run_cp2")
    ctx.recorder = TraceRecorder("run_cp2", "support_agent", "q", checkpoint=True)

    await ex.execute_loop(_agent(), [{"role": "user", "content": "q"}], ctx, _run_state())

    final = [s for s in ctx.recorder.trace.steps if s.kind is StepKind.LLM_CALL][-1]
    snap = final.messages_snapshot
    assert snap is not None
    assert snap[-1]["role"] == "tool"  # the resume point, NOT the assistant answer
    assert all(m.get("content") != "Final answer." for m in snap)


async def test_agent_stack_stamped_on_steps() -> None:
    """Each step records the handoff stack active when it ran (root → … → agent),
    so fork() can resume the right agent in a multi-agent run."""
    llm = SimpleNamespace(chat=AsyncMock(return_value=_llm_response("ok")))
    ex = Executor(llm_client=llm)
    ctx = _context()
    rs = RunState(run_id="run_x")
    rs.agent_stack = ["triage", "support_agent"]

    await ex.execute_loop(_agent(), [{"role": "user", "content": "q"}], ctx, rs)

    assert ctx.recorder.trace.steps[0].agent_stack == ["triage", "support_agent"]


async def test_return_to_parent_restores_parent_stack() -> None:
    """After A hands off to B with return_to_parent=True and B returns, A's
    continuation steps must be stamped with A's stack — not B's. Regression for
    the stack mis-stamping bug (the executor must reset the recorder's stack
    after popping the child)."""
    from orchestrator.agent.config import AgentMemoryConfig
    from orchestrator.agent.types import Handoff, HandoffResult, ResponseStatus
    from orchestrator.agent.types import AgentResponse as _AR

    agent_a = BaseAgent(
        name="agent-a",
        instructions="a",
        handoffs=[Handoff(target_agent="agent-b", description="go to b", return_to_parent=True)],
        config=AgentConfig(),
        memory_config=AgentMemoryConfig(),
    )

    # A's LLM: turn 1 → handoff tool call; turn 2 → final answer (after B returns).
    handoff_tc = _FakeToolCall("h1", "handoff_to_agent-b", '{"reason": "need b"}')
    llm = SimpleNamespace(
        chat=AsyncMock(
            side_effect=[
                _llm_response("", tool_calls=[handoff_tc]),
                _llm_response("Final answer from A."),
            ]
        )
    )

    class _StubHandoff:
        """Simulates entering B: pushes B and stamps the recorder with the deeper
        stack (as B's real loop would), then returns a successful response."""

        _executor = None

        async def execute_handoff(self, *, agent, target_name, tool_call, messages, context, run_state):
            run_state.push_agent(target_name)
            if context.recorder is not None:
                context.recorder.current_agent_stack = run_state.get_agent_stack_snapshot()
            return HandoffResult(
                handoff_id="h1",
                from_agent=agent.name,
                to_agent=target_name,
                success=True,
                response=_AR(content="B result", agent_name=target_name, status=ResponseStatus.SUCCESS),
            )

    ex = Executor(llm_client=llm, handoff_executor=_StubHandoff())
    ctx = RunContext(run_id="run_rtp")
    ctx.recorder = TraceRecorder("run_rtp", "agent-a", "q")
    rs = RunState(run_id="run_rtp")
    rs.push_agent("agent-a")

    await ex.execute_loop(agent_a, [{"role": "user", "content": "q"}], ctx, rs)

    # A's final LLM step (its turn-2 continuation after B returned) carries A's stack.
    final = [s for s in ctx.recorder.trace.steps if s.kind is StepKind.LLM_CALL][-1]
    assert final.agent_stack == ["agent-a"], f"parent continuation mis-stamped: {final.agent_stack}"


async def test_no_recorder_is_a_noop() -> None:
    """With no recorder on the context, the loop runs unchanged."""
    llm = SimpleNamespace(chat=AsyncMock(return_value=_llm_response("hi")))
    ex = Executor(llm_client=llm)
    ctx = RunContext(run_id="run_y")  # recorder stays None

    resp = await ex.execute_loop(_agent(), [{"role": "user", "content": "q"}], ctx, _run_state())
    assert resp.content == "hi"
    assert ctx.recorder is None
