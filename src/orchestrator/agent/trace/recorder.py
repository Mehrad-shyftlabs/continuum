"""
Decision-trace recorder.

A :class:`TraceRecorder` lives on the :class:`RunContext` for a run. The executor
(and, later, handoff/workflow code) calls ``record_*`` at the same points where
spans are already opened; each recorded step is stamped with the current Langfuse
``span_id`` so the trace and the observability view stay in sync — one source of
truth, two projections.

The recorder is intentionally dumb and synchronous: appending a step is a cheap
in-memory operation that never does I/O and never raises into the run. When
tracing is disabled, no recorder is created and the executor's ``if recorder``
guards skip everything.
"""

from __future__ import annotations

from typing import Any

from orchestrator.agent.trace.types import DecisionStep, DecisionTrace, StepKind
from orchestrator.observability.trace_context import get_current_span_id


class TraceRecorder:
    """Collects :class:`DecisionStep` records and assembles a :class:`DecisionTrace`."""

    def __init__(
        self, run_id: str, root_agent: str, user_query: str = "", *, checkpoint: bool = False
    ) -> None:
        self._trace = DecisionTrace(run_id=run_id, root_agent=root_agent, user_query=user_query)
        self._counter = 0
        self.checkpoint = checkpoint
        # The handoff stack of the currently-executing agent (root → … → current).
        # The executor sets this at each loop entry so every recorded step is
        # stamped with the agent path that produced it (used by fork()).
        self.current_agent_stack: list[str] = []

    # -- recording --------------------------------------------------------- #
    def record(
        self,
        kind: StepKind,
        agent_name: str,
        *,
        turn: int = 0,
        parent_id: str | None = None,
        input: Any = None,
        decision: Any = None,
        rationale: str | None = None,
        output: Any = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        latency_ms: int = 0,
        status: str = "ok",
        error: str | None = None,
        messages_snapshot: list[Any] | None = None,
    ) -> str:
        """Append a step and return its id (so callers can nest children under it)."""
        self._counter += 1
        step = DecisionStep(
            step_id=f"s{self._counter}",
            kind=kind,
            agent_name=agent_name,
            turn=turn,
            parent_id=parent_id,
            agent_stack=list(self.current_agent_stack),
            input=input,
            decision=decision,
            rationale=rationale,
            output=output,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            status=status,
            error=error,
            span_id=get_current_span_id(),
            messages_snapshot=messages_snapshot,
        )
        self._trace.add(step)
        return step.step_id

    # -- convenience wrappers (the executor's vocabulary) ------------------ #
    def record_llm_call(
        self,
        agent_name: str,
        turn: int,
        *,
        output: str = "",
        parent_id: str | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        latency_ms: int = 0,
        decision: Any = None,
        messages_snapshot: list[Any] | None = None,
    ) -> str:
        return self.record(
            StepKind.LLM_CALL,
            agent_name,
            turn=turn,
            parent_id=parent_id,
            output=output,
            decision=decision,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            messages_snapshot=messages_snapshot if self.checkpoint else None,
        )

    def record_reasoning(
        self, agent_name: str, turn: int, thought: str, *, parent_id: str | None = None
    ) -> str:
        return self.record(
            StepKind.REASONING,
            agent_name,
            turn=turn,
            parent_id=parent_id,
            decision="think",
            rationale=thought,
        )

    def record_tool_call(
        self,
        agent_name: str,
        turn: int,
        tool_name: str,
        args: Any,
        output: Any,
        *,
        parent_id: str | None = None,
        latency_ms: int = 0,
        status: str = "ok",
        error: str | None = None,
    ) -> str:
        return self.record(
            StepKind.TOOL_CALL,
            agent_name,
            turn=turn,
            parent_id=parent_id,
            input={"tool": tool_name, "args": args},
            decision=f"call {tool_name}",
            output=output,
            latency_ms=latency_ms,
            status=status,
            error=error,
        )

    def record_handoff(
        self,
        from_agent: str,
        to_agent: str,
        turn: int,
        reason: str = "",
        *,
        parent_id: str | None = None,
    ) -> str:
        self._trace.handoff_chain.append(to_agent)
        return self.record(
            StepKind.HANDOFF,
            from_agent,
            turn=turn,
            parent_id=parent_id,
            decision={"handoff_to": to_agent},
            rationale=reason,
        )

    # -- assembly ---------------------------------------------------------- #
    def build_trace(
        self,
        *,
        final_response: str = "",
        status: str = "success",
        completed_at: Any = None,
    ) -> DecisionTrace:
        self._trace.final_response = final_response
        self._trace.status = status
        self._trace.completed_at = completed_at
        return self._trace

    @property
    def trace(self) -> DecisionTrace:
        return self._trace

    def last_step_id(self) -> str | None:
        return self._trace.steps[-1].step_id if self._trace.steps else None
