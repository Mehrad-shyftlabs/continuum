"""
Forkable — the resume protocol for workflow orchestrators.

``runner.fork`` knows how to resume a single agent loop (and, since Phase 2, a
handoff chain). Workflow orchestrators (Sequential, Router, Loop, …) have their
own control flow that lives in the orchestrator, not the executor loop, so they
cannot be resumed by replaying one loop. Instead each opts in by implementing
:class:`Forkable`: given the parent trace and the step to resume from, it re-runs
its own control flow forward from that point with the edit applied.

``runner.fork`` detects a Forkable orchestrator (passed as ``agent=`` or resolved
as the run's root agent) and delegates to :meth:`resume_from`; otherwise it uses
the built-in single-agent / handoff resume.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from continuum.agent.runner import AgentRunner
    from continuum.agent.trace.types import DecisionTrace
    from continuum.agent.types import AgentResponse, RunContext


@runtime_checkable
class Forkable(Protocol):
    """A workflow orchestrator that can resume its control flow from a step."""

    async def resume_from(
        self,
        *,
        parent_trace: DecisionTrace,
        from_step: str,
        override: dict[str, Any] | None,
        runner: AgentRunner,
        context: RunContext,
    ) -> AgentResponse:
        """Re-run this orchestrator forward from ``from_step``, applying ``override``.

        Implementations replay/skip work up to the step's position (reusing the
        parent trace), apply the edit, run the remaining control flow, and record
        a new trace on ``context`` whose ``parent_run_id``/``forked_from_step``
        link back to the parent run. The parent run is never mutated.
        """
        ...
