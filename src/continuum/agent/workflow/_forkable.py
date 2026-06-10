"""
Shared helpers for Forkable workflow orchestrators (Phases 4-6).

Workflow orchestrators (Loop, Reflection, Supervised, Planner, Parallel, Scatter,
Debate) own a single decision trace spanning their sub-agents, mark stage / branch
boundaries with ``WORKFLOW_STEP`` records, and resume their control flow from a
step via the :class:`~continuum.agent.interfaces.forkable.Forkable` protocol.

These functions centralize the boilerplate that is identical across them so each
orchestrator only has to express its own control flow:

* :func:`segment_by_markers` — map each step to the stage / branch index that owns
  it, using the ``WORKFLOW_STEP`` markers (with an agent-name-transition fallback
  for traces recorded before markers existed).
* :func:`resumed_input` — recover the input a resumed stage should re-run with,
  from that stage's first checkpoint, with the fork ``override`` applied.
* :func:`link_lineage` — stamp the forked run's recorder with its parentage.
"""

from __future__ import annotations

from typing import Any


def segment_by_markers(trace: Any) -> tuple[dict[str, int], dict[int, Any]]:
    """Return ``(step_id -> stage_index, stage_index -> first snapshot-bearing step)``.

    Preferred: read the ``WORKFLOW_STEP`` markers, which carry the authoritative
    ``stage`` index in their ``decision`` — correct even when consecutive stages
    reuse the same agent, or run concurrently and were recorded out of order.
    Fallback (traces without markers): infer boundaries from agent-name
    transitions.
    """
    from continuum.agent.trace.types import StepKind

    step_stage: dict[str, int] = {}
    stage_first: dict[int, Any] = {}

    has_markers = any(s.kind == StepKind.WORKFLOW_STEP for s in trace.steps)
    if has_markers:
        cur = 0
        for s in trace.steps:
            if s.kind == StepKind.WORKFLOW_STEP:
                if isinstance(s.decision, dict) and "stage" in s.decision:
                    cur = s.decision["stage"]
                step_stage[s.step_id] = cur
                continue
            step_stage[s.step_id] = cur
            stage_first.setdefault(cur, s)
        return step_stage, stage_first

    stage = -1
    prev_agent: str | None = None
    for s in trace.steps:
        if s.agent_name != prev_agent:
            stage += 1
            prev_agent = s.agent_name
            stage_first[stage] = s
        step_stage[s.step_id] = stage
    return step_stage, stage_first


def resumed_input(stage_first_step: Any, override: dict[str, Any] | None, fallback: str) -> str:
    """Recover a resumed stage's input: the last user message in its first step's
    message checkpoint, with ``override`` applied; fall back to the run query."""
    from continuum.agent.trace.fork import apply_override

    snap = getattr(stage_first_step, "messages_snapshot", None) or []
    edited = apply_override(snap, override)
    for m in reversed(edited):
        if m.get("role") == "user":
            return str(m.get("content", ""))
    return fallback


def link_lineage(
    context: Any, parent_trace: Any, from_step: str, override: dict[str, Any] | None, stage: int
) -> None:
    """Stamp the forked run's recorder so its trace links back to the parent."""
    if context.recorder is not None:
        context.recorder.trace.parent_run_id = parent_trace.run_id
        context.recorder.trace.forked_from_step = from_step
        context.recorder.trace.edit = {"override": override, "stage": stage}


# --------------------------------------------------------------------------- #
# Phase 6 — ordered capture for concurrent orchestrators (Parallel/Scatter/Debate)
# --------------------------------------------------------------------------- #
def branch_recorder_context(context: Any, *, index: int) -> tuple[Any, Any]:
    """Return ``(branch_context, branch_recorder)`` for one concurrent branch.

    The branch gets an isolated :class:`RunContext` (``branch_copy``) AND its own
    fresh recorder, so concurrent branches record into separate traces instead of
    interleaving into the shared one. After the branches finish, the orchestrator
    merges each branch recorder's steps back in deterministic order with
    ``recorder.absorb(...)``. ``branch_recorder`` is ``None`` when tracing is off.
    """
    branch_ctx = context.branch_copy()
    if context.recorder is None:
        branch_ctx.recorder = None
        return branch_ctx, None
    from continuum.agent.trace.recorder import TraceRecorder

    parent_rec = context.recorder
    branch_rec = TraceRecorder(
        f"{context.run_id}:b{index}",
        parent_rec.trace.root_agent,
        "",
        checkpoint=getattr(parent_rec, "checkpoint", False),
    )
    branch_ctx.recorder = branch_rec
    return branch_ctx, branch_rec


def branch_outputs_from_trace(parent_trace: Any) -> dict[int, str]:
    """Recover each branch's final output text from a concurrent parent trace,
    keyed by stage (branch) index — used to re-merge sibling branches when a fork
    re-runs only one branch. The final output of a stage is the last non-marker
    step's ``output`` within that stage segment.
    """
    step_stage, _ = segment_by_markers(parent_trace)
    from continuum.agent.trace.types import StepKind

    outputs: dict[int, str] = {}
    for s in parent_trace.steps:
        if s.kind == StepKind.WORKFLOW_STEP:
            continue
        stage = step_stage.get(s.step_id)
        if stage is None:
            continue
        if s.output:
            outputs[stage] = str(s.output)
    return outputs
