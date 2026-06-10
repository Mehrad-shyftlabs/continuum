"""
Phase 6 — ordered capture: TraceRecorder.absorb merges concurrent branches into
one deterministic, contiguous, re-numbered, segmentable trace.
"""

from __future__ import annotations

from continuum.agent.trace.recorder import TraceRecorder
from continuum.agent.trace.types import StepKind
from continuum.agent.workflow._forkable import segment_by_markers


def _branch(root: str, agent: str, n: int) -> TraceRecorder:
    rec = TraceRecorder(f"{root}:b", root_agent=root, checkpoint=False)
    parent = rec.record_llm_call(agent, 1, output=f"{agent} llm", agent_stack=[agent])
    rec.record_tool_call(
        agent, 1, "t", {"q": 1}, f"{agent} tool out", parent_id=parent, agent_stack=[agent]
    )
    assert len(rec.trace.steps) == n
    return rec


def test_absorb_orders_and_renumbers_branches() -> None:
    main = TraceRecorder("par", root_agent="parallel", checkpoint=False)
    b0 = _branch("parallel", "web", 2)
    b1 = _branch("parallel", "db", 2)

    # Branches finished in arbitrary real order; absorb in deterministic branch order.
    main.absorb(
        b1.trace.steps, stage=1, label="db", orchestrator_name="parallel"
    )  # out of order on purpose
    main.absorb(b0.trace.steps, stage=0, label="web", orchestrator_name="parallel")

    steps = main.trace.steps
    # 2 markers + 2 + 2 branch steps
    assert sum(1 for s in steps if s.kind == StepKind.WORKFLOW_STEP) == 2
    # All step ids are unique after renumbering.
    ids = [s.step_id for s in steps]
    assert len(ids) == len(set(ids))
    # Orchestrator prepended to every absorbed branch step's stack.
    for s in steps:
        if s.kind != StepKind.WORKFLOW_STEP:
            assert s.agent_stack[0] == "parallel"
    # parent_id remap stayed internally consistent (the tool step points at the
    # llm step's NEW id, which exists in the merged trace).
    tool_steps = [s for s in steps if s.kind == StepKind.TOOL_CALL]
    for ts in tool_steps:
        assert ts.parent_id in ids

    # Segmentation recovers the branches by stage index.
    step_stage, stage_first = segment_by_markers(main.trace)
    stages = {step_stage[s.step_id] for s in steps if s.kind != StepKind.WORKFLOW_STEP}
    assert stages == {0, 1}
    assert set(stage_first) == {0, 1}
