"""
Planner embeds its plan in the stage-0 WORKFLOW_STEP marker so a mid-plan fork
can recover it. That trace must survive the JSON round-trip the Redis store uses
(`json.dumps(trace.to_dict())`) — the in-memory store used in other tests never
serializes, so this guards the persisted path explicitly.
"""

from __future__ import annotations

import json

from continuum.agent.base import BaseAgent
from continuum.agent.trace.recorder import TraceRecorder
from continuum.agent.trace.types import DecisionTrace, StepKind
from continuum.agent.utils.context_utils import create_run_context
from continuum.agent.workflow.planner import PlannerAgent


def test_planner_plan_marker_survives_json_roundtrip() -> None:
    planner = PlannerAgent(
        name="planner",
        agents=[BaseAgent(name="worker", instructions="do the step")],
    )

    ctx = create_run_context(max_turns=5)
    ctx.recorder = TraceRecorder(ctx.run_id, "planner", "achieve the goal", checkpoint=True)

    # A realistic plan as produced by _generate_plan (list of plain dicts).
    plan_steps = [
        {"agent": "worker", "task": "gather data", "step": 1},
        {"agent": "worker", "task": "summarize", "step": 2},
    ]
    planner._record_plan_marker(ctx, plan_steps)

    trace = ctx.recorder.trace

    # Exactly the Redis store's save path.
    blob = json.dumps(trace.to_dict())
    restored = DecisionTrace.from_dict(json.loads(blob))

    # The plan embedded in the stage-0 marker survives the round-trip intact.
    markers = [s for s in restored.steps if s.kind == StepKind.WORKFLOW_STEP]
    assert markers, "expected a stage-0 plan marker"
    stage0 = markers[0]
    assert stage0.decision["stage"] == 0
    assert stage0.decision["plan"] == plan_steps
