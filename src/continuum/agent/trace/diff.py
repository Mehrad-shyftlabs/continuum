"""
Trace diff — compare two runs to see how a changed decision affected reasoning.

``diff_traces`` aligns two traces step-by-step (by position) and surfaces what
changed: the headline final answer plus per-step divergences in kind / decision /
output. This is the read-side of the "what-if": run, fork-with-an-edit, diff.
"""

from __future__ import annotations

from typing import Any

from continuum.agent.trace.types import DecisionStep, DecisionTrace


def _step_view(s: DecisionStep) -> dict[str, Any]:
    return {"step_id": s.step_id, "kind": s.kind.value, "decision": s.decision, "output": s.output}


def diff_traces(a: DecisionTrace, b: DecisionTrace) -> dict[str, Any]:
    """Structured diff of two traces (``a`` = before, ``b`` = after)."""
    changed_final = a.final_response != b.final_response

    step_diffs: list[dict[str, Any]] = []
    for i in range(max(len(a.steps), len(b.steps))):
        sa = a.steps[i] if i < len(a.steps) else None
        sb = b.steps[i] if i < len(b.steps) else None
        if sa is None and sb is not None:
            step_diffs.append({"index": i, "kind": "added", "after": _step_view(sb)})
        elif sb is None and sa is not None:
            step_diffs.append({"index": i, "kind": "removed", "before": _step_view(sa)})
        elif (
            sa is not None
            and sb is not None
            and (sa.kind != sb.kind or sa.decision != sb.decision or sa.output != sb.output)
        ):
            step_diffs.append(
                {
                    "index": i,
                    "kind": "changed",
                    "before": _step_view(sa),
                    "after": _step_view(sb),
                }
            )

    return {
        "run_a": a.run_id,
        "run_b": b.run_id,
        "final_response": {
            "changed": changed_final,
            "before": a.final_response,
            "after": b.final_response,
        },
        "metrics": {"before": a.metrics(), "after": b.metrics()},
        "steps_changed": len(step_diffs),
        "step_diffs": step_diffs,
    }
