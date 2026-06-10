"""
Decision trace: a durable, ordered record of how an agent reached its answer.

Every meaningful decision during a run (LLM call, ReAct thought, tool call,
handoff, memory retrieval, routing) is captured as a :class:`DecisionStep`. The
steps form a tree (via ``parent_id``) so the trace spans multi-turn loops and
multi-agent handoffs. The assembled :class:`DecisionTrace` is attached to the
``AgentResponse`` (response metadata) and persisted via a :class:`TraceStore`.

This package is additive and feature-flagged (``DECISION_TRACE_ENABLED``).
The pure core here (types) has no infra dependency; the store and recorder are
imported lazily by callers so importing this module is always cheap and safe.
"""

from __future__ import annotations

from continuum.agent.trace.diff import diff_traces
from continuum.agent.trace.fork import apply_override
from continuum.agent.trace.types import (
    SCHEMA_VERSION,
    DecisionStep,
    DecisionTrace,
    StepKind,
    TraceDetail,
)

__all__ = [
    "SCHEMA_VERSION",
    "DecisionStep",
    "DecisionTrace",
    "StepKind",
    "TraceDetail",
    "apply_override",
    "diff_traces",
]
