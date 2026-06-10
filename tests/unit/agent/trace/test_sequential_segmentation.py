"""
Unit test for SequentialAgent._segment_stages — the WORKFLOW_STEP-based stage
mapping that fixes mis-segmentation when adjacent stages reuse the same agent.
No LLM, no network — builds a trace by hand.
"""

from __future__ import annotations

from continuum.agent.base import BaseAgent
from continuum.agent.config import AgentConfig
from continuum.agent.trace.recorder import TraceRecorder
from continuum.agent.workflow.sequential import SequentialAgent


def _agent(name: str) -> BaseAgent:
    return BaseAgent(name=name, instructions="x", config=AgentConfig())


def test_adjacent_same_agent_stages_segment_correctly() -> None:
    """Pipeline [writer, reviewer, reviewer]: stages 1 and 2 share 'reviewer'.
    Using the WORKFLOW_STEP markers, a step in stage 2 maps to stage 2 — not
    merged into stage 1 (which is what agent-name segmentation would do)."""
    rec = TraceRecorder(run_id="seg", root_agent="pipe")

    rec.record_workflow_step("pipe", stage=0, label="writer")
    rec.record_llm_call("writer", 1, output="draft", decision="final_answer")

    rec.record_workflow_step("pipe", stage=1, label="reviewer")
    s_stage1 = rec.record_llm_call("reviewer", 1, output="review-1", decision="final_answer")

    rec.record_workflow_step("pipe", stage=2, label="reviewer")
    s_stage2 = rec.record_llm_call("reviewer", 1, output="review-2", decision="final_answer")

    pipeline = SequentialAgent(
        name="pipe", agents=[_agent("writer"), _agent("reviewer"), _agent("reviewer")]
    )
    step_stage, stage_first = pipeline._segment_stages(rec.trace)

    # The two reviewer stages are kept distinct (the bug would merge them).
    assert step_stage[s_stage1] == 1
    assert step_stage[s_stage2] == 2
    assert stage_first[2].step_id == s_stage2
    assert stage_first[1].step_id == s_stage1


def test_fallback_segments_by_agent_name_without_markers() -> None:
    """Traces with no WORKFLOW_STEP markers (e.g. legacy) still segment by
    agent-name transitions."""
    rec = TraceRecorder(run_id="seg2", root_agent="pipe")
    a = rec.record_llm_call("intake", 1, output="i", decision="final_answer")
    b = rec.record_llm_call("assessor", 1, output="a", decision="final_answer")
    c = rec.record_llm_call("decider", 1, output="d", decision="final_answer")

    pipeline = SequentialAgent(
        name="pipe", agents=[_agent("intake"), _agent("assessor"), _agent("decider")]
    )
    step_stage, _ = pipeline._segment_stages(rec.trace)
    assert (step_stage[a], step_stage[b], step_stage[c]) == (0, 1, 2)
