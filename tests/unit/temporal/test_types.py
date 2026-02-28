"""Tests for Temporal serializable types."""

import pytest
from datetime import datetime, UTC

from orchestrator.temporal.types import (
    AgentActivityParams,
    AgentActivityResult,
    AgentStep,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStep,
    ConditionalStep,
    NotificationParams,
    ParallelStep,
    StepType,
    WaitStep,
    WorkflowInput,
    WorkflowResult,
    parse_step,
)
import logging

logger = logging.getLogger(__name__)


class TestStepType:
    def test_enum_values(self):
        logger.info("StepType: enum values")
        assert StepType.AGENT == "agent"
        assert StepType.APPROVAL == "approval"
        assert StepType.PARALLEL == "parallel"
        assert StepType.CONDITIONAL == "conditional"
        assert StepType.WAIT == "wait"


class TestAgentStep:
    def test_defaults(self):
        logger.info("AgentStep: defaults")
        step = AgentStep(agent_name="test")
        assert step.type == "agent"
        assert step.agent_name == "test"
        assert step.input is None
        assert step.timeout == 300
        assert step.retries == 3
        assert step.metadata == {}

    def test_round_trip(self):
        logger.info("AgentStep: round trip")
        step = AgentStep(agent_name="foo", input="hello", timeout=60)
        data = step.model_dump()
        restored = AgentStep.model_validate(data)
        assert restored == step

    def test_type_literal(self):
        logger.info("AgentStep: type literal")
        step = AgentStep(agent_name="x")
        assert step.model_dump()["type"] == "agent"


class TestApprovalStep:
    def test_defaults(self):
        logger.info("ApprovalStep: defaults")
        step = ApprovalStep(description="Review this")
        assert step.type == "approval"
        assert step.timeout == 86400
        assert step.approvers == []
        assert step.auto_approve_if is None

    def test_round_trip(self):
        logger.info("ApprovalStep: round trip")
        step = ApprovalStep(
            description="Check work",
            approvers=["admin@example.com"],
            timeout=3600,
        )
        data = step.model_dump()
        restored = ApprovalStep.model_validate(data)
        assert restored == step


class TestParallelStep:
    def test_defaults(self):
        logger.info("ParallelStep: defaults")
        step = ParallelStep(agents=[AgentStep(agent_name="a")])
        assert step.type == "parallel"
        assert step.merge_strategy == "concatenate"

    def test_round_trip(self):
        logger.info("ParallelStep: round trip")
        step = ParallelStep(
            agents=[
                AgentStep(agent_name="a"),
                AgentStep(agent_name="b"),
            ],
            merge_strategy="first_success",
        )
        data = step.model_dump()
        restored = ParallelStep.model_validate(data)
        assert len(restored.agents) == 2
        assert restored.merge_strategy == "first_success"


class TestConditionalStep:
    def test_defaults(self):
        logger.info("ConditionalStep: defaults")
        step = ConditionalStep(condition_agent="evaluator")
        assert step.type == "conditional"
        assert step.if_true == []
        assert step.if_false == []

    def test_with_branches(self):
        logger.info("ConditionalStep: with branches")
        step = ConditionalStep(
            condition_agent="evaluator",
            if_true=[{"type": "agent", "agent_name": "a"}],
            if_false=[{"type": "agent", "agent_name": "b"}],
        )
        data = step.model_dump()
        restored = ConditionalStep.model_validate(data)
        assert len(restored.if_true) == 1
        assert len(restored.if_false) == 1


class TestWaitStep:
    def test_creation(self):
        logger.info("WaitStep: creation")
        step = WaitStep(duration_seconds=60)
        assert step.type == "wait"
        assert step.duration_seconds == 60

    def test_round_trip(self):
        logger.info("WaitStep: round trip")
        step = WaitStep(duration_seconds=3600)
        data = step.model_dump()
        restored = WaitStep.model_validate(data)
        assert restored == step


class TestParseStep:
    def test_parse_agent_step(self):
        logger.info("ParseStep: parse agent step")
        step = parse_step({"type": "agent", "agent_name": "foo"})
        assert isinstance(step, AgentStep)
        assert step.agent_name == "foo"

    def test_parse_approval_step(self):
        logger.info("ParseStep: parse approval step")
        step = parse_step({"type": "approval", "description": "check"})
        assert isinstance(step, ApprovalStep)

    def test_parse_parallel_step(self):
        logger.info("ParseStep: parse parallel step")
        step = parse_step({
            "type": "parallel",
            "agents": [{"type": "agent", "agent_name": "a"}],
        })
        assert isinstance(step, ParallelStep)

    def test_parse_conditional_step(self):
        logger.info("ParseStep: parse conditional step")
        step = parse_step({
            "type": "conditional",
            "condition_agent": "eval",
        })
        assert isinstance(step, ConditionalStep)

    def test_parse_wait_step(self):
        logger.info("ParseStep: parse wait step")
        step = parse_step({"type": "wait", "duration_seconds": 10})
        assert isinstance(step, WaitStep)

    def test_parse_unknown_raises(self):
        logger.info("ParseStep: parse unknown raises")
        with pytest.raises(ValueError, match="Unknown step type"):
            parse_step({"type": "unknown"})


class TestAgentActivityParams:
    def test_defaults(self):
        logger.info("AgentActivityParams: defaults")
        p = AgentActivityParams(agent_name="a", input="hello")
        assert p.session_id is None
        assert p.user_id is None
        assert p.metadata == {}
        assert p.tags == []

    def test_round_trip(self):
        logger.info("AgentActivityParams: round trip")
        p = AgentActivityParams(
            agent_name="a",
            input="test",
            session_id="s1",
            user_id="u1",
            metadata={"k": "v"},
            tags=["t1"],
        )
        data = p.model_dump()
        restored = AgentActivityParams.model_validate(data)
        assert restored == p


class TestAgentActivityResult:
    def test_defaults(self):
        logger.info("AgentActivityResult: defaults")
        r = AgentActivityResult(content="hi", status="success")
        assert r.structured_output is None
        assert r.usage == {}
        assert r.agents_used == []
        assert r.error is None

    def test_from_agent_response(self):
        logger.info("AgentActivityResult: from agent response")
        from unittest.mock import MagicMock

        resp = MagicMock()
        resp.content = "test content"
        resp.status.value = "success"
        resp.structured_output = None
        resp.usage.prompt_tokens = 10
        resp.usage.completion_tokens = 20
        resp.usage.total_tokens = 30
        resp.agents_used = ["a1"]
        resp.error = None

        result = AgentActivityResult.from_agent_response(resp)
        assert result.content == "test content"
        assert result.status == "success"
        assert result.usage["total_tokens"] == 30
        assert result.agents_used == ["a1"]

    def test_round_trip(self):
        logger.info("AgentActivityResult: round trip")
        r = AgentActivityResult(
            content="output",
            status="success",
            usage={"total_tokens": 50},
            agents_used=["a"],
        )
        data = r.model_dump()
        restored = AgentActivityResult.model_validate(data)
        assert restored == r


class TestApprovalRequest:
    def test_creation(self):
        logger.info("ApprovalRequest: creation")
        req = ApprovalRequest(
            request_id="r1",
            workflow_id="wf-1",
            step_index=2,
            description="Review",
        )
        assert req.request_id == "r1"
        assert req.timeout == 86400
        assert req.created_at  # auto-generated


class TestApprovalDecision:
    def test_creation(self):
        logger.info("ApprovalDecision: creation")
        d = ApprovalDecision(
            request_id="r1",
            decision="approved",
            decided_by="admin",
        )
        assert d.decision == "approved"
        assert d.decided_at  # auto-generated

    def test_round_trip(self):
        logger.info("ApprovalDecision: round trip")
        d = ApprovalDecision(
            request_id="r1",
            decision="rejected",
            decided_by="user",
            reason="Not ready",
        )
        data = d.model_dump()
        restored = ApprovalDecision.model_validate(data)
        assert restored == d


class TestWorkflowInput:
    def test_creation(self):
        logger.info("WorkflowInput: creation")
        wf = WorkflowInput(
            steps=[{"type": "agent", "agent_name": "a"}],
            initial_input="hello",
        )
        assert len(wf.steps) == 1
        assert wf.session_id is None

    def test_round_trip(self):
        logger.info("WorkflowInput: round trip")
        wf = WorkflowInput(
            steps=[
                {"type": "agent", "agent_name": "a"},
                {"type": "approval", "description": "check"},
            ],
            initial_input="input",
            session_id="s1",
            user_id="u1",
            metadata={"key": "val"},
        )
        data = wf.model_dump()
        restored = WorkflowInput.model_validate(data)
        assert restored == wf


class TestWorkflowResult:
    def test_completed(self):
        logger.info("WorkflowResult: completed")
        r = WorkflowResult(status="completed", content="done")
        assert r.status == "completed"
        assert r.step_results == []
        assert r.error is None

    def test_failed(self):
        logger.info("WorkflowResult: failed")
        r = WorkflowResult(status="failed", error="boom")
        assert r.error == "boom"

    def test_round_trip(self):
        logger.info("WorkflowResult: round trip")
        r = WorkflowResult(
            status="completed",
            content="result",
            step_results=[
                AgentActivityResult(content="a", status="success"),
            ],
            approval_decisions=[
                ApprovalDecision(
                    request_id="r1",
                    decision="approved",
                    decided_by="admin",
                ),
            ],
        )
        data = r.model_dump()
        restored = WorkflowResult.model_validate(data)
        assert restored.status == "completed"
        assert len(restored.step_results) == 1
        assert len(restored.approval_decisions) == 1
