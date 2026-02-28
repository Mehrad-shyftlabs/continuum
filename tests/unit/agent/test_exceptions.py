"""Unit tests for agent exceptions."""

import pytest

from orchestrator.agent.exceptions import (
    AgentError,
    AgentExecutionError,
    HandoffDepthExceededError,
    HandoffError,
    HandoffTargetNotFoundError,
    MaxTurnsExceededError,
    RunStateError,
    WorkflowError,
)
import logging

logger = logging.getLogger(__name__)


class TestAgentExceptions:
    def test_agent_error_hierarchy(self):
        logger.info("AgentExceptions: agent error hierarchy")
        err = AgentError("test")
        assert isinstance(err, Exception)
        assert err.message == "test"

    def test_agent_execution_error_attributes(self):
        logger.info("AgentExceptions: agent execution error attributes")
        err = AgentExecutionError(
            "exec failed",
            agent_name="test-agent",
            run_id="r1",
            trace_id="t1",
        )
        assert err.agent_name == "test-agent"

    def test_max_turns_exceeded(self):
        logger.info("AgentExceptions: max turns exceeded")
        err = MaxTurnsExceededError("too many turns")
        assert isinstance(err, AgentError)

    def test_handoff_error_types(self):
        logger.info("AgentExceptions: handoff error types")
        err = HandoffError("handoff failed")
        assert isinstance(err, AgentError)

    def test_handoff_target_not_found(self):
        logger.info("AgentExceptions: handoff target not found")
        err = HandoffTargetNotFoundError(from_agent="agent1", to_agent="other")
        assert isinstance(err, HandoffError)

    def test_handoff_depth_exceeded(self):
        logger.info("AgentExceptions: handoff depth exceeded")
        err = HandoffDepthExceededError(current_depth=5, max_depth=3)
        assert isinstance(err, HandoffError)

    def test_workflow_error(self):
        logger.info("AgentExceptions: workflow error")
        err = WorkflowError("workflow failed")
        assert isinstance(err, AgentError)

    def test_run_state_error(self):
        logger.info("AgentExceptions: run state error")
        err = RunStateError("state error")
        assert isinstance(err, AgentError)
