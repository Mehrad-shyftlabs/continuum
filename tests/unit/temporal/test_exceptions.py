"""Tests for Temporal exception hierarchy."""

import pytest

from orchestrator.exceptions import OrchestratorError
from orchestrator.temporal.exceptions import (
    AgentNotRegisteredError,
    ApprovalTimeoutError,
    TemporalActivityError,
    TemporalConnectionError,
    TemporalError,
    TemporalWorkflowError,
    WorkflowCancelledError,
)
import logging

logger = logging.getLogger(__name__)


class TestExceptionHierarchy:
    def test_temporal_error_inherits_orchestrator_error(self):
        logger.info("ExceptionHierarchy: temporal error inherits orchestrator error")
        err = TemporalError("test")
        assert isinstance(err, OrchestratorError)

    def test_connection_error_inherits_temporal_error(self):
        logger.info("ExceptionHierarchy: connection error inherits temporal error")
        err = TemporalConnectionError("fail")
        assert isinstance(err, TemporalError)
        assert isinstance(err, OrchestratorError)

    def test_workflow_error_inherits_temporal_error(self):
        logger.info("ExceptionHierarchy: workflow error inherits temporal error")
        err = TemporalWorkflowError("fail")
        assert isinstance(err, TemporalError)

    def test_activity_error_inherits_temporal_error(self):
        logger.info("ExceptionHierarchy: activity error inherits temporal error")
        err = TemporalActivityError("fail")
        assert isinstance(err, TemporalError)

    def test_agent_not_registered_error(self):
        logger.info("ExceptionHierarchy: agent not registered error")
        err = AgentNotRegisteredError(
            "Agent 'foo' not found",
            agent_name="foo",
            available_agents=["bar", "baz"],
        )
        assert isinstance(err, TemporalError)
        assert err.context["agent_name"] == "foo"
        assert "bar" in err.context["available_agents"]

    def test_approval_timeout_error(self):
        logger.info("ExceptionHierarchy: approval timeout error")
        err = ApprovalTimeoutError(
            "Timed out",
            request_id="req-123",
            timeout_seconds=3600,
        )
        assert err.context["request_id"] == "req-123"
        assert err.context["timeout_seconds"] == 3600

    def test_workflow_cancelled_error(self):
        logger.info("ExceptionHierarchy: workflow cancelled error")
        err = WorkflowCancelledError(
            "Cancelled",
            workflow_id="wf-abc",
        )
        assert err.context["workflow_id"] == "wf-abc"

    def test_connection_error_context(self):
        logger.info("ExceptionHierarchy: connection error context")
        err = TemporalConnectionError(
            "Cannot connect",
            host="localhost:7233",
            namespace="default",
        )
        assert err.context["host"] == "localhost:7233"
        assert err.context["namespace"] == "default"

    def test_workflow_error_context(self):
        logger.info("ExceptionHierarchy: workflow error context")
        err = TemporalWorkflowError(
            "Workflow failed",
            workflow_id="wf-1",
            workflow_type="AgentWorkflow",
        )
        assert err.context["workflow_id"] == "wf-1"

    def test_activity_error_context(self):
        logger.info("ExceptionHierarchy: activity error context")
        err = TemporalActivityError(
            "Activity failed",
            activity_name="run_agent",
            agent_name="test-agent",
        )
        assert err.context["agent_name"] == "test-agent"

    def test_default_messages(self):
        logger.info("ExceptionHierarchy: default messages")
        assert "Temporal" in TemporalError().message
        assert "connect" in TemporalConnectionError().message.lower()
        assert "workflow" in TemporalWorkflowError().message.lower()
