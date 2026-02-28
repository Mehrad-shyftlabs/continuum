"""Unit tests for PrepareRunResult (Issue 7) and _prepare_run return type."""

import pytest

from orchestrator.agent.types import (
    AgentResponse,
    PrepareRunResult,
    ResponseStatus,
    RunContext,
    RunState,
    RunStatus,
)
import logging

logger = logging.getLogger(__name__)


class TestPrepareRunResult:
    """Tests for the PrepareRunResult dataclass."""

    def test_prepare_run_result_fields(self):
        logger.info("PrepareRunResult: prepare run result fields")
        ctx = RunContext(run_id="run_1")
        rs = RunState(run_id="run_1")
        result = PrepareRunResult(
            success=True,
            context=ctx,
            run_state=rs,
            initial_message_count=3,
            tool_context_state={"key": "value"},
        )
        assert result.success is True
        assert result.context is ctx
        assert result.run_state is rs
        assert result.initial_message_count == 3
        assert result.tool_context_state == {"key": "value"}
        assert result.error_response is None

    def test_prepare_run_success(self):
        logger.info("PrepareRunResult: prepare run success")
        result = PrepareRunResult(
            success=True,
            context=RunContext(run_id="run_2"),
            run_state=RunState(run_id="run_2"),
        )
        assert result.success is True
        assert result.context is not None
        assert result.run_state is not None

    def test_prepare_run_validation_failure_returns_error(self):
        logger.info("PrepareRunResult: prepare run validation failure returns error")
        error_resp = AgentResponse(
            content="Input validation failed",
            agent_name="test-agent",
            status=ResponseStatus.ERROR,
            error="Input validation failed",
        )
        result = PrepareRunResult(success=False, error_response=error_resp)
        assert result.success is False
        assert result.context is None
        assert result.run_state is None
        assert result.error_response is not None
        assert result.error_response.status == ResponseStatus.ERROR

    def test_prepare_run_result_defaults(self):
        logger.info("PrepareRunResult: prepare run result defaults")
        result = PrepareRunResult(success=False)
        assert result.context is None
        assert result.run_state is None
        assert result.initial_message_count == 0
        assert result.tool_context_state is None
        assert result.error_response is None
