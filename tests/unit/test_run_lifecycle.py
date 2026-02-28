"""Unit tests for RunLifecycle (extracted tracing logic)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.agent.execution.run_lifecycle import RunLifecycle
from orchestrator.agent.types import (
    AgentResponse,
    ResponseStatus,
    RunContext,
    RunState,
)
import logging

logger = logging.getLogger(__name__)


@pytest.fixture
def lifecycle():
    return RunLifecycle()


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.name = "test-agent"
    agent.model = "gpt-4"
    agent.tags = ["test"]
    agent.tools = []
    return agent


@pytest.fixture
def run_context():
    return RunContext(
        run_id="run_1",
        session_id="session_1",
        user_id="user_1",
        trace_id="",
        tags=["tag1"],
    )


@pytest.fixture
def run_state():
    return RunState(
        run_id="run_1",
        session_id="session_1",
        agent_stack=["test-agent"],
        handoff_chain=[],
    )


class TestStartTrace:
    @pytest.mark.asyncio
    async def test_start_trace_creates_new_trace(self, lifecycle, mock_agent, run_context, run_state):
        logger.info("StartTrace: start trace creates new trace")
        with (
            patch("orchestrator.agent.execution.run_lifecycle.get_current_trace_id", return_value=None),
            patch("orchestrator.agent.execution.run_lifecycle.set_trace_context") as mock_set,
        ):
            mock_trace = MagicMock()
            mock_trace.id = "new-trace-id"
            mock_trace.langfuse_trace = MagicMock()

            with patch("orchestrator.observability.TracingManager") as MockTM:
                MockTM.return_value.create_trace.return_value = mock_trace
                await lifecycle.start_trace(mock_agent, run_context, run_state, "hello")

            assert run_context.trace_id == "new-trace-id"
            mock_set.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_trace_uses_existing_trace(self, lifecycle, mock_agent, run_context, run_state):
        logger.info("StartTrace: start trace uses existing trace")
        with (
            patch("orchestrator.agent.execution.run_lifecycle.get_current_trace_id", return_value="existing-trace"),
            patch("orchestrator.agent.execution.run_lifecycle.get_current_trace_client", return_value=MagicMock()),
            patch("orchestrator.agent.execution.run_lifecycle.set_trace_context") as mock_set,
        ):
            await lifecycle.start_trace(mock_agent, run_context, run_state, "hello")

        assert run_context.trace_id == "existing-trace"
        mock_set.assert_called_once()

    @pytest.mark.asyncio
    async def test_end_trace_clears_context(self, lifecycle, mock_agent, run_context):
        logger.info("StartTrace: end trace clears context")
        response = AgentResponse(content="done", agent_name="test-agent", status=ResponseStatus.SUCCESS)
        run_context._langfuse_trace = MagicMock()

        with patch("orchestrator.agent.execution.run_lifecycle.clear_trace_context") as mock_clear:
            await lifecycle.end_trace(mock_agent, run_context, response)

        mock_clear.assert_called_once()

    @pytest.mark.asyncio
    async def test_report_error_sends_to_reporter(self, lifecycle, mock_agent, run_context, run_state):
        logger.info("StartTrace: report error sends to reporter")
        error = RuntimeError("boom")

        with (
            patch("orchestrator.agent.execution.run_lifecycle.report_error") as mock_report,
            patch("orchestrator.agent.execution.run_lifecycle.clear_trace_context"),
        ):
            await lifecycle.report_error(mock_agent, run_context, error, run_state)

        mock_report.assert_called_once()
        call_args = mock_report.call_args
        assert call_args[0][0] is error
