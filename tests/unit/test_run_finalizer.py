"""Unit tests for RunFinalizer (extracted post-execution logic)."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.agent.execution.run_finalizer import RunFinalizer
from orchestrator.agent.types import (
    AgentResponse,
    ResponseStatus,
    RunContext,
    RunState,
    RunStatus,
    TokenUsage,
)
import logging

logger = logging.getLogger(__name__)


@pytest.fixture
def mock_lifecycle():
    lc = AsyncMock()
    return lc


@pytest.fixture
def finalizer(mock_lifecycle):
    session_service = AsyncMock()
    context_service = AsyncMock()
    return RunFinalizer(
        session_service=session_service,
        context_service=context_service,
        lifecycle=mock_lifecycle,
        tool_executor=None,
        session_client=MagicMock(is_enabled=True),
    )


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.name = "test-agent"
    agent.model = "gpt-4"
    agent.config.log_to_session = True
    agent.tool_executor = None
    agent.on_error = None
    return agent


@pytest.fixture
def run_context():
    return RunContext(
        run_id="run_1",
        session_id="session_1",
        user_id="user_1",
        trace_id="trace_1",
    )


@pytest.fixture
def run_state():
    return RunState(
        run_id="run_1",
        session_id="session_1",
        agent_stack=["test-agent"],
        handoff_chain=[],
    )


class TestFinalize:
    @pytest.mark.asyncio
    async def test_finalize_updates_response(self, finalizer, mock_agent, run_context, run_state):
        logger.info("Finalize: finalize updates response")
        response = AgentResponse(content="Done", agent_name="test-agent", status=ResponseStatus.SUCCESS)
        start_time = time.time() - 0.5

        with patch("orchestrator.agent.execution.run_finalizer.get_metrics_collector") as mock_mc:
            mock_mc.return_value = MagicMock()
            await finalizer.finalize(mock_agent, run_context, run_state, response, 1, None, start_time)

        assert response.run_id == "run_1"
        assert response.trace_id == "trace_1"
        assert response.latency_ms > 0
        assert response.agents_used == ["test-agent"]

    @pytest.mark.asyncio
    async def test_finalize_saves_state(self, finalizer, mock_agent, run_context, run_state):
        logger.info("Finalize: finalize saves state")
        response = AgentResponse(content="Done", agent_name="test-agent", status=ResponseStatus.SUCCESS)

        with patch("orchestrator.agent.execution.run_finalizer.get_metrics_collector") as mock_mc:
            mock_mc.return_value = MagicMock()
            await finalizer.finalize(mock_agent, run_context, run_state, response, 1, None, time.time())

        assert run_state.status == RunStatus.COMPLETED
        finalizer._context_service.save_run_state.assert_awaited_once_with(run_state)

    @pytest.mark.asyncio
    async def test_finalize_records_token_metrics(self, finalizer, mock_agent, run_context, run_state):
        logger.info("Finalize: finalize records token metrics")
        response = AgentResponse(
            content="Done", agent_name="test-agent", status=ResponseStatus.SUCCESS,
            usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        )

        with patch("orchestrator.agent.execution.run_finalizer.get_metrics_collector") as mock_mc:
            metrics = MagicMock()
            mock_mc.return_value = metrics
            await finalizer.finalize(mock_agent, run_context, run_state, response, 1, None, time.time())

        metrics.track_tokens.assert_called_once()

    @pytest.mark.asyncio
    async def test_finalize_calls_lifecycle_end(self, finalizer, mock_lifecycle, mock_agent, run_context, run_state):
        logger.info("Finalize: finalize calls lifecycle end")
        response = AgentResponse(content="Done", agent_name="test-agent", status=ResponseStatus.SUCCESS)

        with patch("orchestrator.agent.execution.run_finalizer.get_metrics_collector") as mock_mc:
            mock_mc.return_value = MagicMock()
            await finalizer.finalize(mock_agent, run_context, run_state, response, 1, None, time.time())

        mock_lifecycle.end_trace.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_save_session_data(self, finalizer, mock_agent, run_context):
        logger.info("Finalize: save session data")
        messages = [{"role": "user", "content": "hi"}]
        await finalizer.save_session_data(mock_agent, run_context, 0, None, None, messages)

        finalizer._session_service.save_messages.assert_awaited_once()


class TestHandleError:
    @pytest.mark.asyncio
    async def test_handle_error_updates_state(self, finalizer, mock_agent, run_context, run_state):
        logger.info("HandleError: handle error updates state")
        error = RuntimeError("test error")

        with patch("orchestrator.agent.execution.run_finalizer.get_metrics_collector") as mock_mc:
            mock_mc.return_value = MagicMock()
            await finalizer.handle_error(mock_agent, run_context, run_state, error, time.time())

        assert run_state.status == RunStatus.FAILED
        assert run_state.metadata["error"] == "test error"

    @pytest.mark.asyncio
    async def test_handle_error_calls_on_error_hook(self, finalizer, mock_agent, run_context, run_state):
        logger.info("HandleError: handle error calls on error hook")
        error = RuntimeError("test error")
        mock_agent.on_error = MagicMock()

        with patch("orchestrator.agent.execution.run_finalizer.get_metrics_collector") as mock_mc:
            mock_mc.return_value = MagicMock()
            await finalizer.handle_error(mock_agent, run_context, run_state, error, time.time())

        mock_agent.on_error.assert_called_once_with(mock_agent, error, {"context": run_context})

    @pytest.mark.asyncio
    async def test_handle_error_reports_to_lifecycle(self, finalizer, mock_lifecycle, mock_agent, run_context, run_state):
        logger.info("HandleError: handle error reports to lifecycle")
        error = RuntimeError("test error")

        with patch("orchestrator.agent.execution.run_finalizer.get_metrics_collector") as mock_mc:
            mock_mc.return_value = MagicMock()
            await finalizer.handle_error(mock_agent, run_context, run_state, error, time.time())

        mock_lifecycle.report_error.assert_awaited_once()


class TestAttachArtifacts:
    def test_attach_artifacts_from_agent_executor(self, finalizer):
        logger.info("AttachArtifacts: attach artifacts from agent executor")
        mock_agent = MagicMock()
        artifacts = MagicMock()
        artifacts.is_empty.return_value = False
        artifacts.to_dict.return_value = {"tool_artifacts": [{"id": "1"}]}
        mock_agent.tool_executor.run_artifacts = artifacts

        response = AgentResponse(content="ok", agent_name="test", status=ResponseStatus.SUCCESS)
        finalizer.attach_run_artifacts(mock_agent, response)

        assert response.run_artifacts == {"tool_artifacts": [{"id": "1"}]}

    def test_attach_artifacts_from_global_executor(self, finalizer):
        logger.info("AttachArtifacts: attach artifacts from global executor")
        mock_agent = MagicMock()
        mock_agent.tool_executor = None

        artifacts = MagicMock()
        artifacts.is_empty.return_value = False
        artifacts.to_dict.return_value = {"tool_artifacts": [{"id": "g1"}]}
        finalizer._tool_executor = MagicMock()
        finalizer._tool_executor.run_artifacts = artifacts

        response = AgentResponse(content="ok", agent_name="test", status=ResponseStatus.SUCCESS)
        finalizer.attach_run_artifacts(mock_agent, response)

        assert response.run_artifacts == {"tool_artifacts": [{"id": "g1"}]}


class TestTrackMcpSession:
    def test_track_mcp_session(self, finalizer, run_context):
        logger.info("TrackMcpSession: track mcp session")
        mock_agent = MagicMock()
        ctx_state = MagicMock()
        ctx_state.get_all_namespaces.return_value = ["ns"]
        ctx_state.get.return_value = "mcp-sess-12345678"
        mock_agent.tool_executor.context_state = ctx_state

        mcp_id, state = finalizer.track_mcp_session(mock_agent, run_context)
        assert mcp_id == "mcp-sess-12345678"
        assert state is ctx_state
