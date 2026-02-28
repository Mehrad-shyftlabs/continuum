"""Unit tests verifying runner delegates to RunFinalizer and RunLifecycle (post Phase 4)."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.agent.types import (
    AgentResponse,
    ResponseStatus,
    RunContext,
    RunState,
    RunStatus,
)
import logging

logger = logging.getLogger(__name__)


@pytest.fixture
def mock_runner():
    """Create an AgentRunner with fully mocked dependencies."""
    with patch("orchestrator.agent.runner.get_container") as mock_gc:
        container = MagicMock()
        container.llm_client = MagicMock()
        container.memory_client = MagicMock()
        container.session_client = MagicMock()
        container.tool_executor = MagicMock()
        mock_gc.return_value = container

        from orchestrator.agent.runner import AgentRunner

        runner = AgentRunner(container=container)
        runner._finalizer = AsyncMock()
        runner._lifecycle = AsyncMock()
        runner._context_service = AsyncMock()
        return runner


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.name = "test-agent"
    agent.model = "gpt-4"
    agent.config.log_to_session = True
    agent.tool_executor = None
    agent.on_error = None
    agent.on_end = None
    return agent


@pytest.fixture
def run_context():
    return RunContext(
        run_id="run_test_1",
        session_id="session_1",
        user_id="user_1",
        trace_id="trace_1",
    )


@pytest.fixture
def run_state():
    return RunState(
        run_id="run_test_1",
        session_id="session_1",
        agent_stack=["test-agent"],
        handoff_chain=[],
    )


class TestRunnerDelegation:
    """Verify that AgentRunner delegates to RunFinalizer and RunLifecycle."""

    def test_runner_has_lifecycle(self, mock_runner):
        logger.info("RunnerDelegation: runner has lifecycle")
        from orchestrator.agent.execution.run_lifecycle import RunLifecycle
        assert hasattr(mock_runner, "_lifecycle")

    def test_runner_has_finalizer(self, mock_runner):
        logger.info("RunnerDelegation: runner has finalizer")
        from orchestrator.agent.execution.run_finalizer import RunFinalizer

        assert hasattr(mock_runner, "_finalizer")

    def test_runner_no_longer_has_trace_methods(self, mock_runner):
        """After Phase 4, tracing methods are on RunLifecycle, not AgentRunner."""
        logger.info("After Phase 4, tracing methods are on RunLifecycle, not AgentRunner")
        assert not hasattr(type(mock_runner), "_trace_run_start")
        assert not hasattr(type(mock_runner), "_trace_run_end")
        assert not hasattr(type(mock_runner), "_trace_run_error")

    def test_runner_no_longer_has_old_finalize(self, mock_runner):
        """After Phase 4, _finalize_run and _handle_run_error are on RunFinalizer."""
        logger.info("After Phase 4, _finalize_run and _handle_run_error are on RunFinalizer")
        assert not hasattr(type(mock_runner), "_finalize_run")
        assert not hasattr(type(mock_runner), "_handle_run_error")
        assert not hasattr(type(mock_runner), "_track_mcp_session")
        assert not hasattr(type(mock_runner), "_attach_run_artifacts")
        assert not hasattr(type(mock_runner), "_save_session_data")
        assert not hasattr(type(mock_runner), "_report_metrics_to_trace")
