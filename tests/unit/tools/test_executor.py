"""Unit tests for tool executor."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.tools.executor import ToolExecutor, ToolExecutorConfig
from orchestrator.tools.types import MCPToolArtifact, RunArtifacts, ToolContextState
import logging

logger = logging.getLogger(__name__)


class TestToolExecutorConfig:
    def test_defaults(self):
        logger.info("ToolExecutorConfig: defaults")
        c = ToolExecutorConfig()
        assert c.max_concurrent_calls > 0
        assert c.timeout_seconds > 0


class TestToolExecutor:
    def test_initialization(self):
        logger.info("ToolExecutor: initialization")
        executor = ToolExecutor()
        assert executor is not None

    def test_run_artifacts(self):
        logger.info("ToolExecutor: run artifacts")
        executor = ToolExecutor()
        assert executor.run_artifacts is not None
        assert isinstance(executor.run_artifacts, RunArtifacts)

    def test_run_artifacts_clear(self):
        logger.info("ToolExecutor: run artifacts clear")
        executor = ToolExecutor()
        executor.run_artifacts.add_artifact(MCPToolArtifact(tool_name="t", server_name="s"))
        assert not executor.run_artifacts.is_empty()
        executor.run_artifacts.clear()
        assert executor.run_artifacts.is_empty()

    def test_context_state(self):
        logger.info("ToolExecutor: context state")
        executor = ToolExecutor()
        assert executor.context_state is not None
        assert isinstance(executor.context_state, ToolContextState)

    def test_tool_registry(self):
        logger.info("ToolExecutor: tool registry")
        mock_server = MagicMock()
        mock_server.name = "test-server"
        executor = ToolExecutor(tool_registry={mock_server: None})
        assert executor is not None
