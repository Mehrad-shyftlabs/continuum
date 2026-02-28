"""Extended tests for tool executor."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.tools.executor import ToolExecutor, ToolExecutorConfig, RateLimiter
from orchestrator.tools.types import ToolContextState
import logging

logger = logging.getLogger(__name__)


class TestToolExecutorConfigExtended:
    def test_custom_config(self):
        logger.info("ToolExecutorConfigExtended: custom config")
        c = ToolExecutorConfig(max_concurrent_calls=10, timeout_seconds=60.0, rate_limit_per_second=20.0)
        assert c.max_concurrent_calls == 10
        assert c.timeout_seconds == 60.0


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_acquire(self):
        logger.info("RateLimiter: acquire")
        rl = RateLimiter(rate_per_second=100.0)
        await rl.acquire()

    @pytest.mark.asyncio
    async def test_multiple_acquires(self):
        logger.info("RateLimiter: multiple acquires")
        rl = RateLimiter(rate_per_second=100.0)
        for _ in range(5):
            await rl.acquire()


class TestToolExecutorExtended:
    def test_with_config(self):
        logger.info("ToolExecutorExtended: with config")
        config = ToolExecutorConfig(max_concurrent_calls=3, timeout_seconds=15.0)
        executor = ToolExecutor(config=config)
        assert executor is not None

    def test_with_tool_registry(self):
        logger.info("ToolExecutorExtended: with tool registry")
        mock_server = MagicMock()
        mock_server.name = "test"
        executor = ToolExecutor(tool_registry={mock_server: ["tool1", "tool2"]})
        assert executor is not None

    def test_with_context_state(self):
        logger.info("ToolExecutorExtended: with context state")
        state = ToolContextState()
        state.set("ns", "k", "v")
        executor = ToolExecutor(context_state=state)
        assert executor.context_state.get("ns", "k") == "v"

    def test_get_available_tools_empty(self):
        logger.info("ToolExecutorExtended: get available tools empty")
        executor = ToolExecutor()
        tools = executor.get_available_tools()
        assert isinstance(tools, list)
        assert tools == []
