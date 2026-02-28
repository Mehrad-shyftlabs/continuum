"""Comprehensive tests for tools/executor.py."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.llm.types import ChatMessage, FunctionCall, ToolCall
from orchestrator.tools.exceptions import MCPToolError
from orchestrator.tools.executor import (
    COMMON_CONTEXT_VARIABLES,
    SENSITIVE_CONTEXT_VARIABLES,
    RateLimiter,
    ToolExecutor,
    ToolExecutorConfig,
)
from orchestrator.tools.types import RunArtifacts, ToolContextState
import logging

logger = logging.getLogger(__name__)


class TestToolExecutorConfig:
    def test_defaults(self):
        logger.info("ToolExecutorConfig: defaults")
        c = ToolExecutorConfig()
        assert c.max_concurrent_calls == 5
        assert c.rate_limit_per_second == 10.0
        assert c.timeout_seconds == 30.0

    def test_custom(self):
        logger.info("ToolExecutorConfig: custom")
        c = ToolExecutorConfig(max_concurrent_calls=10, rate_limit_per_second=20.0, timeout_seconds=60.0)
        assert c.max_concurrent_calls == 10


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_acquire_basic(self):
        logger.info("RateLimiter: acquire basic")
        rl = RateLimiter(rate_per_second=100.0)
        await rl.acquire()
        assert rl.last_update is not None

    @pytest.mark.asyncio
    async def test_acquire_disabled(self):
        logger.info("RateLimiter: acquire disabled")
        rl = RateLimiter(rate_per_second=0)
        await rl.acquire()

    @pytest.mark.asyncio
    async def test_multiple_acquires(self):
        logger.info("RateLimiter: multiple acquires")
        rl = RateLimiter(rate_per_second=100.0)
        for _ in range(5):
            await rl.acquire()


class TestToolExecutor:
    def test_init_defaults(self):
        logger.info("ToolExecutor: init defaults")
        executor = ToolExecutor()
        assert executor.tool_registry == {}
        assert isinstance(executor.context_state, ToolContextState)
        assert isinstance(executor.run_artifacts, RunArtifacts)

    def test_init_with_config(self):
        logger.info("ToolExecutor: init with config")
        config = ToolExecutorConfig(max_concurrent_calls=3)
        executor = ToolExecutor(config=config)
        assert executor._config.max_concurrent_calls == 3

    def test_init_with_context_state(self):
        logger.info("ToolExecutor: init with context state")
        state = ToolContextState()
        state.set("ns", "key", "val")
        executor = ToolExecutor(context_state=state)
        assert executor.context_state.get("ns", "key") == "val"

    def test_context_state_setter(self):
        logger.info("ToolExecutor: context state setter")
        executor = ToolExecutor()
        new_state = ToolContextState()
        new_state.set("ns", "k", "v")
        executor.context_state = new_state
        assert executor.context_state.get("ns", "k") == "v"

    def test_clear_run_artifacts(self):
        logger.info("ToolExecutor: clear run artifacts")
        executor = ToolExecutor()
        executor.clear_run_artifacts(run_id="run-1")
        assert executor.run_artifacts.run_id == "run-1"

    def test_get_available_tools_empty(self):
        logger.info("ToolExecutor: get available tools empty")
        executor = ToolExecutor()
        assert executor.get_available_tools() == []

    def test_get_available_tools_with_registry(self):
        logger.info("ToolExecutor: get available tools with registry")
        executor = ToolExecutor()
        mock_server = MagicMock()
        mock_tool = MagicMock()
        executor.tool_registry = {"tool1": (mock_server, mock_tool)}
        assert executor.get_available_tools() == ["tool1"]

    @pytest.mark.asyncio
    async def test_initialize(self):
        logger.info("ToolExecutor: initialize")
        mock_server = AsyncMock()
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_server.list_tools.return_value = [mock_tool]

        executor = ToolExecutor(tool_registry={mock_server: None})
        await executor.initialize()
        assert "test_tool" in executor.tool_registry

    @pytest.mark.asyncio
    async def test_initialize_with_allowed_tools(self):
        logger.info("ToolExecutor: initialize with allowed tools")
        mock_server = AsyncMock()
        mock_tool1 = MagicMock()
        mock_tool1.name = "allowed_tool"
        mock_tool2 = MagicMock()
        mock_tool2.name = "blocked_tool"
        mock_server.list_tools.return_value = [mock_tool1, mock_tool2]

        executor = ToolExecutor(tool_registry={mock_server: ["allowed_tool"]})
        await executor.initialize()
        assert "allowed_tool" in executor.tool_registry
        assert "blocked_tool" not in executor.tool_registry

    @pytest.mark.asyncio
    async def test_execute_tool_call_not_found(self):
        logger.info("ToolExecutor: execute tool call not found")
        executor = ToolExecutor()
        tc = ToolCall(
            id="tc1", type="function",
            function=FunctionCall(name="nonexistent", arguments="{}"),
        )
        with pytest.raises(MCPToolError):
            await executor.execute_tool_call(tc)

    @pytest.mark.asyncio
    async def test_execute_tool_call_success(self):
        logger.info("ToolExecutor: execute tool call success")
        executor = ToolExecutor()
        mock_server = MagicMock()
        mock_server.name = "test-server"
        mock_server.context_config = MagicMock()
        mock_server.context_config.namespace = None
        mock_server.context_config.should_inject.return_value = False
        mock_server.context_config.auto_capture_common = False
        mock_server.context_config.should_capture.return_value = False

        mock_tool = MagicMock()
        mock_tool.name = "my_tool"
        mock_tool.inputSchema = {"properties": {}}

        executor.tool_registry = {"my_tool": (mock_server, mock_tool)}

        with patch("orchestrator.tools.executor.MCPUtil") as mock_util:
            mock_artifact = MagicMock()
            mock_util.invoke_mcp_tool_with_artifact = AsyncMock(
                return_value=('{"result": "ok"}', mock_artifact)
            )
            tc = ToolCall(
                id="tc1", type="function",
                function=FunctionCall(name="my_tool", arguments='{"arg1": "val1"}'),
            )
            result = await executor.execute_tool_call(tc)
            assert result.role == "tool"
            assert "ok" in result.content

    @pytest.mark.asyncio
    async def test_execute_tool_call_timeout(self):
        logger.info("ToolExecutor: execute tool call timeout")
        executor = ToolExecutor(config=ToolExecutorConfig(timeout_seconds=0.001))
        mock_server = MagicMock()
        mock_server.name = "test-server"
        mock_server.context_config = MagicMock()
        mock_server.context_config.namespace = None
        mock_server.context_config.should_inject.return_value = False
        mock_server.context_config.auto_capture_common = False
        mock_server.context_config.should_capture.return_value = False

        mock_tool = MagicMock()
        mock_tool.name = "slow_tool"
        mock_tool.inputSchema = {"properties": {}}

        executor.tool_registry = {"slow_tool": (mock_server, mock_tool)}

        with patch("orchestrator.tools.executor.MCPUtil") as mock_util:
            async def slow_call(*args, **kwargs):
                await asyncio.sleep(10)
                return ('{"r": "ok"}', MagicMock())
            mock_util.invoke_mcp_tool_with_artifact = slow_call

            tc = ToolCall(
                id="tc1", type="function",
                function=FunctionCall(name="slow_tool", arguments="{}"),
            )
            result = await executor.execute_tool_call(tc)
            assert "timed out" in result.content

    @pytest.mark.asyncio
    async def test_execute_tool_call_exception(self):
        logger.info("ToolExecutor: execute tool call exception")
        executor = ToolExecutor()
        mock_server = MagicMock()
        mock_server.name = "test-server"
        mock_server.context_config = MagicMock()
        mock_server.context_config.namespace = None
        mock_server.context_config.should_inject.return_value = False
        mock_server.context_config.auto_capture_common = False
        mock_server.context_config.should_capture.return_value = False

        mock_tool = MagicMock()
        mock_tool.name = "err_tool"
        mock_tool.inputSchema = {"properties": {}}

        executor.tool_registry = {"err_tool": (mock_server, mock_tool)}

        with patch("orchestrator.tools.executor.MCPUtil") as mock_util:
            mock_util.invoke_mcp_tool_with_artifact = AsyncMock(
                side_effect=RuntimeError("tool failed")
            )
            tc = ToolCall(
                id="tc1", type="function",
                function=FunctionCall(name="err_tool", arguments="{}"),
            )
            result = await executor.execute_tool_call(tc)
            assert "error" in result.content

    @pytest.mark.asyncio
    async def test_execute_tool_calls_multiple(self):
        logger.info("ToolExecutor: execute tool calls multiple")
        executor = ToolExecutor()
        mock_server = MagicMock()
        mock_server.name = "test"
        mock_server.context_config = MagicMock()
        mock_server.context_config.namespace = None
        mock_server.context_config.should_inject.return_value = False
        mock_server.context_config.auto_capture_common = False
        mock_server.context_config.should_capture.return_value = False

        mock_tool = MagicMock()
        mock_tool.name = "tool1"
        mock_tool.inputSchema = {"properties": {}}

        executor.tool_registry = {"tool1": (mock_server, mock_tool)}

        with patch("orchestrator.tools.executor.MCPUtil") as mock_util:
            mock_util.invoke_mcp_tool_with_artifact = AsyncMock(
                return_value=('{"r": "ok"}', MagicMock())
            )
            tcs = [
                ToolCall(id="tc1", type="function", function=FunctionCall(name="tool1", arguments="{}")),
                ToolCall(id="tc2", type="function", function=FunctionCall(name="tool1", arguments="{}")),
            ]
            results = await executor.execute_tool_calls(tcs)
            assert len(results) == 2

    @pytest.mark.asyncio
    async def test_refresh_registry(self):
        logger.info("ToolExecutor: refresh registry")
        executor = ToolExecutor()
        mock_server = AsyncMock()
        mock_tool = MagicMock()
        mock_tool.name = "new_tool"
        mock_server.list_tools.return_value = [mock_tool]

        await executor.refresh_registry({mock_server: None})
        assert "new_tool" in executor.tool_registry

    def test_capture_context_variables(self):
        logger.info("ToolExecutor: capture context variables")
        executor = ToolExecutor()
        mock_server = MagicMock()
        mock_server.name = "test-server"
        mock_server.context_config = MagicMock()
        mock_server.context_config.namespace = None
        mock_server.context_config.should_capture.return_value = True
        mock_server.context_config.get_scope.return_value = "session"
        mock_server.context_config.auto_capture_common = False

        result = json.dumps({"session_id": "s123", "status": "ok"})
        executor._capture_context_variables(mock_server, "create_session", result)
        assert executor.context_state.get("test-server", "session_id") == "s123"

    def test_capture_context_variables_invalid_json(self):
        logger.info("ToolExecutor: capture context variables invalid json")
        executor = ToolExecutor()
        mock_server = MagicMock()
        mock_server.name = "test"
        mock_server.context_config = MagicMock()
        mock_server.context_config.namespace = None
        executor._capture_context_variables(mock_server, "tool", "not json")


class TestConstants:
    def test_common_variables(self):
        logger.info("Constants: common variables")
        assert "session_id" in COMMON_CONTEXT_VARIABLES
        assert "user_id" in COMMON_CONTEXT_VARIABLES

    def test_sensitive_variables(self):
        logger.info("Constants: sensitive variables")
        assert "auth_token" in SENSITIVE_CONTEXT_VARIABLES
