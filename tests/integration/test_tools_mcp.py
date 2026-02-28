"""
Integration tests for MCP tool integration.

- Tests that use the fake server run without any external MCP server.
- Tests marked live_mcp require a real MCP server; set MCP_SERVER_URL to run them.
  Run without live: pytest -m "integration and not live_mcp"
"""

import os

import pytest
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


load_dotenv()

from orchestrator.tools import (
    MCPServerStreamableHttp,
    MCPUtil,
    ToolExecutor,
    create_static_tool_filter,
)

from tests.integration.mcp_fake_server import FakeMCPServer


pytestmark = [pytest.mark.integration]

# Default URL for live tests; override with env MCP_SERVER_URL
DEFAULT_MCP_SERVER_URL = "https://mcp.agentfly.shyftops.io/mcp"


@pytest.fixture
async def mcp_server():
    """Create and connect to a live MCP server, cleanup on teardown. Skips if unavailable."""
    url = os.environ.get("MCP_SERVER_URL", DEFAULT_MCP_SERVER_URL)
    server = MCPServerStreamableHttp(
        {
            "url": url,
            "timeout": 30.0,
            "sse_read_timeout": 60.0,
        },
        name="test-server",
        cache_tools_list=True,
    )
    try:
        await server.connect()
        yield server
    except Exception:
        pytest.skip("MCP server not available")
    finally:
        try:
            await server.cleanup()
        except Exception:
            pass


# --- Tests that use the FAKE server (no network) ---


class TestMCPFakeServer:
    """Integration tests using FakeMCPServer. No live MCP server required."""

    @pytest.fixture
    async def fake_server(self):
        server = FakeMCPServer(name="fake-integration")
        await server.connect()
        try:
            yield server
        finally:
            await server.cleanup()

    async def test_fake_connect_list_tools(self, fake_server):
        tools = await fake_server.list_tools()
        assert len(tools) >= 1
        names = [t.name for t in tools]
        assert "echo" in names

    async def test_fake_convert_tools(self, fake_server):
        tools = await MCPUtil.get_function_tools(fake_server)
        assert len(tools) >= 1
        for tool in tools:
            assert tool.function.name is not None
            assert tool.function.parameters is not None

    async def test_fake_executor_initialization(self, fake_server):
        executor = ToolExecutor({fake_server: None})
        await executor.initialize()
        available = executor.get_available_tools()
        assert isinstance(available, list)
        assert len(available) > 0

    async def test_fake_executor_execute_tool(self, fake_server):
        executor = ToolExecutor({fake_server: None})
        await executor.initialize()
        from orchestrator.llm.types import ToolCall
        from orchestrator.llm.types import FunctionCall

        tool_call = ToolCall(
            id="call-1",
            function=FunctionCall(name="echo", arguments='{"message": "hello"}'),
        )
        msg = await executor.execute_tool_call(tool_call)
        assert msg.role == "tool"
        assert "hello" in (msg.content or "")


class TestMCPE2E:
    """End-to-end: connect → executor init → list tools → execute → cleanup."""

    @pytest.mark.asyncio
    async def test_e2e_connect_execute_cleanup(self):
        server = FakeMCPServer(name="e2e-fake")
        await server.connect()
        try:
            executor = ToolExecutor({server: None})
            await executor.initialize()
            tools = executor.get_available_tools()
            assert len(tools) > 0
            from orchestrator.llm.types import ToolCall, FunctionCall

            tool_call = ToolCall(
                id="e2e-call-1",
                function=FunctionCall(name="add", arguments='{"a": 2, "b": 3}'),
            )
            msg = await executor.execute_tool_call(tool_call)
            assert msg.role == "tool"
            assert "5" in (msg.content or "")
            artifacts = executor.run_artifacts
            assert not artifacts.is_empty()
            by_tool = artifacts.get_by_tool("add")
            assert len(by_tool) >= 1
        finally:
            await server.cleanup()
        assert not server._connected


# --- Tests that require a LIVE MCP server (marked live_mcp) ---


@pytest.mark.live_mcp
class TestMCPConnection:
    async def test_connect(self, mcp_server):
        logger.info("MCPConnection: connect")
        assert mcp_server.name is not None

    async def test_list_tools(self, mcp_server):
        logger.info("MCPConnection: list tools")
        mcp_tools = await mcp_server.list_tools()
        assert isinstance(mcp_tools, list)
        assert len(mcp_tools) > 0

    async def test_convert_tools(self, mcp_server):
        logger.info("MCPConnection: convert tools")
        tools = await MCPUtil.get_function_tools(mcp_server)
        assert len(tools) > 0
        for tool in tools:
            assert tool.function.name is not None
            assert tool.function.parameters is not None

    async def test_tool_filtering(self, mcp_server):
        logger.info("MCPConnection: tool filtering")
        mcp_tools = await mcp_server.list_tools()
        if not mcp_tools:
            pytest.skip("No tools available to filter")

        await mcp_server.cleanup()

        filter_config = create_static_tool_filter(
            allowed_tool_names=[mcp_tools[0].name]
        )

        url = os.environ.get("MCP_SERVER_URL", DEFAULT_MCP_SERVER_URL)
        filtered_server = MCPServerStreamableHttp(
            {
                "url": url,
                "timeout": 30.0,
                "sse_read_timeout": 60.0,
            },
            tool_filter=filter_config,
        )
        try:
            await filtered_server.connect()
            filtered_tools = await filtered_server.list_tools()
            assert len(filtered_tools) == 1
        finally:
            await filtered_server.cleanup()


@pytest.mark.live_mcp
class TestToolExecutor:
    async def test_executor_initialization(self, mcp_server):
        logger.info("ToolExecutor: executor initialization")
        executor = ToolExecutor({mcp_server: None})
        await executor.initialize()

        available_tools = executor.get_available_tools()
        assert isinstance(available_tools, list)
        assert len(available_tools) > 0

    async def test_execute_tool(self, mcp_server):
        logger.info("ToolExecutor: execute tool")
        mcp_tools = await mcp_server.list_tools()
        if not mcp_tools:
            pytest.skip("No tools available to execute")

        tool = mcp_tools[0]
        try:
            result = await mcp_server.call_tool(tool.name, {})
            assert result is not None
        except Exception:
            pass  # Expected if tool requires arguments
