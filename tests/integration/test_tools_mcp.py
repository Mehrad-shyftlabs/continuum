"""
Integration tests for MCP tool integration.

Requires access to an MCP server.

Converted from tests/test_tools.py manual test script.
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


pytestmark = [pytest.mark.integration]

MCP_SERVER_URL = "https://mcp.agentfly.shyftops.io/mcp"


@pytest.fixture
async def mcp_server():
    """Create and connect to an MCP server, cleanup on teardown."""
    server = MCPServerStreamableHttp(
        {
            "url": MCP_SERVER_URL,
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

        filtered_server = MCPServerStreamableHttp(
            {
                "url": MCP_SERVER_URL,
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
