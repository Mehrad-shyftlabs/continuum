"""Unit tests for MCP server connection module (orchestrator.tools.mcp)."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.tools.exceptions import MCPConnectionError, MCPError
from orchestrator.tools.mcp import (
    MCPServerStreamableHttp,
    MCPServerSse,
    MCPServerStdio,
)

import logging

logger = logging.getLogger(__name__)


def _make_mock_tool(name: str = "echo", description: str = "Echo tool"):
    from mcp.types import Tool

    return Tool(name=name, description=description, inputSchema={"type": "object"})


def _make_mock_list_tools_result(tools=None):
    from mcp.types import ListToolsResult

    return ListToolsResult(tools=tools or [_make_mock_tool()])


def _make_mock_call_tool_result(text: str = "ok"):
    from mcp.types import CallToolResult, TextContent

    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        isError=False,
    )


@asynccontextmanager
async def _fake_streams():
    """Fake transport yielding (read, write, None) for streamable HTTP / SSE."""
    read = MagicMock()
    write = MagicMock()
    yield (read, write, None)


@asynccontextmanager
async def _fake_streams_stdio():
    """Fake transport for stdio: (read, write) only."""
    read = MagicMock()
    write = MagicMock()
    yield (read, write, None)


class TestMCPServerStreamableHttpConnect:
    """Test MCPServerStreamableHttp connect, list_tools, call_tool, cleanup."""

    @pytest.mark.asyncio
    @patch("orchestrator.tools.mcp.ClientSession")
    @patch("orchestrator.tools.mcp.streamablehttp_client")
    async def test_connect_list_tools_call_tool_cleanup(
        self, mock_streamablehttp, mock_client_session_cls
    ):
        mock_streamablehttp.return_value = _fake_streams()

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock(return_value=MagicMock())
        mock_session.list_tools = AsyncMock(return_value=_make_mock_list_tools_result())
        mock_session.call_tool = AsyncMock(return_value=_make_mock_call_tool_result())

        async def enter_session(*args, **kwargs):
            yield mock_session

        mock_client_session_cls.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        mock_client_session_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        server = MCPServerStreamableHttp(
            {"url": "https://test.example.com/mcp"},
            name="test-http",
        )
        await server.connect()
        assert server.session is mock_session
        assert server.name == "test-http"

        tools = await server.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "echo"

        result = await server.call_tool("echo", {"x": 1})
        assert result is not None
        assert result.content[0].text == "ok"

        await server.cleanup()
        assert server.session is None

    @pytest.mark.asyncio
    @patch("orchestrator.tools.mcp.ClientSession")
    @patch("orchestrator.tools.mcp.streamablehttp_client")
    async def test_connect_failure_raises_mcp_connection_error(
        self, mock_streamablehttp, mock_client_session_cls
    ):
        mock_streamablehttp.return_value = _fake_streams()
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock(side_effect=ConnectionError("refused"))

        mock_client_session_cls.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        mock_client_session_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        server = MCPServerStreamableHttp(
            {"url": "https://test.example.com/mcp"},
            name="test-http",
        )
        with pytest.raises(MCPConnectionError) as exc_info:
            await server.connect()
        assert "test-http" in str(exc_info.value)
        assert server.session is None

    @pytest.mark.asyncio
    @patch("orchestrator.tools.mcp.ClientSession")
    @patch("orchestrator.tools.mcp.streamablehttp_client")
    async def test_list_tools_without_connect_raises(
        self, mock_streamablehttp, mock_client_session_cls
    ):
        server = MCPServerStreamableHttp(
            {"url": "https://test.example.com/mcp"},
            name="test-http",
        )
        with pytest.raises(MCPError) as exc_info:
            await server.list_tools()
        assert "connect" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    @patch("orchestrator.tools.mcp.ClientSession")
    @patch("orchestrator.tools.mcp.streamablehttp_client")
    async def test_validate_on_connect_calls_list_tools(
        self, mock_streamablehttp, mock_client_session_cls
    ):
        mock_streamablehttp.return_value = _fake_streams()
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock(return_value=MagicMock())
        list_tools_result = _make_mock_list_tools_result()
        mock_session.list_tools = AsyncMock(return_value=list_tools_result)

        mock_client_session_cls.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        mock_client_session_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        server = MCPServerStreamableHttp(
            {"url": "https://test.example.com/mcp"},
            name="test-http",
            validate_on_connect=True,
        )
        await server.connect()
        assert mock_session.list_tools.await_count == 1
        await server.cleanup()


class TestMCPServerSseConnect:
    """Test MCPServerSse with mocked transport."""

    @pytest.mark.asyncio
    @patch("orchestrator.tools.mcp.ClientSession")
    @patch("orchestrator.tools.mcp.sse_client")
    async def test_connect_and_list_tools(self, mock_sse_client, mock_client_session_cls):
        mock_sse_client.return_value = _fake_streams()

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock(return_value=MagicMock())
        mock_session.list_tools = AsyncMock(return_value=_make_mock_list_tools_result())

        mock_client_session_cls.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        mock_client_session_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        server = MCPServerSse(
            {"url": "https://sse.example.com/mcp"},
            name="test-sse",
        )
        await server.connect()
        tools = await server.list_tools()
        assert len(tools) == 1
        await server.cleanup()
        assert server.session is None


class TestMCPServerStdioConnect:
    """Test MCPServerStdio with mocked stdio_client."""

    @pytest.mark.asyncio
    @patch("orchestrator.tools.mcp.ClientSession")
    @patch("orchestrator.tools.mcp.stdio_client")
    async def test_connect_and_list_tools(self, mock_stdio_client, mock_client_session_cls):
        mock_stdio_client.return_value = _fake_streams_stdio()

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock(return_value=MagicMock())
        mock_session.list_tools = AsyncMock(return_value=_make_mock_list_tools_result())

        mock_client_session_cls.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        mock_client_session_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        server = MCPServerStdio(
            {"command": "python", "args": ["-m", "fake_mcp"]},
            name="test-stdio",
        )
        await server.connect()
        tools = await server.list_tools()
        assert len(tools) == 1
        await server.cleanup()
        assert server.session is None


class TestMCPRetryAndCleanup:
    """Test retry behavior and cleanup on cancel."""

    @pytest.mark.asyncio
    @patch("orchestrator.tools.mcp.ClientSession")
    @patch("orchestrator.tools.mcp.streamablehttp_client")
    async def test_retry_list_tools_then_succeed(
        self, mock_streamablehttp, mock_client_session_cls
    ):
        mock_streamablehttp.return_value = _fake_streams()
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock(return_value=MagicMock())
        list_tools_result = _make_mock_list_tools_result()
        mock_session.list_tools = AsyncMock(
            side_effect=[TimeoutError("first attempt"), list_tools_result]
        )

        mock_client_session_cls.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        mock_client_session_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        server = MCPServerStreamableHttp(
            {"url": "https://test.example.com/mcp"},
            name="test-http",
            max_retry_attempts=1,
            retry_backoff_seconds_base=0.01,
        )
        await server.connect()
        tools = await server.list_tools()
        assert len(tools) == 1
        assert mock_session.list_tools.await_count == 2
        await server.cleanup()

    @pytest.mark.asyncio
    @patch("orchestrator.tools.mcp.ClientSession")
    @patch("orchestrator.tools.mcp.streamablehttp_client")
    async def test_cleanup_idempotent(self, mock_streamablehttp, mock_client_session_cls):
        mock_streamablehttp.return_value = _fake_streams()
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock(return_value=MagicMock())

        mock_client_session_cls.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        mock_client_session_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        server = MCPServerStreamableHttp(
            {"url": "https://test.example.com/mcp"},
            name="test-http",
        )
        await server.connect()
        await server.cleanup()
        await server.cleanup()
        assert server.session is None
