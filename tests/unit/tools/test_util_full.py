"""Comprehensive tests for tools/util.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.tools.util import MCPUtil
import logging

logger = logging.getLogger(__name__)


class TestMCPUtilGetAllTools:
    @pytest.mark.asyncio
    async def test_get_all_function_tools_empty(self):
        logger.info("MCPUtilGetAllTools: get all function tools empty")
        result = await MCPUtil.get_all_function_tools([])
        assert result == []

    @pytest.mark.asyncio
    async def test_get_all_function_tools_with_server(self):
        logger.info("MCPUtilGetAllTools: get all function tools with server")
        mock_server = MagicMock()
        mock_server.name = "test-server"

        mock_tool = MagicMock()
        mock_tool.name = "my_tool"
        mock_tool.description = "A test tool"
        mock_tool.inputSchema = {"type": "object", "properties": {}}

        mock_server.list_tools = AsyncMock(return_value=[mock_tool])

        result = await MCPUtil.get_all_function_tools([mock_server])
        assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_get_all_function_tools_server_error(self):
        logger.info("MCPUtilGetAllTools: get all function tools server error")
        mock_server = MagicMock()
        mock_server.name = "broken"
        mock_server.list_tools = AsyncMock(side_effect=Exception("connection failed"))

        with pytest.raises(Exception):
            await MCPUtil.get_all_function_tools([mock_server])

    @pytest.mark.asyncio
    async def test_get_all_function_tools_multiple_servers(self):
        logger.info("MCPUtilGetAllTools: get all function tools multiple servers")
        mock_tool1 = MagicMock()
        mock_tool1.name = "tool1"
        mock_tool1.description = "Tool 1"
        mock_tool1.inputSchema = {"type": "object", "properties": {}}

        mock_tool2 = MagicMock()
        mock_tool2.name = "tool2"
        mock_tool2.description = "Tool 2"
        mock_tool2.inputSchema = {"type": "object", "properties": {}}

        server1 = MagicMock()
        server1.name = "server1"
        server1.list_tools = AsyncMock(return_value=[mock_tool1])

        server2 = MagicMock()
        server2.name = "server2"
        server2.list_tools = AsyncMock(return_value=[mock_tool2])

        result = await MCPUtil.get_all_function_tools([server1, server2])
        assert len(result) >= 2


class TestMCPUtilInvoke:
    @pytest.mark.asyncio
    @patch("orchestrator.observability.provider_manager.get_provider_manager")
    async def test_invoke_mcp_tool(self, mock_get_pm):
        logger.info("MCPUtilInvoke: invoke mcp tool")
        mock_pm = MagicMock()
        mock_pm.is_enabled = False
        mock_get_pm.return_value = mock_pm

        mock_server = MagicMock()
        mock_server.name = "test"

        mock_tool = MagicMock()
        mock_tool.name = "tool1"
        mock_tool.inputSchema = {"properties": {}}

        mock_content = MagicMock()
        mock_content.text = '{"result": "ok"}'
        mock_content.type = "text"
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_response.meta = None
        mock_response.isError = False
        mock_response.model_dump_json.return_value = '{"content": [{"text": "ok"}]}'
        mock_server.call_tool = AsyncMock(return_value=mock_response)

        result = await MCPUtil.invoke_mcp_tool(
            mock_server, mock_tool, '{"key": "val"}'
        )
        assert result is not None
