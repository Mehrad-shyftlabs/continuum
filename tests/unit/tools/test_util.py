"""Unit tests for tools util (MCPUtil)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.tools.util import MCPUtil
import logging

logger = logging.getLogger(__name__)


class TestMCPUtil:
    @pytest.mark.asyncio
    async def test_get_all_function_tools(self):
        logger.info("MCPUtil: get all function tools")
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.inputSchema = {"type": "object", "properties": {}}
        mock_server = MagicMock()
        mock_server.name = "test"
        mock_server.list_tools = AsyncMock(return_value=[mock_tool])
        tools = await MCPUtil.get_all_function_tools([mock_server])
        assert len(tools) >= 1

    @pytest.mark.asyncio
    async def test_get_all_function_tools_empty(self):
        logger.info("MCPUtil: get all function tools empty")
        mock_server = MagicMock()
        mock_server.name = "test"
        mock_server.list_tools = AsyncMock(return_value=[])
        tools = await MCPUtil.get_all_function_tools([mock_server])
        assert tools == []
