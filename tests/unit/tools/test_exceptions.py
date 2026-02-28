"""Unit tests for tools exceptions."""

import pytest

from orchestrator.tools.exceptions import MCPToolError, ToolError
import logging

logger = logging.getLogger(__name__)


class TestToolExceptions:
    def test_tool_error(self):
        logger.info("ToolExceptions: tool error")
        err = ToolError("test")
        assert isinstance(err, Exception)

    def test_mcp_tool_error(self):
        logger.info("ToolExceptions: mcp tool error")
        err = MCPToolError("mcp error")
        assert isinstance(err, ToolError)
