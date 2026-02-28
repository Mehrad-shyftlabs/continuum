"""
Fake MCP server for integration tests that run without a live MCP server.

Implements the MCPServer interface (connect, cleanup, name, list_tools, call_tool,
list_prompts, get_prompt) so ToolExecutor and MCPUtil can be tested end-to-end.
"""

from __future__ import annotations

from mcp.types import (
    CallToolResult,
    GetPromptResult,
    ListPromptsResult,
    TextContent,
    Tool,
)

from orchestrator.tools.types import ToolContextConfig


class FakeMCPServer:
    """Fake MCP server for integration tests. No network; all in-memory."""

    def __init__(
        self,
        name: str = "fake-mcp",
        tools: list[Tool] | None = None,
        context_config: ToolContextConfig | None = None,
    ):
        self._name = name
        self.context_config = context_config or ToolContextConfig()
        self._tools = tools or [
            Tool(
                name="echo",
                description="Echo back the input",
                inputSchema={
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                },
            ),
            Tool(
                name="add",
                description="Add two numbers",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                },
            ),
        ]
        self._connected = False

    @property
    def name(self) -> str:
        return self._name

    async def connect(self) -> None:
        self._connected = True

    async def cleanup(self) -> None:
        self._connected = False

    async def list_tools(
        self,
        metadata: dict | None = None,
    ) -> list[Tool]:
        if not self._connected:
            raise RuntimeError("FakeMCPServer not connected")
        return self._tools

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict | None,
    ) -> CallToolResult:
        if not self._connected:
            raise RuntimeError("FakeMCPServer not connected")
        arguments = arguments or {}
        if tool_name == "echo":
            msg = arguments.get("message", "")
            return CallToolResult(
                content=[TextContent(type="text", text=f"Echo: {msg}")],
                isError=False,
            )
        if tool_name == "add":
            a = arguments.get("a", 0)
            b = arguments.get("b", 0)
            return CallToolResult(
                content=[TextContent(type="text", text=str(float(a) + float(b)))],
                isError=False,
            )
        return CallToolResult(
            content=[TextContent(type="text", text=f"Unknown tool: {tool_name}")],
            isError=True,
        )

    async def list_prompts(self) -> ListPromptsResult:
        return ListPromptsResult(prompts=[])

    async def get_prompt(
        self,
        name: str,
        arguments: dict | None = None,
    ) -> GetPromptResult:
        return GetPromptResult(messages=[])
