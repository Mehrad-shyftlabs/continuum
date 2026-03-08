"""
Fetch Agent Configuration.
"""

from dataclasses import dataclass


@dataclass
class FetchAgentConfig:
    """Configuration for the Fetch agent."""

    # MCP Server - uses uvx to run mcp-server-fetch locally
    mcp_command: str = "uvx"
    mcp_args: list = None

    # Agent Settings
    agent_name: str = "fetch-assistant"
    agent_model: str = "gemini/gemini-2.5-flash"
    agent_temperature: float = 0.7
    max_turns: int = 10

    # Memory & Session
    enable_memory: bool = True
    enable_session: bool = True

    system_instructions: str = """You are a helpful web assistant that can fetch and read web pages.

When the user asks about a URL or web page, use the fetch tool to retrieve its content.
Summarize and answer questions based on what you find.
Be concise and helpful."""

    def __post_init__(self):
        if self.mcp_args is None:
            self.mcp_args = ["mcp-server-fetch"]


default_config = FetchAgentConfig()
