"""
Agent factory functions for multi-agent architecture.

Exports:
- build_tool_catalog: Convert MCP tools to markdown catalog
- create_orchestrator_agent: Create orchestrator agent with memory
- create_executor_agent: Create executor agent with MCP tools
"""

from .executor import create_executor_agent
from .orchestrator import build_tool_catalog, create_orchestrator_agent

__all__ = [
    "build_tool_catalog",
    "create_orchestrator_agent",
    "create_executor_agent",
]
