"""
Petco Retail Agent Configuration.

Configuration for the Petco retail shopping agent.
"""

from dataclasses import dataclass


@dataclass
class PetcoConfig:
    """Configuration for Petco retail agent."""

    # MCP Server
    mcp_url: str = "https://mcp.agentfly.shyftops.io/mcp"
    mcp_timeout: float = 30.0
    mcp_sse_timeout: float = 300.0

    # Agent Settings
    agent_name: str = "petco-assistant"
    agent_model: str = "gemini/gemini-2.5-flash"  # Use Gemini for better performance
    agent_temperature: float = 0.7
    max_turns: int = 25

    # Memory Settings
    enable_memory: bool = True
    memory_search_limit: int = 5

    # Session Settings
    enable_session: bool = True
    session_ttl: int = 3600 * 24  # 24 hours

    # Observability
    enable_tracing: bool = True

    # Multi-Agent Settings
    orchestrator_model: str = "gemini/gemini-2.5-flash"
    orchestrator_temperature: float = 0.3  # Lower for consistent planning
    executor_model: str = "gemini/gemini-2.5-flash"
    executor_temperature: float = 0.5
    use_multi_agent: bool = True  # Toggle between single and multi-agent

    # Agent Instructions - Minimal, focused on behavior rules only
    # Tool descriptions come from MCP server, don't duplicate here
    system_instructions: str = """You are Petco's friendly shopping assistant.

## PERSONA
- Warm, knowledgeable about pet care and products
- Brief and conversational
- Use **bold** for product names and prices
- Ask for confirmation before checkout or significant actions

## CONTEXT AWARENESS
- You have access to Petco shopping tools via MCP
- Tool descriptions explain when/how to use each tool
- System will provide a session_id - use it for all cart/checkout operations
- Memory context shows this user's past preferences and pet info

## RESPONSE STYLE
- Keep responses concise (2-3 sentences for simple queries)
- Format product info clearly (name, price, key features)
- Offer relevant follow-up suggestions naturally"""


# Default configuration
default_config = PetcoConfig()
