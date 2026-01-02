"""
Configuration for Memory Modes Demo.

Demonstrates the new provider-based memory architecture with:
- MemoryScope for scope management
- Mem0Provider for memory operations
- Custom prompts support
"""

from dataclasses import dataclass, field


@dataclass
class MemoryModesConfig:
    """Configuration for memory modes demo."""

    # Agent Settings
    agent_name: str = "memory-demo-agent"
    agent_model: str = "gemini/gemini-2.5-flash"
    agent_temperature: float = 0.7
    max_turns: int = 25

    # Memory Settings
    enable_memory: bool = True
    memory_search_limit: int = 5

    # Session Settings
    enable_session: bool = True
    session_ttl: int = 3600 * 24  # 24 hours

    # Demo Users for testing isolation
    demo_users: list[str] = field(default_factory=lambda: ["alice", "bob", "charlie"])

    # Demo Agents for testing isolation
    demo_agents: list[str] = field(default_factory=lambda: ["assistant", "helper", "expert"])

    # System Instructions
    system_instructions: str = """You are a helpful assistant demonstrating memory capabilities.

Your role is to:
1. Remember facts and preferences users tell you
2. Recall information from previous conversations
3. Demonstrate how memory works across different isolation modes

When users tell you information about themselves:
- Remember their preferences, facts, and context
- Use this information naturally in future conversations
- Acknowledge when you remember something from a previous conversation

Be friendly, helpful, and conversational. Show that you have memory!"""


# Default configuration
default_config = MemoryModesConfig()
