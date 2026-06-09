"""
Persistence module for agent state.

Provides state management using Redis for hot storage.
"""

from continuum.agent.persistence.state import (
    RunStateManager,
    get_global_state_manager,
    initialize_global_state_manager,
)

__all__ = [
    "RunStateManager",
    "get_global_state_manager",
    "initialize_global_state_manager",
]
