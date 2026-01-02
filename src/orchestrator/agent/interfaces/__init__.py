"""
Agent interfaces and abstractions.

Defines clear contracts for extensibility and testability.
"""

from orchestrator.agent.interfaces.executor_interface import (
    IExecutor,
    IStreamExecutor,
)
from orchestrator.agent.interfaces.handler_interface import (
    IHandoffExecutor,
    IMessageBuilder,
    IToolHandler,
)
from orchestrator.agent.interfaces.service_interface import (
    IContextService,
    IExecutionService,
    IMemoryService,
    ISessionService,
    IToolService,
)

__all__ = [
    # Executor interfaces
    "IExecutor",
    "IStreamExecutor",
    # Service interfaces
    "IExecutionService",
    "IContextService",
    "ISessionService",
    "IMemoryService",
    "IToolService",
    # Handler interfaces
    "IMessageBuilder",
    "IToolHandler",
    "IHandoffExecutor",
]
