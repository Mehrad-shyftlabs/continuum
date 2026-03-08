"""
Workflow agents for multi-agent orchestration.

Provides specialized agents for workflow patterns:
- RouterAgent: Dynamic routing/triage
- SequentialAgent: Pipeline execution
- ParallelAgent: Concurrent execution
- LoopAgent: Iterative execution
"""

from orchestrator.agent.workflow.loop import LoopAgent
from orchestrator.agent.workflow.parallel import ParallelAgent
from orchestrator.agent.workflow.reflection import ReflectionAgent, create_reflection_agent, generate_critique_prompt
from orchestrator.agent.workflow.router import RouterAgent
from orchestrator.agent.workflow.sequential import SequentialAgent

__all__ = [
    "RouterAgent",
    "SequentialAgent",
    "ParallelAgent",
    "LoopAgent",
    "ReflectionAgent",
    "create_reflection_agent",
    "generate_critique_prompt",
]
