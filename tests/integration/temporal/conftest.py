"""
Shared fixtures for Temporal integration tests.

Uses Temporal's built-in time-skipping test environment.
No real Temporal server or real LLM calls required.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from temporalio.contrib.pydantic import pydantic_data_converter

from orchestrator.temporal.registry import AgentRegistry, get_agent_registry, reset_agent_registry
from orchestrator.temporal.types import AgentActivityResult


def _make_agent(name: str) -> MagicMock:
    agent = MagicMock()
    agent.name = name
    agent.instructions = f"You are {name}"
    return agent


def _make_mock_runner(results_map: dict[str, str]) -> MagicMock:
    """Create a mock AgentRunner that returns predefined outputs per agent name."""
    runner = MagicMock()

    async def mock_run(agent, input, **kwargs):
        content = results_map.get(agent.name, f"Default output from {agent.name}")
        resp = MagicMock()
        resp.content = content
        resp.status = MagicMock()
        resp.status.value = "success"
        resp.structured_output = None
        resp.usage = MagicMock()
        resp.usage.prompt_tokens = 10
        resp.usage.completion_tokens = 20
        resp.usage.total_tokens = 30
        resp.agents_used = [agent.name]
        resp.error = None
        return resp

    runner.run = AsyncMock(side_effect=mock_run)
    return runner


@pytest.fixture
def mock_registry():
    """Create a fresh registry with test agents and a mock runner."""
    reset_agent_registry()
    registry = get_agent_registry()
    registry.register(_make_agent("agent-a"))
    registry.register(_make_agent("agent-b"))
    registry.register(_make_agent("agent-c"))
    registry.register(_make_agent("evaluator"))

    runner = _make_mock_runner({
        "agent-a": "Result from A",
        "agent-b": "Result from B",
        "agent-c": "Result from C",
        "evaluator": "true",
    })
    registry.set_runner_factory(lambda: runner)

    yield registry
    reset_agent_registry()


@pytest.fixture
async def temporal_env():
    """Time-skipping Temporal test environment with Pydantic data converter."""
    from temporalio.testing import WorkflowEnvironment

    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter,
    ) as env:
        yield env
