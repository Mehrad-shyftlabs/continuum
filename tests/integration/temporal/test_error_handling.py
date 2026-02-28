"""Integration tests for error handling in Temporal workflows."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from temporalio.worker import Worker

from orchestrator.temporal.activities import run_agent_activity, send_notification_activity
from orchestrator.temporal.registry import get_agent_registry, reset_agent_registry
from orchestrator.temporal.types import (
    AgentStep,
    WorkflowInput,
    WorkflowResult,
)
from orchestrator.temporal.workflows.agent_workflow import AgentWorkflow
import logging

logger = logging.getLogger(__name__)


TASK_QUEUE = "test-errors"


@pytest.fixture
def error_registry():
    """Registry where one agent always errors."""
    reset_agent_registry()
    registry = get_agent_registry()

    good_agent = MagicMock()
    good_agent.name = "good-agent"

    error_agent = MagicMock()
    error_agent.name = "error-agent"

    async def mock_run(agent, input, **kwargs):
        if agent.name == "error-agent":
            raise RuntimeError("Agent exploded!")
        resp = MagicMock()
        resp.content = "Good result"
        resp.status = MagicMock()
        resp.status.value = "success"
        resp.structured_output = None
        resp.usage = MagicMock()
        resp.usage.prompt_tokens = 5
        resp.usage.completion_tokens = 10
        resp.usage.total_tokens = 15
        resp.agents_used = [agent.name]
        resp.error = None
        return resp

    runner = MagicMock()
    runner.run = AsyncMock(side_effect=mock_run)

    registry.register(good_agent)
    registry.register(error_agent)
    registry.set_runner_factory(lambda: runner)

    yield registry
    reset_agent_registry()


@pytest.mark.asyncio
async def test_agent_error_captured_in_result(temporal_env, error_registry):
    """Agent failure is captured in step results (not workflow crash)."""
    logger.info("Agent failure is captured in step results (not workflow crash)")
    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[AgentWorkflow],
        activities=[run_agent_activity, send_notification_activity],
    ):
        handle = await temporal_env.client.start_workflow(
            AgentWorkflow.run,
            WorkflowInput(
                initial_input="Test error",
                steps=[AgentStep(agent_name="error-agent", retries=1).model_dump()],
            ),
            id="test-error-1",
            task_queue=TASK_QUEUE,
        )
        result = await handle.result()
        assert result.status == "completed"
        assert len(result.step_results) == 1
        assert result.step_results[0].status == "error"
        assert "exploded" in result.step_results[0].error


@pytest.mark.asyncio
async def test_good_after_error(temporal_env, error_registry):
    """Good agent runs after error agent."""
    logger.info("Good agent runs after error agent")
    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[AgentWorkflow],
        activities=[run_agent_activity, send_notification_activity],
    ):
        handle = await temporal_env.client.start_workflow(
            AgentWorkflow.run,
            WorkflowInput(
                initial_input="Test",
                steps=[
                    AgentStep(agent_name="error-agent", retries=1).model_dump(),
                    AgentStep(agent_name="good-agent").model_dump(),
                ],
            ),
            id="test-good-after-error-1",
            task_queue=TASK_QUEUE,
        )
        result = await handle.result()
        assert result.status == "completed"
        assert len(result.step_results) == 2
        assert result.step_results[0].status == "error"
        assert result.step_results[1].status == "success"


@pytest.mark.asyncio
async def test_cancel_signal(temporal_env, mock_registry):
    """Cancel signal stops workflow execution."""
    logger.info("Cancel signal stops workflow execution")
    import asyncio

    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[AgentWorkflow],
        activities=[run_agent_activity, send_notification_activity],
    ):
        handle = await temporal_env.client.start_workflow(
            AgentWorkflow.run,
            WorkflowInput(
                initial_input="Test",
                steps=[
                    AgentStep(agent_name="agent-a").model_dump(),
                    # Wait step gives us time to cancel
                    {"type": "wait", "duration_seconds": 60},
                    AgentStep(agent_name="agent-b").model_dump(),
                ],
            ),
            id="test-cancel-1",
            task_queue=TASK_QUEUE,
        )

        await asyncio.sleep(0.5)
        await handle.signal(AgentWorkflow.cancel_workflow)

        result = await handle.result()
        assert result.status == "cancelled"
