"""Integration tests for LoopAgentWorkflow."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from temporalio.worker import Worker

from orchestrator.temporal.activities import run_agent_activity, send_notification_activity
from orchestrator.temporal.registry import get_agent_registry, reset_agent_registry
from orchestrator.temporal.types import WorkflowResult
from orchestrator.temporal.workflows.loop_workflow import (
    LoopAgentWorkflow,
    LoopWorkflowInput,
)
import logging

logger = logging.getLogger(__name__)


TASK_QUEUE = "test-loop"


@pytest.fixture
def loop_registry():
    """Registry where agent returns 'COMPLETE' after 3 iterations."""
    reset_agent_registry()
    registry = get_agent_registry()

    agent = MagicMock()
    agent.name = "loop-agent"

    call_count = 0

    async def mock_run(agent, input, **kwargs):
        nonlocal call_count
        call_count += 1
        resp = MagicMock()
        resp.content = f"Iteration {call_count}" if call_count < 3 else "COMPLETE"
        resp.status = MagicMock()
        resp.status.value = "success"
        resp.structured_output = None
        resp.usage = MagicMock()
        resp.usage.prompt_tokens = 5
        resp.usage.completion_tokens = 10
        resp.usage.total_tokens = 15
        resp.agents_used = ["loop-agent"]
        resp.error = None
        return resp

    runner = MagicMock()
    runner.run = AsyncMock(side_effect=mock_run)
    registry.register(agent)
    registry.set_runner_factory(lambda: runner)

    yield registry
    reset_agent_registry()


@pytest.mark.asyncio
async def test_loop_terminates_on_phrase(temporal_env, loop_registry):
    """Loop terminates when agent output contains termination phrase."""
    logger.info("Loop terminates when agent output contains termination phrase")
    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[LoopAgentWorkflow],
        activities=[run_agent_activity, send_notification_activity],
    ):
        handle = await temporal_env.client.start_workflow(
            LoopAgentWorkflow.run,
            LoopWorkflowInput(
                agent_name="loop-agent",
                initial_input="Start looping",
                max_iterations=10,
                termination_phrase="COMPLETE",
            ),
            id="test-loop-term-1",
            task_queue=TASK_QUEUE,
        )
        result = await handle.result()
        assert result.status == "completed"
        assert len(result.step_results) == 3


@pytest.mark.asyncio
async def test_loop_max_iterations(temporal_env, mock_registry):
    """Loop stops at max iterations."""
    logger.info("Loop stops at max iterations")
    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[LoopAgentWorkflow],
        activities=[run_agent_activity, send_notification_activity],
    ):
        handle = await temporal_env.client.start_workflow(
            LoopAgentWorkflow.run,
            LoopWorkflowInput(
                agent_name="agent-a",
                initial_input="Loop forever",
                max_iterations=3,
                termination_phrase="NEVER_MATCHES",
            ),
            id="test-loop-max-1",
            task_queue=TASK_QUEUE,
        )
        result = await handle.result()
        assert result.status == "completed"
        assert len(result.step_results) == 3


@pytest.mark.asyncio
async def test_loop_single_iteration(temporal_env, mock_registry):
    """Loop with max_iterations=1 runs once."""
    logger.info("Loop with max_iterations=1 runs once")
    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[LoopAgentWorkflow],
        activities=[run_agent_activity, send_notification_activity],
    ):
        handle = await temporal_env.client.start_workflow(
            LoopAgentWorkflow.run,
            LoopWorkflowInput(
                agent_name="agent-a",
                initial_input="Just once",
                max_iterations=1,
            ),
            id="test-loop-single-1",
            task_queue=TASK_QUEUE,
        )
        result = await handle.result()
        assert result.status == "completed"
        assert len(result.step_results) == 1
