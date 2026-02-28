"""Integration tests for parallel execution."""

import pytest

from temporalio.worker import Worker

from orchestrator.temporal.activities import run_agent_activity, send_notification_activity
from orchestrator.temporal.types import (
    AgentStep,
    ParallelStep,
    WorkflowInput,
    WorkflowResult,
)
from orchestrator.temporal.workflows.agent_workflow import AgentWorkflow
from orchestrator.temporal.workflows.parallel_workflow import (
    ParallelAgentWorkflow,
    ParallelWorkflowInput,
)
import logging

logger = logging.getLogger(__name__)


TASK_QUEUE = "test-parallel"


@pytest.mark.asyncio
async def test_parallel_step_concatenate(temporal_env, mock_registry):
    """Parallel step with concatenate merge strategy."""
    logger.info("Parallel step with concatenate merge strategy")
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
                    ParallelStep(
                        agents=[
                            AgentStep(agent_name="agent-a"),
                            AgentStep(agent_name="agent-b"),
                        ],
                        merge_strategy="concatenate",
                    ).model_dump(),
                ],
            ),
            id="test-par-concat-1",
            task_queue=TASK_QUEUE,
        )
        result = await handle.result()
        assert result.status == "completed"
        assert "Result from A" in result.content
        assert "Result from B" in result.content


@pytest.mark.asyncio
async def test_parallel_step_first_success(temporal_env, mock_registry):
    """Parallel step with first_success merge strategy."""
    logger.info("Parallel step with first_success merge strategy")
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
                    ParallelStep(
                        agents=[
                            AgentStep(agent_name="agent-a"),
                            AgentStep(agent_name="agent-b"),
                        ],
                        merge_strategy="first_success",
                    ).model_dump(),
                ],
            ),
            id="test-par-first-1",
            task_queue=TASK_QUEUE,
        )
        result = await handle.result()
        assert result.status == "completed"
        assert result.content in ("Result from A", "Result from B")


@pytest.mark.asyncio
async def test_parallel_step_structured(temporal_env, mock_registry):
    """Parallel step with structured merge strategy."""
    logger.info("Parallel step with structured merge strategy")
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
                    ParallelStep(
                        agents=[
                            AgentStep(agent_name="agent-a"),
                            AgentStep(agent_name="agent-b"),
                        ],
                        merge_strategy="structured",
                    ).model_dump(),
                ],
            ),
            id="test-par-struct-1",
            task_queue=TASK_QUEUE,
        )
        result = await handle.result()
        assert result.status == "completed"
        assert "Result from A" in result.content
        assert "Result from B" in result.content


@pytest.mark.asyncio
async def test_parallel_three_agents(temporal_env, mock_registry):
    """Parallel step with three agents."""
    logger.info("Parallel step with three agents")
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
                    ParallelStep(
                        agents=[
                            AgentStep(agent_name="agent-a"),
                            AgentStep(agent_name="agent-b"),
                            AgentStep(agent_name="agent-c"),
                        ],
                    ).model_dump(),
                ],
            ),
            id="test-par-three-1",
            task_queue=TASK_QUEUE,
        )
        result = await handle.result()
        assert result.status == "completed"
        assert len(result.step_results) == 3


@pytest.mark.asyncio
async def test_parallel_workflow_convenience(temporal_env, mock_registry):
    """ParallelAgentWorkflow convenience wrapper."""
    logger.info("ParallelAgentWorkflow convenience wrapper")
    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[ParallelAgentWorkflow],
        activities=[run_agent_activity, send_notification_activity],
    ):
        handle = await temporal_env.client.start_workflow(
            ParallelAgentWorkflow.run,
            ParallelWorkflowInput(
                agent_names=["agent-a", "agent-b"],
                initial_input="Parallel convenience test",
            ),
            id="test-par-conv-1",
            task_queue=TASK_QUEUE,
        )
        result = await handle.result()
        assert result.status == "completed"
        assert len(result.step_results) == 2
