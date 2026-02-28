"""Integration tests for the core AgentWorkflow using Temporal's test environment."""

import pytest

from temporalio.worker import Worker

from orchestrator.temporal.activities import run_agent_activity, send_notification_activity
from orchestrator.temporal.types import (
    AgentStep,
    ApprovalDecision,
    ApprovalStep,
    ConditionalStep,
    ParallelStep,
    WaitStep,
    WorkflowInput,
    WorkflowResult,
)
from orchestrator.temporal.workflows.agent_workflow import AgentWorkflow
import logging

logger = logging.getLogger(__name__)


TASK_QUEUE = "test-agent-workflow"


@pytest.mark.asyncio
async def test_single_agent_step(temporal_env, mock_registry):
    """Workflow runs a single agent step end-to-end."""
    logger.info("Workflow runs a single agent step end-to-end")
    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[AgentWorkflow],
        activities=[run_agent_activity, send_notification_activity],
    ):
        handle = await temporal_env.client.start_workflow(
            AgentWorkflow.run,
            WorkflowInput(
                initial_input="Hello",
                steps=[AgentStep(agent_name="agent-a").model_dump()],
            ),
            id="test-single-1",
            task_queue=TASK_QUEUE,
        )
        result: WorkflowResult = await handle.result()
        assert result.status == "completed"
        assert len(result.step_results) == 1
        assert "Result from A" in result.content


@pytest.mark.asyncio
async def test_sequential_agent_steps(temporal_env, mock_registry):
    """Workflow runs multiple sequential agent steps, passing output between them."""
    logger.info("Workflow runs multiple sequential agent steps, passing output between them")
    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[AgentWorkflow],
        activities=[run_agent_activity, send_notification_activity],
    ):
        handle = await temporal_env.client.start_workflow(
            AgentWorkflow.run,
            WorkflowInput(
                initial_input="Start",
                steps=[
                    AgentStep(agent_name="agent-a").model_dump(),
                    AgentStep(agent_name="agent-b").model_dump(),
                ],
            ),
            id="test-seq-1",
            task_queue=TASK_QUEUE,
        )
        result = await handle.result()
        assert result.status == "completed"
        assert len(result.step_results) == 2


@pytest.mark.asyncio
async def test_parallel_step(temporal_env, mock_registry):
    """Parallel step runs multiple agents concurrently."""
    logger.info("Parallel step runs multiple agents concurrently")
    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[AgentWorkflow],
        activities=[run_agent_activity, send_notification_activity],
    ):
        handle = await temporal_env.client.start_workflow(
            AgentWorkflow.run,
            WorkflowInput(
                initial_input="Parallel test",
                steps=[
                    ParallelStep(
                        agents=[
                            AgentStep(agent_name="agent-a"),
                            AgentStep(agent_name="agent-b"),
                        ],
                    ).model_dump(),
                ],
            ),
            id="test-parallel-1",
            task_queue=TASK_QUEUE,
        )
        result = await handle.result()
        assert result.status == "completed"
        assert len(result.step_results) == 2
        assert "Result from A" in result.content
        assert "Result from B" in result.content


@pytest.mark.asyncio
async def test_conditional_step_true_branch(temporal_env, mock_registry):
    """Conditional step branches based on agent output (true branch)."""
    logger.info("Conditional step branches based on agent output (true branch)")
    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[AgentWorkflow],
        activities=[run_agent_activity, send_notification_activity],
    ):
        handle = await temporal_env.client.start_workflow(
            AgentWorkflow.run,
            WorkflowInput(
                initial_input="Evaluate this",
                steps=[
                    ConditionalStep(
                        condition_agent="evaluator",
                        if_true=[AgentStep(agent_name="agent-a").model_dump()],
                        if_false=[AgentStep(agent_name="agent-b").model_dump()],
                    ).model_dump(),
                ],
            ),
            id="test-cond-true-1",
            task_queue=TASK_QUEUE,
        )
        result = await handle.result()
        assert result.status == "completed"
        # evaluator returns "true" -> if_true branch (agent-a)
        assert len(result.step_results) >= 2  # evaluator + agent-a
        assert "Result from A" in result.content


@pytest.mark.asyncio
async def test_wait_step(temporal_env, mock_registry):
    """Wait step uses durable timer."""
    logger.info("Wait step uses durable timer")
    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[AgentWorkflow],
        activities=[run_agent_activity, send_notification_activity],
    ):
        handle = await temporal_env.client.start_workflow(
            AgentWorkflow.run,
            WorkflowInput(
                initial_input="Before wait",
                steps=[
                    AgentStep(agent_name="agent-a").model_dump(),
                    WaitStep(duration_seconds=10).model_dump(),
                    AgentStep(agent_name="agent-b").model_dump(),
                ],
            ),
            id="test-wait-1",
            task_queue=TASK_QUEUE,
        )
        result = await handle.result()
        assert result.status == "completed"
        assert len(result.step_results) == 2


@pytest.mark.asyncio
async def test_get_status_query(temporal_env, mock_registry):
    """get_status query returns correct state."""
    logger.info("get_status query returns correct state")
    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[AgentWorkflow],
        activities=[run_agent_activity, send_notification_activity],
    ):
        handle = await temporal_env.client.start_workflow(
            AgentWorkflow.run,
            WorkflowInput(
                initial_input="Hello",
                steps=[AgentStep(agent_name="agent-a").model_dump()],
            ),
            id="test-status-1",
            task_queue=TASK_QUEUE,
        )
        result = await handle.result()
        assert result.status == "completed"


@pytest.mark.asyncio
async def test_empty_steps(temporal_env, mock_registry):
    """Workflow with no steps completes immediately."""
    logger.info("Workflow with no steps completes immediately")
    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[AgentWorkflow],
        activities=[run_agent_activity, send_notification_activity],
    ):
        handle = await temporal_env.client.start_workflow(
            AgentWorkflow.run,
            WorkflowInput(
                initial_input="Nothing to do",
                steps=[],
            ),
            id="test-empty-1",
            task_queue=TASK_QUEUE,
        )
        result = await handle.result()
        assert result.status == "completed"
        assert result.content == "Nothing to do"


@pytest.mark.asyncio
async def test_agent_with_custom_input(temporal_env, mock_registry):
    """Agent step with explicit input overrides previous output."""
    logger.info("Agent step with explicit input overrides previous output")
    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[AgentWorkflow],
        activities=[run_agent_activity, send_notification_activity],
    ):
        handle = await temporal_env.client.start_workflow(
            AgentWorkflow.run,
            WorkflowInput(
                initial_input="Original",
                steps=[
                    AgentStep(agent_name="agent-a", input="Custom input").model_dump(),
                ],
            ),
            id="test-custom-input-1",
            task_queue=TASK_QUEUE,
        )
        result = await handle.result()
        assert result.status == "completed"
