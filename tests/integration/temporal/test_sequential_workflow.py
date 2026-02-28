"""Integration tests for SequentialAgentWorkflow."""

import asyncio

import pytest

from temporalio.worker import Worker

from orchestrator.temporal.activities import run_agent_activity, send_notification_activity
from orchestrator.temporal.types import ApprovalDecision, WorkflowResult
from orchestrator.temporal.workflows.sequential_workflow import (
    SequentialAgentWorkflow,
    SequentialWorkflowInput,
)
import logging

logger = logging.getLogger(__name__)


TASK_QUEUE = "test-sequential"


@pytest.mark.asyncio
async def test_sequential_two_agents(temporal_env, mock_registry):
    """Sequential workflow runs agents in order."""
    logger.info("Sequential workflow runs agents in order")
    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[SequentialAgentWorkflow],
        activities=[run_agent_activity, send_notification_activity],
    ):
        handle = await temporal_env.client.start_workflow(
            SequentialAgentWorkflow.run,
            SequentialWorkflowInput(
                agent_names=["agent-a", "agent-b"],
                initial_input="Start here",
            ),
            id="test-seq-conv-1",
            task_queue=TASK_QUEUE,
        )
        result = await handle.result()
        assert result.status == "completed"
        assert len(result.step_results) == 2


@pytest.mark.asyncio
async def test_sequential_three_agents(temporal_env, mock_registry):
    """Sequential workflow with three agents."""
    logger.info("Sequential workflow with three agents")
    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[SequentialAgentWorkflow],
        activities=[run_agent_activity, send_notification_activity],
    ):
        handle = await temporal_env.client.start_workflow(
            SequentialAgentWorkflow.run,
            SequentialWorkflowInput(
                agent_names=["agent-a", "agent-b", "agent-c"],
                initial_input="Pipeline start",
            ),
            id="test-seq-three-1",
            task_queue=TASK_QUEUE,
        )
        result = await handle.result()
        assert result.status == "completed"
        assert len(result.step_results) == 3


@pytest.mark.asyncio
async def test_sequential_with_approval_gates(temporal_env, mock_registry):
    """Sequential workflow with approval between steps."""
    logger.info("Sequential workflow with approval between steps")
    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[SequentialAgentWorkflow],
        activities=[run_agent_activity, send_notification_activity],
    ):
        handle = await temporal_env.client.start_workflow(
            SequentialAgentWorkflow.run,
            SequentialWorkflowInput(
                agent_names=["agent-a", "agent-b"],
                initial_input="With approvals",
                approval_between_steps=True,
                approval_timeout=3600,
            ),
            id="test-seq-approval-1",
            task_queue=TASK_QUEUE,
        )

        await asyncio.sleep(1)
        await handle.signal(
            SequentialAgentWorkflow.submit_approval,
            ApprovalDecision(
                request_id="r1",
                decision="approved",
                decided_by="admin",
            ),
        )

        result = await handle.result()
        assert result.status == "completed"
        assert len(result.step_results) == 2


@pytest.mark.asyncio
async def test_sequential_cancel(temporal_env, mock_registry):
    """Cancel signal stops sequential workflow."""
    logger.info("Cancel signal stops sequential workflow")
    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[SequentialAgentWorkflow],
        activities=[run_agent_activity, send_notification_activity],
    ):
        handle = await temporal_env.client.start_workflow(
            SequentialAgentWorkflow.run,
            SequentialWorkflowInput(
                agent_names=["agent-a", "agent-b"],
                initial_input="Cancel me",
                approval_between_steps=True,
                approval_timeout=3600,
            ),
            id="test-seq-cancel-1",
            task_queue=TASK_QUEUE,
        )

        await asyncio.sleep(1)
        await handle.signal(SequentialAgentWorkflow.cancel_workflow)

        result = await handle.result()
        assert result.status in ("cancelled", "rejected")
