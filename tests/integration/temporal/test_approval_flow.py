"""Integration tests for HITL approval flows."""

import asyncio

import pytest

from temporalio.worker import Worker

from orchestrator.temporal.activities import run_agent_activity, send_notification_activity
from orchestrator.temporal.types import (
    AgentStep,
    ApprovalDecision,
    ApprovalStep,
    WorkflowInput,
    WorkflowResult,
)
from orchestrator.temporal.workflows.agent_workflow import AgentWorkflow
import logging

logger = logging.getLogger(__name__)


TASK_QUEUE = "test-approval"


@pytest.mark.asyncio
async def test_approval_approved(temporal_env, mock_registry):
    """Workflow resumes when human approves."""
    logger.info("Workflow resumes when human approves")
    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[AgentWorkflow],
        activities=[run_agent_activity, send_notification_activity],
    ):
        handle = await temporal_env.client.start_workflow(
            AgentWorkflow.run,
            WorkflowInput(
                initial_input="Test input",
                steps=[
                    AgentStep(agent_name="agent-a").model_dump(),
                    ApprovalStep(description="Review research", timeout=3600).model_dump(),
                    AgentStep(agent_name="agent-b").model_dump(),
                ],
            ),
            id="test-approval-1",
            task_queue=TASK_QUEUE,
        )

        # Allow time for the workflow to reach the approval step
        await asyncio.sleep(1)

        # Signal approval
        await handle.signal(
            AgentWorkflow.submit_approval,
            ApprovalDecision(
                request_id="any",
                decision="approved",
                decided_by="admin",
            ),
        )

        result = await handle.result()
        assert result.status == "completed"
        assert len(result.step_results) == 2  # agent-a + agent-b
        assert len(result.approval_decisions) == 1
        assert result.approval_decisions[0].decision == "approved"


@pytest.mark.asyncio
async def test_approval_rejected(temporal_env, mock_registry):
    """Workflow stops when human rejects."""
    logger.info("Workflow stops when human rejects")
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
                    ApprovalStep(description="Review", timeout=3600).model_dump(),
                    AgentStep(agent_name="agent-b").model_dump(),
                ],
            ),
            id="test-rejection-1",
            task_queue=TASK_QUEUE,
        )

        await asyncio.sleep(1)

        await handle.signal(
            AgentWorkflow.submit_approval,
            ApprovalDecision(
                request_id="any",
                decision="rejected",
                decided_by="admin",
                reason="Not acceptable",
            ),
        )

        result = await handle.result()
        assert result.status == "rejected"
        assert len(result.step_results) == 1  # Only agent-a ran


@pytest.mark.asyncio
async def test_approval_timeout(temporal_env, mock_registry):
    """Approval gate times out when no human responds."""
    logger.info("Approval gate times out when no human responds")
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
                    ApprovalStep(description="Review", timeout=5).model_dump(),
                    AgentStep(agent_name="agent-b").model_dump(),
                ],
            ),
            id="test-timeout-1",
            task_queue=TASK_QUEUE,
        )

        # Time-skipping environment will advance past the 5-second timeout
        result = await handle.result()
        assert result.status in ("rejected", "timed_out")
        assert len(result.step_results) == 1  # Only agent-a ran


@pytest.mark.asyncio
async def test_multiple_approval_gates(temporal_env, mock_registry):
    """Multiple approval steps in a single workflow."""
    logger.info("Multiple approval steps in a single workflow")
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
                    ApprovalStep(description="First review", timeout=3600).model_dump(),
                    AgentStep(agent_name="agent-b").model_dump(),
                    ApprovalStep(description="Second review", timeout=3600).model_dump(),
                    AgentStep(agent_name="agent-c").model_dump(),
                ],
            ),
            id="test-multi-approval-1",
            task_queue=TASK_QUEUE,
        )

        await asyncio.sleep(1)
        await handle.signal(
            AgentWorkflow.submit_approval,
            ApprovalDecision(request_id="r1", decision="approved", decided_by="admin"),
        )

        await asyncio.sleep(1)
        await handle.signal(
            AgentWorkflow.submit_approval,
            ApprovalDecision(request_id="r2", decision="approved", decided_by="admin"),
        )

        result = await handle.result()
        assert result.status == "completed"
        assert len(result.step_results) == 3
        assert len(result.approval_decisions) == 2


@pytest.mark.asyncio
async def test_cancel_during_approval(temporal_env, mock_registry):
    """Cancel signal stops workflow during approval wait."""
    logger.info("Cancel signal stops workflow during approval wait")
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
                    ApprovalStep(description="Review", timeout=3600).model_dump(),
                    AgentStep(agent_name="agent-b").model_dump(),
                ],
            ),
            id="test-cancel-approval-1",
            task_queue=TASK_QUEUE,
        )

        await asyncio.sleep(1)
        await handle.signal(AgentWorkflow.cancel_workflow)

        result = await handle.result()
        assert result.status in ("cancelled", "rejected")


@pytest.mark.asyncio
async def test_approval_at_start(temporal_env, mock_registry):
    """Approval step as the very first step."""
    logger.info("Approval step as the very first step")
    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[AgentWorkflow],
        activities=[run_agent_activity, send_notification_activity],
    ):
        handle = await temporal_env.client.start_workflow(
            AgentWorkflow.run,
            WorkflowInput(
                initial_input="Need approval first",
                steps=[
                    ApprovalStep(description="Pre-approval", timeout=3600).model_dump(),
                    AgentStep(agent_name="agent-a").model_dump(),
                ],
            ),
            id="test-approval-start-1",
            task_queue=TASK_QUEUE,
        )

        await asyncio.sleep(1)
        await handle.signal(
            AgentWorkflow.submit_approval,
            ApprovalDecision(request_id="r1", decision="approved", decided_by="admin"),
        )

        result = await handle.result()
        assert result.status == "completed"
        assert len(result.step_results) == 1
