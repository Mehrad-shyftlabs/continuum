"""Tests for HumanInLoopManager."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from orchestrator.temporal.human_in_loop import (
    ApprovalNotificationConfig,
    HumanInLoopManager,
)
from orchestrator.temporal.types import ApprovalDecision
import logging

logger = logging.getLogger(__name__)


class TestHumanInLoopManager:
    def setup_method(self):
        self.mock_client = MagicMock()
        self.mock_handle = MagicMock()
        self.mock_handle.signal = AsyncMock()
        self.mock_handle.query = AsyncMock()
        self.mock_client.get_workflow_handle = AsyncMock(return_value=self.mock_handle)
        self.manager = HumanInLoopManager(client=self.mock_client)

    @pytest.mark.asyncio
    async def test_approve(self):
        logger.info("HumanInLoopManager: approve")
        await self.manager.approve(
            workflow_id="wf-1",
            request_id="req-1",
            decided_by="admin",
            reason="Looks good",
        )
        self.mock_handle.signal.assert_called_once()
        call_args = self.mock_handle.signal.call_args
        decision = call_args.args[1]
        assert isinstance(decision, ApprovalDecision)
        assert decision.decision == "approved"
        assert decision.decided_by == "admin"

    @pytest.mark.asyncio
    async def test_reject(self):
        logger.info("HumanInLoopManager: reject")
        await self.manager.reject(
            workflow_id="wf-1",
            request_id="req-1",
            decided_by="admin",
            reason="Not ready",
        )
        self.mock_handle.signal.assert_called_once()
        call_args = self.mock_handle.signal.call_args
        decision = call_args.args[1]
        assert decision.decision == "rejected"

    @pytest.mark.asyncio
    async def test_submit_decision(self):
        logger.info("HumanInLoopManager: submit decision")
        decision = ApprovalDecision(
            request_id="req-1",
            decision="approved",
            decided_by="user",
        )
        await self.manager.submit_decision("wf-1", decision)
        self.mock_handle.signal.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_pending_approvals(self):
        logger.info("HumanInLoopManager: get pending approvals")
        self.mock_handle.query = AsyncMock(
            return_value=[{"request_id": "r1", "description": "Review"}]
        )
        result = await self.manager.get_pending_approvals("wf-1")
        assert len(result) == 1
        assert result[0]["request_id"] == "r1"

    @pytest.mark.asyncio
    async def test_get_workflow_status(self):
        logger.info("HumanInLoopManager: get workflow status")
        self.mock_handle.query = AsyncMock(
            return_value={"status": "running", "current_step_index": 2}
        )
        result = await self.manager.get_workflow_status("wf-1")
        assert result["status"] == "running"

    @pytest.mark.asyncio
    async def test_escalate(self):
        logger.info("HumanInLoopManager: escalate")
        self.mock_handle.query = AsyncMock(
            return_value=[
                {
                    "request_id": "req-1",
                    "workflow_id": "wf-1",
                    "step_index": 0,
                    "description": "Review",
                }
            ]
        )
        await self.manager.escalate("wf-1", "req-1", "senior-admin")
        self.mock_handle.signal.assert_called_once()
        call_args = self.mock_handle.signal.call_args
        decision = call_args.args[1]
        assert decision.decision == "escalated"

    @pytest.mark.asyncio
    async def test_escalate_with_handler(self):
        logger.info("HumanInLoopManager: escalate with handler")
        handler = AsyncMock()
        config = ApprovalNotificationConfig(escalation_handler=handler)
        manager = HumanInLoopManager(
            client=self.mock_client,
            notification_config=config,
        )
        self.mock_handle.query = AsyncMock(
            return_value=[
                {
                    "request_id": "req-1",
                    "workflow_id": "wf-1",
                    "step_index": 0,
                    "description": "Review",
                }
            ]
        )
        await manager.escalate("wf-1", "req-1", "senior")
        handler.assert_called_once()


class TestApprovalNotificationConfig:
    def test_defaults(self):
        logger.info("ApprovalNotificationConfig: defaults")
        config = ApprovalNotificationConfig()
        assert config.handler is None
        assert config.webhook_url is None
        assert config.timeout_seconds == 86400
        assert config.escalation_timeout == 7200
        assert config.auto_approve_conditions == []

    def test_custom(self):
        logger.info("ApprovalNotificationConfig: custom")
        handler = AsyncMock()
        config = ApprovalNotificationConfig(
            handler=handler,
            webhook_url="https://hooks.example.com/approve",
            timeout_seconds=3600,
        )
        assert config.handler is handler
        assert config.webhook_url == "https://hooks.example.com/approve"
        assert config.timeout_seconds == 3600
