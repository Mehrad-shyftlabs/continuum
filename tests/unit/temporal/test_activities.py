"""Tests for Temporal activities."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from orchestrator.temporal.types import (
    AgentActivityParams,
    AgentActivityResult,
    NotificationParams,
)
import logging

logger = logging.getLogger(__name__)


class TestRunAgentActivity:
    @pytest.mark.asyncio
    async def test_success(self):
        logger.info("RunAgentActivity: success")
        mock_response = MagicMock()
        mock_response.content = "Agent output"
        mock_response.status.value = "success"
        mock_response.structured_output = None
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_response.agents_used = ["test-agent"]
        mock_response.error = None

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_response)

        mock_agent = MagicMock()
        mock_agent.name = "test-agent"

        mock_registry = MagicMock()
        mock_registry.get_runner.return_value = mock_runner
        mock_registry.get.return_value = mock_agent

        with (
            patch("orchestrator.temporal.activities.get_agent_registry", return_value=mock_registry),
            patch("orchestrator.temporal.activities.activity") as mock_activity,
        ):
            mock_activity.heartbeat = MagicMock()

            from orchestrator.temporal.activities import run_agent_activity

            params = AgentActivityParams(agent_name="test-agent", input="Hello")
            result = await run_agent_activity(params)

            assert result.content == "Agent output"
            assert result.status == "success"
            assert result.usage["total_tokens"] == 30
            mock_runner.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_error_captured(self):
        logger.info("RunAgentActivity: agent error captured")
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(side_effect=RuntimeError("LLM failed"))

        mock_agent = MagicMock()
        mock_agent.name = "fail-agent"

        mock_registry = MagicMock()
        mock_registry.get_runner.return_value = mock_runner
        mock_registry.get.return_value = mock_agent

        with (
            patch("orchestrator.temporal.activities.get_agent_registry", return_value=mock_registry),
            patch("orchestrator.temporal.activities.activity") as mock_activity,
        ):
            mock_activity.heartbeat = MagicMock()
            mock_activity.logger = MagicMock()

            from orchestrator.temporal.activities import run_agent_activity

            params = AgentActivityParams(agent_name="fail-agent", input="Hello")
            result = await run_agent_activity(params)

            assert result.status == "error"
            assert "LLM failed" in result.error
            assert result.content == ""

    @pytest.mark.asyncio
    async def test_missing_agent_raises(self):
        logger.info("RunAgentActivity: missing agent raises")
        from orchestrator.temporal.exceptions import AgentNotRegisteredError

        mock_registry = MagicMock()
        mock_registry.get_runner.return_value = MagicMock()
        mock_registry.get.side_effect = AgentNotRegisteredError("Not found")

        with (
            patch("orchestrator.temporal.activities.get_agent_registry", return_value=mock_registry),
            patch("orchestrator.temporal.activities.activity") as mock_activity,
        ):
            mock_activity.heartbeat = MagicMock()
            mock_activity.logger = MagicMock()

            from orchestrator.temporal.activities import run_agent_activity

            params = AgentActivityParams(agent_name="missing", input="Hello")
            result = await run_agent_activity(params)
            assert result.status == "error"
            assert "Not found" in result.error

    @pytest.mark.asyncio
    async def test_heartbeat_called(self):
        logger.info("RunAgentActivity: heartbeat called")
        mock_response = MagicMock()
        mock_response.content = "ok"
        mock_response.status.value = "success"
        mock_response.structured_output = None
        mock_response.usage.prompt_tokens = 0
        mock_response.usage.completion_tokens = 0
        mock_response.usage.total_tokens = 0
        mock_response.agents_used = []
        mock_response.error = None

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_response)
        mock_agent = MagicMock()
        mock_agent.name = "hb-agent"

        mock_registry = MagicMock()
        mock_registry.get_runner.return_value = mock_runner
        mock_registry.get.return_value = mock_agent

        with (
            patch("orchestrator.temporal.activities.get_agent_registry", return_value=mock_registry),
            patch("orchestrator.temporal.activities.activity") as mock_activity,
        ):
            mock_activity.heartbeat = MagicMock()

            from orchestrator.temporal.activities import run_agent_activity

            params = AgentActivityParams(agent_name="hb-agent", input="test")
            await run_agent_activity(params)
            mock_activity.heartbeat.assert_called_once()

    @pytest.mark.asyncio
    async def test_params_passed_to_runner(self):
        logger.info("RunAgentActivity: params passed to runner")
        mock_response = MagicMock()
        mock_response.content = "ok"
        mock_response.status.value = "success"
        mock_response.structured_output = None
        mock_response.usage.prompt_tokens = 0
        mock_response.usage.completion_tokens = 0
        mock_response.usage.total_tokens = 0
        mock_response.agents_used = []
        mock_response.error = None

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_response)
        mock_agent = MagicMock()
        mock_agent.name = "param-agent"

        mock_registry = MagicMock()
        mock_registry.get_runner.return_value = mock_runner
        mock_registry.get.return_value = mock_agent

        with (
            patch("orchestrator.temporal.activities.get_agent_registry", return_value=mock_registry),
            patch("orchestrator.temporal.activities.activity") as mock_activity,
        ):
            mock_activity.heartbeat = MagicMock()

            from orchestrator.temporal.activities import run_agent_activity

            params = AgentActivityParams(
                agent_name="param-agent",
                input="hello",
                session_id="s1",
                user_id="u1",
                metadata={"key": "val"},
                tags=["tag1"],
            )
            await run_agent_activity(params)

            call_kwargs = mock_runner.run.call_args
            assert call_kwargs.kwargs["session_id"] == "s1"
            assert call_kwargs.kwargs["user_id"] == "u1"


class TestSendNotificationActivity:
    @pytest.mark.asyncio
    async def test_with_handler(self):
        logger.info("SendNotificationActivity: with handler")
        mock_handler = AsyncMock()
        mock_registry = MagicMock()
        mock_registry.get_notification_handler.return_value = mock_handler

        with (
            patch("orchestrator.temporal.activities.get_agent_registry", return_value=mock_registry),
            patch("orchestrator.temporal.activities.activity") as mock_activity,
        ):
            mock_activity.logger = MagicMock()

            from orchestrator.temporal.activities import send_notification_activity

            params = NotificationParams(
                type="approval_required",
                payload={"workflow_id": "wf-1"},
            )
            await send_notification_activity(params)
            mock_handler.assert_called_once_with(params)

    @pytest.mark.asyncio
    async def test_without_handler(self):
        logger.info("SendNotificationActivity: without handler")
        mock_registry = MagicMock()
        mock_registry.get_notification_handler.return_value = None

        with (
            patch("orchestrator.temporal.activities.get_agent_registry", return_value=mock_registry),
            patch("orchestrator.temporal.activities.activity") as mock_activity,
        ):
            mock_activity.logger = MagicMock()

            from orchestrator.temporal.activities import send_notification_activity

            params = NotificationParams(type="test", payload={})
            await send_notification_activity(params)
            mock_activity.logger.warning.assert_called_once()
