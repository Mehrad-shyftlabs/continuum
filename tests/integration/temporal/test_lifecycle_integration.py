"""Integration tests for lifecycle and health check integration."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from orchestrator.core.health import HealthCheck, HealthStatus
import logging

logger = logging.getLogger(__name__)


class TestTemporalHealthCheck:
    @pytest.mark.asyncio
    async def test_temporal_disabled(self):
        logger.info("TemporalHealthCheck: temporal disabled")
        health = HealthCheck()
        with patch("orchestrator.core.health.settings") as mock_settings:
            mock_settings.temporal_enabled = False
            result = await health.check_temporal()
        assert result.status == HealthStatus.HEALTHY
        assert result.details["enabled"] is False

    @pytest.mark.asyncio
    async def test_temporal_not_installed(self):
        logger.info("TemporalHealthCheck: temporal not installed")
        health = HealthCheck()
        with (
            patch("orchestrator.core.health.settings") as mock_settings,
            patch.dict("sys.modules", {"temporalio": None, "temporalio.client": None}),
            patch("builtins.__import__", side_effect=ImportError("No module named 'temporalio'")),
        ):
            mock_settings.temporal_enabled = True
            mock_settings.temporal_host = "localhost:7233"
            mock_settings.temporal_namespace = "default"
            result = await health._check_temporal()
        assert result.status in (HealthStatus.DEGRADED, HealthStatus.UNHEALTHY)

    @pytest.mark.asyncio
    async def test_temporal_connection_failure(self):
        logger.info("TemporalHealthCheck: temporal connection failure")
        health = HealthCheck()
        with (
            patch("orchestrator.core.health.settings") as mock_settings,
            patch(
                "temporalio.client.Client.connect",
                new_callable=AsyncMock,
                side_effect=Exception("Connection refused"),
            ),
        ):
            mock_settings.temporal_enabled = True
            mock_settings.temporal_host = "localhost:7233"
            mock_settings.temporal_namespace = "default"
            result = await health._check_temporal()
        assert result.status == HealthStatus.UNHEALTHY
        assert "Connection refused" in result.message


class TestLifecycleIntegration:
    @pytest.mark.asyncio
    async def test_temporal_init_when_disabled(self):
        """Temporal init is skipped when temporal_enabled=False."""
        logger.info("Temporal init is skipped when temporal_enabled=False")
        from orchestrator.core.lifecycle import OrchestratorLifecycle

        lifecycle = OrchestratorLifecycle(
            verify_connections=False,
            enable_signal_handlers=False,
        )
        with patch("orchestrator.core.lifecycle.settings") as mock_settings:
            mock_settings.temporal_enabled = False
            mock_settings.memory_enabled = False
            mock_settings.session_enabled = False
            mock_settings.langfuse_enabled = False
            mock_settings.shared_services_enabled = True

            result = await lifecycle.initialize()

        assert "temporal" not in lifecycle._initialized_components

    @pytest.mark.asyncio
    async def test_temporal_init_when_enabled(self):
        """Temporal client connects when temporal_enabled=True."""
        logger.info("Temporal client connects when temporal_enabled=True")
        from orchestrator.core.lifecycle import OrchestratorLifecycle

        lifecycle = OrchestratorLifecycle(
            verify_connections=False,
            enable_signal_handlers=False,
        )

        mock_client = MagicMock()
        mock_client.connect = AsyncMock()

        with (
            patch("orchestrator.core.lifecycle.settings") as mock_settings,
            patch(
                "orchestrator.temporal.get_temporal_client",
                return_value=mock_client,
            ),
        ):
            mock_settings.temporal_enabled = True
            mock_settings.memory_enabled = False
            mock_settings.session_enabled = False
            mock_settings.langfuse_enabled = False
            mock_settings.shared_services_enabled = True

            await lifecycle._initialize_clients()

        assert "temporal" in lifecycle._initialized_components
        mock_client.connect.assert_called_once()
