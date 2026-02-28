"""Tests for Temporal configuration."""

from unittest.mock import patch

import pytest

from orchestrator.temporal.config import TemporalConfig
import logging

logger = logging.getLogger(__name__)


class TestTemporalConfig:
    def test_default_values(self):
        logger.info("TemporalConfig: default values")
        config = TemporalConfig()
        assert config.enabled is False
        assert config.host == "localhost:7233"
        assert config.namespace == "default"
        assert config.task_queue == "orchestrator-agents"
        assert config.enable_human_in_loop is True
        assert config.approval_timeout_seconds == 86400
        assert config.workflow_execution_timeout == 86400 * 7
        assert config.activity_start_to_close_timeout == 300
        assert config.activity_retry_max_attempts == 3

    def test_custom_values(self):
        logger.info("TemporalConfig: custom values")
        config = TemporalConfig(
            enabled=True,
            host="temporal.example.com:7233",
            namespace="production",
            task_queue="my-agents",
            approval_timeout_seconds=7200,
        )
        assert config.enabled is True
        assert config.host == "temporal.example.com:7233"
        assert config.namespace == "production"
        assert config.task_queue == "my-agents"
        assert config.approval_timeout_seconds == 7200

    def test_from_settings(self):
        logger.info("TemporalConfig: from settings")
        config = TemporalConfig.from_settings()
        assert isinstance(config, TemporalConfig)
        assert isinstance(config.host, str)
        assert isinstance(config.namespace, str)

    def test_enabled_default_false(self):
        logger.info("TemporalConfig: enabled default false")
        with patch("orchestrator.temporal.config.settings") as mock_settings:
            mock_settings.temporal_enabled = False
            mock_settings.temporal_host = "localhost:7233"
            mock_settings.temporal_namespace = "default"
            mock_settings.temporal_task_queue = "orchestrator-agents"
            mock_settings.temporal_enable_human_in_loop = True
            mock_settings.temporal_approval_timeout_seconds = 86400
            mock_settings.temporal_workflow_execution_timeout = 604800
            mock_settings.temporal_activity_start_to_close_timeout = 300
            mock_settings.temporal_activity_retry_max_attempts = 3
            config = TemporalConfig.from_settings()
            assert config.enabled is False

    def test_workflow_timeout_is_7_days(self):
        logger.info("TemporalConfig: workflow timeout is 7 days")
        config = TemporalConfig()
        assert config.workflow_execution_timeout == 604800

    def test_approval_timeout_is_24h(self):
        logger.info("TemporalConfig: approval timeout is 24h")
        config = TemporalConfig()
        assert config.approval_timeout_seconds == 86400

    def test_activity_timeout_is_5min(self):
        logger.info("TemporalConfig: activity timeout is 5min")
        config = TemporalConfig()
        assert config.activity_start_to_close_timeout == 300

    def test_retry_max_attempts(self):
        logger.info("TemporalConfig: retry max attempts")
        config = TemporalConfig()
        assert config.activity_retry_max_attempts == 3
