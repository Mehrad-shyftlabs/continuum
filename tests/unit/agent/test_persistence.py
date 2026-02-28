"""Tests for agent/persistence/state.py."""

from unittest.mock import MagicMock, patch

import pytest

from orchestrator.agent.types import RunState, RunStatus
import logging

logger = logging.getLogger(__name__)


class TestRunStateManager:
    @patch("orchestrator.agent.persistence.state.settings")
    def test_init_no_auto(self, mock_settings):
        logger.info("RunStateManager: init no auto")
        mock_settings.session_redis_host = "localhost"
        mock_settings.session_redis_port = 6379
        mock_settings.session_redis_password = None
        mock_settings.session_redis_db = 0

        from orchestrator.agent.persistence.state import RunStateManager
        mgr = RunStateManager(auto_initialize=False)
        assert mgr.is_enabled is False
        assert mgr._initialized is False

    @patch("orchestrator.agent.persistence.state.settings")
    def test_key_prefix(self, mock_settings):
        logger.info("RunStateManager: key prefix")
        mock_settings.session_redis_host = "localhost"
        mock_settings.session_redis_port = 6379
        mock_settings.session_redis_password = None
        mock_settings.session_redis_db = 0

        from orchestrator.agent.persistence.state import RunStateManager
        assert RunStateManager.KEY_PREFIX == "orchestrator:run_state"

    @patch("orchestrator.agent.persistence.state.settings")
    def test_custom_ttl(self, mock_settings):
        logger.info("RunStateManager: custom ttl")
        mock_settings.session_redis_host = "localhost"
        mock_settings.session_redis_port = 6379
        mock_settings.session_redis_password = None
        mock_settings.session_redis_db = 0

        from orchestrator.agent.persistence.state import RunStateManager
        mgr = RunStateManager(state_ttl=7200, auto_initialize=False)
        assert mgr._state_ttl == 7200

    @patch("orchestrator.agent.persistence.state.settings")
    def test_custom_redis_params(self, mock_settings):
        logger.info("RunStateManager: custom redis params")
        mock_settings.session_redis_host = "default"
        mock_settings.session_redis_port = 6379
        mock_settings.session_redis_password = None
        mock_settings.session_redis_db = 0

        from orchestrator.agent.persistence.state import RunStateManager
        mgr = RunStateManager(
            redis_host="custom-host",
            redis_port=6380,
            redis_password="secret",
            redis_db=2,
            auto_initialize=False,
        )
        assert mgr._redis_host == "custom-host"
        assert mgr._redis_port == 6380
        assert mgr._redis_password == "secret"
        assert mgr._redis_db == 2

    @patch("orchestrator.agent.persistence.state.settings")
    def test_get_key(self, mock_settings):
        logger.info("RunStateManager: get key")
        mock_settings.session_redis_host = "localhost"
        mock_settings.session_redis_port = 6379
        mock_settings.session_redis_password = None
        mock_settings.session_redis_db = 0

        from orchestrator.agent.persistence.state import RunStateManager
        mgr = RunStateManager(auto_initialize=False)
        key = mgr._get_key("run-123")
        assert "orchestrator:run_state" in key
        assert "run-123" in key

    @patch("orchestrator.agent.persistence.state.settings")
    def test_get_index_key(self, mock_settings):
        logger.info("RunStateManager: get index key")
        mock_settings.session_redis_host = "localhost"
        mock_settings.session_redis_port = 6379
        mock_settings.session_redis_password = None
        mock_settings.session_redis_db = 0

        from orchestrator.agent.persistence.state import RunStateManager

        mgr = RunStateManager(auto_initialize=False)
        key = mgr._get_index_key("session", "s1")
        assert "idx" in key
        assert "session" in key
        assert "s1" in key
