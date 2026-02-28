"""Tests for agent/services/ modules."""

from unittest.mock import MagicMock

import pytest
import logging

logger = logging.getLogger(__name__)


class TestContextService:
    def test_init_no_args(self):
        logger.info("ContextService: init no args")
        from orchestrator.agent.services.context_service import ContextService
        cs = ContextService()
        assert cs._state_manager is None

    def test_init_with_args(self):
        logger.info("ContextService: init with args")
        from orchestrator.agent.services.context_service import ContextService
        mock_state = MagicMock()
        mock_config = MagicMock()
        cs = ContextService(state_manager=mock_state, config=mock_config)
        assert cs._state_manager is mock_state
        assert cs._config is mock_config


class TestMemoryService:
    def test_init_no_args(self):
        logger.info("MemoryService: init no args")
        from orchestrator.agent.services.memory_service import MemoryService
        ms = MemoryService()
        assert ms._memory_client is None

    def test_init_with_clients(self):
        logger.info("MemoryService: init with clients")
        from orchestrator.agent.services.memory_service import MemoryService
        mock_mem = MagicMock()
        mock_sess = MagicMock()
        ms = MemoryService(memory_client=mock_mem, session_client=mock_sess)
        assert ms._memory_client is mock_mem
        assert ms._session_client is mock_sess


class TestSessionService:
    def test_init_no_args(self):
        logger.info("SessionService: init no args")
        from orchestrator.agent.services.session_service import SessionService
        ss = SessionService()
        assert ss._session_client is None

    def test_init_with_client(self):
        logger.info("SessionService: init with client")
        from orchestrator.agent.services.session_service import SessionService
        mock_client = MagicMock()
        ss = SessionService(session_client=mock_client)
        assert ss._session_client is mock_client


class TestToolService:
    def test_init_no_args(self):
        logger.info("ToolService: init no args")
        from orchestrator.agent.services.tool_service import ToolService
        ts = ToolService()
        assert ts._tool_executor is None

    def test_init_with_executor(self):
        logger.info("ToolService: init with executor")
        from orchestrator.agent.services.tool_service import ToolService

        mock_executor = MagicMock()
        ts = ToolService(tool_executor=mock_executor)
        assert ts._tool_executor is mock_executor
