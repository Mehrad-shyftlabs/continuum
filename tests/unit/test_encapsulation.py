"""Unit tests for encapsulation fixes (Issue 10 - no private attribute access)."""

from unittest.mock import MagicMock

import pytest

from orchestrator.agent.execution.handoff_executor import HandoffExecutor
import logging

logger = logging.getLogger(__name__)


class TestHandoffExecutorSetExecutor:
    """Tests for HandoffExecutor.set_executor() method."""

    def test_handoff_executor_set_executor(self):
        logger.info("HandoffExecutorSetExecutor: handoff executor set executor")
        he = HandoffExecutor()
        mock_executor = MagicMock()
        he.set_executor(mock_executor)
        assert he._executor is mock_executor

    def test_handoff_executor_set_executor_replaces(self):
        logger.info("HandoffExecutorSetExecutor: handoff executor set executor replaces")
        he = HandoffExecutor(executor=MagicMock())
        new_executor = MagicMock()
        he.set_executor(new_executor)
        assert he._executor is new_executor


class TestSessionClientSetProvider:
    """Tests for SessionClient.set_provider() method."""

    def test_session_client_set_provider(self):
        logger.info("SessionClientSetProvider: session client set provider")
        from orchestrator.session.client import SessionClient

        client = SessionClient(auto_initialize=False)
        mock_provider = MagicMock()
        client.set_provider(mock_provider)
        assert client._provider is mock_provider

    def test_session_client_set_provider_replaces(self):
        logger.info("SessionClientSetProvider: session client set provider replaces")
        from orchestrator.session.client import SessionClient

        old_provider = MagicMock()
        client = SessionClient(provider=old_provider, auto_initialize=False)
        new_provider = MagicMock()
        client.set_provider(new_provider)
        assert client._provider is new_provider
