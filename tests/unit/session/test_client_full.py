"""Comprehensive tests for session/client.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.session.client import SessionClient
from orchestrator.session.config import SessionConfig
import logging

logger = logging.getLogger(__name__)


class TestSessionClientInit:
    def test_default_disabled(self):
        logger.info("SessionClientInit: default disabled")
        config = SessionConfig(enabled=False)
        client = SessionClient(session_config=config, auto_initialize=False)
        assert client is not None
        assert client.is_enabled is False

    def test_config_property(self):
        logger.info("SessionClientInit: config property")
        config = SessionConfig(enabled=False)
        client = SessionClient(session_config=config, auto_initialize=False)
        assert client.config is config


class TestSessionClientOperationsFull:
    def _make_client(self):
        config = SessionConfig(enabled=True)
        client = SessionClient.__new__(SessionClient)
        client._session_config = config
        client._provider = MagicMock()
        client._memory_client = None
        client._initialized = True
        import threading
        client._lock = threading.Lock()
        return client

    @pytest.mark.asyncio
    async def test_get_or_create_session(self):
        logger.info("SessionClientOperationsFull: get or create session")
        client = self._make_client()
        mock_metadata = MagicMock()
        mock_metadata.session_id = "s1"
        client._provider.get_or_create_session = AsyncMock(return_value=mock_metadata)
        result = await client.get_or_create_session("s1")
        assert result is not None

    @pytest.mark.asyncio
    async def test_add_message(self):
        logger.info("SessionClientOperationsFull: add message")
        client = self._make_client()
        client._provider.add_message = AsyncMock()
        from orchestrator.llm.types import ChatMessage

        msg = ChatMessage(role="user", content="hello")
        await client.add_message("s1", msg)
        client._provider.add_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_conversation_history(self):
        logger.info("SessionClientOperationsFull: get conversation history")
        client = self._make_client()
        client._provider.get_messages = AsyncMock(return_value=[])
        result = await client.get_conversation_history("s1")
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_session_metadata(self):
        logger.info("SessionClientOperationsFull: get session metadata")
        client = self._make_client()
        mock_meta = MagicMock()
        client._provider.get_session_metadata = AsyncMock(return_value=mock_meta)
        result = await client.get_session_metadata("s1")
        assert result is mock_meta

    @pytest.mark.asyncio
    async def test_clear_session(self):
        logger.info("SessionClientOperationsFull: clear session")
        client = self._make_client()
        client._provider.clear_session = AsyncMock(return_value=True)
        result = await client.clear_session("s1")
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_session(self):
        logger.info("SessionClientOperationsFull: delete session")
        client = self._make_client()
        client._provider.delete_session = AsyncMock(return_value=True)
        result = await client.delete_session("s1")
        assert result is True

    def test_set_provider(self):
        logger.info("SessionClientOperationsFull: set provider")
        client = self._make_client()
        mock_provider = MagicMock()
        client.set_provider(mock_provider)
        assert client._provider is mock_provider
