"""Unit tests for session client."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.session.client import SessionClient
from orchestrator.session.config import SessionConfig
from orchestrator.session.exceptions import SessionNotEnabledError
import logging

logger = logging.getLogger(__name__)


class TestSessionClientInit:
    def test_client_initialization_disabled(self):
        logger.info("SessionClientInit: client initialization disabled")
        config = SessionConfig(enabled=False)
        client = SessionClient(session_config=config, auto_initialize=False)
        assert client.is_enabled is False

    def test_client_not_enabled_raises(self):
        logger.info("SessionClientInit: client not enabled raises")
        config = SessionConfig(enabled=False)
        client = SessionClient(session_config=config, auto_initialize=False)
        with pytest.raises(SessionNotEnabledError):
            client._ensure_enabled()


class TestSessionClientOperations:
    def _make_enabled_client(self):
        config = SessionConfig(enabled=True)
        client = SessionClient(session_config=config, auto_initialize=False)
        mock_provider = AsyncMock()
        client._provider = mock_provider
        return client, mock_provider

    @pytest.mark.asyncio
    async def test_get_or_create_session(self):
        logger.info("SessionClientOperations: get or create session")
        client, mock_provider = self._make_enabled_client()
        mock_provider.get_or_create_session.return_value = "session-123"
        result = await client.get_or_create_session(user_id="u1", agent_id="a1")
        assert result == "session-123"

    @pytest.mark.asyncio
    async def test_add_message(self):
        logger.info("SessionClientOperations: add message")
        client, mock_provider = self._make_enabled_client()
        client._memory_client = None
        from orchestrator.session.types import ChatMessage
        msg = ChatMessage(role="user", content="hello")
        await client.add_message("s1", msg)
        mock_provider.add_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_conversation_history(self):
        logger.info("SessionClientOperations: get conversation history")
        client, mock_provider = self._make_enabled_client()
        mock_provider.get_messages.return_value = []
        result = await client.get_conversation_history("s1")
        assert result is not None

    @pytest.mark.asyncio
    async def test_clear_session(self):
        logger.info("SessionClientOperations: clear session")
        client, mock_provider = self._make_enabled_client()
        await client.clear_session("s1")
        mock_provider.clear_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_session_metadata(self):
        logger.info("SessionClientOperations: get session metadata")
        from datetime import datetime
        from orchestrator.session.types import SessionMetadata

        client, mock_provider = self._make_enabled_client()
        mock_provider.get_session_metadata.return_value = SessionMetadata(
            session_id="s1", message_count=5,
            created_at=datetime.now(), last_accessed_at=datetime.now(),
        )
        result = await client.get_session_metadata("s1")
        assert result.session_id == "s1"
