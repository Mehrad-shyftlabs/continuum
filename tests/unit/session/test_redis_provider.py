"""Comprehensive tests for session/providers/redis.py."""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.session.config import SessionConfig
from orchestrator.session.exceptions import (
    SessionConnectionError,
    SessionMessageLimitError,
    SessionNotEnabledError,
    SessionNotFoundError,
)
from orchestrator.session.types import ChatMessage, SessionMessage, SessionMetadata
import logging

logger = logging.getLogger(__name__)


class TestRedisSessionProvider:
    def _make_config(self, enabled=True, configured=True):
        config = MagicMock(spec=SessionConfig)
        config.enabled = enabled
        config.is_configured.return_value = configured
        config.redis_host = "localhost"
        config.redis_port = 6379
        config.redis_password = None
        config.redis_db = 0
        config.redis_max_connections = 10
        config.ttl_seconds = 86400
        config.max_messages = 100
        config.key_prefix = "session"
        config.message_limit_strategy = "sliding_window"
        config.sliding_window_trim_count = 10
        return config

    def _make_provider(self, config=None):
        with patch("orchestrator.session.providers.redis.redis") as mock_redis:
            mock_redis.ConnectionPool.return_value = MagicMock()
            mock_redis.Redis.return_value = AsyncMock()

            from orchestrator.session.providers.redis import RedisSessionProvider

            if config is None:
                config = self._make_config()
            provider = RedisSessionProvider(config, auto_initialize=True)
            return provider

    def test_provider_name(self):
        logger.info("RedisSessionProvider: provider name")
        provider = self._make_provider()
        assert provider.provider_name == "redis"

    def test_config_property(self):
        logger.info("RedisSessionProvider: config property")
        config = self._make_config()
        provider = self._make_provider(config)
        assert provider.config is config

    def test_is_initialized(self):
        logger.info("RedisSessionProvider: is initialized")
        provider = self._make_provider()
        assert provider.is_initialized is True

    def test_is_initialized_disabled(self):
        logger.info("RedisSessionProvider: is initialized disabled")
        config = self._make_config(enabled=False)
        provider = self._make_provider(config)
        assert provider.is_initialized is False

    def test_ensure_enabled_raises(self):
        logger.info("RedisSessionProvider: ensure enabled raises")
        config = self._make_config(enabled=False)
        provider = self._make_provider(config)
        with pytest.raises(SessionNotEnabledError):
            provider._ensure_enabled()

    def test_get_session_key(self):
        logger.info("RedisSessionProvider: get session key")
        provider = self._make_provider()
        key = provider._get_session_key("s1")
        assert "s1" in key
        assert "messages" in key

    def test_get_metadata_key(self):
        logger.info("RedisSessionProvider: get metadata key")
        provider = self._make_provider()
        key = provider._get_metadata_key("s1")
        assert "s1" in key
        assert "metadata" in key

    def test_get_user_agent_session_key(self):
        logger.info("RedisSessionProvider: get user agent session key")
        provider = self._make_provider()
        key = provider._get_user_agent_session_key("u1", "a1")
        assert "u1" in key
        assert "a1" in key

    @pytest.mark.asyncio
    async def test_get_or_create_session_new(self):
        logger.info("RedisSessionProvider: get or create session new")
        provider = self._make_provider()
        provider._redis.get = AsyncMock(return_value=None)
        provider._redis.setex = AsyncMock()

        session_id = await provider.get_or_create_session(user_id="u1", agent_id="a1")
        assert session_id is not None

    @pytest.mark.asyncio
    async def test_get_or_create_session_existing_by_id(self):
        logger.info("RedisSessionProvider: get or create session existing by id")
        provider = self._make_provider()
        metadata = SessionMetadata(
            session_id="s1", user_id="u1",
            created_at=datetime.now(), last_accessed_at=datetime.now(),
            message_count=5,
        )
        provider._redis.get = AsyncMock(return_value=json.dumps(metadata.to_dict()))
        provider._redis.setex = AsyncMock()

        result = await provider.get_or_create_session(session_id="s1")
        assert result == "s1"

    @pytest.mark.asyncio
    async def test_get_or_create_session_id_not_found_creates(self):
        logger.info("RedisSessionProvider: get or create session id not found creates")
        provider = self._make_provider()
        provider._redis.get = AsyncMock(return_value=None)
        provider._redis.setex = AsyncMock()

        result = await provider.get_or_create_session(session_id="s-new", user_id="u1", agent_id="a1")
        assert result == "s-new"

    @pytest.mark.asyncio
    async def test_get_or_create_session_by_user_agent(self):
        logger.info("RedisSessionProvider: get or create session by user agent")
        provider = self._make_provider()
        metadata = SessionMetadata(
            session_id="existing-s", user_id="u1", agent_id="a1",
            created_at=datetime.now(), last_accessed_at=datetime.now(),
            message_count=5,
        )

        async def mock_get(key):
            if "user:" in key:
                return "existing-s"
            if "metadata" in key:
                return json.dumps(metadata.to_dict())
            return None

        provider._redis.get = AsyncMock(side_effect=mock_get)
        provider._redis.setex = AsyncMock()

        result = await provider.get_or_create_session(user_id="u1", agent_id="a1")
        assert result == "existing-s"

    @pytest.mark.asyncio
    async def test_get_or_create_session_stale_mapping(self):
        logger.info("RedisSessionProvider: get or create session stale mapping")
        provider = self._make_provider()

        call_count = 0
        async def mock_get(key):
            nonlocal call_count
            call_count += 1
            if "user:" in key:
                return "stale-session"
            return None

        provider._redis.get = AsyncMock(side_effect=mock_get)
        provider._redis.setex = AsyncMock()
        provider._redis.delete = AsyncMock()

        result = await provider.get_or_create_session(user_id="u1", agent_id="a1")
        assert result is not None

    @pytest.mark.asyncio
    async def test_get_or_create_session_exception(self):
        logger.info("RedisSessionProvider: get or create session exception")
        provider = self._make_provider()
        provider._redis.get = AsyncMock(side_effect=RuntimeError("redis down"))

        with pytest.raises(SessionConnectionError):
            await provider.get_or_create_session(session_id="s1")

    @pytest.mark.asyncio
    async def test_add_message(self):
        logger.info("RedisSessionProvider: add message")
        provider = self._make_provider()
        metadata = SessionMetadata(
            session_id="s1", created_at=datetime.now(),
            last_accessed_at=datetime.now(), message_count=5,
        )
        provider._redis.get = AsyncMock(return_value=json.dumps(metadata.to_dict()))
        provider._redis.rpush = AsyncMock()
        provider._redis.setex = AsyncMock()
        provider._redis.expire = AsyncMock()

        msg = ChatMessage(role="user", content="hello")
        await provider.add_message("s1", msg)
        provider._redis.rpush.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_message_session_not_found(self):
        logger.info("RedisSessionProvider: add message session not found")
        provider = self._make_provider()
        provider._redis.get = AsyncMock(return_value=None)

        msg = ChatMessage(role="user", content="hello")
        with pytest.raises(SessionNotFoundError):
            await provider.add_message("s1", msg)

    @pytest.mark.asyncio
    async def test_add_message_limit_error_strategy(self):
        logger.info("RedisSessionProvider: add message limit error strategy")
        provider = self._make_provider()
        provider._config.message_limit_strategy = "error"
        metadata = SessionMetadata(
            session_id="s1", created_at=datetime.now(),
            last_accessed_at=datetime.now(), message_count=100,
        )
        provider._redis.get = AsyncMock(return_value=json.dumps(metadata.to_dict()))

        msg = ChatMessage(role="user", content="hello")
        with pytest.raises(SessionMessageLimitError):
            await provider.add_message("s1", msg)

    @pytest.mark.asyncio
    async def test_add_message_sliding_window(self):
        logger.info("RedisSessionProvider: add message sliding window")
        provider = self._make_provider()
        provider._config.message_limit_strategy = "sliding_window"
        metadata = SessionMetadata(
            session_id="s1", created_at=datetime.now(),
            last_accessed_at=datetime.now(), message_count=100,
        )
        provider._redis.get = AsyncMock(return_value=json.dumps(metadata.to_dict()))
        provider._redis.ltrim = AsyncMock()
        provider._redis.llen = AsyncMock(return_value=90)
        provider._redis.rpush = AsyncMock()
        provider._redis.setex = AsyncMock()
        provider._redis.expire = AsyncMock()

        msg = ChatMessage(role="user", content="hello")
        await provider.add_message("s1", msg)
        provider._redis.ltrim.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_message_exception(self):
        logger.info("RedisSessionProvider: add message exception")
        provider = self._make_provider()
        provider._redis.get = AsyncMock(side_effect=RuntimeError("err"))

        msg = ChatMessage(role="user", content="hello")
        with pytest.raises(SessionConnectionError):
            await provider.add_message("s1", msg)

    @pytest.mark.asyncio
    async def test_get_messages(self):
        logger.info("RedisSessionProvider: get messages")
        provider = self._make_provider()
        metadata = SessionMetadata(
            session_id="s1", created_at=datetime.now(),
            last_accessed_at=datetime.now(), message_count=2,
        )
        sm1 = SessionMessage(
            message=ChatMessage(role="user", content="hi"),
            timestamp=datetime.now(), metadata={},
        )
        sm2 = SessionMessage(
            message=ChatMessage(role="assistant", content="hello"),
            timestamp=datetime.now(), metadata={},
        )

        provider._redis.get = AsyncMock(return_value=json.dumps(metadata.to_dict()))
        provider._redis.lrange = AsyncMock(
            return_value=[json.dumps(sm1.to_dict()), json.dumps(sm2.to_dict())]
        )
        provider._redis.setex = AsyncMock()

        messages = await provider.get_messages("s1")
        assert len(messages) == 2
        assert messages[0].role == "user"

    @pytest.mark.asyncio
    async def test_get_messages_with_limit(self):
        logger.info("RedisSessionProvider: get messages with limit")
        provider = self._make_provider()
        metadata = SessionMetadata(
            session_id="s1", created_at=datetime.now(),
            last_accessed_at=datetime.now(), message_count=3,
        )
        msgs = [
            SessionMessage(
                message=ChatMessage(role="user", content=f"msg{i}"),
                timestamp=datetime.now(), metadata={},
            )
            for i in range(3)
        ]
        provider._redis.get = AsyncMock(return_value=json.dumps(metadata.to_dict()))
        provider._redis.lrange = AsyncMock(
            return_value=[json.dumps(m.to_dict()) for m in msgs]
        )
        provider._redis.setex = AsyncMock()

        messages = await provider.get_messages("s1", limit=2)
        assert len(messages) == 2

    @pytest.mark.asyncio
    async def test_get_messages_session_not_found(self):
        logger.info("RedisSessionProvider: get messages session not found")
        provider = self._make_provider()
        provider._redis.get = AsyncMock(return_value=None)
        with pytest.raises(SessionNotFoundError):
            await provider.get_messages("s1")

    @pytest.mark.asyncio
    async def test_get_messages_empty(self):
        logger.info("RedisSessionProvider: get messages empty")
        provider = self._make_provider()
        metadata = SessionMetadata(
            session_id="s1", created_at=datetime.now(),
            last_accessed_at=datetime.now(), message_count=0,
        )
        provider._redis.get = AsyncMock(return_value=json.dumps(metadata.to_dict()))
        provider._redis.lrange = AsyncMock(return_value=[])
        provider._redis.setex = AsyncMock()

        messages = await provider.get_messages("s1")
        assert messages == []

    @pytest.mark.asyncio
    async def test_get_messages_exception(self):
        logger.info("RedisSessionProvider: get messages exception")
        provider = self._make_provider()
        provider._redis.get = AsyncMock(side_effect=RuntimeError("err"))
        with pytest.raises(SessionConnectionError):
            await provider.get_messages("s1")

    @pytest.mark.asyncio
    async def test_get_session_metadata(self):
        logger.info("RedisSessionProvider: get session metadata")
        provider = self._make_provider()
        metadata = SessionMetadata(
            session_id="s1", created_at=datetime.now(),
            last_accessed_at=datetime.now(), message_count=5,
        )
        provider._redis.get = AsyncMock(return_value=json.dumps(metadata.to_dict()))

        result = await provider.get_session_metadata("s1")
        assert result is not None
        assert result.session_id == "s1"

    @pytest.mark.asyncio
    async def test_get_session_metadata_not_found(self):
        logger.info("RedisSessionProvider: get session metadata not found")
        provider = self._make_provider()
        provider._redis.get = AsyncMock(return_value=None)
        result = await provider.get_session_metadata("s1")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_session_metadata_exception(self):
        logger.info("RedisSessionProvider: get session metadata exception")
        provider = self._make_provider()
        provider._redis.get = AsyncMock(side_effect=RuntimeError("err"))
        with pytest.raises(SessionConnectionError):
            await provider.get_session_metadata("s1")

    @pytest.mark.asyncio
    async def test_clear_session(self):
        logger.info("RedisSessionProvider: clear session")
        provider = self._make_provider()
        metadata = SessionMetadata(
            session_id="s1", created_at=datetime.now(),
            last_accessed_at=datetime.now(), message_count=10,
        )
        provider._redis.delete = AsyncMock()
        provider._redis.get = AsyncMock(return_value=json.dumps(metadata.to_dict()))
        provider._redis.setex = AsyncMock()

        result = await provider.clear_session("s1")
        assert result is True

    @pytest.mark.asyncio
    async def test_clear_session_no_metadata(self):
        logger.info("RedisSessionProvider: clear session no metadata")
        provider = self._make_provider()
        provider._redis.delete = AsyncMock()
        provider._redis.get = AsyncMock(return_value=None)

        result = await provider.clear_session("s1")
        assert result is True

    @pytest.mark.asyncio
    async def test_clear_session_exception(self):
        logger.info("RedisSessionProvider: clear session exception")
        provider = self._make_provider()
        provider._redis.delete = AsyncMock(side_effect=RuntimeError("err"))
        with pytest.raises(SessionConnectionError):
            await provider.clear_session("s1")

    @pytest.mark.asyncio
    async def test_delete_session(self):
        logger.info("RedisSessionProvider: delete session")
        provider = self._make_provider()
        metadata = SessionMetadata(
            session_id="s1", user_id="u1", agent_id="a1",
            created_at=datetime.now(), last_accessed_at=datetime.now(),
            message_count=5,
        )
        provider._redis.get = AsyncMock(return_value=json.dumps(metadata.to_dict()))
        provider._redis.delete = AsyncMock()

        result = await provider.delete_session("s1")
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_session_no_metadata(self):
        logger.info("RedisSessionProvider: delete session no metadata")
        provider = self._make_provider()
        provider._redis.get = AsyncMock(return_value=None)
        provider._redis.delete = AsyncMock()

        result = await provider.delete_session("s1")
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_session_exception(self):
        logger.info("RedisSessionProvider: delete session exception")
        provider = self._make_provider()
        provider._redis.get = AsyncMock(side_effect=RuntimeError("err"))
        with pytest.raises(SessionConnectionError):
            await provider.delete_session("s1")

    @pytest.mark.asyncio
    async def test_close(self):
        logger.info("RedisSessionProvider: close")
        provider = self._make_provider()
        provider._pool = AsyncMock()
        provider._redis = AsyncMock()
        with patch("orchestrator.config.settings") as mock_settings:
            mock_settings.shared_services_enabled = False
            await provider.close()
            assert provider._initialized is False

    @pytest.mark.asyncio
    async def test_close_shared_services(self):
        logger.info("RedisSessionProvider: close shared services")
        provider = self._make_provider()
        with patch("orchestrator.config.settings") as mock_settings:
            mock_settings.shared_services_enabled = True
            await provider.close()
            assert provider._initialized is True

    @pytest.mark.asyncio
    async def test_close_not_initialized(self):
        logger.info("RedisSessionProvider: close not initialized")
        provider = self._make_provider()
        provider._initialized = False
        provider._redis = None
        await provider.close()

    def test_initialize_disabled(self):
        logger.info("RedisSessionProvider: initialize disabled")
        config = self._make_config(enabled=False)
        provider = self._make_provider(config)
        assert provider._redis is not None or provider._initialized is True

    def test_initialize_not_configured(self):
        logger.info("RedisSessionProvider: initialize not configured")
        config = self._make_config(enabled=True, configured=False)
        with patch("orchestrator.session.providers.redis.redis") as mock_redis:
            mock_redis.ConnectionPool.return_value = MagicMock()
            mock_redis.Redis.return_value = AsyncMock()
            from orchestrator.session.providers.redis import RedisSessionProvider

            provider = RedisSessionProvider(config, auto_initialize=True)
            assert provider._redis is None
