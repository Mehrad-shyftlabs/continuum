"""
Session Manager - Redis backend for short-term conversation history.

Manages session creation, message storage, and retrieval using Redis.
Uses Redis Lists for efficient message storage and JSON for metadata.
"""

import json
import threading
from datetime import datetime
from typing import Any

from orchestrator.logging import get_logger
from orchestrator.session.config import SessionConfig
from orchestrator.session.exceptions import (
    SessionConnectionError,
    SessionMessageLimitError,
    SessionNotEnabledError,
    SessionNotFoundError,
)
from orchestrator.session.types import (
    ChatMessage,
    SessionMessage,
    SessionMetadata,
    generate_session_id,
)

logger = get_logger(__name__)

# Global manager state
_global_lock = threading.Lock()
_global_manager: "SessionManager | None" = None
_initialized = False


class SessionManager:
    """
    Manages sessions using Redis for short-term conversation history.

    Features:
        - Auto-create sessions on first use
        - Store complete conversation history (user, assistant, tool calls, tool results)
        - Configurable TTL and message limits
        - Efficient Redis storage using Lists and JSON
        - Thread-safe operations

    Example:
        ```python
        from orchestrator.session import SessionManager

        manager = SessionManager()

        # Create or get session
        session_id = await manager.get_or_create_session(
            user_id="user-123",
            agent_id="agent-456"
        )

        # Add message
        await manager.add_message(
            session_id=session_id,
            message=ChatMessage(role="user", content="Hello")
        )

        # Get conversation history
        messages = await manager.get_messages(session_id)
        ```
    """

    def __init__(
        self,
        config: SessionConfig | None = None,
        auto_initialize: bool = True,
    ):
        """
        Initialize the Session Manager.

        Args:
            config: Optional session configuration. Uses global settings if not provided.
            auto_initialize: Whether to initialize the Redis connection immediately.
        """
        self._config = config or SessionConfig()
        self._redis: Any = None  # redis.Redis instance
        self._initialized = False
        self._lock = threading.Lock()

        if auto_initialize:
            self.initialize()

    @property
    def config(self) -> SessionConfig:
        """Get the current configuration."""
        return self._config

    @property
    def is_enabled(self) -> bool:
        """Check if sessions are enabled and initialized."""
        return self._config.enabled and self._initialized and self._redis is not None

    def initialize(self) -> bool:
        """
        Initialize the Redis connection.

        Thread-safe initialization that only runs once.

        Returns:
            True if initialization was successful, False otherwise.
        """
        with self._lock:
            if self._initialized:
                return self._redis is not None

            if not self._config.enabled:
                logger.info("Sessions are disabled. Set SESSION_ENABLED=true to enable.")
                self._initialized = True
                return False

            if not self._config.is_configured():
                logger.warning(
                    "Session not properly configured. Check required settings: "
                    "SESSION_REDIS_HOST, SESSION_REDIS_PORT"
                )
                self._initialized = True
                return False

            try:
                import redis.asyncio as redis

                # Create connection pool for better performance
                self._pool = redis.ConnectionPool(
                    host=self._config.redis_host,
                    port=self._config.redis_port,
                    password=self._config.redis_password,
                    db=self._config.redis_db,
                    max_connections=self._config.redis_max_connections,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                )

                # Create async Redis connection using pool
                self._redis = redis.Redis(
                    connection_pool=self._pool,
                    decode_responses=True,
                )

                self._initialized = True
                logger.info(
                    "Session Manager initialized successfully",
                    extra={
                        "redis_host": self._config.redis_host,
                        "redis_port": self._config.redis_port,
                        "ttl_seconds": self._config.ttl_seconds,
                        "max_messages": self._config.max_messages,
                        "max_connections": self._config.redis_max_connections,
                    },
                )
                return True

            except ImportError:
                logger.error("redis package not installed. Run: pip install redis")
                self._initialized = True
                return False
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Failed to initialize Session Manager: {error_msg}")
                self._initialized = True
                return False

    def _ensure_enabled(self) -> None:
        """Raise error if sessions are not enabled."""
        if not self.is_enabled:
            raise SessionNotEnabledError(
                "Session operations require sessions to be enabled. "
                "Set SESSION_ENABLED=true in your environment."
            )

    def _get_session_key(self, session_id: str) -> str:
        """Get Redis key for session messages."""
        return f"{self._config.key_prefix}:{session_id}:messages"

    def _get_metadata_key(self, session_id: str) -> str:
        """Get Redis key for session metadata."""
        return f"{self._config.key_prefix}:{session_id}:metadata"

    async def get_or_create_session(
        self,
        session_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
    ) -> str:
        """
        Get existing session or create a new one.

        Args:
            session_id: Optional session ID. If not provided, generates a new UUID.
            user_id: Optional user identifier.
            agent_id: Optional agent identifier.

        Returns:
            Session ID (existing or newly created).

        Raises:
            SessionNotEnabledError: If sessions are disabled.
            SessionConnectionError: If Redis connection fails.
        """
        self._ensure_enabled()

        # Generate session ID if not provided
        if not session_id:
            session_id = generate_session_id()

        try:
            # Check if session exists
            metadata_key = self._get_metadata_key(session_id)
            metadata_json = await self._redis.get(metadata_key)

            if metadata_json:
                # Session exists, update last_accessed_at
                metadata = SessionMetadata.from_dict(json.loads(metadata_json))
                metadata.last_accessed_at = datetime.now()
                await self._redis.setex(
                    metadata_key,
                    self._config.ttl_seconds,
                    json.dumps(metadata.to_dict()),
                )
                logger.debug(f"Retrieved existing session: {session_id}")
            else:
                # Create new session
                metadata = SessionMetadata(
                    session_id=session_id,
                    user_id=user_id,
                    agent_id=agent_id,
                    created_at=datetime.now(),
                    last_accessed_at=datetime.now(),
                    message_count=0,
                )
                await self._redis.setex(
                    metadata_key,
                    self._config.ttl_seconds,
                    json.dumps(metadata.to_dict()),
                )
                logger.info(
                    f"Created new session: {session_id}",
                    extra={"user_id": user_id, "agent_id": agent_id},
                )

            return session_id

        except Exception as e:
            logger.error(f"Failed to get or create session: {e}")
            raise SessionConnectionError(
                f"Failed to get or create session: {str(e)}",
                session_id=session_id,
                original_error=e,
            ) from e

    async def add_message(
        self,
        session_id: str,
        message: ChatMessage,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Add a message to the session.

        Args:
            session_id: Session ID.
            message: Chat message to add.
            metadata: Additional metadata for the message.

        Raises:
            SessionNotEnabledError: If sessions are disabled.
            SessionNotFoundError: If session doesn't exist.
            SessionMessageLimitError: If message limit is exceeded (when strategy='error').
            SessionConnectionError: If Redis operation fails.
        """
        self._ensure_enabled()

        try:
            # Check if session exists
            metadata_key = self._get_metadata_key(session_id)
            metadata_json = await self._redis.get(metadata_key)

            if not metadata_json:
                raise SessionNotFoundError(
                    f"Session not found: {session_id}",
                    session_id=session_id,
                )

            # Update session metadata
            session_metadata = SessionMetadata.from_dict(json.loads(metadata_json))
            session_metadata.last_accessed_at = datetime.now()

            messages_key = self._get_session_key(session_id)

            # Check message limit and apply strategy
            if session_metadata.message_count >= self._config.max_messages:
                if self._config.message_limit_strategy == "error":
                    raise SessionMessageLimitError(
                        f"Session message limit exceeded: {session_metadata.message_count} >= {self._config.max_messages}",
                        session_id=session_id,
                        current_count=session_metadata.message_count,
                        max_messages=self._config.max_messages,
                    )
                else:
                    # Sliding window strategy: remove oldest messages
                    trim_count = self._config.sliding_window_trim_count

                    # Use LTRIM to remove oldest messages (keep from trim_count to end)
                    await self._redis.ltrim(messages_key, trim_count, -1)

                    # Update message count after trim
                    actual_count = await self._redis.llen(messages_key)
                    session_metadata.message_count = actual_count

                    logger.info(
                        f"Sliding window triggered: trimmed {trim_count} oldest messages",
                        extra={
                            "session_id": session_id,
                            "trimmed_count": trim_count,
                            "new_count": actual_count,
                        },
                    )

            # Increment count for the new message
            session_metadata.message_count += 1

            # Create session message with metadata
            session_message = SessionMessage(
                message=message,
                timestamp=datetime.now(),
                metadata=metadata or {},
            )

            # Store message in Redis List (append to end)
            message_json = json.dumps(session_message.to_dict())
            await self._redis.rpush(messages_key, message_json)

            # Update session metadata
            await self._redis.setex(
                metadata_key,
                self._config.ttl_seconds,
                json.dumps(session_metadata.to_dict()),
            )

            # Set TTL on messages list (same as session TTL)
            await self._redis.expire(messages_key, self._config.ttl_seconds)

            logger.debug(
                f"Added message to session: {session_id}",
                extra={"message_count": session_metadata.message_count, "role": message.role},
            )

        except SessionNotFoundError:
            raise
        except SessionMessageLimitError:
            raise
        except Exception as e:
            logger.error(f"Failed to add message to session: {e}")
            raise SessionConnectionError(
                f"Failed to add message to session: {str(e)}",
                session_id=session_id,
                original_error=e,
            ) from e

    async def get_messages(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[ChatMessage]:
        """
        Get conversation history for a session.

        Args:
            session_id: Session ID.
            limit: Optional limit on number of messages to retrieve.

        Returns:
            List of ChatMessage objects in chronological order.

        Raises:
            SessionNotEnabledError: If sessions are disabled.
            SessionNotFoundError: If session doesn't exist.
            SessionConnectionError: If Redis operation fails.
        """
        self._ensure_enabled()

        try:
            # Check if session exists
            metadata_key = self._get_metadata_key(session_id)
            metadata_json = await self._redis.get(metadata_key)

            if not metadata_json:
                raise SessionNotFoundError(
                    f"Session not found: {session_id}",
                    session_id=session_id,
                )

            # Get messages from Redis List
            messages_key = self._get_session_key(session_id)
            message_jsons = await self._redis.lrange(messages_key, 0, -1)

            if not message_jsons:
                return []

            # Parse messages
            session_messages = [
                SessionMessage.from_dict(json.loads(msg_json)) for msg_json in message_jsons
            ]

            # Convert to ChatMessage list
            messages = [sm.message for sm in session_messages]

            # Apply limit if specified
            if limit and limit > 0:
                messages = messages[-limit:]  # Get last N messages

            # Update last_accessed_at
            session_metadata = SessionMetadata.from_dict(json.loads(metadata_json))
            session_metadata.last_accessed_at = datetime.now()
            await self._redis.setex(
                metadata_key,
                self._config.ttl_seconds,
                json.dumps(session_metadata.to_dict()),
            )

            logger.debug(
                f"Retrieved {len(messages)} messages from session: {session_id}",
                extra={"limit": limit},
            )

            return messages

        except SessionNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to get messages from session: {e}")
            raise SessionConnectionError(
                f"Failed to get messages from session: {str(e)}",
                session_id=session_id,
                original_error=e,
            ) from e

    async def get_session_metadata(self, session_id: str) -> SessionMetadata | None:
        """
        Get session metadata.

        Args:
            session_id: Session ID.

        Returns:
            SessionMetadata if found, None otherwise.

        Raises:
            SessionNotEnabledError: If sessions are disabled.
            SessionConnectionError: If Redis operation fails.
        """
        self._ensure_enabled()

        try:
            metadata_key = self._get_metadata_key(session_id)
            metadata_json = await self._redis.get(metadata_key)

            if not metadata_json:
                return None

            return SessionMetadata.from_dict(json.loads(metadata_json))

        except Exception as e:
            logger.error(f"Failed to get session metadata: {e}")
            raise SessionConnectionError(
                f"Failed to get session metadata: {str(e)}",
                session_id=session_id,
                original_error=e,
            ) from e

    async def clear_session(self, session_id: str) -> bool:
        """
        Clear all messages from a session (but keep metadata).

        Args:
            session_id: Session ID.

        Returns:
            True if cleared successfully.

        Raises:
            SessionNotEnabledError: If sessions are disabled.
            SessionConnectionError: If Redis operation fails.
        """
        self._ensure_enabled()

        try:
            messages_key = self._get_session_key(session_id)
            await self._redis.delete(messages_key)

            # Reset message count in metadata
            metadata_key = self._get_metadata_key(session_id)
            metadata_json = await self._redis.get(metadata_key)
            if metadata_json:
                session_metadata = SessionMetadata.from_dict(json.loads(metadata_json))
                session_metadata.message_count = 0
                session_metadata.last_accessed_at = datetime.now()
                await self._redis.setex(
                    metadata_key,
                    self._config.ttl_seconds,
                    json.dumps(session_metadata.to_dict()),
                )

            logger.info(f"Cleared session: {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to clear session: {e}")
            raise SessionConnectionError(
                f"Failed to clear session: {str(e)}",
                session_id=session_id,
                original_error=e,
            ) from e

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session completely (messages and metadata).

        Args:
            session_id: Session ID.

        Returns:
            True if deleted successfully.

        Raises:
            SessionNotEnabledError: If sessions are disabled.
            SessionConnectionError: If Redis operation fails.
        """
        self._ensure_enabled()

        try:
            messages_key = self._get_session_key(session_id)
            metadata_key = self._get_metadata_key(session_id)

            await self._redis.delete(messages_key, metadata_key)

            logger.info(f"Deleted session: {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete session: {e}")
            raise SessionConnectionError(
                f"Failed to delete session: {str(e)}",
                session_id=session_id,
                original_error=e,
            ) from e

    async def close(self) -> None:
        """
        Close Redis connections and cleanup resources.

        Respects shared_services_enabled setting - if True, doesn't close
        Redis connections (they persist as a shared service).
        """
        from orchestrator.config import settings

        if not self._initialized or self._redis is None:
            return

        # If Redis is a shared service, don't close connections
        if settings.shared_services_enabled:
            logger.debug("Redis is a shared service, skipping connection close")
            return

        with self._lock:
            try:
                # Close Redis connection pool
                if hasattr(self, "_pool") and self._pool is not None:
                    await self._pool.aclose()
                    logger.debug("Redis connection pool closed")

                # Close Redis client
                if self._redis is not None:
                    await self._redis.aclose()
                    logger.debug("Redis client closed")

                self._redis = None
                self._initialized = False
                logger.info("Session Manager closed")

            except Exception as e:
                logger.warning(f"Error closing Session Manager: {e}")
                self._initialized = False


# =============================================================================
# Global Session Manager Functions
# =============================================================================


def initialize_global_session_manager(config: SessionConfig | None = None) -> bool:
    """
    Initialize the global Session Manager.

    This should be called once at application startup. Subsequent calls
    will return the existing initialization status.

    Args:
        config: Optional configuration. Uses environment variables if not provided.

    Returns:
        True if initialization was successful.
    """
    global _global_manager, _initialized

    with _global_lock:
        if _initialized:
            return _global_manager is not None and _global_manager.is_enabled

        _global_manager = SessionManager(config=config, auto_initialize=True)
        _initialized = True

        return _global_manager.is_enabled


def get_global_session_manager() -> SessionManager:
    """
    Get the global Session Manager.

    Auto-initializes if not already initialized.

    Returns:
        The global SessionManager instance.
    """
    global _global_manager, _initialized

    if not _initialized:
        initialize_global_session_manager()

    if _global_manager is None:
        with _global_lock:
            if _global_manager is None:
                _global_manager = SessionManager(auto_initialize=True)
                _initialized = True

    return _global_manager
