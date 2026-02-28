"""
Pytest tests for the Session module.

Tests session functionality with the new provider abstraction pattern.

Usage:
    pytest tests/test_session.py -v
    pytest tests/test_session.py::test_basic_session_operations -v
"""

import os
import pytest
from orchestrator.config import settings
from orchestrator.llm.types import ChatMessage
from orchestrator.session import (
    SessionClient,
    SessionConfig,
    get_global_session_client,
    initialize_global_session_client,
)
from orchestrator.session.exceptions import (
    SessionConnectionError,
    SessionMessageLimitError,
    SessionNotFoundError,
    SessionNotEnabledError,
)
from orchestrator.session.providers import create_provider, list_providers
import logging

logger = logging.getLogger(__name__)


# Pytest marker for tests that require Redis
pytestmark = pytest.mark.asyncio


# Test constants
TEST_USER_ID = "test-user-session-123"
TEST_AGENT_ID = "test-agent-session-456"


@pytest.fixture
def session_config():
    """Create a test session configuration using environment settings."""
    return SessionConfig(
        enabled=True,
        provider="redis",
        redis_host=os.getenv("SESSION_REDIS_HOST", settings.session_redis_host),
        redis_port=int(os.getenv("SESSION_REDIS_PORT", str(settings.session_redis_port))),
        redis_password=os.getenv("SESSION_REDIS_PASSWORD", settings.session_redis_password),
        redis_db=int(os.getenv("SESSION_REDIS_DB", str(settings.session_redis_db))),
        ttl_seconds=3600,
        max_messages=1000,
    )


@pytest.fixture
async def session_client(session_config):
    """Create a test session client."""
    # Skip if Redis is not configured or not available
    if not session_config.is_configured():
        pytest.skip("Redis not configured for testing")
    
    client = SessionClient(session_config=session_config, auto_initialize=True)
    
    # Check if client is actually enabled (Redis connection successful)
    if not client.is_enabled:
        pytest.skip("Session client not enabled (Redis connection failed)")
    
    yield client
    # Cleanup: close provider if needed
    if hasattr(client, 'provider') and client.provider:
        try:
            await client.provider.close()
        except Exception:
            pass


@pytest.fixture
async def test_session_id(session_client):
    """Create a test session and return its ID."""
    logger.info("Create a test session and return its ID")
    try:
        session_id = await session_client.get_or_create_session(
            user_id=TEST_USER_ID,
            agent_id=TEST_AGENT_ID,
        )
        yield session_id
    except SessionConnectionError as e:
        pytest.skip(f"Redis connection failed: {e}")
    finally:
        # Cleanup
        try:
            if 'session_id' in locals():
                await session_client.delete_session(session_id)
        except Exception:
            pass


async def test_provider_registry():
    """Test that providers can be listed and created."""
    logger.info("Test that providers can be listed and created")
    providers = list_providers()
    assert len(providers) > 0, "At least one provider should be available"
    assert "redis" in providers, "Redis provider should be available"
    
    # Test creating a provider
    config = SessionConfig(provider="redis")
    provider = create_provider("redis", config)
    assert provider is not None
    assert provider.provider_name == "redis"
    
    # Cleanup
    try:
        await provider.close()
    except Exception:
        pass


async def test_basic_session_operations(session_client, test_session_id):
    """Test basic session operations: create, add message, get history."""
    logger.info("Test basic session operations: create, add message, get history")
    session_id = test_session_id
    
    # Get session metadata
    metadata = await session_client.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata.user_id == TEST_USER_ID
    assert metadata.agent_id == TEST_AGENT_ID
    assert metadata.message_count == 0
    
    # Add user message
    user_message = ChatMessage(role="user", content="Hello, this is a test message!")
    await session_client.add_message(session_id, user_message, store_in_memory=False)
    
    # Add assistant message
    assistant_message = ChatMessage(
        role="assistant", content="Hello! How can I help you today?"
    )
    await session_client.add_message(session_id, assistant_message, store_in_memory=False)
    
    # Get conversation history
    messages = await session_client.get_conversation_history(session_id)
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[0].content == "Hello, this is a test message!"
    assert messages[1].role == "assistant"
    
    # Verify metadata updated
    metadata = await session_client.get_session_metadata(session_id)
    assert metadata.message_count == 2


async def test_conversation_history(session_client, test_session_id):
    """Test conversation history with multiple messages."""
    logger.info("Test conversation history with multiple messages")
    session_id = test_session_id
    
    # Add multiple messages
    conversation = [
        ("user", "What is Python?"),
        ("assistant", "Python is a high-level programming language."),
        ("user", "What are its main features?"),
        ("assistant", "Python features include simplicity, readability, and a large standard library."),
        ("user", "Thanks!"),
        ("assistant", "You're welcome!"),
    ]
    
    for role, content in conversation:
        message = ChatMessage(role=role, content=content)
        await session_client.add_message(session_id, message, store_in_memory=False)
    
    # Get full history
    messages = await session_client.get_conversation_history(session_id)
    assert len(messages) == len(conversation)
    
    # Verify conversation flow
    for i, (expected_role, expected_content) in enumerate(conversation):
        assert messages[i].role == expected_role
        assert messages[i].content == expected_content
    
    # Test limit
    limited_messages = await session_client.get_conversation_history(session_id, limit=3)
    assert len(limited_messages) == 3
    # Should get last 3 messages
    assert limited_messages[0].content == conversation[-3][1]


async def test_tool_calls_in_session(session_client, test_session_id):
    """Test storing tool calls and tool results in session."""
    logger.info("Test storing tool calls and tool results in session")
    from orchestrator.llm.types import FunctionCall, ToolCall
    
    session_id = test_session_id
    
    # Add message with tool calls
    tool_call = ToolCall(
        id="call_123",
        type="function",
        function=FunctionCall(
            name="get_weather",
            arguments='{"location": "San Francisco"}',
        ),
    )
    
    assistant_message = ChatMessage(
        role="assistant",
        content=None,
        tool_calls=[tool_call],
    )
    
    await session_client.add_message(session_id, assistant_message, store_in_memory=False)
    
    # Add tool result
    tool_result = ChatMessage(
        role="tool",
        content='{"temperature": 72, "condition": "sunny"}',
        tool_call_id="call_123",
    )
    
    await session_client.add_message(session_id, tool_result, store_in_memory=False)
    
    # Get history
    messages = await session_client.get_conversation_history(session_id)
    
    # Verify tool call
    assistant_msg = messages[0]
    assert assistant_msg.tool_calls is not None
    assert len(assistant_msg.tool_calls) == 1
    assert assistant_msg.tool_calls[0].function.name == "get_weather"
    
    # Verify tool result
    tool_msg = messages[1]
    assert tool_msg.role == "tool"
    assert tool_msg.tool_call_id == "call_123"


async def test_session_clear(session_client, test_session_id):
    """Test clearing session messages."""
    logger.info("Test clearing session messages")
    session_id = test_session_id
    
    # Add multiple messages
    for i in range(5):
        message = ChatMessage(role="user", content=f"Message {i}")
        await session_client.add_message(session_id, message, store_in_memory=False)
    
    # Verify messages exist
    messages = await session_client.get_conversation_history(session_id)
    assert len(messages) == 5
    
    # Clear session
    await session_client.clear_session(session_id)
    
    # Verify messages are gone
    messages_after = await session_client.get_conversation_history(session_id)
    assert len(messages_after) == 0
    
    # Verify metadata still exists
    metadata = await session_client.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata.message_count == 0


async def test_session_delete(session_client):
    """Test deleting a session completely."""
    # Create session
    logger.info("Test deleting a session completely")
    session_id = await session_client.get_or_create_session(
        user_id=TEST_USER_ID,
        agent_id=TEST_AGENT_ID,
    )
    
    # Add messages
    message = ChatMessage(role="user", content="Test message")
    await session_client.add_message(session_id, message, store_in_memory=False)
    
    # Delete session
    result = await session_client.delete_session(session_id)
    assert result is True
    
    # Verify session is gone
    with pytest.raises(SessionNotFoundError):
        await session_client.get_conversation_history(session_id)
    
    # Verify metadata is gone
    metadata = await session_client.get_session_metadata(session_id)
    assert metadata is None


async def test_session_errors(session_client):
    """Test error handling in session operations."""
    # Test getting non-existent session
    logger.info("Test error handling in session operations")
    with pytest.raises(SessionNotFoundError):
        await session_client.get_conversation_history("non-existent-session-id")
    
    # Test adding message to non-existent session
    with pytest.raises(SessionNotFoundError):
        await session_client.add_message(
            "non-existent-session-id",
            ChatMessage(role="user", content="test"),
        )


async def test_session_metadata(session_client, test_session_id):
    """Test session metadata operations."""
    logger.info("Test session metadata operations")
    session_id = test_session_id
    
    # Get metadata
    metadata = await session_client.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata.session_id == session_id
    assert metadata.user_id == TEST_USER_ID
    assert metadata.agent_id == TEST_AGENT_ID
    assert metadata.message_count == 0
    
    # Add message and verify count updates
    initial_count = metadata.message_count
    
    await session_client.add_message(
        session_id,
        ChatMessage(role="user", content="Test message"),
        store_in_memory=False,
    )
    
    # Get updated metadata
    updated_metadata = await session_client.get_session_metadata(session_id)
    assert updated_metadata is not None
    assert updated_metadata.message_count == initial_count + 1


async def test_session_not_enabled():
    """Test that operations fail when sessions are disabled."""
    logger.info("Test that operations fail when sessions are disabled")
    config = SessionConfig(enabled=False, provider="redis")
    client = SessionClient(session_config=config, auto_initialize=True)
    
    with pytest.raises(SessionNotEnabledError):
        await client.get_or_create_session(user_id=TEST_USER_ID)
    
    # Cleanup
    if hasattr(client, 'provider') and client.provider:
        try:
            await client.provider.close()
        except Exception:
            pass


async def test_update_session_metadata(session_client, test_session_id):
    """Test updating session metadata."""
    logger.info("Test updating session metadata")
    from orchestrator.session.types import SessionMetadata
    from datetime import datetime

    session_id = test_session_id
    
    # Get current metadata
    metadata = await session_client.get_session_metadata(session_id)
    assert metadata is not None
    
    # Update custom metadata
    metadata.custom["test_key"] = "test_value"
    
    # Update metadata
    result = await session_client.update_session_metadata(session_id, metadata)
    assert result is True
    
    # Verify update
    updated_metadata = await session_client.get_session_metadata(session_id)
    assert updated_metadata is not None
    assert updated_metadata.custom.get("test_key") == "test_value"


async def test_provider_abstraction(session_config):
    """Test that provider abstraction works correctly."""
    # Skip if Redis is not configured
    logger.info("Test that provider abstraction works correctly")
    if not session_config.is_configured():
        pytest.skip("Redis not configured for testing")
    
    # Create provider directly
    provider = create_provider("redis", session_config)
    
    try:
        assert provider.provider_name == "redis"
        
        # Check if provider initialized successfully
        if not provider.is_initialized:
            pytest.skip("Provider not initialized (Redis connection failed)")
        
        # Test provider methods
        session_id = await provider.get_or_create_session(
            user_id=TEST_USER_ID,
            agent_id=TEST_AGENT_ID,
        )
        assert session_id is not None
        
        # Add message
        message = ChatMessage(role="user", content="Test")
        await provider.add_message(session_id, message)
        
        # Get messages
        messages = await provider.get_messages(session_id)
        assert len(messages) == 1
        
        # Get metadata
        metadata = await provider.get_session_metadata(session_id)
        assert metadata is not None
        
        # Cleanup
        await provider.delete_session(session_id)
    finally:
        try:
            await provider.close()
        except Exception:
            pass
