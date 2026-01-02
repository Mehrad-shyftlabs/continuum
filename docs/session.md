# Session Module

Short-term conversation history management with Redis.

## Overview

- **SessionClient**: Session management interface
- **Conversation History**: Message storage and retrieval
- **Auto-creation**: Sessions created on first use
- **TTL Management**: Configurable session expiration

## SessionClient

```python
from orchestrator.session import SessionClient, SessionConfig

# Initialize
config = SessionConfig(
    enabled=True,
    redis_host="localhost",
    redis_port=6380,
)

client = SessionClient(session_config=config)

# Get or create session
session_id = await client.get_or_create_session(
    user_id="user-123",
    agent_id="agent-456",
)

# Add message
await client.add_message(
    session_id=session_id,
    message=ChatMessage(role="user", content="Hello!"),
    store_in_memory=True,  # Also store in long-term memory
)

# Get conversation history
history = await client.get_conversation_history(
    session_id=session_id,
    limit=50,
)
```

## Configuration

```python
from orchestrator.session import SessionConfig

config = SessionConfig(
    enabled=True,
    redis_host="localhost",
    redis_port=6380,
    ttl_seconds=86400 * 7,  # 7 days
    max_messages=1000,
)
```

## Session Metadata

```python
# Get session metadata
metadata = await client.get_session_metadata(session_id)

# Update metadata
metadata.custom["key"] = "value"
await client.update_session_metadata(session_id, metadata)
```

## Integration with Memory

Sessions automatically integrate with long-term memory when `store_in_memory=True`:

```python
await client.add_message(
    session_id=session_id,
    message=message,
    store_in_memory=True,  # Stores in mem0
    metadata={"run_id": run_id},  # For RUN-scoped isolation
)
```

## Types

- `Session`: Session object
- `SessionMetadata`: Session metadata
- `SessionMessage`: Message in session

## Exceptions

- `SessionError`: Base session error
- `SessionConfigurationError`: Configuration issues
- `SessionConnectionError`: Redis connection failures
- `SessionNotFoundError`: Session doesn't exist
- `SessionMessageLimitError`: Message limit exceeded
