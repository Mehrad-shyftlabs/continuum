# Memory Module

Long-term memory with mem0 and Qdrant vector store.

## Overview

- **MemoryClient**: High-level memory interface
- **Memory Scopes**: User, agent, run, and shared isolation
- **Semantic Search**: Vector similarity search
- **Automatic Fact Extraction**: LLM-powered memory consolidation

## MemoryClient

```python
from orchestrator.memory import MemoryClient, MemoryConfig

# Initialize with config
config = MemoryConfig(
    enabled=True,
    qdrant_host="localhost",
    qdrant_port=6333,
    memory_llm_model="gpt-4o-mini",
    embedder_provider="openai",
    embedder_model="text-embedding-3-small",
)

client = MemoryClient(config=config)

# Add memory
await client.add(
    "User prefers dark mode",
    user_id="user-123",
    metadata={"category": "preferences"}
)

# Search memories
results = await client.search(
    "What are the user's preferences?",
    user_id="user-123",
    limit=5
)
```

## Memory Scopes

Control memory isolation:

```python
from orchestrator.memory import MemoryScope

# User-scoped (default)
scope = MemoryScope.USER

# Agent-scoped
scope = MemoryScope.AGENT

# Run-scoped (session-level)
scope = MemoryScope.RUN

# Shared (all users/agents)
scope = MemoryScope.SHARED
```

## Memory Isolation Modes

Configure via `MEMORY_ISOLATION` environment variable:

- `user`: Memories isolated per user
- `agent`: Memories isolated per agent
- `run`: Memories isolated per session/run
- `shared`: All memories shared

## Configuration

```python
from orchestrator.memory import MemoryConfig

config = MemoryConfig(
    enabled=True,
    qdrant_host="localhost",
    qdrant_port=6333,
    memory_llm_model="gpt-4o-mini",
    embedder_provider="openai",
    embedder_model="text-embedding-3-small",
    embedding_dims=1536,
    memory_isolation="user",
)
```

## Supported Embedder Providers

- OpenAI (text-embedding-3-small, text-embedding-3-large)
- Azure OpenAI
- Hugging Face
- Cohere
- Google Gemini / Vertex AI
- Ollama (local)

## Types

- `MemoryEntry`: Memory entry with metadata
- `MemorySearchResult`: Search results with scores
- `MemoryAddResult`: Result of adding memory
- `MemoryScope`: Isolation scope

## Exceptions

- `MemoryError`: Base memory error
- `MemoryConfigurationError`: Configuration issues
- `MemoryConnectionError`: Connection failures
- `MemorySearchError`: Search failures
