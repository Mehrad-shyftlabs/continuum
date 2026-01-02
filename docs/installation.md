# Installation

## Requirements

- Python 3.13 or higher
- Docker (for local development with services)

**Important**: Langfuse must be version 2.x (not 3.x) for LiteLLM compatibility. The SDK pins Langfuse to `>=2.57.0,<3.0.0`.

## Install from Source

```bash
# Clone the repository
git clone https://github.com/shyftlabs/continuum.git
cd continuum

# Install in development mode
pip install -e .

# Or install in production mode
pip install .
```

## Install with Optional Dependencies

```bash
# With embedding support (HuggingFace)
pip install -e ".[embeddings]"

# With Cohere embeddings
pip install -e ".[cohere]"

# With development tools
pip install -e ".[dev]"
```

## Environment Setup

Create a `.env` file in the project root:

```env
# LLM Provider API Keys (at least one required)
OPENAI_API_KEY=your-openai-key
# OR
GEMINI_API_KEY=your-gemini-key
# OR
ANTHROPIC_API_KEY=your-anthropic-key

# Langfuse Configuration (optional)
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=your-public-key
LANGFUSE_SECRET_KEY=your-secret-key
LANGFUSE_HOST=http://localhost:3000

# Memory Configuration (optional)
MEMORY_ENABLED=true
QDRANT_HOST=localhost
QDRANT_PORT=6333
MEMORY_LLM_MODEL=gpt-4o-mini
EMBEDDER_PROVIDER=openai
EMBEDDER_MODEL=text-embedding-3-small

# Session Configuration (optional)
SESSION_ENABLED=true
SESSION_REDIS_HOST=localhost
SESSION_REDIS_PORT=6380
```

## Docker Services (Development)

Start required services using Docker Compose:

```bash
docker-compose up -d
```

This starts:
- Langfuse (port 3000)
- Qdrant (port 6333)
- Redis for sessions (port 6380)
- PostgreSQL (port 5433)
- ClickHouse (port 8123)

## Verify Installation

```bash
# Run health check
python scripts/health_check.py
```

## Next Steps

- See [Agent Documentation](agent.md) to create your first agent
- Check [Configuration](core.md#configuration) for advanced setup options
