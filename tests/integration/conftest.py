"""
Integration test fixtures - may require running services.
"""

import os

import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture
def has_openai_key() -> bool:
    """Check if OpenAI API key is available."""
    return bool(os.getenv("OPENAI_API_KEY"))


@pytest.fixture
def has_redis() -> bool:
    """Check if Redis is available."""
    try:
        import redis

        r = redis.Redis(
            host=os.getenv("SESSION_REDIS_HOST", "localhost"),
            port=int(os.getenv("SESSION_REDIS_PORT", "6379")),
            socket_connect_timeout=2,
        )
        r.ping()
        r.close()
        return True
    except Exception:
        return False


@pytest.fixture
def has_qdrant() -> bool:
    """Check if Qdrant is available."""
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
            timeout=2,
        )
        client.get_collections()
        client.close()
        return True
    except Exception:
        return False
