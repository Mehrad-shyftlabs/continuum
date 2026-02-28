"""
Integration tests for memory operations.

Requires Qdrant to be running and configured.

Converted from tests/test_memory.py manual test script.
"""

import os

import pytest
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


load_dotenv()

from orchestrator.config import settings
from orchestrator.memory import (
    MemoryClient,
    MemoryConfig,
    MemoryFilter,
    MemoryMetadata,
    get_global_memory_client,
    initialize_global_memory,
)
from orchestrator.memory.exceptions import (
    MemoryAddError,
    MemoryError,
    MemoryIdentifierError,
    MemoryNotEnabledError,
    MemorySearchError,
)


pytestmark = [pytest.mark.integration]

TEST_USER_ID = "test-user-integration-123"
TEST_AGENT_ID = "test-agent-integration-456"
TEST_RUN_ID = "test-run-integration-789"
TEST_SESSION_ID = "test-session-integration-abc"


@pytest.fixture
def memory_client():
    """Create a memory client, skip if not enabled."""
    client = MemoryClient()
    if not client.is_enabled:
        pytest.skip("Memory not enabled (Qdrant may not be running)")
    return client


@pytest.fixture
async def cleanup_memory(memory_client):
    """Clean up test data before and after tests."""
    try:
        await memory_client.delete_all(user_id=TEST_USER_ID)
    except Exception:
        pass
    yield
    try:
        await memory_client.delete_all(user_id=TEST_USER_ID)
    except Exception:
        pass


class TestBasicOperations:
    @pytest.mark.skipif(not settings.memory_enabled, reason="Memory not enabled")
    async def test_add_memory(self, memory_client, cleanup_memory):
        logger.info("BasicOperations: add memory")
        result = await memory_client.add(
            "User loves Python programming and prefers dark mode",
            user_id=TEST_USER_ID,
            metadata={"category": "preferences", "test": True},
        )
        assert result.message is not None
        assert len(result.results) >= 0

    @pytest.mark.skipif(not settings.memory_enabled, reason="Memory not enabled")
    async def test_search_memory(self, memory_client, cleanup_memory):
        logger.info("BasicOperations: search memory")
        await memory_client.add(
            "User loves Python programming",
            user_id=TEST_USER_ID,
        )

        search_results = await memory_client.search(
            "What does the user like?",
            user_id=TEST_USER_ID,
            limit=5,
        )
        assert search_results is not None
        assert hasattr(search_results, "results")

    @pytest.mark.skipif(not settings.memory_enabled, reason="Memory not enabled")
    async def test_get_all_memories(self, memory_client, cleanup_memory):
        logger.info("BasicOperations: get all memories")
        await memory_client.add(
            "User loves Python",
            user_id=TEST_USER_ID,
        )

        all_memories = await memory_client.get_all(user_id=TEST_USER_ID)
        assert isinstance(all_memories, list)

    @pytest.mark.skipif(not settings.memory_enabled, reason="Memory not enabled")
    async def test_delete_memory(self, memory_client, cleanup_memory):
        logger.info("BasicOperations: delete memory")
        await memory_client.add(
            "User loves Python",
            user_id=TEST_USER_ID,
        )
        all_memories = await memory_client.get_all(user_id=TEST_USER_ID)
        if all_memories:
            deleted = await memory_client.delete(all_memories[0].id, user_id=TEST_USER_ID)
            assert deleted is True


class TestErrorHandling:
    def test_memory_not_enabled_error(self):
        logger.info("ErrorHandling: memory not enabled error")
        config = MemoryConfig(enabled=False)
        memory = MemoryClient(config=config)

        with pytest.raises(MemoryNotEnabledError):
            import asyncio

            asyncio.get_event_loop().run_until_complete(
                memory.add("test", user_id=TEST_USER_ID)
            )

    @pytest.mark.skipif(not settings.memory_enabled, reason="Memory not enabled")
    async def test_missing_identifier_error(self, memory_client):
        logger.info("ErrorHandling: missing identifier error")
        config = MemoryConfig(memory_isolation="user", enabled=True)
        memory = MemoryClient(config=config)
        if memory.is_enabled:
            with pytest.raises(MemoryIdentifierError):
                await memory.add("test")


class TestEmbedderConfig:
    """Test embedder configuration generation (no external services needed)."""

    def test_openai_provider_config(self):
        logger.info("EmbedderConfig: openai provider config")
        config = MemoryConfig(
            embedder_provider="openai",
            embedder_model="text-embedding-3-small",
            embedding_dims=1536,
        )
        mem0_cfg = config.to_mem0_config()
        embedder = mem0_cfg["embedder"]
        assert embedder["provider"] == "openai"
        assert embedder["config"]["model"] == "text-embedding-3-small"

    def test_ollama_provider_config(self):
        logger.info("EmbedderConfig: ollama provider config")
        config = MemoryConfig(
            embedder_provider="ollama",
            embedder_model="nomic-embed-text",
            embedding_dims=768,
        )
        mem0_cfg = config.to_mem0_config()
        embedder = mem0_cfg["embedder"]
        assert embedder["provider"] == "ollama"
        assert "host" in embedder["config"]

    def test_azure_openai_provider_config(self):
        logger.info("EmbedderConfig: azure openai provider config")
        config = MemoryConfig(
            embedder_provider="azure_openai",
            embedder_model="my-embedding-deployment",
            embedding_dims=1536,
            embedder_api_base="https://myresource.openai.azure.com",
        )
        mem0_cfg = config.to_mem0_config()
        embedder = mem0_cfg["embedder"]
        assert embedder["provider"] == "azure_openai"
        assert embedder["config"].get("azure_endpoint") == "https://myresource.openai.azure.com"

    def test_huggingface_provider_config(self):
        logger.info("EmbedderConfig: huggingface provider config")
        config = MemoryConfig(
            embedder_provider="huggingface",
            embedder_model="BAAI/bge-small-en-v1.5",
            embedding_dims=384,
        )
        mem0_cfg = config.to_mem0_config()
        embedder = mem0_cfg["embedder"]
        assert embedder["provider"] == "huggingface"
        assert embedder["config"]["model"] == "BAAI/bge-small-en-v1.5"

    def test_cohere_provider_config(self):
        logger.info("EmbedderConfig: cohere provider config")
        config = MemoryConfig(
            embedder_provider="cohere",
            embedder_model="embed-english-v3.0",
            embedding_dims=1024,
        )
        mem0_cfg = config.to_mem0_config()
        embedder = mem0_cfg["embedder"]
        assert embedder["provider"] == "cohere"

    def test_gemini_provider_config(self):
        logger.info("EmbedderConfig: gemini provider config")
        config = MemoryConfig(
            embedder_provider="gemini",
            embedder_model="models/embedding-001",
            embedding_dims=768,
        )
        mem0_cfg = config.to_mem0_config()
        embedder = mem0_cfg["embedder"]
        assert embedder["provider"] == "gemini"

    def test_explicit_api_key_config(self):
        logger.info("EmbedderConfig: explicit api key config")
        config = MemoryConfig(
            embedder_provider="openai",
            embedder_model="text-embedding-3-small",
            embedding_dims=1536,
            embedder_api_key="test-api-key",
            embedder_api_base="https://custom.api.com/v1",
        )
        mem0_cfg = config.to_mem0_config()
        embedder = mem0_cfg["embedder"]
        assert embedder["config"].get("api_key") == "test-api-key"
        assert embedder["config"].get("api_base") == "https://custom.api.com/v1"
