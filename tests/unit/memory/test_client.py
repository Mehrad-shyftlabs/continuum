"""Unit tests for memory client."""

from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from orchestrator.memory.client import MemoryClient
from orchestrator.memory.config import MemoryConfig
from orchestrator.memory.exceptions import MemoryNotEnabledError
import logging

logger = logging.getLogger(__name__)


class TestMemoryClientInit:
    def test_client_initialization_disabled(self):
        logger.info("MemoryClientInit: client initialization disabled")
        config = MemoryConfig(enabled=False)
        client = MemoryClient(config=config)
        assert client.is_enabled is False

    def test_client_ensure_enabled_raises(self):
        logger.info("MemoryClientInit: client ensure enabled raises")
        config = MemoryConfig(enabled=False)
        client = MemoryClient(config=config)
        with pytest.raises(MemoryNotEnabledError):
            client._ensure_enabled()

    def test_client_build_scope_user(self):
        logger.info("MemoryClientInit: client build scope user")
        config = MemoryConfig(enabled=False)
        client = MemoryClient(config=config)
        scope = client._build_scope(user_id="u1")
        assert scope is not None


class TestMemoryClientOperations:
    @pytest.mark.asyncio
    async def test_client_add(self):
        logger.info("MemoryClientOperations: client add")
        config = MemoryConfig(enabled=False)
        client = MemoryClient(config=config)
        mock_provider = AsyncMock()
        mock_provider.add.return_value = {"message": "ok", "results": [], "relations": []}
        client._provider = mock_provider
        client._initialized = True
        client._config = MemoryConfig(enabled=True)
        result = await client.add("test memory", user_id="u1")
        assert result is not None

    @pytest.mark.asyncio
    async def test_client_search(self):
        logger.info("MemoryClientOperations: client search")
        config = MemoryConfig(enabled=False)
        client = MemoryClient(config=config)
        mock_provider = AsyncMock()
        mock_result = MagicMock()
        mock_result.results = []
        mock_result.query = "query"
        mock_result.limit = 5
        mock_result.total_results = 0
        mock_provider.search.return_value = mock_result
        client._provider = mock_provider
        client._initialized = True
        client._config = MemoryConfig(enabled=True)
        result = await client.search("query", user_id="u1")
        assert result is not None

    @pytest.mark.asyncio
    async def test_client_get_all(self):
        logger.info("MemoryClientOperations: client get all")
        config = MemoryConfig(enabled=False)
        client = MemoryClient(config=config)
        mock_provider = AsyncMock()
        mock_provider.get_all.return_value = []
        client._provider = mock_provider
        client._initialized = True
        client._config = MemoryConfig(enabled=True)
        result = await client.get_all(user_id="u1")
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_client_delete(self):
        logger.info("MemoryClientOperations: client delete")
        config = MemoryConfig(enabled=False)
        client = MemoryClient(config=config)
        mock_provider = AsyncMock()
        mock_provider.delete.return_value = True
        client._provider = mock_provider
        client._initialized = True
        client._config = MemoryConfig(enabled=True)
        result = await client.delete("mem-1")
        assert result is True

    @pytest.mark.asyncio
    async def test_client_reset(self):
        logger.info("MemoryClientOperations: client reset")
        config = MemoryConfig(enabled=False)
        client = MemoryClient(config=config)
        mock_provider = AsyncMock()
        mock_provider.reset.return_value = True
        client._provider = mock_provider
        client._initialized = True
        client._config = MemoryConfig(enabled=True)
        result = await client.reset()
        assert result is True
