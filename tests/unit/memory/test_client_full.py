"""Comprehensive tests for memory/client.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.memory.client import MemoryClient
from orchestrator.memory.config import MemoryConfig
import logging

logger = logging.getLogger(__name__)


class TestMemoryClientInit:
    def test_default_init(self):
        logger.info("MemoryClientInit: default init")
        config = MemoryConfig(enabled=False)
        client = MemoryClient(config=config)
        assert client is not None

    def test_is_enabled_false(self):
        logger.info("MemoryClientInit: is enabled false")
        config = MemoryConfig(enabled=False)
        client = MemoryClient(config=config)
        assert client.is_enabled is False


class TestMemoryClientOperations:
    def _make_client(self):
        config = MemoryConfig(enabled=True)
        client = MemoryClient.__new__(MemoryClient)
        client._config = config
        client._provider = MagicMock()
        client._initialized = True
        client._provider.is_initialized = True
        return client

    @pytest.mark.asyncio
    async def test_add(self):
        logger.info("MemoryClientOperations: add")
        client = self._make_client()
        mock_result = MagicMock()
        mock_result.message = "ok"
        client._provider.add = AsyncMock(return_value=mock_result)
        result = await client.add("test message", user_id="u1")
        assert result is not None

    @pytest.mark.asyncio
    async def test_search(self):
        logger.info("MemoryClientOperations: search")
        client = self._make_client()
        mock_result = MagicMock()
        mock_result.results = []
        client._provider.search = AsyncMock(return_value=mock_result)
        result = await client.search("query", user_id="u1")
        assert result is not None

    @pytest.mark.asyncio
    async def test_get_all(self):
        logger.info("MemoryClientOperations: get all")
        client = self._make_client()
        mock_result = MagicMock()
        mock_result.results = []
        client._provider.get_all = AsyncMock(return_value=mock_result)
        result = await client.get_all(user_id="u1")
        assert result is not None

    @pytest.mark.asyncio
    async def test_delete(self):
        logger.info("MemoryClientOperations: delete")
        client = self._make_client()
        client._provider.delete = AsyncMock(return_value=True)
        result = await client.delete("mem-123")
        assert result is True

    @pytest.mark.asyncio
    async def test_close(self):
        logger.info("MemoryClientOperations: close")
        client = self._make_client()
        client._provider.close = AsyncMock()
        await client.close()
