"""Comprehensive tests for memory/providers/mem0.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.memory.config import MemoryConfig
from orchestrator.memory.exceptions import MemoryConfigurationError, MemoryError, MemoryUpdateError
from orchestrator.memory.types import MemoryAddResult, MemoryEntry, MemorySearchResult
import logging

logger = logging.getLogger(__name__)


class TestMem0Provider:
    @patch("orchestrator.memory.providers.mem0.MEM0_AVAILABLE", True)
    @patch("orchestrator.memory.providers.mem0.Memory")
    def _make_provider(self, mock_memory_cls):
        from orchestrator.memory.providers.mem0 import Mem0Provider

        mock_memory_cls.from_config.return_value = MagicMock()
        config = MagicMock(spec=MemoryConfig)
        config.enabled = True
        config.is_configured.return_value = True
        config.to_mem0_config.return_value = {"vector_store": {}}
        config.vector_store_provider = "qdrant"
        config.qdrant_host = "localhost"
        config.embedder_provider = "openai"
        config.embedder_model = "text-embedding-3-small"
        config.memory_isolation = "user"

        provider = Mem0Provider(config)
        return provider, mock_memory_cls

    def test_provider_name(self):
        logger.info("Mem0Provider: provider name")
        provider, _ = self._make_provider()
        assert provider.provider_name == "mem0"

    def test_is_initialized(self):
        logger.info("Mem0Provider: is initialized")
        provider, _ = self._make_provider()
        assert provider.is_initialized is True

    def test_build_identifiers(self):
        logger.info("Mem0Provider: build identifiers")
        provider, _ = self._make_provider()
        ids = provider._build_identifiers(user_id="u1", agent_id="a1")
        assert ids == {"user_id": "u1", "agent_id": "a1"}

    def test_build_identifiers_empty(self):
        logger.info("Mem0Provider: build identifiers empty")
        provider, _ = self._make_provider()
        ids = provider._build_identifiers()
        assert ids == {}

    def test_ensure_initialized_fails(self):
        logger.info("Mem0Provider: ensure initialized fails")
        provider, _ = self._make_provider()
        provider._initialized = False
        with pytest.raises(MemoryConfigurationError):
            provider._ensure_initialized()

    @pytest.mark.asyncio
    async def test_add(self):
        logger.info("Mem0Provider: add")
        provider, _ = self._make_provider()
        provider._sync_memory.add = MagicMock(return_value={"results": [], "message": "ok"})

        with patch.object(MemoryAddResult, "from_mem0_response") as mock_from:
            mock_from.return_value = MemoryAddResult(message="ok", results=[])
            result = await provider.add("test msg", user_id="u1")
            assert result.message == "ok"

    @pytest.mark.asyncio
    async def test_add_exception(self):
        logger.info("Mem0Provider: add exception")
        provider, _ = self._make_provider()
        provider._sync_memory.add = MagicMock(side_effect=Exception("boom"))

        result = await provider.add("test msg", user_id="u1")
        assert result.message == "Memory operation failed"

    @pytest.mark.asyncio
    async def test_add_with_metadata_and_prompt(self):
        logger.info("Mem0Provider: add with metadata and prompt")
        provider, _ = self._make_provider()
        provider._sync_memory.add = MagicMock(return_value={"results": [], "message": "ok"})

        with patch.object(MemoryAddResult, "from_mem0_response") as mock_from:
            mock_from.return_value = MemoryAddResult(message="ok", results=[])
            result = await provider.add(
                "test", user_id="u1",
                metadata={"k": "v"}, custom_prompt="extract facts",
            )
            call_kwargs = provider._sync_memory.add.call_args
            assert "metadata" in call_kwargs.kwargs or "metadata" in str(call_kwargs)

    @pytest.mark.asyncio
    async def test_search(self):
        logger.info("Mem0Provider: search")
        provider, _ = self._make_provider()
        provider._sync_memory.search = MagicMock(return_value={"results": []})

        with patch.object(MemorySearchResult, "from_mem0_response") as mock_from:
            mock_from.return_value = MemorySearchResult(
                results=[], query="q", limit=5, total_results=0,
            )
            result = await provider.search("query", user_id="u1")
            assert result.total_results == 0

    @pytest.mark.asyncio
    async def test_search_with_filters(self):
        logger.info("Mem0Provider: search with filters")
        provider, _ = self._make_provider()
        provider._sync_memory.search = MagicMock(return_value={"results": []})

        with patch.object(MemorySearchResult, "from_mem0_response") as mock_from:
            mock_from.return_value = MemorySearchResult(
                results=[], query="q", limit=5, total_results=0,
            )
            await provider.search("q", filters={"category": "work"})

    @pytest.mark.asyncio
    async def test_search_exception(self):
        logger.info("Mem0Provider: search exception")
        provider, _ = self._make_provider()
        provider._sync_memory.search = MagicMock(side_effect=Exception("err"))
        result = await provider.search("q")
        assert result.total_results == 0

    @pytest.mark.asyncio
    async def test_get(self):
        logger.info("Mem0Provider: get")
        provider, _ = self._make_provider()
        provider._sync_memory.get = MagicMock(return_value={"id": "m1", "memory": "fact"})

        with patch.object(MemoryEntry, "from_mem0_result") as mock_from:
            mock_from.return_value = MagicMock(spec=MemoryEntry)
            result = await provider.get("m1")
            assert result is not None

    @pytest.mark.asyncio
    async def test_get_not_found(self):
        logger.info("Mem0Provider: get not found")
        provider, _ = self._make_provider()
        provider._sync_memory.get = MagicMock(return_value=None)
        result = await provider.get("m1")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_exception(self):
        logger.info("Mem0Provider: get exception")
        provider, _ = self._make_provider()
        provider._sync_memory.get = MagicMock(side_effect=Exception("err"))
        result = await provider.get("m1")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_all(self):
        logger.info("Mem0Provider: get all")
        provider, _ = self._make_provider()
        provider._sync_memory.get_all = MagicMock(
            return_value={"results": [{"id": "m1", "memory": "fact"}]}
        )

        with patch.object(MemoryEntry, "from_mem0_result") as mock_from:
            mock_from.return_value = MagicMock(spec=MemoryEntry)
            result = await provider.get_all(user_id="u1")
            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_all_exception(self):
        logger.info("Mem0Provider: get all exception")
        provider, _ = self._make_provider()
        provider._sync_memory.get_all = MagicMock(side_effect=Exception("err"))
        result = await provider.get_all()
        assert result == []

    @pytest.mark.asyncio
    async def test_delete(self):
        logger.info("Mem0Provider: delete")
        provider, _ = self._make_provider()
        provider._sync_memory.delete = MagicMock()
        result = await provider.delete("m1")
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_exception(self):
        logger.info("Mem0Provider: delete exception")
        provider, _ = self._make_provider()
        provider._sync_memory.delete = MagicMock(side_effect=Exception("err"))
        result = await provider.delete("m1")
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_all(self):
        logger.info("Mem0Provider: delete all")
        provider, _ = self._make_provider()
        provider._sync_memory.delete_all = MagicMock()
        result = await provider.delete_all(user_id="u1")
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_all_exception(self):
        logger.info("Mem0Provider: delete all exception")
        provider, _ = self._make_provider()
        provider._sync_memory.delete_all = MagicMock(side_effect=Exception("err"))
        result = await provider.delete_all()
        assert result is False

    @pytest.mark.asyncio
    async def test_update(self):
        logger.info("Mem0Provider: update")
        provider, _ = self._make_provider()
        provider._sync_memory.update = MagicMock(return_value={"id": "m1", "memory": "updated"})

        with patch.object(MemoryEntry, "from_mem0_result") as mock_from:
            mock_from.return_value = MagicMock(spec=MemoryEntry)
            result = await provider.update("m1", "new data")
            assert result is not None

    @pytest.mark.asyncio
    async def test_update_returns_none(self):
        logger.info("Mem0Provider: update returns none")
        provider, _ = self._make_provider()
        provider._sync_memory.update = MagicMock(return_value=None)
        with pytest.raises(MemoryUpdateError):
            await provider.update("m1", "data")

    @pytest.mark.asyncio
    async def test_update_exception(self):
        logger.info("Mem0Provider: update exception")
        provider, _ = self._make_provider()
        provider._sync_memory.update = MagicMock(side_effect=RuntimeError("err"))
        with pytest.raises(MemoryUpdateError):
            await provider.update("m1", "data")

    @pytest.mark.asyncio
    async def test_history(self):
        logger.info("Mem0Provider: history")
        provider, _ = self._make_provider()
        provider._sync_memory.history = MagicMock(return_value=[{"v": 1}])
        result = await provider.history("m1")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_history_exception(self):
        logger.info("Mem0Provider: history exception")
        provider, _ = self._make_provider()
        provider._sync_memory.history = MagicMock(side_effect=Exception("err"))
        result = await provider.history("m1")
        assert result == []

    @pytest.mark.asyncio
    async def test_reset(self):
        logger.info("Mem0Provider: reset")
        provider, _ = self._make_provider()
        provider._sync_memory.reset = MagicMock()
        result = await provider.reset()
        assert result is True

    @pytest.mark.asyncio
    async def test_reset_exception(self):
        logger.info("Mem0Provider: reset exception")
        provider, _ = self._make_provider()
        provider._sync_memory.reset = MagicMock(side_effect=Exception("err"))
        with pytest.raises(MemoryError):
            await provider.reset()

    @pytest.mark.asyncio
    async def test_close(self):
        logger.info("Mem0Provider: close")
        provider, _ = self._make_provider()
        await provider.close()
        assert provider._initialized is False
        assert provider._sync_memory is None

    # Sync methods
    def test_add_sync(self):
        logger.info("Mem0Provider: add sync")
        provider, _ = self._make_provider()
        provider._sync_memory.add = MagicMock(return_value={"results": [], "message": "ok"})
        with patch.object(MemoryAddResult, "from_mem0_response") as mock_from:
            mock_from.return_value = MemoryAddResult(message="ok", results=[])
            result = provider.add_sync("msg", user_id="u1")
            assert result.message == "ok"

    def test_add_sync_exception(self):
        logger.info("Mem0Provider: add sync exception")
        provider, _ = self._make_provider()
        provider._sync_memory.add = MagicMock(side_effect=Exception("err"))
        result = provider.add_sync("msg")
        assert result.message == "Memory operation failed"

    def test_search_sync(self):
        logger.info("Mem0Provider: search sync")
        provider, _ = self._make_provider()
        provider._sync_memory.search = MagicMock(return_value={"results": []})
        with patch.object(MemorySearchResult, "from_mem0_response") as mock_from:
            mock_from.return_value = MemorySearchResult(
                results=[], query="q", limit=5, total_results=0,
            )
            result = provider.search_sync("q")
            assert result.total_results == 0

    def test_search_sync_exception(self):
        logger.info("Mem0Provider: search sync exception")
        provider, _ = self._make_provider()
        provider._sync_memory.search = MagicMock(side_effect=Exception("err"))
        result = provider.search_sync("q")
        assert result.total_results == 0

    def test_get_sync(self):
        logger.info("Mem0Provider: get sync")
        provider, _ = self._make_provider()
        provider._sync_memory.get = MagicMock(return_value={"id": "m1"})
        with patch.object(MemoryEntry, "from_mem0_result") as mock_from:
            mock_from.return_value = MagicMock()
            result = provider.get_sync("m1")
            assert result is not None

    def test_get_sync_not_found(self):
        logger.info("Mem0Provider: get sync not found")
        provider, _ = self._make_provider()
        provider._sync_memory.get = MagicMock(return_value=None)
        result = provider.get_sync("m1")
        assert result is None

    def test_get_sync_exception(self):
        logger.info("Mem0Provider: get sync exception")
        provider, _ = self._make_provider()
        provider._sync_memory.get = MagicMock(side_effect=Exception("err"))
        result = provider.get_sync("m1")
        assert result is None

    def test_get_all_sync(self):
        logger.info("Mem0Provider: get all sync")
        provider, _ = self._make_provider()
        provider._sync_memory.get_all = MagicMock(return_value={"results": [{"id": "1"}]})
        with patch.object(MemoryEntry, "from_mem0_result") as mock_from:
            mock_from.return_value = MagicMock()
            result = provider.get_all_sync(user_id="u1")
            assert len(result) == 1

    def test_get_all_sync_exception(self):
        logger.info("Mem0Provider: get all sync exception")
        provider, _ = self._make_provider()
        provider._sync_memory.get_all = MagicMock(side_effect=Exception("err"))
        result = provider.get_all_sync()
        assert result == []

    def test_delete_sync(self):
        logger.info("Mem0Provider: delete sync")
        provider, _ = self._make_provider()
        provider._sync_memory.delete = MagicMock()
        assert provider.delete_sync("m1") is True

    def test_delete_sync_exception(self):
        logger.info("Mem0Provider: delete sync exception")
        provider, _ = self._make_provider()
        provider._sync_memory.delete = MagicMock(side_effect=Exception("err"))
        assert provider.delete_sync("m1") is False

    def test_delete_all_sync(self):
        logger.info("Mem0Provider: delete all sync")
        provider, _ = self._make_provider()
        provider._sync_memory.delete_all = MagicMock()
        assert provider.delete_all_sync(user_id="u1") is True

    def test_delete_all_sync_exception(self):
        logger.info("Mem0Provider: delete all sync exception")
        provider, _ = self._make_provider()
        provider._sync_memory.delete_all = MagicMock(side_effect=Exception("err"))
        assert provider.delete_all_sync() is False

    def test_update_sync(self):
        logger.info("Mem0Provider: update sync")
        provider, _ = self._make_provider()
        provider._sync_memory.update = MagicMock(return_value={"id": "m1"})
        with patch.object(MemoryEntry, "from_mem0_result") as mock_from:
            mock_from.return_value = MagicMock()
            result = provider.update_sync("m1", "new data")
            assert result is not None

    def test_update_sync_returns_none(self):
        logger.info("Mem0Provider: update sync returns none")
        provider, _ = self._make_provider()
        provider._sync_memory.update = MagicMock(return_value=None)
        with pytest.raises(MemoryUpdateError):
            provider.update_sync("m1", "data")

    def test_update_sync_exception(self):
        logger.info("Mem0Provider: update sync exception")
        provider, _ = self._make_provider()
        provider._sync_memory.update = MagicMock(side_effect=RuntimeError("err"))
        with pytest.raises(MemoryUpdateError):
            provider.update_sync("m1", "data")

    def test_history_sync(self):
        logger.info("Mem0Provider: history sync")
        provider, _ = self._make_provider()
        provider._sync_memory.history = MagicMock(return_value=[{"v": 1}])
        result = provider.history_sync("m1")
        assert len(result) == 1

    def test_history_sync_exception(self):
        logger.info("Mem0Provider: history sync exception")
        provider, _ = self._make_provider()
        provider._sync_memory.history = MagicMock(side_effect=Exception("err"))
        result = provider.history_sync("m1")
        assert result == []

    def test_reset_sync(self):
        logger.info("Mem0Provider: reset sync")
        provider, _ = self._make_provider()
        provider._sync_memory.reset = MagicMock()
        assert provider.reset_sync() is True

    def test_reset_sync_exception(self):
        logger.info("Mem0Provider: reset sync exception")
        provider, _ = self._make_provider()
        provider._sync_memory.reset = MagicMock(side_effect=Exception("err"))
        with pytest.raises(MemoryError):
            provider.reset_sync()
