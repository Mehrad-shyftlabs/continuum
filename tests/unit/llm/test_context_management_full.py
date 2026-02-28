"""Comprehensive tests for llm/context_management.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.llm.context_management import (
    CompressionResult,
    CompressionStrategy,
    ContextManagementConfig,
    ProgressiveContextManager,
    SummaryCache,
)
import logging

logger = logging.getLogger(__name__)


class TestContextManagementConfig:
    def test_defaults(self):
        logger.info("ContextManagementConfig: defaults")
        c = ContextManagementConfig()
        assert c.enabled is not None
        assert c.compression_threshold > 0

    def test_to_dict(self):
        logger.info("ContextManagementConfig: to dict")
        c = ContextManagementConfig()
        d = c.to_dict()
        assert "enabled" in d
        assert "compression_threshold" in d


class TestCompressionStrategy:
    def test_values(self):
        logger.info("CompressionStrategy: values")
        assert CompressionStrategy.SUMMARIZE_OLD is not None
        assert CompressionStrategy.TRUNCATE_OLDEST is not None
        assert CompressionStrategy.SMART is not None


class TestCompressionResult:
    def test_creation(self):
        logger.info("CompressionResult: creation")
        r = CompressionResult(
            original_token_count=1000,
            compressed_token_count=500,
            messages_before=10,
            messages_after=5,
            was_compressed=True,
            strategy_used="summarize_old",
            compression_ratio=0.5,
            latency_ms=150.0,
        )
        assert r.original_token_count == 1000
        assert r.compressed_token_count == 500
        assert r.was_compressed is True

    def test_to_dict(self):
        logger.info("CompressionResult: to dict")
        r = CompressionResult(
            original_token_count=1000,
            compressed_token_count=500,
            messages_before=10,
            messages_after=5,
            was_compressed=True,
            strategy_used="smart",
            compression_ratio=0.5,
            latency_ms=100.0,
        )
        d = r.to_dict()
        assert d["original_token_count"] == 1000


class TestSummaryCache:
    def test_creation(self):
        logger.info("SummaryCache: creation")
        cache = SummaryCache()
        assert cache is not None

    def test_get_missing(self):
        logger.info("SummaryCache: get missing")
        cache = SummaryCache()
        messages = [{"role": "user", "content": "hello"}]
        result = cache.get(messages)
        assert result is None

    def test_set_and_get(self):
        logger.info("SummaryCache: set and get")
        cache = SummaryCache()
        messages = [{"role": "user", "content": "hello"}]
        summary = [{"role": "system", "content": "Summary: user said hello"}]
        cache.set(messages, summary)
        result = cache.get(messages)
        assert result == summary

    def test_clear(self):
        logger.info("SummaryCache: clear")
        cache = SummaryCache()
        messages = [{"role": "user", "content": "hello"}]
        cache.set(messages, [{"role": "system", "content": "summary"}])
        cache.clear()
        assert cache.get(messages) is None


class TestProgressiveContextManager:
    def test_creation(self):
        logger.info("ProgressiveContextManager: creation")
        config = ContextManagementConfig()
        mgr = ProgressiveContextManager(config)
        assert mgr is not None

    def test_creation_defaults(self):
        logger.info("ProgressiveContextManager: creation defaults")
        mgr = ProgressiveContextManager()
        assert mgr is not None

    @pytest.mark.asyncio
    async def test_compress_if_needed_disabled(self):
        logger.info("ProgressiveContextManager: compress if needed disabled")
        config = ContextManagementConfig(enabled=False)
        mgr = ProgressiveContextManager(config)
        messages = [{"role": "user", "content": "hi"}]
        result_msgs, result = await mgr.compress_if_needed(messages, model="gpt-4o")
        assert isinstance(result, CompressionResult)
        assert result.was_compressed is False
