"""Unit tests for LLM context management (progressive compression)."""

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


class TestCompressionStrategy:
    def test_values(self):
        logger.info("CompressionStrategy: values")
        assert CompressionStrategy.SUMMARIZE_OLD == "summarize_old"
        assert CompressionStrategy.TRUNCATE_OLDEST == "truncate_oldest"
        assert CompressionStrategy.SMART == "smart"


class TestContextManagementConfig:
    def test_defaults(self):
        logger.info("ContextManagementConfig: defaults")
        c = ContextManagementConfig()
        assert isinstance(c.compression_threshold, float)
        assert isinstance(c.keep_recent_messages, int)
        assert isinstance(c.enabled, bool)

    def test_to_dict(self):
        logger.info("ContextManagementConfig: to dict")
        c = ContextManagementConfig()
        d = c.to_dict()
        assert "enabled" in d
        assert "compression_threshold" in d


class TestCompressionResult:
    def test_creation(self):
        logger.info("CompressionResult: creation")
        r = CompressionResult(
            original_token_count=1000,
            compressed_token_count=500,
            messages_before=20,
            messages_after=10,
            was_compressed=True,
            strategy_used="smart",
            compression_ratio=0.5,
            latency_ms=10.0,
        )
        assert r.was_compressed is True
        assert r.messages_before == 20

    def test_to_dict(self):
        logger.info("CompressionResult: to dict")
        r = CompressionResult(
            original_token_count=1000,
            compressed_token_count=500,
            messages_before=20,
            messages_after=10,
            was_compressed=True,
            strategy_used="smart",
            compression_ratio=0.5,
            latency_ms=10.0,
        )
        d = r.to_dict()
        assert d["was_compressed"] is True


class TestSummaryCache:
    def test_creation(self):
        logger.info("SummaryCache: creation")
        cache = SummaryCache()
        assert cache is not None


class TestProgressiveContextManager:
    def test_init(self):
        logger.info("ProgressiveContextManager: init")
        mgr = ProgressiveContextManager()
        assert mgr is not None

    def test_config_defaults(self):
        logger.info("ProgressiveContextManager: config defaults")
        mgr = ProgressiveContextManager()
        assert mgr.config is not None
