"""Unit tests for orchestrator config module."""

from unittest.mock import patch

import pytest

from orchestrator.config import Settings, get_settings
import logging

logger = logging.getLogger(__name__)


class TestSettings:
    def test_settings_has_expected_fields(self):
        logger.info("Settings: settings has expected fields")
        s = Settings()
        assert hasattr(s, "default_llm_model")
        assert hasattr(s, "default_llm_temperature")
        assert hasattr(s, "default_llm_max_tokens")
        assert hasattr(s, "llm_request_timeout")
        assert hasattr(s, "log_level")
        assert hasattr(s, "environment")
        assert hasattr(s, "memory_enabled")
        assert hasattr(s, "session_enabled")

    def test_settings_types(self):
        logger.info("Settings: settings types")
        s = Settings()
        assert isinstance(s.default_llm_model, str)
        assert isinstance(s.default_llm_temperature, float)
        assert isinstance(s.default_llm_max_tokens, int)
        assert isinstance(s.memory_enabled, bool)
        assert isinstance(s.session_enabled, bool)

    def test_get_settings_cached(self):
        logger.info("Settings: get settings cached")
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_settings_repr_masks_secrets(self):
        logger.info("Settings: settings repr masks secrets")
        s = Settings(openai_api_key="sk-secret123")
        r = repr(s)
        assert "sk-secret123" not in r

    def test_settings_memory_fields(self):
        logger.info("Settings: settings memory fields")
        s = Settings()
        assert hasattr(s, "qdrant_host")
        assert hasattr(s, "qdrant_port")
        assert hasattr(s, "embedder_provider")
        assert hasattr(s, "memory_search_limit")

    def test_settings_session_fields(self):
        logger.info("Settings: settings session fields")
        s = Settings()
        assert hasattr(s, "session_redis_host")
        assert hasattr(s, "session_redis_port")
        assert isinstance(s.session_ttl_seconds, int)
        assert isinstance(s.session_max_messages, int)

    def test_settings_context_management_fields(self):
        logger.info("Settings: settings context management fields")
        s = Settings()
        assert isinstance(s.context_management_enabled, bool)
        assert isinstance(s.context_compression_threshold, float)
        assert isinstance(s.context_keep_recent_messages, int)
