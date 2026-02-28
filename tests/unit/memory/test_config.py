"""Unit tests for memory config."""

import pytest

from orchestrator.memory.config import MemoryConfig
import logging

logger = logging.getLogger(__name__)


class TestMemoryConfig:
    def test_defaults(self):
        logger.info("MemoryConfig: defaults")
        c = MemoryConfig()
        assert c.enabled is True
        assert c.search_limit == 5

    def test_is_configured(self):
        logger.info("MemoryConfig: is configured")
        c = MemoryConfig()
        assert isinstance(c.is_configured(), bool)
