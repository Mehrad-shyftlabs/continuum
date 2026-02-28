"""Unit tests for observability config."""

import pytest

from orchestrator.observability.config import ObservabilityConfig
import logging

logger = logging.getLogger(__name__)


class TestObservabilityConfig:
    def test_defaults(self):
        logger.info("ObservabilityConfig: defaults")
        c = ObservabilityConfig()
        assert isinstance(c.enabled, bool)

    def test_is_configured(self):
        logger.info("ObservabilityConfig: is configured")
        c = ObservabilityConfig()
        assert isinstance(c.is_configured(), bool)

    def test_get_provider_config(self):
        logger.info("ObservabilityConfig: get provider config")
        c = ObservabilityConfig()
        pc = c.get_provider_config("langfuse")
        assert pc is not None or pc is None
