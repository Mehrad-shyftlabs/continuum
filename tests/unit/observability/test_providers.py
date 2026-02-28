"""Unit tests for observability providers."""

from unittest.mock import MagicMock

import pytest

from orchestrator.observability.providers.base import (
    ObservabilityProvider,
    ProviderCapabilities,
)
from orchestrator.observability.providers.registry import ProviderRegistry, get_provider_registry
import logging

logger = logging.getLogger(__name__)


class TestProviderCapabilities:
    def test_values(self):
        logger.info("ProviderCapabilities: values")
        assert ProviderCapabilities.TRACE == "trace"
        assert ProviderCapabilities.SPAN == "span"
        assert ProviderCapabilities.GENERATION == "generation"
        assert ProviderCapabilities.EVENT == "event"
        assert ProviderCapabilities.METRICS == "metrics"
        assert ProviderCapabilities.STREAMING == "streaming"


class TestProviderRegistry:
    def test_init(self):
        logger.info("ProviderRegistry: init")
        pr = ProviderRegistry()
        assert pr is not None

    def test_register_and_get(self):
        logger.info("ProviderRegistry: register and get")
        pr = ProviderRegistry()
        mock_provider = MagicMock(spec=ObservabilityProvider)
        mock_provider.name = "test_p"
        mock_provider.is_enabled = True
        pr.register("test_p", mock_provider)
        result = pr.get("test_p")
        assert result is mock_provider

    def test_get_missing(self):
        logger.info("ProviderRegistry: get missing")
        pr = ProviderRegistry()
        result = pr.get("nonexistent")
        assert result is None

    def test_get_enabled(self):
        logger.info("ProviderRegistry: get enabled")
        pr = ProviderRegistry()
        mock_provider = MagicMock(spec=ObservabilityProvider)
        mock_provider.name = "test_p"
        mock_provider.is_enabled = True
        pr.register("test_p", mock_provider)
        enabled = pr.get_enabled()
        assert len(enabled) >= 1

    def test_get_provider_registry(self):
        logger.info("ProviderRegistry: get provider registry")
        registry = get_provider_registry()
        assert registry is not None
