"""Unit tests for observability provider manager."""

from unittest.mock import MagicMock

import pytest

from orchestrator.observability.provider_manager import ProviderManager
import logging

logger = logging.getLogger(__name__)


class TestProviderManager:
    def test_init_with_mock_registry(self):
        logger.info("ProviderManager: init with mock registry")
        mock_registry = MagicMock()
        mock_registry.get_enabled.return_value = []
        pm = ProviderManager(registry=mock_registry)
        assert pm.is_enabled is False

    def test_is_enabled_with_provider(self):
        logger.info("ProviderManager: is enabled with provider")
        mock_registry = MagicMock()
        mock_provider = MagicMock()
        mock_registry.get_enabled.return_value = [mock_provider]
        pm = ProviderManager(registry=mock_registry)
        assert pm.is_enabled is True

    def test_shutdown(self):
        logger.info("ProviderManager: shutdown")
        mock_registry = MagicMock()
        mock_registry.get_enabled.return_value = []
        pm = ProviderManager(registry=mock_registry)
        pm.shutdown()
