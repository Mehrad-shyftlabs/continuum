"""Unit tests for observability initialization."""

from unittest.mock import MagicMock, patch

import pytest

from orchestrator.observability.config import ObservabilityConfig
from orchestrator.observability.initialization import initialize_observability
import logging

logger = logging.getLogger(__name__)


class TestInitializeObservability:
    def test_initialize_returns_manager(self):
        logger.info("InitializeObservability: initialize returns manager")
        config = ObservabilityConfig()
        manager = initialize_observability(config)
        assert manager is not None

    def test_initialize_idempotent(self):
        logger.info("InitializeObservability: initialize idempotent")
        config = ObservabilityConfig()
        m1 = initialize_observability(config)
        m2 = initialize_observability(config)
        assert m1 is m2
