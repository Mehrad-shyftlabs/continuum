"""Unit tests for session config."""

import pytest

from orchestrator.session.config import SessionConfig
import logging

logger = logging.getLogger(__name__)


class TestSessionConfig:
    def test_defaults(self):
        logger.info("SessionConfig: defaults")
        c = SessionConfig()
        assert c.enabled is True
        assert c.ttl_seconds > 0
        assert c.max_messages > 0

    def test_is_configured(self):
        logger.info("SessionConfig: is configured")
        c = SessionConfig()
        assert isinstance(c.is_configured(), bool)
