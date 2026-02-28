"""Unit tests for session exceptions."""

import pytest

from orchestrator.session.exceptions import (
    SessionError,
    SessionMessageLimitError,
    SessionNotEnabledError,
    SessionNotFoundError,
)
import logging

logger = logging.getLogger(__name__)


class TestSessionExceptions:
    def test_session_error(self):
        logger.info("SessionExceptions: session error")
        err = SessionError("test")
        assert isinstance(err, Exception)

    def test_session_not_enabled(self):
        logger.info("SessionExceptions: session not enabled")
        err = SessionNotEnabledError("not enabled")
        assert isinstance(err, SessionError)

    def test_session_not_found(self):
        logger.info("SessionExceptions: session not found")
        err = SessionNotFoundError("not found")
        assert isinstance(err, SessionError)

    def test_session_message_limit(self):
        logger.info("SessionExceptions: session message limit")
        err = SessionMessageLimitError("limit", session_id="s1", current_count=100, max_messages=50)
        assert isinstance(err, SessionError)
