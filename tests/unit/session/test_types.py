"""Unit tests for session types."""

from datetime import datetime

import pytest

from orchestrator.session.types import (
    ChatMessage,
    SessionMessage,
    SessionMetadata,
    generate_session_id,
)
import logging

logger = logging.getLogger(__name__)


class TestSessionTypes:
    def test_generate_session_id(self):
        logger.info("SessionTypes: generate session id")
        sid = generate_session_id()
        assert isinstance(sid, str)
        assert len(sid) > 0

    def test_generate_session_id_unique(self):
        logger.info("SessionTypes: generate session id unique")
        ids = {generate_session_id() for _ in range(100)}
        assert len(ids) == 100

    def test_chat_message(self):
        logger.info("SessionTypes: chat message")
        m = ChatMessage(role="user", content="hello")
        assert m.role == "user"
        assert m.content == "hello"

    def test_session_metadata(self):
        logger.info("SessionTypes: session metadata")
        m = SessionMetadata(
            session_id="s1", message_count=5,
            created_at=datetime.now(), last_accessed_at=datetime.now(),
        )
        assert m.session_id == "s1"
        d = m.to_dict()
        assert d["session_id"] == "s1"

    def test_session_metadata_from_dict(self):
        logger.info("SessionTypes: session metadata from dict")
        d = {
            "session_id": "s1", "message_count": 3,
            "created_at": datetime.now().isoformat(),
            "last_accessed_at": datetime.now().isoformat(),
        }
        m = SessionMetadata.from_dict(d)
        assert m.session_id == "s1"

    def test_session_message(self):
        logger.info("SessionTypes: session message")
        msg = ChatMessage(role="user", content="hi")
        sm = SessionMessage(message=msg, timestamp=datetime.now())
        d = sm.to_dict()
        assert d["message"]["role"] == "user"

    def test_session_message_from_dict(self):
        logger.info("SessionTypes: session message from dict")
        d = {
            "message": {"role": "user", "content": "hi"},
            "metadata": {},
            "timestamp": datetime.now().isoformat(),
        }
        sm = SessionMessage.from_dict(d)
        assert sm.message.role == "user"
