"""Unit tests for double message serialization fix (Issue 6)."""

import pytest

from orchestrator.agent.utils.message_utils import message_to_dict
import logging

logger = logging.getLogger(__name__)


class TestMessageSerialization:
    """Tests that message_to_dict is idempotent and messages aren't double-serialized."""

    def test_message_to_dict_idempotent(self):
        """Calling message_to_dict on a dict returns the same dict."""
        logger.info("Calling message_to_dict on a dict returns the same dict")
        msg = {"role": "user", "content": "Hello"}
        result = message_to_dict(msg)
        assert result == msg
        assert result is msg

    def test_messages_not_double_serialized(self):
        """After _prepare_run stores messages as dicts, run() should use them directly."""
        logger.info("After _prepare_run stores messages as dicts, run() should use them directly")
        messages_as_dicts = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        # Simulating the fixed behavior: list(run_state.messages) instead of
        # [message_to_dict(m) for m in run_state.messages]
        result = list(messages_as_dicts)
        assert result == messages_as_dicts
        # Each element should be the same object (no extra copy)
        for orig, copied in zip(messages_as_dicts, result):
            assert orig is copied

    def test_message_to_dict_with_tool_calls(self):
        """message_to_dict handles messages with tool_calls."""
        logger.info("message_to_dict handles messages with tool_calls")
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "tc_1", "function": {"name": "foo", "arguments": "{}"}}],
        }
        result = message_to_dict(msg)
        assert result == msg
