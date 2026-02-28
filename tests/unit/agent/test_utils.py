"""Unit tests for agent utils."""

from unittest.mock import MagicMock

import pytest

from orchestrator.agent.utils.context_utils import create_run_context, inject_tool_context_to_prompt
from orchestrator.agent.utils.message_utils import message_to_dict
import logging

logger = logging.getLogger(__name__)


class TestCreateRunContext:
    def test_create_run_context_defaults(self):
        logger.info("CreateRunContext: create run context defaults")
        ctx = create_run_context()
        assert ctx.run_id is not None
        assert ctx.max_turns == 25

    def test_create_run_context_custom(self):
        logger.info("CreateRunContext: create run context custom")
        ctx = create_run_context(
            session_id="s1", user_id="u1", trace_id="t1", max_turns=10,
            metadata={"k": "v"}, tags=["tag1"],
        )
        assert ctx.session_id == "s1"
        assert ctx.max_turns == 10
        assert ctx.metadata["k"] == "v"
        assert "tag1" in ctx.tags


class TestInjectToolContext:
    def test_inject_empty(self):
        logger.info("InjectToolContext: inject empty")
        state = MagicMock()
        state.is_empty.return_value = True
        result = inject_tool_context_to_prompt(state)
        assert result is None

    def test_inject_with_session_id(self):
        logger.info("InjectToolContext: inject with session id")
        state = MagicMock()
        state.is_empty.return_value = False
        state.to_prompt_context.return_value = "Context info"
        state.get_all_namespaces.return_value = ["ns1"]
        state.get.return_value = "session-123"
        result = inject_tool_context_to_prompt(state)
        assert "IMPORTANT" in result
        assert "session already exists" in result

    def test_inject_without_session_id(self):
        logger.info("InjectToolContext: inject without session id")
        state = MagicMock()
        state.is_empty.return_value = False
        state.to_prompt_context.return_value = "Context info"
        state.get_all_namespaces.return_value = ["ns1"]
        state.get.return_value = None
        result = inject_tool_context_to_prompt(state)
        assert result == "Context info"


class TestMessageToDict:
    def test_message_to_dict_from_dict(self):
        logger.info("MessageToDict: message to dict from dict")
        msg = {"role": "user", "content": "hello"}
        result = message_to_dict(msg)
        assert result == msg

    def test_message_to_dict_from_chat_message(self):
        logger.info("MessageToDict: message to dict from chat message")
        from orchestrator.llm.types import ChatMessage

        msg = ChatMessage(role="user", content="hello")
        result = message_to_dict(msg)
        assert result["role"] == "user"
        assert result["content"] == "hello"
