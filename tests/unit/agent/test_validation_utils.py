"""Tests for agent/utils/validation_utils.py."""

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from orchestrator.agent.types import RunContext
from orchestrator.agent.utils.validation_utils import validate_input
import logging

logger = logging.getLogger(__name__)


class SampleSchema(BaseModel):
    input: str


class TestValidateInput:
    @pytest.mark.asyncio
    async def test_no_schema_returns_none(self):
        logger.info("ValidateInput: no schema returns none")
        agent = MagicMock()
        agent.input_schema = None
        ctx = RunContext(run_id="r1")
        result = await validate_input(agent, "hello", ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_string_input_passes(self):
        logger.info("ValidateInput: string input passes")
        agent = MagicMock()
        agent.input_schema = SampleSchema
        agent.name = "test"
        ctx = RunContext(run_id="r1")
        result = await validate_input(agent, "hello", ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_list_dict_input(self):
        logger.info("ValidateInput: list dict input")
        agent = MagicMock()
        agent.input_schema = SampleSchema
        agent.name = "test"
        ctx = RunContext(run_id="r1")
        result = await validate_input(
            agent,
            [{"role": "user", "content": "hello"}],
            ctx,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_list_with_model(self):
        logger.info("ValidateInput: list with model")
        agent = MagicMock()
        agent.input_schema = SampleSchema
        agent.name = "test"
        ctx = RunContext(run_id="r1")
        msg = MagicMock()
        msg.content = "hello"
        result = await validate_input(agent, [msg], ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_input(self):
        logger.info("ValidateInput: empty input")
        agent = MagicMock()
        agent.input_schema = SampleSchema
        agent.name = "test"
        ctx = RunContext(run_id="r1")
        result = await validate_input(agent, "", ctx)
        assert result is None


class TestMessageToDict:
    def test_dict_passthrough(self):
        logger.info("MessageToDict: dict passthrough")
        from orchestrator.agent.utils.message_utils import message_to_dict
        msg = {"role": "user", "content": "hi"}
        assert message_to_dict(msg) == msg

    def test_to_dict_method(self):
        logger.info("MessageToDict: to dict method")
        from orchestrator.agent.utils.message_utils import message_to_dict
        msg = MagicMock()
        msg.to_dict.return_value = {"role": "user", "content": "hi"}
        del msg.model_dump
        result = message_to_dict(msg)
        assert result == {"role": "user", "content": "hi"}

    def test_model_dump_method(self):
        logger.info("MessageToDict: model dump method")
        from orchestrator.agent.utils.message_utils import message_to_dict
        msg = MagicMock(spec=[])
        msg.model_dump = MagicMock(return_value={"role": "user", "content": "hi"})
        result = message_to_dict(msg)
        assert result == {"role": "user", "content": "hi"}

    def test_fallback_to_str(self):
        logger.info("MessageToDict: fallback to str")
        from orchestrator.agent.utils.message_utils import message_to_dict

        result = message_to_dict("hello")
        assert result["role"] == "user"
        assert result["content"] == "hello"
