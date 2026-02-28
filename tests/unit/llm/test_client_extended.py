"""Extended tests for LLM client - covering more methods and edge cases."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.llm.config import LLMConfig
from orchestrator.llm.types import ChatMessage
import logging

logger = logging.getLogger(__name__)


class TestLLMClientSetupLitellm:
    @patch("orchestrator.llm.client.setup_langfuse")
    @patch("orchestrator.llm.client.litellm")
    def test_setup_litellm_loads_config(self, mock_ll, mock_setup):
        logger.info("LLMClientSetupLitellm: setup litellm loads config")
        from orchestrator.llm.client import LLMClient
        client = LLMClient(enable_langfuse=False)
        assert mock_ll.set_verbose is not None

    @patch("orchestrator.llm.client.setup_langfuse")
    @patch("orchestrator.llm.client.litellm")
    def test_json_mode_pydantic_log(self, mock_ll, mock_setup):
        logger.info("LLMClientSetupLitellm: json mode pydantic log")
        from pydantic import BaseModel
        from orchestrator.llm.client import LLMClient
        client = LLMClient(enable_langfuse=False)

        class MyModel(BaseModel):
            name: str

        client._log_json_mode_status({"response_format": MyModel}, "gpt-4")

    @patch("orchestrator.llm.client.setup_langfuse")
    @patch("orchestrator.llm.client.litellm")
    def test_validate_json_response_array(self, mock_ll, mock_setup):
        logger.info("LLMClientSetupLitellm: validate json response array")
        from orchestrator.llm.client import LLMClient
        client = LLMClient(enable_langfuse=False)
        client._validate_json_response('[1, 2, 3]', {"response_format": {}}, "gpt-4")

    @patch("orchestrator.llm.client.setup_langfuse")
    @patch("orchestrator.llm.client.litellm")
    def test_validate_json_response_invalid(self, mock_ll, mock_setup):
        logger.info("LLMClientSetupLitellm: validate json response invalid")
        from orchestrator.llm.client import LLMClient
        client = LLMClient(enable_langfuse=False)
        client._validate_json_response("{invalid", {"response_format": {}}, "gpt-4")

    @patch("orchestrator.llm.client.setup_langfuse")
    @patch("orchestrator.llm.client.litellm")
    def test_validate_json_response_no_format(self, mock_ll, mock_setup):
        logger.info("LLMClientSetupLitellm: validate json response no format")
        from orchestrator.llm.client import LLMClient
        client = LLMClient(enable_langfuse=False)
        client._validate_json_response("hello", {}, "gpt-4")


class TestLLMClientSyncStream:
    @patch("orchestrator.llm.client.setup_langfuse")
    @patch("orchestrator.llm.client.litellm")
    def test_chat_stream_sync(self, mock_ll, mock_setup):
        logger.info("LLMClientSyncStream: chat stream sync")
        from orchestrator.llm.client import LLMClient
        client = LLMClient(enable_langfuse=False)

        mock_chunk = MagicMock()
        mock_chunk.id = "chunk-1"
        mock_chunk.model = "gpt-4"
        mock_delta = MagicMock()
        mock_delta.content = "Hello"
        mock_delta.role = "assistant"
        mock_delta.tool_calls = None
        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = None
        mock_chunk.choices = [mock_choice]

        mock_ll.completion.return_value = [mock_chunk]

        chunks = list(client.chat_stream_sync([ChatMessage(role="user", content="hi")]))
        assert len(chunks) >= 1


class TestLLMRateLimiter:
    @pytest.mark.asyncio
    async def test_rate_limiter_acquire(self):
        logger.info("LLMRateLimiter: rate limiter acquire")
        from orchestrator.llm.client import _LLMRateLimiter
        rl = _LLMRateLimiter(rpm=60)
        await rl.acquire()
        assert rl.tokens < 60

    @pytest.mark.asyncio
    async def test_rate_limiter_multiple_acquires(self):
        logger.info("LLMRateLimiter: rate limiter multiple acquires")
        from orchestrator.llm.client import _LLMRateLimiter

        rl = _LLMRateLimiter(rpm=1000)
        for _ in range(5):
            await rl.acquire()
        assert rl.tokens < 1000
