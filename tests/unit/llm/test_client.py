"""Unit tests for LLM client."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.llm.config import LLMConfig
from orchestrator.llm.exceptions import (
    LLMAuthenticationError,
    LLMContextLengthError,
    LLMError,
    LLMInvalidRequestError,
    LLMRateLimitError,
    LLMServiceUnavailableError,
    LLMTimeoutError,
)
from orchestrator.llm.types import ChatMessage, LLMResponse, ToolDefinition
import logging

logger = logging.getLogger(__name__)


class TestLLMClientInit:
    @patch("orchestrator.llm.client.setup_langfuse")
    @patch("orchestrator.llm.client.litellm")
    def test_client_initialization(self, mock_ll, mock_setup):
        logger.info("LLMClientInit: client initialization")
        from orchestrator.llm.client import LLMClient
        client = LLMClient(enable_langfuse=False)
        assert client.default_config is not None
        assert client._langfuse_enabled is False

    @patch("orchestrator.llm.client.setup_langfuse", side_effect=Exception("no langfuse"))
    @patch("orchestrator.llm.client.litellm")
    def test_client_initialization_langfuse_fails(self, mock_ll, mock_setup):
        logger.info("LLMClientInit: client initialization langfuse fails")
        from orchestrator.llm.client import LLMClient
        client = LLMClient(enable_langfuse=True)
        assert client._langfuse_enabled is True

    @patch("orchestrator.llm.client.setup_langfuse")
    @patch("orchestrator.llm.client.litellm")
    def test_client_with_rate_limiter(self, mock_ll, mock_setup):
        logger.info("LLMClientInit: client with rate limiter")
        from orchestrator.llm.client import LLMClient
        config = LLMConfig(rate_limit_rpm=60)
        client = LLMClient(config=config, enable_langfuse=False)
        assert client._rate_limiter is not None


class TestLLMClientConversions:
    @patch("orchestrator.llm.client.setup_langfuse")
    @patch("orchestrator.llm.client.litellm")
    def test_convert_messages_from_chat_message(self, mock_ll, mock_setup):
        logger.info("LLMClientConversions: convert messages from chat message")
        from orchestrator.llm.client import LLMClient
        client = LLMClient(enable_langfuse=False)
        msgs = [ChatMessage(role="user", content="hello")]
        result = client._convert_messages(msgs)
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "hello"

    @patch("orchestrator.llm.client.setup_langfuse")
    @patch("orchestrator.llm.client.litellm")
    def test_convert_messages_from_dict(self, mock_ll, mock_setup):
        logger.info("LLMClientConversions: convert messages from dict")
        from orchestrator.llm.client import LLMClient
        client = LLMClient(enable_langfuse=False)
        msgs = [{"role": "user", "content": "hello"}]
        result = client._convert_messages(msgs)
        assert result[0] == msgs[0]

    @patch("orchestrator.llm.client.setup_langfuse")
    @patch("orchestrator.llm.client.litellm")
    def test_convert_tools(self, mock_ll, mock_setup):
        logger.info("LLMClientConversions: convert tools")
        from orchestrator.llm.client import LLMClient
        from orchestrator.llm.types import FunctionDefinition
        client = LLMClient(enable_langfuse=False)
        tools = [ToolDefinition(function=FunctionDefinition(name="fn", description="d"))]
        result = client._convert_tools(tools)
        assert result[0]["type"] == "function"

    @patch("orchestrator.llm.client.setup_langfuse")
    @patch("orchestrator.llm.client.litellm")
    def test_convert_tools_none(self, mock_ll, mock_setup):
        logger.info("LLMClientConversions: convert tools none")
        from orchestrator.llm.client import LLMClient
        client = LLMClient(enable_langfuse=False)
        assert client._convert_tools(None) is None


class TestLLMClientMetadata:
    @patch("orchestrator.llm.client.setup_langfuse")
    @patch("orchestrator.llm.client.litellm")
    def test_build_metadata_without_langfuse(self, mock_ll, mock_setup):
        logger.info("LLMClientMetadata: build metadata without langfuse")
        from orchestrator.llm.client import LLMClient
        client = LLMClient(enable_langfuse=False)
        metadata = client._build_metadata(trace_metadata={"key": "val"})
        assert metadata == {"key": "val"}

    @patch("orchestrator.llm.client.get_langfuse_metadata", return_value={"trace_id": "t1"})
    @patch("orchestrator.llm.client.setup_langfuse")
    @patch("orchestrator.llm.client.litellm")
    def test_build_metadata_with_langfuse(self, mock_ll, mock_setup, mock_meta):
        logger.info("LLMClientMetadata: build metadata with langfuse")
        from orchestrator.llm.client import LLMClient
        client = LLMClient(enable_langfuse=False)
        client._langfuse_enabled = True
        metadata = client._build_metadata()
        assert "trace_id" in metadata


class TestLLMClientExceptionHandling:
    @patch("orchestrator.llm.client.setup_langfuse")
    @patch("orchestrator.llm.client.litellm")
    def _make_client(self, mock_ll, mock_setup):
        from orchestrator.llm.client import LLMClient
        return LLMClient(enable_langfuse=False)

    def test_handle_exception_auth_error(self):
        logger.info("LLMClientExceptionHandling: handle exception auth error")
        from litellm import AuthenticationError
        client = self._make_client()
        with pytest.raises(LLMAuthenticationError):
            client._handle_exception(AuthenticationError("auth", "gpt-4", "openai"), "gpt-4")

    def test_handle_exception_rate_limit(self):
        logger.info("LLMClientExceptionHandling: handle exception rate limit")
        from litellm import RateLimitError
        client = self._make_client()
        with pytest.raises(LLMRateLimitError):
            client._handle_exception(RateLimitError("rate", "gpt-4", "openai"), "gpt-4")

    def test_handle_exception_timeout(self):
        logger.info("LLMClientExceptionHandling: handle exception timeout")
        from litellm import Timeout
        client = self._make_client()
        with pytest.raises(LLMTimeoutError):
            client._handle_exception(Timeout("timeout", "gpt-4", "openai"), "gpt-4")

    def test_handle_exception_context_length(self):
        logger.info("LLMClientExceptionHandling: handle exception context length")
        from litellm import ContextWindowExceededError
        client = self._make_client()
        with pytest.raises(LLMContextLengthError):
            client._handle_exception(ContextWindowExceededError("ctx", "gpt-4", "openai"), "gpt-4")

    def test_handle_exception_bad_request(self):
        logger.info("LLMClientExceptionHandling: handle exception bad request")
        from litellm import BadRequestError
        client = self._make_client()
        with pytest.raises(LLMInvalidRequestError):
            client._handle_exception(BadRequestError("bad", "gpt-4", "openai"), "gpt-4")

    def test_handle_exception_service_unavailable(self):
        logger.info("LLMClientExceptionHandling: handle exception service unavailable")
        from litellm import ServiceUnavailableError
        client = self._make_client()
        with pytest.raises(LLMServiceUnavailableError):
            client._handle_exception(ServiceUnavailableError("down", "gpt-4", "openai"), "gpt-4")

    def test_handle_exception_generic(self):
        logger.info("LLMClientExceptionHandling: handle exception generic")
        client = self._make_client()
        with pytest.raises(LLMError):
            client._handle_exception(RuntimeError("generic"), "gpt-4")


class TestLLMClientProvider:
    @patch("orchestrator.llm.client.setup_langfuse")
    @patch("orchestrator.llm.client.litellm")
    def test_get_provider_from_model(self, mock_ll, mock_setup):
        logger.info("LLMClientProvider: get provider from model")
        from orchestrator.llm.client import LLMClient
        client = LLMClient(enable_langfuse=False)
        assert client._get_provider_from_model("gpt-4") == "openai"
        assert client._get_provider_from_model("claude-3-opus") == "anthropic"
        assert client._get_provider_from_model("gemini/gemini-pro") == "gemini"
        assert client._get_provider_from_model("unknown-model") == "unknown"


class TestLLMClientJsonHelpers:
    @patch("orchestrator.llm.client.setup_langfuse")
    @patch("orchestrator.llm.client.litellm")
    def test_log_json_mode_status(self, mock_ll, mock_setup):
        logger.info("LLMClientJsonHelpers: log json mode status")
        from orchestrator.llm.client import LLMClient
        client = LLMClient(enable_langfuse=False)
        client._log_json_mode_status({"response_format": {"type": "json_object"}}, "gpt-4")
        client._log_json_mode_status({"response_format": {"type": "json_schema", "json_schema": {}}}, "gpt-4")
        client._log_json_mode_status({}, "gpt-4")

    @patch("orchestrator.llm.client.setup_langfuse")
    @patch("orchestrator.llm.client.litellm")
    def test_validate_json_response(self, mock_ll, mock_setup):
        logger.info("LLMClientJsonHelpers: validate json response")
        from orchestrator.llm.client import LLMClient
        client = LLMClient(enable_langfuse=False)
        client._validate_json_response('{"key": "val"}', {"response_format": {"type": "json_object"}}, "gpt-4")
        client._validate_json_response("not json", {"response_format": {"type": "json_object"}}, "gpt-4")
        client._validate_json_response(None, {"response_format": {}}, "gpt-4")
        client._validate_json_response('{"invalid json', {"response_format": {}}, "gpt-4")

    @patch("orchestrator.llm.client.setup_langfuse")
    @patch("orchestrator.llm.client.litellm")
    def test_handle_tools_with_json_mode(self, mock_ll, mock_setup):
        logger.info("LLMClientJsonHelpers: handle tools with json mode")
        from orchestrator.llm.client import LLMClient
        client = LLMClient(enable_langfuse=False)
        config = LLMConfig(json_mode=True, model="gemini/gemini-2.5-flash")
        llm_kwargs = {"response_format": {"type": "json_object"}}
        tools = [{"type": "function", "function": {"name": "fn"}}]
        client._handle_tools_with_json_mode(llm_kwargs, config, tools)
        assert "response_format" not in llm_kwargs


class TestLLMClientChatSync:
    @patch("orchestrator.llm.client.setup_langfuse")
    @patch("orchestrator.llm.client.litellm")
    def test_chat_sync(self, mock_ll, mock_setup):
        logger.info("LLMClientChatSync: chat sync")
        from orchestrator.llm.client import LLMClient
        client = LLMClient(enable_langfuse=False)

        mock_resp = MagicMock()
        mock_resp.id = "resp-1"
        mock_resp.model = "gpt-4"
        mock_choice = MagicMock()
        mock_choice.message.content = "Hello!"
        mock_choice.message.role = "assistant"
        mock_choice.message.tool_calls = None
        mock_choice.message.function_call = None
        mock_choice.finish_reason = "stop"
        mock_resp.choices = [mock_choice]
        mock_resp.usage.prompt_tokens = 10
        mock_resp.usage.completion_tokens = 5
        mock_resp.usage.total_tokens = 15
        mock_resp.model_dump.return_value = {}
        mock_ll.completion.return_value = mock_resp

        result = client.chat_sync([ChatMessage(role="user", content="hi")])
        assert result.content == "Hello!"
        assert result.model == "gpt-4"


class TestLLMClientChatAsync:
    @patch("orchestrator.llm.client.setup_langfuse")
    @patch("orchestrator.llm.client.litellm")
    @pytest.mark.asyncio
    async def test_chat_async(self, mock_ll, mock_setup):
        logger.info("LLMClientChatAsync: chat async")
        from orchestrator.llm.client import LLMClient

        client = LLMClient(enable_langfuse=False)

        mock_resp = MagicMock()
        mock_resp.id = "resp-1"
        mock_resp.model = "gpt-4"
        mock_choice = MagicMock()
        mock_choice.message.content = "Hello async!"
        mock_choice.message.role = "assistant"
        mock_choice.message.tool_calls = None
        mock_choice.message.function_call = None
        mock_choice.finish_reason = "stop"
        mock_resp.choices = [mock_choice]
        mock_resp.usage.prompt_tokens = 10
        mock_resp.usage.completion_tokens = 5
        mock_resp.usage.total_tokens = 15
        mock_resp.model_dump.return_value = {}
        mock_ll.acompletion = AsyncMock(return_value=mock_resp)

        result = await client.chat(
            [ChatMessage(role="user", content="hi")],
            auto_session=False,
        )
        assert result.content == "Hello async!"
