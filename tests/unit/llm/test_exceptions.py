"""Unit tests for LLM exceptions."""

import pytest

from orchestrator.llm.exceptions import (
    LLMAuthenticationError,
    LLMContentFilterError,
    LLMContextLengthError,
    LLMError,
    LLMFallbackExhaustedError,
    LLMInvalidRequestError,
    LLMRateLimitError,
    LLMServiceUnavailableError,
    LLMStreamingError,
    LLMTimeoutError,
    LLMToolCallError,
)
import logging

logger = logging.getLogger(__name__)


class TestLLMExceptions:
    def test_llm_error_hierarchy(self):
        logger.info("LLMExceptions: llm error hierarchy")
        err = LLMError("test", model="gpt-4", provider="openai", should_report=False)
        assert err.model == "gpt-4"
        assert err.provider == "openai"
        assert err.error_code == "LLM_ERROR"

    def test_auth_error(self):
        logger.info("LLMExceptions: auth error")
        err = LLMAuthenticationError(should_report=False)
        assert err.error_code == "LLM_AUTH_ERROR"

    def test_rate_limit_error(self):
        logger.info("LLMExceptions: rate limit error")
        err = LLMRateLimitError("rate limited", retry_after=30.0, should_report=False)
        assert err.retry_after == 30.0
        assert err.context["retry_after_seconds"] == 30.0

    def test_timeout_error(self):
        logger.info("LLMExceptions: timeout error")
        err = LLMTimeoutError("timed out", timeout=60.0, should_report=False)
        assert err.timeout == 60.0

    def test_context_length_error(self):
        logger.info("LLMExceptions: context length error")
        err = LLMContextLengthError("too long", max_tokens=4096, requested_tokens=8000, should_report=False)
        assert err.max_tokens == 4096
        assert err.requested_tokens == 8000

    def test_invalid_request_error(self):
        logger.info("LLMExceptions: invalid request error")
        err = LLMInvalidRequestError(should_report=False)
        assert err.error_code == "LLM_INVALID_REQUEST"

    def test_service_unavailable_error(self):
        logger.info("LLMExceptions: service unavailable error")
        err = LLMServiceUnavailableError(should_report=False)
        assert err.error_code == "LLM_SERVICE_UNAVAILABLE"

    def test_fallback_exhausted_error(self):
        logger.info("LLMExceptions: fallback exhausted error")
        errs = [ValueError("e1"), RuntimeError("e2")]
        err = LLMFallbackExhaustedError(
            "all failed",
            attempted_models=["m1", "m2"],
            errors=errs,
            should_report=False,
        )
        assert err.attempted_models == ["m1", "m2"]
        assert err.context["error_count"] == 2

    def test_tool_call_error(self):
        logger.info("LLMExceptions: tool call error")
        err = LLMToolCallError("tool failed", tool_name="my_tool", tool_arguments={"a": 1}, should_report=False)
        assert err.tool_name == "my_tool"
        assert err.context["tool_name"] == "my_tool"

    def test_streaming_error(self):
        logger.info("LLMExceptions: streaming error")
        err = LLMStreamingError(should_report=False)
        assert err.error_code == "LLM_STREAMING_ERROR"

    def test_content_filter_error(self):
        logger.info("LLMExceptions: content filter error")
        err = LLMContentFilterError("filtered", filter_reason="violence", should_report=False)
        assert err.filter_reason == "violence"

    def test_all_have_correct_defaults(self):
        logger.info("LLMExceptions: all have correct defaults")
        classes = [
            LLMError, LLMAuthenticationError, LLMRateLimitError, LLMTimeoutError,
            LLMContextLengthError, LLMInvalidRequestError, LLMServiceUnavailableError,
            LLMFallbackExhaustedError, LLMToolCallError, LLMStreamingError, LLMContentFilterError,
        ]
        for cls in classes:
            assert hasattr(cls, "default_error_code")
            assert hasattr(cls, "default_message")
