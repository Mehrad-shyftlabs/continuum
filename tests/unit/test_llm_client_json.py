"""Unit tests for LLMClient JSON mode helpers (Issue 11 - extracted methods)."""

import json
from unittest.mock import MagicMock, call, patch

import pytest

from orchestrator.llm.config import LLMConfig
import logging

logger = logging.getLogger(__name__)


@pytest.fixture
def llm_client():
    """Create an LLMClient with mocked external dependencies."""
    with patch("orchestrator.llm.client.setup_langfuse"):
        from orchestrator.llm.client import LLMClient

        client = LLMClient(config=LLMConfig(model="test-model"))
        return client


class TestLogJsonModeStatus:
    """Tests for _log_json_mode_status()."""

    def test_log_json_mode_json_object(self, llm_client):
        logger.info("LogJsonModeStatus: log json mode json object")
        llm_kwargs = {"response_format": {"type": "json_object"}}
        with patch("orchestrator.llm.client.logger") as mock_logger:
            llm_client._log_json_mode_status(llm_kwargs, "test-model")
            mock_logger.info.assert_called_once()
            assert "json_object" in mock_logger.info.call_args[0][0]

    def test_log_json_mode_json_schema(self, llm_client):
        logger.info("LogJsonModeStatus: log json mode json schema")
        llm_kwargs = {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"strict": True},
            }
        }
        with patch("orchestrator.llm.client.logger") as mock_logger:
            llm_client._log_json_mode_status(llm_kwargs, "test-model")
            mock_logger.info.assert_called_once()
            assert "json_schema" in mock_logger.info.call_args[0][0]

    def test_log_json_mode_pydantic(self, llm_client):
        logger.info("LogJsonModeStatus: log json mode pydantic")
        class MyModel:
            __name__ = "MyModel"

        llm_kwargs = {"response_format": MyModel}
        with patch("orchestrator.llm.client.logger") as mock_logger:
            llm_client._log_json_mode_status(llm_kwargs, "test-model")
            mock_logger.info.assert_called_once()
            log_msg = mock_logger.info.call_args[0][0]
            assert "Pydantic" in log_msg or "MyModel" in log_msg

    def test_log_json_mode_no_format(self, llm_client):
        logger.info("LogJsonModeStatus: log json mode no format")
        with patch("orchestrator.llm.client.logger") as mock_logger:
            llm_client._log_json_mode_status({}, "test-model")
            mock_logger.info.assert_not_called()


class TestValidateJsonResponse:
    """Tests for _validate_json_response()."""

    def test_validate_json_response_valid(self, llm_client):
        logger.info("ValidateJsonResponse: validate json response valid")
        content = json.dumps({"key": "value"})
        llm_kwargs = {"response_format": {"type": "json_object"}}
        with patch("orchestrator.llm.client.logger") as mock_logger:
            llm_client._validate_json_response(content, llm_kwargs, "test-model")
            mock_logger.info.assert_called_once()
            assert "valid JSON" in mock_logger.info.call_args[0][0]

    def test_validate_json_response_valid_array(self, llm_client):
        logger.info("ValidateJsonResponse: validate json response valid array")
        content = json.dumps([1, 2, 3])
        llm_kwargs = {"response_format": {"type": "json_object"}}
        with patch("orchestrator.llm.client.logger") as mock_logger:
            llm_client._validate_json_response(content, llm_kwargs, "test-model")
            mock_logger.info.assert_called_once()
            assert "valid JSON" in mock_logger.info.call_args[0][0]

    def test_validate_json_response_invalid(self, llm_client):
        """Content that looks like JSON (starts/ends with braces) but isn't valid."""
        logger.info("Content that looks like JSON (starts/ends with braces) but isn't valid")
        content = "{not: valid json}"
        llm_kwargs = {"response_format": {"type": "json_object"}}
        with patch("orchestrator.llm.client.logger") as mock_logger:
            llm_client._validate_json_response(content, llm_kwargs, "test-model")
            mock_logger.warning.assert_called_once()
            assert "not valid JSON" in mock_logger.warning.call_args[0][0]

    def test_validate_json_response_not_json(self, llm_client):
        logger.info("ValidateJsonResponse: validate json response not json")
        content = "Just plain text response"
        llm_kwargs = {"response_format": {"type": "json_object"}}
        with patch("orchestrator.llm.client.logger") as mock_logger:
            llm_client._validate_json_response(content, llm_kwargs, "test-model")
            mock_logger.warning.assert_called_once()
            assert "doesn't appear to be JSON" in mock_logger.warning.call_args[0][0]

    def test_validate_json_response_no_format(self, llm_client):
        """If no response_format requested, no validation happens."""
        logger.info("If no response_format requested, no validation happens")
        with patch("orchestrator.llm.client.logger") as mock_logger:
            llm_client._validate_json_response("plain text", {}, "test-model")
            mock_logger.info.assert_not_called()
            mock_logger.warning.assert_not_called()

    def test_validate_json_response_none_content(self, llm_client):
        """If content is None, no validation happens."""
        logger.info("If content is None, no validation happens")
        llm_kwargs = {"response_format": {"type": "json_object"}}
        with patch("orchestrator.llm.client.logger") as mock_logger:
            llm_client._validate_json_response(None, llm_kwargs, "test-model")
            mock_logger.info.assert_not_called()
            mock_logger.warning.assert_not_called()
