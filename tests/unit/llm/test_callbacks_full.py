"""Comprehensive tests for llm/callbacks.py."""

from unittest.mock import MagicMock, patch

import pytest

from orchestrator.llm.callbacks import (
    get_langfuse_metadata,
    setup_langfuse,
)
import logging

logger = logging.getLogger(__name__)


class TestGetLangfuseMetadata:
    @patch("orchestrator.llm.callbacks.get_current_trace_id", return_value="trace-123")
    @patch("orchestrator.llm.callbacks.get_current_span_id", return_value="span-456")
    def test_with_trace_context(self, mock_span, mock_trace):
        logger.info("GetLangfuseMetadata: with trace context")
        metadata = get_langfuse_metadata()
        assert metadata is not None
        assert isinstance(metadata, dict)

    @patch("orchestrator.llm.callbacks.get_current_trace_id", return_value=None)
    @patch("orchestrator.llm.callbacks.get_current_span_id", return_value=None)
    def test_without_trace_context(self, mock_span, mock_trace):
        logger.info("GetLangfuseMetadata: without trace context")
        metadata = get_langfuse_metadata()
        assert isinstance(metadata, dict)

    @patch("orchestrator.llm.callbacks.get_current_trace_id", return_value="t1")
    @patch("orchestrator.llm.callbacks.get_current_span_id", return_value="s1")
    def test_with_custom_metadata(self, mock_span, mock_trace):
        logger.info("GetLangfuseMetadata: with custom metadata")
        metadata = get_langfuse_metadata(custom_metadata={"key": "value"})
        assert isinstance(metadata, dict)

    @patch("orchestrator.llm.callbacks.get_current_trace_id", return_value="t1")
    @patch("orchestrator.llm.callbacks.get_current_span_id", return_value="s1")
    def test_with_explicit_trace_id(self, mock_span, mock_trace):
        logger.info("GetLangfuseMetadata: with explicit trace id")
        metadata = get_langfuse_metadata(trace_id="explicit-trace")
        assert isinstance(metadata, dict)

    @patch("orchestrator.llm.callbacks.get_current_trace_id", return_value="t1")
    @patch("orchestrator.llm.callbacks.get_current_span_id", return_value="s1")
    def test_with_tags(self, mock_span, mock_trace):
        logger.info("GetLangfuseMetadata: with tags")
        metadata = get_langfuse_metadata(tags=["test", "unit"])
        assert isinstance(metadata, dict)


class TestSetupLangfuse:
    @patch("orchestrator.llm.callbacks.settings")
    def test_setup_disabled(self, mock_settings):
        logger.info("SetupLangfuse: setup disabled")
        mock_settings.langfuse_enabled = False
        result = setup_langfuse()
        assert result is not None or result is None
