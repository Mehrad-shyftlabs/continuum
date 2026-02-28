"""Unit tests for LLM callbacks."""

from unittest.mock import MagicMock, patch

import pytest

from orchestrator.llm.callbacks import get_langfuse_metadata
import logging

logger = logging.getLogger(__name__)


class TestGetLangfuseMetadata:
    def test_metadata_basic(self):
        logger.info("GetLangfuseMetadata: metadata basic")
        metadata = get_langfuse_metadata()
        assert isinstance(metadata, dict)

    def test_metadata_with_custom(self):
        logger.info("GetLangfuseMetadata: metadata with custom")
        custom = {"task": "test", "agent": "helper"}
        metadata = get_langfuse_metadata(custom_metadata=custom)
        assert metadata["task"] == "test"

    def test_metadata_with_trace_context(self):
        logger.info("GetLangfuseMetadata: metadata with trace context")
        from orchestrator.observability.trace_context import clear_trace_context, set_trace_context

        set_trace_context(trace_id="t1", user_id="u1", session_id="s1")
        try:
            metadata = get_langfuse_metadata()
            assert isinstance(metadata, dict)
        finally:
            clear_trace_context()
