"""Extended tests for tracing module - covering TracingManager methods."""

from unittest.mock import MagicMock, patch

import pytest

from orchestrator.observability.tracing import (
    GenerationData,
    SpanData,
    SpanLevel,
    TraceData,
    TracingManager,
)
import logging

logger = logging.getLogger(__name__)


class TestTracingManagerExtended:
    def test_create_trace_with_metadata(self):
        logger.info("TracingManagerExtended: create trace with metadata")
        tm = TracingManager()
        trace = tm.create_trace(
            name="test", user_id="u1", session_id="s1",
            metadata={"key": "val"}, tags=["prod"],
        )
        assert trace is not None

    def test_create_trace_returns_trace_data(self):
        logger.info("TracingManagerExtended: create trace returns trace data")
        tm = TracingManager()
        trace = tm.create_trace(name="basic")
        assert trace is not None


class TestGenerationData:
    def test_creation(self):
        logger.info("GenerationData: creation")
        gd = GenerationData(name="gen1", model="gpt-4")
        assert gd.name == "gen1"
        assert gd.model == "gpt-4"

    def test_model_dump(self):
        logger.info("GenerationData: model dump")
        gd = GenerationData(name="gen1", model="gpt-4")
        d = gd.model_dump()
        assert d["name"] == "gen1"

    def test_with_usage(self):
        logger.info("GenerationData: with usage")
        gd = GenerationData(
            name="gen1", model="gpt-4",
            usage_prompt_tokens=100,
            usage_completion_tokens=50,
            usage_total_tokens=150,
        )
        assert gd.usage_total_tokens == 150


class TestSpanDataExtended:
    def test_with_metadata(self):
        logger.info("SpanDataExtended: with metadata")
        sd = SpanData(name="s", metadata={"key": "val"})
        assert sd.metadata["key"] == "val"

    def test_with_input_output(self):
        logger.info("SpanDataExtended: with input output")
        sd = SpanData(name="s", input={"q": "test"}, output={"r": "result"})
        assert sd.input["q"] == "test"
        assert sd.output["r"] == "result"

    def test_with_level(self):
        logger.info("SpanDataExtended: with level")
        sd = SpanData(name="s", level=SpanLevel.WARNING)
        assert sd.level == SpanLevel.WARNING
