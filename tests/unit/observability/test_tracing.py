"""Unit tests for observability tracing."""

import pytest

from orchestrator.observability.tracing import (
    SpanData,
    SpanKind,
    SpanLevel,
    TraceData,
    TracingManager,
)
import logging

logger = logging.getLogger(__name__)


class TestTraceData:
    def test_creation(self):
        logger.info("TraceData: creation")
        td = TraceData(name="test-trace", user_id="u1", session_id="s1")
        assert td.name == "test-trace"
        assert td.id is not None
        assert td.user_id == "u1"

    def test_model_dump(self):
        logger.info("TraceData: model dump")
        td = TraceData(name="t")
        d = td.model_dump()
        assert d["name"] == "t"
        assert "id" in d


class TestSpanData:
    def test_creation(self):
        logger.info("SpanData: creation")
        sd = SpanData(name="test-span")
        assert sd.name == "test-span"
        assert sd.id is not None

    def test_model_dump(self):
        logger.info("SpanData: model dump")
        sd = SpanData(name="s")
        d = sd.model_dump()
        assert d["name"] == "s"

    def test_level(self):
        logger.info("SpanData: level")
        sd = SpanData(name="s", level=SpanLevel.ERROR)
        assert sd.level == SpanLevel.ERROR


class TestSpanEnums:
    def test_span_levels(self):
        logger.info("SpanEnums: span levels")
        assert SpanLevel.DEBUG == "DEBUG"
        assert SpanLevel.DEFAULT == "DEFAULT"
        assert SpanLevel.WARNING == "WARNING"
        assert SpanLevel.ERROR == "ERROR"

    def test_span_kinds(self):
        logger.info("SpanEnums: span kinds")
        assert SpanKind.SPAN == "span"
        assert SpanKind.GENERATION == "generation"
        assert SpanKind.EVENT == "event"


class TestTracingManager:
    def test_init(self):
        logger.info("TracingManager: init")
        tm = TracingManager()
        assert tm is not None

    def test_create_trace(self):
        logger.info("TracingManager: create trace")
        tm = TracingManager()
        trace = tm.create_trace(name="test", user_id="u1")
        assert trace is not None
