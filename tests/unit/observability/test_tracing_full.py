"""Comprehensive tests for observability/tracing.py - Span, GenerationSpan, Trace, TracingManager."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from orchestrator.observability.tracing import (
    GenerationData,
    GenerationSpan,
    Span,
    SpanData,
    SpanKind,
    SpanLevel,
    Trace,
    TraceData,
    TracingManager,
)
import logging

logger = logging.getLogger(__name__)


class TestSpanLevel:
    def test_values(self):
        logger.info("SpanLevel: values")
        assert SpanLevel.DEBUG == "DEBUG"
        assert SpanLevel.DEFAULT == "DEFAULT"
        assert SpanLevel.WARNING == "WARNING"
        assert SpanLevel.ERROR == "ERROR"


class TestSpanKind:
    def test_values(self):
        logger.info("SpanKind: values")
        assert SpanKind.SPAN == "span"
        assert SpanKind.GENERATION == "generation"
        assert SpanKind.EVENT == "event"


class TestTraceData:
    def test_defaults(self):
        logger.info("TraceData: defaults")
        td = TraceData(name="test")
        assert td.name == "test"
        assert td.id is not None
        assert td.user_id is None
        assert td.metadata == {}
        assert td.tags == []
        assert td.public is False

    def test_model_dump(self):
        logger.info("TraceData: model dump")
        td = TraceData(name="test", user_id="u1", tags=["t1"])
        d = td.model_dump()
        assert d["name"] == "test"
        assert d["user_id"] == "u1"


class TestSpanData:
    def test_defaults(self):
        logger.info("SpanData: defaults")
        sd = SpanData(name="span")
        assert sd.name == "span"
        assert sd.level == SpanLevel.DEFAULT
        assert sd.end_time is None

    def test_with_all_fields(self):
        logger.info("SpanData: with all fields")
        sd = SpanData(
            name="s", input={"q": "test"}, output={"r": "res"},
            metadata={"k": "v"}, level=SpanLevel.WARNING,
        )
        assert sd.metadata["k"] == "v"
        assert sd.level == SpanLevel.WARNING


class TestGenerationData:
    def test_defaults(self):
        logger.info("GenerationData: defaults")
        gd = GenerationData(name="gen", model="gpt-4")
        assert gd.model == "gpt-4"
        assert gd.usage_prompt_tokens is None

    def test_with_usage(self):
        logger.info("GenerationData: with usage")
        gd = GenerationData(
            name="gen", model="gpt-4",
            usage_prompt_tokens=100, usage_completion_tokens=50,
            usage_total_tokens=150,
        )
        assert gd.usage_total_tokens == 150


class TestSpan:
    def test_create_without_langfuse(self):
        logger.info("Span: create without langfuse")
        data = SpanData(name="test-span")
        span = Span(None, data)
        assert span.id == data.id
        assert span.name == "test-span"

    def test_update_without_langfuse(self):
        logger.info("Span: update without langfuse")
        data = SpanData(name="test-span")
        span = Span(None, data)
        result = span.update(
            name="updated", input={"q": "test"},
            output={"r": "res"}, metadata={"k": "v"},
            level=SpanLevel.WARNING, status_message="ok",
        )
        assert result is span
        assert span._data.name == "updated"
        assert span._data.level == SpanLevel.WARNING

    def test_end_without_langfuse(self):
        logger.info("Span: end without langfuse")
        data = SpanData(name="test-span")
        span = Span(None, data)
        span.end(
            output={"result": "done"}, metadata={"k": "v"},
            level=SpanLevel.ERROR, status_message="failed",
        )
        assert span._data.end_time is not None
        assert span._data.output == {"result": "done"}

    def test_update_with_langfuse(self):
        logger.info("Span: update with langfuse")
        mock_lf = MagicMock()
        data = SpanData(name="s")
        span = Span(mock_lf, data)
        span.update(name="new", level=SpanLevel.DEBUG)
        mock_lf.update.assert_called_once()

    def test_update_with_langfuse_exception(self):
        logger.info("Span: update with langfuse exception")
        mock_lf = MagicMock()
        mock_lf.update.side_effect = Exception("langfuse down")
        data = SpanData(name="s")
        span = Span(mock_lf, data)
        span.update(name="new")

    def test_end_with_langfuse(self):
        logger.info("Span: end with langfuse")
        mock_lf = MagicMock()
        data = SpanData(name="s")
        span = Span(mock_lf, data)
        span.end(output="done")
        mock_lf.end.assert_called_once()

    def test_end_with_langfuse_exception(self):
        logger.info("Span: end with langfuse exception")
        mock_lf = MagicMock()
        mock_lf.end.side_effect = Exception("fail")
        span = Span(mock_lf, SpanData(name="s"))
        span.end()

    def test_child_span(self):
        logger.info("Span: child span")
        data = SpanData(name="parent")
        span = Span(None, data)
        child = span.span("child", input={"q": "test"})
        assert child.name == "child"
        assert len(span._children) == 1

    def test_child_span_with_langfuse(self):
        logger.info("Span: child span with langfuse")
        mock_lf = MagicMock()
        mock_lf.span.return_value = MagicMock()
        span = Span(mock_lf, SpanData(name="parent"))
        child = span.span("child")
        assert child is not None
        mock_lf.span.assert_called_once()

    def test_generation(self):
        logger.info("Span: generation")
        span = Span(None, SpanData(name="parent"))
        gen = span.generation("llm-call", model="gpt-4", input={"msg": "hi"})
        assert gen.name == "llm-call"
        assert len(span._children) == 1

    def test_generation_with_langfuse(self):
        logger.info("Span: generation with langfuse")
        mock_lf = MagicMock()
        mock_lf.generation.return_value = MagicMock()
        span = Span(mock_lf, SpanData(name="parent"))
        gen = span.generation("llm-call", model="gpt-4")
        mock_lf.generation.assert_called_once()

    def test_event_without_langfuse(self):
        logger.info("Span: event without langfuse")
        span = Span(None, SpanData(name="s"))
        span.event("test-event", input={"x": 1})

    def test_event_with_langfuse(self):
        logger.info("Span: event with langfuse")
        mock_lf = MagicMock()
        span = Span(mock_lf, SpanData(name="s"))
        span.event("test-event", output={"y": 2})
        mock_lf.event.assert_called_once()

    def test_score_without_langfuse(self):
        logger.info("Span: score without langfuse")
        span = Span(None, SpanData(name="s"))
        span.score("quality", 0.9)

    def test_score_with_langfuse(self):
        logger.info("Span: score with langfuse")
        mock_lf = MagicMock()
        span = Span(mock_lf, SpanData(name="s"))
        span.score("quality", 0.9, comment="good")
        mock_lf.score.assert_called_once()


class TestGenerationSpan:
    def test_create(self):
        logger.info("GenerationSpan: create")
        data = GenerationData(name="gen", model="gpt-4")
        gen = GenerationSpan(None, data)
        assert gen.id == data.id
        assert gen.name == "gen"

    def test_update_without_langfuse(self):
        logger.info("GenerationSpan: update without langfuse")
        gen = GenerationSpan(None, GenerationData(name="gen", model="gpt-4"))
        result = gen.update(
            name="updated", model="gpt-4o", output="result",
            usage_prompt_tokens=100, usage_completion_tokens=50,
            usage_total_tokens=150,
        )
        assert result is gen
        assert gen._data.model == "gpt-4o"
        assert gen._data.usage_total_tokens == 150

    def test_update_with_langfuse(self):
        logger.info("GenerationSpan: update with langfuse")
        mock_lf = MagicMock()
        gen = GenerationSpan(mock_lf, GenerationData(name="gen"))
        gen.update(usage_prompt_tokens=10)
        mock_lf.update.assert_called_once()

    def test_end_without_langfuse(self):
        logger.info("GenerationSpan: end without langfuse")
        gen = GenerationSpan(None, GenerationData(name="gen"))
        gen.end(
            output="done", usage_prompt_tokens=100,
            usage_completion_tokens=50, usage_total_tokens=150,
        )
        assert gen._data.end_time is not None
        assert gen._data.usage_total_tokens == 150

    def test_end_with_langfuse(self):
        logger.info("GenerationSpan: end with langfuse")
        mock_lf = MagicMock()
        gen = GenerationSpan(mock_lf, GenerationData(name="gen"))
        gen.end(output="done", usage_prompt_tokens=10)
        mock_lf.end.assert_called_once()

    def test_end_with_langfuse_exception(self):
        logger.info("GenerationSpan: end with langfuse exception")
        mock_lf = MagicMock()
        mock_lf.end.side_effect = Exception("fail")
        gen = GenerationSpan(mock_lf, GenerationData(name="gen"))
        gen.end()

    def test_score(self):
        logger.info("GenerationSpan: score")
        mock_lf = MagicMock()
        gen = GenerationSpan(mock_lf, GenerationData(name="gen"))
        gen.score("quality", 0.8, comment="ok")
        mock_lf.score.assert_called_once()


class TestTrace:
    def test_create_without_langfuse(self):
        logger.info("Trace: create without langfuse")
        data = TraceData(name="trace")
        trace = Trace(None, data)
        assert trace.id == data.id
        assert trace.name == "trace"
        assert trace.langfuse_trace is None

    def test_update_without_langfuse(self):
        logger.info("Trace: update without langfuse")
        trace = Trace(None, TraceData(name="t"))
        result = trace.update(
            name="updated", user_id="u1", session_id="s1",
            input={"q": "test"}, output={"r": "res"},
            metadata={"k": "v"}, tags=["t1"], public=True,
        )
        assert result is trace
        assert trace._data.user_id == "u1"
        assert "t1" in trace._data.tags
        assert trace._data.public is True

    def test_update_with_langfuse(self):
        logger.info("Trace: update with langfuse")
        mock_lf = MagicMock()
        trace = Trace(mock_lf, TraceData(name="t"))
        trace.update(name="new")
        mock_lf.update.assert_called_once()

    def test_span(self):
        logger.info("Trace: span")
        trace = Trace(None, TraceData(name="t"))
        span = trace.span("my-span", input={"q": "test"})
        assert span.name == "my-span"
        assert len(trace._spans) == 1

    def test_span_with_langfuse(self):
        logger.info("Trace: span with langfuse")
        mock_lf = MagicMock()
        mock_lf.span.return_value = MagicMock()
        trace = Trace(mock_lf, TraceData(name="t"))
        span = trace.span("my-span")
        mock_lf.span.assert_called_once()

    def test_generation(self):
        logger.info("Trace: generation")
        trace = Trace(None, TraceData(name="t"))
        gen = trace.generation("llm-call", model="gpt-4")
        assert gen.name == "llm-call"
        assert len(trace._generations) == 1

    def test_generation_with_langfuse(self):
        logger.info("Trace: generation with langfuse")
        mock_lf = MagicMock()
        mock_lf.generation.return_value = MagicMock()
        trace = Trace(mock_lf, TraceData(name="t"))
        gen = trace.generation("llm-call", model="gpt-4")
        mock_lf.generation.assert_called_once()

    def test_event_without_langfuse(self):
        logger.info("Trace: event without langfuse")
        trace = Trace(None, TraceData(name="t"))
        trace.event("evt", input={"x": 1})

    def test_event_with_langfuse(self):
        logger.info("Trace: event with langfuse")
        mock_lf = MagicMock()
        trace = Trace(mock_lf, TraceData(name="t"))
        trace.event("evt")
        mock_lf.event.assert_called_once()

    def test_score(self):
        logger.info("Trace: score")
        mock_lf = MagicMock()
        trace = Trace(mock_lf, TraceData(name="t"))
        trace.score("quality", 0.95)
        mock_lf.score.assert_called_once()

    def test_get_trace_url_without_langfuse(self):
        logger.info("Trace: get trace url without langfuse")
        trace = Trace(None, TraceData(name="t"))
        assert trace.get_trace_url() is None

    def test_get_trace_url_with_langfuse(self):
        logger.info("Trace: get trace url with langfuse")
        mock_lf = MagicMock()
        mock_lf.get_trace_url.return_value = "http://example.com/trace/123"
        trace = Trace(mock_lf, TraceData(name="t"))
        assert trace.get_trace_url() == "http://example.com/trace/123"


class TestTracingManager:
    def test_init(self):
        logger.info("TracingManager: init")
        tm = TracingManager()
        assert tm._provider_manager is None

    @patch("orchestrator.observability.tracing.get_current_trace_client", return_value=None)
    @patch("orchestrator.observability.tracing.get_current_trace_id", return_value=None)
    def test_current_trace_none(self, mock_id, mock_client):
        logger.info("TracingManager: current trace none")
        tm = TracingManager()
        assert tm._current_trace is None

    @patch("orchestrator.observability.tracing.get_current_span_client", return_value=None)
    @patch("orchestrator.observability.tracing.get_current_span_id", return_value=None)
    def test_current_span_none(self, mock_id, mock_client):
        logger.info("TracingManager: current span none")
        tm = TracingManager()
        assert tm._current_span is None

    @patch("orchestrator.observability.tracing.set_trace_context")
    @patch("orchestrator.observability.tracing.get_current_trace_id", return_value=None)
    def test_create_trace(self, mock_get_id, mock_set_ctx):
        logger.info("TracingManager: create trace")
        tm = TracingManager()
        mock_pm = MagicMock()
        mock_pm.is_enabled = False
        tm._provider_manager = mock_pm

        trace = tm.create_trace("test-trace", user_id="u1")
        assert trace is not None
        assert trace.name == "test-trace"

    @patch("orchestrator.observability.tracing.set_trace_context")
    @patch("orchestrator.observability.tracing.get_current_trace_id", return_value=None)
    def test_create_trace_with_enabled_provider(self, mock_get_id, mock_set_ctx):
        logger.info("TracingManager: create trace with enabled provider")
        tm = TracingManager()
        mock_pm = MagicMock()
        mock_pm.is_enabled = True
        mock_pm.trace.return_value = MagicMock()
        tm._provider_manager = mock_pm

        trace = tm.create_trace("test", force=True)
        mock_pm.trace.assert_called_once()

    @patch("orchestrator.observability.tracing.set_trace_context")
    @patch("orchestrator.observability.tracing.get_current_trace_id", return_value="existing-id")
    @patch("orchestrator.observability.tracing.get_current_trace_client", return_value=MagicMock())
    def test_create_trace_existing_context(self, mock_client, mock_id, mock_set_ctx):
        logger.info("TracingManager: create trace existing context")
        tm = TracingManager()
        trace = tm.create_trace("test")
        assert trace is not None

    def test_trace_context_manager(self):
        logger.info("TracingManager: trace context manager")
        tm = TracingManager()
        mock_pm = MagicMock()
        mock_pm.is_enabled = False
        tm._provider_manager = mock_pm

        with patch("orchestrator.observability.tracing.get_current_trace_id", return_value=None):
            with patch("orchestrator.observability.tracing.set_trace_context"):
                with patch("orchestrator.observability.tracing.TraceScope"):
                    with tm.trace("test-trace") as trace:
                        assert trace is not None

    def test_span_context_manager_no_parent(self):
        logger.info("TracingManager: span context manager no parent")
        tm = TracingManager()
        with patch.object(type(tm), "_current_span", new_callable=lambda: property(lambda s: None)):
            with patch.object(type(tm), "_current_trace", new_callable=lambda: property(lambda s: None)):
                with tm.span("test-span") as span:
                    assert span is not None

    def test_flush(self):
        logger.info("TracingManager: flush")
        tm = TracingManager()
        mock_pm = MagicMock()
        tm._provider_manager = mock_pm
        tm.flush()
        mock_pm.flush.assert_called_once()

    def test_shutdown(self):
        logger.info("TracingManager: shutdown")
        tm = TracingManager()
        mock_pm = MagicMock()
        tm._provider_manager = mock_pm
        tm.shutdown()
        mock_pm.flush.assert_called_once()
        mock_pm.shutdown.assert_called_once()

    def test_flush_exception(self):
        logger.info("TracingManager: flush exception")
        tm = TracingManager()
        mock_pm = MagicMock()
        mock_pm.flush.side_effect = Exception("fail")
        tm._provider_manager = mock_pm
        tm.flush()
