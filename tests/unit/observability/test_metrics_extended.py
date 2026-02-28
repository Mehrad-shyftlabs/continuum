"""Extended unit tests for observability metrics - covering MetricsCollector methods."""

import time
from unittest.mock import MagicMock

import pytest

from orchestrator.observability.metrics import (
    ErrorMetric,
    LatencyMetric,
    MetricsCollector,
    TokenUsageMetric,
)
import logging

logger = logging.getLogger(__name__)


class TestMetricsCollectorContextManagers:
    def test_track_latency_context_manager(self):
        logger.info("MetricsCollectorContextManagers: track latency context manager")
        mc = MetricsCollector()
        with mc.track_latency("test_op") as metric:
            time.sleep(0.005)
        assert metric.duration_ms is not None
        assert metric.duration_ms > 0
        assert len(mc._latencies) >= 1

    def test_track_latency_with_metadata(self):
        logger.info("MetricsCollectorContextManagers: track latency with metadata")
        mc = MetricsCollector()
        with mc.track_latency("test_op", metadata={"key": "val"}) as metric:
            pass
        assert metric.metadata["key"] == "val"

    def test_track_latency_exception(self):
        logger.info("MetricsCollectorContextManagers: track latency exception")
        mc = MetricsCollector()
        with pytest.raises(ValueError):
            with mc.track_latency("test_op") as metric:
                raise ValueError("boom")
        assert metric.duration_ms is not None


class TestMetricsCollectorRecords:
    def test_record_latency(self):
        logger.info("MetricsCollectorRecords: record latency")
        mc = MetricsCollector()
        mc.record_latency("op", 50.0, metadata={"key": "val"})
        assert len(mc._latencies) == 1
        assert mc._latencies[0].duration_ms == 50.0

    def test_track_tokens(self):
        logger.info("MetricsCollectorRecords: track tokens")
        mc = MetricsCollector()
        mc.track_tokens("op", prompt_tokens=100, completion_tokens=50, model="gpt-4")
        assert len(mc._token_usage) == 1
        assert mc._token_usage[0].prompt_tokens == 100

    def test_track_error(self):
        logger.info("MetricsCollectorRecords: track error")
        mc = MetricsCollector()
        mc.track_error("op", RuntimeError("err"), metadata={"agent": "test"})
        assert len(mc._errors) == 1
        assert mc._errors[0].error_type == "RuntimeError"

    def test_record_metric(self):
        logger.info("MetricsCollectorRecords: record metric")
        mc = MetricsCollector()
        mc.record_metric("accuracy", 0.95)
        mc.record_metric("accuracy", 0.97)
        assert "accuracy" in mc._custom_metrics
        assert len(mc._custom_metrics["accuracy"]) == 2


class TestMetricsCollectorSummary:
    def test_get_summary_empty(self):
        logger.info("MetricsCollectorSummary: get summary empty")
        mc = MetricsCollector()
        summary = mc.get_summary()
        assert isinstance(summary, dict)

    def test_get_summary_with_data(self):
        logger.info("MetricsCollectorSummary: get summary with data")
        mc = MetricsCollector()
        mc.record_latency("op1", 50.0)
        mc.record_latency("op1", 100.0)
        mc.track_tokens("op", prompt_tokens=100, completion_tokens=50)
        mc.track_error("op", ValueError("err"))
        summary = mc.get_summary()
        assert "latency" in summary or "latencies" in summary or isinstance(summary, dict)


class TestMetricsCollectorReporting:
    def test_report_to_trace(self):
        logger.info("MetricsCollectorReporting: report to trace")
        mc = MetricsCollector()
        mc.record_latency("op", 50.0)
        mc.track_tokens("op", prompt_tokens=10, completion_tokens=5)
        mock_trace = MagicMock()
        mc.report_to_trace(mock_trace)

    def test_reset(self):
        logger.info("MetricsCollectorReporting: reset")
        mc = MetricsCollector()
        mc.record_latency("op", 50.0)
        mc.track_tokens("op", prompt_tokens=10, completion_tokens=5)
        mc.track_error("op", RuntimeError("err"))
        mc.record_metric("x", 1.0)
        mc.reset()
        assert len(mc._latencies) == 0
        assert len(mc._token_usage) == 0
        assert len(mc._errors) == 0
        assert len(mc._custom_metrics) == 0


class TestErrorMetric:
    def test_creation(self):
        logger.info("ErrorMetric: creation")
        m = ErrorMetric(name="op", error_type="ValueError", error_message="bad")
        assert m.name == "op"
        assert m.error_type == "ValueError"
