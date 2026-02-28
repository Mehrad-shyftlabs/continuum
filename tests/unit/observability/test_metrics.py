"""Unit tests for observability metrics."""

import time
from unittest.mock import MagicMock

import pytest

from orchestrator.observability.metrics import (
    LatencyMetric,
    MetricsCollector,
    TokenUsageMetric,
    get_metrics_collector,
    reset_metrics,
)
import logging

logger = logging.getLogger(__name__)


class TestLatencyMetric:
    def test_stop(self):
        logger.info("LatencyMetric: stop")
        m = LatencyMetric(name="test", start_time=time.perf_counter())
        time.sleep(0.01)
        duration = m.stop()
        assert duration > 0
        assert m.end_time is not None


class TestTokenUsageMetric:
    def test_creation(self):
        logger.info("TokenUsageMetric: creation")
        m = TokenUsageMetric(name="test", prompt_tokens=10, completion_tokens=5, total_tokens=15, model="gpt-4")
        assert m.prompt_tokens == 10
        assert m.total_tokens == 15


class TestMetricsCollector:
    def test_record_latency(self):
        logger.info("MetricsCollector: record latency")
        mc = MetricsCollector()
        mc.record_latency("test_op", 100.0, metadata={"key": "val"})

    def test_track_tokens(self):
        logger.info("MetricsCollector: track tokens")
        mc = MetricsCollector()
        mc.track_tokens("test_op", prompt_tokens=100, completion_tokens=50, model="gpt-4")

    def test_track_error(self):
        logger.info("MetricsCollector: track error")
        mc = MetricsCollector()
        err = RuntimeError("test")
        mc.track_error("test_op", err, metadata={"agent": "test"})

    def test_get_summary(self):
        logger.info("MetricsCollector: get summary")
        mc = MetricsCollector()
        mc.record_latency("op", 50.0)
        mc.track_tokens("op", prompt_tokens=10, completion_tokens=5)
        summary = mc.get_summary()
        assert isinstance(summary, dict)

    def test_report_to_trace(self):
        logger.info("MetricsCollector: report to trace")
        mc = MetricsCollector()
        mc.record_latency("op", 50.0)
        mock_trace = MagicMock()
        mc.report_to_trace(mock_trace)

    def test_reset(self):
        logger.info("MetricsCollector: reset")
        mc = MetricsCollector()
        mc.record_latency("op", 50.0)
        mc.reset()
        summary = mc.get_summary()
        assert isinstance(summary, dict)


class TestGlobalMetrics:
    def test_get_metrics_collector_singleton(self):
        logger.info("GlobalMetrics: get metrics collector singleton")
        mc1 = get_metrics_collector()
        mc2 = get_metrics_collector()
        assert mc1 is mc2

    def test_reset_metrics(self):
        logger.info("GlobalMetrics: reset metrics")
        mc = get_metrics_collector()
        mc.record_latency("op", 50.0)
        reset_metrics()
