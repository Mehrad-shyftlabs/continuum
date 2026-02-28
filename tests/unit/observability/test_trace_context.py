"""Unit tests for trace context management."""

import pytest

from orchestrator.observability.trace_context import (
    clear_trace_context,
    get_current_agent_name,
    get_current_run_id,
    get_current_session_id,
    get_current_span_id,
    get_current_trace_id,
    get_current_user_id,
    set_trace_context,
    truncate_data,
)
import logging

logger = logging.getLogger(__name__)


class TestTraceContext:
    def setup_method(self):
        clear_trace_context()

    def test_set_get_trace_context(self):
        logger.info("TraceContext: set get trace context")
        set_trace_context(
            trace_id="t1", user_id="u1", session_id="s1",
            agent_name="agent1", run_id="r1",
        )
        assert get_current_trace_id() == "t1"
        assert get_current_user_id() == "u1"
        assert get_current_session_id() == "s1"
        assert get_current_agent_name() == "agent1"
        assert get_current_run_id() == "r1"

    def test_clear_trace_context(self):
        logger.info("TraceContext: clear trace context")
        set_trace_context(trace_id="t1", user_id="u1")
        clear_trace_context()
        assert get_current_trace_id() is None
        assert get_current_user_id() is None

    def test_get_current_span_id_default(self):
        logger.info("TraceContext: get current span id default")
        assert get_current_span_id() is None


class TestTruncateData:
    def test_truncate_short_string(self):
        logger.info("TruncateData: truncate short string")
        result = truncate_data("short")
        assert result == "short"

    def test_truncate_dict(self):
        logger.info("TruncateData: truncate dict")
        data = {"key": "x" * 10000}
        result = truncate_data(data)
        assert isinstance(result, dict)

    def test_truncate_none(self):
        logger.info("TruncateData: truncate none")
        assert truncate_data(None) is None

    def test_truncate_list(self):
        logger.info("TruncateData: truncate list")
        data = list(range(100))
        result = truncate_data(data)
        assert isinstance(result, list)
