"""Unit tests for error reporter."""

from unittest.mock import MagicMock, patch

import pytest

from orchestrator.observability.error_reporter import (
    ErrorReporter,
    ErrorReportingContext,
    disable_error_reporting,
    enable_error_reporting,
    report_error,
    report_exception,
)
import logging

logger = logging.getLogger(__name__)


class TestErrorReporter:
    def test_report_error(self):
        logger.info("ErrorReporter: report error")
        err = RuntimeError("test error")
        report_error(err, context="test")

    def test_report_exception(self):
        logger.info("ErrorReporter: report exception")
        try:
            raise ValueError("boom")
        except ValueError:
            report_exception(context="test")

    def test_error_reporting_context(self):
        logger.info("ErrorReporter: error reporting context")
        ctx = ErrorReportingContext(
            context="agent_run",
            trace_id="t1",
            user_id="u1",
            session_id="s1",
        )
        assert ctx.context == "agent_run"

    def test_enable_disable_reporting(self):
        logger.info("ErrorReporter: enable disable reporting")
        disable_error_reporting()
        enable_error_reporting()
