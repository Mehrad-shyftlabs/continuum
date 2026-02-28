"""Unit tests for orchestrator logging module."""

import json

import pytest

from orchestrator.logging import (
    DevelopmentFormatter,
    JSONFormatter,
    LangfuseHandler,
    LogContext,
    LogLevel,
    OrchestratorLogger,
    clear_log_context,
    clear_loggers,
    get_child_logger,
    get_log_context,
    get_logger,
    get_logger_count,
    logger_for_module,
    set_log_context,
    setup_logging,
)
import logging

logger = logging.getLogger(__name__)


class TestGetLogger:
    def test_get_logger(self):
        logger.info("GetLogger: get logger")
        result = get_logger("test_module")
        assert result is not None
        assert "orchestrator" in result.name

    def test_get_logger_none_name(self):
        logger.info("GetLogger: get logger none name")
        result = get_logger(None)
        assert result.name == "orchestrator"

    def test_get_logger_non_orchestrator_prefix(self):
        logger.info("GetLogger: get logger non orchestrator prefix")
        result = get_logger("mymodule")
        assert result.name == "orchestrator.mymodule"

    def test_get_logger_orchestrator_prefix(self):
        logger.info("GetLogger: get logger orchestrator prefix")
        result = get_logger("orchestrator.llm")
        assert result.name == "orchestrator.llm"

    def test_logger_for_module(self):
        logger.info("GetLogger: logger for module")
        result = logger_for_module("orchestrator.agent")
        assert result.name == "orchestrator.agent"

    def test_logger_for_module_non_orchestrator(self):
        logger.info("GetLogger: logger for module non orchestrator")
        result = logger_for_module("mylib")
        assert result.name == "orchestrator.mylib"

    def test_get_child_logger(self):
        logger.info("GetLogger: get child logger")
        result = get_child_logger("orchestrator.agent", "runner")
        assert "runner" in result.name


class TestSetupLogging:
    def test_setup_logging_default(self):
        logger.info("SetupLogging: setup logging default")
        setup_logging(level="WARNING", json_format=False, enable_langfuse_handler=False)

    def test_setup_logging_json_format(self):
        logger.info("SetupLogging: setup logging json format")
        setup_logging(level="INFO", json_format=True, enable_langfuse_handler=False)

    def test_setup_logging_with_log_level_enum(self):
        logger.info("SetupLogging: setup logging with log level enum")
        setup_logging(level=LogLevel.DEBUG, json_format=False, enable_langfuse_handler=False)


class TestLogContext:
    def test_log_context_sets_and_clears(self):
        logger.info("LogContext: log context sets and clears")
        with LogContext(trace_id="t1", user_id="u1", span_id="s1", session_id="sess1"):
            ctx = get_log_context()
            assert ctx["trace_id"] == "t1"
            assert ctx["user_id"] == "u1"
            assert ctx["span_id"] == "s1"
            assert ctx["session_id"] == "sess1"
        ctx = get_log_context()
        assert ctx["trace_id"] is None

    def test_set_log_context(self):
        logger.info("LogContext: set log context")
        clear_log_context()
        set_log_context(trace_id="abc")
        assert get_log_context()["trace_id"] == "abc"
        clear_log_context()

    def test_clear_log_context(self):
        logger.info("LogContext: clear log context")
        set_log_context(trace_id="x", user_id="y")
        clear_log_context()
        ctx = get_log_context()
        assert ctx["trace_id"] is None
        assert ctx["user_id"] is None


class TestFormatters:
    def test_json_formatter(self):
        logger.info("Formatters: json formatter")
        formatter = JSONFormatter()
        record = logging.LogRecord("test", logging.INFO, "file", 1, "hello", (), None)
        result = formatter.format(record)
        data = json.loads(result)
        assert data["message"] == "hello"
        assert data["level"] == "INFO"

    def test_json_formatter_with_exception(self):
        logger.info("Formatters: json formatter with exception")
        formatter = JSONFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            record = logging.LogRecord("test", logging.ERROR, "file", 1, "err", (), sys.exc_info())
        result = formatter.format(record)
        data = json.loads(result)
        assert "exception" in data
        assert data["exception"]["type"] == "ValueError"

    def test_development_formatter(self):
        logger.info("Formatters: development formatter")
        formatter = DevelopmentFormatter()
        record = logging.LogRecord("test", logging.INFO, "file", 1, "hello dev", (), None)
        result = formatter.format(record)
        assert "hello dev" in result
        assert "INFO" in result

    def test_development_formatter_with_context(self):
        logger.info("Formatters: development formatter with context")
        formatter = DevelopmentFormatter()
        with LogContext(trace_id="abcdefgh", user_id="user1"):
            record = logging.LogRecord("test", logging.DEBUG, "file", 1, "msg", (), None)
            result = formatter.format(record)
            assert "trace=abcdefgh" in result


class TestLangfuseHandler:
    def test_langfuse_handler_no_client(self):
        logger.info("LangfuseHandler: langfuse handler no client")
        handler = LangfuseHandler()
        record = logging.LogRecord("test", logging.ERROR, "file", 1, "error msg", (), None)
        handler.emit(record)


class TestLoggerCache:
    def test_clear_loggers(self):
        logger.info("LoggerCache: clear loggers")
        get_logger("test_cache_1")
        count = clear_loggers()
        assert count >= 1

    def test_get_logger_count(self):
        logger.info("LoggerCache: get logger count")
        clear_loggers()
        get_logger("test_count_a")
        get_logger("test_count_b")
        assert get_logger_count() >= 2


class TestLogLevel:
    def test_log_level_values(self):
        logger.info("LogLevel: log level values")
        assert LogLevel.DEBUG == "DEBUG"
        assert LogLevel.INFO == "INFO"
        assert LogLevel.WARNING == "WARNING"
        assert LogLevel.ERROR == "ERROR"
        assert LogLevel.CRITICAL == "CRITICAL"
