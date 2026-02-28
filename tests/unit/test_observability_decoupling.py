"""Unit tests for observability decoupling (Issue 19)."""

from unittest.mock import MagicMock, patch

import pytest
import logging

logger = logging.getLogger(__name__)


class TestExceptionErrorReporter:
    def test_exception_without_reporter_configured(self):
        """Exceptions should work fine when no reporter is set."""
        logger.info("Exceptions should work fine when no reporter is set")
        from orchestrator.exceptions import OrchestratorError, set_error_reporter

        original = None
        try:
            from orchestrator.exceptions import _error_reporter
            original = _error_reporter
            set_error_reporter(None)

            err = OrchestratorError("test error", should_report=True)
            assert err.message == "test error"
        finally:
            set_error_reporter(original)

    def test_exception_with_reporter_configured(self):
        """When a reporter is set, it should be called on exception creation."""
        logger.info("When a reporter is set, it should be called on exception creation")
        from orchestrator.exceptions import OrchestratorError, set_error_reporter

        mock_reporter = MagicMock()
        original = None
        try:
            from orchestrator.exceptions import _error_reporter
            original = _error_reporter
            set_error_reporter(mock_reporter)

            err = OrchestratorError("test error", should_report=True)
            mock_reporter.assert_called_once_with(err)
        finally:
            set_error_reporter(original)

    def test_exception_reporter_not_called_when_should_report_false(self):
        """Reporter should not be called when should_report=False."""
        logger.info("Reporter should not be called when should_report=False")
        from orchestrator.exceptions import OrchestratorError, set_error_reporter

        mock_reporter = MagicMock()
        original = None
        try:
            from orchestrator.exceptions import _error_reporter
            original = _error_reporter
            set_error_reporter(mock_reporter)

            OrchestratorError("test error", should_report=False)
            mock_reporter.assert_not_called()
        finally:
            set_error_reporter(original)

    def test_set_error_reporter(self):
        logger.info("ExceptionErrorReporter: set error reporter")
        from orchestrator.exceptions import get_error_reporter, set_error_reporter

        original = get_error_reporter()
        try:
            sentinel = lambda e: None
            set_error_reporter(sentinel)
            assert get_error_reporter() is sentinel
        finally:
            set_error_reporter(original)

    def test_reporter_exception_does_not_propagate(self):
        """If the reporter itself raises, the exception init should still succeed."""
        logger.info("If the reporter itself raises, the exception init should still succeed")
        from orchestrator.exceptions import OrchestratorError, set_error_reporter

        def bad_reporter(e):
            raise RuntimeError("reporter broke")

        original = None
        try:
            from orchestrator.exceptions import _error_reporter
            original = _error_reporter
            set_error_reporter(bad_reporter)

            err = OrchestratorError("test error", should_report=True)
            assert err.message == "test error"
        finally:
            set_error_reporter(original)


class TestLLMClientWithoutLangfuse:
    def test_llm_client_works_without_langfuse(self):
        """LLMClient should initialize even when Langfuse setup fails."""
        logger.info("LLMClient should initialize even when Langfuse setup fails")
        with patch("orchestrator.llm.client.setup_langfuse", side_effect=Exception("no langfuse")):
            from orchestrator.llm import LLMClient

            client = LLMClient(enable_langfuse=True)
            assert client._langfuse_enabled is True

    def test_llm_client_works_with_langfuse_disabled(self):
        """LLMClient should skip Langfuse when disabled."""
        logger.info("LLMClient should skip Langfuse when disabled")
        with patch("orchestrator.llm.client.setup_langfuse") as mock_setup:
            from orchestrator.llm import LLMClient

            client = LLMClient(enable_langfuse=False)
            mock_setup.assert_not_called()
            assert client._langfuse_enabled is False
