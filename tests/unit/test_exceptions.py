"""Unit tests for orchestrator exceptions module."""

import pytest

from orchestrator.exceptions import (
    ConfigurationError,
    ErrorCategory,
    ErrorSeverity,
    LangfuseError,
    NetworkError,
    ObservabilityError,
    OrchestratorError,
    ProviderError,
    TracingError,
    ValidationError,
    set_error_reporter,
    wrap_exception,
)
import logging

logger = logging.getLogger(__name__)


class TestOrchestratorError:
    def test_orchestrator_error_defaults(self):
        logger.info("OrchestratorError: orchestrator error defaults")
        err = OrchestratorError(should_report=False)
        assert err.message == "An error occurred in the Orchestrator SDK"
        assert err.error_code == "ORCHESTRATOR_ERROR"
        assert err.category == ErrorCategory.INTERNAL
        assert err.severity == ErrorSeverity.HIGH

    def test_orchestrator_error_custom(self):
        logger.info("OrchestratorError: orchestrator error custom")
        err = OrchestratorError(
            "custom msg",
            error_code="CUSTOM",
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.LOW,
            context={"key": "val"},
            trace_id="t1",
            span_id="s1",
            should_report=False,
        )
        assert err.message == "custom msg"
        assert err.error_code == "CUSTOM"
        assert err.context["key"] == "val"
        assert err.trace_id == "t1"
        assert err.span_id == "s1"

    def test_orchestrator_error_to_dict(self):
        logger.info("OrchestratorError: orchestrator error to dict")
        err = OrchestratorError("test", trace_id="t1", span_id="s1", should_report=False)
        d = err.to_dict()
        assert d["error_type"] == "OrchestratorError"
        assert d["error_code"] == "ORCHESTRATOR_ERROR"
        assert "trace_id" in d
        assert "span_id" in d
        assert "timestamp" in d

    def test_orchestrator_error_str(self):
        logger.info("OrchestratorError: orchestrator error str")
        err = OrchestratorError("test msg", should_report=False)
        s = str(err)
        assert "ORCHESTRATOR_ERROR" in s
        assert "test msg" in s

    def test_orchestrator_error_repr(self):
        logger.info("OrchestratorError: orchestrator error repr")
        err = OrchestratorError("test", should_report=False)
        r = repr(err)
        assert "OrchestratorError" in r

    def test_orchestrator_error_with_original(self):
        logger.info("OrchestratorError: orchestrator error with original")
        orig = ValueError("original")
        err = OrchestratorError("wrapper", original_error=orig, should_report=False)
        assert err.original_error is orig
        d = err.to_dict()
        assert d["original_error"]["type"] == "ValueError"

    def test_orchestrator_error_get_traceback(self):
        logger.info("OrchestratorError: orchestrator error get traceback")
        try:
            raise ValueError("boom")
        except ValueError as e:
            err = OrchestratorError("wrapped", original_error=e, should_report=False)
            tb = err.get_traceback()
            assert "ValueError" in tb

    def test_orchestrator_error_get_traceback_none(self):
        logger.info("OrchestratorError: orchestrator error get traceback none")
        err = OrchestratorError("no orig", should_report=False)
        assert err.get_traceback() is None


class TestSubclassErrors:
    def test_configuration_error(self):
        logger.info("SubclassErrors: configuration error")
        err = ConfigurationError("missing key", config_key="API_KEY", expected_type="str", should_report=False)
        assert err.error_code == "CONFIG_ERROR"
        assert err.category == ErrorCategory.CONFIGURATION
        assert err.context["config_key"] == "API_KEY"

    def test_validation_error(self):
        logger.info("SubclassErrors: validation error")
        err = ValidationError("bad input", field="name", value="x" * 200, expected="string", should_report=False)
        assert err.error_code == "VALIDATION_ERROR"
        assert err.context["field"] == "name"
        assert len(err.context["value"]) <= 100

    def test_observability_error(self):
        logger.info("SubclassErrors: observability error")
        err = ObservabilityError("obs err", should_report=False)
        assert err.severity == ErrorSeverity.MEDIUM

    def test_langfuse_error_no_self_report(self):
        logger.info("SubclassErrors: langfuse error no self report")
        err = LangfuseError("langfuse err", operation="create_trace")
        assert err.should_report is False
        assert err.context["operation"] == "create_trace"

    def test_tracing_error(self):
        logger.info("SubclassErrors: tracing error")
        err = TracingError("trace err", trace_name="t", span_name="s", should_report=False)
        assert err.context["trace_name"] == "t"
        assert err.context["span_name"] == "s"

    def test_network_error(self):
        logger.info("SubclassErrors: network error")
        err = NetworkError("net err", url="http://x", status_code=500, should_report=False)
        assert err.context["url"] == "http://x"
        assert err.context["status_code"] == 500

    def test_provider_error(self):
        logger.info("SubclassErrors: provider error")
        err = ProviderError("prov err", provider="openai", provider_error_code="429", should_report=False)
        assert err.context["provider"] == "openai"


class TestWrapException:
    def test_wrap_standard_exception(self):
        logger.info("WrapException: wrap standard exception")
        orig = ValueError("val err")
        wrapped = wrap_exception(orig, ValidationError, field="x")
        assert isinstance(wrapped, ValidationError)
        assert wrapped.original_error is orig

    def test_wrap_already_orchestrator_error(self):
        logger.info("WrapException: wrap already orchestrator error")
        orig = ConfigurationError("already wrapped", should_report=False)
        result = wrap_exception(orig, ValidationError)
        assert result is orig


class TestErrorEnums:
    def test_error_severity_values(self):
        logger.info("ErrorEnums: error severity values")
        assert ErrorSeverity.LOW == "low"
        assert ErrorSeverity.MEDIUM == "medium"
        assert ErrorSeverity.HIGH == "high"
        assert ErrorSeverity.CRITICAL == "critical"

    def test_error_category_values(self):
        logger.info("ErrorEnums: error category values")
        assert ErrorCategory.CONFIGURATION == "configuration"
        assert ErrorCategory.AUTHENTICATION == "authentication"
        assert ErrorCategory.RATE_LIMIT == "rate_limit"
        assert ErrorCategory.TIMEOUT == "timeout"
        assert ErrorCategory.VALIDATION == "validation"
        assert ErrorCategory.NETWORK == "network"
        assert ErrorCategory.PROVIDER == "provider"
        assert ErrorCategory.INTERNAL == "internal"
        assert ErrorCategory.OBSERVABILITY == "observability"
        assert ErrorCategory.UNKNOWN == "unknown"
