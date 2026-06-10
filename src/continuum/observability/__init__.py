"""
Observability Module - Monitoring and tracing for the Orchestrator SDK.

Provides comprehensive observability features including:
- Multi-provider support (Langfuse, Google Vertex, etc.)
- Structured tracing with automatic provider routing
- Span management for LLM calls, tool invocations, and agent decisions
- Metrics collection (latency, token usage, error rates)
- Automatic error reporting to observability providers
- Decorators for easy instrumentation
- Async-safe trace context propagation
"""

from continuum.observability.config import ObservabilityConfig
from continuum.observability.decorators import (
    ObservationContext,
    observe,
    trace_agent,
    trace_tool,
)
from continuum.observability.error_reporter import (
    ErrorReporter,
    ErrorReportingContext,
    disable_error_reporting,
    enable_error_reporting,
    flush_errors,
    get_error_reporter,
    report_error,
    report_exception,
)
from continuum.observability.initialization import (
    initialize_observability,
    is_initialized,
    reset_initialization,
)
from continuum.observability.metrics import (
    MetricsCollector,
    get_metrics_collector,
    get_metrics_summary,
    initialize_metrics_collector,
    reset_metrics,
)
from continuum.observability.provider_manager import (
    ProviderManager,
    get_provider_manager,
)
from continuum.observability.providers.base import (
    ObservabilityProvider,
    ProviderCapabilities,
)
from continuum.observability.providers.langfuse import LangfuseProvider
from continuum.observability.providers.registry import (
    ProviderRegistry,
    get_provider,
    get_provider_registry,
    register_provider,
)
from continuum.observability.trace_context import (
    SpanScope,
    TraceContextToken,
    TraceScope,
    build_langfuse_metadata,
    clear_trace_context,
    get_current_agent_name,
    get_current_run_id,
    get_current_session_id,
    get_current_span_client,
    get_current_span_id,
    get_current_trace_client,
    get_current_trace_id,
    get_current_user_id,
    get_parent_client,
    get_parent_observation_id,
    get_trace_metadata,
    restore_trace_context,
    set_trace_context,
    traced_operation,
    truncate_data,
)
from continuum.observability.tracing import (
    GenerationSpan,
    Span,
    SpanLevel,
    Trace,
    TracingManager,
)

__all__ = [
    # Config
    "ObservabilityConfig",
    # Tracing
    "TracingManager",
    "Trace",
    "Span",
    "GenerationSpan",
    "SpanLevel",
    # Trace Context (Async-Safe)
    "TraceScope",
    "SpanScope",
    "TraceContextToken",
    "get_current_trace_id",
    "get_current_trace_client",
    "get_current_span_id",
    "get_current_span_client",
    "get_current_user_id",
    "get_current_session_id",
    "get_current_agent_name",
    "get_current_run_id",
    "get_parent_observation_id",
    "get_parent_client",
    "set_trace_context",
    "restore_trace_context",
    "clear_trace_context",
    "get_trace_metadata",
    "build_langfuse_metadata",
    "truncate_data",
    "traced_operation",
    # Decorators
    "observe",
    "trace_tool",
    "trace_agent",
    "ObservationContext",
    # Error Reporting
    "ErrorReporter",
    "ErrorReportingContext",
    "get_error_reporter",
    "report_error",
    "report_exception",
    "enable_error_reporting",
    "disable_error_reporting",
    "flush_errors",
    # Metrics
    "MetricsCollector",
    "get_metrics_collector",
    "get_metrics_summary",
    "initialize_metrics_collector",
    "reset_metrics",
    # Provider System
    "ObservabilityProvider",
    "ProviderCapabilities",
    "ProviderManager",
    "ProviderRegistry",
    "LangfuseProvider",
    "get_provider_manager",
    "get_provider",
    "get_provider_registry",
    "register_provider",
    "initialize_observability",
    "is_initialized",
    "reset_initialization",
]
