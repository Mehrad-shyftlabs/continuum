# Observability Module

Tracing, metrics, and error reporting with Langfuse integration.

## Overview

- **TracingManager**: Structured tracing for agent execution
- **MetricsCollector**: Latency, token usage, error rates
- **Error Reporting**: Automatic error reporting to Langfuse
- **Decorators**: Easy instrumentation with `@observe`

## Tracing

```python
from orchestrator.observability import TracingManager

manager = TracingManager()

# Create trace
trace = manager.create_trace(
    name="agent-run",
    user_id="user-123",
    session_id="session-456",
    input={"query": "Hello"},
)

# Create span
span = trace.create_span(
    name="llm-call",
    level=SpanLevel.DEFAULT,
)

# Update span
span.update(
    output={"response": "Hi there!"},
    metadata={"model": "gpt-4o"},
)
```

## Decorators

Automatically trace functions:

```python
from orchestrator.observability import observe

@observe(name="my_function", capture_output=True)
async def my_function(input: str) -> str:
    # Function is automatically traced
    return "result"
```

## Metrics

```python
from orchestrator.observability import get_metrics_collector

metrics = get_metrics_collector()

# Record latency
metrics.record_latency("operation", 150.5, metadata={"type": "llm"})

# Track tokens
metrics.track_tokens(
    "llm_call",
    prompt_tokens=100,
    completion_tokens=50,
    model="gpt-4o",
)

# Track errors
metrics.track_error("operation", exception, metadata={})
```

## Error Reporting

```python
from orchestrator.observability import report_error

try:
    # ... code ...
except Exception as e:
    report_error(
        e,
        context="agent_execution",
        trace_id=trace_id,
        metadata={"agent": "my-agent"},
    )
```

## Trace Context

Async-safe trace context propagation:

```python
from orchestrator.observability import (
    set_trace_context,
    get_current_trace_id,
    clear_trace_context,
)

# Set context
set_trace_context(
    trace_id="trace-123",
    user_id="user-123",
    session_id="session-456",
)

# Get current trace ID (anywhere in async call chain)
trace_id = get_current_trace_id()

# Clear context
clear_trace_context()
```

## Configuration

```python
from orchestrator.observability import ObservabilityConfig, initialize_observability

config = ObservabilityConfig(
    langfuse_enabled=True,
    langfuse_public_key="pk-...",
    langfuse_secret_key="sk-...",
    langfuse_host="http://localhost:3000",
)

initialize_observability(config)
```

**Note**: Langfuse v2.x is required for LiteLLM compatibility. The SDK automatically pins to `langfuse>=2.57.0,<3.0.0`.

## Types

- `Trace`: Trace object
- `Span`: Span object
- `GenerationSpan`: LLM generation span
- `SpanLevel`: Span level (DEFAULT, LLM, TOOL, etc.)

## Health Check

```bash
python scripts/health_check.py
```
