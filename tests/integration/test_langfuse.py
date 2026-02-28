"""
Integration tests for Langfuse observability module.

Requires Langfuse to be configured.

Converted from tests/test_langfuse.py manual test script.
"""

import asyncio
import os

import pytest
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


load_dotenv()

from orchestrator.config import settings


pytestmark = [pytest.mark.integration]

skip_no_langfuse = pytest.mark.skipif(
    not settings.langfuse_enabled,
    reason="Langfuse not configured",
)


@skip_no_langfuse
class TestLangfuseConnection:
    def test_client_creation(self):
        logger.info("LangfuseConnection: client creation")
        from orchestrator.observability import LangfuseClient

        client = LangfuseClient()
        assert client.is_enabled

    def test_auth_check(self):
        logger.info("LangfuseConnection: auth check")
        from orchestrator.observability import LangfuseClient

        client = LangfuseClient()
        if client.is_enabled:
            assert client.auth_check()


@skip_no_langfuse
class TestBasicTracing:
    def test_create_trace(self):
        logger.info("BasicTracing: create trace")
        from orchestrator.observability import LangfuseClient, TracingManager

        client = LangfuseClient()
        if not client.is_enabled:
            pytest.skip("Langfuse not enabled")

        manager = TracingManager(client.client)

        with manager.trace(
            "test-trace",
            user_id="test-user",
            session_id="test-session",
            metadata={"test": True},
            tags=["test", "automated"],
        ) as trace:
            assert trace.id is not None
            trace.event("test-event", output={"status": "started"})

            with manager.span("test-span", input={"data": "test"}) as span:
                assert span.id is not None
                span.end(output={"result": "success"})

            gen = trace.generation(
                "test-generation",
                model="test-model",
                input=[{"role": "user", "content": "Hello"}],
            )
            gen.end(
                output="Hello! How can I help?",
                usage_prompt_tokens=10,
                usage_completion_tokens=8,
            )

            trace.score("test-score", 0.95, comment="Test score")
            trace.update(output={"final": "result"})

        client.flush()


@skip_no_langfuse
class TestDecorators:
    async def test_observe_decorator(self):
        logger.info("Decorators: observe decorator")
        from orchestrator.observability import LangfuseClient, observe

        client = LangfuseClient()
        if not client.is_enabled:
            pytest.skip("Langfuse not enabled")

        @observe(name="test-observe")
        def sync_function(x: int, y: int) -> int:
            return x + y

        @observe()
        async def async_function(message: str) -> str:
            await asyncio.sleep(0.01)
            return f"Received: {message}"

        result = sync_function(2, 3)
        assert result == 5

        result = await async_function("Hello")
        assert result == "Received: Hello"

        client.flush()

    async def test_trace_tool_decorator(self):
        logger.info("Decorators: trace tool decorator")
        from orchestrator.observability import LangfuseClient, trace_tool

        client = LangfuseClient()
        if not client.is_enabled:
            pytest.skip("Langfuse not enabled")

        @trace_tool(tool_type="calculator")
        def calculate(operation: str, a: float, b: float) -> float:
            if operation == "add":
                return a + b
            elif operation == "multiply":
                return a * b
            return 0

        result = calculate("add", 5, 3)
        assert result == 8

        client.flush()

    async def test_trace_agent_decorator(self):
        logger.info("Decorators: trace agent decorator")
        from orchestrator.observability import LangfuseClient, trace_agent

        client = LangfuseClient()
        if not client.is_enabled:
            pytest.skip("Langfuse not enabled")

        @trace_agent(name="test-agent")
        async def run_test_agent(query: str, user_id: str = "test") -> str:
            await asyncio.sleep(0.01)
            return f"Response to: {query}"

        result = await run_test_agent("What is Python?", user_id="test-user")
        assert result == "Response to: What is Python?"

        client.flush()


@skip_no_langfuse
class TestMetrics:
    async def test_latency_tracking(self):
        logger.info("Metrics: latency tracking")
        from orchestrator.observability import LangfuseClient, MetricsCollector

        client = LangfuseClient()
        metrics = MetricsCollector(client=client)

        with metrics.track_latency("test-operation"):
            await asyncio.sleep(0.01)

        latency_stats = metrics.get_latency_stats()
        assert isinstance(latency_stats, dict)

    def test_token_tracking(self):
        logger.info("Metrics: token tracking")
        from orchestrator.observability import LangfuseClient, MetricsCollector

        client = LangfuseClient()
        metrics = MetricsCollector(client=client)

        metrics.track_tokens(
            "llm-call-1",
            prompt_tokens=100,
            completion_tokens=50,
            model="gpt-4o",
        )

        token_stats = metrics.get_token_stats()
        assert isinstance(token_stats, dict)

    def test_error_tracking(self):
        logger.info("Metrics: error tracking")
        from orchestrator.observability import LangfuseClient, MetricsCollector

        client = LangfuseClient()
        metrics = MetricsCollector(client=client)

        try:
            raise ValueError("Test error")
        except Exception as e:
            metrics.track_error("test-operation", e)

        error_stats = metrics.get_error_stats()
        assert isinstance(error_stats, dict)

    def test_custom_metrics(self):
        logger.info("Metrics: custom metrics")
        from orchestrator.observability import LangfuseClient, MetricsCollector

        client = LangfuseClient()
        metrics = MetricsCollector(client=client)

        metrics.record_metric("items_processed", 42)
        metrics.increment("api_calls")
        metrics.increment("api_calls")

        summary = metrics.get_summary()
        assert isinstance(summary, dict)
