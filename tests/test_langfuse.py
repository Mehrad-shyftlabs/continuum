"""
Manual testing script for the Langfuse observability module.

Run this script to verify Langfuse integration is working correctly.

Usage:
    1. Start Langfuse locally: docker compose up -d (from langfuse repo)
    2. Copy .env.template to .env and add Langfuse API keys
    3. Run: python -m tests.test_langfuse

Or run individual tests:
    python -m tests.test_langfuse --test connection
    python -m tests.test_langfuse --test tracing
    python -m tests.test_langfuse --test decorators
    python -m tests.test_langfuse --test metrics
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

from orchestrator.config import settings


# =============================================================================
# Test Helpers
# =============================================================================


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"✅ {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"❌ {message}")


def print_info(message: str, end: str = "\n") -> None:
    """Print an info message."""
    print(f"ℹ️  {message}", end=end)


def print_config() -> None:
    """Print current Langfuse configuration."""
    print_header("Langfuse Configuration")
    print_info(f"Enabled: {settings.langfuse_enabled}")
    print_info(f"Host: {settings.langfuse_host}")
    print_info(f"Public Key: {settings.langfuse_public_key[:20] + '...' if settings.langfuse_public_key else 'Not set'}")
    print_info(f"Secret Key: {'***' if settings.langfuse_secret_key else 'Not set'}")
    print_info(f"Sample Rate: {settings.langfuse_sample_rate}")
    print_info(f"Environment: {settings.environment}")


# =============================================================================
# Tests
# =============================================================================


async def test_connection() -> bool:
    """Test Langfuse connection."""
    print_header("Testing Langfuse Connection")

    try:
        from orchestrator.observability import LangfuseClient

        client = LangfuseClient()

        print_info(f"Langfuse host: {client.config.host}")
        print_info(f"Client enabled: {client.is_enabled}")

        if not client.is_enabled:
            print_info("Langfuse not configured (check API keys)")
            return True  # Not a failure, just not configured

        # Test auth
        print_info("Testing authentication...")
        if client.auth_check():
            print_success("Langfuse connection successful!")
            return True
        else:
            print_error("Langfuse authentication failed")
            return False

    except Exception as e:
        print_error(f"Connection test failed: {e}")
        return False


async def test_basic_tracing() -> bool:
    """Test basic tracing functionality."""
    print_header("Testing Basic Tracing")

    try:
        from orchestrator.observability import LangfuseClient, TracingManager

        client = LangfuseClient()

        if not client.is_enabled:
            print_info("Langfuse not configured, skipping tracing test")
            return True

        manager = TracingManager(client.client)

        # Create a trace
        print_info("Creating test trace...")
        with manager.trace(
            "test-trace",
            user_id="test-user",
            session_id="test-session",
            metadata={"test": True},
            tags=["test", "manual"],
        ) as trace:
            print_info(f"Trace ID: {trace.id}")

            # Log an event
            trace.event("test-event", output={"status": "started"})

            # Create a span
            print_info("Creating test span...")
            with manager.span("test-span", input={"data": "test"}) as span:
                print_info(f"Span ID: {span.id}")
                span.end(output={"result": "success"})

            # Create a generation span
            print_info("Creating test generation...")
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

            # Add a score
            trace.score("test-score", 0.95, comment="Test score")

            # Update trace output
            trace.update(output={"final": "result"})

            # Get trace URL
            url = trace.get_trace_url()
            if url:
                print_info(f"Trace URL: {url}")

        # Flush
        client.flush()
        print_success("Basic tracing test passed!")
        return True

    except Exception as e:
        print_error(f"Tracing test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_decorators() -> bool:
    """Test tracing decorators."""
    print_header("Testing Tracing Decorators")

    try:
        from orchestrator.observability import (
            LangfuseClient,
            observe,
            trace_agent,
            trace_tool,
        )

        client = LangfuseClient()

        if not client.is_enabled:
            print_info("Langfuse not configured, skipping decorator test")
            return True

        # Test @observe decorator
        @observe(name="test-observe")
        def sync_function(x: int, y: int) -> int:
            return x + y

        @observe()
        async def async_function(message: str) -> str:
            await asyncio.sleep(0.1)
            return f"Received: {message}"

        # Test @trace_tool decorator
        @trace_tool(tool_type="calculator")
        def calculate(operation: str, a: float, b: float) -> float:
            if operation == "add":
                return a + b
            elif operation == "multiply":
                return a * b
            return 0

        # Test @trace_agent decorator
        @trace_agent(name="test-agent")
        async def run_test_agent(query: str, user_id: str = "test") -> str:
            await asyncio.sleep(0.1)
            return f"Response to: {query}"

        print_info("Testing @observe (sync)...")
        result = sync_function(2, 3)
        print_info(f"Result: {result}")

        print_info("Testing @observe (async)...")
        result = await async_function("Hello")
        print_info(f"Result: {result}")

        print_info("Testing @trace_tool...")
        result = calculate("add", 5, 3)
        print_info(f"Result: {result}")

        print_info("Testing @trace_agent...")
        result = await run_test_agent("What is Python?", user_id="test-user")
        print_info(f"Result: {result}")

        client.flush()
        print_success("Decorator tests passed!")
        return True

    except Exception as e:
        print_error(f"Decorator test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_metrics() -> bool:
    """Test metrics collection."""
    print_header("Testing Metrics Collection")

    try:
        from orchestrator.observability import (
            LangfuseClient,
            MetricsCollector,
            TracingManager,
        )

        client = LangfuseClient()
        metrics = MetricsCollector(client=client)

        # Test latency tracking
        print_info("Testing latency tracking...")
        with metrics.track_latency("test-operation"):
            await asyncio.sleep(0.1)

        with metrics.track_latency("another-operation"):
            await asyncio.sleep(0.05)

        # Test direct latency recording
        metrics.record_latency("manual-latency", 150.5)

        latency_stats = metrics.get_latency_stats()
        print_info(f"Latency stats: {latency_stats}")

        # Test token tracking
        print_info("Testing token tracking...")
        metrics.track_tokens(
            "llm-call-1",
            prompt_tokens=100,
            completion_tokens=50,
            model="gpt-4o",
        )
        metrics.track_tokens(
            "llm-call-2",
            prompt_tokens=200,
            completion_tokens=100,
            model="gpt-4o-mini",
        )

        token_stats = metrics.get_token_stats()
        print_info(f"Token stats: {token_stats}")

        # Test error tracking
        print_info("Testing error tracking...")
        try:
            raise ValueError("Test error")
        except Exception as e:
            metrics.track_error("test-operation", e)

        error_stats = metrics.get_error_stats()
        print_info(f"Error stats: {error_stats}")

        # Test custom metrics
        print_info("Testing custom metrics...")
        metrics.record_metric("items_processed", 42)
        metrics.increment("api_calls")
        metrics.increment("api_calls")

        # Get full summary
        summary = metrics.get_summary()
        print_info(f"Full summary: {summary}")

        # Report to trace if Langfuse is enabled
        if client.is_enabled:
            print_info("Reporting metrics to trace...")
            manager = TracingManager(client.client)
            with manager.trace("metrics-test") as trace:
                metrics.report_to_trace(trace)

            client.flush()

        print_success("Metrics tests passed!")
        return True

    except Exception as e:
        print_error(f"Metrics test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_context_manager() -> bool:
    """Test ObservationContext."""
    print_header("Testing ObservationContext")

    try:
        from orchestrator.observability import LangfuseClient, TracingManager
        from orchestrator.observability.decorators import ObservationContext

        client = LangfuseClient()

        if not client.is_enabled:
            print_info("Langfuse not configured, skipping context test")
            return True

        manager = TracingManager(client.client)

        print_info("Testing ObservationContext...")
        with manager.trace("context-test") as trace:
            with ObservationContext("manual-observation", manager=manager) as ctx:
                ctx.set_input({"query": "test"})
                ctx.add_metadata({"step": 1})

                # Simulate work
                await asyncio.sleep(0.1)

                ctx.set_output({"result": "success"})

        client.flush()
        print_success("Context manager test passed!")
        return True

    except Exception as e:
        print_error(f"Context manager test failed: {e}")
        return False


async def test_prompts_and_datasets() -> bool:
    """Test prompt and dataset management."""
    print_header("Testing Prompts and Datasets")

    try:
        from orchestrator.observability import LangfuseClient

        client = LangfuseClient()

        if not client.is_enabled:
            print_info("Langfuse not configured, skipping prompt/dataset test")
            return True

        # Test prompt creation
        print_info("Creating test prompt...")
        prompt = client.create_prompt(
            name="test-prompt",
            prompt="You are a helpful assistant. User query: {{query}}",
            config={"temperature": 0.7},
            labels=["test"],
        )
        if prompt:
            print_info(f"Prompt created: {prompt}")

        # Test getting prompt
        print_info("Getting test prompt...")
        retrieved = client.get_prompt("test-prompt")
        if retrieved:
            print_info(f"Prompt retrieved: {retrieved}")

        # Test dataset creation
        print_info("Creating test dataset...")
        dataset = client.create_dataset(
            name="test-dataset",
            description="Test dataset for SDK testing",
            metadata={"type": "test"},
        )
        if dataset:
            print_info(f"Dataset created: {dataset}")

        # Add dataset item
        print_info("Adding dataset item...")
        item = client.create_dataset_item(
            dataset_name="test-dataset",
            input={"query": "What is Python?"},
            expected_output="Python is a programming language...",
            metadata={"difficulty": "easy"},
        )
        if item:
            print_info(f"Dataset item created: {item}")

        client.flush()
        print_success("Prompts and datasets test passed!")
        return True

    except Exception as e:
        print_error(f"Prompts/datasets test failed: {e}")
        return False


async def test_with_llm() -> bool:
    """Test integration with LLM client."""
    print_header("Testing LLM Integration")

    try:
        from orchestrator.llm import ChatMessage, LLMClient
        from orchestrator.observability import LangfuseClient, TracingManager

        # Initialize clients
        langfuse_client = LangfuseClient()
        llm_client = LLMClient(enable_langfuse=langfuse_client.is_enabled)

        if not langfuse_client.is_enabled:
            print_info("Langfuse not configured, testing LLM without tracing")

        manager = TracingManager(langfuse_client.client if langfuse_client.is_enabled else None)

        print_info("Testing LLM call with tracing...")
        with manager.trace(
            "llm-integration-test",
            user_id="test-user",
            metadata={"test": True},
        ) as trace:
            messages = [
                ChatMessage(role="user", content="Say 'hello' and nothing else."),
            ]

            # Make LLM call
            print_info("Making LLM call...")
            response = await llm_client.chat(messages)
            print_info(f"Response: {response.content}")

            # Add score
            trace.score("test_integration", 1.0, comment="Integration test successful")
            trace.update(output={"response": response.content})

            if langfuse_client.is_enabled:
                url = trace.get_trace_url()
                if url:
                    print_info(f"View trace: {url}")

        if langfuse_client.is_enabled:
            langfuse_client.flush()

        print_success("LLM integration test passed!")
        return True

    except Exception as e:
        print_error(f"LLM integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def run_all_tests() -> None:
    """Run all tests."""
    print_header("Orchestrator SDK - Langfuse Observability Tests")
    print("Running comprehensive tests for the observability module...")

    # Print configuration
    print_config()

    results = {
        "Connection": await test_connection(),
        "Basic Tracing": await test_basic_tracing(),
        "Decorators": await test_decorators(),
        "Metrics": await test_metrics(),
        "Context Manager": await test_context_manager(),
        "Prompts & Datasets": await test_prompts_and_datasets(),
        "LLM Integration": await test_with_llm(),
    }

    # Summary
    print_header("Test Results Summary")
    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")

    print()
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print_success("All tests passed!")
    else:
        print_error(f"{total - passed} test(s) failed")


async def run_specific_test(test_name: str) -> None:
    """Run a specific test."""
    tests = {
        "connection": test_connection,
        "tracing": test_basic_tracing,
        "decorators": test_decorators,
        "metrics": test_metrics,
        "context": test_context_manager,
        "prompts": test_prompts_and_datasets,
        "llm": test_with_llm,
    }

    # Print config before running
    print_config()

    if test_name in tests:
        await tests[test_name]()
    else:
        print_error(f"Unknown test: {test_name}")
        print_info(f"Available tests: {', '.join(tests.keys())}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test the Langfuse observability module")
    parser.add_argument(
        "--test",
        type=str,
        help="Run a specific test (connection, tracing, decorators, metrics, context, prompts, llm)",
    )
    args = parser.parse_args()

    if args.test:
        asyncio.run(run_specific_test(args.test))
    else:
        asyncio.run(run_all_tests())


if __name__ == "__main__":
    main()

