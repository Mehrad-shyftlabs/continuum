"""
Manual testing script for the LLM module.

Run this script to verify the SDK is working correctly with your configured providers.

Usage:
    1. Copy .env.template to .env and fill in your API keys
    2. Run: python -m tests.test_llm

Or run individual tests:
    python -m tests.test_llm --test openai
    python -m tests.test_llm --test gemini
    python -m tests.test_llm --test streaming
    python -m tests.test_llm --test function_calling
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

from orchestrator.config import settings
from orchestrator.llm import (
    ChatMessage,
    LLMClient,
    LLMConfig,
    ToolDefinition,
    FunctionDefinition,
)

# =============================================================================
# Test Configuration from Environment Variables
# =============================================================================

# Model names from env (with defaults)
OPENAI_MODEL = os.getenv("TEST_OPENAI_MODEL", "gpt-4o-mini")
GEMINI_MODEL = os.getenv("TEST_GEMINI_MODEL", "gemini/gemini-2.5-pro")

# Max tokens from env
TEST_MAX_TOKENS = int(os.getenv("DEFAULT_LLM_MAX_TOKENS", "100"))


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
    """Print current test configuration."""
    print_header("Test Configuration")
    print_info(f"OpenAI Model: {OPENAI_MODEL}")
    print_info(f"Gemini Model: {GEMINI_MODEL}")
    print_info(f"Test Max Tokens: {TEST_MAX_TOKENS}")
    print_info(f"Default Temperature: {settings.default_llm_temperature}")
    print_info(f"Fallback Enabled: {settings.llm_enable_fallback}")
    print_info(f"Langfuse Enabled: {settings.langfuse_enabled}")


async def test_openai_basic() -> bool:
    """Test basic OpenAI completion."""
    print_header("Testing OpenAI Basic Completion")

    try:
        client = LLMClient(
            config=LLMConfig(
                model=OPENAI_MODEL,
                max_tokens=TEST_MAX_TOKENS,
                temperature=settings.default_llm_temperature,
            ),
            enable_langfuse=False,  # Disable for basic testing
        )

        messages = [
            ChatMessage(role="system", content="You are a helpful assistant. Be concise."),
            ChatMessage(role="user", content="What is 2 + 2? Answer in one word."),
        ]

        print_info(f"Using model: {OPENAI_MODEL}")
        print_info("Sending request to OpenAI...")
        response = await client.chat(messages)

        print_info(f"Model: {response.model}")
        print_info(f"Response: {response.content}")
        print_info(f"Tokens used: {response.usage.total_tokens if response.usage else 'N/A'}")
        print_info(f"Finish reason: {response.finish_reason}")

        print_success("OpenAI basic completion working!")
        return True

    except Exception as e:
        print_error(f"OpenAI test failed: {e}")
        return False


async def test_gemini_basic() -> bool:
    """Test basic Gemini completion."""
    print_header("Testing Google Gemini Basic Completion")

    try:
        client = LLMClient(
            config=LLMConfig(
                model=GEMINI_MODEL,
                max_tokens=TEST_MAX_TOKENS,
                temperature=settings.default_llm_temperature,
            ),
            enable_langfuse=False,
        )

        messages = [
            ChatMessage(role="user", content="What is the capital of France? Answer in one word."),
        ]

        print_info(f"Using model: {GEMINI_MODEL}")
        print_info("Sending request to Google Gemini...")
        response = await client.chat(messages)

        print_info(f"Model: {response.model}")
        print_info(f"Response: {response.content}")
        print_info(f"Tokens used: {response.usage.total_tokens if response.usage else 'N/A'}")

        print_success("Gemini basic completion working!")
        return True

    except Exception as e:
        print_error(f"Gemini test failed: {e}")
        return False


async def test_streaming_openai() -> bool:
    """Test streaming with OpenAI."""
    print_header("Testing OpenAI Streaming")

    try:
        client = LLMClient(
            config=LLMConfig(
                model=OPENAI_MODEL,
                max_tokens=TEST_MAX_TOKENS,
                temperature=settings.default_llm_temperature,
            ),
            enable_langfuse=False,
        )

        messages = [
            ChatMessage(role="user", content="Count from 1 to 5, one number per line."),
        ]

        print_info(f"Using model: {OPENAI_MODEL}")
        print_info("Starting stream from OpenAI...")
        print_info("Response: ", end="")

        full_content = ""
        async for chunk in client.chat_stream(messages):
            if chunk.content:
                print(chunk.content, end="", flush=True)
                full_content += chunk.content

        print()  # New line after stream
        print_info(f"Stream finished. Total characters: {len(full_content)}")
        print_success("OpenAI streaming working!")
        return True

    except Exception as e:
        print_error(f"OpenAI streaming test failed: {e}")
        return False


async def test_streaming_gemini() -> bool:
    """Test streaming with Gemini."""
    print_header("Testing Gemini Streaming")

    try:
        client = LLMClient(
            config=LLMConfig(
                model=GEMINI_MODEL,
                max_tokens=TEST_MAX_TOKENS,
                temperature=settings.default_llm_temperature,
            ),
            enable_langfuse=False,
        )

        messages = [
            ChatMessage(role="user", content="List 3 colors, one per line."),
        ]

        print_info(f"Using model: {GEMINI_MODEL}")
        print_info("Starting stream from Gemini...")
        print_info("Response: ", end="")

        full_content = ""
        async for chunk in client.chat_stream(messages):
            if chunk.content:
                print(chunk.content, end="", flush=True)
                full_content += chunk.content

        print()
        print_info(f"Stream finished. Total characters: {len(full_content)}")
        print_success("Gemini streaming working!")
        return True

    except Exception as e:
        print_error(f"Gemini streaming test failed: {e}")
        return False


async def test_function_calling() -> bool:
    """Test function/tool calling."""
    print_header("Testing Function Calling")

    # Function calling needs more tokens - use at least 500
    func_call_max_tokens = max(TEST_MAX_TOKENS, 500)
    
    try:
        client = LLMClient(
            config=LLMConfig(
                model=OPENAI_MODEL,
                max_tokens=func_call_max_tokens,
                temperature=settings.default_llm_temperature,
            ),
            enable_langfuse=False,
        )

        # Define a weather tool
        weather_tool = ToolDefinition(
            type="function",
            function=FunctionDefinition(
                name="get_weather",
                description="Get the current weather for a location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit",
                        },
                    },
                    "required": ["location"],
                },
            ),
        )

        messages = [
            ChatMessage(role="user", content="What's the weather like in New York?"),
        ]

        print_info(f"Using model: {OPENAI_MODEL}")
        print_info(f"Max tokens for function calling: {func_call_max_tokens}")
        print_info("Sending request with tool definition...")
        
        # Use tool_choice to encourage/require tool usage
        response = await client.chat(
            messages, 
            tools=[weather_tool],
            tool_choice="auto"  # Can also use "required" to force tool use
        )

        print_info(f"Model: {response.model}")
        print_info(f"Finish reason: {response.finish_reason}")

        if response.tool_calls:
            print_info(f"Tool calls received: {len(response.tool_calls)}")
            for tc in response.tool_calls:
                print_info(f"  - Function: {tc.function.name}")
                print_info(f"    Arguments: {tc.function.arguments}")

                # Parse and display arguments
                args = json.loads(tc.function.arguments)
                print_info(f"    Parsed: {args}")

            print_success("Function calling working!")
            return True
        else:
            print_info(f"Response content: {response.content}")
            print_error("No tool calls in response (model didn't use the tool)")
            return False

    except Exception as e:
        print_error(f"Function calling test failed: {e}")
        return False


async def test_fallback() -> bool:
    """Test fallback mechanism."""
    print_header("Testing Fallback Mechanism")

    try:
        # Configure with a non-existent primary model
        fallback_model = OPENAI_MODEL
        client = LLMClient(
            config=LLMConfig(
                model="non-existent-model",
                fallback_models=[fallback_model],
                enable_fallback=True,
                max_tokens=TEST_MAX_TOKENS,
                temperature=settings.default_llm_temperature,
            ),
            enable_langfuse=False,
        )

        messages = [
            ChatMessage(role="user", content="Say 'fallback works'"),
        ]

        print_info(f"Testing fallback from non-existent model to {fallback_model}...")
        response = await client.chat(messages)

        print_info(f"Response model: {response.model}")
        print_info(f"Response: {response.content}")

        print_success("Fallback mechanism working!")
        return True

    except Exception as e:
        print_error(f"Fallback test failed: {e}")
        return False


async def test_sync_completion() -> bool:
    """Test synchronous completion."""
    print_header("Testing Synchronous Completion")

    try:
        client = LLMClient(
            config=LLMConfig(
                model=OPENAI_MODEL,
                max_tokens=TEST_MAX_TOKENS,
                temperature=settings.default_llm_temperature,
            ),
            enable_langfuse=False,
        )

        messages = [
            ChatMessage(role="user", content="Say 'sync works' in exactly those words."),
        ]

        print_info(f"Using model: {OPENAI_MODEL}")
        print_info("Testing synchronous completion...")
        response = client.chat_sync(messages)

        print_info(f"Model: {response.model}")
        print_info(f"Response: {response.content}")

        print_success("Synchronous completion working!")
        return True

    except Exception as e:
        print_error(f"Sync completion test failed: {e}")
        return False


async def test_json_mode() -> bool:
    """Test JSON mode."""
    print_header("Testing JSON Mode")

    try:
        client = LLMClient(
            config=LLMConfig(
                model=OPENAI_MODEL,
                max_tokens=TEST_MAX_TOKENS,
                temperature=settings.default_llm_temperature,
                json_mode=True,
            ),
            enable_langfuse=False,
        )

        messages = [
            ChatMessage(
                role="system",
                content="You are a helpful assistant that responds in JSON format.",
            ),
            ChatMessage(
                role="user",
                content="Give me a JSON object with name='test' and value=42",
            ),
        ]

        print_info(f"Using model: {OPENAI_MODEL}")
        print_info("Testing JSON mode...")
        response = await client.chat(messages)

        print_info(f"Response: {response.content}")

        # Try to parse as JSON
        if response.content:
            parsed = json.loads(response.content)
            print_info(f"Parsed JSON: {parsed}")
            print_success("JSON mode working!")
            return True
        else:
            print_error("No content in response")
            return False

    except json.JSONDecodeError:
        print_error("Response is not valid JSON")
        return False
    except Exception as e:
        print_error(f"JSON mode test failed: {e}")
        return False


async def test_model_info() -> bool:
    """Test model information utilities."""
    print_header("Testing Model Info Utilities")

    try:
        client = LLMClient(enable_langfuse=False)

        # Test get_model_info
        print_info(f"Getting model info for {OPENAI_MODEL}...")
        info = client.get_model_info(OPENAI_MODEL)
        if info:
            print_info(f"Model info: {info}")
        else:
            print_info("No model info available")

        # Test get_max_tokens
        max_tokens = client.get_max_tokens(OPENAI_MODEL)
        print_info(f"Max tokens for {OPENAI_MODEL}: {max_tokens}")

        # Test count_tokens
        messages = [
            ChatMessage(role="user", content="Hello, how are you doing today?"),
        ]
        token_count = client.count_tokens(messages, OPENAI_MODEL)
        print_info(f"Token count for test message: {token_count}")

        print_success("Model info utilities working!")
        return True

    except Exception as e:
        print_error(f"Model info test failed: {e}")
        return False


async def test_langfuse_integration() -> bool:
    """Test Langfuse integration."""
    print_header("Testing Langfuse Integration")

    try:
        from orchestrator.llm.callbacks import setup_langfuse, get_langfuse_callback

        print_info("Setting up Langfuse...")
        success = setup_langfuse()

        if success:
            callback = get_langfuse_callback()
            print_info(f"Langfuse callback: {callback}")

            # Make a test call
            client = LLMClient(
                config=LLMConfig(
                    model=OPENAI_MODEL,
                    max_tokens=TEST_MAX_TOKENS,
                    temperature=settings.default_llm_temperature,
                ),
                enable_langfuse=True,
            )

            messages = [
                ChatMessage(role="user", content="Say 'langfuse test'"),
            ]

            print_info(f"Using model: {OPENAI_MODEL}")
            print_info("Making test call with Langfuse logging...")
            response = await client.chat(messages)
            print_info(f"Response: {response.content}")

            print_success("Langfuse integration working!")
            return True
        else:
            print_info("Langfuse not configured (check LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)")
            return True  # Not a failure, just not configured

    except Exception as e:
        print_error(f"Langfuse test failed: {e}")
        return False


async def test_default_config() -> bool:
    """Test using default configuration from environment."""
    print_header("Testing Default Configuration from Environment")

    try:
        # Use default config from settings - no overrides
        client = LLMClient(enable_langfuse=False)

        messages = [
            ChatMessage(role="user", content="Say 'defaults work'"),
        ]

        print_info(f"Default model: {settings.default_llm_model}")
        print_info(f"Default max tokens: {settings.default_llm_max_tokens}")
        print_info(f"Default temperature: {settings.default_llm_temperature}")
        print_info("Testing with default configuration...")

        response = await client.chat(messages)

        print_info(f"Model used: {response.model}")
        print_info(f"Response: {response.content}")

        print_success("Default configuration working!")
        return True

    except Exception as e:
        print_error(f"Default config test failed: {e}")
        return False


async def run_all_tests() -> None:
    """Run all tests."""
    print_header("Orchestrator SDK - LLM Module Tests")
    print("Running comprehensive tests for the LLM module...")
    
    # Print configuration first
    print_config()

    results = {
        "Default Config": await test_default_config(),
        "OpenAI Basic": await test_openai_basic(),
        "Gemini Basic": await test_gemini_basic(),
        "OpenAI Streaming": await test_streaming_openai(),
        "Gemini Streaming": await test_streaming_gemini(),
        "Function Calling": await test_function_calling(),
        "Fallback": await test_fallback(),
        "Sync Completion": await test_sync_completion(),
        "JSON Mode": await test_json_mode(),
        "Model Info": await test_model_info(),
        "Langfuse": await test_langfuse_integration(),
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
        "default": test_default_config,
        "openai": test_openai_basic,
        "gemini": test_gemini_basic,
        "streaming": test_streaming_openai,
        "streaming_gemini": test_streaming_gemini,
        "function_calling": test_function_calling,
        "fallback": test_fallback,
        "sync": test_sync_completion,
        "json": test_json_mode,
        "model_info": test_model_info,
        "langfuse": test_langfuse_integration,
    }

    # Print config before running test
    print_config()

    if test_name in tests:
        await tests[test_name]()
    else:
        print_error(f"Unknown test: {test_name}")
        print_info(f"Available tests: {', '.join(tests.keys())}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test the Orchestrator SDK LLM module")
    parser.add_argument(
        "--test",
        type=str,
        help="Run a specific test (default, openai, gemini, streaming, function_calling, fallback, sync, json, model_info, langfuse)",
    )
    args = parser.parse_args()

    if args.test:
        asyncio.run(run_specific_test(args.test))
    else:
        asyncio.run(run_all_tests())


if __name__ == "__main__":
    main()
