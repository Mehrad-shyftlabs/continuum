"""
Pytest fixtures for LLM module tests.
"""

import os
import pytest
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from orchestrator.llm import LLMClient, LLMConfig
from orchestrator.config import settings


# Override pytest-asyncio's event_loop fixture to be session-scoped
# This ensures all tests use the same event loop, preventing cleanup issues
@pytest.fixture(scope="session")
def event_loop():
    """
    Create a session-scoped event loop for all async tests.
    
    This ensures LiteLLM's async clients are properly cleaned up
    before the event loop closes, preventing RuntimeWarnings.
    """
    loop = asyncio.new_event_loop()
    yield loop
    
    # Cleanup LiteLLM async clients BEFORE closing the event loop
    # This prevents the RuntimeWarning about unawaited coroutines
    try:
        import litellm
        if hasattr(litellm, "close_litellm_async_clients"):
            # Run cleanup synchronously before loop closes
            if not loop.is_closed():
                try:
                    loop.run_until_complete(litellm.close_litellm_async_clients())
                except Exception:
                    # Ignore cleanup errors - resources will be freed on process exit
                    pass
    except Exception:
        pass
    
    # Close the event loop
    loop.close()


@pytest.fixture
def test_model() -> str:
    """Get test model from environment or use default."""
    return os.getenv("TEST_OPENAI_MODEL", "gpt-4o-mini")


@pytest.fixture
def test_max_tokens() -> int:
    """Get test max tokens from environment or use default."""
    return int(os.getenv("DEFAULT_LLM_MAX_TOKENS", "100"))


@pytest.fixture
def llm_client(test_model: str, test_max_tokens: int) -> LLMClient:
    """
    Create an LLMClient for testing.
    
    Note: Cleanup of LiteLLM async clients is handled by the session-scoped
    event_loop fixture, so no per-test cleanup is needed here.
    """
    return LLMClient(
        config=LLMConfig(
            model=test_model,
            max_tokens=test_max_tokens,
            temperature=settings.default_llm_temperature,
        ),
        enable_langfuse=False,  # Disable Langfuse for unit tests
    )


@pytest.fixture
def llm_client_with_langfuse(test_model: str, test_max_tokens: int) -> LLMClient:
    """
    Create an LLMClient with Langfuse enabled for testing.
    
    Note: Cleanup of LiteLLM async clients is handled by the session-scoped
    event_loop fixture, so no per-test cleanup is needed here.
    """
    return LLMClient(
        config=LLMConfig(
            model=test_model,
            max_tokens=test_max_tokens,
            temperature=settings.default_llm_temperature,
        ),
        enable_langfuse=True,
    )
