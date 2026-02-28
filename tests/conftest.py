"""
Root conftest.py - Shared fixtures for all tests.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client that returns predictable responses."""
    from orchestrator.llm.types import LLMResponse, Usage

    client = AsyncMock()
    client.chat = AsyncMock(
        return_value=LLMResponse(
            id="test-id",
            model="test-model",
            content="Test response",
            role="assistant",
            usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            finish_reason="stop",
        )
    )
    client.chat_stream = AsyncMock()
    client.count_tokens = MagicMock(return_value=100)
    client.get_max_tokens = MagicMock(return_value=4096)
    client.default_config = MagicMock()
    client.default_config.model = "test-model"
    return client


@pytest.fixture
def mock_memory_client():
    """Create a mock memory client."""
    client = AsyncMock()
    client.is_enabled = False
    client.search = AsyncMock(return_value=MagicMock(results=[]))
    client.add = AsyncMock()
    return client


@pytest.fixture
def mock_session_client():
    """Create a mock session client."""
    client = AsyncMock()
    client.is_enabled = False
    client.get_conversation_history = AsyncMock(return_value=[])
    client.add_message = AsyncMock()
    return client


@pytest.fixture
def mock_container(mock_llm_client, mock_memory_client, mock_session_client):
    """Create a mock container with all clients."""
    from orchestrator.core.container import Container, ContainerConfig

    config = ContainerConfig(auto_initialize=False)
    container = Container(config=config)
    container.set_llm_client(mock_llm_client)
    container.set_memory_client(mock_memory_client)
    container.set_session_client(mock_session_client)
    return container


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global state before each test."""
    yield
    from orchestrator.core.container import reset_container

    reset_container()
