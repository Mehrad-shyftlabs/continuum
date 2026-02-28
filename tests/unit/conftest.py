"""
Unit test fixtures - all external dependencies are mocked.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_litellm_response():
    """Create a mock LiteLLM response object."""

    class MockMessage:
        def __init__(self):
            self.content = "Test response"
            self.tool_calls = None
            self.role = "assistant"

    mock_choice = MagicMock()
    mock_choice.message = MockMessage()
    mock_choice.finish_reason = "stop"

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 20
    mock_usage.total_tokens = 30

    response = MagicMock(spec_set=["id", "choices", "usage", "model"])
    response.id = "test-response-id"
    response.choices = [mock_choice]
    response.usage = mock_usage
    response.model = "test-model"
    return response
