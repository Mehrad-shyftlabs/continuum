"""
Integration tests for Progressive Context Management module.

Tests proactive context compression, summarization, and integration
with LLMClient and AgentRunner.

Converted from tests/test_context_management.py manual test script.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from orchestrator.llm import (
    ChatMessage,
    LLMClient,
    LLMConfig,
    CompressionStrategy,
    CompressionResult,
    ContextManagementConfig,
    ProgressiveContextManager,
    get_progressive_context_manager,
)
from orchestrator.llm.context_window import get_context_window_manager
from orchestrator.agent import BaseAgent, AgentRunner
from orchestrator.observability.metrics import get_metrics_collector
import logging

logger = logging.getLogger(__name__)


pytestmark = [pytest.mark.integration]


def create_large_messages(count: int, tokens_per_message: int = 100) -> list[dict]:
    """Create a list of messages that approximate a certain token count."""
    chars_per_message = tokens_per_message * 4

    messages = []
    for i in range(count):
        content = f"User message {i}: " + "x" * (
            chars_per_message - len(f"User message {i}: ")
        )
        messages.append(
            {
                "role": "user" if i % 2 == 0 else "assistant",
                "content": content,
            }
        )
    return messages


def create_messages_exceeding_threshold(
    model: str = "gpt-4o", threshold: float = 0.8
) -> list[dict]:
    """Create messages that exceed the compression threshold for a model."""
    window_manager = get_context_window_manager()
    limits = window_manager.get_model_limits(model)
    threshold_tokens = int(limits.effective_input_limit * threshold)
    target_tokens = int(threshold_tokens * 1.3)

    messages = []
    current_tokens = 0

    system_msg = {"role": "system", "content": "You are a helpful assistant." * 10}
    messages.append(system_msg)
    current_tokens = window_manager.count_tokens(messages, model)

    i = 0
    while current_tokens < target_tokens:
        content = (
            f"User message {i}: "
            + "This is a test message with enough content to approximate tokens. " * 100
        )
        messages.append(
            {
                "role": "user" if i % 2 == 0 else "assistant",
                "content": content,
            }
        )
        current_tokens = window_manager.count_tokens(messages, model)
        i += 1
        if i > 1000:
            break

    return messages


class TestBasicCompression:
    async def test_truncate_oldest_strategy(self):
        logger.info("BasicCompression: truncate oldest strategy")
        manager = ProgressiveContextManager(
            config=ContextManagementConfig(
                enabled=True,
                compression_threshold=0.8,
                compression_strategy=CompressionStrategy.TRUNCATE_OLDEST,
            )
        )

        model = "gpt-3.5-turbo"
        messages = create_messages_exceeding_threshold(model, threshold=0.8)

        window_manager = get_context_window_manager()
        limits = window_manager.get_model_limits(model)
        threshold_tokens = int(limits.effective_input_limit * 0.8)
        original_tokens = window_manager.count_tokens(messages, model)

        while original_tokens <= threshold_tokens:
            content = "Additional message: " + "x" * 500
            messages.append({"role": "user", "content": content})
            original_tokens = window_manager.count_tokens(messages, model)

        compressed, result = await manager.compress_if_needed(
            messages=messages,
            model=model,
        )

        compressed_tokens = window_manager.count_tokens(compressed, model)
        assert result.was_compressed
        assert compressed_tokens < original_tokens
        assert len(compressed) <= len(messages)

    async def test_no_compression_below_threshold(self):
        logger.info("BasicCompression: no compression below threshold")
        manager = ProgressiveContextManager(
            config=ContextManagementConfig(
                enabled=True,
                compression_threshold=0.8,
            )
        )

        model = "gpt-4o"
        window_manager = get_context_window_manager()
        limits = window_manager.get_model_limits(model)

        small_messages = create_large_messages(5, tokens_per_message=50)
        small_tokens = window_manager.count_tokens(small_messages, model)
        threshold_tokens = int(limits.effective_input_limit * 0.8)

        if small_tokens < threshold_tokens:
            compressed, result = await manager.compress_if_needed(
                messages=small_messages,
                model=model,
            )
            assert not result.was_compressed
            assert len(compressed) == len(small_messages)


class TestCompressionStrategies:
    async def test_summarize_strategy_with_mock(self):
        logger.info("CompressionStrategies: summarize strategy with mock")
        mock_response = MagicMock()
        mock_response.content = "This is a summary of the previous conversation."

        mock_llm_client = AsyncMock()
        mock_llm_client.chat = AsyncMock(return_value=mock_response)

        manager = ProgressiveContextManager(
            config=ContextManagementConfig(
                enabled=True,
                compression_strategy=CompressionStrategy.SUMMARIZE_OLD,
                keep_recent_messages=5,
            )
        )

        with patch.object(manager, "_get_llm_client", return_value=mock_llm_client):
            model = "gpt-3.5-turbo"
            messages = create_messages_exceeding_threshold(model, threshold=0.8)

            window_manager = get_context_window_manager()
            limits = window_manager.get_model_limits(model)
            threshold_tokens = int(limits.effective_input_limit * 0.8)
            original_tokens = window_manager.count_tokens(messages, model)

            while original_tokens <= threshold_tokens or len(messages) <= 5:
                content = "Additional message for summarization: " + "x" * 500
                messages.append({"role": "user", "content": content})
                original_tokens = window_manager.count_tokens(messages, model)

            compressed, result = await manager.compress_if_needed(
                messages=messages,
                model=model,
            )

            assert result.was_compressed
            assert result.strategy_used == "summarize_old"
            assert result.summarization_used
            mock_llm_client.chat.assert_called()

    async def test_smart_strategy(self):
        logger.info("CompressionStrategies: smart strategy")
        manager = ProgressiveContextManager(
            config=ContextManagementConfig(
                enabled=True,
                compression_strategy=CompressionStrategy.SMART,
                keep_recent_messages=5,
            )
        )

        model = "gpt-3.5-turbo"
        messages = create_messages_exceeding_threshold(model, threshold=0.8)

        window_manager = get_context_window_manager()
        limits = window_manager.get_model_limits(model)
        threshold_tokens = int(limits.effective_input_limit * 0.8)
        original_tokens = window_manager.count_tokens(messages, model)

        while original_tokens <= threshold_tokens:
            content = "Additional: " + "x" * 500
            messages.append({"role": "user", "content": content})
            original_tokens = window_manager.count_tokens(messages, model)

        compressed, result = await manager.compress_if_needed(
            messages=messages,
            model=model,
        )

        assert result.was_compressed
        assert result.strategy_used in [
            "smart_summarize",
            "smart_summarize_truncate",
            "truncate_oldest",
        ]
        assert len(compressed) <= len(messages)


class TestMetricsTracking:
    async def test_compression_latency(self):
        logger.info("MetricsTracking: compression latency")
        metrics = get_metrics_collector()
        metrics.reset()

        manager = ProgressiveContextManager(
            config=ContextManagementConfig(
                enabled=True,
                compression_strategy=CompressionStrategy.TRUNCATE_OLDEST,
            )
        )

        model = "gpt-4o"
        messages = create_messages_exceeding_threshold(model)

        compressed, result = await manager.compress_if_needed(
            messages=messages,
            model=model,
        )

        assert result.latency_ms > 0


class TestConfiguration:
    async def test_global_config_defaults(self):
        logger.info("Configuration: global config defaults")
        manager = get_progressive_context_manager()
        assert manager.config.enabled
        assert manager.config.compression_threshold == 0.8

    async def test_custom_config(self):
        logger.info("Configuration: custom config")
        custom_config = ContextManagementConfig(
            enabled=True,
            compression_threshold=0.9,
            compression_strategy=CompressionStrategy.SUMMARIZE_OLD,
            keep_recent_messages=15,
        )
        custom_manager = ProgressiveContextManager(config=custom_config)
        assert custom_manager.config.compression_threshold == 0.9
        assert custom_manager.config.keep_recent_messages == 15
        assert custom_manager.config.compression_strategy == CompressionStrategy.SUMMARIZE_OLD

    async def test_disabled_config(self):
        logger.info("Configuration: disabled config")
        disabled_config = ContextManagementConfig(enabled=False)
        disabled_manager = ProgressiveContextManager(config=disabled_config)

        model = "gpt-4o"
        messages = create_messages_exceeding_threshold(model)

        compressed, result = await disabled_manager.compress_if_needed(
            messages=messages,
            model=model,
        )

        assert not result.was_compressed
        assert len(compressed) == len(messages)


class TestErrorHandling:
    async def test_fallback_on_llm_failure(self):
        logger.info("ErrorHandling: fallback on llm failure")
        manager = ProgressiveContextManager(
            config=ContextManagementConfig(
                enabled=True,
                compression_strategy=CompressionStrategy.SMART,
            )
        )

        model = "gpt-4o"
        messages = create_messages_exceeding_threshold(model)

        with patch.object(manager, "_get_llm_client", return_value=None):
            compressed, result = await manager.compress_if_needed(
                messages=messages,
                model=model,
            )

            assert len(compressed) <= len(messages)
            assert result.strategy_used in [
                "truncate_oldest",
                "fallback_truncate",
                "smart_summarize",
                "summarize_old",
            ]

    async def test_compression_never_fails_request(self):
        logger.info("ErrorHandling: compression never fails request")
        manager = ProgressiveContextManager(
            config=ContextManagementConfig(
                enabled=True,
                compression_strategy=CompressionStrategy.SMART,
            )
        )

        model = "gpt-4o"
        messages = create_messages_exceeding_threshold(model)

        compressed, result = await manager.compress_if_needed(
            messages=messages,
            model=model,
        )

        assert compressed is not None
        assert result is not None


class TestLLMClientIntegration:
    async def test_chat_with_context_management(self):
        logger.info("LLMClientIntegration: chat with context management")
        client = LLMClient(
            config=LLMConfig(model="gpt-4o-mini", max_tokens=100),
            enable_langfuse=False,
        )

        model = "gpt-4o-mini"
        messages = create_messages_exceeding_threshold(model, threshold=0.7)
        chat_messages = [
            ChatMessage(role=msg["role"], content=msg["content"]) for msg in messages
        ]

        class MockMessage:
            def __init__(self):
                self.content = "Test response"
                self.tool_calls = None
                self.role = "assistant"

        mock_message = MockMessage()

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        mock_usage.total_tokens = 150

        mock_response = Mock(spec_set=["id", "choices", "usage", "model"])
        mock_response.id = "test-response-id"
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        mock_response.model = model

        async_mock_response = AsyncMock(return_value=mock_response)

        with patch(
            "orchestrator.llm.client.litellm.acompletion",
            side_effect=async_mock_response,
        ):
            response = await client.chat(
                messages=chat_messages,
                auto_session=False,
            )
            assert async_mock_response.called
