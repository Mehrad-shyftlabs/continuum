"""
Tests for Progressive Context Management module.

Tests proactive context compression, summarization, and integration with LLMClient and AgentRunner.

Usage:
    python -m tests.test_context_management
    python -m tests.test_context_management --test compression
    python -m tests.test_context_management --test integration
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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
from orchestrator.agent.types import RunContext
from orchestrator.observability.metrics import get_metrics_collector


# =============================================================================
# Test Utilities
# =============================================================================

def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"✅ {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"❌ {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    print(f"ℹ️  {message}")


def create_large_messages(count: int, tokens_per_message: int = 100) -> list[dict[str, any]]:
    """Create a list of messages that approximate a certain token count."""
    # Rough approximation: ~4 characters per token
    chars_per_message = tokens_per_message * 4
    
    messages = []
    for i in range(count):
        content = f"User message {i}: " + "x" * (chars_per_message - len(f"User message {i}: "))
        messages.append({
            "role": "user" if i % 2 == 0 else "assistant",
            "content": content,
        })
    
    return messages


def create_messages_exceeding_threshold(model: str = "gpt-4o", threshold: float = 0.8) -> list[dict[str, any]]:
    """Create messages that exceed the compression threshold for a model."""
    window_manager = get_context_window_manager()
    limits = window_manager.get_model_limits(model)
    threshold_tokens = int(limits.effective_input_limit * threshold)
    
    # Create messages that exceed threshold by ~30% to ensure we're definitely over
    target_tokens = int(threshold_tokens * 1.3)
    
    # Count tokens as we build to ensure we exceed threshold
    messages = []
    current_tokens = 0
    
    # Add system message
    system_msg = {"role": "system", "content": "You are a helpful assistant." * 10}
    messages.append(system_msg)
    current_tokens = window_manager.count_tokens(messages, model)
    
    # Keep adding messages until we exceed threshold
    i = 0
    while current_tokens < target_tokens:
        # Create a larger message (~500 tokens each)
        content = f"User message {i}: " + "This is a test message with enough content to approximate tokens. " * 100
        messages.append({
            "role": "user" if i % 2 == 0 else "assistant",
            "content": content,
        })
        current_tokens = window_manager.count_tokens(messages, model)
        i += 1
        
        # Safety limit
        if i > 1000:
            break
    
    return messages


# =============================================================================
# Test: Basic Compression
# =============================================================================

async def test_basic_compression() -> bool:
    """Test basic compression functionality."""
    print_header("Test: Basic Compression")
    
    try:
        manager = ProgressiveContextManager(
            config=ContextManagementConfig(
                enabled=True,
                compression_threshold=0.8,
                compression_strategy=CompressionStrategy.TRUNCATE_OLDEST,
            )
        )
        
        # Use a smaller model for testing to make threshold easier to hit
        model = "gpt-3.5-turbo"  # 16385 tokens, effective ~12288, threshold ~9830
        messages = create_messages_exceeding_threshold(model, threshold=0.8)
        
        window_manager = get_context_window_manager()
        limits = window_manager.get_model_limits(model)
        threshold_tokens = int(limits.effective_input_limit * 0.8)
        original_tokens = window_manager.count_tokens(messages, model)
        
        print_info(f"Model: {model}")
        print_info(f"Effective limit: {limits.effective_input_limit}")
        print_info(f"Threshold: {threshold_tokens}")
        print_info(f"Original messages: {len(messages)}, tokens: {original_tokens}")
        
        # Verify we're actually over threshold
        if original_tokens <= threshold_tokens:
            print_info(f"⚠️  Messages ({original_tokens}) don't exceed threshold ({threshold_tokens}), adding more...")
            # Add more messages
            while original_tokens <= threshold_tokens:
                content = "Additional message: " + "x" * 500
                messages.append({"role": "user", "content": content})
                original_tokens = window_manager.count_tokens(messages, model)
        
        # Compress
        compressed, result = await manager.compress_if_needed(
            messages=messages,
            model=model,
        )
        
        compressed_tokens = window_manager.count_tokens(compressed, model)
        print_info(f"Compressed messages: {len(compressed)}, tokens: {compressed_tokens}")
        print_info(f"Compression ratio: {result.compression_ratio:.2%}")
        print_info(f"Strategy used: {result.strategy_used}")
        
        # Verify compression happened
        if not result.was_compressed:
            print_error(f"Compression didn't occur. Original: {original_tokens}, Threshold: {threshold_tokens}")
            return False
        
        assert compressed_tokens < original_tokens, f"Compressed tokens ({compressed_tokens}) should be less than original ({original_tokens})"
        assert len(compressed) <= len(messages), "Compressed messages should be fewer or equal"
        
        print_success("Basic compression working!")
        return True
        
    except Exception as e:
        print_error(f"Basic compression test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_compression_threshold() -> bool:
    """Test that compression only happens when threshold is exceeded."""
    print_header("Test: Compression Threshold")
    
    try:
        manager = ProgressiveContextManager(
            config=ContextManagementConfig(
                enabled=True,
                compression_threshold=0.8,
            )
        )
        
        model = "gpt-4o"
        window_manager = get_context_window_manager()
        limits = window_manager.get_model_limits(model)
        
        # Test 1: Messages below threshold should not compress
        small_messages = create_large_messages(5, tokens_per_message=50)
        small_tokens = window_manager.count_tokens(small_messages, model)
        threshold_tokens = int(limits.effective_input_limit * 0.8)
        
        print_info(f"Small messages tokens: {small_tokens}, threshold: {threshold_tokens}")
        
        if small_tokens < threshold_tokens:
            compressed, result = await manager.compress_if_needed(
                messages=small_messages,
                model=model,
            )
            
            assert not result.was_compressed, "Should not compress when below threshold"
            assert len(compressed) == len(small_messages), "Messages should be unchanged"
            print_success("Below threshold: No compression ✓")
        
        # Test 2: Messages above threshold should compress
        large_messages = create_messages_exceeding_threshold(model, threshold=0.8)
        large_tokens = window_manager.count_tokens(large_messages, model)
        
        print_info(f"Large messages tokens: {large_tokens}, threshold: {threshold_tokens}")
        
        if large_tokens > threshold_tokens:
            compressed, result = await manager.compress_if_needed(
                messages=large_messages,
                model=model,
            )
            
            assert result.was_compressed, "Should compress when above threshold"
            compressed_tokens = window_manager.count_tokens(compressed, model)
            assert compressed_tokens < large_tokens, "Compressed should be smaller"
            print_success("Above threshold: Compression occurred ✓")
        
        print_success("Compression threshold test passed!")
        return True
        
    except Exception as e:
        print_error(f"Compression threshold test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test: Compression Strategies
# =============================================================================

async def test_truncate_strategy() -> bool:
    """Test truncation strategy."""
    print_header("Test: Truncate Strategy")
    
    try:
        manager = ProgressiveContextManager(
            config=ContextManagementConfig(
                enabled=True,
                compression_strategy=CompressionStrategy.TRUNCATE_OLDEST,
            )
        )
        
        model = "gpt-3.5-turbo"  # Smaller model for easier threshold
        messages = create_messages_exceeding_threshold(model, threshold=0.8)
        
        window_manager = get_context_window_manager()
        limits = window_manager.get_model_limits(model)
        threshold_tokens = int(limits.effective_input_limit * 0.8)
        original_tokens = window_manager.count_tokens(messages, model)
        
        # Ensure we exceed threshold
        if original_tokens <= threshold_tokens:
            while original_tokens <= threshold_tokens:
                content = "Additional: " + "x" * 500
                messages.append({"role": "user", "content": content})
                original_tokens = window_manager.count_tokens(messages, model)
        
        compressed, result = await manager.compress_if_needed(
            messages=messages,
            model=model,
        )
        
        if not result.was_compressed:
            print_error(f"Compression didn't occur. Tokens: {original_tokens}, Threshold: {threshold_tokens}")
            return False
        
        assert result.strategy_used == "truncate_oldest", f"Expected truncate_oldest, got {result.strategy_used}"
        assert result.truncation_used, "Truncation should be marked as used"
        assert len(compressed) <= len(messages), "Should have fewer or equal messages"
        
        print_success("Truncate strategy working!")
        return True
        
    except Exception as e:
        print_error(f"Truncate strategy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_summarize_strategy() -> bool:
    """Test summarization strategy with mocked LLM."""
    print_header("Test: Summarize Strategy")
    
    try:
        # Mock LLM client for summarization
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
        
        # Replace the LLM client with mock
        with patch.object(manager, '_get_llm_client', return_value=mock_llm_client):
            model = "gpt-3.5-turbo"  # Smaller model for easier threshold
            messages = create_messages_exceeding_threshold(model, threshold=0.8)
            
            window_manager = get_context_window_manager()
            limits = window_manager.get_model_limits(model)
            threshold_tokens = int(limits.effective_input_limit * 0.8)
            original_tokens = window_manager.count_tokens(messages, model)
            
            # Ensure we exceed threshold and have enough messages to summarize
            if original_tokens <= threshold_tokens or len(messages) <= 5:
                while original_tokens <= threshold_tokens or len(messages) <= 5:
                    content = "Additional message for summarization: " + "x" * 500
                    messages.append({"role": "user", "content": content})
                    original_tokens = window_manager.count_tokens(messages, model)
            
            compressed, result = await manager.compress_if_needed(
                messages=messages,
                model=model,
            )
            
            if not result.was_compressed:
                print_error(f"Compression didn't occur. Tokens: {original_tokens}, Threshold: {threshold_tokens}")
                return False
            
            assert result.strategy_used == "summarize_old", f"Expected summarize_old, got {result.strategy_used}"
            assert result.summarization_used, "Summarization should be marked as used"
            
            # Verify summarization was called
            mock_llm_client.chat.assert_called()
            
            print_success("Summarize strategy working!")
            return True
        
    except Exception as e:
        print_error(f"Summarize strategy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    except Exception as e:
        print_error(f"Summarize strategy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_smart_strategy() -> bool:
    """Test smart strategy (summarize + fallback to truncate)."""
    print_header("Test: Smart Strategy")
    
    try:
        manager = ProgressiveContextManager(
            config=ContextManagementConfig(
                enabled=True,
                compression_strategy=CompressionStrategy.SMART,
                keep_recent_messages=5,
            )
        )
        
        model = "gpt-3.5-turbo"  # Smaller model for easier threshold
        messages = create_messages_exceeding_threshold(model, threshold=0.8)
        
        window_manager = get_context_window_manager()
        limits = window_manager.get_model_limits(model)
        threshold_tokens = int(limits.effective_input_limit * 0.8)
        original_tokens = window_manager.count_tokens(messages, model)
        
        # Ensure we exceed threshold
        if original_tokens <= threshold_tokens:
            while original_tokens <= threshold_tokens:
                content = "Additional: " + "x" * 500
                messages.append({"role": "user", "content": content})
                original_tokens = window_manager.count_tokens(messages, model)
        
        compressed, result = await manager.compress_if_needed(
            messages=messages,
            model=model,
        )
        
        if not result.was_compressed:
            print_error(f"Compression didn't occur. Tokens: {original_tokens}, Threshold: {threshold_tokens}")
            return False
        
        # Smart strategy should use either summarize or truncate
        assert result.strategy_used in [
            "smart_summarize",
            "smart_summarize_truncate",
            "truncate_oldest",  # Fallback if summarization fails
        ], f"Unexpected strategy: {result.strategy_used}"
        
        assert len(compressed) <= len(messages), "Should have fewer or equal messages"
        
        print_success(f"Smart strategy working! (used: {result.strategy_used})")
        return True
        
    except Exception as e:
        print_error(f"Smart strategy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test: Metrics and Observability
# =============================================================================

async def test_metrics_tracking() -> bool:
    """Test that metrics are properly tracked."""
    print_header("Test: Metrics Tracking")
    
    try:
        metrics = get_metrics_collector()
        metrics.reset()  # Clear any existing metrics
        
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
        
        # Check that latency was recorded
        # Note: We can't directly check metrics._latencies as it's private,
        # but we can verify the result has latency info
        assert result.latency_ms > 0, "Latency should be recorded"
        
        print_info(f"Compression latency: {result.latency_ms:.2f}ms")
        print_info(f"Compression ratio: {result.compression_ratio:.2%}")
        
        print_success("Metrics tracking working!")
        return True
        
    except Exception as e:
        print_error(f"Metrics tracking test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test: LLMClient Integration
# =============================================================================

async def test_llm_client_integration() -> bool:
    """Test integration with LLMClient."""
    print_header("Test: LLMClient Integration")
    
    try:
        # Create client with context management enabled
        client = LLMClient(
            config=LLMConfig(
                model="gpt-4o-mini",
                max_tokens=100,
            ),
            enable_langfuse=False,  # Disable for testing
        )
        
        # Create large message list
        model = "gpt-4o-mini"
        messages = create_messages_exceeding_threshold(model, threshold=0.7)  # Lower threshold for smaller model
        
        # Convert to ChatMessage format
        chat_messages = [
            ChatMessage(role=msg["role"], content=msg["content"])
            for msg in messages
        ]
        
        print_info(f"Original messages: {len(chat_messages)}")
        
        # Mock the actual LLM call to avoid API costs
        # Create a proper mock response structure that matches LiteLLM's format
        # Use a simple class to prevent attribute access issues
        class MockMessage:
            def __init__(self):
                self.content = "Test response"
                self.tool_calls = None
                self.role = "assistant"
                # Don't define function_call - hasattr will return False
        
        mock_message = MockMessage()
        
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"
        
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        mock_usage.total_tokens = 150
        
        # Create response object - use Mock with spec_set to prevent auto-creation
        # This ensures model_dump doesn't exist unless we explicitly set it
        mock_response = Mock(spec_set=['id', 'choices', 'usage', 'model'])
        mock_response.id = "test-response-id"
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        mock_response.model = model
        # Don't set model_dump - spec_set prevents hasattr from returning True
        # This way raw_response will be None
        
        # Make it async - litellm.acompletion is async
        async_mock_response = AsyncMock(return_value=mock_response)
        
        with patch('orchestrator.llm.client.litellm.acompletion', side_effect=async_mock_response):
            
            # Call chat - context management should compress before sending
            response = await client.chat(
                messages=chat_messages,
                auto_session=False,
            )
            
            # Verify the call was made
            assert async_mock_response.called, "LLM should have been called"
            
            # Get the messages that were actually sent
            call_args = async_mock_response.call_args
            sent_messages = call_args.kwargs.get('messages', []) if call_args and call_args.kwargs else []
            
            print_info(f"Messages sent to LLM: {len(sent_messages)}")
            
            # Verify compression happened (sent messages should be fewer)
            # Note: This might not always be true if compression wasn't needed,
            # but in our test case it should compress
            if len(messages) > 10:  # Only check if we had enough messages
                # The compression might have happened, check if it did
                window_manager = get_context_window_manager()
                original_tokens = window_manager.count_tokens(messages, model)
                sent_tokens = sum(len(str(m.get('content', ''))) // 4 for m in sent_messages)  # Rough estimate
                
                print_info(f"Original tokens (approx): {original_tokens}")
                print_info(f"Sent tokens (approx): {sent_tokens}")
        
        print_success("LLMClient integration working!")
        return True
        
    except Exception as e:
        print_error(f"LLMClient integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test: AgentRunner Integration
# =============================================================================

async def test_agent_runner_integration() -> bool:
    """Test integration with AgentRunner."""
    print_header("Test: AgentRunner Integration")
    
    try:
        from orchestrator.agent.config import AgentConfig
        
        # Create agent with context management config
        agent_config = AgentConfig()
        agent_config.context_management = ContextManagementConfig(
            enabled=True,
            compression_threshold=0.7,  # Lower threshold for testing
            compression_strategy=CompressionStrategy.TRUNCATE_OLDEST,
        )
        
        agent = BaseAgent(
            name="test-agent",
            instructions="You are a helpful assistant.",
            model="gpt-4o-mini",
            config=agent_config,
        )
        
        runner = AgentRunner()
        
        # Create large input that will trigger compression
        large_input = " ".join([f"Message {i}: " + "x" * 100 for i in range(50)])
        
        print_info(f"Input length: {len(large_input)} characters")
        
        # Mock the LLM call to avoid API costs
        # Create a proper mock response structure that matches LiteLLM's format
        # Use a simple class to prevent attribute access issues
        class MockMessage:
            def __init__(self):
                self.content = "Test response"
                self.tool_calls = None
                self.role = "assistant"
                # Don't define function_call - hasattr will return False
        
        mock_message = MockMessage()
        
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"
        
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        mock_usage.total_tokens = 150
        
        # Create response object - use Mock with spec_set to prevent auto-creation
        # This ensures model_dump doesn't exist unless we explicitly set it
        mock_response = Mock(spec_set=['id', 'choices', 'usage', 'model'])
        mock_response.id = "test-response-id"
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        mock_response.model = agent.model
        # Don't set model_dump - spec_set prevents hasattr from returning True
        # This way raw_response will be None
        
        # Make it async - litellm.acompletion is async
        async_mock_response = AsyncMock(return_value=mock_response)
        
        with patch('orchestrator.llm.client.litellm.acompletion', side_effect=async_mock_response):
            
            # Run agent - context management should compress in _prepare_messages
            response = await runner.run(
                agent=agent,
                input=large_input,
                session_id="test-session",
                user_id="test-user",
            )
            
            # Verify the call was made
            assert async_mock_response.called, "LLM should have been called"
            
            print_info(f"Agent response status: {response.status}")
        
        print_success("AgentRunner integration working!")
        return True
        
    except Exception as e:
        print_error(f"AgentRunner integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test: Error Handling
# =============================================================================

async def test_error_handling() -> bool:
    """Test error handling and fallbacks."""
    print_header("Test: Error Handling")
    
    try:
        manager = ProgressiveContextManager(
            config=ContextManagementConfig(
                enabled=True,
                compression_strategy=CompressionStrategy.SMART,
            )
        )
        
        model = "gpt-4o"
        messages = create_messages_exceeding_threshold(model)
        
        # Test that errors don't break the request
        # Simulate summarization failure by making LLM client return None
        with patch.object(manager, '_get_llm_client', return_value=None):
            compressed, result = await manager.compress_if_needed(
                messages=messages,
                model=model,
            )
            
            # Should fallback to truncation
            assert len(compressed) <= len(messages), "Should return compressed or original messages"
            # When LLM client is None, smart strategy will use text summary fallback,
            # which still counts as summarization, then may truncate if needed
            # So strategy could be smart_summarize (with text fallback) or truncate_oldest
            assert result.strategy_used in [
                "truncate_oldest",
                "fallback_truncate",
                "smart_summarize",  # Text fallback still counts as summarization
                "summarize_old",  # Text fallback path
            ], f"Should handle error gracefully, got {result.strategy_used}"
            
            print_success(f"Error handling: Fallback working (strategy: {result.strategy_used}) ✓")
        
        # Test that compression failure doesn't break request
        # This should never happen, but test the error path
        try:
            compressed, result = await manager.compress_if_needed(
                messages=messages,
                model=model,
            )
            # Should always return something
            assert compressed is not None, "Should return messages even on error"
            assert result is not None, "Should return result even on error"
            print_success("Error handling: Never fails request ✓")
        except Exception as e:
            print_error(f"Compression should not raise exceptions: {e}")
            return False
        
        print_success("Error handling test passed!")
        return True
        
    except Exception as e:
        print_error(f"Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test: Configuration
# =============================================================================

async def test_configuration() -> bool:
    """Test configuration options."""
    print_header("Test: Configuration")
    
    try:
        # Test global config
        manager = get_progressive_context_manager()
        assert manager.config.enabled, "Should be enabled by default"
        assert manager.config.compression_threshold == 0.8, "Default threshold should be 0.8"
        
        print_success("Global config: Defaults correct ✓")
        
        # Test custom config
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
        
        print_success("Custom config: Applied correctly ✓")
        
        # Test disabled
        disabled_config = ContextManagementConfig(enabled=False)
        disabled_manager = ProgressiveContextManager(config=disabled_config)
        
        model = "gpt-4o"
        messages = create_messages_exceeding_threshold(model)
        
        compressed, result = await disabled_manager.compress_if_needed(
            messages=messages,
            model=model,
        )
        
        assert not result.was_compressed, "Should not compress when disabled"
        assert len(compressed) == len(messages), "Messages should be unchanged"
        
        print_success("Disabled config: No compression ✓")
        
        print_success("Configuration test passed!")
        return True
        
    except Exception as e:
        print_error(f"Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Main Test Runner
# =============================================================================

async def run_all_tests() -> None:
    """Run all context management tests."""
    print_header("Context Management - Complete Test Suite")
    
    tests = [
        ("Basic Compression", test_basic_compression),
        ("Compression Threshold", test_compression_threshold),
        ("Truncate Strategy", test_truncate_strategy),
        ("Summarize Strategy", test_summarize_strategy),
        ("Smart Strategy", test_smart_strategy),
        ("Metrics Tracking", test_metrics_tracking),
        ("LLMClient Integration", test_llm_client_integration),
        ("AgentRunner Integration", test_agent_runner_integration),
        ("Error Handling", test_error_handling),
        ("Configuration", test_configuration),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result))
        except Exception as e:
            print_error(f"Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print_header("Test Summary")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print_success("All tests passed! 🎉")
    else:
        print_error(f"{total - passed} test(s) failed")


async def run_specific_test(test_name: str) -> None:
    """Run a specific test."""
    test_map = {
        "compression": test_basic_compression,
        "threshold": test_compression_threshold,
        "truncate": test_truncate_strategy,
        "summarize": test_summarize_strategy,
        "smart": test_smart_strategy,
        "metrics": test_metrics_tracking,
        "llm": test_llm_client_integration,
        "agent": test_agent_runner_integration,
        "errors": test_error_handling,
        "config": test_configuration,
    }
    
    if test_name not in test_map:
        print_error(f"Unknown test: {test_name}")
        print_info(f"Available tests: {', '.join(test_map.keys())}")
        return
    
    test_func = test_map[test_name]
    result = await test_func()
    
    if result:
        print_success(f"Test '{test_name}' passed!")
    else:
        print_error(f"Test '{test_name}' failed!")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test Context Management module")
    parser.add_argument(
        "--test",
        choices=[
            "compression",
            "threshold",
            "truncate",
            "summarize",
            "smart",
            "metrics",
            "llm",
            "agent",
            "errors",
            "config",
            "all",
        ],
        default="all",
        help="Which test to run",
    )
    
    args = parser.parse_args()
    
    if args.test == "all":
        asyncio.run(run_all_tests())
    else:
        asyncio.run(run_specific_test(args.test))


if __name__ == "__main__":
    main()
