"""
Manual testing script for the Memory module.

Run this script to verify memory functionality is working correctly.

Usage:
    1. Start Qdrant: docker-compose up -d qdrant
    2. Copy .env.template to .env and add API keys
    3. Run: python -m tests.test_memory

Or run individual tests:
    python -m tests.test_memory --test basic
    python -m tests.test_memory --test search
    python -m tests.test_memory --test errors
    python -m tests.test_memory --test isolation
    python -m tests.test_memory --test langfuse
    
Embedder-specific tests (requires API keys):
    python -m tests.test_memory --test embedder_config      # Test config generation (no API needed)
    python -m tests.test_memory --test embedder_huggingface # Requires HUGGINGFACE_API_KEY
    python -m tests.test_memory --test embedder_cohere      # Requires COHERE_API_KEY
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

from orchestrator.config import settings
from orchestrator.memory import (
    MemoryClient,
    MemoryConfig,
    MemoryFilter,
    MemoryMetadata,
    get_global_memory_client,
    initialize_global_memory,
)
from orchestrator.memory.exceptions import (
    MemoryAddError,
    MemoryError,
    MemoryIdentifierError,
    MemoryNotEnabledError,
    MemorySearchError,
)
from orchestrator.observability import get_global_langfuse_client

# =============================================================================
# Test Configuration
# =============================================================================

TEST_USER_ID = "test-user-123"
TEST_AGENT_ID = "test-agent-456"
TEST_RUN_ID = "test-run-789"
TEST_SESSION_ID = "test-session-abc"

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


def print_info(message: str) -> None:
    """Print an info message."""
    print(f"ℹ️  {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"⚠️  {message}")


async def cleanup_test_data(memory: MemoryClient, user_id: str) -> None:
    """Clean up test data."""
    try:
        await memory.delete_all(user_id=user_id)
        print_info(f"Cleaned up test data for user: {user_id}")
    except Exception as e:
        print_warning(f"Cleanup failed (may not exist): {e}")


# =============================================================================
# Test Functions
# =============================================================================


async def test_basic_operations() -> bool:
    """Test basic memory operations: add, search, get, delete."""
    print_header("Basic Memory Operations")

    try:
        memory = MemoryClient()

        if not memory.is_enabled:
            print_error("Memory is not enabled. Set MEMORY_ENABLED=true")
            return False

        print_success("Memory client initialized")

        # Clean up any existing test data
        await cleanup_test_data(memory, TEST_USER_ID)

        # Test 1: Add memory
        print_info("Test 1: Adding memory...")
        result = await memory.add(
            "User loves Python programming and prefers dark mode",
            user_id=TEST_USER_ID,
            metadata={"category": "preferences", "test": True},
        )
        print_success(f"Memory added: {result.message}")
        print_info(f"Extracted {len(result.results)} facts")

        # Test 2: Search memory
        print_info("Test 2: Searching memories...")
        search_results = await memory.search(
            "What does the user like?",
            user_id=TEST_USER_ID,
            limit=5,
        )
        print_success(f"Search found {len(search_results.results)} results")
        for i, entry in enumerate(search_results.results, 1):
            print(f"   {i}. {entry.memory} (score: {entry.score:.3f})")

        # Test 3: Get all memories
        print_info("Test 3: Getting all memories...")
        all_memories = await memory.get_all(user_id=TEST_USER_ID)
        print_success(f"Retrieved {len(all_memories)} total memories")
        for entry in all_memories:
            print(f"   - {entry.memory}")

        # Test 4: Get specific memory
        if all_memories:
            print_info("Test 4: Getting specific memory...")
            memory_id = all_memories[0].id
            specific = await memory.get(memory_id, user_id=TEST_USER_ID)
            if specific:
                print_success(f"Retrieved memory: {specific.memory}")
            else:
                print_error("Memory not found")

        # Test 5: Update memory
        if all_memories:
            print_info("Test 5: Updating memory...")
            memory_id = all_memories[0].id
            updated = await memory.update(
                memory_id,
                "User loves Python programming, prefers dark mode, and uses VS Code",
                user_id=TEST_USER_ID,
            )
            print_success(f"Updated memory: {updated.memory}")

        # Test 6: Delete specific memory
        if all_memories:
            print_info("Test 6: Deleting specific memory...")
            memory_id = all_memories[0].id
            deleted = await memory.delete(memory_id, user_id=TEST_USER_ID)
            if deleted:
                print_success("Memory deleted successfully")

        # Cleanup
        await cleanup_test_data(memory, TEST_USER_ID)
        print_success("All basic operations completed successfully")
        return True

    except Exception as e:
        print_error(f"Basic operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_search_functionality() -> bool:
    """Test advanced search functionality."""
    print_header("Search Functionality Tests")

    try:
        memory = MemoryClient()

        if not memory.is_enabled:
            print_error("Memory is not enabled")
            return False

        # Clean up
        await cleanup_test_data(memory, TEST_USER_ID)

        # Add multiple memories
        print_info("Adding test memories...")
        test_memories = [
            "User works at Google as a software engineer",
            "User's favorite programming language is Python",
            "User prefers dark mode in all applications",
            "User is allergic to peanuts",
            "User's birthday is January 15th",
        ]

        for msg in test_memories:
            await memory.add(msg, user_id=TEST_USER_ID)

        print_success(f"Added {len(test_memories)} test memories")

        # Test 1: Basic search
        print_info("Test 1: Basic semantic search...")
        results = await memory.search(
            "What does the user do for work?",
            user_id=TEST_USER_ID,
            limit=3,
        )
        print_success(f"Found {len(results.results)} relevant results")
        for entry in results.results:
            print(f"   - {entry.memory} (score: {entry.score:.3f})")

        # Test 2: Search with filters
        print_info("Test 2: Search with metadata filters...")
        filter_obj = MemoryFilter(
            user_id=TEST_USER_ID,
            category="preferences",
        )
        results = await memory.search(
            "user preferences",
            user_id=TEST_USER_ID,
            filters=filter_obj,
            limit=5,
        )
        print_success(f"Filtered search found {len(results.results)} results")

        # Test 3: Different query types
        print_info("Test 3: Testing different query types...")
        queries = [
            "What are the user's allergies?",
            "When is the user's birthday?",
            "What programming languages does the user know?",
        ]

        for query in queries:
            results = await memory.search(query, user_id=TEST_USER_ID, limit=2)
            print(f"   Query: '{query}'")
            print(f"   Results: {len(results.results)}")
            if results.results:
                print(f"   Top result: {results.results[0].memory}")

        # Test 4: Get memory strings
        print_info("Test 4: Getting memory strings...")
        results = await memory.search("user information", user_id=TEST_USER_ID)
        memory_strings = results.get_memory_strings()
        print_success(f"Retrieved {len(memory_strings)} memory strings")
        for s in memory_strings[:3]:
            print(f"   - {s}")

        # Test 5: Top K results
        print_info("Test 5: Getting top K results...")
        results = await memory.search("user details", user_id=TEST_USER_ID, limit=10)
        top_3 = results.get_top_k(3)
        print_success(f"Top 3 results: {len(top_3)}")
        for entry in top_3:
            print(f"   - {entry.memory} (score: {entry.score:.3f})")

        # Cleanup
        await cleanup_test_data(memory, TEST_USER_ID)
        print_success("All search tests completed successfully")
        return True

    except Exception as e:
        print_error(f"Search functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_isolation_levels() -> bool:
    """Test different memory isolation levels."""
    print_header("Memory Isolation Level Tests")

    try:
        # Test with different isolation levels
        isolation_levels = ["shared", "user", "agent", "run"]

        for isolation in isolation_levels:
            print_info(f"\nTesting isolation level: {isolation}")

            config = MemoryConfig(memory_isolation=isolation)
            memory = MemoryClient(config=config)

            if not memory.is_enabled:
                print_warning(f"Memory not enabled for {isolation}")
                continue

            try:
                # Clean up
                await cleanup_test_data(memory, TEST_USER_ID)

                # Add memory with appropriate identifiers
                identifiers: dict[str, str] = {}
                if isolation == "user":
                    identifiers["user_id"] = TEST_USER_ID
                elif isolation == "agent":
                    identifiers["agent_id"] = TEST_AGENT_ID
                elif isolation == "run":
                    identifiers["run_id"] = TEST_RUN_ID
                elif isolation == "shared":
                    identifiers["agent_id"] = "shared"

                await memory.add(
                    f"Test memory for {isolation} isolation",
                    **identifiers,
                )
                print_success(f"Added memory with {isolation} isolation")

                # Search
                results = await memory.search(
                    "test memory",
                    **identifiers,
                    limit=5,
                )
                print_success(f"Search found {len(results.results)} results")

                # Cleanup
                await cleanup_test_data(memory, TEST_USER_ID)

            except MemoryIdentifierError as e:
                print_warning(f"Identifier error (expected for {isolation}): {e}")
            except Exception as e:
                print_error(f"Test failed for {isolation}: {e}")

        print_success("Isolation level tests completed")
        return True

    except Exception as e:
        print_error(f"Isolation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_error_cases() -> bool:
    """Test error handling and failure cases."""
    print_header("Error Handling Tests")

    try:
        # Test 1: Memory not enabled
        print_info("Test 1: Memory not enabled error...")
        config = MemoryConfig(enabled=False)
        memory = MemoryClient(config=config)

        try:
            await memory.add("test", user_id=TEST_USER_ID)
            print_error("Should have raised MemoryNotEnabledError")
            return False
        except MemoryNotEnabledError:
            print_success("Correctly raised MemoryNotEnabledError")

        # Test 2: Missing required identifier
        print_info("Test 2: Missing required identifier...")
        # Only test if memory can be enabled (Qdrant available)
        if settings.memory_enabled:
            config = MemoryConfig(memory_isolation="user", enabled=True)
            memory = MemoryClient(config=config)
            
            if memory.is_enabled:
                try:
                    await memory.add("test")  # Missing user_id
                    print_error("Should have raised MemoryIdentifierError")
                    return False
                except MemoryIdentifierError:
                    print_success("Correctly raised MemoryIdentifierError")
                except MemoryNotEnabledError:
                    print_warning("Memory not enabled (Qdrant may not be running)")
            else:
                print_warning("Memory not enabled, skipping identifier test")
        else:
            print_warning("Memory disabled in config, skipping identifier test")

        # Test 3: Invalid memory ID
        print_info("Test 3: Invalid memory ID...")
        config = MemoryConfig(enabled=True)
        memory = MemoryClient(config=config)

        if memory.is_enabled:
            try:
                result = await memory.get("invalid-memory-id-12345", user_id=TEST_USER_ID)
                if result is None:
                    print_success("Correctly returned None for invalid ID")
                else:
                    print_warning("Unexpected result for invalid ID")
            except Exception as e:
                print_info(f"Exception (may be expected): {e}")

        # Test 4: Search with invalid query
        print_info("Test 4: Search with empty query...")
        if memory.is_enabled:
            try:
                results = await memory.search("", user_id=TEST_USER_ID)
                print_info(f"Empty query returned {len(results.results)} results")
            except Exception as e:
                print_info(f"Exception (may be expected): {e}")

        # Test 5: Delete non-existent memory
        print_info("Test 5: Delete non-existent memory...")
        if memory.is_enabled:
            try:
                deleted = await memory.delete("non-existent-id", user_id=TEST_USER_ID)
                print_info(f"Delete returned: {deleted}")
            except Exception as e:
                print_info(f"Exception (may be expected): {e}")

        print_success("Error handling tests completed")
        return True

    except Exception as e:
        print_error(f"Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_langfuse_integration() -> bool:
    """Test Langfuse tracing integration."""
    print_header("Langfuse Integration Tests")

    try:
        memory = MemoryClient()

        if not memory.is_enabled:
            print_error("Memory is not enabled")
            return False

        # Check Langfuse client
        langfuse = get_global_langfuse_client()
        if not langfuse.is_enabled:
            print_warning("Langfuse is not enabled. Errors will still be logged.")
        else:
            print_success("Langfuse is enabled")

        # Clean up
        await cleanup_test_data(memory, TEST_USER_ID)

        # Test 1: Add with tracing
        print_info("Test 1: Adding memory with trace context...")
        trace_id = f"test-trace-{os.urandom(4).hex()}"
        try:
            result = await memory.add(
                "User prefers Python and dark mode",
                user_id=TEST_USER_ID,
                trace_id=trace_id,
                span_id="test-span-1",
                session_id=TEST_SESSION_ID,
            )
            print_success(f"Memory added with trace_id: {trace_id}")
            print_info("Check Langfuse UI for trace: " + trace_id)
        except Exception as e:
            print_error(f"Add with tracing failed: {e}")
            print_info("This error should appear in Langfuse")

        # Test 2: Search with tracing
        print_info("Test 2: Searching with trace context...")
        trace_id = f"test-trace-{os.urandom(4).hex()}"
        try:
            results = await memory.search(
                "user preferences",
                user_id=TEST_USER_ID,
                trace_id=trace_id,
                span_id="test-span-2",
                session_id=TEST_SESSION_ID,
            )
            print_success(f"Search completed with trace_id: {trace_id}")
            print_info(f"Found {len(results.results)} results")
            print_info("Check Langfuse UI for trace: " + trace_id)
        except Exception as e:
            print_error(f"Search with tracing failed: {e}")
            print_info("This error should appear in Langfuse")

        # Test 3: Error that should be reported to Langfuse
        print_info("Test 3: Testing error reporting to Langfuse...")
        trace_id = f"test-trace-error-{os.urandom(4).hex()}"
        try:
            # This should fail and be reported
            config = MemoryConfig(memory_isolation="user")
            error_memory = MemoryClient(config=config)
            await error_memory.add("test")  # Missing user_id
        except MemoryIdentifierError as e:
            print_success("Error correctly raised and should be in Langfuse")
            print_info(f"Error trace_id: {trace_id}")
            print_info("Check Langfuse UI for error report")

        # Test 4: Multiple operations in same trace
        print_info("Test 4: Multiple operations in same trace...")
        trace_id = f"test-trace-multi-{os.urandom(4).hex()}"
        try:
            await memory.add(
                "User works at Google",
                user_id=TEST_USER_ID,
                trace_id=trace_id,
                span_id="span-1",
            )
            await memory.add(
                "User likes Python",
                user_id=TEST_USER_ID,
                trace_id=trace_id,
                span_id="span-2",
            )
            results = await memory.search(
                "user information",
                user_id=TEST_USER_ID,
                trace_id=trace_id,
                span_id="span-3",
            )
            print_success(f"Multiple operations in trace: {trace_id}")
            print_info(f"All operations should appear in same trace in Langfuse")
        except Exception as e:
            print_error(f"Multi-operation trace failed: {e}")

        # Cleanup
        await cleanup_test_data(memory, TEST_USER_ID)

        print_success("Langfuse integration tests completed")
        print_info("\n📊 Check Langfuse UI for traces:")
        print_info(f"   URL: {settings.langfuse_host}")
        print_info("   Look for traces with 'memory_' prefix in context")
        return True

    except Exception as e:
        print_error(f"Langfuse integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_metadata_and_filters() -> bool:
    """Test metadata and filtering functionality."""
    print_header("Metadata and Filter Tests")

    try:
        memory = MemoryClient()

        if not memory.is_enabled:
            print_error("Memory is not enabled")
            return False

        # Clean up
        await cleanup_test_data(memory, TEST_USER_ID)

        # Test 1: Add with metadata
        print_info("Test 1: Adding memories with metadata...")
        metadata = MemoryMetadata(
            category="preferences",
            tags=["ui", "theme"],
            source="user_input",
            confidence=0.95,
            custom={"priority": "high"},
        )

        result = await memory.add(
            "User prefers dark mode",
            user_id=TEST_USER_ID,
            metadata=metadata,
        )
        print_success("Memory added with rich metadata")

        # Test 2: Search with metadata filters
        print_info("Test 2: Searching with metadata filters...")
        filter_obj = MemoryFilter(
            user_id=TEST_USER_ID,
            category="preferences",
            tags=["ui"],
        )
        results = await memory.search(
            "user preferences",
            user_id=TEST_USER_ID,
            filters=filter_obj,
        )
        print_success(f"Filtered search found {len(results.results)} results")

        # Test 3: Add with dict metadata
        print_info("Test 3: Adding with dict metadata...")
        await memory.add(
            "User is vegetarian",
            user_id=TEST_USER_ID,
            metadata={"category": "dietary", "tags": ["health", "preference"]},
        )
        print_success("Memory added with dict metadata")

        # Test 4: Multiple categories
        print_info("Test 4: Testing multiple categories...")
        categories = ["preferences", "dietary", "work", "personal"]
        for cat in categories:
            await memory.add(
                f"Test memory for {cat} category",
                user_id=TEST_USER_ID,
                metadata={"category": cat},
            )

        # Search by category
        for cat in categories:
            results = await memory.search(
                "test memory",
                user_id=TEST_USER_ID,
                filters={"category": cat},
            )
            print(f"   Category '{cat}': {len(results.results)} results")

        # Cleanup
        await cleanup_test_data(memory, TEST_USER_ID)
        print_success("Metadata and filter tests completed")
        return True

    except Exception as e:
        print_error(f"Metadata test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_conversation_format() -> bool:
    """Test adding memories from conversation format."""
    print_header("Conversation Format Tests")

    try:
        memory = MemoryClient()

        if not memory.is_enabled:
            print_error("Memory is not enabled")
            return False

        # Clean up
        await cleanup_test_data(memory, TEST_USER_ID)

        # Test 1: Single string
        print_info("Test 1: Adding single string...")
        await memory.add("User likes Python", user_id=TEST_USER_ID)
        print_success("Single string added")

        # Test 2: List of strings
        print_info("Test 2: Adding list of strings...")
        await memory.add(
            ["User works at Google", "User likes dark mode"],
            user_id=TEST_USER_ID,
        )
        print_success("List of strings added")

        # Test 3: Chat messages format
        print_info("Test 3: Adding chat messages...")
        messages = [
            {"role": "user", "content": "I'm allergic to peanuts"},
            {"role": "assistant", "content": "I'll remember that you're allergic to peanuts."},
            {"role": "user", "content": "My favorite color is blue"},
            {"role": "assistant", "content": "Got it, blue is your favorite color."},
        ]
        result = await memory.add(messages, user_id=TEST_USER_ID)
        print_success(f"Chat messages added: {result.message}")
        print_info(f"Extracted {len(result.results)} facts from conversation")

        # Test 4: Search to verify extraction
        print_info("Test 4: Verifying extracted facts...")
        results = await memory.search("user allergies", user_id=TEST_USER_ID)
        print_success(f"Found {len(results.results)} results about allergies")
        for entry in results.results:
            if "peanut" in entry.memory.lower():
                print(f"   ✓ {entry.memory}")

        results = await memory.search("favorite color", user_id=TEST_USER_ID)
        print_success(f"Found {len(results.results)} results about color")
        for entry in results.results:
            if "blue" in entry.memory.lower():
                print(f"   ✓ {entry.memory}")

        # Cleanup
        await cleanup_test_data(memory, TEST_USER_ID)
        print_success("Conversation format tests completed")
        return True

    except Exception as e:
        print_error(f"Conversation format test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_global_client() -> bool:
    """Test global memory client."""
    print_header("Global Client Tests")

    try:
        # Test 1: Initialize global client
        print_info("Test 1: Initializing global client...")
        initialized = initialize_global_memory()
        if initialized:
            print_success("Global memory initialized")
        else:
            print_warning("Global memory not initialized (may be disabled)")

        # Test 2: Get global client
        print_info("Test 2: Getting global client...")
        memory = get_global_memory_client()
        print_success("Global client retrieved")

        if memory.is_enabled:
            # Test 3: Use global client
            print_info("Test 3: Using global client...")
            await cleanup_test_data(memory, TEST_USER_ID)

            await memory.add("Test with global client", user_id=TEST_USER_ID)
            results = await memory.search("test", user_id=TEST_USER_ID)
            print_success(f"Global client works: found {len(results.results)} results")

            await cleanup_test_data(memory, TEST_USER_ID)
        else:
            print_warning("Global client not enabled")

        print_success("Global client tests completed")
        return True

    except Exception as e:
        print_error(f"Global client test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_performance() -> bool:
    """Test memory performance with multiple operations."""
    print_header("Performance Tests")

    try:
        memory = MemoryClient()

        if not memory.is_enabled:
            print_error("Memory is not enabled")
            return False

        # Clean up
        await cleanup_test_data(memory, TEST_USER_ID)

        # Test 1: Batch add
        print_info("Test 1: Adding multiple memories...")
        import time

        start = time.time()
        test_memories = [
            f"User fact {i}: Test memory number {i}"
            for i in range(10)
        ]

        for msg in test_memories:
            await memory.add(msg, user_id=TEST_USER_ID)

        elapsed = time.time() - start
        print_success(f"Added {len(test_memories)} memories in {elapsed:.2f}s")
        print_info(f"Average: {elapsed/len(test_memories):.3f}s per memory")

        # Test 2: Search performance
        print_info("Test 2: Testing search performance...")
        start = time.time()
        results = await memory.search("user facts", user_id=TEST_USER_ID, limit=10)
        elapsed = time.time() - start
        print_success(f"Search completed in {elapsed:.3f}s")
        print_info(f"Found {len(results.results)} results")

        # Test 3: Get all performance
        print_info("Test 3: Testing get_all performance...")
        start = time.time()
        all_memories = await memory.get_all(user_id=TEST_USER_ID)
        elapsed = time.time() - start
        print_success(f"Retrieved {len(all_memories)} memories in {elapsed:.3f}s")

        # Cleanup
        await cleanup_test_data(memory, TEST_USER_ID)
        print_success("Performance tests completed")
        return True

    except Exception as e:
        print_error(f"Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_embedder_huggingface() -> bool:
    """Test memory with Hugging Face embedder (native mem0 provider)."""
    print_header("Hugging Face Embedder Tests")

    try:
        # Check if HuggingFace API key is available
        hf_key = os.environ.get("HUGGINGFACE_API_KEY") or os.environ.get("HF_TOKEN")
        if not hf_key:
            print_warning("HUGGINGFACE_API_KEY or HF_TOKEN not set, skipping HuggingFace test")
            print_info("Set HUGGINGFACE_API_KEY=hf_xxx to run this test")
            return True  # Skip but don't fail

        print_info("Testing Hugging Face embedder configuration...")
        
        # Create config with HuggingFace embedder (native mem0 provider)
        # Note: mem0 uses 'huggingface' provider directly, NOT 'litellm'
        config = MemoryConfig(
            embedder_provider="huggingface",
            embedder_model="BAAI/bge-m3",  # Popular multilingual model (just model name, no prefix)
            embedding_dims=1024,  # bge-m3 outputs 1024 dims
        )
        
        # Log configuration
        print_info(f"Embedder provider: {config.embedder_provider}")
        print_info(f"Embedder model: {config.embedder_model}")
        print_info(f"Embedding dims: {config.embedding_dims}")
        
        # Verify mem0 config generation
        mem0_config = config.to_mem0_config()
        embedder_section = mem0_config.get("embedder", {})
        print_info(f"mem0 embedder config: {embedder_section}")
        
        if embedder_section.get("provider") != "huggingface":
            print_error(f"Expected provider 'huggingface', got '{embedder_section.get('provider')}'")
            return False
        print_success("Embedder configuration generated correctly")
        
        # Initialize memory client
        memory = MemoryClient(config=config)
        
        if not memory.is_enabled:
            print_warning("Memory not enabled (Qdrant may not be running)")
            print_info("Start Qdrant with: docker-compose up -d qdrant")
            return True  # Skip but don't fail
        
        print_success("Memory client initialized with HuggingFace embedder")
        
        # Clean up
        hf_test_user = "test-user-hf"
        await cleanup_test_data(memory, hf_test_user)
        
        # Test 1: Add memory
        print_info("Test 1: Adding memory with HuggingFace embeddings...")
        result = await memory.add(
            "User prefers dark mode and uses VS Code for Python development",
            user_id=hf_test_user,
            metadata={"test": "huggingface_embedder"},
        )
        print_success(f"Memory added: {result.message}")
        print_info(f"Extracted {len(result.results)} facts")
        
        # Test 2: Search memory
        print_info("Test 2: Searching with HuggingFace embeddings...")
        results = await memory.search(
            "What IDE does the user prefer?",
            user_id=hf_test_user,
            limit=3,
        )
        print_success(f"Search found {len(results.results)} results")
        for entry in results.results:
            print(f"   - {entry.memory} (score: {entry.score:.3f})")
        
        # Cleanup
        await cleanup_test_data(memory, hf_test_user)
        print_success("HuggingFace embedder test completed successfully! 🤗")
        return True

    except Exception as e:
        print_error(f"HuggingFace embedder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_embedder_cohere() -> bool:
    """Test memory with Cohere embedder (native mem0 provider)."""
    print_header("Cohere Embedder Tests")

    try:
        # Check if Cohere API key is available
        cohere_key = os.environ.get("COHERE_API_KEY")
        if not cohere_key:
            print_warning("COHERE_API_KEY not set, skipping Cohere test")
            print_info("Set COHERE_API_KEY=xxx to run this test")
            return True  # Skip but don't fail

        print_info("Testing Cohere embedder configuration...")
        
        # Create config with Cohere embedder (native mem0 provider)
        # Note: mem0 uses 'cohere' provider directly, NOT 'litellm'
        config = MemoryConfig(
            embedder_provider="cohere",
            embedder_model="embed-english-v3.0",  # Cohere's latest English model (just model name)
            embedding_dims=1024,  # embed-english-v3.0 outputs 1024 dims
        )
        
        # Log configuration
        print_info(f"Embedder provider: {config.embedder_provider}")
        print_info(f"Embedder model: {config.embedder_model}")
        print_info(f"Embedding dims: {config.embedding_dims}")
        
        # Verify mem0 config generation
        mem0_config = config.to_mem0_config()
        embedder_section = mem0_config.get("embedder", {})
        print_info(f"mem0 embedder config: {embedder_section}")
        
        if embedder_section.get("provider") != "cohere":
            print_error(f"Expected provider 'cohere', got '{embedder_section.get('provider')}'")
            return False
        print_success("Embedder configuration generated correctly")
        
        # Initialize memory client
        memory = MemoryClient(config=config)
        
        if not memory.is_enabled:
            print_warning("Memory not enabled (Qdrant may not be running)")
            print_info("Start Qdrant with: docker-compose up -d qdrant")
            return True  # Skip but don't fail
        
        print_success("Memory client initialized with Cohere embedder")
        
        # Clean up
        cohere_test_user = "test-user-cohere"
        await cleanup_test_data(memory, cohere_test_user)
        
        # Test 1: Add memory
        print_info("Test 1: Adding memory with Cohere embeddings...")
        result = await memory.add(
            "User is a senior software engineer at Google working on AI projects",
            user_id=cohere_test_user,
            metadata={"test": "cohere_embedder"},
        )
        print_success(f"Memory added: {result.message}")
        print_info(f"Extracted {len(result.results)} facts")
        
        # Test 2: Search memory
        print_info("Test 2: Searching with Cohere embeddings...")
        results = await memory.search(
            "Where does the user work and what do they do?",
            user_id=cohere_test_user,
            limit=3,
        )
        print_success(f"Search found {len(results.results)} results")
        for entry in results.results:
            print(f"   - {entry.memory} (score: {entry.score:.3f})")
        
        # Cleanup
        await cleanup_test_data(memory, cohere_test_user)
        print_success("Cohere embedder test completed successfully! 🔷")
        return True

    except Exception as e:
        print_error(f"Cohere embedder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_embedder_config() -> bool:
    """Test embedder configuration generation for various providers supported by mem0."""
    print_header("Embedder Configuration Tests")

    try:
        print_info("Testing embedder configuration generation...")
        print_info("Note: mem0 supports these embedder providers: openai, azure_openai, huggingface, ollama, gemini, vertexai, cohere")
        
        # Test 1: OpenAI provider (default)
        print_info("Test 1: OpenAI provider configuration...")
        config = MemoryConfig(
            embedder_provider="openai",
            embedder_model="text-embedding-3-small",
            embedding_dims=1536,
        )
        mem0_cfg = config.to_mem0_config()
        embedder = mem0_cfg["embedder"]
        assert embedder["provider"] == "openai", f"Expected 'openai', got '{embedder['provider']}'"
        assert embedder["config"]["model"] == "text-embedding-3-small"
        print_success("OpenAI config: ✓")
        
        # Test 2: Ollama provider (local)
        print_info("Test 2: Ollama provider configuration...")
        config = MemoryConfig(
            embedder_provider="ollama",
            embedder_model="nomic-embed-text",
            embedding_dims=768,
        )
        mem0_cfg = config.to_mem0_config()
        embedder = mem0_cfg["embedder"]
        assert embedder["provider"] == "ollama", f"Expected 'ollama', got '{embedder['provider']}'"
        assert "host" in embedder["config"], "Ollama config should have 'host'"
        print_success(f"Ollama config: ✓ (host={embedder['config']['host']})")
        
        # Test 3: Azure OpenAI provider
        print_info("Test 3: Azure OpenAI provider configuration...")
        config = MemoryConfig(
            embedder_provider="azure_openai",
            embedder_model="my-embedding-deployment",
            embedding_dims=1536,
            embedder_api_base="https://myresource.openai.azure.com",
        )
        mem0_cfg = config.to_mem0_config()
        embedder = mem0_cfg["embedder"]
        assert embedder["provider"] == "azure_openai", f"Expected 'azure_openai', got '{embedder['provider']}'"
        assert embedder["config"].get("azure_endpoint") == "https://myresource.openai.azure.com"
        print_success("Azure OpenAI config: ✓")
        
        # Test 4: HuggingFace provider
        print_info("Test 4: HuggingFace provider configuration...")
        config = MemoryConfig(
            embedder_provider="huggingface",
            embedder_model="BAAI/bge-small-en-v1.5",
            embedding_dims=384,
        )
        mem0_cfg = config.to_mem0_config()
        embedder = mem0_cfg["embedder"]
        assert embedder["provider"] == "huggingface", f"Expected 'huggingface', got '{embedder['provider']}'"
        assert embedder["config"]["model"] == "BAAI/bge-small-en-v1.5"
        print_success("HuggingFace config: ✓")
        
        # Test 5: Cohere provider
        print_info("Test 5: Cohere provider configuration...")
        config = MemoryConfig(
            embedder_provider="cohere",
            embedder_model="embed-english-v3.0",
            embedding_dims=1024,
        )
        mem0_cfg = config.to_mem0_config()
        embedder = mem0_cfg["embedder"]
        assert embedder["provider"] == "cohere", f"Expected 'cohere', got '{embedder['provider']}'"
        print_success("Cohere config: ✓")
        
        # Test 6: Gemini provider
        print_info("Test 6: Gemini provider configuration...")
        config = MemoryConfig(
            embedder_provider="gemini",
            embedder_model="models/embedding-001",
            embedding_dims=768,
        )
        mem0_cfg = config.to_mem0_config()
        embedder = mem0_cfg["embedder"]
        assert embedder["provider"] == "gemini", f"Expected 'gemini', got '{embedder['provider']}'"
        print_success("Gemini config: ✓")
        
        # Test 7: Explicit API key with OpenAI
        print_info("Test 7: Explicit API key configuration...")
        config = MemoryConfig(
            embedder_provider="openai",
            embedder_model="text-embedding-3-small",
            embedding_dims=1536,
            embedder_api_key="test-api-key",
            embedder_api_base="https://custom.api.com/v1",
        )
        mem0_cfg = config.to_mem0_config()
        embedder = mem0_cfg["embedder"]
        assert embedder["config"].get("api_key") == "test-api-key"
        assert embedder["config"].get("api_base") == "https://custom.api.com/v1"
        print_success("Explicit API key/base config: ✓")
        
        print_success("All embedder configuration tests passed! 🎉")
        return True

    except AssertionError as e:
        print_error(f"Assertion failed: {e}")
        return False
    except Exception as e:
        print_error(f"Embedder config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_session_id_alignment() -> bool:
    """Test session_id to run_id alignment pattern (for session integration)."""
    print_header("Session ID Alignment Tests")

    try:
        memory = MemoryClient()

        if not memory.is_enabled:
            print_error("Memory is not enabled")
            return False

        # Clean up
        await cleanup_test_data(memory, TEST_USER_ID)

        # Test: session_id maps to run_id for alignment
        # This demonstrates the pattern used by SessionClient
        print_info("Test: session_id alignment with run_id...")
        
        # Use session_id as run_id (alignment pattern)
        session_id = TEST_SESSION_ID
        run_id = session_id  # Align session_id with run_id
        
        # Add memory with session_id and run_id aligned
        result = await memory.add(
            "User prefers dark mode and loves Python programming",
            user_id=TEST_USER_ID,
            agent_id=TEST_AGENT_ID,
            run_id=run_id,  # session_id used as run_id
            session_id=session_id,  # For tracing
            metadata={"source": "session_test"},
        )
        print_success(f"Memory added with aligned IDs: session_id={session_id}, run_id={run_id}")
        print_info(f"Extracted {len(result.results)} facts")

        # Search using the same aligned IDs
        results = await memory.search(
            "What does the user prefer?",
            user_id=TEST_USER_ID,
            agent_id=TEST_AGENT_ID,
            run_id=run_id,  # Use same run_id (aligned with session_id)
            session_id=session_id,  # For tracing
        )
        print_success(f"Search with aligned IDs found {len(results.results)} results")
        
        # Verify memories are scoped correctly
        if results.results:
            for entry in results.results:
                print_info(f"  - {entry.memory[:60]}...")
                # Verify run_id alignment
                if entry.run_id == run_id:
                    print_success(f"    ✓ Memory has correct run_id: {entry.run_id}")
                else:
                    print_warning(f"    ⚠ Memory run_id mismatch: {entry.run_id} != {run_id}")

        # Test: Get all memories with run_id scope
        all_memories = await memory.get_all(
            user_id=TEST_USER_ID,
            agent_id=TEST_AGENT_ID,
            run_id=run_id,
        )
        print_success(f"Retrieved {len(all_memories)} memories scoped to run_id={run_id}")

        # Cleanup
        await cleanup_test_data(memory, TEST_USER_ID)
        print_success("Session ID alignment tests completed")
        print_info("Note: This demonstrates the ID alignment pattern used by SessionClient")
        return True

    except Exception as e:
        print_error(f"Session ID alignment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Main Test Runner
# =============================================================================


async def run_all_tests() -> None:
    """Run all tests."""
    print_header("Memory Module - Complete Test Suite")

    tests = [
        ("Basic Operations", test_basic_operations),
        ("Search Functionality", test_search_functionality),
        ("Isolation Levels", test_isolation_levels),
        ("Error Handling", test_error_cases),
        ("Langfuse Integration", test_langfuse_integration),
        ("Metadata and Filters", test_metadata_and_filters),
        ("Conversation Format", test_conversation_format),
        ("Global Client", test_global_client),
        ("Performance", test_performance),
        ("Session ID Alignment", test_session_id_alignment),
        ("Embedder Configuration", test_embedder_config),
        ("HuggingFace Embedder", test_embedder_huggingface),
        ("Cohere Embedder", test_embedder_cohere),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result))
        except Exception as e:
            print_error(f"Test '{name}' crashed: {e}")
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


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test Memory module")
    parser.add_argument(
        "--test",
        choices=[
            "basic",
            "search",
            "isolation",
            "errors",
            "langfuse",
            "metadata",
            "conversation",
            "global",
            "performance",
            "session_alignment",
            "embedder_config",
            "embedder_huggingface",
            "embedder_cohere",
            "all",
        ],
        default="all",
        help="Which test to run",
    )

    args = parser.parse_args()

    test_map = {
        "basic": test_basic_operations,
        "search": test_search_functionality,
        "isolation": test_isolation_levels,
        "errors": test_error_cases,
        "langfuse": test_langfuse_integration,
        "metadata": test_metadata_and_filters,
        "conversation": test_conversation_format,
        "global": test_global_client,
        "performance": test_performance,
        "session_alignment": test_session_id_alignment,
        "embedder_config": test_embedder_config,
        "embedder_huggingface": test_embedder_huggingface,
        "embedder_cohere": test_embedder_cohere,
        "all": run_all_tests,
    }

    test_func = test_map[args.test]

    if args.test == "all":
        await test_func()
    else:
        await test_func()


if __name__ == "__main__":
    asyncio.run(main())

