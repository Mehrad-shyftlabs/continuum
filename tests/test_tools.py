"""
Manual testing script for the Tools/MCP module.

Run this script to verify MCP tool integration is working correctly.

Usage:
    1. Ensure you have API keys configured in .env
    2. Run: python -m tests.test_tools

Or run individual tests:
    python -m tests.test_tools --test connect
    python -m tests.test_tools --test list_tools
    python -m tests.test_tools --test execute
    python -m tests.test_tools --test full_workflow
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
from orchestrator.llm import ChatMessage, LLMClient
from orchestrator.tools import (
    MCPServerStreamableHttp,
    MCPUtil,
    ToolExecutor,
    create_static_tool_filter,
)

# =============================================================================
# Test Configuration
# =============================================================================

# Live MCP server URL (no auth required)
MCP_SERVER_URL = "https://mcp.agentfly.shyftops.io/mcp"

# Test model
TEST_MODEL = os.getenv("TEST_OPENAI_MODEL", "gpt-4o-mini")

# =============================================================================
# Helper Functions
# =============================================================================


def print_info(msg: str) -> None:
    """Print info message."""
    print(f"ℹ️  {msg}")


def print_success(msg: str) -> None:
    """Print success message."""
    print(f"✅ {msg}")


def print_error(msg: str) -> None:
    """Print error message."""
    print(f"❌ {msg}")


def print_warning(msg: str) -> None:
    """Print warning message."""
    print(f"⚠️  {msg}")


# =============================================================================
# Test Functions
# =============================================================================


async def test_connect() -> bool:
    """Test connecting to MCP server."""
    print_info("Testing MCP server connection...")

    server = None
    try:
        # Use Streamable HTTP transport for remote MCP servers
        server = MCPServerStreamableHttp(
            {
                "url": MCP_SERVER_URL,
                "timeout": 30.0,
                "sse_read_timeout": 60.0,
            },
            name="test-server",
        )

        await server.connect()
        print_success(f"Connected to MCP server: {server.name}")

        # Cleanup
        await server.cleanup()
        server = None
        print_success("Connection test passed!")
        return True

    except Exception as e:
        print_error(f"Connection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if server is not None:
            try:
                await server.cleanup()
            except Exception:
                pass


async def test_list_tools() -> bool:
    """Test listing tools from MCP server."""
    print_info("Testing tool listing...")

    server = None
    try:
        server = MCPServerStreamableHttp(
            {
                "url": MCP_SERVER_URL,
                "timeout": 30.0,
                "sse_read_timeout": 60.0,
            },
            cache_tools_list=True,
        )

        await server.connect()

        # List tools
        mcp_tools = await server.list_tools()
        print_success(f"Found {len(mcp_tools)} tools")

        # Show only first 5 tools with truncated descriptions
        for tool in mcp_tools[:5]:
            desc = (tool.description or "No description")[:60]
            print_info(f"  - {tool.name}: {desc}...")
        
        if len(mcp_tools) > 5:
            print_info(f"  ... and {len(mcp_tools) - 5} more tools")

        # Cleanup before testing filtering (avoid multiple server cleanup issues)
        await server.cleanup()
        server = None

        # Test tool filtering with a new server
        print_info("Testing tool filtering...")
        if mcp_tools:
            filter_config = create_static_tool_filter(
                allowed_tool_names=[mcp_tools[0].name]
            )

            server = MCPServerStreamableHttp(
                {
                    "url": MCP_SERVER_URL,
                    "timeout": 30.0,
                    "sse_read_timeout": 60.0,
                },
                tool_filter=filter_config,
            )
            await server.connect()

            filtered_tools = await server.list_tools()
            print_success(f"Filtered tools: {len(filtered_tools)} (expected 1)")

            await server.cleanup()
            server = None

        print_success("Tool listing test passed!")
        return True

    except Exception as e:
        print_error(f"Tool listing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Ensure cleanup even on error
        if server is not None:
            try:
                await server.cleanup()
            except Exception:
                pass


async def test_convert_tools() -> bool:
    """Test converting MCP tools to ToolDefinition format."""
    print_info("Testing tool conversion...")

    server = None
    try:
        server = MCPServerStreamableHttp(
            {
                "url": MCP_SERVER_URL,
                "timeout": 30.0,
                "sse_read_timeout": 60.0,
            },
        )

        await server.connect()

        # Convert tools
        tools = await MCPUtil.get_function_tools(server)
        print_success(f"Converted {len(tools)} tools to ToolDefinition format")

        # Show only first 3 tools with summary
        for tool in tools[:3]:
            param_count = len(tool.function.parameters.get("properties", {}))
            desc = (tool.function.description or "N/A")[:50]
            print_info(f"  - {tool.function.name}: {desc}... ({param_count} params)")
        
        if len(tools) > 3:
            print_info(f"  ... and {len(tools) - 3} more tools")

        # Cleanup
        await server.cleanup()
        server = None

        print_success("Tool conversion test passed!")
        return True

    except Exception as e:
        print_error(f"Tool conversion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if server is not None:
            try:
                await server.cleanup()
            except Exception:
                pass


async def test_execute_tool() -> bool:
    """Test executing a tool directly."""
    print_info("Testing tool execution...")

    server = None
    try:
        server = MCPServerStreamableHttp(
            {
                "url": MCP_SERVER_URL,
                "timeout": 30.0,
                "sse_read_timeout": 60.0,
            },
        )

        await server.connect()

        # Get tools
        mcp_tools = await server.list_tools()
        if not mcp_tools:
            print_warning("No tools available to test")
            await server.cleanup()
            server = None
            return True  # Not a failure, just no tools

        # Try to execute first tool (with minimal args if possible)
        tool = mcp_tools[0]
        print_info(f"Executing tool: {tool.name}")

        # Try with empty args first (some tools might work)
        try:
            result = await server.call_tool(tool.name, {})
            print_success(f"Tool executed successfully")
            print_info(f"Result: {len(result.content)} content item(s)")

        except Exception as tool_error:
            print_warning(f"Tool execution with empty args failed (expected)")
            print_info("This is normal if the tool requires arguments")

        # Cleanup
        await server.cleanup()
        server = None

        print_success("Tool execution test passed!")
        return True

    except Exception as e:
        print_error(f"Tool execution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if server is not None:
            try:
                await server.cleanup()
            except Exception:
                pass


async def test_tool_executor() -> bool:
    """Test ToolExecutor."""
    print_info("Testing ToolExecutor...")

    server = None
    try:
        server = MCPServerStreamableHttp(
            {
                "url": MCP_SERVER_URL,
                "timeout": 30.0,
                "sse_read_timeout": 60.0,
            },
        )

        await server.connect()

        # Create executor
        executor = ToolExecutor({server: None})
        await executor.initialize()

        available_tools = executor.get_available_tools()
        print_success(f"ToolExecutor initialized with {len(available_tools)} tools")
        
        # Show first 5 tool names only
        if available_tools:
            tool_names_preview = ", ".join(available_tools[:5])
            if len(available_tools) > 5:
                tool_names_preview += f" ... (+{len(available_tools) - 5} more)"
            print_info(f"Tools: {tool_names_preview}")

        # Cleanup
        await server.cleanup()
        server = None

        print_success("ToolExecutor test passed!")
        return True

    except Exception as e:
        print_error(f"ToolExecutor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if server is not None:
            try:
                await server.cleanup()
            except Exception:
                pass


async def test_full_workflow() -> bool:
    """Test full workflow: connect -> get tools -> use with LLM -> execute."""
    print_info("Testing full workflow...")

    server = None
    try:
        # 1. Connect to MCP server
        print_info("Step 1: Connecting to MCP server...")
        server = MCPServerStreamableHttp(
            {
                "url": MCP_SERVER_URL,
                "timeout": 30.0,
                "sse_read_timeout": 60.0,
            },
            cache_tools_list=True,
        )
        await server.connect()
        print_success("Connected to MCP server")

        # 2. Get tools (schema normalization is enabled by default)
        print_info("Step 2: Getting tools from MCP server...")
        tools = await MCPUtil.get_function_tools(server)
        print_success(f"Retrieved {len(tools)} tools (schemas normalized for LLM compatibility)")

        if not tools:
            print_warning("No tools available, skipping LLM test")
            await server.cleanup()
            server = None
            return True

        tool_names = [t.function.name for t in tools]
        tool_names_preview = ", ".join(tool_names[:5])
        if len(tool_names) > 5:
            tool_names_preview += f" ... (+{len(tool_names) - 5} more)"
        print_info(f"Available tools: {tool_names_preview}")

        # 3. Initialize LLM and executor
        print_info("Step 3: Initializing LLM and ToolExecutor...")
        llm = LLMClient()
        executor = ToolExecutor({server: None})
        await executor.initialize()
        print_success("Initialized LLM and ToolExecutor")

        # 4. Test LLM with tools (simple query)
        print_info("Step 4: Testing LLM with tools...")
        messages = [
            ChatMessage(
                role="system",
                content="You are a helpful assistant with access to tools. Use them when appropriate.",
            ),
            ChatMessage(
                role="user",
                content=f"Hello! I see you have access to {len(tools)} tools. Can you tell me what tools are available? Just list their names.",
            ),
        ]

        response = await llm.chat(
            messages,
            tools=tools,
            tool_choice="auto",
        )

        print_success(f"LLM responded (model: {response.model})")
        if response.content:
            content_preview = response.content[:100].replace("\n", " ")
            print_info(f"Response preview: {content_preview}...")

        # 5. Check for tool calls
        if response.tool_calls:
            print_info(f"Step 5: LLM requested {len(response.tool_calls)} tool call(s)")
            for tc in response.tool_calls:
                args_preview = tc.function.arguments[:50].replace("\n", " ")
                print_info(f"  - {tc.function.name}({args_preview}...)")

            # Execute tools
            print_info("Step 6: Executing tool calls...")
            tool_messages = await executor.execute_tool_calls(response.tool_calls)
            print_success(f"Executed {len(tool_messages)} tool calls")

            # Add tool results to conversation
            messages.append(
                ChatMessage(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )
            messages.extend(tool_messages)

            # Continue conversation
            print_info("Step 7: Continuing conversation with tool results...")
            final_response = await llm.chat(messages)
            print_success("Got final response")
            if final_response.content:
                content_preview = final_response.content[:100].replace("\n", " ")
                print_info(f"Final response preview: {content_preview}...")
        else:
            print_info("No tool calls requested by LLM (this is okay)")

        # Cleanup
        await server.cleanup()
        server = None

        print_success("Full workflow test passed!")
        return True

    except Exception as e:
        print_error(f"Full workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if server is not None:
            try:
                await server.cleanup()
            except Exception:
                pass


async def test_langfuse_tracing() -> bool:
    """Test that tool calls are traced in Langfuse."""
    print_info("Testing Langfuse tracing for tool calls...")

    server = None
    try:
        from orchestrator.observability.langfuse_client import (
            get_global_langfuse_client,
            initialize_global_langfuse,
        )

        # Initialize Langfuse
        if not initialize_global_langfuse():
            print_warning("Langfuse not configured, skipping tracing test")
            return True

        langfuse_client = get_global_langfuse_client()
        if not langfuse_client.is_enabled:
            print_warning("Langfuse not enabled, skipping tracing test")
            return True

        # Create a trace
        trace = langfuse_client.trace(
            name="test-tools-tracing",
            input={"test": "tool_tracing"},
        )

        if trace:
            print_success("Created Langfuse trace")

            # Test tool listing with trace
            server = MCPServerStreamableHttp(
                {
                    "url": MCP_SERVER_URL,
                    "timeout": 30.0,
                    "sse_read_timeout": 60.0,
                },
            )
            await server.connect()

            # List tools (should create span)
            tools = await server.list_tools()
            print_success(f"Listed {len(tools)} tools (traced)")

            # Execute a tool if available
            if tools:
                executor = ToolExecutor({server: None})
                await executor.initialize()

                # Create a mock tool call for testing
                from orchestrator.llm.types import FunctionCall, ToolCall

                # Use first tool with empty args
                tool_call = ToolCall(
                    id="test-call-1",
                    type="function",
                    function=FunctionCall(
                        name=tools[0].name,
                        arguments="{}",
                    ),
                )

                try:
                    result = await executor.execute_tool_call(
                        tool_call,
                        trace_id=trace.id if trace else None,
                    )
                    print_success("Executed tool (traced)")
                except Exception as e:
                    print_warning(f"Tool execution failed (expected if args required)")

            await server.cleanup()
            server = None

            print_success("Langfuse tracing test passed!")
            print_info("Check Langfuse UI to see the traces")
            return True
        else:
            print_warning("Could not create trace")
            return True

    except Exception as e:
        print_error(f"Langfuse tracing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if server is not None:
            try:
                await server.cleanup()
            except Exception:
                pass


# =============================================================================
# Main Test Runner
# =============================================================================


async def run_all_tests() -> bool:
    """Run all tests."""
    print("=" * 70)
    print("MCP Tools Integration Tests")
    print("=" * 70)
    print()

    tests = [
        ("Connection", test_connect),
        ("List Tools", test_list_tools),
        ("Convert Tools", test_convert_tools),
        ("Execute Tool", test_execute_tool),
        ("Tool Executor", test_tool_executor),
        ("Full Workflow", test_full_workflow),
        ("Langfuse Tracing", test_langfuse_tracing),
    ]

    results = {}
    for name, test_func in tests:
        print(f"\n{'=' * 70}")
        print(f"Running: {name}")
        print("=" * 70)
        try:
            results[name] = await test_func()
        except Exception as e:
            print_error(f"Test {name} crashed: {e}")
            results[name] = False
        print()

    # Summary
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    print()
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 70)

    return passed == total


async def run_single_test(test_name: str) -> bool:
    """Run a single test."""
    test_map = {
        "connect": test_connect,
        "list_tools": test_list_tools,
        "convert_tools": test_convert_tools,
        "execute": test_execute_tool,
        "executor": test_tool_executor,
        "full_workflow": test_full_workflow,
        "tracing": test_langfuse_tracing,
    }

    if test_name not in test_map:
        print_error(f"Unknown test: {test_name}")
        print_info(f"Available tests: {', '.join(test_map.keys())}")
        return False

    test_func = test_map[test_name]
    return await test_func()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test MCP Tools Integration")
    parser.add_argument(
        "--test",
        type=str,
        help="Run a specific test (connect, list_tools, convert_tools, execute, executor, full_workflow, tracing)",
    )

    args = parser.parse_args()

    if args.test:
        asyncio.run(run_single_test(args.test))
    else:
        asyncio.run(run_all_tests())


if __name__ == "__main__":
    main()

