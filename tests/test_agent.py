"""
Tests for the Agent module.

Tests agent creation, execution, handoffs, and workflow agents.
"""

import asyncio
import os
import sys
from typing import Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f" {text}")
    print(f"{'='*60}\n")


def print_result(name: str, success: bool, details: str = "") -> None:
    """Print test result."""
    status = "✓ PASS" if success else "✗ FAIL"
    print(f"  {status}: {name}")
    if details:
        print(f"         {details}")


# =============================================================================
# Test: Agent Creation
# =============================================================================


async def test_agent_creation() -> bool:
    """Test creating basic agents."""
    print_header("Test: Agent Creation")
    
    try:
        from orchestrator.agent import (
            BaseAgent,
            create_agent,
            Handoff,
            AgentMemoryConfig,
            MemoryScope,
        )
        
        # Test 1: Basic agent creation
        agent = BaseAgent(
            name="test-agent",
            instructions="You are a helpful assistant.",
            model="gpt-4o-mini",
        )
        
        assert agent.name == "test-agent"
        assert agent.model == "gpt-4o-mini"
        print_result("Basic agent creation", True)
        
        # Test 2: Agent with tools
        tool = {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search for information",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        
        agent_with_tools = BaseAgent(
            name="tool-agent",
            instructions="You can search.",
            tools=[tool],
        )
        
        assert len(agent_with_tools.tools) == 1
        print_result("Agent with tools", True)
        
        # Test 3: Agent with handoffs
        agent_with_handoffs = BaseAgent(
            name="triage-agent",
            instructions="Route requests.",
            handoffs=[
                Handoff(
                    target_agent="specialist",
                    description="Hand off to specialist",
                ),
            ],
        )
        
        assert len(agent_with_handoffs.handoffs) == 1
        assert agent_with_handoffs.can_handoff_to("specialist")
        assert not agent_with_handoffs.can_handoff_to("unknown")
        print_result("Agent with handoffs", True)
        
        # Test 4: Factory function
        agent_factory = create_agent(
            name="factory-agent",
            instructions="Created via factory.",
            memory_scope=MemoryScope.USER,
            store_memories=True,
        )
        
        assert agent_factory.name == "factory-agent"
        assert agent_factory.memory_config.store_memories is True
        print_result("Factory function", True)
        
        # Test 5: Get tools for LLM (includes handoffs)
        tools = agent_with_handoffs.get_tools_for_llm()
        assert len(tools) == 1  # Handoff as tool
        assert "handoff_to_specialist" in tools[0]["function"]["name"]
        print_result("Get tools for LLM", True)
        
        # Test 6: Is handoff tool call
        is_handoff, target = agent_with_handoffs.is_handoff_tool_call("handoff_to_specialist")
        assert is_handoff is True
        assert target == "specialist"
        print_result("Is handoff tool call", True)
        
        print(f"\n  All agent creation tests passed!")
        return True
        
    except Exception as e:
        print_result("Agent creation", False, str(e))
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test: Agent Types
# =============================================================================


async def test_agent_types() -> bool:
    """Test agent type definitions."""
    print_header("Test: Agent Types")
    
    try:
        from orchestrator.agent import (
            AgentResponse,
            AgentEvent,
            EventType,
            ResponseStatus,
            RunStatus,
            RunState,
            RunContext,
            TokenUsage,
            Handoff,
            HandoffData,
            generate_run_id,
            generate_handoff_id,
        )
        
        # Test 1: Generate IDs
        run_id = generate_run_id()
        assert run_id.startswith("run_")
        print_result("Generate run ID", True, run_id)
        
        handoff_id = generate_handoff_id()
        assert handoff_id.startswith("handoff_")
        print_result("Generate handoff ID", True, handoff_id)
        
        # Test 2: Token usage
        usage1 = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        usage2 = TokenUsage(prompt_tokens=200, completion_tokens=100, total_tokens=300)
        combined = usage1.add(usage2)
        
        assert combined.prompt_tokens == 300
        assert combined.total_tokens == 450
        print_result("Token usage", True, f"Combined: {combined.total_tokens} tokens")
        
        # Test 3: Run state
        state = RunState(
            run_id=run_id,
            session_id="session-123",
            user_id="user-456",
            current_agent="test-agent",
            status=RunStatus.RUNNING,
        )
        
        state_dict = state.to_dict()
        restored = RunState.from_dict(state_dict)
        
        assert restored.run_id == state.run_id
        assert restored.status == RunStatus.RUNNING
        print_result("Run state serialization", True)
        
        # Test 4: Agent response
        response = AgentResponse(
            content="Hello, world!",
            agent_name="test-agent",
            status=ResponseStatus.SUCCESS,
            usage=usage1,
            turn_count=1,
        )
        
        response_dict = response.to_dict()
        assert response_dict["content"] == "Hello, world!"
        assert response_dict["status"] == "success"
        print_result("Agent response", True)
        
        # Test 5: Agent event
        event = AgentEvent(
            type=EventType.CONTENT_DELTA,
            agent_name="test-agent",
            run_id=run_id,
            data={"content": "Hello"},
        )
        
        event_dict = event.to_dict()
        assert event_dict["type"] == "content_delta"
        print_result("Agent event", True)
        
        # Test 6: Handoff to tool definition
        handoff = Handoff(
            target_agent="specialist",
            description="Handle complex queries",
        )
        
        tool_def = handoff.to_tool_definition()
        assert tool_def["type"] == "function"
        assert "handoff_to_specialist" in tool_def["function"]["name"]
        print_result("Handoff to tool definition", True)
        
        print(f"\n  All agent type tests passed!")
        return True
        
    except Exception as e:
        print_result("Agent types", False, str(e))
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test: Workflow Agents
# =============================================================================


async def test_workflow_agents() -> bool:
    """Test workflow agent types."""
    print_header("Test: Workflow Agents")
    
    try:
        from orchestrator.agent import (
            BaseAgent,
            RouterAgent,
            SequentialAgent,
            ParallelAgent,
            LoopAgent,
            Route,
            TerminationConfig,
            TerminationType,
            MergeStrategy,
            FailStrategy,
            create_router_agent,
            create_sequential_agent,
            create_parallel_agent,
            create_loop_agent,
        )
        
        # Create base agents for workflows
        agent1 = BaseAgent(name="agent-1", instructions="Agent 1")
        agent2 = BaseAgent(name="agent-2", instructions="Agent 2")
        agent3 = BaseAgent(name="agent-3", instructions="Agent 3")
        
        # Test 1: RouterAgent
        router = RouterAgent(
            name="router",
            routes=[
                Route(agent_name="agent-1", description="Handle type A requests"),
                Route(agent_name="agent-2", description="Handle type B requests"),
            ],
            fallback_agent_name="agent-3",
        )
        
        assert len(router.routes) == 2
        assert router.get_route("agent-1") is not None
        print_result("RouterAgent creation", True)
        
        # Test 2: SequentialAgent
        sequential = SequentialAgent(
            name="pipeline",
            agents=[agent1, agent2, agent3],
        )
        
        assert len(sequential.agents) == 3
        print_result("SequentialAgent creation", True)
        
        # Test 3: ParallelAgent
        parallel = ParallelAgent(
            name="parallel",
            agents=[agent1, agent2],
        )
        
        assert len(parallel.agents) == 2
        print_result("ParallelAgent creation", True)
        
        # Test 4: LoopAgent
        loop = LoopAgent(
            name="loop",
            agent=agent1,
            termination=TerminationConfig(
                type=TerminationType.LLM_DECISION,
                max_iterations=5,
            ),
        )
        
        assert loop.termination.max_iterations == 5
        print_result("LoopAgent creation", True)
        
        # Test 5: Factory functions
        router_factory = create_router_agent(
            name="triage",
            routes=[
                ("billing", "Handle billing questions"),
                ("technical", "Handle technical issues"),
            ],
            fallback="general",
        )
        
        assert len(router_factory.routes) == 2
        print_result("create_router_agent factory", True)
        
        sequential_factory = create_sequential_agent(
            name="process",
            agents=[agent1, agent2],
            pass_full_history=True,
        )
        
        assert sequential_factory.sequential_config.pass_full_history is True
        print_result("create_sequential_agent factory", True)
        
        parallel_factory = create_parallel_agent(
            name="gather",
            agents=[agent1, agent2],
            merge_strategy=MergeStrategy.CONCATENATE,
        )
        
        assert parallel_factory.parallel_config.merge_strategy == MergeStrategy.CONCATENATE
        print_result("create_parallel_agent factory", True)
        
        loop_factory = create_loop_agent(
            name="iterate",
            agent=agent1,
            max_iterations=10,
            termination_type=TerminationType.OUTPUT_MATCH,
            termination_pattern="DONE",
        )
        
        assert loop_factory.termination.pattern == "DONE"
        print_result("create_loop_agent factory", True)
        
        print(f"\n  All workflow agent tests passed!")
        return True
        
    except Exception as e:
        print_result("Workflow agents", False, str(e))
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test: History Summarization
# =============================================================================


async def test_history_summarization() -> bool:
    """Test history summarization for handoffs."""
    print_header("Test: History Summarization")
    
    try:
        from orchestrator.agent.handoff import (
            HistorySummarizer,
            summarize_conversation,
            extract_nested_history,
            flatten_nested_history,
            format_message_for_summary,
        )
        from orchestrator.agent.types import HistorySummarizationMode
        
        # Sample conversation
        messages = [
            {"role": "user", "content": "Hello, I need help"},
            {"role": "assistant", "content": "Hi! How can I help you today?"},
            {"role": "user", "content": "I have a billing question"},
            {"role": "assistant", "content": "I'll transfer you to billing."},
        ]
        
        # Test 1: Full mode
        full = summarize_conversation(messages, mode=HistorySummarizationMode.FULL)
        assert len(full) == 4
        print_result("Full mode", True, f"{len(full)} messages")
        
        # Test 2: Recent N mode
        recent = summarize_conversation(messages, mode=HistorySummarizationMode.RECENT_N, recent_n=2)
        assert len(recent) == 2
        print_result("Recent N mode", True, f"{len(recent)} messages")
        
        # Test 3: Summary mode
        summary = summarize_conversation(messages, mode=HistorySummarizationMode.SUMMARY)
        assert len(summary) == 1
        assert "<CONVERSATION HISTORY>" in summary[0]["content"]
        print_result("Summary mode", True)
        
        # Test 4: Hybrid mode
        hybrid = summarize_conversation(messages, mode=HistorySummarizationMode.HYBRID, recent_n=2)
        # Should have summary + 2 recent
        assert len(hybrid) == 3
        print_result("Hybrid mode", True, f"{len(hybrid)} messages")
        
        # Test 5: Format message for summary
        tool_msg = {
            "role": "assistant",
            "content": "Let me search",
            "tool_calls": [{"function": {"name": "search"}}],
        }
        formatted = format_message_for_summary(tool_msg)
        assert "search" in formatted
        print_result("Format message with tools", True)
        
        # Test 6: Extract nested history
        summary_msg = summary[0]
        extracted = extract_nested_history(summary_msg)
        assert extracted is not None
        assert len(extracted) > 0
        print_result("Extract nested history", True, f"{len(extracted)} messages extracted")
        
        print(f"\n  All history summarization tests passed!")
        return True
        
    except Exception as e:
        print_result("History summarization", False, str(e))
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test: Agent Runner (Mock)
# =============================================================================


async def test_agent_runner_mock() -> bool:
    """Test AgentRunner without actual LLM calls."""
    print_header("Test: Agent Runner (Mock)")
    
    try:
        from orchestrator.agent import (
            BaseAgent,
            AgentRunner,
            RunnerConfig,
        )
        
        # Create a runner with mocked dependencies
        runner = AgentRunner(
            config=RunnerConfig(
                persist_state=False,  # Don't use Redis for this test
            ),
        )
        
        # Test 1: Agent registration
        agent = BaseAgent(
            name="test-agent",
            instructions="You are a test agent.",
        )
        
        runner.register_agent(agent)
        assert runner.get_agent("test-agent") is not None
        assert runner.get_agent("unknown") is None
        print_result("Agent registration", True)
        
        # Test 2: Register multiple agents
        agent2 = BaseAgent(name="agent-2", instructions="Agent 2")
        runner.register_agent(agent2)
        
        assert runner.get_agent("test-agent") is not None
        assert runner.get_agent("agent-2") is not None
        print_result("Multiple agent registration", True)
        
        # Test 3: Runner config
        assert runner._config.persist_state is False
        print_result("Runner config", True)
        
        print(f"\n  All agent runner mock tests passed!")
        return True
        
    except Exception as e:
        print_result("Agent runner mock", False, str(e))
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test: State Manager
# =============================================================================


async def test_state_manager() -> bool:
    """Test RunStateManager."""
    print_header("Test: State Manager")
    
    try:
        from orchestrator.agent import (
            RunStateManager,
            RunState,
            RunStatus,
            generate_run_id,
        )
        
        # Create state manager (may not connect if Redis not available)
        manager = RunStateManager(auto_initialize=False)
        
        # Test 1: Create run state
        run_id = generate_run_id()
        state = RunState(
            run_id=run_id,
            session_id="session-test",
            user_id="user-test",
            current_agent="test-agent",
            status=RunStatus.RUNNING,
        )
        
        assert state.run_id == run_id
        assert state.status == RunStatus.RUNNING
        print_result("Create run state", True)
        
        # Test 2: Serialize/deserialize
        state_dict = state.to_dict()
        restored = RunState.from_dict(state_dict)
        
        assert restored.run_id == state.run_id
        assert restored.session_id == state.session_id
        assert restored.status == state.status
        print_result("State serialization", True)
        
        # Test 3: Update timestamp
        old_time = state.updated_at
        import time
        time.sleep(0.01)
        state.update_timestamp()
        
        assert state.updated_at > old_time
        print_result("Update timestamp", True)
        
        # Test 4: Try to initialize (may fail without Redis)
        try:
            success = manager.initialize()
            if success:
                print_result("Redis connection", True, "Connected")
            else:
                print_result("Redis connection", True, "Not available (expected in test)")
        except Exception:
            print_result("Redis connection", True, "Not available (expected in test)")
        
        print(f"\n  All state manager tests passed!")
        return True
        
    except Exception as e:
        print_result("State manager", False, str(e))
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test: Agent as Tool
# =============================================================================


async def test_agent_as_tool() -> bool:
    """Test using an agent as a tool."""
    print_header("Test: Agent as Tool")
    
    try:
        from orchestrator.agent import (
            BaseAgent,
            agent_as_tool,
        )
        
        # Create specialist agent
        math_agent = BaseAgent(
            name="math-expert",
            instructions="You solve math problems.",
            description="Expert in mathematics and calculations",
        )
        
        # Test 1: Convert to tool
        tool_def = agent_as_tool(math_agent)
        
        assert tool_def["type"] == "function"
        assert "consult_math_expert" in tool_def["function"]["name"]
        assert "query" in tool_def["function"]["parameters"]["properties"]
        print_result("Agent to tool conversion", True)
        
        # Test 2: Custom description
        tool_def2 = agent_as_tool(math_agent, "Use for complex math problems")
        
        assert "complex math" in tool_def2["function"]["description"]
        print_result("Custom tool description", True)
        
        # Test 3: Use agent's to_tool_definition method
        tool_def3 = math_agent.to_tool_definition()
        
        assert tool_def3["function"]["name"] == "consult_math_expert"
        print_result("Agent.to_tool_definition()", True)
        
        # Test 4: Create main agent with sub-agent as tool
        main_agent = BaseAgent(
            name="main-agent",
            instructions="You are a general assistant.",
            tools=[agent_as_tool(math_agent)],
        )
        
        tools = main_agent.get_tools_for_llm()
        assert len(tools) == 1
        assert "math_expert" in tools[0]["function"]["name"]
        print_result("Agent with sub-agent tool", True)
        
        print(f"\n  All agent as tool tests passed!")
        return True
        
    except Exception as e:
        print_result("Agent as tool", False, str(e))
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Main Test Runner
# =============================================================================


async def run_all_tests() -> None:
    """Run all agent tests."""
    print("\n" + "="*60)
    print(" AGENT MODULE TESTS")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Agent Creation", await test_agent_creation()))
    results.append(("Agent Types", await test_agent_types()))
    results.append(("Workflow Agents", await test_workflow_agents()))
    results.append(("History Summarization", await test_history_summarization()))
    results.append(("Agent Runner (Mock)", await test_agent_runner_mock()))
    results.append(("State Manager", await test_state_manager()))
    results.append(("Agent as Tool", await test_agent_as_tool()))
    
    # Print summary
    print("\n" + "="*60)
    print(" TEST SUMMARY")
    print("="*60 + "\n")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 All tests passed!")
    else:
        print(f"\n  ⚠️  {total - passed} test(s) failed")


if __name__ == "__main__":
    asyncio.run(run_all_tests())

