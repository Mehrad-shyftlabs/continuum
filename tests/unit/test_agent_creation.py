"""
Unit tests for agent creation, registration, and agent-as-tool functionality.

Converted from tests/test_agent.py manual test script.
"""

import pytest

from orchestrator.agent import (
    BaseAgent,
    Handoff,
    AgentMemoryConfig,
    MemoryScope,
    create_agent,
    AgentRunner,
    RunnerConfig,
    RunStateManager,
    RunState,
    RunStatus,
    generate_run_id,
    agent_as_tool,
)
import logging

logger = logging.getLogger(__name__)


class TestAgentCreation:
    """Tests for creating agents with various configurations."""

    def test_basic_agent_creation(self):
        logger.info("AgentCreation: basic agent creation")
        agent = BaseAgent(
            name="test-agent",
            instructions="You are a helpful assistant.",
            model="gpt-4o-mini",
        )
        assert agent.name == "test-agent"
        assert agent.model == "gpt-4o-mini"

    def test_agent_with_tools(self):
        logger.info("AgentCreation: agent with tools")
        tool = {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search for information",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        agent = BaseAgent(
            name="tool-agent",
            instructions="You can search.",
            tools=[tool],
        )
        assert len(agent.tools) == 1

    def test_agent_with_handoffs(self):
        logger.info("AgentCreation: agent with handoffs")
        agent = BaseAgent(
            name="triage-agent",
            instructions="Route requests.",
            handoffs=[
                Handoff(
                    target_agent="specialist",
                    description="Hand off to specialist",
                ),
            ],
        )
        assert len(agent.handoffs) == 1
        assert agent.can_handoff_to("specialist")
        assert not agent.can_handoff_to("unknown")

    def test_factory_function(self):
        logger.info("AgentCreation: factory function")
        agent = create_agent(
            name="factory-agent",
            instructions="Created via factory.",
            memory_scope=MemoryScope.USER,
            store_memories=True,
        )
        assert agent.name == "factory-agent"
        assert agent.memory_config.store_memories is True

    def test_get_tools_for_llm_includes_handoffs(self):
        logger.info("AgentCreation: get tools for llm includes handoffs")
        agent = BaseAgent(
            name="triage-agent",
            instructions="Route requests.",
            handoffs=[
                Handoff(
                    target_agent="specialist",
                    description="Hand off to specialist",
                ),
            ],
        )
        tools = agent.get_tools_for_llm()
        assert len(tools) == 1
        assert "handoff_to_specialist" in tools[0]["function"]["name"]

    def test_is_handoff_tool_call(self):
        logger.info("AgentCreation: is handoff tool call")
        agent = BaseAgent(
            name="triage-agent",
            instructions="Route requests.",
            handoffs=[
                Handoff(
                    target_agent="specialist",
                    description="Hand off to specialist",
                ),
            ],
        )
        is_handoff, target = agent.is_handoff_tool_call("handoff_to_specialist")
        assert is_handoff is True
        assert target == "specialist"


class TestAgentRunner:
    """Tests for AgentRunner without actual LLM calls."""

    def test_agent_registration(self):
        logger.info("AgentRunner: agent registration")
        runner = AgentRunner(config=RunnerConfig(persist_state=False))
        agent = BaseAgent(name="test-agent", instructions="You are a test agent.")

        runner.register_agent(agent)
        assert runner.get_agent("test-agent") is not None
        assert runner.get_agent("unknown") is None

    def test_multiple_agent_registration(self):
        logger.info("AgentRunner: multiple agent registration")
        runner = AgentRunner(config=RunnerConfig(persist_state=False))
        agent1 = BaseAgent(name="agent-1", instructions="Agent 1")
        agent2 = BaseAgent(name="agent-2", instructions="Agent 2")

        runner.register_agent(agent1)
        runner.register_agent(agent2)

        assert runner.get_agent("agent-1") is not None
        assert runner.get_agent("agent-2") is not None

    def test_runner_config(self):
        logger.info("AgentRunner: runner config")
        runner = AgentRunner(config=RunnerConfig(persist_state=False))
        assert runner._config.persist_state is False


class TestStateManager:
    """Tests for RunStateManager."""

    def test_create_run_state(self):
        logger.info("StateManager: create run state")
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

    def test_state_serialization(self):
        logger.info("StateManager: state serialization")
        run_id = generate_run_id()
        state = RunState(
            run_id=run_id,
            session_id="session-test",
            user_id="user-test",
            current_agent="test-agent",
            status=RunStatus.RUNNING,
        )
        state_dict = state.to_dict()
        restored = RunState.from_dict(state_dict)

        assert restored.run_id == state.run_id
        assert restored.session_id == state.session_id
        assert restored.status == state.status

    def test_update_timestamp(self):
        logger.info("StateManager: update timestamp")
        import time

        state = RunState(
            run_id=generate_run_id(),
            session_id="session-test",
            user_id="user-test",
            current_agent="test-agent",
            status=RunStatus.RUNNING,
        )
        old_time = state.updated_at
        time.sleep(0.01)
        state.update_timestamp()
        assert state.updated_at > old_time


class TestAgentAsTool:
    """Tests for using an agent as a tool."""

    def test_agent_to_tool_conversion(self):
        logger.info("AgentAsTool: agent to tool conversion")
        math_agent = BaseAgent(
            name="math-expert",
            instructions="You solve math problems.",
            description="Expert in mathematics and calculations",
        )
        tool_def = agent_as_tool(math_agent)

        assert tool_def["type"] == "function"
        assert "consult_math_expert" in tool_def["function"]["name"]
        assert "query" in tool_def["function"]["parameters"]["properties"]

    def test_custom_tool_description(self):
        logger.info("AgentAsTool: custom tool description")
        math_agent = BaseAgent(
            name="math-expert",
            instructions="You solve math problems.",
            description="Expert in mathematics and calculations",
        )
        tool_def = agent_as_tool(math_agent, "Use for complex math problems")
        assert "complex math" in tool_def["function"]["description"]

    def test_agent_to_tool_definition_method(self):
        logger.info("AgentAsTool: agent to tool definition method")
        math_agent = BaseAgent(
            name="math-expert",
            instructions="You solve math problems.",
            description="Expert in mathematics and calculations",
        )
        tool_def = math_agent.to_tool_definition()
        assert tool_def["function"]["name"] == "consult_math_expert"

    def test_agent_with_sub_agent_tool(self):
        logger.info("AgentAsTool: agent with sub agent tool")
        math_agent = BaseAgent(
            name="math-expert",
            instructions="You solve math problems.",
            description="Expert in mathematics and calculations",
        )
        main_agent = BaseAgent(
            name="main-agent",
            instructions="You are a general assistant.",
            tools=[agent_as_tool(math_agent)],
        )
        tools = main_agent.get_tools_for_llm()
        assert len(tools) == 1
        assert "math_expert" in tools[0]["function"]["name"]
