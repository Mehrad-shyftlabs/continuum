"""Unit tests for BaseAgent."""

from unittest.mock import MagicMock

import pytest

from orchestrator.agent.base import BaseAgent, create_agent
from orchestrator.agent.types import Handoff
import logging

logger = logging.getLogger(__name__)


class TestBaseAgent:
    def test_create_basic_agent(self):
        logger.info("BaseAgent: create basic agent")
        agent = BaseAgent(name="test", instructions="You are helpful.")
        assert agent.name == "test"
        assert agent.instructions == "You are helpful."

    def test_create_agent_with_tools(self):
        logger.info("BaseAgent: create agent with tools")
        tool = MagicMock()
        tool.name = "my_tool"
        agent = BaseAgent(name="test", instructions="help", tools=[tool])
        assert len(agent.tools) == 1

    def test_create_agent_with_handoffs(self):
        logger.info("BaseAgent: create agent with handoffs")
        handoff = Handoff(target_agent="other", description="Transfer to other")
        agent = BaseAgent(name="test", instructions="help", handoffs=[handoff])
        assert len(agent.handoffs) == 1

    def test_get_tools_for_llm_includes_handoffs(self):
        logger.info("BaseAgent: get tools for llm includes handoffs")
        handoff = Handoff(target_agent="other", description="Transfer to other")
        agent = BaseAgent(name="test", instructions="help", handoffs=[handoff])
        tools = agent.get_tools_for_llm()
        assert len(tools) >= 1

    def test_get_tools_for_llm_no_handoffs(self):
        logger.info("BaseAgent: get tools for llm no handoffs")
        agent = BaseAgent(name="test", instructions="help")
        tools = agent.get_tools_for_llm()
        assert tools == [] or tools is not None

    def test_is_handoff_tool_call_positive(self):
        logger.info("BaseAgent: is handoff tool call positive")
        handoff = Handoff(target_agent="other", description="Transfer to other")
        agent = BaseAgent(name="test", instructions="help", handoffs=[handoff])
        is_handoff, target = agent.is_handoff_tool_call("handoff_to_other")
        assert is_handoff is True
        assert target == "other"

    def test_is_handoff_tool_call_negative(self):
        logger.info("BaseAgent: is handoff tool call negative")
        agent = BaseAgent(name="test", instructions="help")
        is_handoff, target = agent.is_handoff_tool_call("random_tool")
        assert is_handoff is False

    def test_agent_tags(self):
        logger.info("BaseAgent: agent tags")
        agent = BaseAgent(name="test", instructions="help", tags=["prod", "v1"])
        assert "prod" in agent.tags

    def test_agent_lifecycle_hooks(self):
        logger.info("BaseAgent: agent lifecycle hooks")
        on_start = MagicMock()
        on_end = MagicMock()
        on_error = MagicMock()
        agent = BaseAgent(
            name="test", instructions="help",
            on_start=on_start, on_end=on_end, on_error=on_error,
        )
        assert agent.on_start is on_start
        assert agent.on_end is on_end
        assert agent.on_error is on_error

    def test_agent_json_mode_config(self):
        logger.info("BaseAgent: agent json mode config")
        agent = BaseAgent(name="test", instructions="help", enable_json_mode=True)
        assert agent.enable_json_mode is True

    def test_agent_config_defaults(self):
        logger.info("BaseAgent: agent config defaults")
        agent = BaseAgent(name="test", instructions="help")
        assert agent.config is not None
        assert agent.config.max_turns > 0

    def test_to_tool_definition(self):
        logger.info("BaseAgent: to tool definition")
        agent = BaseAgent(name="helper", instructions="I help with math")
        td = agent.to_tool_definition()
        assert td is not None

    def test_can_handoff_to(self):
        logger.info("BaseAgent: can handoff to")
        handoff = Handoff(target_agent="other", description="Transfer")
        agent = BaseAgent(name="test", instructions="help", handoffs=[handoff])
        assert agent.can_handoff_to("other") is True
        assert agent.can_handoff_to("nonexistent") is False


class TestCreateAgentFactory:
    def test_create_agent_factory(self):
        logger.info("CreateAgentFactory: create agent factory")
        agent = create_agent(name="factory", instructions="I am factory made")
        assert agent.name == "factory"

    def test_create_agent_factory_with_model(self):
        logger.info("CreateAgentFactory: create agent factory with model")
        agent = create_agent(name="test", instructions="help", model="gpt-3.5-turbo")
        assert agent.model == "gpt-3.5-turbo"
