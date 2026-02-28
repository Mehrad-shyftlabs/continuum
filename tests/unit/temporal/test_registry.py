"""Tests for AgentRegistry."""

import pytest
from unittest.mock import MagicMock

from orchestrator.temporal.exceptions import AgentNotRegisteredError
from orchestrator.temporal.registry import (
    AgentRegistry,
    get_agent_registry,
    reset_agent_registry,
)
import logging

logger = logging.getLogger(__name__)


def _make_agent(name: str) -> MagicMock:
    agent = MagicMock()
    agent.name = name
    return agent


class TestAgentRegistry:
    def setup_method(self):
        self.registry = AgentRegistry()

    def test_register_and_get(self):
        logger.info("AgentRegistry: register and get")
        agent = _make_agent("test-agent")
        self.registry.register(agent)
        assert self.registry.get("test-agent") is agent

    def test_get_missing_raises(self):
        logger.info("AgentRegistry: get missing raises")
        with pytest.raises(AgentNotRegisteredError) as exc_info:
            self.registry.get("nonexistent")
        assert "nonexistent" in str(exc_info.value)

    def test_get_missing_includes_available(self):
        logger.info("AgentRegistry: get missing includes available")
        self.registry.register(_make_agent("agent-a"))
        with pytest.raises(AgentNotRegisteredError) as exc_info:
            self.registry.get("missing")
        assert "agent-a" in exc_info.value.context["available_agents"]

    def test_register_many(self):
        logger.info("AgentRegistry: register many")
        agents = [_make_agent("a"), _make_agent("b"), _make_agent("c")]
        self.registry.register_many(agents)
        assert self.registry.list_agents() == ["a", "b", "c"]

    def test_list_agents_empty(self):
        logger.info("AgentRegistry: list agents empty")
        assert self.registry.list_agents() == []

    def test_list_agents(self):
        logger.info("AgentRegistry: list agents")
        self.registry.register(_make_agent("x"))
        self.registry.register(_make_agent("y"))
        assert set(self.registry.list_agents()) == {"x", "y"}

    def test_register_overwrites(self):
        logger.info("AgentRegistry: register overwrites")
        agent1 = _make_agent("same")
        agent2 = _make_agent("same")
        self.registry.register(agent1)
        self.registry.register(agent2)
        assert self.registry.get("same") is agent2

    def test_set_runner_factory(self):
        logger.info("AgentRegistry: set runner factory")
        mock_runner = MagicMock()
        self.registry.set_runner_factory(lambda: mock_runner)
        assert self.registry.get_runner() is mock_runner

    def test_get_runner_caches(self):
        logger.info("AgentRegistry: get runner caches")
        mock_runner = MagicMock()
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return mock_runner

        self.registry.set_runner_factory(factory)
        self.registry.get_runner()
        self.registry.get_runner()
        assert call_count == 1

    def test_set_notification_handler(self):
        logger.info("AgentRegistry: set notification handler")
        handler = MagicMock()
        self.registry.set_notification_handler(handler)
        assert self.registry.get_notification_handler() is handler

    def test_get_notification_handler_default_none(self):
        logger.info("AgentRegistry: get notification handler default none")
        assert self.registry.get_notification_handler() is None

    def test_clear(self):
        logger.info("AgentRegistry: clear")
        self.registry.register(_make_agent("a"))
        self.registry.set_runner_factory(lambda: MagicMock())
        self.registry.set_notification_handler(MagicMock())
        self.registry.clear()
        assert self.registry.list_agents() == []
        assert self.registry.get_notification_handler() is None


class TestGlobalRegistry:
    def setup_method(self):
        reset_agent_registry()

    def teardown_method(self):
        reset_agent_registry()

    def test_singleton(self):
        logger.info("GlobalRegistry: singleton")
        r1 = get_agent_registry()
        r2 = get_agent_registry()
        assert r1 is r2

    def test_reset(self):
        logger.info("GlobalRegistry: reset")
        r1 = get_agent_registry()
        reset_agent_registry()
        r2 = get_agent_registry()
        assert r1 is not r2
