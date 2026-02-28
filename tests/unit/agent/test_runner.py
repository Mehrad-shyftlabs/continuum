"""Comprehensive tests for agent/runner.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.agent.base import BaseAgent
from orchestrator.agent.config import RunnerConfig
from orchestrator.agent.types import (
    AgentResponse,
    EventType,
    ResponseStatus,
    RunContext,
    RunState,
    RunStatus,
)
import logging

logger = logging.getLogger(__name__)


class TestAgentRunner:
    def _make_runner(self, **kwargs):
        from orchestrator.agent.runner import AgentRunner

        mock_container = MagicMock()
        mock_container.llm_client = MagicMock()
        mock_container.memory_client = MagicMock()
        mock_container.session_client = MagicMock()
        mock_container.tool_executor = MagicMock()

        return AgentRunner(container=mock_container, **kwargs)

    def test_init_defaults(self):
        logger.info("AgentRunner: init defaults")
        runner = self._make_runner()
        assert runner._config is not None
        assert runner._agent_registry == {}
        assert runner._circuit_breaker is not None

    def test_init_with_config(self):
        logger.info("AgentRunner: init with config")
        config = RunnerConfig(default_max_turns=5)
        runner = self._make_runner(config=config)
        assert runner._config.default_max_turns == 5

    def test_init_with_registry(self):
        logger.info("AgentRunner: init with registry")
        agent = MagicMock(spec=BaseAgent)
        agent.name = "test"
        runner = self._make_runner(agent_registry={"test": agent})
        assert "test" in runner._agent_registry

    def test_init_creates_services(self):
        logger.info("AgentRunner: init creates services")
        runner = self._make_runner()
        assert runner._context_service is not None
        assert runner._memory_service is not None
        assert runner._session_service is not None
        assert runner._tool_service is not None
        assert runner._handoff_manager is not None

    def test_init_with_explicit_clients(self):
        logger.info("AgentRunner: init with explicit clients")
        mock_llm = MagicMock()
        mock_mem = MagicMock()
        mock_sess = MagicMock()
        mock_tool = MagicMock()
        mock_tracing = MagicMock()

        runner = self._make_runner(
            llm_client=mock_llm,
            memory_client=mock_mem,
            session_client=mock_sess,
            tool_executor=mock_tool,
            tracing_manager=mock_tracing,
        )
        assert runner._llm_client is mock_llm
        assert runner._memory_client is mock_mem
        assert runner._session_client is mock_sess
        assert runner._tool_executor is mock_tool
        assert runner._tracing_manager is mock_tracing


class TestRunnerConfig:
    def test_defaults(self):
        logger.info("RunnerConfig: defaults")
        config = RunnerConfig()
        assert config.default_max_turns == 25
        assert config.default_timeout == 300
        assert config.persist_state is True
        assert config.parallel_tool_calls is True
        assert config.retry_on_error is True
        assert config.max_retries == 3
        assert config.trace_enabled is True

    def test_custom(self):
        logger.info("RunnerConfig: custom")
        config = RunnerConfig(default_max_turns=10, default_timeout=60)
        assert config.default_max_turns == 10
        assert config.default_timeout == 60

    def test_circuit_breaker(self):
        logger.info("RunnerConfig: circuit breaker")
        config = RunnerConfig(circuit_breaker_threshold=10, circuit_breaker_cooldown=120)
        assert config.circuit_breaker_threshold == 10
        assert config.circuit_breaker_cooldown == 120

    def test_to_dict(self):
        logger.info("RunnerConfig: to dict")
        config = RunnerConfig()
        d = config.to_dict()
        assert "default_max_turns" in d
        assert "default_timeout" in d
