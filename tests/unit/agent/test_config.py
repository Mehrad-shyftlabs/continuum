"""Unit tests for agent config."""

import pytest

from orchestrator.agent.config import (
    AgentConfig,
    AgentMemoryConfig,
    HandoffConfig,
    LoopConfig,
    ParallelConfig,
    RunnerConfig,
    SequentialConfig,
)
import logging

logger = logging.getLogger(__name__)


class TestAgentConfig:
    def test_defaults(self):
        logger.info("AgentConfig: defaults")
        c = AgentConfig()
        assert c.max_turns > 0
        assert c.log_to_session is True

    def test_to_dict(self):
        logger.info("AgentConfig: to dict")
        c = AgentConfig()
        d = c.to_dict()
        assert "max_turns" in d


class TestAgentMemoryConfig:
    def test_defaults(self):
        logger.info("AgentMemoryConfig: defaults")
        c = AgentMemoryConfig()
        assert isinstance(c.search_memories, bool)
        assert isinstance(c.store_memories, bool)
        assert c.search_limit == 5


class TestHandoffConfig:
    def test_defaults(self):
        logger.info("HandoffConfig: defaults")
        c = HandoffConfig()
        assert isinstance(c.to_dict(), dict)


class TestRunnerConfig:
    def test_defaults(self):
        logger.info("RunnerConfig: defaults")
        c = RunnerConfig()
        assert c.circuit_breaker_threshold > 0

    def test_to_dict(self):
        logger.info("RunnerConfig: to dict")
        c = RunnerConfig()
        d = c.to_dict()
        assert "circuit_breaker_threshold" in d


class TestWorkflowConfigs:
    def test_sequential_defaults(self):
        logger.info("WorkflowConfigs: sequential defaults")
        c = SequentialConfig()
        assert isinstance(c, SequentialConfig)

    def test_parallel_defaults(self):
        logger.info("WorkflowConfigs: parallel defaults")
        c = ParallelConfig()
        assert isinstance(c, ParallelConfig)

    def test_loop_defaults(self):
        logger.info("WorkflowConfigs: loop defaults")
        c = LoopConfig()
        assert isinstance(c, LoopConfig)
