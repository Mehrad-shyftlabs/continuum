"""Comprehensive tests for agent/config.py."""

import pytest

from orchestrator.agent.config import (
    AgentConfig,
    AgentMemoryConfig,
    HandoffConfig,
    RunnerConfig,
)
from orchestrator.agent.types import (
    HistorySummarizationMode,
    MemoryScope,
)
import logging

logger = logging.getLogger(__name__)


class TestAgentMemoryConfig:
    def test_defaults(self):
        logger.info("AgentMemoryConfig: defaults")
        c = AgentMemoryConfig()
        assert c.search_memories is True
        assert c.search_scope == MemoryScope.USER
        assert c.search_limit == 5
        assert c.search_threshold == 0.0
        assert c.store_memories is True
        assert c.store_scope == MemoryScope.USER
        assert c.store_assistant_messages is True
        assert c.store_user_messages is True
        assert c.broadcast_learnings is False
        assert c.broadcast_to is None
        assert c.broadcast_threshold == 0.8

    def test_custom(self):
        logger.info("AgentMemoryConfig: custom")
        c = AgentMemoryConfig(
            search_memories=False,
            search_scope=MemoryScope.AGENT,
            search_limit=10,
            store_memories=False,
        )
        assert c.search_memories is False
        assert c.search_scope == MemoryScope.AGENT
        assert c.search_limit == 10

    def test_to_dict(self):
        logger.info("AgentMemoryConfig: to dict")
        c = AgentMemoryConfig()
        d = c.to_dict()
        assert d["search_memories"] is True
        assert d["search_scope"] == "user"
        assert "broadcast_learnings" in d

    def test_from_dict(self):
        logger.info("AgentMemoryConfig: from dict")
        data = {
            "search_memories": False,
            "search_scope": "agent",
            "search_limit": 10,
        }
        c = AgentMemoryConfig.from_dict(data)
        assert c.search_memories is False
        assert c.search_scope == MemoryScope.AGENT
        assert c.search_limit == 10

    def test_from_dict_defaults(self):
        logger.info("AgentMemoryConfig: from dict defaults")
        c = AgentMemoryConfig.from_dict({})
        assert c.search_memories is True
        assert c.search_scope == MemoryScope.USER


class TestHandoffConfig:
    def test_defaults(self):
        logger.info("HandoffConfig: defaults")
        c = HandoffConfig()
        assert c.transfer_history is True
        assert c.summarize_history is True
        assert c.summarization_mode == HistorySummarizationMode.HYBRID
        assert c.recent_messages == 5
        assert c.summary_model is None
        assert c.return_to_parent is True
        assert c.max_handoff_depth == 10

    def test_custom(self):
        logger.info("HandoffConfig: custom")
        c = HandoffConfig(
            max_handoff_depth=5,
            summarization_mode=HistorySummarizationMode.FULL,
        )
        assert c.max_handoff_depth == 5
        assert c.summarization_mode == HistorySummarizationMode.FULL

    def test_to_dict(self):
        logger.info("HandoffConfig: to dict")
        c = HandoffConfig()
        d = c.to_dict()
        assert d["transfer_history"] is True
        assert d["summarization_mode"] == "hybrid"


class TestAgentConfig:
    def test_defaults(self):
        logger.info("AgentConfig: defaults")
        c = AgentConfig()
        assert c.temperature == 0.7
        assert c.max_tokens is None
        assert c.max_turns == 25
        assert c.timeout == 300
        assert c.retry_count == 3
        assert c.input_sanitization is True
        assert c.injection_detection is False
        assert c.output_type == "text"
        assert c.trace_all_turns is True
        assert c.log_to_session is True

    def test_custom(self):
        logger.info("AgentConfig: custom")
        c = AgentConfig(temperature=0.5, max_turns=10)
        assert c.temperature == 0.5
        assert c.max_turns == 10

    def test_memory_config(self):
        logger.info("AgentConfig: memory config")
        c = AgentConfig()
        assert isinstance(c.memory, AgentMemoryConfig)

    def test_handoff_config(self):
        logger.info("AgentConfig: handoff config")
        c = AgentConfig()
        assert isinstance(c.handoff, HandoffConfig)

    def test_to_dict(self):
        logger.info("AgentConfig: to dict")
        c = AgentConfig()
        d = c.to_dict()
        assert "model" in d
        assert "temperature" in d
        assert "memory" in d
        assert "handoff" in d
        assert d["context_management"] is None


class TestRunnerConfig:
    def test_defaults(self):
        logger.info("RunnerConfig: defaults")
        c = RunnerConfig()
        assert c.default_max_turns == 25
        assert c.default_timeout == 300
        assert c.persist_state is True
        assert c.parallel_tool_calls is True
        assert c.max_parallel_tools == 5
        assert c.tool_timeout == 60
        assert c.retry_on_error is True
        assert c.max_retries == 3
        assert c.circuit_breaker_threshold == 5
        assert c.circuit_breaker_cooldown == 60
        assert c.trace_enabled is True

    def test_custom(self):
        logger.info("RunnerConfig: custom")
        c = RunnerConfig(
            default_max_turns=50,
            persist_state=False,
            max_retries=5,
        )
        assert c.default_max_turns == 50
        assert c.persist_state is False
        assert c.max_retries == 5

    def test_to_dict(self):
        logger.info("RunnerConfig: to dict")
        c = RunnerConfig()
        d = c.to_dict()
        assert "default_max_turns" in d
        assert "persist_state" in d
        assert "circuit_breaker_threshold" in d
