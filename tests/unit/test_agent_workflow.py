"""
Unit tests for workflow agents (Router, Sequential, Parallel, Loop).

Converted from tests/test_agent.py manual test script.
"""

import pytest

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
from orchestrator.agent.handoff import (
    HistorySummarizer,
    summarize_conversation,
    extract_nested_history,
    flatten_nested_history,
    format_message_for_summary,
)
from orchestrator.agent.types import HistorySummarizationMode
import logging

logger = logging.getLogger(__name__)


@pytest.fixture
def base_agents():
    """Create a set of base agents for workflow tests."""
    return [
        BaseAgent(name="agent-1", instructions="Agent 1"),
        BaseAgent(name="agent-2", instructions="Agent 2"),
        BaseAgent(name="agent-3", instructions="Agent 3"),
    ]


class TestRouterAgent:
    def test_creation(self):
        logger.info("RouterAgent: creation")
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

    def test_factory_function(self):
        logger.info("RouterAgent: factory function")
        router = create_router_agent(
            name="triage",
            routes=[
                ("billing", "Handle billing questions"),
                ("technical", "Handle technical issues"),
            ],
            fallback="general",
        )
        assert len(router.routes) == 2


class TestSequentialAgent:
    def test_creation(self, base_agents):
        logger.info("SequentialAgent: creation")
        sequential = SequentialAgent(
            name="pipeline",
            agents=base_agents,
        )
        assert len(sequential.agents) == 3

    def test_factory_function(self, base_agents):
        logger.info("SequentialAgent: factory function")
        sequential = create_sequential_agent(
            name="process",
            agents=base_agents[:2],
            pass_full_history=True,
        )
        assert sequential.sequential_config.pass_full_history is True


class TestParallelAgent:
    def test_creation(self, base_agents):
        logger.info("ParallelAgent: creation")
        parallel = ParallelAgent(
            name="parallel",
            agents=base_agents[:2],
        )
        assert len(parallel.agents) == 2

    def test_factory_function(self, base_agents):
        logger.info("ParallelAgent: factory function")
        parallel = create_parallel_agent(
            name="gather",
            agents=base_agents[:2],
            merge_strategy=MergeStrategy.CONCATENATE,
        )
        assert parallel.parallel_config.merge_strategy == MergeStrategy.CONCATENATE


class TestLoopAgent:
    def test_creation(self, base_agents):
        logger.info("LoopAgent: creation")
        loop = LoopAgent(
            name="loop",
            agent=base_agents[0],
            termination=TerminationConfig(
                type=TerminationType.LLM_DECISION,
                max_iterations=5,
            ),
        )
        assert loop.termination.max_iterations == 5

    def test_factory_function(self, base_agents):
        logger.info("LoopAgent: factory function")
        loop = create_loop_agent(
            name="iterate",
            agent=base_agents[0],
            max_iterations=10,
            termination_type=TerminationType.OUTPUT_MATCH,
            termination_pattern="DONE",
        )
        assert loop.termination.pattern == "DONE"


class TestHistorySummarization:
    """Tests for history summarization during handoffs."""

    @pytest.fixture
    def sample_messages(self):
        return [
            {"role": "user", "content": "Hello, I need help"},
            {"role": "assistant", "content": "Hi! How can I help you today?"},
            {"role": "user", "content": "I have a billing question"},
            {"role": "assistant", "content": "I'll transfer you to billing."},
        ]

    def test_full_mode(self, sample_messages):
        logger.info("HistorySummarization: full mode")
        full = summarize_conversation(
            sample_messages, mode=HistorySummarizationMode.FULL
        )
        assert len(full) == 4

    def test_recent_n_mode(self, sample_messages):
        logger.info("HistorySummarization: recent n mode")
        recent = summarize_conversation(
            sample_messages, mode=HistorySummarizationMode.RECENT_N, recent_n=2
        )
        assert len(recent) == 2

    def test_summary_mode(self, sample_messages):
        logger.info("HistorySummarization: summary mode")
        summary = summarize_conversation(
            sample_messages, mode=HistorySummarizationMode.SUMMARY
        )
        assert len(summary) == 1
        assert "<CONVERSATION HISTORY>" in summary[0]["content"]

    def test_hybrid_mode(self, sample_messages):
        logger.info("HistorySummarization: hybrid mode")
        hybrid = summarize_conversation(
            sample_messages, mode=HistorySummarizationMode.HYBRID, recent_n=2
        )
        assert len(hybrid) == 3

    def test_format_message_with_tools(self):
        logger.info("HistorySummarization: format message with tools")
        tool_msg = {
            "role": "assistant",
            "content": "Let me search",
            "tool_calls": [{"function": {"name": "search"}}],
        }
        formatted = format_message_for_summary(tool_msg)
        assert "search" in formatted

    def test_extract_nested_history(self, sample_messages):
        logger.info("HistorySummarization: extract nested history")
        summary = summarize_conversation(
            sample_messages, mode=HistorySummarizationMode.SUMMARY
        )
        summary_msg = summary[0]
        extracted = extract_nested_history(summary_msg)
        assert extracted is not None
        assert len(extracted) > 0
