"""
Unit tests for agent type definitions and data classes.

Converted from tests/test_agent.py manual test script.
"""

import pytest

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
import logging

logger = logging.getLogger(__name__)


class TestGenerateIds:
    """Tests for ID generation utilities."""

    def test_generate_run_id(self):
        logger.info("GenerateIds: generate run id")
        run_id = generate_run_id()
        assert run_id.startswith("run_")

    def test_generate_handoff_id(self):
        logger.info("GenerateIds: generate handoff id")
        handoff_id = generate_handoff_id()
        assert handoff_id.startswith("handoff_")

    def test_unique_run_ids(self):
        logger.info("GenerateIds: unique run ids")
        ids = {generate_run_id() for _ in range(100)}
        assert len(ids) == 100

    def test_unique_handoff_ids(self):
        logger.info("GenerateIds: unique handoff ids")
        ids = {generate_handoff_id() for _ in range(100)}
        assert len(ids) == 100


class TestTokenUsage:
    """Tests for TokenUsage data class."""

    def test_token_usage_creation(self):
        logger.info("TokenUsage: token usage creation")
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150

    def test_token_usage_add(self):
        logger.info("TokenUsage: token usage add")
        usage1 = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        usage2 = TokenUsage(prompt_tokens=200, completion_tokens=100, total_tokens=300)
        combined = usage1.add(usage2)

        assert combined.prompt_tokens == 300
        assert combined.completion_tokens == 150
        assert combined.total_tokens == 450


class TestRunState:
    """Tests for RunState serialization."""

    def test_run_state_serialization(self):
        logger.info("RunState: run state serialization")
        run_id = generate_run_id()
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


class TestAgentResponse:
    """Tests for AgentResponse data class."""

    def test_agent_response_to_dict(self):
        logger.info("AgentResponse: agent response to dict")
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        response = AgentResponse(
            content="Hello, world!",
            agent_name="test-agent",
            status=ResponseStatus.SUCCESS,
            usage=usage,
            turn_count=1,
        )

        response_dict = response.to_dict()
        assert response_dict["content"] == "Hello, world!"
        assert response_dict["status"] == "success"


class TestAgentEvent:
    """Tests for AgentEvent data class."""

    def test_agent_event_to_dict(self):
        logger.info("AgentEvent: agent event to dict")
        run_id = generate_run_id()
        event = AgentEvent(
            type=EventType.CONTENT_DELTA,
            agent_name="test-agent",
            run_id=run_id,
            data={"content": "Hello"},
        )

        event_dict = event.to_dict()
        assert event_dict["type"] == "content_delta"


class TestHandoff:
    """Tests for Handoff data class."""

    def test_handoff_to_tool_definition(self):
        logger.info("Handoff: handoff to tool definition")
        handoff = Handoff(
            target_agent="specialist",
            description="Handle complex queries",
        )

        tool_def = handoff.to_tool_definition()
        assert tool_def["type"] == "function"
        assert "handoff_to_specialist" in tool_def["function"]["name"]
