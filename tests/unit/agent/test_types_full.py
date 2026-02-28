"""Comprehensive tests for agent/types.py."""

import pytest

from orchestrator.agent.types import (
    AgentEvent,
    AgentResponse,
    EventType,
    FailStrategy,
    Handoff,
    HandoffData,
    HandoffResult,
    HistorySummarizationMode,
    MemoryScope,
    MergeStrategy,
    PrepareRunResult,
    ResponseStatus,
    RunContext,
    RunState,
    RunStatus,
    StepResult,
    TerminationType,
    TokenUsage,
    ToolExecutionSummary,
    generate_handoff_id,
    generate_run_id,
)
import logging

logger = logging.getLogger(__name__)


class TestRunStatus:
    def test_values(self):
        logger.info("RunStatus: values")
        assert RunStatus.PENDING == "pending"
        assert RunStatus.RUNNING == "running"
        assert RunStatus.COMPLETED == "completed"
        assert RunStatus.FAILED == "failed"
        assert RunStatus.CANCELLED == "cancelled"
        assert RunStatus.PAUSED == "paused"
        assert RunStatus.WAITING_FOR_INPUT == "waiting_for_input"
        assert RunStatus.WAITING_FOR_TOOL == "waiting_for_tool"
        assert RunStatus.HANDOFF_PENDING == "handoff_pending"


class TestResponseStatus:
    def test_values(self):
        logger.info("ResponseStatus: values")
        assert ResponseStatus.SUCCESS == "success"
        assert ResponseStatus.ERROR == "error"
        assert ResponseStatus.HANDOFF == "handoff"
        assert ResponseStatus.TOOL_CALL == "tool_call"
        assert ResponseStatus.MAX_TURNS_REACHED == "max_turns_reached"
        assert ResponseStatus.CANCELLED == "cancelled"


class TestEventType:
    def test_run_events(self):
        logger.info("EventType: run events")
        assert EventType.RUN_START == "run_start"
        assert EventType.RUN_END == "run_end"
        assert EventType.RUN_ERROR == "run_error"

    def test_agent_events(self):
        logger.info("EventType: agent events")
        assert EventType.AGENT_START == "agent_start"
        assert EventType.AGENT_END == "agent_end"

    def test_tool_events(self):
        logger.info("EventType: tool events")
        assert EventType.TOOL_CALL_START == "tool_call_start"
        assert EventType.TOOL_CALL_END == "tool_call_end"
        assert EventType.TOOL_CALL_ERROR == "tool_call_error"

    def test_handoff_events(self):
        logger.info("EventType: handoff events")
        assert EventType.HANDOFF_START == "handoff_start"
        assert EventType.HANDOFF_END == "handoff_end"
        assert EventType.HANDOFF_RETURN == "handoff_return"

    def test_memory_events(self):
        logger.info("EventType: memory events")
        assert EventType.MEMORY_RETRIEVAL == "memory_retrieval"
        assert EventType.MEMORY_STORAGE == "memory_storage"


class TestHistorySummarizationMode:
    def test_values(self):
        logger.info("HistorySummarizationMode: values")
        assert HistorySummarizationMode.FULL == "full"
        assert HistorySummarizationMode.SUMMARY == "summary"
        assert HistorySummarizationMode.RECENT_N == "recent_n"
        assert HistorySummarizationMode.HYBRID == "hybrid"


class TestMemoryScope:
    def test_values(self):
        logger.info("MemoryScope: values")
        assert MemoryScope.SHARED == "shared"
        assert MemoryScope.USER == "user"
        assert MemoryScope.AGENT == "agent"
        assert MemoryScope.RUN == "run"


class TestMergeStrategy:
    def test_values(self):
        logger.info("MergeStrategy: values")
        assert MergeStrategy.CONCATENATE == "concatenate"
        assert MergeStrategy.LLM_SUMMARIZE == "llm_summarize"
        assert MergeStrategy.STRUCTURED == "structured"
        assert MergeStrategy.FIRST_SUCCESS == "first_success"


class TestFailStrategy:
    def test_values(self):
        logger.info("FailStrategy: values")
        assert FailStrategy.FAIL_FAST == "fail_fast"
        assert FailStrategy.CONTINUE_ON_ERROR == "continue"
        assert FailStrategy.REQUIRE_ALL == "require_all"


class TestTerminationType:
    def test_values(self):
        logger.info("TerminationType: values")
        assert TerminationType.LLM_DECISION == "llm_decision"
        assert TerminationType.TOOL_CALL == "tool_call"
        assert TerminationType.OUTPUT_MATCH == "output_match"
        assert TerminationType.CUSTOM == "custom"


class TestGenerateRunId:
    def test_unique(self):
        logger.info("GenerateRunId: unique")
        id1 = generate_run_id()
        id2 = generate_run_id()
        assert id1 != id2
        assert id1.startswith("run_")

    def test_format(self):
        logger.info("GenerateRunId: format")
        run_id = generate_run_id()
        assert isinstance(run_id, str)
        assert len(run_id) > 4


class TestGenerateHandoffId:
    def test_unique(self):
        logger.info("GenerateHandoffId: unique")
        id1 = generate_handoff_id()
        id2 = generate_handoff_id()
        assert id1 != id2
        assert id1.startswith("handoff_")


class TestTokenUsage:
    def test_defaults(self):
        logger.info("TokenUsage: defaults")
        tu = TokenUsage()
        assert tu.prompt_tokens == 0
        assert tu.completion_tokens == 0
        assert tu.total_tokens == 0

    def test_to_dict(self):
        logger.info("TokenUsage: to dict")
        tu = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        d = tu.to_dict()
        assert d["prompt_tokens"] == 100
        assert d["completion_tokens"] == 50
        assert d["total_tokens"] == 150


class TestToolExecutionSummary:
    def test_defaults(self):
        logger.info("ToolExecutionSummary: defaults")
        s = ToolExecutionSummary()
        assert s.tool_count == 0
        assert s.success_count == 0
        assert s.error_count == 0
        assert s.tools_used == []
        assert s.total_latency_ms == 0.0

    def test_add_tool_execution_success(self):
        logger.info("ToolExecutionSummary: add tool execution success")
        s = ToolExecutionSummary()
        s.add_tool_execution("search", 50.0, server_name="mcp-server", success=True)
        assert s.tool_count == 1
        assert s.success_count == 1
        assert s.error_count == 0
        assert "search" in s.tools_used
        assert s.total_latency_ms == 50.0

    def test_add_tool_execution_failure(self):
        logger.info("ToolExecutionSummary: add tool execution failure")
        s = ToolExecutionSummary()
        s.add_tool_execution("broken_tool", 10.0, success=False, error="timeout")
        assert s.tool_count == 1
        assert s.success_count == 0
        assert s.error_count == 1
        assert any("timeout" in e for e in s.errors)

    def test_to_dict(self):
        logger.info("ToolExecutionSummary: to dict")
        s = ToolExecutionSummary()
        s.add_tool_execution("tool1", 25.0, success=True)
        d = s.to_dict()
        assert d["tool_count"] == 1


class TestRunContext:
    def test_creation(self):
        logger.info("RunContext: creation")
        ctx = RunContext(run_id="r1")
        assert ctx.run_id == "r1"
        assert ctx.user_id is None
        assert ctx.session_id is None
        assert ctx.max_turns == 25
        assert ctx.agent_stack == []
        assert ctx.tags == []

    def test_with_user(self):
        logger.info("RunContext: with user")
        ctx = RunContext(run_id="r1", user_id="u1", session_id="s1")
        assert ctx.user_id == "u1"
        assert ctx.session_id == "s1"

    def test_to_dict(self):
        logger.info("RunContext: to dict")
        ctx = RunContext(run_id="r1", user_id="u1")
        d = ctx.to_dict()
        assert d["run_id"] == "r1"
        assert d["user_id"] == "u1"
        assert "max_turns" in d
        assert "usage" in d


class TestRunState:
    def test_creation(self):
        logger.info("RunState: creation")
        state = RunState(run_id="r1")
        assert state.run_id == "r1"
        assert state.status == RunStatus.PENDING
        assert state.turn_count == 0
        assert state.max_turns == 25

    def test_to_dict(self):
        logger.info("RunState: to dict")
        state = RunState(run_id="r1", current_agent="a1")
        d = state.to_dict()
        assert d["run_id"] == "r1"
        assert d["current_agent"] == "a1"
        assert d["status"] == "pending"


class TestAgentEvent:
    def test_creation(self):
        logger.info("AgentEvent: creation")
        evt = AgentEvent(
            type=EventType.RUN_START,
            agent_name="a1",
            run_id="r1",
        )
        assert evt.type == EventType.RUN_START
        assert evt.agent_name == "a1"

    def test_to_dict(self):
        logger.info("AgentEvent: to dict")
        evt = AgentEvent(
            type=EventType.TOOL_CALL_START,
            agent_name="a1",
            run_id="r1",
            data={"tool": "search"},
        )
        d = evt.to_dict()
        assert d["type"] == "tool_call_start"
        assert d["data"]["tool"] == "search"


class TestAgentResponse:
    def test_creation(self):
        logger.info("AgentResponse: creation")
        r = AgentResponse(
            content="Hello!",
            run_id="r1",
            agent_name="a1",
        )
        assert r.content == "Hello!"
        assert r.status == ResponseStatus.SUCCESS

    def test_with_error(self):
        logger.info("AgentResponse: with error")
        r = AgentResponse(
            content="",
            status=ResponseStatus.ERROR,
            run_id="r1",
            agent_name="a1",
            error="something failed",
        )
        assert r.error == "something failed"

    def test_to_dict(self):
        logger.info("AgentResponse: to dict")
        r = AgentResponse(content="Hi", run_id="r1", agent_name="a1")
        d = r.to_dict()
        assert d["content"] == "Hi"
        assert "status" in d


class TestStepResult:
    def test_creation(self):
        logger.info("StepResult: creation")
        state = RunState(run_id="r1")
        sr = StepResult(run_state=state)
        assert sr.is_complete is False
        assert sr.requires_input is False
        assert sr.response is None


class TestPrepareRunResult:
    def test_creation(self):
        logger.info("PrepareRunResult: creation")
        ctx = RunContext(run_id="r1")
        state = RunState(run_id="r1")
        p = PrepareRunResult(
            success=True,
            context=ctx,
            run_state=state,
        )
        assert p.success is True
        assert p.context is ctx
        assert p.run_state is state

    def test_failure(self):
        logger.info("PrepareRunResult: failure")
        r = AgentResponse(content="", status=ResponseStatus.ERROR, error="bad input")
        p = PrepareRunResult(success=False, error_response=r)
        assert p.success is False
        assert p.error_response is not None
