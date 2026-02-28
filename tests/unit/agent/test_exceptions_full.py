"""Comprehensive tests for agent/exceptions.py."""

import pytest

from orchestrator.agent.exceptions import (
    AgentConfigurationError,
    AgentError,
    AgentExecutionError,
    AgentNotFoundError,
    AgentTimeoutError,
    AgentToolError,
    HandoffCycleDetectedError,
    HandoffDepthExceededError,
    HandoffError,
    HandoffNotAllowedError,
    HandoffTargetNotFoundError,
    LoopMaxIterationsError,
    LoopWorkflowError,
    MaxTurnsExceededError,
    NoRouteFoundError,
    ParallelWorkflowError,
    RouterError,
    RunStateError,
    RunStateNotFoundError,
    RunStatePersistenceError,
    SequentialWorkflowError,
    WorkflowError,
)
import logging

logger = logging.getLogger(__name__)


class TestAgentError:
    def test_basic(self):
        logger.info("AgentError: basic")
        err = AgentError("test")
        assert isinstance(err, Exception)

    def test_with_all_context(self):
        logger.info("AgentError: with all context")
        err = AgentError("err", agent_name="a1", run_id="r1", trace_id="t1",
                         context={"key": "val"}, original_error=ValueError("orig"))
        assert err.agent_name == "a1"
        assert err.run_id == "r1"
        assert err.trace_id == "t1"
        assert err.context["agent_name"] == "a1"
        assert err.context["run_id"] == "r1"
        assert err.context["trace_id"] == "t1"
        assert isinstance(err.original_error, ValueError)


class TestAgentNotFoundError:
    def test_default_message(self):
        logger.info("AgentNotFoundError: default message")
        err = AgentNotFoundError(agent_name="missing-agent")
        assert "missing-agent" in str(err)
        assert isinstance(err, AgentError)

    def test_custom_message(self):
        logger.info("AgentNotFoundError: custom message")
        err = AgentNotFoundError(agent_name="a1", message="custom msg")
        assert "custom msg" in str(err)


class TestAgentConfigurationError:
    def test_basic(self):
        logger.info("AgentConfigurationError: basic")
        err = AgentConfigurationError("bad config")
        assert isinstance(err, AgentError)

    def test_with_config_key(self):
        logger.info("AgentConfigurationError: with config key")
        err = AgentConfigurationError("bad", config_key="model")
        assert err.config_key == "model"
        assert err.context["config_key"] == "model"


class TestAgentExecutionError:
    def test_basic(self):
        logger.info("AgentExecutionError: basic")
        err = AgentExecutionError("failed")
        assert isinstance(err, AgentError)

    def test_with_turn(self):
        logger.info("AgentExecutionError: with turn")
        err = AgentExecutionError("failed", turn=3)
        assert err.turn == 3
        assert err.context["turn"] == 3


class TestAgentTimeoutError:
    def test_basic(self):
        logger.info("AgentTimeoutError: basic")
        err = AgentTimeoutError("timed out")
        assert isinstance(err, AgentError)

    def test_with_timeout(self):
        logger.info("AgentTimeoutError: with timeout")
        err = AgentTimeoutError("timed out", timeout=30)
        assert err.timeout == 30
        assert err.context["timeout"] == 30


class TestMaxTurnsExceededError:
    def test_default_message(self):
        logger.info("MaxTurnsExceededError: default message")
        err = MaxTurnsExceededError(max_turns=10, current_turn=11)
        assert "10" in str(err)
        assert err.max_turns == 10
        assert err.current_turn == 11
        assert isinstance(err, AgentError)

    def test_custom_message(self):
        logger.info("MaxTurnsExceededError: custom message")
        err = MaxTurnsExceededError(message="custom", max_turns=5)
        assert "custom" in str(err)


class TestHandoffError:
    def test_basic(self):
        logger.info("HandoffError: basic")
        err = HandoffError("handoff failed")
        assert isinstance(err, AgentError)

    def test_with_agents(self):
        logger.info("HandoffError: with agents")
        err = HandoffError("fail", from_agent="a1", to_agent="a2", handoff_id="h1")
        assert err.from_agent == "a1"
        assert err.to_agent == "a2"
        assert err.handoff_id == "h1"
        assert err.context["from_agent"] == "a1"
        assert err.context["to_agent"] == "a2"
        assert err.context["handoff_id"] == "h1"


class TestHandoffNotAllowedError:
    def test_basic(self):
        logger.info("HandoffNotAllowedError: basic")
        err = HandoffNotAllowedError(from_agent="a1", to_agent="a2")
        assert isinstance(err, HandoffError)
        assert "a1" in str(err)
        assert "a2" in str(err)

    def test_with_reason(self):
        logger.info("HandoffNotAllowedError: with reason")
        err = HandoffNotAllowedError(from_agent="a1", to_agent="a2", reason="no permission")
        assert err.reason == "no permission"


class TestHandoffDepthExceededError:
    def test_basic(self):
        logger.info("HandoffDepthExceededError: basic")
        err = HandoffDepthExceededError(current_depth=6, max_depth=5)
        assert isinstance(err, HandoffError)
        assert err.current_depth == 6
        assert err.max_depth == 5
        assert err.context["current_depth"] == 6
        assert err.context["max_depth"] == 5


class TestHandoffTargetNotFoundError:
    def test_basic(self):
        logger.info("HandoffTargetNotFoundError: basic")
        err = HandoffTargetNotFoundError(from_agent="a1", to_agent="a2")
        assert isinstance(err, HandoffError)
        assert "a2" in str(err)
        assert "a1" in str(err)


class TestHandoffCycleDetectedError:
    def test_basic(self):
        logger.info("HandoffCycleDetectedError: basic")
        err = HandoffCycleDetectedError(
            from_agent="a1", to_agent="a2",
            agent_stack=["a0", "a1"],
        )
        assert isinstance(err, HandoffError)
        assert err.agent_stack == ["a0", "a1"]
        assert "cycle" in str(err).lower()


class TestAgentToolError:
    def test_basic(self):
        logger.info("AgentToolError: basic")
        err = AgentToolError("tool failed")
        assert isinstance(err, AgentError)

    def test_with_details(self):
        logger.info("AgentToolError: with details")
        err = AgentToolError("fail", tool_name="search", tool_args={"q": "test"})
        assert err.tool_name == "search"
        assert err.tool_args == {"q": "test"}
        assert err.context["tool_name"] == "search"


class TestWorkflowError:
    def test_basic(self):
        logger.info("WorkflowError: basic")
        err = WorkflowError("failed")
        assert isinstance(err, AgentError)

    def test_with_details(self):
        logger.info("WorkflowError: with details")
        err = WorkflowError("failed", workflow_type="sequential", step=2)
        assert err.workflow_type == "sequential"
        assert err.step == 2
        assert err.context["workflow_type"] == "sequential"
        assert err.context["step"] == 2


class TestSequentialWorkflowError:
    def test_basic(self):
        logger.info("SequentialWorkflowError: basic")
        err = SequentialWorkflowError("failed")
        assert isinstance(err, WorkflowError)
        assert err.workflow_type == "sequential"

    def test_with_failed_agent(self):
        logger.info("SequentialWorkflowError: with failed agent")
        err = SequentialWorkflowError("fail", failed_agent="a1")
        assert err.failed_agent == "a1"


class TestParallelWorkflowError:
    def test_basic(self):
        logger.info("ParallelWorkflowError: basic")
        err = ParallelWorkflowError("failed")
        assert isinstance(err, WorkflowError)
        assert err.workflow_type == "parallel"

    def test_with_failed_agents(self):
        logger.info("ParallelWorkflowError: with failed agents")
        err = ParallelWorkflowError("fail", failed_agents=["a1", "a2"])
        assert err.failed_agents == ["a1", "a2"]


class TestLoopWorkflowError:
    def test_basic(self):
        logger.info("LoopWorkflowError: basic")
        err = LoopWorkflowError("failed")
        assert isinstance(err, WorkflowError)
        assert err.workflow_type == "loop"

    def test_with_iteration(self):
        logger.info("LoopWorkflowError: with iteration")
        err = LoopWorkflowError("fail", iteration=3)
        assert err.iteration == 3


class TestLoopMaxIterationsError:
    def test_basic(self):
        logger.info("LoopMaxIterationsError: basic")
        err = LoopMaxIterationsError(max_iterations=100)
        assert isinstance(err, LoopWorkflowError)
        assert err.max_iterations == 100
        assert "100" in str(err)


class TestRunStateError:
    def test_basic(self):
        logger.info("RunStateError: basic")
        err = RunStateError("state error")
        assert isinstance(err, AgentError)


class TestRunStateNotFoundError:
    def test_basic(self):
        logger.info("RunStateNotFoundError: basic")
        err = RunStateNotFoundError(run_id="r1")
        assert isinstance(err, RunStateError)
        assert "r1" in str(err)


class TestRunStatePersistenceError:
    def test_basic(self):
        logger.info("RunStatePersistenceError: basic")
        err = RunStatePersistenceError("persist failed")
        assert isinstance(err, RunStateError)


class TestRouterError:
    def test_basic(self):
        logger.info("RouterError: basic")
        err = RouterError("routing failed")
        assert isinstance(err, AgentError)

    def test_with_input(self):
        logger.info("RouterError: with input")
        err = RouterError("fail", input_text="some user input")
        assert err.context["input_preview"] == "some user input"

    def test_long_input_truncated(self):
        logger.info("RouterError: long input truncated")
        long_input = "x" * 300
        err = RouterError("fail", input_text=long_input)
        assert len(err.context["input_preview"]) < 300
        assert err.context["input_preview"].endswith("...")


class TestNoRouteFoundError:
    def test_basic(self):
        logger.info("NoRouteFoundError: basic")
        err = NoRouteFoundError()
        assert isinstance(err, RouterError)

    def test_with_routes(self):
        logger.info("NoRouteFoundError: with routes")
        err = NoRouteFoundError(available_routes=["r1", "r2"])
        assert err.available_routes == ["r1", "r2"]
        assert err.context["available_routes"] == ["r1", "r2"]

    def test_custom_message(self):
        logger.info("NoRouteFoundError: custom message")
        err = NoRouteFoundError(message="no match")
        assert "no match" in str(err)
