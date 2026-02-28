"""Tests for agent/execution/ modules."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.agent.types import (
    TokenUsage,
    ToolExecutionSummary,
)
import logging

logger = logging.getLogger(__name__)


class TestExecutor:
    def test_init_no_args(self):
        logger.info("Executor: init no args")
        from orchestrator.agent.execution.executor import Executor
        ex = Executor()
        assert ex._llm_client is None
        assert ex._tool_handler is None
        assert ex._handoff_executor is None

    def test_init_with_clients(self):
        logger.info("Executor: init with clients")
        from orchestrator.agent.execution.executor import Executor
        mock_llm = MagicMock()
        mock_tool = MagicMock()
        mock_handoff = MagicMock()
        ex = Executor(llm_client=mock_llm, tool_handler=mock_tool, handoff_executor=mock_handoff)
        assert ex._llm_client is mock_llm
        assert ex._tool_handler is mock_tool
        assert ex._handoff_executor is mock_handoff


class TestToolHandler:
    def test_init_no_args(self):
        logger.info("ToolHandler: init no args")
        from orchestrator.agent.execution.tool_handler import ToolHandler
        th = ToolHandler()
        assert th._tool_service is None

    def test_init_with_service(self):
        logger.info("ToolHandler: init with service")
        from orchestrator.agent.execution.tool_handler import ToolHandler
        mock_service = MagicMock()
        th = ToolHandler(tool_service=mock_service)
        assert th._tool_service is mock_service


class TestHandoffExecutor:
    def test_init_no_args(self):
        logger.info("HandoffExecutor: init no args")
        from orchestrator.agent.execution.handoff_executor import HandoffExecutor
        he = HandoffExecutor()
        assert he._handoff_manager is None
        assert he._agent_registry is None or he._agent_registry == {}

    def test_init_with_args(self):
        logger.info("HandoffExecutor: init with args")
        from orchestrator.agent.execution.handoff_executor import HandoffExecutor
        mock_manager = MagicMock()
        mock_registry = {"a1": MagicMock()}
        he = HandoffExecutor(handoff_manager=mock_manager, agent_registry=mock_registry)
        assert he._handoff_manager is mock_manager


class TestMessageBuilder:
    def test_init_no_args(self):
        logger.info("MessageBuilder: init no args")
        from orchestrator.agent.execution.message_builder import MessageBuilder
        mb = MessageBuilder()
        assert mb._memory_service is None
        assert mb._session_service is None

    def test_init_with_services(self):
        logger.info("MessageBuilder: init with services")
        from orchestrator.agent.execution.message_builder import MessageBuilder
        mock_mem = MagicMock()
        mock_sess = MagicMock()
        mb = MessageBuilder(memory_service=mock_mem, session_service=mock_sess)
        assert mb._memory_service is mock_mem
        assert mb._session_service is mock_sess


class TestRunFinalizer:
    def test_init(self):
        logger.info("RunFinalizer: init")
        from orchestrator.agent.execution.run_finalizer import RunFinalizer
        mock_sess = MagicMock()
        mock_ctx = MagicMock()
        mock_lc = MagicMock()
        rf = RunFinalizer(
            session_service=mock_sess,
            context_service=mock_ctx,
            lifecycle=mock_lc,
        )
        assert rf._session_service is mock_sess
        assert rf._context_service is mock_ctx
        assert rf._lifecycle is mock_lc


class TestRunLifecycle:
    def test_class_exists(self):
        logger.info("RunLifecycle: class exists")
        from orchestrator.agent.execution.run_lifecycle import RunLifecycle
        rl = RunLifecycle()
        assert rl is not None


class TestStreamExecutor:
    def test_init_no_args(self):
        logger.info("StreamExecutor: init no args")
        from orchestrator.agent.execution.stream_executor import StreamExecutor
        se = StreamExecutor()
        assert se._llm_client is None

    def test_init_with_client(self):
        logger.info("StreamExecutor: init with client")
        from orchestrator.agent.execution.stream_executor import StreamExecutor

        mock_llm = MagicMock()
        se = StreamExecutor(llm_client=mock_llm)
        assert se._llm_client is mock_llm


class TestTokenUsage:
    def test_defaults(self):
        logger.info("TokenUsage: defaults")
        tu = TokenUsage()
        assert tu.prompt_tokens == 0
        assert tu.completion_tokens == 0
        assert tu.total_tokens == 0

    def test_with_values(self):
        logger.info("TokenUsage: with values")
        tu = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        assert tu.total_tokens == 150


class TestToolExecutionSummary:
    def test_defaults(self):
        logger.info("ToolExecutionSummary: defaults")
        s = ToolExecutionSummary()
        assert s.tool_count == 0
        assert s.success_count == 0
        assert s.error_count == 0
