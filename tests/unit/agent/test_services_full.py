"""Comprehensive tests for agent/services/ modules."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import logging

logger = logging.getLogger(__name__)


class TestContextServiceInit:
    def test_init_no_args(self):
        logger.info("ContextServiceInit: init no args")
        from orchestrator.agent.services.context_service import ContextService
        cs = ContextService()
        assert cs._state_manager is None
        assert cs._config is None

    def test_init_with_args(self):
        logger.info("ContextServiceInit: init with args")
        from orchestrator.agent.services.context_service import ContextService
        mock_state = MagicMock()
        mock_config = MagicMock()
        cs = ContextService(state_manager=mock_state, config=mock_config)
        assert cs._state_manager is mock_state
        assert cs._config is mock_config


class TestContextServiceCreateRunState:
    @pytest.mark.asyncio
    async def test_create_run_state(self):
        logger.info("ContextServiceCreateRunState: create run state")
        from orchestrator.agent.services.context_service import ContextService
        from orchestrator.agent.types import RunContext, RunState, RunStatus

        mock_state_mgr = MagicMock()
        mock_state_mgr.save = AsyncMock()
        mock_config = MagicMock()
        mock_config.persist_state = False

        cs = ContextService(state_manager=mock_state_mgr, config=mock_config)

        agent = MagicMock()
        agent.name = "test-agent"
        ctx = RunContext(run_id="r1", user_id="u1")

        state = await cs.create_run_state(agent, ctx)
        assert isinstance(state, RunState)
        assert state.run_id == "r1"
        assert state.current_agent == "test-agent"
        assert state.status == RunStatus.RUNNING

    @pytest.mark.asyncio
    async def test_create_run_state_persists(self):
        logger.info("ContextServiceCreateRunState: create run state persists")
        from orchestrator.agent.services.context_service import ContextService
        from orchestrator.agent.types import RunContext

        mock_state_mgr = MagicMock()
        mock_state_mgr.save = AsyncMock()
        mock_config = MagicMock()
        mock_config.persist_state = True

        cs = ContextService(state_manager=mock_state_mgr, config=mock_config)

        agent = MagicMock()
        agent.name = "agent1"
        ctx = RunContext(run_id="r1")

        await cs.create_run_state(agent, ctx)
        mock_state_mgr.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_run_state(self):
        logger.info("ContextServiceCreateRunState: save run state")
        from orchestrator.agent.services.context_service import ContextService
        from orchestrator.agent.types import RunState

        mock_state_mgr = MagicMock()
        mock_state_mgr.save = AsyncMock()
        mock_config = MagicMock()
        mock_config.persist_state = True

        cs = ContextService(state_manager=mock_state_mgr, config=mock_config)
        state = RunState(run_id="r1")
        await cs.save_run_state(state)
        mock_state_mgr.save.assert_called_once_with(state)

    @pytest.mark.asyncio
    async def test_load_run_state(self):
        logger.info("ContextServiceCreateRunState: load run state")
        from orchestrator.agent.services.context_service import ContextService
        from orchestrator.agent.types import RunState

        mock_state_mgr = MagicMock()
        mock_state = RunState(run_id="r1")
        mock_state_mgr.load = AsyncMock(return_value=mock_state)

        cs = ContextService(state_manager=mock_state_mgr)
        result = await cs.load_run_state("r1")
        assert result is mock_state


class TestMemoryServiceInit:
    def test_init_no_args(self):
        logger.info("MemoryServiceInit: init no args")
        from orchestrator.agent.services.memory_service import MemoryService
        ms = MemoryService()
        assert ms._memory_client is None
        assert ms._session_client is None

    def test_memory_client_property(self):
        logger.info("MemoryServiceInit: memory client property")
        from orchestrator.agent.services.memory_service import MemoryService
        mock_client = MagicMock()
        ms = MemoryService(memory_client=mock_client)
        assert ms.memory_client is mock_client


class TestMemoryServiceRetrieve:
    @pytest.mark.asyncio
    async def test_skip_when_no_client(self):
        logger.info("MemoryServiceRetrieve: skip when no client")
        from orchestrator.agent.services.memory_service import MemoryService
        from orchestrator.agent.types import RunContext

        ms = MemoryService(memory_client=None)
        agent = MagicMock()
        agent.memory_config.search_memories = True
        ctx = RunContext(run_id="r1")

        result = await ms.retrieve_memories(agent, "query", ctx)
        assert result == []

    @pytest.mark.asyncio
    async def test_skip_when_search_disabled(self):
        logger.info("MemoryServiceRetrieve: skip when search disabled")
        from orchestrator.agent.services.memory_service import MemoryService
        from orchestrator.agent.types import RunContext

        mock_client = MagicMock()
        ms = MemoryService(memory_client=mock_client)
        agent = MagicMock()
        agent.memory_config.search_memories = False
        ctx = RunContext(run_id="r1")

        result = await ms.retrieve_memories(agent, "query", ctx)
        assert result == []


class TestSessionServiceInit:
    def test_init_no_args(self):
        logger.info("SessionServiceInit: init no args")
        from orchestrator.agent.services.session_service import SessionService
        ss = SessionService()
        assert ss._session_client is None

    def test_session_client_property(self):
        logger.info("SessionServiceInit: session client property")
        from orchestrator.agent.services.session_service import SessionService
        mock_client = MagicMock()
        ss = SessionService(session_client=mock_client)
        assert ss.session_client is mock_client


class TestSessionServiceSaveMessages:
    @pytest.mark.asyncio
    async def test_skip_when_no_client(self):
        logger.info("SessionServiceSaveMessages: skip when no client")
        from orchestrator.agent.services.session_service import SessionService

        ss = SessionService(session_client=None)
        agent = MagicMock()
        await ss.save_messages(agent, [], 0, "s1")

    @pytest.mark.asyncio
    async def test_skip_when_disabled(self):
        logger.info("SessionServiceSaveMessages: skip when disabled")
        from orchestrator.agent.services.session_service import SessionService

        mock_client = MagicMock()
        mock_client.is_enabled = False
        ss = SessionService(session_client=mock_client)
        agent = MagicMock()
        await ss.save_messages(agent, [], 0, "s1")


class TestToolServiceInit:
    def test_init_no_args(self):
        logger.info("ToolServiceInit: init no args")
        from orchestrator.agent.services.tool_service import ToolService
        ts = ToolService()
        assert ts._tool_executor is None

    def test_init_with_executor(self):
        logger.info("ToolServiceInit: init with executor")
        from orchestrator.agent.services.tool_service import ToolService
        mock_executor = MagicMock()
        ts = ToolService(tool_executor=mock_executor)
        assert ts._tool_executor is mock_executor


class TestToolHandlerDeep:
    @pytest.mark.asyncio
    async def test_execute_tool_call_no_service(self):
        logger.info("ToolHandlerDeep: execute tool call no service")
        from orchestrator.agent.execution.tool_handler import ToolHandler
        th = ToolHandler()
        with pytest.raises(RuntimeError):
            await th.execute_tool_call(MagicMock(), MagicMock(), MagicMock())

    @pytest.mark.asyncio
    async def test_execute_tools_batch_no_service(self):
        logger.info("ToolHandlerDeep: execute tools batch no service")
        from orchestrator.agent.execution.tool_handler import ToolHandler
        th = ToolHandler()
        with pytest.raises(RuntimeError):
            await th.execute_tools_batch(MagicMock(), [], MagicMock())

    @pytest.mark.asyncio
    async def test_execute_tool_call_with_service(self):
        logger.info("ToolHandlerDeep: execute tool call with service")
        from orchestrator.agent.execution.tool_handler import ToolHandler

        mock_service = MagicMock()
        mock_service.execute_tool_call = AsyncMock(return_value=({"result": "ok"}, {}))
        th = ToolHandler(tool_service=mock_service)

        result = await th.execute_tool_call(MagicMock(), MagicMock(), MagicMock())
        mock_service.execute_tool_call.assert_called_once()
