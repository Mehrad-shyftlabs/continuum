"""Tests for WorkerManager."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from orchestrator.temporal.worker import (
    WorkerManager,
    get_worker_manager,
    reset_worker_manager,
)
from orchestrator.temporal.config import TemporalConfig
import logging

logger = logging.getLogger(__name__)


class TestWorkerManager:
    def setup_method(self):
        self.mock_client = MagicMock()
        self.mock_client.raw_client = MagicMock()
        self.mock_registry = MagicMock()
        self.config = TemporalConfig(task_queue="test-queue")
        self.manager = WorkerManager(
            client=self.mock_client,
            registry=self.mock_registry,
            config=self.config,
        )

    def test_not_running_initially(self):
        logger.info("WorkerManager: not running initially")
        assert self.manager.is_running is False

    def test_register_workflow(self):
        logger.info("WorkerManager: register workflow")
        class MyWorkflow:
            pass

        self.manager.register_workflow(MyWorkflow)
        assert MyWorkflow in self.manager._custom_workflows

    def test_register_activity(self):
        logger.info("WorkerManager: register activity")
        async def my_activity():
            pass

        self.manager.register_activity(my_activity)
        assert my_activity in self.manager._custom_activities

    @pytest.mark.asyncio
    async def test_start(self):
        logger.info("WorkerManager: start")
        with patch("orchestrator.temporal.worker.Worker") as MockWorker:
            mock_worker_instance = MagicMock()
            mock_worker_instance.run = AsyncMock()
            MockWorker.return_value = mock_worker_instance

            await self.manager.start()
            assert self.manager.is_running is True
            MockWorker.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_custom_queue(self):
        logger.info("WorkerManager: start custom queue")
        with patch("orchestrator.temporal.worker.Worker") as MockWorker:
            mock_worker_instance = MagicMock()
            mock_worker_instance.run = AsyncMock()
            MockWorker.return_value = mock_worker_instance

            await self.manager.start(task_queue="custom-queue")
            call_kwargs = MockWorker.call_args
            assert call_kwargs.kwargs["task_queue"] == "custom-queue"

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        logger.info("WorkerManager: start idempotent")
        with patch("orchestrator.temporal.worker.Worker") as MockWorker:
            mock_worker_instance = MagicMock()
            mock_worker_instance.run = AsyncMock()
            MockWorker.return_value = mock_worker_instance

            await self.manager.start()
            await self.manager.start()  # Should not start a second worker
            assert MockWorker.call_count == 1

    @pytest.mark.asyncio
    async def test_stop(self):
        logger.info("WorkerManager: stop")
        with patch("orchestrator.temporal.worker.Worker") as MockWorker:
            mock_worker_instance = MagicMock()
            mock_worker_instance.run = AsyncMock()
            mock_worker_instance.shutdown = MagicMock()
            MockWorker.return_value = mock_worker_instance

            await self.manager.start()
            await self.manager.stop()
            assert self.manager.is_running is False
            mock_worker_instance.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self):
        logger.info("WorkerManager: stop when not running")
        await self.manager.stop()  # Should not raise

    @pytest.mark.asyncio
    async def test_includes_builtin_workflows(self):
        logger.info("WorkerManager: includes builtin workflows")
        with patch("orchestrator.temporal.worker.Worker") as MockWorker:
            mock_worker_instance = MagicMock()
            mock_worker_instance.run = AsyncMock()
            MockWorker.return_value = mock_worker_instance

            await self.manager.start()
            call_kwargs = MockWorker.call_args
            workflows = call_kwargs.kwargs["workflows"]
            assert len(workflows) >= 4  # 4 built-in workflows

    @pytest.mark.asyncio
    async def test_includes_builtin_activities(self):
        logger.info("WorkerManager: includes builtin activities")
        with patch("orchestrator.temporal.worker.Worker") as MockWorker:
            mock_worker_instance = MagicMock()
            mock_worker_instance.run = AsyncMock()
            MockWorker.return_value = mock_worker_instance

            await self.manager.start()
            call_kwargs = MockWorker.call_args
            activities = call_kwargs.kwargs["activities"]
            assert len(activities) >= 2  # run_agent + send_notification

    @pytest.mark.asyncio
    async def test_includes_custom_workflows_and_activities(self):
        logger.info("WorkerManager: includes custom workflows and activities")
        class CustomWF:
            pass

        async def custom_act():
            pass

        self.manager.register_workflow(CustomWF)
        self.manager.register_activity(custom_act)

        with patch("orchestrator.temporal.worker.Worker") as MockWorker:
            mock_worker_instance = MagicMock()
            mock_worker_instance.run = AsyncMock()
            MockWorker.return_value = mock_worker_instance

            await self.manager.start()
            call_kwargs = MockWorker.call_args
            workflows = call_kwargs.kwargs["workflows"]
            activities = call_kwargs.kwargs["activities"]
            assert CustomWF in workflows
            assert custom_act in activities


class TestGlobalWorkerManager:
    def setup_method(self):
        reset_worker_manager()

    def teardown_method(self):
        reset_worker_manager()

    def test_singleton(self):
        logger.info("GlobalWorkerManager: singleton")
        m1 = get_worker_manager()
        m2 = get_worker_manager()
        assert m1 is m2

    def test_reset(self):
        logger.info("GlobalWorkerManager: reset")
        m1 = get_worker_manager()
        reset_worker_manager()
        m2 = get_worker_manager()
        assert m1 is not m2
