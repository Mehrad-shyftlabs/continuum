"""Tests for TemporalClient."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from orchestrator.temporal.client import (
    TemporalClient,
    get_temporal_client,
    reset_temporal_client,
)
from orchestrator.temporal.config import TemporalConfig
from orchestrator.temporal.exceptions import TemporalConnectionError
import logging

logger = logging.getLogger(__name__)


class TestTemporalClient:
    def test_init_default_config(self):
        logger.info("TemporalClient: init default config")
        client = TemporalClient()
        assert client._config is not None
        assert client.is_connected is False

    def test_init_custom_config(self):
        logger.info("TemporalClient: init custom config")
        config = TemporalConfig(host="custom:7233", namespace="prod")
        client = TemporalClient(config=config)
        assert client._config.host == "custom:7233"
        assert client._config.namespace == "prod"

    def test_is_connected_false_initially(self):
        logger.info("TemporalClient: is connected false initially")
        client = TemporalClient()
        assert client.is_connected is False

    def test_raw_client_raises_when_not_connected(self):
        logger.info("TemporalClient: raw client raises when not connected")
        client = TemporalClient()
        with pytest.raises(TemporalConnectionError):
            _ = client.raw_client

    @pytest.mark.asyncio
    async def test_connect_success(self):
        logger.info("TemporalClient: connect success")
        client = TemporalClient()
        mock_temporal_client = MagicMock()

        with patch(
            "orchestrator.temporal.client.Client.connect",
            new_callable=AsyncMock,
            return_value=mock_temporal_client,
        ):
            await client.connect()
            assert client.is_connected is True
            assert client.raw_client is mock_temporal_client

    @pytest.mark.asyncio
    async def test_connect_failure(self):
        logger.info("TemporalClient: connect failure")
        client = TemporalClient()
        with patch(
            "orchestrator.temporal.client.Client.connect",
            new_callable=AsyncMock,
            side_effect=Exception("Connection refused"),
        ):
            with pytest.raises(TemporalConnectionError):
                await client.connect()

    @pytest.mark.asyncio
    async def test_disconnect(self):
        logger.info("TemporalClient: disconnect")
        client = TemporalClient()
        client._client = MagicMock()
        assert client.is_connected is True
        await client.disconnect()
        assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_start_workflow(self):
        logger.info("TemporalClient: start workflow")
        client = TemporalClient()
        mock_handle = MagicMock()
        mock_raw = MagicMock()
        mock_raw.start_workflow = AsyncMock(return_value=mock_handle)
        client._client = mock_raw

        handle = await client.start_workflow(
            "test_workflow",
            {"input": "test"},
            id="wf-1",
            task_queue="test-queue",
        )
        assert handle is mock_handle
        mock_raw.start_workflow.assert_called_once()

    @pytest.mark.asyncio
    async def test_signal_workflow(self):
        logger.info("TemporalClient: signal workflow")
        client = TemporalClient()
        mock_handle = MagicMock()
        mock_handle.signal = AsyncMock()
        mock_raw = MagicMock()
        mock_raw.get_workflow_handle.return_value = mock_handle
        client._client = mock_raw

        await client.signal_workflow("wf-1", "my_signal", {"data": "test"})
        mock_handle.signal.assert_called_once_with("my_signal", {"data": "test"})

    @pytest.mark.asyncio
    async def test_query_workflow(self):
        logger.info("TemporalClient: query workflow")
        client = TemporalClient()
        mock_handle = MagicMock()
        mock_handle.query = AsyncMock(return_value={"status": "running"})
        mock_raw = MagicMock()
        mock_raw.get_workflow_handle.return_value = mock_handle
        client._client = mock_raw

        result = await client.query_workflow("wf-1", "get_status")
        assert result == {"status": "running"}

    @pytest.mark.asyncio
    async def test_cancel_workflow(self):
        logger.info("TemporalClient: cancel workflow")
        client = TemporalClient()
        mock_handle = MagicMock()
        mock_handle.cancel = AsyncMock()
        mock_raw = MagicMock()
        mock_raw.get_workflow_handle.return_value = mock_handle
        client._client = mock_raw

        await client.cancel_workflow("wf-1")
        mock_handle.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_workflow_handle(self):
        logger.info("TemporalClient: get workflow handle")
        client = TemporalClient()
        mock_handle = MagicMock()
        mock_raw = MagicMock()
        mock_raw.get_workflow_handle.return_value = mock_handle
        client._client = mock_raw

        handle = await client.get_workflow_handle("wf-1")
        assert handle is mock_handle

    @pytest.mark.asyncio
    async def test_get_workflow_result(self):
        logger.info("TemporalClient: get workflow result")
        client = TemporalClient()
        mock_handle = MagicMock()
        mock_handle.result = AsyncMock(return_value="done")
        mock_raw = MagicMock()
        mock_raw.get_workflow_handle.return_value = mock_handle
        client._client = mock_raw

        result = await client.get_workflow_result("wf-1")
        assert result == "done"


class TestGlobalClient:
    def setup_method(self):
        reset_temporal_client()

    def teardown_method(self):
        reset_temporal_client()

    def test_singleton(self):
        logger.info("GlobalClient: singleton")
        c1 = get_temporal_client()
        c2 = get_temporal_client()
        assert c1 is c2

    def test_reset(self):
        logger.info("GlobalClient: reset")
        c1 = get_temporal_client()
        reset_temporal_client()
        c2 = get_temporal_client()
        assert c1 is not c2
