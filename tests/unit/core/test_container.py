"""Unit tests for core container."""

from unittest.mock import MagicMock, patch

import pytest

from orchestrator.core.container import Container, ContainerConfig, get_container, reset_container
import logging

logger = logging.getLogger(__name__)


class TestContainerConfig:
    def test_defaults(self):
        logger.info("ContainerConfig: defaults")
        c = ContainerConfig(auto_initialize=False)
        assert c.auto_initialize is False


class TestContainer:
    def test_container_init(self):
        logger.info("Container: container init")
        c = Container(config=ContainerConfig(auto_initialize=False))
        assert c._llm_initialized is False

    def test_container_set_llm_client(self):
        logger.info("Container: container set llm client")
        c = Container(config=ContainerConfig(auto_initialize=False))
        mock = MagicMock()
        c.set_llm_client(mock)
        assert c._llm_client is mock
        assert c._llm_initialized is True

    def test_container_set_memory_client(self):
        logger.info("Container: container set memory client")
        c = Container(config=ContainerConfig(auto_initialize=False))
        mock = MagicMock()
        c.set_memory_client(mock)
        assert c._memory_client is mock

    def test_container_set_session_client(self):
        logger.info("Container: container set session client")
        c = Container(config=ContainerConfig(auto_initialize=False))
        mock = MagicMock()
        c.set_session_client(mock)
        assert c._session_client is mock

    def test_container_llm_not_initialized_raises(self):
        logger.info("Container: container llm not initialized raises")
        c = Container(config=ContainerConfig(auto_initialize=False))
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = c.llm_client

    def test_container_has_memory_client(self):
        logger.info("Container: container has memory client")
        c = Container(config=ContainerConfig(auto_initialize=False, enable_memory=False))
        assert c.has_memory_client() is False

    def test_container_has_session_client(self):
        logger.info("Container: container has session client")
        c = Container(config=ContainerConfig(auto_initialize=False, enable_session=False))
        assert c.has_session_client() is False

    def test_container_reset(self):
        logger.info("Container: container reset")
        c = Container(config=ContainerConfig(auto_initialize=False))
        c.set_llm_client(MagicMock())
        c.reset()
        assert c._llm_initialized is False

    def test_container_set_tool_executor(self):
        logger.info("Container: container set tool executor")
        c = Container(config=ContainerConfig(auto_initialize=False))
        mock = MagicMock()
        c.set_tool_executor(mock)
        assert c._tool_executor is mock


class TestGlobalContainer:
    def test_get_container_singleton(self):
        logger.info("GlobalContainer: get container singleton")
        c1 = get_container()
        c2 = get_container()
        assert c1 is c2

    def test_reset_container(self):
        logger.info("GlobalContainer: reset container")
        _ = get_container()
        reset_container()
