"""Comprehensive tests for core/container.py."""

from unittest.mock import MagicMock, patch

import pytest

from orchestrator.core.container import Container, ContainerConfig, get_container
import logging

logger = logging.getLogger(__name__)


class TestContainerConfig:
    def test_defaults(self):
        logger.info("ContainerConfig: defaults")
        config = ContainerConfig()
        assert config.auto_initialize is True
        assert config.llm_config is None

    def test_custom(self):
        logger.info("ContainerConfig: custom")
        config = ContainerConfig(
            auto_initialize=False,
            enable_memory=False,
            enable_session=False,
        )
        assert config.auto_initialize is False
        assert config.enable_memory is False
        assert config.enable_session is False


class TestContainer:
    def test_init(self):
        logger.info("Container: init")
        config = ContainerConfig(auto_initialize=False)
        c = Container(config=config)
        assert c._config.auto_initialize is False

    def test_set_llm_client(self):
        logger.info("Container: set llm client")
        config = ContainerConfig(auto_initialize=False)
        c = Container(config=config)
        mock = MagicMock()
        c.set_llm_client(mock)
        assert c.llm_client is mock
        assert c.has_llm_client() is True

    def test_llm_client_not_initialized(self):
        logger.info("Container: llm client not initialized")
        config = ContainerConfig(auto_initialize=False)
        c = Container(config=config)
        with pytest.raises(RuntimeError):
            _ = c.llm_client

    def test_set_memory_client(self):
        logger.info("Container: set memory client")
        config = ContainerConfig(auto_initialize=False, enable_memory=True)
        c = Container(config=config)
        mock = MagicMock()
        mock.is_enabled = True
        c.set_memory_client(mock)
        assert c.memory_client is mock

    def test_memory_client_none_when_not_set(self):
        logger.info("Container: memory client none when not set")
        config = ContainerConfig(auto_initialize=False, enable_memory=False)
        c = Container(config=config)
        assert c.memory_client is None

    def test_set_session_client(self):
        logger.info("Container: set session client")
        config = ContainerConfig(auto_initialize=False)
        c = Container(config=config)
        mock = MagicMock()
        c.set_session_client(mock)
        assert c.session_client is mock
        assert c.has_session_client() is True

    def test_session_client_none_when_not_set(self):
        logger.info("Container: session client none when not set")
        config = ContainerConfig(auto_initialize=False, enable_session=False)
        c = Container(config=config)
        assert c.session_client is None

    def test_set_tool_executor(self):
        logger.info("Container: set tool executor")
        config = ContainerConfig(auto_initialize=False)
        c = Container(config=config)
        mock = MagicMock()
        c.set_tool_executor(mock)
        assert c.tool_executor is mock

    def test_has_llm_client_false(self):
        logger.info("Container: has llm client false")
        config = ContainerConfig(auto_initialize=False)
        c = Container(config=config)
        assert c.has_llm_client() is False

    def test_has_memory_client_false(self):
        logger.info("Container: has memory client false")
        config = ContainerConfig(auto_initialize=False)
        c = Container(config=config)
        assert c.has_memory_client() is False

    def test_has_session_client_false(self):
        logger.info("Container: has session client false")
        config = ContainerConfig(auto_initialize=False)
        c = Container(config=config)
        assert c.has_session_client() is False


class TestGetContainer:
    def test_returns_instance(self):
        logger.info("GetContainer: returns instance")
        c = get_container()
        assert isinstance(c, Container)

    def test_singleton(self):
        logger.info("GetContainer: singleton")
        c1 = get_container()
        c2 = get_container()
        assert c1 is c2
