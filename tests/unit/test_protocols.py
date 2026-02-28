"""Unit tests for SDK-level protocols and Container protocol acceptance."""

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from orchestrator.protocols import ILLMClient, IMemoryClient, ISessionClient
import logging

logger = logging.getLogger(__name__)


class FakeLLMClient:
    """Custom LLM client implementing ILLMClient protocol."""

    async def chat(self, messages: list, **kwargs: Any) -> Any:
        return {"content": "fake response"}

    async def chat_stream(self, messages: list, **kwargs: Any) -> AsyncIterator:
        yield {"content": "chunk"}

    def count_tokens(self, messages: list, model: str | None = None) -> int:
        return 42


class FakeMemoryClient:
    """Custom memory client implementing IMemoryClient protocol."""

    @property
    def is_enabled(self) -> bool:
        return True

    async def search(self, query: str, **kwargs: Any) -> Any:
        return []

    async def add(self, messages: Any, **kwargs: Any) -> Any:
        return None


class FakeSessionClient:
    """Custom session client implementing ISessionClient protocol."""

    @property
    def is_enabled(self) -> bool:
        return True

    async def get_conversation_history(self, session_id: str, **kwargs: Any) -> list:
        return []

    async def add_message(self, session_id: str, message: Any, **kwargs: Any) -> None:
        pass


class TestProtocolImplementation:
    def test_llm_client_implements_protocol(self):
        logger.info("ProtocolImplementation: llm client implements protocol")
        from orchestrator.llm import LLMClient

        assert hasattr(LLMClient, "chat")
        assert hasattr(LLMClient, "chat_stream")
        assert hasattr(LLMClient, "count_tokens")

    def test_memory_client_implements_protocol(self):
        logger.info("ProtocolImplementation: memory client implements protocol")
        from orchestrator.memory import MemoryClient

        assert hasattr(MemoryClient, "is_enabled")
        assert hasattr(MemoryClient, "search")
        assert hasattr(MemoryClient, "add")

    def test_session_client_implements_protocol(self):
        logger.info("ProtocolImplementation: session client implements protocol")
        from orchestrator.session import SessionClient

        assert hasattr(SessionClient, "is_enabled")
        assert hasattr(SessionClient, "get_conversation_history")
        assert hasattr(SessionClient, "add_message")

    def test_custom_llm_client_via_protocol(self):
        logger.info("ProtocolImplementation: custom llm client via protocol")
        client = FakeLLMClient()
        assert isinstance(client, ILLMClient)

    def test_custom_memory_client_via_protocol(self):
        logger.info("ProtocolImplementation: custom memory client via protocol")
        client = FakeMemoryClient()
        assert isinstance(client, IMemoryClient)

    def test_custom_session_client_via_protocol(self):
        logger.info("ProtocolImplementation: custom session client via protocol")
        client = FakeSessionClient()
        assert isinstance(client, ISessionClient)


class TestContainerAcceptsProtocols:
    def test_container_accepts_protocol_implementations(self):
        logger.info("ContainerAcceptsProtocols: container accepts protocol implementations")
        with patch("orchestrator.core.container.settings") as mock_settings:
            mock_settings.default_model = "gpt-4"
            mock_settings.enable_memory = False
            mock_settings.enable_session = False
            mock_settings.enable_tracing = False
            mock_settings.shared_services_enabled = False

            from orchestrator.core.container import Container, ContainerConfig

            container = Container(config=ContainerConfig(auto_initialize=False))

            fake_llm = FakeLLMClient()
            container.set_llm_client(fake_llm)
            assert container._llm_client is fake_llm
            assert container._llm_initialized is True

            fake_mem = FakeMemoryClient()
            container.set_memory_client(fake_mem)
            assert container._memory_client is fake_mem

            fake_sess = FakeSessionClient()
            container.set_session_client(fake_sess)
            assert container._session_client is fake_sess
