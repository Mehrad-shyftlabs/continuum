"""Unit tests for thread safety and global state management."""

import threading
from unittest.mock import patch

import pytest

from orchestrator.core.container import Container, ContainerConfig, get_container, reset_container
import logging

logger = logging.getLogger(__name__)


class TestContainerConcurrentAccess:
    def test_container_concurrent_access(self):
        """Multiple threads getting the container should not raise."""
        logger.info("Multiple threads getting the container should not raise")
        reset_container()
        results = []
        errors = []

        def get():
            try:
                c = get_container()
                results.append(c)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # All threads should get the same container instance
        assert all(r is results[0] for r in results)
        reset_container()


class TestContainerResetDuringAccess:
    def test_container_reset_during_access(self):
        """Resetting while other threads access should not crash."""
        logger.info("Resetting while other threads access should not crash")
        reset_container()
        errors = []

        def access_loop():
            for _ in range(20):
                try:
                    _ = get_container()
                except Exception as e:
                    errors.append(e)

        def reset_loop():
            for _ in range(5):
                try:
                    reset_container()
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=access_loop),
            threading.Thread(target=access_loop),
            threading.Thread(target=reset_loop),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        reset_container()


class TestGlobalStateIsolation:
    def test_global_state_isolation_between_tests(self):
        """reset_container should fully clean up all global state."""
        logger.info("reset_container should fully clean up all global state")
        reset_container()

        c1 = get_container()
        assert c1 is not None

        reset_container()

        c2 = get_container()
        assert c2 is not None
        assert c2 is not c1

        reset_container()

    def test_reset_container_clears_memory_globals(self):
        """reset_container should also reset memory and session module globals."""
        logger.info("reset_container should also reset memory and session module globals")
        from orchestrator.memory.client import _global_memory_client, _initialized

        reset_container()

        from orchestrator.memory import client as mem_mod

        assert mem_mod._global_memory_client is None
        assert mem_mod._initialized is False

    def test_reset_container_clears_session_globals(self):
        """reset_container should also reset session module globals."""
        logger.info("reset_container should also reset session module globals")
        reset_container()

        from orchestrator.session import client as sess_mod

        assert sess_mod._global_session_client is None
        assert sess_mod._initialized is False
