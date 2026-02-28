"""Unit tests for session module cleanup (Issue 20)."""

import importlib

import pytest
import logging

logger = logging.getLogger(__name__)


class TestSessionModule:
    def test_session_client_has_all_public_methods(self):
        """SessionClient should have the core public API methods."""
        logger.info("SessionClient should have the core public API methods")
        from orchestrator.session import SessionClient

        required_methods = [
            "get_or_create_session",
            "add_message",
            "get_conversation_history",
            "get_session_metadata",
            "clear_session",
            "delete_session",
        ]
        for method_name in required_methods:
            assert hasattr(SessionClient, method_name), f"SessionClient missing {method_name}"

    def test_no_import_of_removed_manager(self):
        """session/manager.py should no longer be importable."""
        logger.info("session/manager.py should no longer be importable")
        with pytest.raises(ImportError):
            import orchestrator.session.manager  # noqa: F401

    def test_session_init_does_not_export_manager(self):
        """The session __init__.py should not reference SessionManager."""
        logger.info("The session __init__.py should not reference SessionManager")
        from orchestrator import session

        all_exports = getattr(session, "__all__", [])
        assert "SessionManager" not in all_exports
