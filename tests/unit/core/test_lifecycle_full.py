"""Comprehensive unit tests for core/lifecycle.py."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.core.lifecycle import (
    ConfigurationError,
    InitializationResult,
    LifecycleState,
    OrchestratorLifecycle,
    get_lifecycle_manager,
    validate_configuration,
)
import logging

logger = logging.getLogger(__name__)


class TestConfigurationError:
    def test_basic(self):
        logger.info("ConfigurationError: basic")
        err = ConfigurationError(field="FOO", message="missing")
        assert err.field == "FOO"
        assert err.message == "missing"
        assert err.severity == "error"

    def test_warning_severity(self):
        logger.info("ConfigurationError: warning severity")
        err = ConfigurationError(field="BAR", message="warn", severity="warning")
        assert err.severity == "warning"

    def test_str_representation(self):
        logger.info("ConfigurationError: str representation")
        err = ConfigurationError(field="FOO", message="missing")
        assert "[ERROR]" in str(err)
        assert "FOO" in str(err)

    def test_str_warning(self):
        logger.info("ConfigurationError: str warning")
        err = ConfigurationError(field="BAR", message="optional", severity="warning")
        assert "[WARNING]" in str(err)


class TestValidateConfiguration:
    @patch("orchestrator.core.lifecycle.settings")
    def test_memory_enabled_no_qdrant(self, mock_settings):
        logger.info("ValidateConfiguration: memory enabled no qdrant")
        mock_settings.memory_enabled = True
        mock_settings.qdrant_host = ""
        mock_settings.memory_llm_model = ""
        mock_settings.embedder_model = ""
        mock_settings.session_enabled = False
        mock_settings.langfuse_enabled = False
        mock_settings.openai_api_key = "sk-test"
        mock_settings.anthropic_api_key = ""
        mock_settings.gemini_api_key = ""
        mock_settings.azure_api_key = ""
        mock_settings.default_llm_model = "gpt-4"

        errors, warnings = validate_configuration()
        error_fields = [e.field for e in errors]
        assert "QDRANT_HOST" in error_fields

    @patch("orchestrator.core.lifecycle.settings")
    def test_session_enabled_no_redis(self, mock_settings):
        logger.info("ValidateConfiguration: session enabled no redis")
        mock_settings.memory_enabled = False
        mock_settings.session_enabled = True
        mock_settings.session_redis_host = ""
        mock_settings.langfuse_enabled = False
        mock_settings.openai_api_key = "sk-test"
        mock_settings.anthropic_api_key = ""
        mock_settings.gemini_api_key = ""
        mock_settings.azure_api_key = ""
        mock_settings.default_llm_model = "gpt-4"

        errors, warnings = validate_configuration()
        error_fields = [e.field for e in errors]
        assert "SESSION_REDIS_HOST" in error_fields

    @patch("orchestrator.core.lifecycle.settings")
    def test_langfuse_enabled_no_keys(self, mock_settings):
        logger.info("ValidateConfiguration: langfuse enabled no keys")
        mock_settings.memory_enabled = False
        mock_settings.session_enabled = False
        mock_settings.langfuse_enabled = True
        mock_settings.langfuse_public_key = ""
        mock_settings.langfuse_secret_key = ""
        mock_settings.openai_api_key = "sk-test"
        mock_settings.anthropic_api_key = ""
        mock_settings.gemini_api_key = ""
        mock_settings.azure_api_key = ""
        mock_settings.default_llm_model = "gpt-4"

        errors, warnings = validate_configuration()
        error_fields = [e.field for e in errors]
        assert "LANGFUSE_PUBLIC_KEY" in error_fields
        assert "LANGFUSE_SECRET_KEY" in error_fields

    @patch("orchestrator.core.lifecycle.settings")
    def test_no_llm_key_warning(self, mock_settings):
        logger.info("ValidateConfiguration: no llm key warning")
        mock_settings.memory_enabled = False
        mock_settings.session_enabled = False
        mock_settings.langfuse_enabled = False
        mock_settings.openai_api_key = ""
        mock_settings.anthropic_api_key = ""
        mock_settings.gemini_api_key = ""
        mock_settings.azure_api_key = ""
        mock_settings.default_llm_model = "gpt-4"

        errors, warnings = validate_configuration()
        warning_fields = [w.field for w in warnings]
        assert "LLM_API_KEY" in warning_fields

    @patch("orchestrator.core.lifecycle.settings")
    def test_no_default_model_warning(self, mock_settings):
        logger.info("ValidateConfiguration: no default model warning")
        mock_settings.memory_enabled = False
        mock_settings.session_enabled = False
        mock_settings.langfuse_enabled = False
        mock_settings.openai_api_key = "sk-test"
        mock_settings.anthropic_api_key = ""
        mock_settings.gemini_api_key = ""
        mock_settings.azure_api_key = ""
        mock_settings.default_llm_model = ""

        errors, warnings = validate_configuration()
        warning_fields = [w.field for w in warnings]
        assert "DEFAULT_LLM_MODEL" in warning_fields


class TestLifecycleState:
    def test_values(self):
        logger.info("LifecycleState: values")
        assert LifecycleState.NOT_INITIALIZED == "not_initialized"
        assert LifecycleState.RUNNING == "running"
        assert LifecycleState.SHUTDOWN == "shutdown"
        assert LifecycleState.FAILED == "failed"


class TestInitializationResult:
    def test_success(self):
        logger.info("InitializationResult: success")
        r = InitializationResult(success=True, state=LifecycleState.RUNNING)
        assert r.success is True
        assert r.errors == []
        assert r.warnings == []

    def test_to_dict(self):
        logger.info("InitializationResult: to dict")
        r = InitializationResult(
            success=False, state=LifecycleState.FAILED,
            errors=["err1"], warnings=["warn1"],
        )
        d = r.to_dict()
        assert d["success"] is False
        assert d["state"] == "failed"
        assert "err1" in d["errors"]
        assert "initialized_at" in d


class TestOrchestratorLifecycle:
    def test_init_defaults(self):
        logger.info("OrchestratorLifecycle: init defaults")
        lc = OrchestratorLifecycle()
        assert lc.state == LifecycleState.NOT_INITIALIZED
        assert lc.is_running is False

    def test_init_custom_params(self):
        logger.info("OrchestratorLifecycle: init custom params")
        lc = OrchestratorLifecycle(
            shutdown_timeout=30.0,
            fail_on_unhealthy=True,
            verify_connections=False,
            enable_signal_handlers=False,
        )
        assert lc._shutdown_timeout == 30.0
        assert lc._fail_on_unhealthy is True

    def test_register_shutdown_callback(self):
        logger.info("OrchestratorLifecycle: register shutdown callback")
        lc = OrchestratorLifecycle()
        async def callback():
            pass
        lc.register_shutdown_callback(callback)
        assert len(lc._shutdown_callbacks) == 1

    @pytest.mark.asyncio
    async def test_initialize_already_running(self):
        logger.info("OrchestratorLifecycle: initialize already running")
        lc = OrchestratorLifecycle(verify_connections=False, enable_signal_handlers=False)
        lc._state = LifecycleState.RUNNING
        result = await lc.initialize()
        assert result.success is True
        assert "already initialized" in result.warnings[0]

    @pytest.mark.asyncio
    async def test_initialize_already_in_progress(self):
        logger.info("OrchestratorLifecycle: initialize already in progress")
        lc = OrchestratorLifecycle(verify_connections=False, enable_signal_handlers=False)
        lc._state = LifecycleState.INITIALIZING
        result = await lc.initialize()
        assert result.success is False

    @pytest.mark.asyncio
    @patch("orchestrator.core.lifecycle.validate_configuration")
    @patch("orchestrator.core.lifecycle.settings")
    async def test_initialize_success_no_verify(self, mock_settings, mock_validate):
        logger.info("OrchestratorLifecycle: initialize success no verify")
        mock_validate.return_value = ([], [])
        mock_settings.memory_enabled = False
        mock_settings.session_enabled = False
        mock_settings.shared_services_enabled = False

        lc = OrchestratorLifecycle(
            verify_connections=False,
            enable_signal_handlers=False,
        )
        result = await lc.initialize()
        assert result.success is True
        assert lc.state == LifecycleState.RUNNING

    @pytest.mark.asyncio
    @patch("orchestrator.core.lifecycle.validate_configuration")
    @patch("orchestrator.core.lifecycle.settings")
    async def test_initialize_config_errors_fail_on_unhealthy(self, mock_settings, mock_validate):
        logger.info("OrchestratorLifecycle: initialize config errors fail on unhealthy")
        mock_validate.return_value = (
            [ConfigurationError(field="X", message="bad")], [],
        )
        mock_settings.memory_enabled = False
        mock_settings.session_enabled = False

        lc = OrchestratorLifecycle(
            verify_connections=False,
            enable_signal_handlers=False,
            fail_on_unhealthy=True,
        )
        result = await lc.initialize()
        assert result.success is False
        assert lc.state == LifecycleState.FAILED

    @pytest.mark.asyncio
    async def test_shutdown_not_running(self):
        logger.info("OrchestratorLifecycle: shutdown not running")
        lc = OrchestratorLifecycle(enable_signal_handlers=False)
        lc._state = LifecycleState.NOT_INITIALIZED
        await lc.shutdown()
        assert lc.state == LifecycleState.NOT_INITIALIZED

    @pytest.mark.asyncio
    async def test_shutdown_already_shutdown(self):
        logger.info("OrchestratorLifecycle: shutdown already shutdown")
        lc = OrchestratorLifecycle(enable_signal_handlers=False)
        lc._state = LifecycleState.SHUTDOWN
        await lc.shutdown()
        assert lc.state == LifecycleState.SHUTDOWN

    @pytest.mark.asyncio
    @patch("orchestrator.core.lifecycle.settings")
    async def test_shutdown_from_running(self, mock_settings):
        logger.info("OrchestratorLifecycle: shutdown from running")
        mock_settings.shared_services_enabled = False
        lc = OrchestratorLifecycle(enable_signal_handlers=False)
        lc._state = LifecycleState.RUNNING
        lc._initialized_components = []
        await lc.shutdown()
        assert lc.state == LifecycleState.SHUTDOWN

    @pytest.mark.asyncio
    @patch("orchestrator.core.lifecycle.settings")
    async def test_shutdown_with_callbacks(self, mock_settings):
        logger.info("OrchestratorLifecycle: shutdown with callbacks")
        mock_settings.shared_services_enabled = False
        lc = OrchestratorLifecycle(enable_signal_handlers=False)
        lc._state = LifecycleState.RUNNING
        lc._initialized_components = []

        callback_called = False
        async def my_callback():
            nonlocal callback_called
            callback_called = True

        lc.register_shutdown_callback(my_callback)
        await lc.shutdown()
        assert callback_called
        assert lc.state == LifecycleState.SHUTDOWN

    @pytest.mark.asyncio
    @patch("orchestrator.core.lifecycle.settings")
    async def test_signal_handler_ignores_during_shutdown(self, mock_settings):
        logger.info("OrchestratorLifecycle: signal handler ignores during shutdown")
        import signal
        lc = OrchestratorLifecycle(enable_signal_handlers=False)
        lc._state = LifecycleState.SHUTTING_DOWN
        await lc._signal_handler(signal.SIGTERM)
        assert lc.state == LifecycleState.SHUTTING_DOWN

    @pytest.mark.asyncio
    async def test_context_manager(self):
        logger.info("OrchestratorLifecycle: context manager")
        with patch("orchestrator.core.lifecycle.validate_configuration", return_value=([], [])):
            with patch("orchestrator.core.lifecycle.settings") as mock_settings:
                mock_settings.memory_enabled = False
                mock_settings.session_enabled = False
                mock_settings.shared_services_enabled = False

                async with OrchestratorLifecycle(
                    verify_connections=False,
                    enable_signal_handlers=False,
                ) as lc:
                    assert lc.state == LifecycleState.RUNNING
                assert lc.state == LifecycleState.SHUTDOWN

    @pytest.mark.asyncio
    async def test_get_health(self):
        logger.info("OrchestratorLifecycle: get health")
        lc = OrchestratorLifecycle(enable_signal_handlers=False)
        mock_checker = MagicMock()
        mock_checker.check_all = AsyncMock(return_value=MagicMock(checks=[]))
        lc._health_checker = mock_checker
        result = await lc.get_health()
        assert result is not None


class TestGetLifecycleManager:
    def test_returns_instance(self):
        logger.info("GetLifecycleManager: returns instance")
        import orchestrator.core.lifecycle as mod
        old = mod._global_lifecycle
        mod._global_lifecycle = None
        try:
            lm = get_lifecycle_manager(enable_signal_handlers=False)
            assert isinstance(lm, OrchestratorLifecycle)
        finally:
            mod._global_lifecycle = old

    def test_returns_same_instance(self):
        logger.info("GetLifecycleManager: returns same instance")
        import orchestrator.core.lifecycle as mod

        old = mod._global_lifecycle
        mod._global_lifecycle = None
        try:
            lm1 = get_lifecycle_manager(enable_signal_handlers=False)
            lm2 = get_lifecycle_manager(enable_signal_handlers=False)
            assert lm1 is lm2
        finally:
            mod._global_lifecycle = old
