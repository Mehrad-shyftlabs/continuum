"""Comprehensive tests for observability/provider_manager.py."""

from unittest.mock import MagicMock

import pytest

from orchestrator.observability.provider_manager import ProviderManager, get_provider_manager
from orchestrator.observability.providers.base import ProviderCapabilities
import logging

logger = logging.getLogger(__name__)


def _make_mock_provider(capabilities=None):
    provider = MagicMock()
    caps = capabilities or [
        ProviderCapabilities.TRACE, ProviderCapabilities.SPAN,
        ProviderCapabilities.GENERATION, ProviderCapabilities.EVENT,
        ProviderCapabilities.SCORE,
    ]
    provider.supports_feature.side_effect = lambda f: f in caps
    return provider


def _make_registry(providers=None):
    registry = MagicMock()
    if providers is None:
        providers = {}
    registry.get_enabled.return_value = providers
    registry.get_all.return_value = providers
    return registry


class TestProviderManager:
    def test_init_default(self):
        logger.info("ProviderManager: init default")
        pm = ProviderManager(registry=_make_registry())
        assert pm is not None

    def test_is_enabled_no_providers(self):
        logger.info("ProviderManager: is enabled no providers")
        pm = ProviderManager(registry=_make_registry({}))
        assert pm.is_enabled is False

    def test_is_enabled_with_providers(self):
        logger.info("ProviderManager: is enabled with providers")
        pm = ProviderManager(registry=_make_registry({"p1": _make_mock_provider()}))
        assert pm.is_enabled is True

    def test_supports_feature(self):
        logger.info("ProviderManager: supports feature")
        provider = _make_mock_provider([ProviderCapabilities.TRACE])
        pm = ProviderManager(registry=_make_registry({"p1": provider}))
        assert pm.supports_feature(ProviderCapabilities.TRACE) is True

    def test_supports_feature_false(self):
        logger.info("ProviderManager: supports feature false")
        provider = MagicMock()
        provider.supports_feature.return_value = False
        pm = ProviderManager(registry=_make_registry({"p1": provider}))
        assert pm.supports_feature(ProviderCapabilities.TRACE) is False

    def test_trace_no_providers(self):
        logger.info("ProviderManager: trace no providers")
        pm = ProviderManager(registry=_make_registry({}))
        result = pm.trace(name="test")
        assert result is None

    def test_trace_with_provider(self):
        logger.info("ProviderManager: trace with provider")
        provider = _make_mock_provider()
        provider.trace.return_value = MagicMock(id="trace-1")
        pm = ProviderManager(registry=_make_registry({"p1": provider}))
        result = pm.trace(name="test", user_id="u1")
        assert result is not None
        provider.trace.assert_called_once()

    def test_trace_provider_exception(self):
        logger.info("ProviderManager: trace provider exception")
        provider = _make_mock_provider()
        provider.trace.side_effect = Exception("fail")
        pm = ProviderManager(registry=_make_registry({"p1": provider}))
        result = pm.trace(name="test")
        assert result is None

    def test_span_no_providers(self):
        logger.info("ProviderManager: span no providers")
        pm = ProviderManager(registry=_make_registry({}))
        result = pm.span(name="test")
        assert result is None

    def test_span_with_provider(self):
        logger.info("ProviderManager: span with provider")
        provider = _make_mock_provider()
        provider.span.return_value = MagicMock(id="span-1")
        pm = ProviderManager(registry=_make_registry({"p1": provider}))
        result = pm.span(name="test", trace_id="t1")
        assert result is not None

    def test_span_provider_exception(self):
        logger.info("ProviderManager: span provider exception")
        provider = _make_mock_provider()
        provider.span.side_effect = Exception("fail")
        pm = ProviderManager(registry=_make_registry({"p1": provider}))
        result = pm.span(name="test")
        assert result is None

    def test_generation_no_providers(self):
        logger.info("ProviderManager: generation no providers")
        pm = ProviderManager(registry=_make_registry({}))
        result = pm.generation(name="test")
        assert result is None

    def test_generation_with_provider(self):
        logger.info("ProviderManager: generation with provider")
        provider = _make_mock_provider()
        provider.generation.return_value = MagicMock(id="gen-1")
        pm = ProviderManager(registry=_make_registry({"p1": provider}))
        result = pm.generation(name="test", model="gpt-4")
        assert result is not None

    def test_generation_provider_exception(self):
        logger.info("ProviderManager: generation provider exception")
        provider = _make_mock_provider()
        provider.generation.side_effect = Exception("fail")
        pm = ProviderManager(registry=_make_registry({"p1": provider}))
        result = pm.generation(name="test")
        assert result is None

    def test_event_no_providers(self):
        logger.info("ProviderManager: event no providers")
        pm = ProviderManager(registry=_make_registry({}))
        result = pm.event(name="test")
        assert result is None

    def test_event_with_provider(self):
        logger.info("ProviderManager: event with provider")
        provider = _make_mock_provider()
        provider.event.return_value = MagicMock()
        pm = ProviderManager(registry=_make_registry({"p1": provider}))
        result = pm.event(name="test")
        assert result is not None

    def test_score_no_providers(self):
        logger.info("ProviderManager: score no providers")
        pm = ProviderManager(registry=_make_registry({}))
        result = pm.score(trace_id="t1", name="quality", value=0.9)
        assert result is None

    def test_score_with_provider(self):
        logger.info("ProviderManager: score with provider")
        provider = _make_mock_provider()
        provider.score.return_value = MagicMock()
        pm = ProviderManager(registry=_make_registry({"p1": provider}))
        result = pm.score(trace_id="t1", name="quality", value=0.9)
        assert result is not None

    def test_score_provider_exception(self):
        logger.info("ProviderManager: score provider exception")
        provider = _make_mock_provider()
        provider.score.side_effect = Exception("fail")
        pm = ProviderManager(registry=_make_registry({"p1": provider}))
        result = pm.score(trace_id="t1", name="quality", value=0.9)
        assert result is None

    def test_flush(self):
        logger.info("ProviderManager: flush")
        provider = _make_mock_provider()
        pm = ProviderManager(registry=_make_registry({"p1": provider}))
        pm.flush()
        provider.flush.assert_called_once()

    def test_flush_exception(self):
        logger.info("ProviderManager: flush exception")
        provider = _make_mock_provider()
        provider.flush.side_effect = Exception("fail")
        pm = ProviderManager(registry=_make_registry({"p1": provider}))
        pm.flush()

    def test_shutdown(self):
        logger.info("ProviderManager: shutdown")
        provider = _make_mock_provider()
        pm = ProviderManager(registry=_make_registry({"p1": provider}))
        pm.shutdown()
        provider.shutdown.assert_called_once()

    def test_shutdown_exception(self):
        logger.info("ProviderManager: shutdown exception")
        provider = _make_mock_provider()
        provider.shutdown.side_effect = Exception("fail")
        pm = ProviderManager(registry=_make_registry({"p1": provider}))
        pm.shutdown()

    def test_create_prompt_no_providers(self):
        logger.info("ProviderManager: create prompt no providers")
        pm = ProviderManager(registry=_make_registry({}))
        result = pm.create_prompt("test", "prompt text")
        assert result is None

    def test_create_prompt_with_provider(self):
        logger.info("ProviderManager: create prompt with provider")
        provider = _make_mock_provider([ProviderCapabilities.PROMPT_MANAGEMENT])
        provider.create_prompt.return_value = MagicMock()
        pm = ProviderManager(registry=_make_registry({"p1": provider}))
        result = pm.create_prompt("test", "prompt text")
        assert result is not None

    def test_get_prompt_no_providers(self):
        logger.info("ProviderManager: get prompt no providers")
        pm = ProviderManager(registry=_make_registry({}))
        result = pm.get_prompt("test")
        assert result is None

    def test_get_prompt_with_provider(self):
        logger.info("ProviderManager: get prompt with provider")
        provider = _make_mock_provider([ProviderCapabilities.PROMPT_MANAGEMENT])
        provider.get_prompt.return_value = "prompt text"
        pm = ProviderManager(registry=_make_registry({"p1": provider}))
        result = pm.get_prompt("test")
        assert result == "prompt text"

    def test_create_dataset_no_providers(self):
        logger.info("ProviderManager: create dataset no providers")
        pm = ProviderManager(registry=_make_registry({}))
        result = pm.create_dataset("test")
        assert result is None

    def test_create_dataset_with_provider(self):
        logger.info("ProviderManager: create dataset with provider")
        provider = _make_mock_provider([ProviderCapabilities.DATASET_MANAGEMENT])
        provider.create_dataset.return_value = MagicMock()
        pm = ProviderManager(registry=_make_registry({"p1": provider}))
        result = pm.create_dataset("test")
        assert result is not None

    def test_get_dataset_no_providers(self):
        logger.info("ProviderManager: get dataset no providers")
        pm = ProviderManager(registry=_make_registry({}))
        result = pm.get_dataset("test")
        assert result is None

    def test_get_dataset_with_provider(self):
        logger.info("ProviderManager: get dataset with provider")
        provider = _make_mock_provider([ProviderCapabilities.DATASET_MANAGEMENT])
        provider.get_dataset.return_value = MagicMock()
        pm = ProviderManager(registry=_make_registry({"p1": provider}))
        result = pm.get_dataset("test")
        assert result is not None

    def test_create_dataset_item_no_providers(self):
        logger.info("ProviderManager: create dataset item no providers")
        pm = ProviderManager(registry=_make_registry({}))
        result = pm.create_dataset_item("ds", input={"x": 1})
        assert result is None

    def test_create_dataset_item_with_provider(self):
        logger.info("ProviderManager: create dataset item with provider")
        provider = _make_mock_provider([ProviderCapabilities.DATASET_MANAGEMENT])
        provider.create_dataset_item.return_value = MagicMock()
        pm = ProviderManager(registry=_make_registry({"p1": provider}))
        result = pm.create_dataset_item("ds", input={"x": 1})
        assert result is not None

    def test_multiple_providers(self):
        logger.info("ProviderManager: multiple providers")
        p1 = _make_mock_provider()
        p1.trace.return_value = None
        p2 = _make_mock_provider()
        p2.trace.return_value = MagicMock(id="from-p2")
        pm = ProviderManager(registry=_make_registry({"p1": p1, "p2": p2}))
        result = pm.trace(name="test")
        assert result is not None


class TestGetProviderManager:
    def test_returns_instance(self):
        logger.info("GetProviderManager: returns instance")
        import orchestrator.observability.provider_manager as mod
        old = mod._global_manager
        mod._global_manager = None
        try:
            pm = get_provider_manager()
            assert isinstance(pm, ProviderManager)
        finally:
            mod._global_manager = old

    def test_returns_same_instance(self):
        logger.info("GetProviderManager: returns same instance")
        import orchestrator.observability.provider_manager as mod

        old = mod._global_manager
        mod._global_manager = None
        try:
            pm1 = get_provider_manager()
            pm2 = get_provider_manager()
            assert pm1 is pm2
        finally:
            mod._global_manager = old
