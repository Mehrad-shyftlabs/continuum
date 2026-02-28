"""Tests for agent/handoff/manager.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.agent.base import BaseAgent
from orchestrator.agent.exceptions import (
    HandoffCycleDetectedError,
    HandoffDepthExceededError,
    HandoffNotAllowedError,
)
from orchestrator.agent.handoff.manager import HandoffManager
from orchestrator.agent.types import Handoff, HistorySummarizationMode
import logging

logger = logging.getLogger(__name__)


class TestHandoffManagerInit:
    def test_default(self):
        logger.info("HandoffManagerInit: default")
        hm = HandoffManager()
        assert hm._max_depth == 10
        assert hm._llm_client is None
        assert hm._tracing_manager is None

    def test_custom_depth(self):
        logger.info("HandoffManagerInit: custom depth")
        hm = HandoffManager(max_depth=3)
        assert hm._max_depth == 3

    def test_with_clients(self):
        logger.info("HandoffManagerInit: with clients")
        mock_llm = MagicMock()
        mock_tracing = MagicMock()
        hm = HandoffManager(llm_client=mock_llm, tracing_manager=mock_tracing)
        assert hm._llm_client is mock_llm
        assert hm._tracing_manager is mock_tracing


class TestHandoffManagerValidation:
    def _make_agent(self, name, handoffs=None):
        agent = MagicMock(spec=BaseAgent)
        agent.name = name
        agent.handoffs = handoffs or []
        agent.get_handoff = MagicMock(return_value=None)
        return agent

    def test_validate_handoff_depth_exceeded(self):
        logger.info("HandoffManagerValidation: validate handoff depth exceeded")
        hm = HandoffManager(max_depth=2)
        agent = self._make_agent("a1")
        with pytest.raises(HandoffDepthExceededError):
            hm.validate_handoff(agent, "a2", current_depth=2)

    def test_validate_handoff_cycle_detected(self):
        logger.info("HandoffManagerValidation: validate handoff cycle detected")
        hm = HandoffManager()
        agent = self._make_agent("a1")
        with pytest.raises(HandoffCycleDetectedError):
            hm.validate_handoff(agent, "a1", agent_stack=["a0", "a1"])

    def test_validate_handoff_not_allowed(self):
        logger.info("HandoffManagerValidation: validate handoff not allowed")
        hm = HandoffManager()
        agent = self._make_agent("a1")
        agent.get_handoff.return_value = None
        with pytest.raises(HandoffNotAllowedError):
            hm.validate_handoff(agent, "a2")

    def test_validate_handoff_success(self):
        logger.info("HandoffManagerValidation: validate handoff success")
        hm = HandoffManager()
        agent = self._make_agent("a1")
        handoff_def = MagicMock(spec=Handoff)
        agent.get_handoff.return_value = handoff_def
        result = hm.validate_handoff(agent, "a2")
        assert result is handoff_def


class TestHandoffManagerDetectCycle:
    def test_no_cycle(self):
        logger.info("HandoffManagerDetectCycle: no cycle")
        hm = HandoffManager()
        assert hm.detect_cycle(["a1", "a2"], "a3") is False

    def test_cycle_detected(self):
        logger.info("HandoffManagerDetectCycle: cycle detected")
        hm = HandoffManager()
        assert hm.detect_cycle(["a1", "a2", "a3"], "a1") is True

    def test_empty_stack(self):
        logger.info("HandoffManagerDetectCycle: empty stack")
        hm = HandoffManager()
        assert hm.detect_cycle([], "a1") is False
