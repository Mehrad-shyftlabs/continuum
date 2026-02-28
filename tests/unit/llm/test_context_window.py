"""Unit tests for LLM context window management."""

from unittest.mock import patch

import pytest

from orchestrator.llm.context_window import (
    ContextWindowManager,
    ModelLimits,
    TruncationResult,
    TruncationStrategy,
)
import logging

logger = logging.getLogger(__name__)


class TestModelLimits:
    def test_effective_input_limit(self):
        logger.info("ModelLimits: effective input limit")
        ml = ModelLimits(model="gpt-4", max_tokens=8192)
        assert ml.effective_input_limit == int(8192 * 0.75)

    def test_effective_input_limit_with_max_input(self):
        logger.info("ModelLimits: effective input limit with max input")
        ml = ModelLimits(model="gpt-4", max_tokens=8192, max_input_tokens=6000)
        assert ml.effective_input_limit == 6000

    def test_to_dict(self):
        logger.info("ModelLimits: to dict")
        ml = ModelLimits(model="gpt-4", max_tokens=8192)
        d = ml.to_dict()
        assert d["model"] == "gpt-4"
        assert d["max_tokens"] == 8192
        assert "effective_input_limit" in d


class TestTruncationResult:
    def test_to_dict(self):
        logger.info("TruncationResult: to dict")
        tr = TruncationResult(
            original_token_count=1000,
            truncated_token_count=500,
            messages_removed=3,
            was_truncated=True,
            strategy_used=TruncationStrategy.OLDEST_FIRST,
        )
        d = tr.to_dict()
        assert d["was_truncated"] is True
        assert d["messages_removed"] == 3
        assert d["strategy_used"] == "oldest_first"


class TestTruncationStrategy:
    def test_values(self):
        logger.info("TruncationStrategy: values")
        assert TruncationStrategy.OLDEST_FIRST == "oldest_first"
        assert TruncationStrategy.KEEP_SYSTEM_AND_RECENT == "keep_system_and_recent"
        assert TruncationStrategy.SMART == "smart"
        assert TruncationStrategy.NONE == "none"


class TestContextWindowManager:
    def test_init(self):
        logger.info("ContextWindowManager: init")
        cwm = ContextWindowManager()
        assert cwm is not None

    @patch("orchestrator.llm.context_window.litellm")
    def test_get_model_limits(self, mock_ll):
        logger.info("ContextWindowManager: get model limits")
        mock_ll.get_model_info.return_value = {
            "max_tokens": 8192,
            "max_input_tokens": 6000,
            "max_output_tokens": 2192,
        }
        cwm = ContextWindowManager()
        limits = cwm.get_model_limits("gpt-4")
        assert limits.max_tokens == 8192

    @patch("orchestrator.llm.context_window.litellm")
    def test_get_model_limits_fallback(self, mock_ll):
        logger.info("ContextWindowManager: get model limits fallback")
        mock_ll.get_model_info.side_effect = Exception("not found")
        cwm = ContextWindowManager()
        limits = cwm.get_model_limits("unknown-model")
        assert limits.max_tokens > 0
