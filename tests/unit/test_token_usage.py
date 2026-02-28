"""Unit tests for TokenUsage.add() (Issue 9 - clear merge logic)."""

import pytest

from orchestrator.agent.types import TokenUsage
import logging

logger = logging.getLogger(__name__)


class TestTokenUsageAdd:
    """Tests for the rewritten TokenUsage.add() method."""

    def test_add_basic(self):
        logger.info("TokenUsageAdd: add basic")
        a = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        b = TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30)
        result = a.add(b)
        assert result.prompt_tokens == 30
        assert result.completion_tokens == 15
        assert result.total_tokens == 45

    def test_add_with_model_usage_both_have_same_model(self):
        logger.info("TokenUsageAdd: add with model usage both have same model")
        a = TokenUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            model_usage={"gpt-4": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
        )
        b = TokenUsage(
            prompt_tokens=20,
            completion_tokens=10,
            total_tokens=30,
            model_usage={"gpt-4": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}},
        )
        result = a.add(b)
        assert result.model_usage["gpt-4"]["prompt_tokens"] == 30
        assert result.model_usage["gpt-4"]["completion_tokens"] == 15
        assert result.model_usage["gpt-4"]["total_tokens"] == 45

    def test_add_with_model_usage_different_models(self):
        logger.info("TokenUsageAdd: add with model usage different models")
        a = TokenUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            model_usage={"gpt-4": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
        )
        b = TokenUsage(
            prompt_tokens=20,
            completion_tokens=10,
            total_tokens=30,
            model_usage={"claude-3": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}},
        )
        result = a.add(b)
        assert "gpt-4" in result.model_usage
        assert "claude-3" in result.model_usage
        assert result.model_usage["gpt-4"]["prompt_tokens"] == 10
        assert result.model_usage["claude-3"]["prompt_tokens"] == 20

    def test_add_with_model_usage_one_empty(self):
        logger.info("TokenUsageAdd: add with model usage one empty")
        a = TokenUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            model_usage={"gpt-4": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
        )
        b = TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        result = a.add(b)
        assert result.model_usage["gpt-4"]["prompt_tokens"] == 10
        assert result.model_usage["gpt-4"]["completion_tokens"] == 5
        assert result.model_usage["gpt-4"]["total_tokens"] == 15

    def test_add_preserves_self_unique_models(self):
        logger.info("TokenUsageAdd: add preserves self unique models")
        a = TokenUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            model_usage={
                "gpt-4": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                "gpt-3.5": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
        )
        b = TokenUsage(
            prompt_tokens=20,
            completion_tokens=10,
            total_tokens=30,
            model_usage={"gpt-4": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}},
        )
        result = a.add(b)
        assert "gpt-3.5" in result.model_usage
        assert result.model_usage["gpt-3.5"]["prompt_tokens"] == 5

    def test_add_preserves_other_unique_models(self):
        logger.info("TokenUsageAdd: add preserves other unique models")
        a = TokenUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            model_usage={"gpt-4": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
        )
        b = TokenUsage(
            prompt_tokens=20,
            completion_tokens=10,
            total_tokens=30,
            model_usage={
                "gpt-4": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
                "claude-3": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11},
            },
        )
        result = a.add(b)
        assert "claude-3" in result.model_usage
        assert result.model_usage["claude-3"]["prompt_tokens"] == 8

    def test_to_dict_roundtrip(self):
        logger.info("TokenUsageAdd: to dict roundtrip")
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            model_usage={"gpt-4": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}},
        )
        d = usage.to_dict()
        restored = TokenUsage.from_dict(d)
        assert restored.prompt_tokens == usage.prompt_tokens
        assert restored.completion_tokens == usage.completion_tokens
        assert restored.total_tokens == usage.total_tokens
        assert restored.model_usage == usage.model_usage
