"""Tests for agent/handoff/history.py."""

import pytest

from orchestrator.agent.handoff.history import (
    HistorySummarizer,
    get_history_markers,
    reset_history_markers,
    set_history_markers,
)
from orchestrator.agent.types import HistorySummarizationMode
import logging

logger = logging.getLogger(__name__)


class TestHistoryMarkers:
    def setup_method(self):
        reset_history_markers()

    def test_get_default_markers(self):
        logger.info("HistoryMarkers: get default markers")
        start, end = get_history_markers()
        assert "CONVERSATION HISTORY" in start
        assert "CONVERSATION HISTORY" in end

    def test_set_custom_markers(self):
        logger.info("HistoryMarkers: set custom markers")
        set_history_markers(start="<START>", end="<END>")
        start, end = get_history_markers()
        assert start == "<START>"
        assert end == "<END>"

    def test_set_partial_start(self):
        logger.info("HistoryMarkers: set partial start")
        set_history_markers(start="<NEW_START>")
        start, end = get_history_markers()
        assert start == "<NEW_START>"
        assert "CONVERSATION HISTORY" in end

    def test_set_partial_end(self):
        logger.info("HistoryMarkers: set partial end")
        set_history_markers(end="<NEW_END>")
        start, end = get_history_markers()
        assert "CONVERSATION HISTORY" in start
        assert end == "<NEW_END>"

    def test_reset_markers(self):
        logger.info("HistoryMarkers: reset markers")
        set_history_markers(start="X", end="Y")
        reset_history_markers()
        start, end = get_history_markers()
        assert "CONVERSATION HISTORY" in start
        assert "CONVERSATION HISTORY" in end

    def teardown_method(self):
        reset_history_markers()


class TestHistorySummarizer:
    def test_defaults(self):
        logger.info("HistorySummarizer: defaults")
        hs = HistorySummarizer()
        assert hs.mode == HistorySummarizationMode.HYBRID
        assert hs.recent_n == 5
        assert hs.max_length == 4000
        assert hs.include_tool_calls is True
        assert hs.include_metadata is False

    def test_custom_mode(self):
        logger.info("HistorySummarizer: custom mode")
        hs = HistorySummarizer(mode=HistorySummarizationMode.FULL)
        assert hs.mode == HistorySummarizationMode.FULL

    def test_recent_n(self):
        logger.info("HistorySummarizer: recent n")
        hs = HistorySummarizer(mode=HistorySummarizationMode.RECENT_N, recent_n=10)
        assert hs.recent_n == 10

    def test_custom_max_length(self):
        logger.info("HistorySummarizer: custom max length")
        hs = HistorySummarizer(max_length=8000)
        assert hs.max_length == 8000

    @pytest.mark.asyncio
    async def test_summarize_full_mode(self):
        logger.info("HistorySummarizer: summarize full mode")
        hs = HistorySummarizer(mode=HistorySummarizationMode.FULL)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = await hs.summarize(messages)
        assert isinstance(result, list)
        assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_summarize_empty_messages(self):
        logger.info("HistorySummarizer: summarize empty messages")
        hs = HistorySummarizer(mode=HistorySummarizationMode.FULL)
        result = await hs.summarize([])
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_summarize_recent_n(self):
        logger.info("HistorySummarizer: summarize recent n")
        hs = HistorySummarizer(mode=HistorySummarizationMode.RECENT_N, recent_n=2)
        messages = [
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "msg2"},
            {"role": "user", "content": "msg3"},
            {"role": "assistant", "content": "msg4"},
            {"role": "user", "content": "msg5"},
        ]
        result = await hs.summarize(messages)
        assert isinstance(result, list)
