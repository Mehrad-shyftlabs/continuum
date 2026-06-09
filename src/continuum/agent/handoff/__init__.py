"""
Handoff module for agent-to-agent transitions.

Provides handoff management with history summarization.
"""

from continuum.agent.handoff.history import (
    HistorySummarizer,
    default_history_mapper,
    extract_nested_history,
    flatten_nested_history,
    format_message_for_summary,
    summarize_conversation,
)
from continuum.agent.handoff.manager import HandoffManager

__all__ = [
    "HandoffManager",
    "HistorySummarizer",
    "default_history_mapper",
    "summarize_conversation",
    "extract_nested_history",
    "flatten_nested_history",
    "format_message_for_summary",
]
