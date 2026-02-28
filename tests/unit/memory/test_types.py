"""Unit tests for memory types."""

from datetime import datetime

import pytest

from orchestrator.memory.types import (
    MemoryAddResult,
    MemoryEntry,
    MemoryFilter,
    MemoryMetadata,
    MemorySearchResult,
)
import logging

logger = logging.getLogger(__name__)


class TestMemoryMetadata:
    def test_to_dict_full(self):
        logger.info("MemoryMetadata: to dict full")
        m = MemoryMetadata(
            category="pref", tags=["t1"], source="chat",
            confidence=0.9, created_at=datetime(2025, 1, 1),
            custom={"extra": "val"},
        )
        d = m.to_dict()
        assert d["category"] == "pref"
        assert d["tags"] == ["t1"]
        assert d["extra"] == "val"
        assert "confidence" in d

    def test_to_dict_empty(self):
        logger.info("MemoryMetadata: to dict empty")
        m = MemoryMetadata()
        d = m.to_dict()
        assert d == {}


class TestMemoryEntry:
    def test_to_dict(self):
        logger.info("MemoryEntry: to dict")
        e = MemoryEntry(id="1", memory="likes coffee", user_id="u1", score=0.95)
        d = e.to_dict()
        assert d["id"] == "1"
        assert d["memory"] == "likes coffee"
        assert d["user_id"] == "u1"
        assert d["score"] == 0.95

    def test_from_mem0_result(self):
        logger.info("MemoryEntry: from mem0 result")
        data = {"id": "1", "memory": "test", "user_id": "u1", "metadata": {"k": "v"}}
        e = MemoryEntry.from_mem0_result(data)
        assert e.id == "1"
        assert e.metadata["k"] == "v"

    def test_from_mem0_result_none_metadata(self):
        logger.info("MemoryEntry: from mem0 result none metadata")
        e = MemoryEntry.from_mem0_result({"id": "1", "memory": "t", "metadata": None})
        assert e.metadata == {}

    def test_from_mem0_result_invalid_metadata(self):
        logger.info("MemoryEntry: from mem0 result invalid metadata")
        e = MemoryEntry.from_mem0_result({"id": "1", "memory": "t", "metadata": "invalid"})
        assert e.metadata == {}


class TestMemorySearchResult:
    def test_from_mem0_response(self):
        logger.info("MemorySearchResult: from mem0 response")
        resp = {"results": [{"id": "1", "memory": "t1"}, {"id": "2", "memory": "t2"}]}
        sr = MemorySearchResult.from_mem0_response(resp, "query", 5)
        assert len(sr.results) == 2
        assert sr.query == "query"
        assert sr.total_results == 2

    def test_get_memory_strings(self):
        logger.info("MemorySearchResult: get memory strings")
        sr = MemorySearchResult(
            results=[MemoryEntry(id="1", memory="a"), MemoryEntry(id="2", memory="b")],
            query="q", limit=5,
        )
        assert sr.get_memory_strings() == ["a", "b"]

    def test_get_top_k(self):
        logger.info("MemorySearchResult: get top k")
        sr = MemorySearchResult(
            results=[
                MemoryEntry(id="1", memory="a", score=0.5),
                MemoryEntry(id="2", memory="b", score=0.9),
                MemoryEntry(id="3", memory="c", score=0.7),
            ],
            query="q", limit=5,
        )
        top = sr.get_top_k(2)
        assert top[0].score == 0.9
        assert len(top) == 2


class TestMemoryAddResult:
    def test_from_mem0_response_dict(self):
        logger.info("MemoryAddResult: from mem0 response dict")
        resp = {"message": "added", "results": [{"id": "1"}], "relations": []}
        r = MemoryAddResult.from_mem0_response(resp)
        assert r.message == "added"
        assert len(r.results) == 1

    def test_from_mem0_response_str(self):
        logger.info("MemoryAddResult: from mem0 response str")
        r = MemoryAddResult.from_mem0_response("ok")
        assert r.message == "ok"


class TestMemoryFilter:
    def test_to_mem0_filter(self):
        logger.info("MemoryFilter: to mem0 filter")
        f = MemoryFilter(user_id="u1", agent_id="a1", category="pref", tags=["t1"])
        d = f.to_mem0_filter()
        assert d["user_id"] == "u1"
        assert d["agent_id"] == "a1"
        assert d["category"] == "pref"
        assert d["tags"] == ["t1"]

    def test_to_mem0_filter_empty(self):
        logger.info("MemoryFilter: to mem0 filter empty")
        f = MemoryFilter()
        d = f.to_mem0_filter()
        assert d == {}
