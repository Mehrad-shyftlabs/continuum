"""Unit tests for tools types."""

import pytest

from orchestrator.tools.types import (
    MCPToolArtifact,
    RunArtifacts,
    ToolContextState,
)
import logging

logger = logging.getLogger(__name__)


class TestToolContextState:
    def test_get_set(self):
        logger.info("ToolContextState: get set")
        state = ToolContextState()
        state.set("ns", "key", "val")
        assert state.get("ns", "key") == "val"

    def test_get_missing(self):
        logger.info("ToolContextState: get missing")
        state = ToolContextState()
        assert state.get("ns", "missing") is None

    def test_is_empty(self):
        logger.info("ToolContextState: is empty")
        state = ToolContextState()
        assert state.is_empty() is True
        state.set("ns", "k", "v")
        assert state.is_empty() is False

    def test_to_dict(self):
        logger.info("ToolContextState: to dict")
        state = ToolContextState()
        state.set("ns1", "k1", "v1")
        d = state.to_dict()
        assert isinstance(d, dict)

    def test_get_all_namespaces(self):
        logger.info("ToolContextState: get all namespaces")
        state = ToolContextState()
        state.set("ns1", "k", "v")
        state.set("ns2", "k", "v")
        ns = state.get_all_namespaces()
        assert "ns1" in ns
        assert "ns2" in ns


class TestMCPToolArtifact:
    def test_creation(self):
        logger.info("MCPToolArtifact: creation")
        a = MCPToolArtifact(tool_name="get_cart", server_name="petco-mcp", meta={"k": "v"})
        assert a.tool_name == "get_cart"
        assert a.server_name == "petco-mcp"
        assert a.meta["k"] == "v"


class TestRunArtifacts:
    def test_add_and_clear(self):
        logger.info("RunArtifacts: add and clear")
        ra = RunArtifacts()
        assert ra.is_empty() is True
        a = MCPToolArtifact(tool_name="t", server_name="s")
        ra.add_artifact(a)
        assert ra.is_empty() is False
        d = ra.to_dict()
        assert len(d["tool_artifacts"]) == 1
        ra.clear()
        assert ra.is_empty() is True
