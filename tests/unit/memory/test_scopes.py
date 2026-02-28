"""Unit tests for memory scopes."""

import pytest

from orchestrator.memory.scopes import (
    MemoryScope,
    ScopeDefinition,
    get_scope_definition,
    is_scope_registered,
    list_scopes,
    register_scope,
)
import logging

logger = logging.getLogger(__name__)


class TestScopeRegistry:
    def test_list_scopes(self):
        logger.info("ScopeRegistry: list scopes")
        scopes = list_scopes()
        assert "user" in scopes
        assert "agent" in scopes
        assert "run" in scopes
        assert "shared" in scopes

    def test_is_scope_registered(self):
        logger.info("ScopeRegistry: is scope registered")
        assert is_scope_registered("user") is True
        assert is_scope_registered("nonexistent") is False

    def test_get_scope_definition(self):
        logger.info("ScopeRegistry: get scope definition")
        sd = get_scope_definition("user")
        assert isinstance(sd, ScopeDefinition)
        assert sd.required_field == "user_id"

    def test_get_scope_definition_unknown(self):
        logger.info("ScopeRegistry: get scope definition unknown")
        with pytest.raises(ValueError, match="Unknown scope type"):
            get_scope_definition("nonexistent")

    def test_register_custom_scope(self):
        logger.info("ScopeRegistry: register custom scope")
        register_scope(name="team", required_field="team_id", description="Team scope")
        assert is_scope_registered("team") is True
        sd = get_scope_definition("team")
        assert sd.required_field == "team_id"


class TestMemoryScope:
    def test_user_scope(self):
        logger.info("MemoryScope: user scope")
        scope = MemoryScope.user("u1")
        ids = scope.to_identifiers()
        assert ids["user_id"] == "u1"

    def test_agent_scope(self):
        logger.info("MemoryScope: agent scope")
        scope = MemoryScope.agent("a1")
        ids = scope.to_identifiers()
        assert ids["agent_id"] == "a1"

    def test_run_scope(self):
        logger.info("MemoryScope: run scope")
        scope = MemoryScope.run("r1")
        ids = scope.to_identifiers()
        assert ids["run_id"] == "r1"

    def test_shared_scope(self):
        logger.info("MemoryScope: shared scope")
        scope = MemoryScope.shared()
        ids = scope.to_identifiers()
        assert "agent_id" in ids

    def test_from_isolation_mode_user(self):
        logger.info("MemoryScope: from isolation mode user")
        scope = MemoryScope.from_isolation_mode("user", user_id="u1")
        assert scope is not None
        ids = scope.to_identifiers()
        assert ids["user_id"] == "u1"

    def test_from_isolation_mode_missing_field(self):
        logger.info("MemoryScope: from isolation mode missing field")
        with pytest.raises(ValueError):
            MemoryScope.from_isolation_mode("user")
