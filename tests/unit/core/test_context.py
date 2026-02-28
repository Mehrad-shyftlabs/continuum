"""Unit tests for core/context.py - ExecutionContext, ContextManager, ContextScope."""

import pytest

from orchestrator.core.context import (
    ContextManager,
    ContextScope,
    ContextToken,
    ExecutionContext,
    get_agent_name,
    get_context_manager,
    get_current_context,
    get_run_id,
    get_session_id,
    get_span_id,
    get_trace_id,
    get_user_id,
)
import logging

logger = logging.getLogger(__name__)


class TestExecutionContext:
    def test_defaults(self):
        logger.info("ExecutionContext: defaults")
        ctx = ExecutionContext()
        assert ctx.trace_id is None
        assert ctx.span_id is None
        assert ctx.user_id is None
        assert ctx.session_id is None
        assert ctx.run_id is None
        assert ctx.agent_name is None
        assert ctx.metadata == {}
        assert ctx.created_at is not None

    def test_with_values(self):
        logger.info("ExecutionContext: with values")
        ctx = ExecutionContext(
            trace_id="t1", span_id="s1", user_id="u1",
            session_id="sess1", run_id="r1", agent_name="a1",
            metadata={"key": "val"},
        )
        assert ctx.trace_id == "t1"
        assert ctx.user_id == "u1"

    def test_to_dict(self):
        logger.info("ExecutionContext: to dict")
        ctx = ExecutionContext(trace_id="t1", user_id="u1")
        d = ctx.to_dict()
        assert d["trace_id"] == "t1"
        assert d["user_id"] == "u1"
        assert "created_at" in d

    def test_with_span(self):
        logger.info("ExecutionContext: with span")
        ctx = ExecutionContext(trace_id="t1", user_id="u1", metadata={"k": "v"})
        new_ctx = ctx.with_span("new-span")
        assert new_ctx.span_id == "new-span"
        assert new_ctx.trace_id == "t1"
        assert new_ctx.user_id == "u1"
        assert new_ctx.metadata == {"k": "v"}

    def test_with_agent(self):
        logger.info("ExecutionContext: with agent")
        ctx = ExecutionContext(trace_id="t1", agent_name="old")
        new_ctx = ctx.with_agent("new-agent")
        assert new_ctx.agent_name == "new-agent"
        assert new_ctx.trace_id == "t1"


class TestContextToken:
    def test_defaults(self):
        logger.info("ContextToken: defaults")
        token = ContextToken()
        assert token.trace_id is None
        assert token.span_id is None

    def test_with_values(self):
        logger.info("ContextToken: with values")
        token = ContextToken(trace_id="t1", span_id="s1")
        assert token.trace_id == "t1"


class TestContextManager:
    def setup_method(self):
        self.cm = ContextManager()
        self.cm.clear_context()

    def test_get_current_default(self):
        logger.info("ContextManager: get current default")
        ctx = self.cm.get_current()
        assert ctx.trace_id is None

    def test_set_and_get_context(self):
        logger.info("ContextManager: set and get context")
        token = self.cm.set_context(trace_id="t1", user_id="u1")
        ctx = self.cm.get_current()
        assert ctx.trace_id == "t1"
        assert ctx.user_id == "u1"
        self.cm.restore_context(token)

    def test_restore_context(self):
        logger.info("ContextManager: restore context")
        self.cm.set_context(trace_id="original")
        token = self.cm.set_context(trace_id="new")
        assert self.cm.get_current().trace_id == "new"
        self.cm.restore_context(token)
        assert self.cm.get_current().trace_id == "original"

    def test_clear_context(self):
        logger.info("ContextManager: clear context")
        self.cm.set_context(trace_id="t1", user_id="u1", run_id="r1")
        self.cm.clear_context()
        ctx = self.cm.get_current()
        assert ctx.trace_id is None
        assert ctx.user_id is None

    def test_context_scope(self):
        logger.info("ContextManager: context scope")
        with self.cm.context(trace_id="t1", user_id="u1") as ctx:
            assert ctx.trace_id == "t1"
            assert ctx.user_id == "u1"
        assert self.cm.get_current().trace_id is None

    def test_nested_context_scopes(self):
        logger.info("ContextManager: nested context scopes")
        with self.cm.context(trace_id="t1") as outer:
            assert outer.trace_id == "t1"
            with self.cm.context(span_id="s1") as inner:
                assert inner.span_id == "s1"
                assert inner.trace_id == "t1"
            assert self.cm.get_current().span_id is None

    def test_span_context(self):
        logger.info("ContextManager: span context")
        self.cm.set_context(trace_id="t1")
        with self.cm.span_context("span-1") as ctx:
            assert ctx.span_id == "span-1"
            assert ctx.trace_id == "t1"
        self.cm.clear_context()

    def test_agent_context(self):
        logger.info("ContextManager: agent context")
        self.cm.set_context(trace_id="t1")
        with self.cm.agent_context("my-agent") as ctx:
            assert ctx.agent_name == "my-agent"
            assert ctx.trace_id == "t1"
        self.cm.clear_context()

    def test_set_context_partial(self):
        logger.info("ContextManager: set context partial")
        self.cm.set_context(trace_id="t1")
        self.cm.set_context(user_id="u1")
        ctx = self.cm.get_current()
        assert ctx.trace_id == "t1"
        assert ctx.user_id == "u1"
        self.cm.clear_context()


class TestContextScope:
    def test_sync_context_manager(self):
        logger.info("ContextScope: sync context manager")
        cm = ContextManager()
        cm.clear_context()
        scope = ContextScope(manager=cm, trace_id="t1")
        with scope as ctx:
            assert ctx.trace_id == "t1"
        assert cm.get_current().trace_id is None

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        logger.info("ContextScope: async context manager")
        cm = ContextManager()
        cm.clear_context()
        scope = ContextScope(manager=cm, trace_id="t2", user_id="u2")
        async with scope as ctx:
            assert ctx.trace_id == "t2"
            assert ctx.user_id == "u2"
        assert cm.get_current().trace_id is None


class TestGlobalFunctions:
    def setup_method(self):
        cm = get_context_manager()
        cm.clear_context()

    def test_get_context_manager_singleton(self):
        logger.info("GlobalFunctions: get context manager singleton")
        cm1 = get_context_manager()
        cm2 = get_context_manager()
        assert cm1 is cm2

    def test_get_current_context(self):
        logger.info("GlobalFunctions: get current context")
        ctx = get_current_context()
        assert isinstance(ctx, ExecutionContext)

    def test_convenience_getters(self):
        logger.info("GlobalFunctions: convenience getters")
        cm = get_context_manager()
        cm.set_context(
            trace_id="t1", span_id="s1", user_id="u1",
            session_id="sess1", run_id="r1", agent_name="a1",
        )
        assert get_trace_id() == "t1"
        assert get_span_id() == "s1"
        assert get_user_id() == "u1"
        assert get_session_id() == "sess1"
        assert get_run_id() == "r1"
        assert get_agent_name() == "a1"
        cm.clear_context()
