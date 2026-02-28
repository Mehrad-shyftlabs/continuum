"""Comprehensive tests for observability/decorators.py."""

from unittest.mock import MagicMock, patch

import pytest

from orchestrator.observability.decorators import (
    ObservationContext,
    _get_function_input,
    _serialize_output,
    observe,
    trace_agent,
    trace_tool,
)
from orchestrator.observability.tracing import SpanLevel
import logging

logger = logging.getLogger(__name__)


class TestGetFunctionInput:
    def test_basic(self):
        logger.info("GetFunctionInput: basic")
        def my_func(a, b, c=3):
            pass
        result = _get_function_input(my_func, (1, 2), {"c": 4})
        assert result == {"a": 1, "b": 2, "c": 4}

    def test_defaults(self):
        logger.info("GetFunctionInput: defaults")
        def my_func(a, b=10):
            pass
        result = _get_function_input(my_func, (1,), {})
        assert result == {"a": 1, "b": 10}


class TestSerializeOutput:
    def test_none(self):
        logger.info("SerializeOutput: none")
        assert _serialize_output(None) is None

    def test_primitives(self):
        logger.info("SerializeOutput: primitives")
        assert _serialize_output("hello") == "hello"
        assert _serialize_output(42) == 42
        assert _serialize_output(3.14) == 3.14
        assert _serialize_output(True) is True

    def test_list(self):
        logger.info("SerializeOutput: list")
        result = _serialize_output([1, 2, 3])
        assert result == [1, 2, 3]

    def test_dict(self):
        logger.info("SerializeOutput: dict")
        result = _serialize_output({"a": 1, "b": "two"})
        assert result == {"a": 1, "b": "two"}

    def test_model_dump(self):
        logger.info("SerializeOutput: model dump")
        obj = MagicMock()
        obj.model_dump.return_value = {"key": "val"}
        del obj.__dict__
        result = _serialize_output(obj)
        assert result == {"key": "val"}

    def test_object_with_dict(self):
        logger.info("SerializeOutput: object with dict")
        class Obj:
            def __init__(self):
                self.x = 1
                self._private = 2
        result = _serialize_output(Obj())
        assert result == {"x": 1}

    def test_fallback_to_string(self):
        logger.info("SerializeOutput: fallback to string")
        class NoDict:
            __slots__ = ()
            def __str__(self):
                return "custom-str"
        obj = NoDict()
        result = _serialize_output(obj)
        assert "custom-str" in str(result)


class TestObserveDecorator:
    def test_sync_function(self):
        logger.info("ObserveDecorator: sync function")
        @observe(name="test-sync")
        def my_func(x):
            return x * 2
        result = my_func(5)
        assert result == 10

    def test_sync_function_default_name(self):
        logger.info("ObserveDecorator: sync function default name")
        @observe()
        def my_func(x):
            return x + 1
        result = my_func(5)
        assert result == 6

    def test_sync_function_no_capture(self):
        logger.info("ObserveDecorator: sync function no capture")
        @observe(capture_input=False, capture_output=False)
        def my_func(x):
            return x
        result = my_func(5)
        assert result == 5

    def test_sync_function_exception(self):
        logger.info("ObserveDecorator: sync function exception")
        @observe(name="test-err")
        def my_func():
            raise ValueError("boom")
        with pytest.raises(ValueError):
            my_func()

    @pytest.mark.asyncio
    async def test_async_function(self):
        logger.info("ObserveDecorator: async function")
        @observe(name="test-async")
        async def my_func(x):
            return x * 3
        result = await my_func(5)
        assert result == 15

    @pytest.mark.asyncio
    async def test_async_function_exception(self):
        logger.info("ObserveDecorator: async function exception")
        @observe(name="test-async-err")
        async def my_func():
            raise RuntimeError("async boom")
        with pytest.raises(RuntimeError):
            await my_func()

    def test_with_metadata(self):
        logger.info("ObserveDecorator: with metadata")
        @observe(name="test", metadata={"env": "test"})
        def my_func():
            return "ok"
        assert my_func() == "ok"


class TestTraceToolDecorator:
    def test_sync(self):
        logger.info("TraceToolDecorator: sync")
        @trace_tool(name="my-tool")
        def tool_func(query):
            return {"result": query}
        result = tool_func("test")
        assert result["result"] == "test"

    def test_with_tool_type(self):
        logger.info("TraceToolDecorator: with tool type")
        @trace_tool(name="api-tool", tool_type="api")
        def api_func():
            return "data"
        assert api_func() == "data"

    @pytest.mark.asyncio
    async def test_async(self):
        logger.info("TraceToolDecorator: async")
        @trace_tool(name="async-tool")
        async def async_tool(x):
            return x
        result = await async_tool(42)
        assert result == 42


class TestTraceAgentDecorator:
    def test_sync_no_trace_context(self):
        logger.info("TraceAgentDecorator: sync no trace context")
        with patch("orchestrator.observability.decorators.get_current_trace_id", return_value=None):
            @trace_agent(name="agent1")
            def agent_func(msg):
                return f"response: {msg}"
            result = agent_func("hello")
            assert result == "response: hello"

    def test_sync_with_trace_context(self):
        logger.info("TraceAgentDecorator: sync with trace context")
        with patch("orchestrator.observability.decorators.get_current_trace_id", return_value="trace-123"):
            @trace_agent(name="agent1")
            def agent_func(msg):
                return f"response: {msg}"
            result = agent_func("hello")
            assert result == "response: hello"

    @pytest.mark.asyncio
    async def test_async_with_trace_context(self):
        logger.info("TraceAgentDecorator: async with trace context")
        with patch("orchestrator.observability.decorators.get_current_trace_id", return_value="trace-123"):
            @trace_agent(name="agent-async")
            async def agent_func(msg):
                return f"response: {msg}"
            result = await agent_func("hello")
            assert result == "response: hello"

    @pytest.mark.asyncio
    async def test_async_no_trace_create_new(self):
        logger.info("TraceAgentDecorator: async no trace create new")
        mock_pm = MagicMock()
        mock_pm.is_enabled = True
        mock_pm.trace.return_value = MagicMock(id="new-trace")
        mock_pm.flush.return_value = None

        with patch("orchestrator.observability.decorators.get_current_trace_id", return_value=None):
            with patch("orchestrator.observability.provider_manager.get_provider_manager", return_value=mock_pm):
                @trace_agent(name="agent-new", create_new_trace=True)
                async def agent_func(msg):
                    return f"response: {msg}"
                result = await agent_func("hello")
                assert result == "response: hello"

    def test_sync_exception_with_trace(self):
        logger.info("TraceAgentDecorator: sync exception with trace")
        with patch("orchestrator.observability.decorators.get_current_trace_id", return_value="trace-123"):
            @trace_agent(name="agent-err")
            def agent_func():
                raise ValueError("agent error")
            with pytest.raises(ValueError):
                agent_func()


class TestObservationContext:
    def test_basic(self):
        logger.info("ObservationContext: basic")
        ctx = ObservationContext("test")
        assert ctx.name == "test"
        assert ctx.span_type == "span"

    def test_set_input(self):
        logger.info("ObservationContext: set input")
        ctx = ObservationContext("test")
        ctx.set_input({"query": "hello"})
        assert ctx._input == {"query": "hello"}

    def test_set_output(self):
        logger.info("ObservationContext: set output")
        ctx = ObservationContext("test")
        ctx.set_output({"result": "world"})
        assert ctx._output == {"result": "world"}

    def test_add_metadata(self):
        logger.info("ObservationContext: add metadata")
        ctx = ObservationContext("test")
        ctx.add_metadata({"key": "val"})
        assert ctx._metadata["key"] == "val"

    def test_set_error(self):
        logger.info("ObservationContext: set error")
        ctx = ObservationContext("test")
        ctx.set_error(ValueError("boom"))
        assert ctx._metadata["error"] == "boom"
        assert ctx._metadata["error_type"] == "ValueError"

    def test_context_manager_no_trace(self):
        logger.info("ObservationContext: context manager no trace")
        mock_manager = MagicMock()
        mock_manager.get_current_trace.return_value = None
        mock_manager.get_current_span.return_value = None

        ctx = ObservationContext("test", manager=mock_manager)
        with ctx as c:
            c.set_input({"x": 1})
            c.set_output({"y": 2})
        assert c._span is None

    def test_context_manager_with_trace(self):
        logger.info("ObservationContext: context manager with trace")
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.span.return_value = mock_span
        mock_manager = MagicMock()
        mock_manager.get_current_trace.return_value = mock_trace
        mock_manager.get_current_span.return_value = None

        ctx = ObservationContext("test", manager=mock_manager)
        with ctx as c:
            c.set_output("result")
        mock_span.end.assert_called_once()

    def test_context_manager_with_exception(self):
        logger.info("ObservationContext: context manager with exception")
        mock_span = MagicMock()
        mock_trace = MagicMock()
        mock_trace.span.return_value = mock_span
        mock_manager = MagicMock()
        mock_manager.get_current_trace.return_value = mock_trace
        mock_manager.get_current_span.return_value = None

        ctx = ObservationContext("test", manager=mock_manager)
        with pytest.raises(ValueError):
            with ctx:
                raise ValueError("boom")
        mock_span.end.assert_called_once()

    def test_set_level_with_span(self):
        logger.info("ObservationContext: set level with span")
        ctx = ObservationContext("test")
        mock_span = MagicMock()
        ctx._span = mock_span
        ctx.set_level(SpanLevel.WARNING)
        mock_span.update.assert_called()
