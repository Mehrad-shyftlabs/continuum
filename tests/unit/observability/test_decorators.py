"""Unit tests for observability decorators."""

from unittest.mock import MagicMock, patch

import pytest

from orchestrator.observability.decorators import ObservationContext, observe
import logging

logger = logging.getLogger(__name__)


class TestObserveDecorator:
    def test_observe_sync_function(self):
        logger.info("ObserveDecorator: observe sync function")
        @observe(name="test_fn")
        def my_func(x):
            return x + 1
        result = my_func(5)
        assert result == 6

    @pytest.mark.asyncio
    async def test_observe_async_function(self):
        logger.info("ObserveDecorator: observe async function")
        @observe(name="test_async_fn")
        async def my_func(x):
            return x * 2
        result = await my_func(5)
        assert result == 10

    def test_observe_handles_exception(self):
        logger.info("ObserveDecorator: observe handles exception")
        @observe(name="test_err")
        def my_func():
            raise ValueError("boom")
        with pytest.raises(ValueError, match="boom"):
            my_func()

    @pytest.mark.asyncio
    async def test_observe_async_handles_exception(self):
        logger.info("ObserveDecorator: observe async handles exception")
        @observe(name="test_async_err")
        async def my_func():
            raise RuntimeError("async boom")
        with pytest.raises(RuntimeError, match="async boom"):
            await my_func()

    def test_observe_captures_output(self):
        logger.info("ObserveDecorator: observe captures output")
        @observe(name="test_output", capture_output=True)
        def my_func():
            return {"result": "ok"}
        result = my_func()
        assert result == {"result": "ok"}

    def test_observe_no_capture(self):
        logger.info("ObserveDecorator: observe no capture")
        @observe(name="test_no_capture", capture_output=False)
        def my_func():
            return "secret"
        result = my_func()
        assert result == "secret"
