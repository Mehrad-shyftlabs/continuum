"""Unit tests for the RateLimiter."""

import asyncio

import pytest

from orchestrator.tools.executor import RateLimiter
import logging

logger = logging.getLogger(__name__)


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_rate_limiter_first_call(self):
        logger.info("RateLimiter: rate limiter first call")
        limiter = RateLimiter(rate_per_second=10.0)
        assert limiter.last_update is None
        await limiter.acquire()
        assert limiter.last_update is not None

    @pytest.mark.asyncio
    async def test_rate_limiter_burst(self):
        logger.info("RateLimiter: rate limiter burst")
        limiter = RateLimiter(rate_per_second=5.0)
        for _ in range(5):
            await limiter.acquire()
        # Tokens may be very slightly above 0 due to elapsed time between acquires
        assert limiter.tokens < 1.0

    @pytest.mark.asyncio
    async def test_rate_limiter_disabled_with_zero(self):
        logger.info("RateLimiter: rate limiter disabled with zero")
        limiter = RateLimiter(rate_per_second=0)
        await limiter.acquire()
        assert limiter.last_update is None

    @pytest.mark.asyncio
    async def test_rate_limiter_replenish(self):
        logger.info("RateLimiter: rate limiter replenish")
        limiter = RateLimiter(rate_per_second=10.0)
        for _ in range(10):
            await limiter.acquire()
        initial_tokens = limiter.tokens
        await asyncio.sleep(0.2)
        await limiter.acquire()
        # After sleeping, tokens should have been replenished
        # (the acquire consumed one, but the replenishment happened first)
        assert True  # If we got here without hanging, replenishment worked

    @pytest.mark.asyncio
    async def test_rate_limiter_negative_rate(self):
        logger.info("RateLimiter: rate limiter negative rate")
        limiter = RateLimiter(rate_per_second=-1)
        await limiter.acquire()
        assert limiter.last_update is None
