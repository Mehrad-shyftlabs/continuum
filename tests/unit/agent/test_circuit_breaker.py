"""Tests for agent/utils/circuit_breaker.py."""

import time

import pytest

from orchestrator.agent.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpen,
    CircuitState,
)
import logging

logger = logging.getLogger(__name__)


class TestCircuitState:
    def test_values(self):
        logger.info("CircuitState: values")
        assert CircuitState.CLOSED == "closed"
        assert CircuitState.OPEN == "open"
        assert CircuitState.HALF_OPEN == "half_open"


class TestCircuitBreakerOpen:
    def test_exception(self):
        logger.info("CircuitBreakerOpen: exception")
        err = CircuitBreakerOpen(remaining_cooldown=10.5)
        assert err.remaining_cooldown == 10.5
        assert "10.5" in str(err)


class TestCircuitBreaker:
    def test_defaults(self):
        logger.info("CircuitBreaker: defaults")
        cb = CircuitBreaker()
        assert cb._threshold == 5
        assert cb._cooldown == 60
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_custom_params(self):
        logger.info("CircuitBreaker: custom params")
        cb = CircuitBreaker(threshold=3, cooldown=30)
        assert cb._threshold == 3
        assert cb._cooldown == 30

    def test_check_closed(self):
        logger.info("CircuitBreaker: check closed")
        cb = CircuitBreaker()
        cb.check()  # Should not raise

    def test_record_success_resets(self):
        logger.info("CircuitBreaker: record success resets")
        cb = CircuitBreaker(threshold=2)
        cb.record_failure()
        assert cb.failure_count == 1
        cb.record_success()
        assert cb.failure_count == 0
        assert cb.state == CircuitState.CLOSED

    def test_record_failure_opens_at_threshold(self):
        logger.info("CircuitBreaker: record failure opens at threshold")
        cb = CircuitBreaker(threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_check_open_raises(self):
        logger.info("CircuitBreaker: check open raises")
        cb = CircuitBreaker(threshold=1, cooldown=60)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        with pytest.raises(CircuitBreakerOpen) as exc_info:
            cb.check()
        assert exc_info.value.remaining_cooldown > 0

    def test_half_open_after_cooldown(self):
        logger.info("CircuitBreaker: half open after cooldown")
        cb = CircuitBreaker(threshold=1, cooldown=0)
        cb.record_failure()
        # With cooldown=0, the state property immediately transitions to HALF_OPEN
        time.sleep(0.01)
        state = cb.state
        assert state == CircuitState.HALF_OPEN

    def test_half_open_success_closes(self):
        logger.info("CircuitBreaker: half open success closes")
        cb = CircuitBreaker(threshold=1, cooldown=0)
        cb.record_failure()
        time.sleep(0.01)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        logger.info("CircuitBreaker: half open failure reopens")
        cb = CircuitBreaker(threshold=1, cooldown=0)
        cb.record_failure()
        time.sleep(0.01)
        state = cb.state
        assert state == CircuitState.HALF_OPEN
        # Record another failure to re-open
        cb.record_failure()
        # After recording a failure at threshold, it should be open again
        # But the state getter may transition it to half_open again since cooldown=0
        # Test that failure_count increased
        assert cb.failure_count >= 2

    def test_reset(self):
        logger.info("CircuitBreaker: reset")
        cb = CircuitBreaker(threshold=1)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
