"""Unit tests for the circuit breaker."""

import time

import pytest

from orchestrator.agent.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpen,
    CircuitState,
)
import logging

logger = logging.getLogger(__name__)


class TestCircuitBreaker:
    def test_circuit_breaker_starts_closed(self):
        logger.info("CircuitBreaker: circuit breaker starts closed")
        cb = CircuitBreaker(threshold=3, cooldown=10)
        assert cb.state == CircuitState.CLOSED

    def test_circuit_breaker_opens_after_threshold(self):
        logger.info("CircuitBreaker: circuit breaker opens after threshold")
        cb = CircuitBreaker(threshold=3, cooldown=60)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_circuit_breaker_open_raises(self):
        logger.info("CircuitBreaker: circuit breaker open raises")
        cb = CircuitBreaker(threshold=2, cooldown=60)
        cb.record_failure()
        cb.record_failure()
        with pytest.raises(CircuitBreakerOpen):
            cb.check()

    def test_circuit_breaker_resets_after_cooldown(self):
        logger.info("CircuitBreaker: circuit breaker resets after cooldown")
        cb = CircuitBreaker(threshold=2, cooldown=0)
        cb.record_failure()
        cb.record_failure()
        # With cooldown=0, state transitions immediately to HALF_OPEN on check
        time.sleep(0.01)
        assert cb.state == CircuitState.HALF_OPEN
        cb.check()  # Should not raise in HALF_OPEN state

    def test_circuit_breaker_allows_after_success(self):
        logger.info("CircuitBreaker: circuit breaker allows after success")
        cb = CircuitBreaker(threshold=2, cooldown=0)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.01)
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_circuit_breaker_success_resets_count(self):
        logger.info("CircuitBreaker: circuit breaker success resets count")
        cb = CircuitBreaker(threshold=5, cooldown=60)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.failure_count == 0
        assert cb.state == CircuitState.CLOSED

    def test_circuit_breaker_manual_reset(self):
        logger.info("CircuitBreaker: circuit breaker manual reset")
        cb = CircuitBreaker(threshold=2, cooldown=60)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_circuit_breaker_open_remaining_cooldown(self):
        logger.info("CircuitBreaker: circuit breaker open remaining cooldown")
        cb = CircuitBreaker(threshold=1, cooldown=60)
        cb.record_failure()
        with pytest.raises(CircuitBreakerOpen) as exc_info:
            cb.check()
        assert exc_info.value.remaining_cooldown > 0
