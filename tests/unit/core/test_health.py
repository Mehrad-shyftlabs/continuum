"""Unit tests for health check module."""

import pytest

from orchestrator.core.health import (
    HealthCheck,
    HealthCheckResult,
    HealthStatus,
    OverallHealthResult,
    get_health_checker,
)
import logging

logger = logging.getLogger(__name__)


class TestHealthStatus:
    def test_values(self):
        logger.info("HealthStatus: values")
        assert HealthStatus.HEALTHY == "healthy"
        assert HealthStatus.UNHEALTHY == "unhealthy"
        assert HealthStatus.DEGRADED == "degraded"
        assert HealthStatus.UNKNOWN == "unknown"


class TestHealthCheckResult:
    def test_to_dict(self):
        logger.info("HealthCheckResult: to dict")
        r = HealthCheckResult(name="redis", status=HealthStatus.HEALTHY, message="ok", latency_ms=5.0)
        d = r.to_dict()
        assert d["name"] == "redis"
        assert d["status"] == "healthy"
        assert d["latency_ms"] == 5.0

    def test_defaults(self):
        logger.info("HealthCheckResult: defaults")
        r = HealthCheckResult(name="test", status=HealthStatus.UNKNOWN)
        assert r.message == ""
        assert r.latency_ms == 0.0


class TestOverallHealthResult:
    def test_to_dict(self):
        logger.info("OverallHealthResult: to dict")
        checks = [
            HealthCheckResult(name="redis", status=HealthStatus.HEALTHY),
            HealthCheckResult(name="qdrant", status=HealthStatus.UNHEALTHY),
        ]
        r = OverallHealthResult(status=HealthStatus.DEGRADED, checks=checks, total_latency_ms=10.0)
        d = r.to_dict()
        assert d["status"] == "degraded"
        assert "redis" in d["checks"]
        assert "qdrant" in d["checks"]


class TestHealthChecker:
    def test_get_health_checker(self):
        logger.info("HealthChecker: get health checker")
        checker = get_health_checker()
        assert checker is not None

    def test_health_check_init(self):
        logger.info("HealthChecker: health check init")
        hc = HealthCheck()
        assert isinstance(hc, HealthCheck)
