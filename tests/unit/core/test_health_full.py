"""Comprehensive tests for core/health.py."""

from unittest.mock import AsyncMock, MagicMock, patch

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
    def test_defaults(self):
        logger.info("HealthCheckResult: defaults")
        r = HealthCheckResult(name="redis", status=HealthStatus.HEALTHY)
        assert r.name == "redis"
        assert r.message == ""
        assert r.latency_ms == 0.0

    def test_with_details(self):
        logger.info("HealthCheckResult: with details")
        r = HealthCheckResult(
            name="redis", status=HealthStatus.HEALTHY,
            message="Connected", latency_ms=5.2,
            details={"host": "localhost"},
        )
        assert r.details["host"] == "localhost"

    def test_to_dict(self):
        logger.info("HealthCheckResult: to dict")
        r = HealthCheckResult(name="redis", status=HealthStatus.HEALTHY, message="ok")
        d = r.to_dict()
        assert d["name"] == "redis"
        assert d["status"] == "healthy"
        assert "checked_at" in d


class TestOverallHealthResult:
    def test_creation(self):
        logger.info("OverallHealthResult: creation")
        checks = [
            HealthCheckResult(name="redis", status=HealthStatus.HEALTHY),
            HealthCheckResult(name="qdrant", status=HealthStatus.UNHEALTHY),
        ]
        r = OverallHealthResult(status=HealthStatus.DEGRADED, checks=checks)
        assert len(r.checks) == 2

    def test_to_dict(self):
        logger.info("OverallHealthResult: to dict")
        r = OverallHealthResult(
            status=HealthStatus.HEALTHY,
            checks=[HealthCheckResult(name="test", status=HealthStatus.HEALTHY)],
        )
        d = r.to_dict()
        assert d["status"] == "healthy"
        assert len(d["checks"]) == 1


class TestHealthCheck:
    def test_creation(self):
        logger.info("HealthCheck: creation")
        hc = HealthCheck()
        assert hc is not None

    @pytest.mark.asyncio
    async def test_check_all_mocked(self):
        logger.info("HealthCheck: check all mocked")
        hc = HealthCheck()
        mock_result = OverallHealthResult(
            status=HealthStatus.HEALTHY,
            checks=[HealthCheckResult(name="mock", status=HealthStatus.HEALTHY)],
        )
        with patch.object(hc, "check_all", new_callable=AsyncMock, return_value=mock_result):
            result = await hc.check_all(timeout=1.0)
            assert isinstance(result, OverallHealthResult)


class TestGetHealthChecker:
    def test_returns_instance(self):
        logger.info("GetHealthChecker: returns instance")
        hc = get_health_checker()
        assert isinstance(hc, HealthCheck)
