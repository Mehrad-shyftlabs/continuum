"""Unit tests for lifecycle module."""

import pytest

from orchestrator.core.lifecycle import (
    ConfigurationError as LifecycleConfigError,
    validate_configuration,
)
import logging

logger = logging.getLogger(__name__)


class TestValidateConfiguration:
    def test_validate_configuration_returns_errors_and_warnings(self):
        logger.info("ValidateConfiguration: validate configuration returns errors and warnings")
        errors, warnings = validate_configuration()
        assert isinstance(errors, list)
        assert isinstance(warnings, list)

    def test_configuration_error_str(self):
        logger.info("ValidateConfiguration: configuration error str")
        err = LifecycleConfigError(field="API_KEY", message="Missing", severity="error")
        s = str(err)
        assert "API_KEY" in s
        assert "Missing" in s
