"""Unit tests for memory exceptions."""

import pytest

from orchestrator.memory.exceptions import (
    MemoryAddError,
    MemoryConfigurationError,
    MemoryConnectionError,
    MemoryDeleteError,
    MemoryError,
    MemoryIdentifierError,
    MemoryNotEnabledError,
    MemorySearchError,
    MemoryUpdateError,
)
import logging

logger = logging.getLogger(__name__)


class TestMemoryExceptions:
    def test_memory_error(self):
        logger.info("MemoryExceptions: memory error")
        err = MemoryError("test", should_report=False)
        assert isinstance(err, Exception)

    def test_memory_not_enabled(self):
        logger.info("MemoryExceptions: memory not enabled")
        err = MemoryNotEnabledError("not enabled")
        assert isinstance(err, MemoryError)

    def test_memory_identifier_error(self):
        logger.info("MemoryExceptions: memory identifier error")
        err = MemoryIdentifierError("bad id")
        assert isinstance(err, MemoryError)

    def test_memory_connection_error(self):
        logger.info("MemoryExceptions: memory connection error")
        err = MemoryConnectionError("connection failed")
        assert isinstance(err, MemoryError)

    def test_memory_search_error(self):
        logger.info("MemoryExceptions: memory search error")
        err = MemorySearchError("search failed")
        assert isinstance(err, MemoryError)

    def test_memory_add_error(self):
        logger.info("MemoryExceptions: memory add error")
        err = MemoryAddError("add failed")
        assert isinstance(err, MemoryError)

    def test_memory_delete_error(self):
        logger.info("MemoryExceptions: memory delete error")
        err = MemoryDeleteError("delete failed")
        assert isinstance(err, MemoryError)

    def test_memory_update_error(self):
        logger.info("MemoryExceptions: memory update error")
        err = MemoryUpdateError("update failed")
        assert isinstance(err, MemoryError)

    def test_memory_configuration_error(self):
        logger.info("MemoryExceptions: memory configuration error")
        err = MemoryConfigurationError("bad config")
        assert isinstance(err, MemoryError)
