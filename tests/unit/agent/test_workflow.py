"""Tests for agent/workflow/ modules."""

from unittest.mock import MagicMock

import pytest

from orchestrator.agent.base import BaseAgent
import logging

logger = logging.getLogger(__name__)


class TestSequentialAgent:
    def test_import(self):
        logger.info("SequentialAgent: import")
        from orchestrator.agent.workflow.sequential import SequentialAgent
        assert SequentialAgent is not None
        assert issubclass(SequentialAgent, BaseAgent)


class TestLoopAgent:
    def test_import(self):
        logger.info("LoopAgent: import")
        from orchestrator.agent.workflow.loop import LoopAgent
        assert LoopAgent is not None
        assert issubclass(LoopAgent, BaseAgent)


class TestParallelAgent:
    def test_import(self):
        logger.info("ParallelAgent: import")
        from orchestrator.agent.workflow.parallel import ParallelAgent
        assert ParallelAgent is not None
        assert issubclass(ParallelAgent, BaseAgent)


class TestRouterAgent:
    def test_import(self):
        logger.info("RouterAgent: import")
        from orchestrator.agent.workflow.router import RouterAgent

        assert RouterAgent is not None
        assert issubclass(RouterAgent, BaseAgent)
