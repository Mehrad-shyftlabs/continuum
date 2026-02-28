"""Unit tests for RunState/RunContext handoff_chain type consistency (Issue 8)."""

import pytest

from orchestrator.agent.types import (
    HandoffData,
    RunContext,
    RunState,
    generate_handoff_id,
)
import logging

logger = logging.getLogger(__name__)


class TestRunStateHandoffChain:
    """Tests for RunState handoff_chain serialization and conversion."""

    def test_run_state_handoff_chain_serialization(self):
        logger.info("RunStateHandoffChain: run state handoff chain serialization")
        state = RunState(run_id="run_1")
        handoff = HandoffData(
            handoff_id=generate_handoff_id(),
            from_agent="agent-a",
            to_agent="agent-b",
            reason="needs specialist",
        )
        state.add_handoff(handoff)

        d = state.to_dict()
        restored = RunState.from_dict(d)
        assert len(restored.handoff_chain) == 1
        assert restored.handoff_chain[0]["from_agent"] == "agent-a"
        assert restored.handoff_chain[0]["to_agent"] == "agent-b"

    def test_run_state_from_dict_with_handoff_data(self):
        logger.info("RunStateHandoffChain: run state from dict with handoff data")
        handoff = HandoffData(
            handoff_id="handoff_1",
            from_agent="agent-a",
            to_agent="agent-b",
            reason="test",
        )
        data = {
            "run_id": "run_1",
            "handoff_chain": [handoff.to_dict()],
        }
        state = RunState.from_dict(data)
        assert len(state.handoff_chain) == 1
        assert state.handoff_chain[0]["to_agent"] == "agent-b"

    def test_run_state_get_handoff_chain_as_data(self):
        logger.info("RunStateHandoffChain: run state get handoff chain as data")
        state = RunState(run_id="run_1")
        handoff = HandoffData(
            handoff_id="handoff_1",
            from_agent="agent-a",
            to_agent="agent-b",
            reason="test",
        )
        state.add_handoff(handoff)

        handoff_data_list = state.get_handoff_chain_as_data()
        assert len(handoff_data_list) == 1
        assert isinstance(handoff_data_list[0], HandoffData)
        assert handoff_data_list[0].from_agent == "agent-a"
        assert handoff_data_list[0].to_agent == "agent-b"

    def test_run_state_add_handoff(self):
        logger.info("RunStateHandoffChain: run state add handoff")
        state = RunState(run_id="run_1")
        h1 = HandoffData(handoff_id="h1", from_agent="a", to_agent="b", reason="r1")
        h2 = HandoffData(handoff_id="h2", from_agent="b", to_agent="c", reason="r2")
        state.add_handoff(h1)
        state.add_handoff(h2)
        assert len(state.handoff_chain) == 2


class TestRunContextHandoffChainConsistency:
    """Tests for RunContext handoff_chain type consistency."""

    def test_run_context_handoff_chain_type_consistency(self):
        """RunContext.handoff_chain stores HandoffData objects."""
        logger.info("RunContext.handoff_chain stores HandoffData objects")
        ctx = RunContext(run_id="run_1")
        handoff = HandoffData(
            handoff_id="handoff_1",
            from_agent="agent-a",
            to_agent="agent-b",
            reason="test",
        )
        ctx.handoff_chain.append(handoff)
        assert isinstance(ctx.handoff_chain[0], HandoffData)

        d = ctx.to_dict()
        assert d["handoff_chain"][0]["from_agent"] == "agent-a"

    def test_run_state_and_context_handoff_interop(self):
        """Handoffs can be created in RunContext and stored in RunState."""
        logger.info("Handoffs can be created in RunContext and stored in RunState")
        ctx = RunContext(run_id="run_1")
        handoff = HandoffData(
            handoff_id="handoff_1",
            from_agent="agent-a",
            to_agent="agent-b",
            reason="test",
        )
        ctx.handoff_chain.append(handoff)

        state = RunState(run_id="run_1")
        for h in ctx.handoff_chain:
            state.add_handoff(h)

        assert len(state.handoff_chain) == 1
        recovered = state.get_handoff_chain_as_data()
        assert recovered[0].to_agent == "agent-b"
