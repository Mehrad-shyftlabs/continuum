"""Unit tests for the pure fork/diff helpers (Phase 4)."""

from __future__ import annotations

from continuum.agent.trace import (
    DecisionStep,
    DecisionTrace,
    StepKind,
    apply_override,
    diff_traces,
)


# --------------------------------------------------------------------------- #
# apply_override
# --------------------------------------------------------------------------- #
def _msgs() -> list[dict]:
    return [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Is my order delayed?"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": '{"status": "shipped"}'},
    ]


class TestApplyOverride:
    def test_none_returns_copy_unchanged(self) -> None:
        msgs = _msgs()
        out = apply_override(msgs, None)
        assert out == msgs
        assert out is not msgs  # never mutate the input

    def test_set_tool_result(self) -> None:
        out = apply_override(
            _msgs(), {"set_tool_result": {"tool_call_id": "c1", "content": '{"status": "DELAYED"}'}}
        )
        tool_msg = next(m for m in out if m.get("role") == "tool")
        assert tool_msg["content"] == '{"status": "DELAYED"}'

    def test_replace_last_user(self) -> None:
        out = apply_override(_msgs(), {"replace_last_user": "Is order #999 delayed?"})
        users = [m for m in out if m["role"] == "user"]
        assert users[-1]["content"] == "Is order #999 delayed?"

    def test_replace_system(self) -> None:
        out = apply_override(_msgs(), {"system": "You are terse."})
        assert out[0] == {"role": "system", "content": "You are terse."}

    def test_prepend_system_when_absent(self) -> None:
        out = apply_override([{"role": "user", "content": "hi"}], {"system": "Be nice."})
        assert out[0] == {"role": "system", "content": "Be nice."}

    def test_append(self) -> None:
        out = apply_override(_msgs(), {"append": {"role": "user", "content": "also check carrier"}})
        assert out[-1] == {"role": "user", "content": "also check carrier"}

    def test_does_not_mutate_input(self) -> None:
        msgs = _msgs()
        apply_override(
            msgs,
            {"replace_last_user": "x", "set_tool_result": {"tool_call_id": "c1", "content": "y"}},
        )
        assert msgs[1]["content"] == "Is my order delayed?"  # original intact
        assert msgs[3]["content"] == '{"status": "shipped"}'


# --------------------------------------------------------------------------- #
# diff_traces
# --------------------------------------------------------------------------- #
def _trace(run_id: str, final: str, tool_out: str) -> DecisionTrace:
    t = DecisionTrace(run_id=run_id, root_agent="a", final_response=final)
    t.add(DecisionStep(step_id="s1", kind=StepKind.LLM_CALL, agent_name="a", decision="tool_call"))
    t.add(
        DecisionStep(
            step_id="s2",
            kind=StepKind.TOOL_CALL,
            agent_name="a",
            decision="call lookup_order",
            output=tool_out,
        )
    )
    t.add(
        DecisionStep(
            step_id="s3",
            kind=StepKind.LLM_CALL,
            agent_name="a",
            decision="final_answer",
            output=final,
        )
    )
    return t


class TestDiffTraces:
    def test_detects_changed_final_response(self) -> None:
        a = _trace("A", "not delayed", '{"status":"shipped"}')
        b = _trace("B", "delayed 2 days", '{"status":"DELAYED"}')
        d = diff_traces(a, b)
        assert d["final_response"]["changed"] is True
        assert d["final_response"]["before"] == "not delayed"
        assert d["final_response"]["after"] == "delayed 2 days"

    def test_identical_traces_show_no_changes(self) -> None:
        a = _trace("A", "same", "out")
        b = _trace("B", "same", "out")
        d = diff_traces(a, b)
        assert d["final_response"]["changed"] is False
        assert d["steps_changed"] == 0

    def test_pinpoints_diverging_step(self) -> None:
        a = _trace("A", "not delayed", '{"status":"shipped"}')
        b = _trace("B", "delayed 2 days", '{"status":"DELAYED"}')
        d = diff_traces(a, b)
        changed_indices = {sd["index"] for sd in d["step_diffs"]}
        assert 1 in changed_indices  # the tool-output step diverged
        assert 2 in changed_indices  # and the final answer
        tool_diff = next(sd for sd in d["step_diffs"] if sd["index"] == 1)
        assert tool_diff["before"]["output"] == '{"status":"shipped"}'
        assert tool_diff["after"]["output"] == '{"status":"DELAYED"}'

    def test_added_and_removed_steps(self) -> None:
        a = _trace("A", "x", "o")
        b = _trace("B", "x", "o")
        b.add(DecisionStep(step_id="s4", kind=StepKind.LLM_CALL, agent_name="a"))
        d = diff_traces(a, b)
        assert any(sd["kind"] == "added" and sd["index"] == 3 for sd in d["step_diffs"])
