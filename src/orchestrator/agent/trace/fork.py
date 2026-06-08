"""
Fork edits — pure functions for the "what-if" half of the feature.

``apply_override`` takes the message array restored from a step checkpoint and
applies a small, well-defined edit before the loop re-runs from that point. Kept
pure (no I/O) so it is trivially testable; :meth:`AgentRunner.fork` wires it to
real execution.

Supported edits (any subset, applied in this order):
* ``system``           — replace (or prepend) the system instruction.
* ``set_tool_result``  — override a recorded tool result by ``tool_call_id``;
                         this is the "what if the tool had returned X?" knob.
* ``replace_last_user``— replace the most recent user message content.
* ``append``           — append a message dict (e.g. an extra instruction).
"""

from __future__ import annotations

import copy
from typing import Any


def apply_override(
    messages: list[dict[str, Any]], override: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Return a new edited copy of ``messages`` (input is never mutated)."""
    msgs = copy.deepcopy(messages)
    if not override:
        return msgs

    if "system" in override:
        text = override["system"]
        if msgs and msgs[0].get("role") == "system":
            msgs[0] = {"role": "system", "content": text}
        else:
            msgs.insert(0, {"role": "system", "content": text})

    if "set_tool_result" in override:
        spec = override["set_tool_result"]
        target_id = spec.get("tool_call_id")
        new_content = spec.get("content")
        for m in msgs:
            if m.get("role") == "tool" and m.get("tool_call_id") == target_id:
                m["content"] = new_content

    if "replace_last_user" in override:
        for m in reversed(msgs):
            if m.get("role") == "user":
                m["content"] = override["replace_last_user"]
                break

    if "append" in override:
        msgs.append(override["append"])

    return msgs
