"""Codex >= 0.149.0 ``additional_tools`` normalization (#3185).

Codex CLI 0.149.0 sends tool definitions as ``input`` items of type
``additional_tools`` instead of a top-level ``tools`` array for models its
capability cache flags (``gpt-5.6-sol``). Without the lift, every tools
consumer (schema compaction, output-shaper stratum, tools token accounting)
sees a tool-less request and records zero tool-schema savings.
"""

from __future__ import annotations

import asyncio
import copy
import json
from typing import Any

from headroom.proxy.handlers.openai import (
    OpenAIHandlerMixin,
    _compact_openai_responses_tools,
    _lift_codex_additional_tools,
    _restore_codex_additional_tools,
)


def _verbose_tool(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": " ".join(["Runs a shell command in the workspace."] * 30),
        "parameters": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "title": name,
            "properties": {
                "command": {
                    "type": "array",
                    "title": "command",
                    "items": {"type": "string"},
                }
            },
            "required": ["command"],
        },
    }


def _codex_0149_payload() -> dict[str, Any]:
    return {
        "model": "gpt-5.6-sol",
        "include": ["reasoning.encrypted_content"],
        "reasoning": {"effort": "low", "context": "all_turns"},
        "tool_choice": "auto",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "do the thing"}],
            },
            {
                "type": "additional_tools",
                "tools": [_verbose_tool("shell"), _verbose_tool("update_plan")],
            },
        ],
    }


def test_lift_moves_additional_tools_to_top_level() -> None:
    payload = _codex_0149_payload()

    lifted = _lift_codex_additional_tools(payload)

    assert lifted == 2
    assert [t["name"] for t in payload["tools"]] == ["shell", "update_plan"]
    # The carrier item is dropped; every other input item survives in order.
    assert [item["type"] for item in payload["input"]] == ["message"]


def test_lift_concatenates_multiple_carrier_items() -> None:
    payload = _codex_0149_payload()
    payload["input"].append({"type": "additional_tools", "tools": [_verbose_tool("view_image")]})

    lifted = _lift_codex_additional_tools(payload)

    assert lifted == 3
    assert [t["name"] for t in payload["tools"]] == ["shell", "update_plan", "view_image"]


def test_lift_is_noop_when_top_level_tools_present() -> None:
    payload = _codex_0149_payload()
    payload["tools"] = [_verbose_tool("shell")]
    before = copy.deepcopy(payload)

    assert _lift_codex_additional_tools(payload) == 0
    assert payload == before


def test_lift_is_noop_without_carrier_items() -> None:
    payload = _codex_0149_payload()
    payload["input"] = [item for item in payload["input"] if item["type"] != "additional_tools"]
    before = copy.deepcopy(payload)

    assert _lift_codex_additional_tools(payload) == 0
    assert payload == before

    assert _lift_codex_additional_tools({"model": "gpt-5.6-sol", "input": "not-a-list"}) == 0
    assert _lift_codex_additional_tools("not-a-dict") == 0  # type: ignore[arg-type]


def test_lift_disabled_by_kill_switch(monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_CODEX_ADDITIONAL_TOOLS_LIFT", "0")
    payload = _codex_0149_payload()
    before = copy.deepcopy(payload)

    assert _lift_codex_additional_tools(payload) == 0
    assert payload == before


def test_lift_logs_with_request_id(caplog) -> None:
    payload = _codex_0149_payload()

    with caplog.at_level("INFO", logger="headroom.proxy"):
        assert _lift_codex_additional_tools(payload, request_id="req_test") == 2

    assert any(
        "req_test" in message and "additional_tools" in message for message in caplog.messages
    )


def test_lift_preserves_empty_carrier_items() -> None:
    payload = _codex_0149_payload()
    payload["input"].append({"type": "additional_tools", "tools": []})

    lifted = _lift_codex_additional_tools(payload)

    # The empty carrier holds no definitions to lift; it is preserved rather
    # than invented into an empty top-level array.
    assert lifted == 2
    assert [item["type"] for item in payload["input"]] == ["message", "additional_tools"]


def test_lifted_tools_reach_schema_compaction() -> None:
    payload = _codex_0149_payload()

    # Without the lift: compaction sees no tools and returns unmodified —
    # the exact production failure.
    _, modified, _, _ = _compact_openai_responses_tools(copy.deepcopy(payload))
    assert modified is False

    _lift_codex_additional_tools(payload)
    compacted, modified, before_bytes, after_bytes = _compact_openai_responses_tools(payload)

    assert modified is True
    assert after_bytes < before_bytes
    # Compaction preserves the invocation shape the model needs.
    assert [t["name"] for t in compacted["tools"]] == ["shell", "update_plan"]


# ---------------------------------------------------------------------------
# Carrier restoration (0.36.3 regression from #3186)
#
# The lift is an internal normalization so the tools consumers engage. It must
# not change the forwarded wire shape: `tools` is a per-request parameter,
# while `additional_tools` is an `input` item and therefore part of the
# conversation transcript. Codex TUI/app-server over WebSocket declares its
# tools once and relies on the transcript for every later turn, so forwarding
# the lifted shape cost the session its whole tool surface after turn one --
# 0.36.3 regressed shell/filesystem access while 0.36.2 worked.
# ---------------------------------------------------------------------------


def _handler(compress=None):  # noqa: ANN001, ANN202
    """A bare mixin with the executor and compressor stubbed out."""
    handler = object.__new__(OpenAIHandlerMixin)

    async def _run_compression(fn, *, timeout):  # noqa: ANN001, ANN202
        return fn()

    def _default_compress(payload, *, model, request_id, **kwargs):  # noqa: ANN001, ANN202
        return (payload, True, 0, [], None, 0, 0, 0, {})

    handler._run_compression_in_executor = _run_compression
    handler._compress_openai_responses_payload = compress or _default_compress
    return handler


def _forward(payload: dict[str, Any], compress=None) -> dict[str, Any]:  # noqa: ANN001
    handler = _handler(compress)
    result = asyncio.run(
        handler._compress_openai_responses_payload_in_executor(
            payload,
            model="gpt-5.6-sol",
            request_id="req-carrier",
        )
    )
    return result[0]


def test_lift_restore_round_trip_is_shape_preserving() -> None:
    payload = _codex_0149_payload()
    before = copy.deepcopy(payload)

    plan: list[dict[str, Any]] = []
    _lift_codex_additional_tools(payload, restore_plan=plan)
    _restore_codex_additional_tools(payload, plan)

    assert payload == before


def test_compressor_sees_tools_but_forwarded_payload_does_not() -> None:
    """The whole point: consumers get top-level tools, the wire keeps the carrier."""
    seen: list[Any] = []

    def _compress(payload, *, model, request_id, **kwargs):  # noqa: ANN001, ANN202
        seen.append(copy.deepcopy(payload.get("tools")))
        return (payload, True, 0, [], None, 0, 0, 0, {})

    payload = _codex_0149_payload()
    before = copy.deepcopy(payload)

    forwarded = _forward(payload, _compress)

    # The savings fix (#3185) still holds: compaction saw real tools.
    assert [t["name"] for t in seen[0]] == ["shell", "update_plan"]
    # The regression fix: the forwarded shape is what Codex sent.
    assert "tools" not in forwarded
    assert forwarded["input"] == before["input"]


def test_stateful_second_turn_still_carries_tools() -> None:
    """Reproduces the 0.36.3 session: turn one worked, then tools vanished.

    A stateful client appends to the transcript it already sent. If Headroom
    forwards turn one without the carrier, the transcript the client builds
    turn two from has no tool definitions at all -- and turn two carries no
    top-level ``tools`` either, so the model is left with no tool surface.
    """
    forwarded_turn_1 = _forward(_codex_0149_payload())

    turn_2 = {
        "model": "gpt-5.6-sol",
        "input": [
            *forwarded_turn_1["input"],
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "now run pwd again"}],
            },
        ],
    }

    # Turn two declares no tools of its own; everything rides the transcript.
    assert "tools" not in turn_2
    assert any(item.get("type") == "additional_tools" for item in turn_2["input"]), (
        "turn two lost every tool definition -- this is the 0.36.3 regression"
    )

    # And turn two survives its own trip through the proxy with tools intact.
    forwarded_turn_2 = _forward(turn_2)
    carriers = [i for i in forwarded_turn_2["input"] if i.get("type") == "additional_tools"]
    assert [t["name"] for c in carriers for t in c["tools"]] == ["shell", "update_plan"]


def test_restore_keeps_the_compacted_schemas() -> None:
    """Restoration returns compaction's output, not the pre-compaction copy."""
    payload = _codex_0149_payload()
    plan: list[dict[str, Any]] = []
    _lift_codex_additional_tools(payload, restore_plan=plan)

    compacted, modified, _before, _after = _compact_openai_responses_tools(payload)
    assert modified, "fixture should be compactable"
    restored = _restore_codex_additional_tools(compacted, plan)

    assert restored == 2
    carrier = next(i for i in compacted["input"] if i.get("type") == "additional_tools")
    assert [t["name"] for t in carrier["tools"]] == ["shell", "update_plan"]
    # The verbose description is gone -- the savings survived the round trip.
    assert len(json.dumps(carrier["tools"])) < len(
        json.dumps(_codex_0149_payload()["input"][1]["tools"])
    )


def test_restore_puts_the_carrier_back_in_position() -> None:
    payload = _codex_0149_payload()
    payload["input"].append(
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "tail"}]}
    )
    before_types = [i["type"] for i in payload["input"]]

    plan: list[dict[str, Any]] = []
    _lift_codex_additional_tools(payload, restore_plan=plan)
    _restore_codex_additional_tools(payload, plan)

    assert [i["type"] for i in payload["input"]] == before_types


def test_restore_handles_multiple_carriers() -> None:
    payload = _codex_0149_payload()
    payload["input"].append({"type": "additional_tools", "tools": [_verbose_tool("view_image")]})
    before = copy.deepcopy(payload)

    plan: list[dict[str, Any]] = []
    assert _lift_codex_additional_tools(payload, restore_plan=plan) == 3
    assert _restore_codex_additional_tools(payload, plan) == 3

    assert payload == before


def test_restore_folds_into_first_carrier_when_the_count_changes() -> None:
    """Deferral/injection rewrites the array; the split no longer maps."""
    payload = _codex_0149_payload()
    payload["input"].append({"type": "additional_tools", "tools": [_verbose_tool("view_image")]})
    plan: list[dict[str, Any]] = []
    _lift_codex_additional_tools(payload, restore_plan=plan)

    payload["tools"] = [_verbose_tool("tool_search")]
    assert _restore_codex_additional_tools(payload, plan) == 1

    carriers = [i for i in payload["input"] if i.get("type") == "additional_tools"]
    assert len(carriers) == 1
    assert [t["name"] for t in carriers[0]["tools"]] == ["tool_search"]
    assert "tools" not in payload


def test_restore_recovers_the_originals_when_the_array_is_emptied() -> None:
    """A tool-less forward is never the safer outcome."""
    payload = _codex_0149_payload()
    plan: list[dict[str, Any]] = []
    _lift_codex_additional_tools(payload, restore_plan=plan)

    payload["tools"] = []
    assert _restore_codex_additional_tools(payload, plan) == 2

    carrier = next(i for i in payload["input"] if i.get("type") == "additional_tools")
    assert [t["name"] for t in carrier["tools"]] == ["shell", "update_plan"]


def test_restore_preserves_other_carrier_keys() -> None:
    payload = _codex_0149_payload()
    payload["input"][1]["id"] = "carrier_abc"
    before = copy.deepcopy(payload)

    plan: list[dict[str, Any]] = []
    _lift_codex_additional_tools(payload, restore_plan=plan)
    _restore_codex_additional_tools(payload, plan)

    assert payload == before


def test_restore_is_a_noop_without_a_plan() -> None:
    payload = _codex_0149_payload()
    before = copy.deepcopy(payload)

    assert _restore_codex_additional_tools(payload, []) == 0
    assert payload == before
    assert _restore_codex_additional_tools("not-a-dict", [{"tools": []}]) == 0  # type: ignore[arg-type]


def test_kill_switch_leaves_the_payload_completely_untouched(monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_CODEX_ADDITIONAL_TOOLS_LIFT", "0")
    payload = _codex_0149_payload()
    before = copy.deepcopy(payload)

    forwarded = _forward(payload)

    assert forwarded["input"] == before["input"]
    assert "tools" not in forwarded


def test_classic_top_level_clients_are_untouched_by_the_restore() -> None:
    """A non-Codex payload never enters the lift, so it never enters the restore."""
    payload = {
        "model": "gpt-5.6-sol",
        "tools": [_verbose_tool("shell")],
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}
        ],
    }
    before = copy.deepcopy(payload)

    forwarded = _forward(payload)

    assert forwarded["tools"] == before["tools"]
    assert forwarded["input"] == before["input"]


def test_unrestorable_payload_warns_instead_of_failing_silently(caplog) -> None:
    """If the carrier cannot go back, say so -- a stateful client will lose tools."""

    def _mangle(payload, *, model, request_id, **kwargs):  # noqa: ANN001, ANN202
        payload["input"] = "collapsed-to-a-string"
        return (payload, True, 0, [], None, 0, 0, 0, {})

    with caplog.at_level("WARNING", logger="headroom.proxy"):
        forwarded = _forward(_codex_0149_payload(), _mangle)

    assert any("could not be restored" in message for message in caplog.messages)
    # Degraded, not broken: this turn still carries its tools.
    assert [t["name"] for t in forwarded["tools"]] == ["shell", "update_plan"]


def test_restore_is_idempotent() -> None:
    """A second restore must not duplicate the definitions."""
    payload = _codex_0149_payload()
    before = copy.deepcopy(payload)

    plan: list[dict[str, Any]] = []
    _lift_codex_additional_tools(payload, restore_plan=plan)
    assert _restore_codex_additional_tools(payload, plan) == 2
    assert _restore_codex_additional_tools(payload, plan) == 0

    assert payload == before
