"""The OpenAI Responses wire shape on ``/v1/compress``.

Codex sends ``input``/``instructions``, not ``messages``. Before this, the
endpoint answered ``400 Missing required field: messages`` and a gateway in
front of Codex failed open on every turn.

The tests that matter here are the ones about what must NOT change. A
Responses transcript carries fields the provider hands back to itself --
``reasoning.encrypted_content`` above all -- and a compressor that rewrites or
reorders them produces a request that is smaller and rejected. Round-trip
fidelity is the feature; the token saving is the easy part.
"""

from __future__ import annotations

import copy

from headroom.proxy.gateway_responses import (
    VIEW_MARKER,
    apply_view,
    build_view,
    carries_view,
    is_responses_body,
    mark_view,
)
from headroom.proxy.gateway_turn import build_provider_body

ENCRYPTED = "gAAAAABqmk3crgbgbbcsDuoytKItfYE_AeLFYB0cl2Ikew"


def codex_body() -> dict:
    """A Responses body with one of each item type Codex actually sends."""
    return {
        "model": "gpt-5-codex",
        "instructions": "You are Codex.",
        "input": [
            {
                "type": "message",
                "id": "msg_1",
                "role": "developer",
                "content": [{"type": "input_text", "text": "project instructions"}],
                "internal_chat_message_metadata_passthrough": {"opaque": True},
            },
            {
                "type": "message",
                "id": "msg_2",
                "role": "user",
                "content": [{"type": "input_text", "text": "list the files"}],
            },
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [],
                "encrypted_content": ENCRYPTED,
            },
            {
                "type": "custom_tool_call",
                "id": "ctc_1",
                "call_id": "call_abc",
                "name": "shell",
                "input": "ls -R",
                "status": "completed",
            },
            {
                "type": "custom_tool_call_output",
                "id": "ctco_1",
                "call_id": "call_abc",
                "output": [{"type": "input_text", "text": "a.py\nb.py\n" * 200}],
            },
        ],
    }


def test_a_responses_body_is_recognised_and_a_chat_body_is_not():
    assert is_responses_body(codex_body())
    assert not is_responses_body({"model": "gpt-4o", "messages": []})


def test_a_chat_body_that_happens_to_carry_input_stays_on_the_chat_path():
    """``messages`` is the veto, not a tiebreak.

    Sniffing for ``input`` alone once cost this path a chat body's entire
    ``messages`` list: it was classified as Responses on the way out and the
    transcript was dropped.
    """
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "input": "stray"}
    assert not is_responses_body(body)
    assert not carries_view(body)

    out = build_provider_body(body, body["messages"], None)
    assert out["messages"] == body["messages"]
    assert out["input"] == "stray"


def test_every_text_slot_becomes_exactly_one_view_message():
    view = build_view(codex_body())
    # instructions + 2 message items + the call's input + the call's output.
    # Only `reasoning` contributes nothing.
    assert len(view.messages) == 5
    assert [m["role"] for m in view.messages] == [
        "system",
        "developer",
        "user",
        "assistant",
        "tool",
    ]
    assert view.messages[-2]["content"] == "ls -R"
    assert view.messages[-1]["tool_call_id"] == "call_abc"


def test_reasoning_is_never_shown_to_the_compressor():
    """What the compressor cannot see, it cannot corrupt.

    Reasoning is the one item that must never reach it: the provider hands
    ``encrypted_content`` back to itself and a single rewritten byte severs the
    model's own chain of thought.
    """
    view = build_view(codex_body())
    blob = "\n".join(str(m["content"]) for m in view.messages)
    assert ENCRYPTED not in blob


def test_an_untouched_view_round_trips_to_an_identical_body():
    body = codex_body()
    view = build_view(body)
    out, changed = apply_view(body, view.messages)
    assert changed is False
    assert out == body


def test_a_rewrite_reaches_the_text_and_nothing_else():
    body = codex_body()
    view = build_view(body)
    rewritten = [dict(m, content="SHORTER") for m in view.messages]
    out, changed = apply_view(body, rewritten)

    assert changed is True
    assert out["instructions"] == "SHORTER"
    assert out["input"][0]["content"][0]["text"] == "SHORTER"
    assert out["input"][4]["output"][0]["text"] == "SHORTER"
    assert out["input"][3]["input"] == "SHORTER"  # the call's own input
    # ...and everything opaque survives byte-for-byte.
    assert out["input"][2] == body["input"][2]
    assert out["input"][2]["encrypted_content"] == ENCRYPTED
    assert out["input"][0]["internal_chat_message_metadata_passthrough"] == {"opaque": True}
    assert [i["type"] for i in out["input"]] == [i["type"] for i in body["input"]]
    assert [i.get("call_id") for i in out["input"]] == [i.get("call_id") for i in body["input"]]


def test_a_view_that_does_not_line_up_is_refused_rather_than_guessed():
    """Fewer messages back than slots means we cannot say which slot lost its
    text. Forwarding the original uncompressed is the only safe answer."""
    body = codex_body()
    view = build_view(body)
    out, changed = apply_view(body, view.messages[:-1])
    assert changed is False
    assert out == body


def test_apply_view_does_not_mutate_the_body_it_was_given():
    body = codex_body()
    before = copy.deepcopy(body)
    view = build_view(body)
    apply_view(body, [dict(m, content="SHORTER") for m in view.messages])
    assert body == before


def test_the_provider_body_goes_back_in_the_shape_the_client_sent():
    body = codex_body()
    body["messages"] = build_view(body).messages
    mark_view(body)

    out = build_provider_body(body, [dict(m, content="SHORTER") for m in body["messages"]], None)

    assert "input" in out and "instructions" in out
    assert "messages" not in out, "the view is ours; the provider must never see it"
    assert VIEW_MARKER not in out, "control keys never reach a provider"
    assert out["input"][0]["content"][0]["text"] == "SHORTER"
    assert out["input"][2]["encrypted_content"] == ENCRYPTED


def test_a_bare_string_input_is_handled():
    body = {"model": "gpt-5-codex", "input": "just a string"}
    view = build_view(body)
    assert [m["content"] for m in view.messages] == ["just a string"]
    out, changed = apply_view(body, [{"role": "user", "content": "shorter"}])
    assert changed is True
    assert out["input"] == "shorter"


def test_string_valued_content_and_output_are_handled():
    """Not every client sends parts; both fields also accept a bare string."""
    body = {
        "model": "gpt-5-codex",
        "input": [
            {"type": "message", "role": "user", "content": "hello"},
            {"type": "function_call_output", "call_id": "c1", "output": "raw output"},
        ],
    }
    view = build_view(body)
    assert [m["content"] for m in view.messages] == ["hello", "raw output"]

    out, changed = apply_view(body, [dict(m, content="X") for m in view.messages])
    assert changed is True
    assert out["input"][0]["content"] == "X"
    assert out["input"][1]["output"] == "X"
    assert out["input"][1]["call_id"] == "c1"


def test_an_empty_input_yields_an_empty_view():
    view = build_view({"model": "gpt-5-codex", "input": []})
    assert view.messages == []
    assert len(view) == 0


def _call_body(item: dict) -> dict:
    return {"model": "gpt-5-codex", "input": [item]}


def test_a_call_input_is_compressible():
    """``custom_tool_call.input`` holds scripts and patches and is resent every
    turn: 8.9% of transcript bytes across real Codex sessions."""
    body = _call_body(
        {"type": "custom_tool_call", "call_id": "c1", "name": "exec", "input": "ls -R " * 500}
    )
    view = build_view(body)
    assert len(view.messages) == 1
    assert view.messages[0]["role"] == "assistant"

    out, changed = apply_view(body, [{"role": "assistant", "content": "ls -R"}])
    assert changed is True
    assert out["input"][0]["input"] == "ls -R"
    assert out["input"][0]["call_id"] == "c1"


def test_json_arguments_are_rewritten_only_when_the_result_still_parses():
    """``function_call.arguments`` is a JSON document the provider parses.

    A compressor works on text and has no reason to emit valid JSON, so
    shortening one would usually turn a working call into a parse error at the
    provider. The field keeps what it had unless the replacement parses.
    """
    body = _call_body(
        {"type": "function_call", "call_id": "c1", "name": "f", "arguments": '{"path": "/tmp/x"}'}
    )
    out, changed = apply_view(body, [{"role": "assistant", "content": "path=/tmp/x"}])
    assert changed is False, "plain text must not replace a JSON document"
    assert out["input"][0]["arguments"] == '{"path": "/tmp/x"}'

    out, changed = apply_view(body, [{"role": "assistant", "content": '{"path":"/tmp/x"}'}])
    assert changed is True, "valid JSON may replace valid JSON"
    assert out["input"][0]["arguments"] == '{"path":"/tmp/x"}'


def test_a_call_input_that_was_never_json_has_nothing_to_protect():
    """The guard is about preserving a parse, not about refusing to compress."""
    body = _call_body(
        {"type": "custom_tool_call", "call_id": "c1", "name": "exec", "input": "pwd && ls"}
    )
    out, changed = apply_view(body, [{"role": "assistant", "content": "pwd"}])
    assert changed is True
    assert out["input"][0]["input"] == "pwd"


# --------------------------------------------------------------------------- #
# CCR re-drive                                                                 #
# --------------------------------------------------------------------------- #


def test_the_ccr_tool_is_declared_in_the_flat_responses_shape():
    """Responses rejects the chat-completions nested declaration.

    Falling through to the OpenAI shape would inject a tool the provider
    refuses, on exactly the turns where compression markers made it necessary.
    """
    from headroom.ccr import CCR_TOOL_NAME
    from headroom.ccr.tool_injection import create_ccr_tool_definition

    flat = create_ccr_tool_definition("openai_responses")
    assert flat["type"] == "function"
    assert flat["name"] == CCR_TOOL_NAME
    assert "function" not in flat, "that is the chat-completions shape"
    assert "hash" in flat["parameters"]["properties"]

    nested = create_ccr_tool_definition("openai")
    assert nested["function"]["name"] == CCR_TOOL_NAME, "chat shape must not change"


def test_arming_ccr_on_a_responses_turn_injects_the_flat_tool():
    from headroom.ccr import CCR_TOOL_NAME
    from headroom.proxy.gateway_turn import (
        GatewayCapabilities,
        RequestTransformResult,
        arm_ccr_redrive,
    )

    caps = GatewayCapabilities(can_redrive=True, can_relay_response=True, session_affinity=True)
    result = RequestTransformResult(
        messages=[], tools=[], ctx=None, transforms=[], redrive_armed=False
    )
    armed = arm_ccr_redrive(
        result,
        provider="openai",
        caps=caps,
        mode="ccr",
        ccr_hashes=["abc123"],
        responses_shape=True,
    )
    assert armed is True
    injected = [t for t in result.tools if t.get("name") == CCR_TOOL_NAME]
    assert len(injected) == 1
    assert "function" not in injected[0], "a Responses turn needs the flat shape"


def test_a_redrive_goes_back_in_the_responses_shape():
    """The second request must look like the first.

    ``_redrive_payload`` wrote ``messages`` unconditionally, so a re-drive on a
    Responses turn handed the gateway a body carrying both ``input`` and
    ``messages`` -- which the provider rejects. This covers every re-drive, CCR
    and tool-router alike.
    """
    from headroom.proxy.gateway_turn import PendingTurn, Step, _redrive_payload

    body = codex_body()
    view = build_view(body)
    turn = PendingTurn(
        turn_id="t1",
        session_key=None,
        session_id=None,
        provider="openai",
        model="gpt-5-codex",
        body=body,
        ctx=None,
        obligations=["redrive"],
        outcome_draft=None,
        created_at=0.0,
        deadline=0.0,
        wire_shape="openai_responses",
    )
    step = Step(
        kind="redrive",
        messages=[dict(m, content="SHORTER") for m in view.messages],
        tools=None,
    )

    sent = _redrive_payload(turn, step)["request"]

    assert "input" in sent
    assert "messages" not in sent, "the provider rejects a body carrying both"
    assert sent["input"][0]["content"][0]["text"] == "SHORTER"
    assert sent["input"][2]["encrypted_content"] == ENCRYPTED


def test_a_responses_body_with_no_view_never_grows_a_messages_key():
    """The bypass header returns before a view is built.

    ``build_provider_body`` used to add ``messages`` unconditionally, so
    ``x-headroom-bypass: true`` -- the path whose whole job is to change
    nothing -- turned every Codex request into a provider 400: "Unsupported
    parameter: 'messages'. In the Responses API, this parameter has moved to
    'input'."
    """
    body = codex_body()
    assert not carries_view(body)

    out = build_provider_body(body, [], None)

    assert "messages" not in out
    assert out["input"] == body["input"]
    assert out["instructions"] == body["instructions"]


def test_a_view_that_cannot_map_back_forwards_the_original_transcript():
    """Fail open on the transcript, not on a half-applied rewrite."""
    body = codex_body()
    body["messages"] = build_view(body).messages
    mark_view(body)

    # One message short: the slots cannot be matched up.
    out = build_provider_body(body, body["messages"][:-1], None)

    assert "messages" not in out
    assert out["input"] == codex_body()["input"], "the original goes out untouched"
