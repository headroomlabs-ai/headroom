"""WebSocket-shaped `/v1/responses` Rust binding tests.

The default Python CLI runtime now compresses WS `response.create` frames
through its CompressionUnit + ContentRouter path. These tests keep the
lower-level PyO3 live-zone binding covered on WebSocket-shaped envelopes
so Rust migration work cannot silently break the exposed bridge.

The tests exercise the compression *transformation logic* in isolation —
they replicate the body-shape handling the WS handler does (envelope
detect, compress inner, re-wrap) without spinning up a full WebSocket
session. Full session-lifecycle coverage already exists in
`test_openai_codex_ws_lifecycle.py`.
"""

from __future__ import annotations

import json
from typing import Any

import pytest


def _ensure_binding():
    """Skip if the Rust extension hasn't been built (mirrors the pattern
    in `test_responses_pyo3_compression.py`)."""
    try:
        from headroom._core import compress_openai_responses_live_zone

        return compress_openai_responses_live_zone
    except ImportError:
        pytest.skip("headroom._core not built — run scripts/build_rust_extension.sh")


def _ws_compress_first_frame(
    first_msg_raw: str,
    auth_mode_value: str = "payg",
    bypass: bool = False,
) -> tuple[str, bool]:
    """Replicates the WS-handler compression block as a pure function.

    Returns ``(new_first_msg_raw, modified)``. The real handler embeds
    this logic inline in `handle_openai_responses_ws`; pulling it out
    here lets us pin the exact byte-shape contract without standing
    up a full WebSocket fixture. If you change the handler's
    compression block, mirror it here so the tests catch the drift.
    """
    if bypass:
        return first_msg_raw, False

    compress = _ensure_binding()

    try:
        send_body: Any = json.loads(first_msg_raw)
    except json.JSONDecodeError:
        return first_msg_raw, False

    if not isinstance(send_body, dict):
        return first_msg_raw, False

    wrapped = "response" in send_body and isinstance(send_body["response"], dict)
    inner = send_body["response"] if wrapped else send_body
    model = (inner.get("model") if isinstance(inner, dict) else None) or ""

    inner_bytes = json.dumps(inner).encode("utf-8")
    new_bytes, modified, _saved, _transforms, _reason = compress(
        inner_bytes, auth_mode_value, model
    )
    if not modified:
        return first_msg_raw, False

    try:
        new_inner = json.loads(new_bytes)
    except json.JSONDecodeError:
        return first_msg_raw, False

    if not isinstance(new_inner, dict):
        return first_msg_raw, False

    if wrapped:
        send_body["response"] = new_inner
    else:
        send_body = new_inner
    return json.dumps(send_body), True


class TestWrappedEnvelopeShape:
    """Codex's WebSocket protocol wraps the Responses payload in a
    ``response.create`` envelope. The WS handler must unwrap to compress
    and re-wrap to forward."""

    def test_passthrough_when_inner_has_no_input_array(self):
        # No `input` array → dispatcher's NoMessagesArray path → passthrough.
        first_msg = json.dumps(
            {
                "type": "response.create",
                "response": {"model": "gpt-5"},
            }
        )
        out, modified = _ws_compress_first_frame(first_msg)
        assert modified is False
        assert out == first_msg

    def test_envelope_preserved_on_passthrough(self):
        first_msg = json.dumps(
            {
                "type": "response.create",
                "response": {
                    "model": "gpt-5",
                    "input": [{"type": "message", "role": "user", "content": "hi"}],
                },
            }
        )
        out, modified = _ws_compress_first_frame(first_msg)
        # Single small user message → no compression applies.
        assert modified is False
        assert json.loads(out) == json.loads(first_msg)

    def test_bypass_header_short_circuits_first_frame(self):
        first_msg = json.dumps(
            {
                "type": "response.create",
                "response": {
                    "model": "gpt-5",
                    "input": [
                        {
                            "type": "function_call_output",
                            "call_id": "call_1",
                            "output": json.dumps(
                                [
                                    {
                                        "id": i,
                                        "name": f"Item {i}",
                                        "desc": "large repeated payload " * 20,
                                    }
                                    for i in range(100)
                                ]
                            ),
                        }
                    ],
                },
            }
        )

        out, modified = _ws_compress_first_frame(first_msg, bypass=True)

        assert modified is False
        assert out == first_msg


class TestUnwrappedShape:
    """Older Codex versions (and some test fixtures) send the Responses
    payload directly as the first frame, without a `response.create`
    envelope. The handler must work for both shapes."""

    def test_passthrough_when_no_input_array(self):
        first_msg = json.dumps({"model": "gpt-5"})
        out, modified = _ws_compress_first_frame(first_msg)
        assert modified is False
        assert out == first_msg

    def test_passthrough_when_empty_input(self):
        first_msg = json.dumps({"model": "gpt-5", "input": []})
        out, modified = _ws_compress_first_frame(first_msg)
        assert modified is False
        assert out == first_msg


class TestNonJsonFirstFrame:
    """If the first frame isn't JSON, we forward it byte-for-byte rather
    than crashing the WS session."""

    def test_garbage_passthrough(self):
        out, modified = _ws_compress_first_frame("not actually json")
        assert modified is False
        assert out == "not actually json"

    def test_json_array_passthrough(self):
        # Top-level array isn't a Responses envelope.
        first_msg = json.dumps([1, 2, 3])
        out, modified = _ws_compress_first_frame(first_msg)
        assert modified is False
        assert out == first_msg

    def test_json_string_passthrough(self):
        first_msg = json.dumps("a string at the top level")
        out, modified = _ws_compress_first_frame(first_msg)
        assert modified is False
        assert out == first_msg


class TestAuthModeForwarded:
    """Every F1 AuthMode value reaches the dispatcher without raising.
    The dispatcher itself currently treats all modes identically (per-mode
    tuning is F2.2 follow-up), but the call must not fail on any value
    the F1 classifier produces."""

    @pytest.mark.parametrize(
        "auth_mode_value",
        ["payg", "oauth", "subscription", "unknown"],
    )
    def test_all_auth_modes_accepted(self, auth_mode_value: str):
        first_msg = json.dumps({"model": "gpt-5", "input": []})
        out, modified = _ws_compress_first_frame(first_msg, auth_mode_value)
        assert modified is False
        assert out == first_msg


class TestNoExceptionLeak:
    """The WS handler wraps the compression block in try/except so a
    JSON-shape edge case can never crash the WS session. This pins the
    contract that no input shape produces an exception in the
    transformation function."""

    @pytest.mark.parametrize(
        "first_msg",
        [
            "",
            "{",
            "{",
            "}",
            "[",
            "null",
            "true",
            "0",
            json.dumps({}),
            json.dumps({"response": "not a dict"}),
            json.dumps({"response": []}),
            json.dumps({"response": None}),
            json.dumps({"response": {"input": "not an array"}}),
            json.dumps({"input": "string instead of array"}),
            json.dumps({"input": None}),
        ],
    )
    def test_no_exception_for_garbage_shapes(self, first_msg: str):
        # Should never raise — return passthrough on anything malformed.
        out, modified = _ws_compress_first_frame(first_msg)
        # Regardless of result, no exception leaked. modified might be
        # False here (garbage input → no compression).
        assert isinstance(out, str)
        assert isinstance(modified, bool)
