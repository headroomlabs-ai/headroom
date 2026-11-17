# tests/test_sse_safeguards_passthrough.py
"""Claude Code reads auto-mode verdicts from message_delta.delta.safeguard_results
(streaming) or top-level safeguard_results (JSON). Headroom's buffered-CCR path
rebuilds SSE from JSON; the rebuild must carry the field or Claude Code falls
back to a billed client-side classifier for the rest of the session."""

from __future__ import annotations

import json

import pytest

from headroom.proxy.handlers.streaming import StreamingMixin


class _Mixin(StreamingMixin):
    pass


def _events(raw: list[bytes]) -> list[dict]:
    out = []
    for chunk in raw:
        for block in chunk.decode().split("\n\n"):
            for line in block.splitlines():
                if line.startswith("data: "):
                    out.append(json.loads(line[6:]))
    return out


def _by_type(events: list[dict], t: str) -> list[dict]:
    return [e for e in events if e.get("type") == t]


_RESULTS = [{"type": "dangerous_tool_use", "verdict": "allow", "confidence": 0.99}]


def _json_response(**extra):
    base = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-x",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 2},
    }
    base.update(extra)
    return base


def test_response_to_sse_carries_safeguard_results_in_message_delta():
    ev = _events(_Mixin()._response_to_sse(_json_response(safeguard_results=_RESULTS), "anthropic"))
    (delta,) = _by_type(ev, "message_delta")
    assert delta["delta"]["safeguard_results"] == _RESULTS


def test_response_to_sse_keeps_stop_reason_alongside_safeguard_results():
    ev = _events(_Mixin()._response_to_sse(_json_response(safeguard_results=_RESULTS), "anthropic"))
    (delta,) = _by_type(ev, "message_delta")
    assert delta["delta"]["stop_reason"] == "end_turn"
    assert delta["delta"]["safeguard_results"] == _RESULTS


def test_response_to_sse_emits_null_safeguard_results():
    ev = _events(_Mixin()._response_to_sse(_json_response(safeguard_results=None), "anthropic"))
    (delta,) = _by_type(ev, "message_delta")
    assert "safeguard_results" in delta["delta"]
    assert delta["delta"]["safeguard_results"] is None


def test_response_to_sse_omits_key_when_source_lacks_it():
    ev = _events(_Mixin()._response_to_sse(_json_response(), "anthropic"))
    (delta,) = _by_type(ev, "message_delta")
    assert "safeguard_results" not in delta["delta"]


def test_response_to_sse_carries_stop_sequence():
    ev = _events(
        _Mixin()._response_to_sse(
            _json_response(stop_reason="stop_sequence", stop_sequence="END"), "anthropic"
        )
    )
    (delta,) = _by_type(ev, "message_delta")
    assert delta["delta"]["stop_sequence"] == "END"


def _sse(*events: dict) -> bytes:
    return b"".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events)


def _stream_with_results_in(where: str) -> bytes:
    start = {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-x",
            "content": [],
            "stop_reason": None,
            "usage": {"input_tokens": 10, "output_tokens": 0},
        },
    }
    delta = {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {"output_tokens": 2},
    }
    if where == "delta":
        delta["delta"]["safeguard_results"] = _RESULTS
    else:
        start["message"]["safeguard_results"] = _RESULTS
    return _sse(
        start,
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
        {"type": "content_block_stop", "index": 0},
        delta,
        {"type": "message_stop"},
    )


@pytest.mark.parametrize("where", ["delta", "start"])
def test_parse_sse_to_response_lifts_safeguard_results(where):
    parsed = _Mixin()._parse_sse_to_response(_stream_with_results_in(where).decode(), "anthropic")
    assert parsed is not None
    assert parsed["safeguard_results"] == _RESULTS


def test_round_trip_preserves_safeguard_results():
    m = _Mixin()
    parsed = m._parse_sse_to_response(_stream_with_results_in("delta").decode(), "anthropic")
    ev = _events(m._response_to_sse(parsed, "anthropic"))
    (delta,) = _by_type(ev, "message_delta")
    assert delta["delta"]["safeguard_results"] == _RESULTS
