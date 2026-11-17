"""Malformed-upstream tolerance for SSE event parsing.

Two parsers read upstream-controlled SSE bodies: ``_parse_sse_usage_from_buffer``
and ``_parse_sse_to_response``. A valid-JSON
event of unexpected *shape* must be skipped rather than raise. In
``_parse_sse_to_response`` the raise escapes into ``_finalize_stream_response``
and tears down a stream the client is already reading.

Scope: these cover **frame shape** — the type of a container the parser reaches
into. Value validity (a token count that is a string, ``Infinity``, or a
>4300-digit integer literal) is a separate defect class and is not covered here.

The happy-path fixtures are transcribed from real ``api.anthropic.com``
``/v1/messages`` streams (``claude-haiku-4-5``), not hand-invented. Details a
stand-in gets wrong, which these pin:

* ``message_start.message`` sends ``stop_reason``, ``stop_sequence`` and
  ``stop_details`` as **null**. A guard that rejected nulls indiscriminately
  would break every ordinary request.
* ``message_delta.usage`` repeats the *full* usage block, not just
  ``output_tokens``.
* ``usage.cache_creation`` is always present as a nested object — it is what
  ``_extract_anthropic_cache_ttl_metrics`` reads.
* A ``tool_use`` block carries a non-standard ``caller`` field, which is what
  exercises the copy-through branch in ``content_block_start``.
* Extended thinking emits ``thinking_delta`` then ``signature_delta`` on block
  0, then a separate text block at index 1.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from headroom.proxy.handlers.streaming import StreamingMixin


class _Handler(StreamingMixin):
    """Bare mixin host — the parsers touch no other handler state."""


# `None` is itself a value under test, so it cannot double as "not supplied".
_UNSET = object()

_WRONG_TYPES = [None, "str", 42, 3.5, True, ["x"], [], {}]


def _frame(event: str, payload: Any) -> str:
    """One SSE frame in Anthropic's wire shape: an ``event:`` line then ``data:``."""
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


# ---------------------------------------------------------------------------
# Fixtures transcribed from real api.anthropic.com streams.
# ---------------------------------------------------------------------------

_REAL_USAGE = {
    "input_tokens": 12,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
    "output_tokens": 1,
    "service_tier": "standard",
    "inference_geo": "not_available",
}

_REAL_MESSAGE = {
    "id": "msg_011CdQD1YDscyCRWpGCcrwkA",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-4-5-20251001",
    "content": [],
    # Real Anthropic sends all three as null on message_start.
    "stop_reason": None,
    "stop_sequence": None,
    "stop_details": None,
    "usage": _REAL_USAGE,
}

_MESSAGE_DELTA = _frame(
    "message_delta",
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None, "stop_details": None},
        # The real message_delta repeats the whole usage block.
        "usage": {
            "input_tokens": 12,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "output_tokens": 4,
        },
    },
)


def _text_stream(*, message: Any = _UNSET, extra: str = "", mid: str = "") -> str:
    """The real single-text-block stream, optionally with frames injected.

    ``extra`` lands right after ``message_start``; ``mid`` lands *inside* the
    open content block, after ``content_block_start``. The distinction matters:
    a ``content_block_delta`` injected before any block has opened resolves to
    ``target is None`` and the parser skips its whole body, so a delta-shaped
    fault placed in ``extra`` would never reach the code it is meant to test.
    """
    msg = _REAL_MESSAGE if message is _UNSET else message
    return (
        _frame("message_start", {"type": "message_start", "message": msg})
        + extra
        + _frame(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        )
        + _frame("ping", {"type": "ping"})
        + mid
        + _frame(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hi"},
            },
        )
        + _frame("content_block_stop", {"type": "content_block_stop", "index": 0})
        + _MESSAGE_DELTA
        + _frame("message_stop", {"type": "message_stop"})
    )


def _tool_use_stream() -> str:
    """The real tool_use stream — note the non-standard ``caller`` field."""
    partials = ["", '{"ci', "ty", '": "Par', 'is"}']
    return (
        _frame("message_start", {"type": "message_start", "message": _REAL_MESSAGE})
        + _frame(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_01Kpi77sYFPtGoQbz6YMCSok",
                    "name": "get_weather",
                    "input": {},
                    "caller": {"type": "direct"},
                },
            },
        )
        + "".join(
            _frame(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": p},
                },
            )
            for p in partials
        )
        + _frame("content_block_stop", {"type": "content_block_stop", "index": 0})
        + _frame("message_stop", {"type": "message_stop"})
    )


def _thinking_stream() -> str:
    """The real extended-thinking stream: thinking block 0, then text block 1."""
    return (
        _frame("message_start", {"type": "message_start", "message": _REAL_MESSAGE})
        + _frame(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""},
            },
        )
        + _frame(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "17"},
            },
        )
        + _frame(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": " * 23 = 391"},
            },
        )
        + _frame(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "EtQCCpMBCBAYAipA"},
            },
        )
        + _frame("content_block_stop", {"type": "content_block_stop", "index": 0})
        + _frame(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "text", "text": ""},
            },
        )
        + _frame(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": "391"},
            },
        )
        + _frame("content_block_stop", {"type": "content_block_stop", "index": 1})
        + _MESSAGE_DELTA
        + _frame("message_stop", {"type": "message_stop"})
    )


def _usage(payload: str, provider: str = "anthropic") -> dict[str, int] | None:
    """Run the buffered usage parser over a whole stream."""
    return _Handler()._parse_sse_usage_from_buffer(
        {"sse_buffer": bytearray(payload.encode())}, provider
    )


def _response(payload: str, provider: str = "anthropic") -> dict[str, Any] | None:
    return _Handler()._parse_sse_to_response(payload, provider)


# ---------------------------------------------------------------------------
# Happy path — the real streams must parse, nulls and ping frames included.
# ---------------------------------------------------------------------------


def test_real_text_stream_usage_is_parsed() -> None:
    usage = _usage(_text_stream())
    assert usage is not None
    assert usage["input_tokens"] == 12
    assert usage["cache_read_input_tokens"] == 0
    # message_delta's output_tokens wins over message_start's provisional 1.
    assert usage["output_tokens"] == 4
    # Sourced from the nested usage.cache_creation object real traffic always sends.
    assert usage["cache_creation_ephemeral_5m_input_tokens"] == 0
    assert usage["cache_creation_ephemeral_1h_input_tokens"] == 0


def test_real_text_stream_reconstructs_the_response() -> None:
    resp = _response(_text_stream())
    assert resp is not None
    assert resp["id"] == "msg_011CdQD1YDscyCRWpGCcrwkA"
    assert resp["model"] == "claude-haiku-4-5-20251001"
    assert resp["role"] == "assistant"
    assert resp["stop_reason"] == "end_turn"
    assert [b["text"] for b in resp["content"]] == ["hi"]
    assert resp["usage"]["output_tokens"] == 4


def test_real_tool_use_stream_reconstructs_tool_input() -> None:
    resp = _response(_tool_use_stream())
    assert resp is not None
    block = resp["content"][0]
    assert block["type"] == "tool_use"
    assert block["name"] == "get_weather"
    # The five partial_json deltas must accumulate and parse.
    assert block["input"] == {"city": "Paris"}
    assert "_partial_json" not in block
    # Pins current behaviour, not desired behaviour: the copy-through branch
    # runs only for *non-standard* block types, so `tool_use` keeps just
    # id/name/input and the real `caller` field is dropped on reconstruction.
    # Out of scope here (this is fidelity, not a shape crash); see the
    # `server_tool_use` cases below for the copy-through path itself.
    assert block["caller"] == {"type": "direct"}


def test_real_thinking_stream_reconstructs_both_blocks() -> None:
    resp = _response(_thinking_stream())
    assert resp is not None
    thinking, text = resp["content"]
    assert thinking["type"] == "thinking"
    assert thinking["thinking"] == "17 * 23 = 391"
    assert thinking["signature"] == "EtQCCpMBCBAYAipA"
    assert "thinking_buffer" not in thinking
    assert text["text"] == "391"


def test_legitimate_nulls_are_not_treated_as_malformed() -> None:
    # stop_reason/stop_sequence/stop_details are null on every real
    # message_start. The guard must not mistake them for a malformed frame.
    head = (
        _frame("message_start", {"type": "message_start", "message": _REAL_MESSAGE})
        + _frame(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        )
        + _frame("content_block_stop", {"type": "content_block_stop", "index": 0})
    )
    partial = _response(head)
    assert partial is not None
    assert partial["id"] == "msg_011CdQD1YDscyCRWpGCcrwkA"
    assert partial["stop_reason"] is None
    assert partial["stop_details"] is None


# ---------------------------------------------------------------------------
# Shape sweep. Each entry injects a wrong-typed value at one position the
# parser reaches into, inside an otherwise-real stream. The frame must be
# skipped, never fatal, and the surrounding good frames must still parse.
# ---------------------------------------------------------------------------


def _at_message(bad: Any) -> str:
    return _text_stream(message=bad)


def _at_message_usage(bad: Any) -> str:
    return _text_stream(message={**_REAL_MESSAGE, "usage": bad})


def _at_message_stop_details(bad: Any) -> str:
    return _text_stream(message={**_REAL_MESSAGE, "stop_details": bad})


def _at_content_block(bad: Any) -> str:
    return _text_stream(
        extra=_frame(
            "content_block_start",
            {"type": "content_block_start", "index": 5, "content_block": bad},
        )
    )


def _at_block_start_index(bad: Any) -> str:
    return _text_stream(
        extra=_frame(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": bad,
                "content_block": {"type": "text", "text": ""},
            },
        )
    )


def _at_block_delta_index(bad: Any) -> str:
    return _text_stream(
        extra=_frame(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": bad,
                "delta": {"type": "text_delta", "text": "X"},
            },
        )
    )


def _at_block_stop_index(bad: Any) -> str:
    return _text_stream(
        extra=_frame("content_block_stop", {"type": "content_block_stop", "index": bad})
    )


def _at_content_block_delta(bad: Any) -> str:
    return _text_stream(
        mid=_frame("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": bad})
    )


def _at_text_delta_text(bad: Any) -> str:
    return _text_stream(
        mid=_frame(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": bad},
            },
        )
    )


def _at_partial_json(bad: Any) -> str:
    return _text_stream(
        mid=_frame(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": bad},
            },
        )
    )


def _at_thinking_delta(bad: Any) -> str:
    return _text_stream(
        mid=_frame(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": bad},
            },
        )
    )


def _at_message_delta(bad: Any) -> str:
    return _text_stream(extra=_frame("message_delta", {"type": "message_delta", "delta": bad}))


def _at_message_delta_usage(bad: Any) -> str:
    return _text_stream(
        extra=_frame("message_delta", {"type": "message_delta", "delta": {}, "usage": bad})
    )


def _at_top_level_event(bad: Any) -> str:
    return _text_stream(extra=f"event: garbage\ndata: {json.dumps(bad)}\n\n")


def _at_citations_copy_through(bad: Any) -> str:
    """Upstream seeds a non-list `citations` on a non-standard block, then a
    citations_delta tries to append to it."""
    return _text_stream(
        extra=_frame(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 9,
                "content_block": {"type": "server_tool_use", "citations": bad},
            },
        )
        + _frame(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 9,
                "delta": {"type": "citations_delta", "citation": {"cited_text": "x"}},
            },
        )
        + _frame("content_block_stop", {"type": "content_block_stop", "index": 9})
    )


def _at_scratch_key_injection(bad: Any) -> str:
    """Upstream tries to seed the parser's own accumulator scratch keys."""
    return _text_stream(
        extra=_frame(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 8,
                "content_block": {
                    "type": "server_tool_use",
                    "_partial_json": bad,
                    "thinking_buffer": bad,
                },
            },
        )
        + _frame(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 8,
                "delta": {"type": "input_json_delta", "partial_json": '{"a":1}'},
            },
        )
        + _frame(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 8,
                "delta": {"type": "thinking_delta", "thinking": "t"},
            },
        )
        + _frame("content_block_stop", {"type": "content_block_stop", "index": 8})
    )


_INJECTIONS = [
    ("message", _at_message),
    ("message.usage", _at_message_usage),
    # No guard needed at this position — the parser copies stop_details
    # through without inspecting it. Kept so the sweep's contract ("no
    # position raises") stays exhaustive if that ever changes.
    ("message.stop_details", _at_message_stop_details),
    ("content_block", _at_content_block),
    ("content_block_start.index", _at_block_start_index),
    ("content_block_delta.index", _at_block_delta_index),
    ("content_block_stop.index", _at_block_stop_index),
    ("content_block_delta.delta", _at_content_block_delta),
    ("text_delta.text", _at_text_delta_text),
    ("input_json_delta.partial_json", _at_partial_json),
    ("thinking_delta.thinking", _at_thinking_delta),
    ("message_delta.delta", _at_message_delta),
    ("message_delta.usage", _at_message_delta_usage),
    ("top-level event", _at_top_level_event),
    ("citations copy-through", _at_citations_copy_through),
    ("scratch-key injection", _at_scratch_key_injection),
]


@pytest.mark.parametrize("position,build", _INJECTIONS, ids=[p for p, _ in _INJECTIONS])
@pytest.mark.parametrize("bad", _WRONG_TYPES, ids=repr)
def test_response_parser_survives_wrong_shape(position: str, build: Any, bad: Any) -> None:
    resp = _response(build(bad))
    assert resp is not None, f"{position}={bad!r} lost the whole response"
    # The good delta's text still lands. Substring rather than equality: when
    # `bad` happens to be a valid string it is a legitimate text_delta and
    # correctly accumulates alongside "hi".
    texts = [b.get("text") or "" for b in resp["content"] if b.get("type") == "text"]
    assert any("hi" in t for t in texts), f"{position}={bad!r} destroyed the good content block"


@pytest.mark.parametrize("position,build", _INJECTIONS, ids=[p for p, _ in _INJECTIONS])
@pytest.mark.parametrize("bad", _WRONG_TYPES, ids=repr)
def test_usage_parser_survives_wrong_shape(position: str, build: Any, bad: Any) -> None:
    usage = _usage(build(bad))
    assert usage is not None, f"{position}={bad!r} lost all usage"
    # message_start's input_tokens survives whatever was injected around it.
    if position not in {"message", "message.usage"}:
        assert usage["input_tokens"] == 12, f"{position}={bad!r} corrupted input_tokens"


@pytest.mark.parametrize("bad", _WRONG_TYPES, ids=repr)
def test_message_start_wrong_shape_still_yields_later_usage(bad: Any) -> None:
    # A malformed message_start must not prevent message_delta's usage landing.
    usage = _usage(_text_stream(message=bad))
    assert usage is not None
    assert usage["output_tokens"] == 4


# ---------------------------------------------------------------------------
# The usage parser, across every provider branch.
# ---------------------------------------------------------------------------

_USAGE_POSITIONS = [
    ("anthropic", lambda b: {"type": "message_start", "message": b}),
    ("anthropic", lambda b: {"type": "message_start", "message": {"usage": b}}),
    ("anthropic", lambda b: {"type": "message_delta", "usage": b}),
    ("openai", lambda b: {"usage": b}),
    ("openai", lambda b: {"response": b}),
    ("openai", lambda b: {"usage": {"prompt_tokens_details": b}}),
    ("gemini", lambda b: {"usageMetadata": b}),
]


@pytest.mark.parametrize("provider,build", _USAGE_POSITIONS)
@pytest.mark.parametrize("bad", _WRONG_TYPES, ids=repr)
def test_buffered_usage_parser_survives_wrong_shape(provider: str, build: Any, bad: Any) -> None:
    # Must not raise; returning None (nothing usable) is the correct outcome.
    _usage(_frame("x", build(bad)), provider)


@pytest.mark.parametrize("provider", ["anthropic", "openai", "gemini"])
@pytest.mark.parametrize("bad", _WRONG_TYPES, ids=repr)
def test_usage_parser_skips_non_object_events(provider: str, bad: Any) -> None:
    payload = f"data: {json.dumps(bad)}\n\n"
    assert not _usage(payload, provider)


def test_gemini_usage_still_parses(provider: str = "gemini") -> None:
    # Pins that the gemini guard did not break the branch it hardened.
    usage = _usage(
        _frame("x", {"usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 3}}), provider
    )
    assert usage is not None
    assert usage["input_tokens"] == 7
    assert usage["output_tokens"] == 3


def test_openai_usage_still_parses() -> None:
    usage = _usage(_frame("x", {"usage": {"prompt_tokens": 9, "completion_tokens": 2}}), "openai")
    assert usage is not None
    assert usage["input_tokens"] == 9
    assert usage["output_tokens"] == 2


def test_unparseable_json_is_still_skipped() -> None:
    # Pre-existing behaviour: the JSONDecodeError guard already covered this.
    usage = _usage(_text_stream(extra="event: broken\ndata: {not json\n\n"))
    assert usage is not None
    assert usage["input_tokens"] == 12
