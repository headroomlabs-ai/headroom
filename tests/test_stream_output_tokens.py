"""Output tokens must be counted from the stream's text, not its wire size.

When an upstream sends no usage chunk, the proxy estimated output tokens as
``total_bytes // 40`` over the RAW SSE WIRE — ``data:`` prefixes, JSON
envelopes, ``role``/``finish_reason``/``id``/``model`` fields and blank-line
framing all included. From a field log (Copilot Chat, 0.36.x):

    Could not parse output_tokens from SSE, estimating 8 from 334 bytes

The divisor is a fudge for "bytes per token including framing", so the error
tracked how chattily the answer was chunked rather than how long it was: the
same answer split into more deltas scores higher purely for being split.

GitHub's Copilot CAPI is one of the upstreams that omits the usage chunk, so
this was every Copilot turn's output number — and output tokens feed the
output-shaping savings estimate and the cost model.
"""

from __future__ import annotations

import json

import pytest

from headroom.proxy.stream_output_tokens import (
    TEXT_CHARS_PER_TOKEN,
    WIRE_BYTES_PER_TOKEN,
    estimate_output_tokens,
    extract_stream_text,
)


def _sse(*objs: dict, done: bool = True) -> str:
    out = "".join(f"data: {json.dumps(o)}\n\n" for o in objs)
    return out + ("data: [DONE]\n\n" if done else "")


def _chat_delta(text: str) -> dict:
    return {"choices": [{"index": 0, "delta": {"content": text}}]}


# --------------------------------------------------------------------------- #
# Extraction, per surface
# --------------------------------------------------------------------------- #
def test_openai_chat_deltas() -> None:
    sse = _sse(
        {"choices": [{"delta": {"role": "assistant"}}]},
        _chat_delta("Hello "),
        _chat_delta("world"),
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    )
    assert extract_stream_text(sse) == "Hello world"


def test_anthropic_content_block_deltas() -> None:
    sse = _sse(
        {"type": "message_start", "message": {"id": "msg_1"}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "abc"}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "def"}},
        {"type": "message_stop"},
        done=False,
    )
    assert extract_stream_text(sse) == "abcdef"


def test_openai_responses_deltas() -> None:
    sse = _sse(
        {"type": "response.output_text.delta", "delta": "part one "},
        {"type": "response.output_text.delta", "delta": "part two"},
    )
    assert extract_stream_text(sse) == "part one part two"


def test_reasoning_and_tool_arguments_are_billed_output_too() -> None:
    """Omitting these under-counts exactly the most expensive turns."""
    sse = _sse(
        {"choices": [{"delta": {"reasoning_content": "thinking hard"}}]},
        {"choices": [{"delta": {"tool_calls": [{"function": {"arguments": '{"path":"a.py"}'}}]}}]},
    )
    text = extract_stream_text(sse)
    assert "thinking hard" in text
    assert '{"path":"a.py"}' in text


def test_anthropic_thinking_and_partial_json() -> None:
    sse = _sse(
        {"type": "content_block_delta", "delta": {"thinking": "plan"}},
        {"type": "content_block_delta", "delta": {"partial_json": '{"a":1}'}},
        done=False,
    )
    assert extract_stream_text(sse) == 'plan{"a":1}'


# --------------------------------------------------------------------------- #
# Malformed input must never raise — this runs on the response path
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "sse",
    [
        "",
        "data: not-json\n\n",
        "data: [DONE]\n\n",
        "garbage without data prefix\n\n",
        'data: {"choices": "not-a-list"}\n\n',
        'data: {"choices": [null]}\n\n',
        'data: {"choices": [{"delta": null}]}\n\n',
        'data: {"choices": [{"delta": {"content": 42}}]}\n\n',
        'data: {"type": "content_block_delta", "delta": "not-a-dict"}\n\n',
        "data:\n\n",
    ],
)
def test_malformed_streams_yield_empty_not_an_exception(sse: str) -> None:
    assert extract_stream_text(sse) == ""


def test_multi_line_data_fields_concatenate() -> None:
    """Per the SSE spec, and real streams do it."""
    payload = json.dumps(_chat_delta("joined"))
    half = len(payload) // 2
    sse = f"data: {payload[:half]}\ndata: {payload[half:]}\n\n"
    assert extract_stream_text(sse) == "joined"


# --------------------------------------------------------------------------- #
# The estimate itself
# --------------------------------------------------------------------------- #
def test_text_beats_the_wire_heuristic_on_the_reported_shape() -> None:
    """A chunky stream: framing dominates the wire, so bytes//40 misreads it."""
    sse = _sse(*[_chat_delta(w) for w in ("The ", "quick ", "brown ", "fox ", "jumps")])
    total_bytes = len(sse.encode())

    tokens, source = estimate_output_tokens(sse_text=sse, total_bytes=total_bytes)

    assert source == "estimated_text"
    # "The quick brown fox jumps" is 25 chars -> 6 tokens.
    assert tokens == len("The quick brown fox jumps") // TEXT_CHARS_PER_TOKEN
    # The old estimator scored this stream far higher purely for its framing.
    assert total_bytes // WIRE_BYTES_PER_TOKEN > tokens


def test_chunking_no_longer_changes_the_answer() -> None:
    """Same text, different delta split — the count must not move."""
    text = "identical content across both streams"
    one = _sse(_chat_delta(text))
    many = _sse(*[_chat_delta(c) for c in text])

    a, _ = estimate_output_tokens(sse_text=one, total_bytes=len(one.encode()))
    b, _ = estimate_output_tokens(sse_text=many, total_bytes=len(many.encode()))

    assert a == b
    # And the wire-based estimator would have disagreed wildly.
    assert len(one.encode()) // WIRE_BYTES_PER_TOKEN != len(many.encode()) // WIRE_BYTES_PER_TOKEN


def test_a_short_answer_is_never_recorded_as_zero() -> None:
    sse = _sse(_chat_delta("OK"))
    tokens, source = estimate_output_tokens(sse_text=sse, total_bytes=len(sse.encode()))
    assert source == "estimated_text"
    assert tokens == 1


def test_falls_back_to_bytes_when_no_text_is_recoverable() -> None:
    """The upstream-error path reaches here with no stream text at all."""
    tokens, source = estimate_output_tokens(sse_text="", total_bytes=800)
    assert source == "estimated_bytes"
    assert tokens == 800 // WIRE_BYTES_PER_TOKEN


def test_negative_or_zero_bytes_are_safe() -> None:
    assert estimate_output_tokens(sse_text="", total_bytes=0) == (0, "estimated_bytes")
    assert estimate_output_tokens(sse_text="", total_bytes=-5) == (0, "estimated_bytes")
