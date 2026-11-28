"""Closed SSE reconstruction must ignore one leading UTF-8 BOM (GH #4255).

The SSE spec (WHATWG "Interpreting an event stream") lets a stream start with a
single U+FEFF byte-order mark, which the parser must skip. The closed-response
parsers treated it as part of the first line, so ``\\ufeffdata: {...}`` was not
recognised as a ``data:`` field and the first event was dropped: an OpenAI
stream lost its first content delta, and a data-only Anthropic stream lost
``message_start`` (message id and initial usage).

Only the one marker at the very start of the payload is ignored. A BOM inside
event text is content and must survive, and the stored Anthropic wire frame
keeps its original bytes so opaque replay stays byte-identical.
"""

from __future__ import annotations

import json
from typing import Any

from headroom.ccr.response_handler import CCRResponseHandler, StreamingCCRHandler
from headroom.proxy.anthropic_wire import AnthropicSSEEnvelope, fold_anthropic_stream

BOM = "﻿".encode()


def _data_only(events: list[dict[str, Any]]) -> bytes:
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()


OPENAI_STREAM = (
    _data_only(
        [
            {
                "id": "chatcmpl_bom",
                "model": "gpt-4o",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}}],
            },
            {
                "id": "chatcmpl_bom",
                "model": "gpt-4o",
                "choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}],
            },
        ]
    )
    + b"data: [DONE]\n\n"
)

ANTHROPIC_STREAM = _data_only(
    [
        {
            "type": "message_start",
            "message": {
                "id": "msg_bom",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [],
                "usage": {"input_tokens": 7, "output_tokens": 1},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 2},
        },
        {"type": "message_stop"},
    ]
)


def test_openai_closed_stream_keeps_first_delta_after_bom() -> None:
    handler = StreamingCCRHandler(CCRResponseHandler(), provider="openai")

    parsed = handler._parse_sse_stream(BOM + OPENAI_STREAM)

    assert parsed["choices"][0]["message"]["content"] == "Hello"
    assert parsed == handler._parse_sse_stream(OPENAI_STREAM)


def test_anthropic_closed_stream_keeps_message_start_after_bom() -> None:
    handler = StreamingCCRHandler(CCRResponseHandler(), provider="anthropic")

    parsed = handler._parse_sse_stream(BOM + ANTHROPIC_STREAM)

    assert parsed["id"] == "msg_bom"
    assert parsed["usage"]["input_tokens"] == 7
    assert parsed["content"] == [{"type": "text", "text": "Hi"}]
    assert parsed == handler._parse_sse_stream(ANTHROPIC_STREAM)


def test_anthropic_envelope_after_bom_is_complete() -> None:
    envelope = AnthropicSSEEnvelope.parse(BOM + ANTHROPIC_STREAM)

    assert envelope.saw_message_start
    assert envelope.is_complete()
    assert fold_anthropic_stream(BOM + ANTHROPIC_STREAM) == fold_anthropic_stream(ANTHROPIC_STREAM)


def test_anthropic_error_event_after_bom_is_reported() -> None:
    error = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}

    assert fold_anthropic_stream(BOM + _data_only([error])) == error


def test_bom_prefixed_opaque_frame_replays_original_bytes() -> None:
    ping = BOM + b'event: ping\ndata: {"type": "ping"}\n\n'

    envelope = AnthropicSSEEnvelope.parse(ping + ANTHROPIC_STREAM)

    assert envelope.saw_message_start
    assert b"".join(envelope.render()).startswith(ping)


def test_bom_inside_event_text_is_preserved() -> None:
    handler = StreamingCCRHandler(CCRResponseHandler(), provider="openai")
    stream = _data_only([{"id": "c", "choices": [{"index": 0, "delta": {"content": "﻿Hi"}}]}])

    parsed = handler._parse_sse_stream(BOM + stream)

    assert parsed["choices"][0]["message"]["content"] == "﻿Hi"
