"""PR-A8 / P1-9: SSE delta arms for thinking, signature, citations.

The proxy used to handle only ``text_delta`` and ``input_json_delta``
events on Anthropic's stream. The remaining delta types
(``thinking_delta``, ``signature_delta``, ``citations_delta``) and the
``redacted_thinking`` content_block_start were silently dropped, so any
non-streaming retry path that reconstructed the response from the SSE
stream produced an unsigned thinking block (rejected by Anthropic on
replay) or empty citations.

These tests pin the new contract:

- ``thinking_delta`` text appends to ``block.thinking_buffer`` and is
  promoted to ``block.thinking`` on ``content_block_stop``.
- ``signature_delta`` sets ``block.signature`` (last-write-wins).
- ``citations_delta`` appends each citation object to ``block.citations``.
- ``redacted_thinking`` content_block_start preserves the opaque
  ``data`` field as-is.
"""

from __future__ import annotations

import json
from typing import Any

from headroom.proxy.handlers.streaming import StreamingMixin


class _Parser(StreamingMixin):
    """Subclass that exposes the parser without the rest of the proxy."""


def _build_sse(events: list[dict[str, Any]]) -> str:
    """Render a list of event dicts as an SSE payload string."""
    out: list[str] = []
    for ev in events:
        out.append(f"event: {ev['type']}")
        out.append(f"data: {json.dumps(ev)}")
        out.append("")  # event terminator
    return "\n".join(out) + "\n"


def test_thinking_delta_accumulated() -> None:
    parser = _Parser()
    events = [
        {"type": "message_start", "message": {"id": "msg_1", "model": "claude-opus-4"}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "Let me consider "},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "the question carefully."},
        },
        {"type": "content_block_stop", "index": 0},
    ]
    sse = _build_sse(events)
    response = parser._parse_sse_to_response(sse, "anthropic")
    assert response is not None
    assert len(response["content"]) == 1
    block = response["content"][0]
    assert block["type"] == "thinking"
    assert block["thinking"] == "Let me consider the question carefully."


def test_signature_delta_preserved() -> None:
    parser = _Parser()
    events = [
        {"type": "message_start", "message": {"id": "msg_1", "model": "claude-opus-4"}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "hmm"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "sig_abc123_v1"},
        },
        {"type": "content_block_stop", "index": 0},
    ]
    sse = _build_sse(events)
    response = parser._parse_sse_to_response(sse, "anthropic")
    assert response is not None
    block = response["content"][0]
    assert block["signature"] == "sig_abc123_v1"
    # Last-write-wins semantics — second signature_delta overrides.
    events2 = events + [
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "sig_xyz999_v2"},
        },
    ]
    # Re-emit with the corrected ordering: stop must come after all deltas.
    events2 = [e for e in events2 if e["type"] != "content_block_stop"]
    events2.append({"type": "content_block_stop", "index": 0})
    response2 = parser._parse_sse_to_response(_build_sse(events2), "anthropic")
    assert response2 is not None
    assert response2["content"][0]["signature"] == "sig_xyz999_v2"


def test_citations_delta_accumulated() -> None:
    parser = _Parser()
    events = [
        {"type": "message_start", "message": {"id": "msg_1", "model": "claude-opus-4"}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Per source A"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "citations_delta",
                "citation": {
                    "type": "page_location",
                    "cited_text": "abc",
                    "document_index": 0,
                },
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "citations_delta",
                "citation": {
                    "type": "page_location",
                    "cited_text": "def",
                    "document_index": 1,
                },
            },
        },
        {"type": "content_block_stop", "index": 0},
    ]
    sse = _build_sse(events)
    response = parser._parse_sse_to_response(sse, "anthropic")
    assert response is not None
    block = response["content"][0]
    citations = block["citations"]
    assert len(citations) == 2
    assert citations[0]["cited_text"] == "abc"
    assert citations[1]["cited_text"] == "def"


def test_redacted_thinking_data_preserved() -> None:
    parser = _Parser()
    redacted_blob = "ENC:" + ("x" * 200)
    events = [
        {"type": "message_start", "message": {"id": "msg_1", "model": "claude-opus-4"}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "redacted_thinking", "data": redacted_blob},
        },
        {"type": "content_block_stop", "index": 0},
    ]
    sse = _build_sse(events)
    response = parser._parse_sse_to_response(sse, "anthropic")
    assert response is not None
    block = response["content"][0]
    assert block["type"] == "redacted_thinking"
    # `data` field MUST be preserved byte-for-byte for signature
    # validation on the next turn.
    assert block["data"] == redacted_blob
