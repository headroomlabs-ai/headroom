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


def _sse_events(sse_text: str) -> list[dict[str, Any]]:
    """Extract JSON data objects from an SSE payload string."""
    events: list[dict[str, Any]] = []
    for line in sse_text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


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


def test_non_standard_block_fields_preserved() -> None:
    # A block whose type isn't text/tool_use/thinking/redacted_thinking (e.g.
    # server_tool_use, web_search_tool_result) must keep its fields on
    # reconstruction, not collapse to a bare {type, index}. The sibling
    # _reconstruct_anthropic_response already does this via dict(block).
    parser = _Parser()
    events = [
        {"type": "message_start", "message": {"id": "msg_1", "model": "claude-opus-4"}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "server_tool_use",
                "id": "srvtoolu_1",
                "name": "web_search",
                "input": {"query": "headroom proxy"},
            },
        },
        {"type": "content_block_stop", "index": 0},
    ]
    sse = _build_sse(events)
    response = parser._parse_sse_to_response(sse, "anthropic")
    assert response is not None
    block = response["content"][0]
    assert block["type"] == "server_tool_use"
    assert block["id"] == "srvtoolu_1"
    assert block["name"] == "web_search"
    assert block["input"] == {"query": "headroom proxy"}


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


def test_response_to_sse_preserves_server_tool_use_blocks() -> None:
    parser = _Parser()
    response = {
        "id": "msg_1",
        "model": "claude-opus-4",
        "role": "assistant",
        "content": [
            {
                "type": "server_tool_use",
                "id": "srv_1",
                "name": "web_search",
                "input": {"query": "headroom"},
            }
        ],
        "stop_reason": "end_turn",
        "usage": {"output_tokens": 1},
    }

    sse_events = b"".join(parser._response_to_sse(response, "anthropic")).decode("utf-8")

    assert '"type": "content_block_start"' in sse_events
    assert '"type": "server_tool_use"' in sse_events
    assert '"name": "web_search"' in sse_events

    round_tripped = parser._parse_sse_to_response(sse_events, "anthropic")
    assert round_tripped is not None
    assert round_tripped["content"][0]["type"] == "server_tool_use"


def test_response_to_sse_preserves_thinking_redacted_and_citations() -> None:
    parser = _Parser()
    redacted_blob = "ENC:" + ("y" * 200)
    stop_details = {"type": "refusal", "message": "policy refusal"}
    response = {
        "id": "msg_2",
        "model": "claude-opus-4",
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "plan carefully", "signature": "sig_123"},
            {
                "type": "text",
                "text": "Per source A",
                "citations": [
                    {
                        "type": "page_location",
                        "cited_text": "abc",
                        "document_index": 0,
                    }
                ],
            },
            {"type": "redacted_thinking", "data": redacted_blob},
        ],
        "stop_reason": "refusal",
        "stop_details": stop_details,
        "usage": {"input_tokens": 10, "output_tokens": 3},
    }

    sse_text = b"".join(parser._response_to_sse(response, "anthropic")).decode("utf-8")

    assert "thinking_delta" in sse_text
    assert "signature_delta" in sse_text
    assert "citations_delta" in sse_text
    assert "redacted_thinking" in sse_text
    assert redacted_blob in sse_text

    round_tripped = parser._parse_sse_to_response(sse_text, "anthropic")
    assert round_tripped is not None
    assert round_tripped["content"][0]["thinking"] == "plan carefully"
    assert round_tripped["content"][0]["signature"] == "sig_123"
    assert round_tripped["content"][1]["citations"][0]["cited_text"] == "abc"
    assert round_tripped["content"][2]["data"] == redacted_blob
    assert round_tripped["stop_reason"] == "refusal"
    assert round_tripped["stop_details"] == stop_details


def test_response_to_sse_does_not_default_missing_stop_reason() -> None:
    parser = _Parser()
    sse_text = b"".join(parser._response_to_sse({"content": []}, "anthropic")).decode("utf-8")
    events = [
        json.loads(line[len("data: ") :])
        for line in sse_text.splitlines()
        if line.startswith("data: ")
    ]
    message_delta = next(event for event in events if event["type"] == "message_delta")

    assert message_delta["delta"] == {}
    assert "end_turn" not in sse_text


def test_response_to_sse_emits_unknown_content_block_verbatim() -> None:
    parser = _Parser()
    block = {"type": "future_block", "payload": {"preserve": ["me"]}}

    sse_text = b"".join(parser._response_to_sse({"content": [block]}, "anthropic")).decode("utf-8")
    events = _sse_events(sse_text)

    block_start = next(ev for ev in events if ev["type"] == "content_block_start")
    assert block_start["content_block"] == block
    assert not any(ev["type"] == "content_block_delta" for ev in events)


def test_response_to_sse_tolerates_malformed_content_and_usage() -> None:
    # `_response_to_sse` runs on provider/reconstruction-controlled JSON and is
    # reached from a call site (anthropic.py buffered CCR path) that only catches
    # ValueError, so a present-but-null `content`/`usage` or a non-dict block must
    # not raise a TypeError/AttributeError that would 500 the streamed request.
    # The sibling `_record_ccr_feedback_from_response` guards `content` the same
    # way. Each of these once crashed the unguarded loop.
    parser = _Parser()

    for response in (
        {"content": None, "usage": {"output_tokens": 5}},
        {"content": "not-a-list"},
        {"content": [None, {"type": "text", "text": "hi"}]},
        {"content": [], "usage": None},
    ):
        sse_text = b"".join(parser._response_to_sse(response, "anthropic")).decode("utf-8")
        # Always a well-formed envelope, regardless of the malformed body.
        assert "event: message_start" in sse_text
        assert "event: message_stop" in sse_text

    # The one valid block alongside a null element is still rendered.
    sse_text = b"".join(
        parser._response_to_sse({"content": [None, {"type": "text", "text": "hi"}]}, "anthropic")
    ).decode("utf-8")
    events = _sse_events(sse_text)
    text_deltas = [
        ev
        for ev in events
        if ev["type"] == "content_block_delta" and ev["delta"].get("type") == "text_delta"
    ]
    assert len(text_deltas) == 1
    assert text_deltas[0]["delta"]["text"] == "hi"


def test_response_to_sse_emits_server_tool_use_without_delta() -> None:
    parser = _Parser()
    server_tool_use = {
        "type": "server_tool_use",
        "id": "srvtoolu_123",
        "name": "web_search",
        "input": {"query": "headroom server_tool_use SSE crash"},
    }
    response = {
        "id": "msg_3",
        "model": "claude-opus-4",
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Searching."},
            server_tool_use,
        ],
        "stop_reason": "end_turn",
        "usage": {"output_tokens": 5},
    }

    sse_text = b"".join(parser._response_to_sse(response, "anthropic")).decode("utf-8")
    events = _sse_events(sse_text)

    block_starts = [ev for ev in events if ev["type"] == "content_block_start"]
    assert block_starts[1]["index"] == 1
    assert block_starts[1]["content_block"] == server_tool_use
    assert not any(ev["type"] == "content_block_delta" and ev["index"] == 1 for ev in events)
    assert any(
        ev["type"] == "content_block_delta"
        and ev["index"] == 0
        and ev["delta"] == {"type": "text_delta", "text": "Searching."}
        for ev in events
    )


# Issue #1876: CCR buffered-stream re-synthesis corrupted extended-thinking
# responses — `content_block_stop` deduped appended blocks by whole-dict
# equality (`target not in response["content"]`), so two distinct blocks
# that happened to be value-identical could collapse into one, or two
# stops for the *same* index with different accumulated content (a
# retried HTTP/2 stream reset redelivering a truncated segment) could
# both slip through as duplicates. Dedup is now keyed by block index.


def test_distinct_empty_thinking_blocks_at_different_indices_both_survive() -> None:
    """Two separate empty `thinking` blocks are two blocks, not one.

    Regression guard: if dedup ever regresses to dict-equality, this
    collapses to a single entry since both blocks are value-identical.
    """
    parser = _Parser()
    events = [
        {"type": "message_start", "message": {"id": "msg_1", "model": "claude-opus-4"}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "thinking", "thinking": ""},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "text_delta", "text": "Here is my answer."},
        },
        {"type": "content_block_stop", "index": 2},
    ]
    response = parser._parse_sse_to_response(_build_sse(events), "anthropic")
    assert response is not None
    assert len(response["content"]) == 3
    assert response["content"][0]["type"] == "thinking"
    assert response["content"][0]["thinking"] == ""
    assert response["content"][1]["type"] == "thinking"
    assert response["content"][1]["thinking"] == ""
    # The text block that followed the two empty thinking blocks must not
    # be dropped — this is the "text blocks are missing entirely" half of
    # the reported corruption.
    assert response["content"][2]["type"] == "text"
    assert response["content"][2]["text"] == "Here is my answer."


def test_redelivered_block_same_index_different_content_collapses_to_one_entry() -> None:
    """A fully redelivered content_block lifecycle (start/delta/stop) for
    an index that was already appended must not produce a second entry —
    even though the redelivered content differs from the first, which is
    exactly the case the old whole-dict-equality dedup missed. Reproduces
    an HTTP/2 stream-reset retry (`_stream_response`'s retry path)
    redelivering a fresh accumulation for the same block index: with the
    old `target not in response["content"]` check, the two dicts have
    unequal `thinking` text, so *both* slipped through as duplicate
    entries for one logical block.
    """
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
            "delta": {"type": "thinking_delta", "thinking": "partial"},
        },
        {"type": "content_block_stop", "index": 0},
        # Full redelivery of the same index with different accumulated
        # content — must be ignored, not appended as a second block.
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "full retried text"},
        },
        {"type": "content_block_stop", "index": 0},
    ]
    response = parser._parse_sse_to_response(_build_sse(events), "anthropic")
    assert response is not None
    assert len(response["content"]) == 1
    assert response["content"][0]["thinking"] == "partial"


def test_buffered_ccr_extended_thinking_round_trip_preserves_all_blocks() -> None:
    """End-to-end shape for issue #1876: a buffered CCR continuation
    response with thinking -> text -> tool_use must reconstruct to SSE
    (the re-synthesis path `anthropic.py` uses for the client-facing
    stream) with the text preserved and the thinking block emitted
    exactly once, unduplicated."""
    parser = _Parser()
    response = {
        "id": "msg_final",
        "model": "claude-opus-4",
        "role": "assistant",
        "content": [
            {
                "type": "thinking",
                "thinking": "Now I have the context, let me answer.",
                "signature": "sig_final",
            },
            {"type": "text", "text": "Based on the retrieved context, here is the answer."},
            {"type": "tool_use", "id": "toolu_real_1", "name": "real_tool", "input": {"y": 2}},
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 8, "output_tokens": 12},
    }

    sse_text = b"".join(parser._response_to_sse(response, "anthropic")).decode("utf-8")
    assert sse_text.count('"type": "thinking"') == 1
    assert "Based on the retrieved context, here is the answer." in sse_text

    round_tripped = parser._parse_sse_to_response(sse_text, "anthropic")
    assert round_tripped is not None
    assert len(round_tripped["content"]) == 3
    assert round_tripped["content"][0]["type"] == "thinking"
    assert round_tripped["content"][0]["thinking"] == "Now I have the context, let me answer."
    assert round_tripped["content"][1]["type"] == "text"
    assert (
        round_tripped["content"][1]["text"] == "Based on the retrieved context, here is the answer."
    )
    assert round_tripped["content"][2]["type"] == "tool_use"
    assert round_tripped["content"][2]["input"] == {"y": 2}


def test_server_tool_use_input_reassembled_from_partial_json() -> None:
    # server_tool_use streams its input via input_json_delta exactly like
    # tool_use: the content_block_start carries an empty input, the real args
    # arrive as partial_json, and content_block_stop must parse them into an
    # object. Gating the parse on type == "tool_use" left server_tool_use.input
    # empty and leaked the `_partial_json` scratch key into replayed assistant
    # history, which Anthropic rejects on the next turn (#2438).
    parser = _Parser()
    events = [
        {"type": "message_start", "message": {"id": "msg_1", "model": "claude-opus-4"}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "server_tool_use",
                "id": "srvtoolu_1",
                "name": "web_search",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"query": "hea'},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": 'droom proxy"}'},
        },
        {"type": "content_block_stop", "index": 0},
    ]
    sse = _build_sse(events)
    response = parser._parse_sse_to_response(sse, "anthropic")
    assert response is not None
    block = response["content"][0]
    assert block["type"] == "server_tool_use"
    assert block["input"] == {"query": "headroom proxy"}
    # Scratch key must never leak into a block that gets replayed as history.
    assert "_partial_json" not in block
