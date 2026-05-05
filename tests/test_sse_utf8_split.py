"""PR-A8 / P1-8: bytes-level SSE buffer survives multi-byte splits.

Pre-A8 the proxy decoded each chunk via ``chunk.decode("utf-8",
errors="ignore")``. A 4-byte emoji or 3-byte CJK character split across
two TCP reads was silently dropped, corrupting the downstream parser's
view of the stream. The fix moves event-boundary detection into bytes
and decodes only complete events.

These tests pin:
- ``parse_sse_events_from_byte_buffer`` preserves emoji split mid-bytes.
- Same for CJK chars.
- A complete event with invalid UTF-8 raises (operator-visible diagnostic).
"""

from __future__ import annotations

import json

import pytest

from headroom.proxy.helpers import parse_sse_events_from_byte_buffer


def _emit_event(name: str, data: str) -> bytes:
    return f"event: {name}\ndata: {data}\n\n".encode()


def test_emoji_split_across_chunks_preserved() -> None:
    """Fire emoji `🔥` split mid-byte: 4 bytes total, split 2/2."""
    # `ensure_ascii=False` keeps the emoji as raw UTF-8 bytes on the
    # wire — that's the case where chunk-boundary splits actually
    # corrupt content with the old `errors="ignore"` decoder.
    payload = json.dumps({"type": "text_delta", "text": "fire 🔥 here"}, ensure_ascii=False)
    full = _emit_event("content_block_delta", payload)
    # Split somewhere inside the 4-byte emoji. The emoji `🔥` is
    # `\xf0\x9f\x94\xa5` — find that sequence and split mid-bytes.
    emoji_bytes = "🔥".encode()
    assert emoji_bytes == b"\xf0\x9f\x94\xa5"
    idx = full.find(emoji_bytes)
    assert idx > 0
    # Split halfway through the emoji.
    chunk_a = full[: idx + 2]
    chunk_b = full[idx + 2 :]

    buf = bytearray()
    buf.extend(chunk_a)
    # First chunk should not produce any complete events because the
    # event terminator `\n\n` is in chunk_b — but more importantly,
    # the helper must not corrupt the partial emoji bytes.
    events = parse_sse_events_from_byte_buffer(buf)
    assert events == []
    # Buffer still contains the partial emoji + everything before it.
    assert b"\xf0\x9f" in bytes(buf)

    buf.extend(chunk_b)
    events = parse_sse_events_from_byte_buffer(buf)
    assert len(events) == 1
    name, data_str = events[0]
    assert name == "content_block_delta"
    parsed = json.loads(data_str)
    assert parsed["text"] == "fire 🔥 here"


def test_cjk_split_across_chunks_preserved() -> None:
    """CJK `語` is 3 bytes (e8 aa 9e); split 1/2 across two reads."""
    payload = json.dumps({"type": "text_delta", "text": "日本語テスト"}, ensure_ascii=False)
    full = _emit_event("content_block_delta", payload)
    needle = "語".encode()
    assert needle == b"\xe8\xaa\x9e"
    idx = full.find(needle)
    assert idx > 0
    chunk_a = full[: idx + 1]
    chunk_b = full[idx + 1 :]

    buf = bytearray()
    buf.extend(chunk_a)
    events = parse_sse_events_from_byte_buffer(buf)
    assert events == []  # no complete event yet

    buf.extend(chunk_b)
    events = parse_sse_events_from_byte_buffer(buf)
    assert len(events) == 1
    _, data_str = events[0]
    parsed = json.loads(data_str)
    assert parsed["text"] == "日本語テスト"


def test_complete_event_with_invalid_utf8_raises_loud() -> None:
    """Invalid UTF-8 in a *complete* event surfaces loudly (not silent corruption)."""
    # Build a complete event whose data field has invalid UTF-8 bytes
    # NOT split across a chunk boundary — this is a true upstream bug
    # we want operators to see, not silently fix-up.
    bad = b'event: content_block_delta\ndata: {"text":"\xff\xfe"}\n\n'
    buf = bytearray()
    buf.extend(bad)
    with pytest.raises(UnicodeDecodeError):
        parse_sse_events_from_byte_buffer(buf)
