from __future__ import annotations

import json

import pytest

from headroom.proxy.sse_byte_buffer_policy import (
    find_sse_event_terminator,
    parse_sse_events_from_byte_buffer,
)


def test_find_sse_event_terminator_returns_earliest_separator() -> None:
    assert find_sse_event_terminator(bytearray(b"data: one\r\n\r\ndata: two\n\n")) == (9, 4)


def test_parse_sse_events_drains_complete_events_and_leaves_tail() -> None:
    buf = bytearray(b": ignored\nevent: delta\ndata: one\ndata: two\n\npartial")

    assert parse_sse_events_from_byte_buffer(buf) == [("delta", "one\ntwo")]
    assert bytes(buf) == b"partial"


def test_parse_sse_events_preserves_split_utf8_tail() -> None:
    smile = "\U0001f642".encode()
    buf = bytearray(b"data: hello " + smile[:2])

    assert parse_sse_events_from_byte_buffer(buf) == []
    buf.extend(smile[2:] + b"\n\n")
    assert parse_sse_events_from_byte_buffer(buf) == [(None, "hello \U0001f642")]


def test_parse_sse_events_splits_crlf_lines() -> None:
    buf = bytearray(b"event: delta\r\ndata: one\r\ndata: two\r\n\r\n")

    assert parse_sse_events_from_byte_buffer(buf) == [("delta", "one\ntwo")]
    assert not buf


def test_parse_sse_events_raises_on_complete_invalid_utf8_event() -> None:
    with pytest.raises(UnicodeDecodeError):
        parse_sse_events_from_byte_buffer(bytearray(b"data: \xff\n\n"))


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
def test_parse_sse_events_preserves_unicode_line_separators_in_json(
    separator: str,
) -> None:
    payload = {"text": f"before{separator}after"}
    wire = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()

    events = parse_sse_events_from_byte_buffer(bytearray(wire))

    assert len(events) == 1
    assert json.loads(events[0][1]) == payload
