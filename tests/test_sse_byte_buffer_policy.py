from __future__ import annotations

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


def test_parse_sse_events_raises_on_complete_invalid_utf8_event() -> None:
    with pytest.raises(UnicodeDecodeError):
        parse_sse_events_from_byte_buffer(bytearray(b"data: \xff\n\n"))
