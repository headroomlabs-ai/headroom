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


@pytest.mark.parametrize(
    "raw",
    [
        b"data: one\n\r\ndata: two\n\n",  # LF line, CRLF blank line
        b"data: one\n\rdata: two\n\n",  # LF line, CR blank line
        b"data: one\r\n\rdata: two\n\n",  # CRLF line, CR blank line
        b"data: one\r\r\ndata: two\n\n",  # CR line, CRLF blank line
        b"data: one\r\rdata: two\r\r\n",  # CR only (last blank line is CRLF)
    ],
)
def test_parse_sse_events_accepts_mixed_and_cr_only_line_endings(raw: bytes) -> None:
    buf = bytearray(raw)

    assert parse_sse_events_from_byte_buffer(buf) == [(None, "one"), (None, "two")]
    assert bytes(buf) == b""


def test_find_sse_event_terminator_prefers_full_crlf_blank_line() -> None:
    assert find_sse_event_terminator(bytearray(b"data: one\r\n\r\nrest")) == (9, 4)


def test_find_sse_event_terminator_keeps_a_crlf_line_ending_whole() -> None:
    # "\r\n" is one line ending, not a CR ending plus a blank LF line.
    assert find_sse_event_terminator(bytearray(b"data: one\r\ndata: two")) is None


@pytest.mark.parametrize("raw", [b"data: one\r\r", b"data: one\n\r", b"data: one\r\n\r"])
def test_find_sse_event_terminator_waits_for_the_byte_after_a_trailing_cr(raw: bytes) -> None:
    # The next byte may be the LF of a CRLF that belongs to this terminator.
    assert find_sse_event_terminator(bytearray(raw)) is None
    assert find_sse_event_terminator(bytearray(raw + b"\n")) == (9, len(raw) - 9 + 1)
    assert find_sse_event_terminator(bytearray(raw + b"d")) == (9, len(raw) - 9)
