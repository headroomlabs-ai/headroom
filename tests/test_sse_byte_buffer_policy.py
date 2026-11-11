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


_TRAILING_CR_TERMINATORS = [b"\r\r", b"\n\r", b"\r\n\r"]


@pytest.mark.parametrize("terminator", _TRAILING_CR_TERMINATORS)
def test_find_sse_event_terminator_accepts_a_trailing_cr_at_eof(terminator: bytes) -> None:
    # Once the stream has ended no LF can follow, so the final CR is the end.
    raw = bytearray(b"data: one" + terminator)
    assert find_sse_event_terminator(raw, at_eof=True) == (9, len(terminator))


@pytest.mark.parametrize("bytewise", [False, True], ids=["full", "bytewise"])
@pytest.mark.parametrize("terminator", _TRAILING_CR_TERMINATORS)
def test_parse_sse_events_drains_a_trailing_cr_event_at_eof(
    terminator: bytes, bytewise: bool
) -> None:
    wire = b"event: delta\rdata: one" + terminator
    buf = bytearray()
    events: list[tuple[str | None, str]] = []
    for chunk in [wire[i : i + 1] for i in range(len(wire))] if bytewise else [wire]:
        buf.extend(chunk)
        events += parse_sse_events_from_byte_buffer(buf)
    # Held back while a CRLF could still be completed ...
    assert events == []
    assert buf == wire
    # ... and released once the stream has ended.
    assert parse_sse_events_from_byte_buffer(buf, at_eof=True) == [("delta", "one")]
    assert buf == bytearray()


def test_parse_sse_events_at_eof_leaves_an_unterminated_tail() -> None:
    # A last event without a blank line, or one cut inside a UTF-8 character,
    # is still incomplete at EOF and stays in the buffer.
    buf = bytearray(b"data: one\r\rdata: caf\xc3")
    assert parse_sse_events_from_byte_buffer(buf, at_eof=True) == [(None, "one")]
    assert buf == bytearray(b"data: caf\xc3")


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
def test_parse_sse_events_preserves_unicode_line_separators_in_json(
    separator: str,
) -> None:
    payload = {"text": f"before{separator}after"}
    wire = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()

    events = parse_sse_events_from_byte_buffer(bytearray(wire))

    assert len(events) == 1
    assert json.loads(events[0][1]) == payload


def test_parse_sse_events_strips_only_one_leading_space() -> None:
    # SSE spec: "If value starts with a U+0020 SPACE character, remove it".
    # Further leading whitespace is part of the payload.
    buf = bytearray(b"event: delta\ndata:   indented\ndata:\ttab\ndata:none\n\n")

    assert parse_sse_events_from_byte_buffer(buf) == [("delta", "  indented\n\ttab\nnone")]
