"""Pure SSE byte-buffer parsing policy."""

from __future__ import annotations

import re

# SSE byte-buffer helper supports LF, CRLF and CR line endings, mixed freely
# (see ``find_sse_event_terminator``). Per the SSE spec the default event name
# is "message"; we return ``None`` so callers can decide whether to apply that
# default. ``SSE_EVENT_TERMINATORS`` lists the two common separators.
SSE_EVENT_TERMINATORS = (b"\n\n", b"\r\n\r\n")
SSE_EVENT_LINE_PREFIX = "event:"
SSE_DATA_LINE_PREFIX = "data:"

# An event ends at a blank line: a line ending directly followed by another.
# The spec lets every line end in CRLF, LF or CR independently, so any pair of
# the three is a terminator (``\n\r``, ``\r\n\r``, ``\r\r\n`` ... as well as
# the common ``\n\n`` and ``\r\n\r\n``). A CR directly followed by LF is one
# CRLF ending, never a CR ending plus a blank LF line.
_SSE_BLANK_LINE_RE = re.compile(rb"(?:\r\n|\n|\r(?!\n))(?:\r\n|\n|\r)")


def find_sse_event_terminator(buf: bytearray) -> tuple[int, int] | None:
    """Return ``(index, length)`` of the earliest complete event terminator.

    Returns ``None`` when ``buf`` holds no blank line yet, or when the
    earliest one ends in a CR that is the last byte of ``buf``: the next byte
    may be the LF of a CRLF, which belongs to the same terminator.
    """
    match = _SSE_BLANK_LINE_RE.search(buf)
    if match is None:
        return None
    if match.end() == len(buf) and buf[-1:] == b"\r":
        return None
    return match.start(), match.end() - match.start()


def parse_sse_events_from_byte_buffer(
    buf: bytearray,
) -> list[tuple[str | None, str]]:
    """Drain complete ``event:`` + ``data:`` events from a bytes buffer.

    Returns list of ``(event_name, data_str)`` tuples for complete events.
    Mutates ``buf`` in-place to leave only partial-event tail bytes.

    Operates on bytes; only decodes complete events as UTF-8 (raises if a
    *complete* event has invalid UTF-8, which is an upstream protocol bug).
    """
    events: list[tuple[str | None, str]] = []
    while True:
        terminator_match = find_sse_event_terminator(buf)
        if terminator_match is None:
            break
        idx, terminator_len = terminator_match
        event_bytes = bytes(buf[:idx])
        del buf[: idx + terminator_len]

        event_text = event_bytes.decode("utf-8")
        event_name: str | None = None
        data_lines: list[str] = []
        for line in event_text.splitlines():
            if not line:
                continue
            if line.startswith(":"):
                continue
            if line.startswith(SSE_EVENT_LINE_PREFIX):
                event_name = line[len(SSE_EVENT_LINE_PREFIX) :].lstrip()
            elif line.startswith(SSE_DATA_LINE_PREFIX):
                data_lines.append(line[len(SSE_DATA_LINE_PREFIX) :].lstrip())

        if data_lines:
            events.append((event_name, "\n".join(data_lines)))
    return events
