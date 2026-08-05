"""Pure SSE byte-buffer parsing policy."""

from __future__ import annotations

# SSE byte-buffer helper supports LF and CRLF event separators. Per the SSE
# spec the default event name is "message"; we return ``None`` so callers can
# decide whether to apply that default.
SSE_EVENT_TERMINATORS = (b"\n\n", b"\r\n\r\n")
SSE_EVENT_LINE_PREFIX = "event:"
SSE_DATA_LINE_PREFIX = "data:"


def find_sse_event_terminator(buf: bytearray) -> tuple[int, int] | None:
    """Return the earliest complete SSE event terminator in ``buf``."""
    matches = [
        (idx, len(terminator))
        for terminator in SSE_EVENT_TERMINATORS
        if (idx := buf.find(terminator)) != -1
    ]
    if not matches:
        return None
    return min(matches, key=lambda match: match[0])


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
