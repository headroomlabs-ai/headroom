"""Client-bound CCR marker scrubbing for streaming responses.

CCR markers are an internal proxy-to-model protocol.  They must never cross
the proxy-to-client boundary, even when an upstream splits a marker across
transport chunks or embeds it in a streamed tool-call argument.  This module
buffers at most one SSE event, rewrites its JSON payload recursively, and
preserves non-data SSE fields verbatim.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

from headroom.cache.compression_store import CompressionStore, get_compression_store
from headroom.ccr.marker_resolution import scrub_client_payload, scrub_markers_for_client

MAX_CCR_SSE_EVENT_BYTES = 8 * 1024 * 1024


class CCRMarkerEgressOverflow(ValueError):
    """Raised when an unterminated SSE event exceeds the bounded holdback."""


class CCRMarkerEgressFilter:
    """Incrementally scrub CCR markers from client-bound SSE bytes."""

    def __init__(self, *, store: CompressionStore | None = None) -> None:
        self._store = store or get_compression_store()
        self._buffer = bytearray()

    def feed(self, chunk: bytes) -> list[bytes]:
        """Consume an upstream chunk and return complete scrubbed SSE events."""
        self._buffer.extend(chunk)
        if len(self._buffer) > MAX_CCR_SSE_EVENT_BYTES:
            size = len(self._buffer)
            self._buffer.clear()
            raise CCRMarkerEgressOverflow(
                f"CCR client-bound SSE event exceeded {MAX_CCR_SSE_EVENT_BYTES} bytes "
                f"without a boundary (received {size})"
            )
        output: list[bytes] = []
        while True:
            boundary = _next_event_boundary(self._buffer)
            if boundary is None:
                break
            end, separator_length = boundary
            event = bytes(self._buffer[:end])
            del self._buffer[: end + separator_length]
            separator = b"\r\n\r\n" if separator_length == 4 else b"\n\n"
            output.append(self._scrub_event(event) + separator)
        return output

    def finish(self) -> list[bytes]:
        """Flush a truncated final event without allowing a marker to leak."""
        if not self._buffer:
            return []
        tail = bytes(self._buffer)
        self._buffer.clear()
        return [self._scrub_event(tail)]

    def _scrub_event(self, event: bytes) -> bytes:
        if b"<<ccr:" not in event:
            return event

        lines = event.splitlines(keepends=True)
        data_indexes = [index for index, line in enumerate(lines) if line.startswith(b"data:")]
        if not data_indexes:
            return _scrub_text_bytes(event, store=self._store)

        data_parts = [lines[index][5:].lstrip(b" ").rstrip(b"\r\n") for index in data_indexes]
        raw_data = b"\n".join(data_parts)
        try:
            payload = json.loads(raw_data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return _scrub_text_bytes(event, store=self._store)

        scrubbed = scrub_client_payload(payload, store=self._store)
        encoded = json.dumps(scrubbed, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        first = data_indexes[0]
        newline = b"\r\n" if lines[first].endswith(b"\r\n") else b"\n"
        lines[first] = b"data: " + encoded + newline
        for index in reversed(data_indexes[1:]):
            del lines[index]
        return b"".join(lines)


def scrub_sse_chunks(
    chunks: Iterable[bytes], *, store: CompressionStore | None = None
) -> list[bytes]:
    """Synchronous convenience helper used by focused tests."""
    marker_filter = CCRMarkerEgressFilter(store=store)
    output: list[bytes] = []
    for chunk in chunks:
        output.extend(marker_filter.feed(chunk))
    output.extend(marker_filter.finish())
    return output


def _next_event_boundary(buffer: bytearray) -> tuple[int, int] | None:
    lf = buffer.find(b"\n\n")
    crlf = buffer.find(b"\r\n\r\n")
    candidates = [(lf, 2), (crlf, 4)]
    present = [candidate for candidate in candidates if candidate[0] >= 0]
    return min(present, default=None)


def _scrub_text_bytes(data: bytes, *, store: CompressionStore) -> bytes:
    text = data.decode("utf-8", errors="replace")
    scrubbed = scrub_markers_for_client(text, store=store, resolve_hits=False)
    assert isinstance(scrubbed, str)
    return scrubbed.encode("utf-8")
