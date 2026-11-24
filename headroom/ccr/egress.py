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


_ARGUMENT_EVENT_TAGS = (
    b"input_json_delta",
    b"function_call_arguments",
    b"content_block_stop",
    b"output_item.done",
)
_MARKER_PREFIX = "<<ccr:"
_MAX_HELD_MARKER_CHARS = 256


def _incomplete_marker_start(text: str) -> int | None:
    """Return where a trailing, possibly unfinished CCR marker begins."""
    open_at = text.rfind(_MARKER_PREFIX)
    if open_at >= 0 and ">>" not in text[open_at:]:
        if len(text) - open_at <= _MAX_HELD_MARKER_CHARS:
            return open_at
        return None
    for size in range(len(_MARKER_PREFIX) - 1, 0, -1):
        if text.endswith(_MARKER_PREFIX[:size]):
            return len(text) - size
    return None


class ToolArgumentMarkerAssembler:
    """Resolve CCR markers that a provider splits across argument delta events.

    Anthropic ``input_json_delta`` and Responses ``function_call_arguments``
    deltas arrive as independent fragments.  A trailing, possibly unfinished
    marker is held until the next fragment of the same tool call completes it;
    any leftover is emitted as a synthesized delta before the call closes.
    """

    def __init__(self, *, store: CompressionStore | None = None) -> None:
        self._store = store or get_compression_store()
        self._held: dict[str, str] = {}

    @property
    def has_held(self) -> bool:
        return bool(self._held)

    def process(self, event: dict[str, object]) -> list[dict[str, object]]:
        event_type = event.get("type")
        if event_type == "content_block_delta":
            delta = event.get("delta")
            if isinstance(delta, dict) and delta.get("type") == "input_json_delta":
                key = f"a:{event.get('index')}"
                fragment = delta.get("partial_json")
                if isinstance(fragment, str):
                    delta["partial_json"] = self._advance(key, fragment)
            return [event]
        if event_type == "response.function_call_arguments.delta":
            fragment = event.get("delta")
            if isinstance(fragment, str):
                key = f"r:{event.get('item_id') or event.get('output_index')}"
                event["delta"] = self._advance(key, fragment)
            return [event]
        if event_type == "content_block_stop":
            leftover = self._held.pop(f"a:{event.get('index')}", "")
            if leftover:
                synthetic: dict[str, object] = {
                    "type": "content_block_delta",
                    "index": event.get("index"),
                    "delta": {"type": "input_json_delta", "partial_json": leftover},
                }
                return [synthetic, event]
            return [event]
        if event_type in {"response.function_call_arguments.done", "response.output_item.done"}:
            item = event.get("item")
            item_id = event.get("item_id") or (item.get("id") if isinstance(item, dict) else None)
            leftover = self._held.pop(f"r:{item_id or event.get('output_index')}", "")
            if leftover:
                synthetic = {
                    "type": "response.function_call_arguments.delta",
                    "item_id": item_id,
                    "output_index": event.get("output_index"),
                    "delta": leftover,
                }
                return [synthetic, event]
        return [event]

    def _advance(self, key: str, fragment: str) -> str:
        text = self._held.pop(key, "") + fragment
        cut = _incomplete_marker_start(text)
        if cut is not None:
            self._held[key] = text[cut:]
            text = text[:cut]
        scrubbed = scrub_markers_for_client(text, store=self._store, resolve_hits=True)
        assert isinstance(scrubbed, str)
        return scrubbed


class CCRMarkerEgressFilter:
    """Incrementally scrub CCR markers from client-bound SSE bytes."""

    def __init__(self, *, store: CompressionStore | None = None) -> None:
        self._store = store or get_compression_store()
        self._buffer = bytearray()
        self._arguments = ToolArgumentMarkerAssembler(store=self._store)

    def feed(self, chunk: bytes) -> list[bytes]:
        """Consume an upstream chunk and return complete scrubbed SSE events."""
        self._buffer.extend(chunk)
        output: list[bytes] = []
        while True:
            boundary = _next_event_boundary(self._buffer)
            if boundary is None:
                break
            end, separator_length = boundary
            if end + separator_length > MAX_CCR_SSE_EVENT_BYTES:
                self._buffer.clear()
                raise CCRMarkerEgressOverflow("CCR client-bound SSE event exceeded the event limit")
            event = bytes(self._buffer[:end])
            del self._buffer[: end + separator_length]
            separator = b"\r\n\r\n" if separator_length == 4 else b"\n\n"
            output.extend(scrubbed + separator for scrubbed in self._scrub_event(event))
        if len(self._buffer) > MAX_CCR_SSE_EVENT_BYTES:
            size = len(self._buffer)
            self._buffer.clear()
            raise CCRMarkerEgressOverflow(
                f"CCR client-bound SSE event exceeded {MAX_CCR_SSE_EVENT_BYTES} bytes "
                f"without a boundary (received {size})"
            )
        return output

    def finish(self) -> list[bytes]:
        """Flush a truncated final event without allowing a marker to leak."""
        if not self._buffer:
            return []
        tail = bytes(self._buffer)
        self._buffer.clear()
        return self._scrub_event(tail)

    def _scrub_event(self, event: bytes) -> list[bytes]:
        """Return the scrubbed event, preceded by any synthesized delta."""
        if b"<<ccr:" not in event and not (
            self._arguments.has_held or any(tag in event for tag in _ARGUMENT_EVENT_TAGS)
        ):
            return [event]

        lines = event.splitlines(keepends=True)
        data_indexes = [index for index, line in enumerate(lines) if line.startswith(b"data:")]
        if not data_indexes:
            return [_scrub_text_bytes(event, store=self._store)]

        data_parts = [lines[index][5:].lstrip(b" ").rstrip(b"\r\n") for index in data_indexes]
        raw_data = b"\n".join(data_parts)
        try:
            payload = json.loads(raw_data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return [_scrub_text_bytes(event, store=self._store)]

        newline = b"\r\n" if lines[data_indexes[0]].endswith(b"\r\n") else b"\n"
        has_event_line = any(line.startswith(b"event:") for line in lines)
        payloads = self._arguments.process(payload) if isinstance(payload, dict) else [payload]
        output: list[bytes] = []
        for item in payloads:
            scrubbed = scrub_client_payload(item, store=self._store)
            encoded = json.dumps(scrubbed, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
            if item is payload:
                first = data_indexes[0]
                lines[first] = b"data: " + encoded + newline
                for index in reversed(data_indexes[1:]):
                    del lines[index]
                output.append(b"".join(lines))
                continue
            head = b""
            if has_event_line and isinstance(item, dict):
                head = b"event: " + str(item.get("type")).encode("utf-8") + newline
            output.append(head + b"data: " + encoded + newline)
        return output


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
