"""Bounded event-level CCR interception and continuation splicing."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from headroom.ccr.egress import MAX_CCR_SSE_EVENT_BYTES, scrub_sse_chunks
from headroom.ccr.marker_resolution import (
    scrub_client_payload,
    strip_internal_retrieve_calls,
)
from headroom.ccr.response_handler import (
    RESIDUAL_CCR_ERROR,
    RESIDUAL_CCR_SKIPPED_MIXED,
    CCRResponseHandler,
    StreamingCCRHandler,
)

MAX_CCR_STREAM_RECONSTRUCTION_BYTES = 8 * 1024 * 1024


class EventLevelCCRInterceptor:
    """Suppress private CCR events, continue server-side, and splice the result."""

    def __init__(
        self,
        response_handler: CCRResponseHandler,
        *,
        provider: str,
        render_response: Callable[[dict[str, Any]], list[bytes]],
    ) -> None:
        if provider not in {"anthropic", "openai_responses"}:
            raise ValueError(f"unsupported CCR stream provider: {provider}")
        self.response_handler = response_handler
        self.provider = provider
        self.render_response = render_response

    async def process(
        self,
        stream: AsyncIterator[bytes],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        api_call_fn: Callable[
            [list[dict[str, Any]], list[dict[str, Any]] | None],
            Awaitable[dict[str, Any]],
        ],
    ) -> AsyncIterator[bytes]:
        decoder = _SSEEventDecoder()
        reconstruction = bytearray()
        reconstruction_available = True
        private_indexes: set[int] = set()
        private_item_ids: set[str] = set()
        detected_private = False
        detected_client_tool = False
        held_terminal: list[bytes] = []
        max_visible_index = -1
        max_sequence = -1
        completed_response: dict[str, Any] | None = None

        async for chunk in stream:
            for raw_event in decoder.feed(chunk):
                if self.provider == "anthropic" and reconstruction_available:
                    if len(reconstruction) + len(raw_event) > MAX_CCR_STREAM_RECONSTRUCTION_BYTES:
                        reconstruction.clear()
                        reconstruction_available = False
                    else:
                        reconstruction.extend(raw_event)
                event = _parse_sse_event(raw_event)
                if event is None:
                    if (
                        self.provider == "openai_responses"
                        and detected_private
                        and _is_done_event(raw_event)
                    ):
                        held_terminal.append(raw_event)
                        continue
                    for scrubbed in _scrub_event(raw_event):
                        yield scrubbed
                    continue
                event_type = str(event.get("type") or "")
                sequence = event.get("sequence_number")
                if isinstance(sequence, int):
                    max_sequence = max(max_sequence, sequence)

                if self.provider == "anthropic":
                    index = event.get("index")
                    if event_type == "content_block_start":
                        block = event.get("content_block")
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            if block.get("name") == "headroom_retrieve":
                                detected_private = True
                                if isinstance(index, int):
                                    private_indexes.add(index)
                                continue
                            detected_client_tool = True
                    if isinstance(index, int) and index in private_indexes:
                        continue
                    if isinstance(index, int):
                        max_visible_index = max(max_visible_index, index)
                    if event_type in {"message_delta", "message_stop"} and detected_private:
                        held_terminal.append(raw_event)
                        continue
                else:
                    item = event.get("item")
                    if isinstance(item, dict) and item.get("type") == "function_call":
                        item_id = str(item.get("id") or item.get("call_id") or "")
                        if item.get("name") == "headroom_retrieve":
                            detected_private = True
                            if item_id:
                                private_item_ids.add(item_id)
                            continue
                        detected_client_tool = True
                    item_id = str(event.get("item_id") or "")
                    if item_id and item_id in private_item_ids:
                        continue
                    output_index = event.get("output_index")
                    if isinstance(output_index, int):
                        max_visible_index = max(max_visible_index, output_index)
                    if event_type == "response.completed":
                        response = event.get("response")
                        if isinstance(response, dict):
                            completed_response = response
                        if detected_private:
                            held_terminal.append(raw_event)
                            continue
                client_event = raw_event
                if self.provider == "openai_responses" and isinstance(event.get("response"), dict):
                    event["response"] = strip_internal_retrieve_calls(
                        event["response"], "openai_responses"
                    )
                    client_event = _encode_sse_event(event)
                for scrubbed in _scrub_event(client_event):
                    yield scrubbed

        for raw_event in decoder.finish():
            if self.provider == "anthropic" and reconstruction_available:
                if len(reconstruction) + len(raw_event) <= MAX_CCR_STREAM_RECONSTRUCTION_BYTES:
                    reconstruction.extend(raw_event)
                else:
                    reconstruction.clear()
                    reconstruction_available = False
            for scrubbed in _scrub_event(raw_event):
                yield scrubbed

        if not detected_private:
            return

        if self.provider == "anthropic" and not reconstruction_available:
            yield _error_event(
                self.provider,
                "CCR stream exceeded the bounded continuation window.",
            )
            return

        if self.provider == "anthropic":
            reconstructed = StreamingCCRHandler(
                self.response_handler, provider="anthropic"
            )._parse_sse_stream(bytes(reconstruction))
        else:
            reconstructed = completed_response or {}

        residual = self.response_handler.residual_ccr_status(reconstructed, self.provider)
        if residual == RESIDUAL_CCR_SKIPPED_MIXED or detected_client_tool:
            for terminal in held_terminal:
                for scrubbed in _scrub_event(
                    _strip_private_from_terminal(terminal, provider=self.provider)
                ):
                    yield scrubbed
            return
        if residual == RESIDUAL_CCR_ERROR:
            final_response = await self.response_handler.handle_response(
                reconstructed,
                messages,
                tools,
                api_call_fn,
                self.provider,
            )
            if (
                self.response_handler.residual_ccr_status(final_response, self.provider)
                == RESIDUAL_CCR_ERROR
            ):
                yield _error_event(self.provider, "Unable to safely complete CCR retrieval.")
                return
        else:
            # A malformed stream claimed a private event but reconstructed no
            # call. Never re-expose it or pretend the turn completed.
            yield _error_event(self.provider, "Unable to reconstruct CCR retrieval request.")
            return

        final_response = _combine_usage(reconstructed, final_response)
        final_response = strip_internal_retrieve_calls(final_response, self.provider)
        final_response = scrub_client_payload(final_response)
        visible_initial = strip_internal_retrieve_calls(reconstructed, self.provider)
        for rendered_event in self.render_response(final_response):
            spliced = _splice_rendered_event(
                rendered_event,
                provider=self.provider,
                index_offset=max_visible_index + 1,
                sequence_offset=max_sequence + 1,
                visible_initial=visible_initial,
                final_response=final_response,
            )
            if spliced is not None:
                for scrubbed in _scrub_event(spliced):
                    yield scrubbed


class _SSEEventDecoder:
    def __init__(self) -> None:
        self.buffer = bytearray()

    def feed(self, chunk: bytes) -> list[bytes]:
        self.buffer.extend(chunk)
        if len(self.buffer) > MAX_CCR_SSE_EVENT_BYTES:
            self.buffer.clear()
            raise ValueError("unterminated CCR SSE event exceeded the bounded holdback")
        events: list[bytes] = []
        while True:
            boundary = _next_boundary(self.buffer)
            if boundary is None:
                break
            end, length = boundary
            events.append(bytes(self.buffer[: end + length]))
            del self.buffer[: end + length]
        return events

    def finish(self) -> list[bytes]:
        if not self.buffer:
            return []
        tail = bytes(self.buffer)
        self.buffer.clear()
        return [tail]


def _next_boundary(buffer: bytearray) -> tuple[int, int] | None:
    candidates = [(buffer.find(b"\n\n"), 2), (buffer.find(b"\r\n\r\n"), 4)]
    present = [candidate for candidate in candidates if candidate[0] >= 0]
    return min(present, default=None)


def _parse_sse_event(raw_event: bytes) -> dict[str, Any] | None:
    data_lines = []
    for line in raw_event.decode("utf-8", errors="replace").splitlines():
        if line.startswith("data:"):
            data_lines.append(line[5:].removeprefix(" "))
    if not data_lines or data_lines == ["[DONE]"]:
        return None
    try:
        value = json.loads("\n".join(data_lines))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _is_done_event(raw_event: bytes) -> bool:
    return any(line.strip() == b"data: [DONE]" for line in raw_event.splitlines())


def _encode_sse_event(event: dict[str, Any]) -> bytes:
    event_type = event.get("type")
    prefix = f"event: {event_type}\n" if event_type else ""
    return f"{prefix}data: {json.dumps(event)}\n\n".encode()


def _scrub_event(raw_event: bytes) -> list[bytes]:
    return scrub_sse_chunks([raw_event])


def _strip_private_from_terminal(raw_event: bytes, *, provider: str) -> bytes:
    event = _parse_sse_event(raw_event)
    if event is None:
        return raw_event
    if provider == "openai_responses" and isinstance(event.get("response"), dict):
        event["response"] = strip_internal_retrieve_calls(event["response"], "openai_responses")
    return _encode_sse_event(scrub_client_payload(event))


def _splice_rendered_event(
    raw_event: bytes,
    *,
    provider: str,
    index_offset: int,
    sequence_offset: int,
    visible_initial: dict[str, Any],
    final_response: dict[str, Any],
) -> bytes | None:
    event = _parse_sse_event(raw_event)
    if event is None:
        return raw_event
    event_type = str(event.get("type") or "")
    if provider == "anthropic":
        if event_type == "message_start":
            return None
        if isinstance(event.get("index"), int):
            event["index"] += index_offset
        return _encode_sse_event(event)

    if event_type in {"response.created", "response.in_progress"}:
        return None
    if isinstance(event.get("output_index"), int):
        event["output_index"] += index_offset
    if isinstance(event.get("sequence_number"), int):
        event["sequence_number"] += sequence_offset
    if event_type == "response.completed" and isinstance(event.get("response"), dict):
        initial_output = visible_initial.get("output") or []
        final_output = final_response.get("output") or []
        event["response"] = {
            **event["response"],
            "output": [*initial_output, *final_output],
        }
    return _encode_sse_event(event)


def _error_event(provider: str, message: str) -> bytes:
    if provider == "anthropic":
        return _encode_sse_event(
            {"type": "error", "error": {"type": "api_error", "message": message}}
        )
    return _encode_sse_event(
        {"type": "error", "error": {"type": "server_error", "message": message}}
    )


def _combine_usage(
    initial_response: dict[str, Any], final_response: dict[str, Any]
) -> dict[str, Any]:
    """Report the total provider usage charged across retrieval and continuation."""
    initial_usage = initial_response.get("usage")
    final_usage = final_response.get("usage")
    if not isinstance(initial_usage, dict) or not isinstance(final_usage, dict):
        return final_response
    combined = dict(final_usage)
    for key, value in initial_usage.items():
        if isinstance(value, int) and isinstance(combined.get(key, 0), int):
            combined[key] = value + int(combined.get(key, 0))
    return {**final_response, "usage": combined}
