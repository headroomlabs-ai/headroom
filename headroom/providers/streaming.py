"""Provider-owned streaming parser and policy helpers."""

from __future__ import annotations

import json
import logging
from typing import Any

from headroom.proxy.helpers import parse_sse_events_from_byte_buffer

logger = logging.getLogger("headroom.proxy")


def extract_anthropic_cache_ttl_metrics(usage: dict[str, Any] | None) -> tuple[int, int]:
    """Extract observed Anthropic cache-write TTL bucket usage."""
    if not isinstance(usage, dict):
        return (0, 0)
    cache_creation = usage.get("cache_creation")
    if not isinstance(cache_creation, dict):
        return (0, 0)
    return (
        int(cache_creation.get("ephemeral_5m_input_tokens", 0) or 0),
        int(cache_creation.get("ephemeral_1h_input_tokens", 0) or 0),
    )


def parse_sse_usage_chunk(chunk: bytes, provider: str) -> dict[str, int] | None:
    """Parse provider usage from a single SSE byte chunk."""
    try:
        buf = bytearray(chunk)
        return parse_sse_usage_events(parse_sse_events_from_byte_buffer(buf), provider)
    except (UnicodeDecodeError, KeyError, TypeError) as exc:
        logger.debug("SSE usage parsing error for %s: %s", provider, exc)
        return None


def parse_sse_usage_buffer(
    stream_state: dict[str, Any],
    provider: str,
) -> dict[str, int] | None:
    """Parse provider usage from a mutable buffered SSE stream state."""
    buffer = stream_state["sse_buffer"]
    return parse_sse_usage_events(parse_sse_events_from_byte_buffer(buffer), provider)


def parse_sse_usage_events(
    events: list[tuple[str | None, str]],
    provider: str,
) -> dict[str, int] | None:
    """Parse provider usage from complete SSE events."""
    usage_found: dict[str, int] = {}

    for _event_name, data_str in events:
        if not data_str or data_str == "[DONE]":
            continue

        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        if provider == "anthropic":
            event_type = data.get("type", "")
            if event_type == "message_start":
                msg = data.get("message", {})
                msg_usage = msg.get("usage", {})
                if msg_usage:
                    usage_found["input_tokens"] = msg_usage.get("input_tokens", 0)
                    usage_found["cache_read_input_tokens"] = msg_usage.get(
                        "cache_read_input_tokens", 0
                    )
                    usage_found["cache_creation_input_tokens"] = msg_usage.get(
                        "cache_creation_input_tokens", 0
                    )
                    cache_write_5m, cache_write_1h = extract_anthropic_cache_ttl_metrics(msg_usage)
                    usage_found["cache_creation_ephemeral_5m_input_tokens"] = cache_write_5m
                    usage_found["cache_creation_ephemeral_1h_input_tokens"] = cache_write_1h
                    logger.debug(
                        "[CACHE] Anthropic usage: input=%s, cache_read=%s, cache_write=%s",
                        usage_found.get("input_tokens"),
                        usage_found.get("cache_read_input_tokens"),
                        usage_found.get("cache_creation_input_tokens"),
                    )
            elif event_type == "message_delta":
                delta_usage = data.get("usage", {})
                if delta_usage:
                    usage_found["output_tokens"] = delta_usage.get("output_tokens", 0)

        elif provider == "openai":
            chunk_usage = data.get("usage")
            if not isinstance(chunk_usage, dict):
                response = data.get("response")
                if isinstance(response, dict):
                    chunk_usage = response.get("usage")
            if isinstance(chunk_usage, dict):
                input_tokens = chunk_usage.get("prompt_tokens")
                if input_tokens is None:
                    input_tokens = chunk_usage.get("input_tokens", 0)
                output_tokens = chunk_usage.get("completion_tokens")
                if output_tokens is None:
                    output_tokens = chunk_usage.get("output_tokens", 0)
                usage_found["input_tokens"] = _usage_int(input_tokens)
                usage_found["output_tokens"] = _usage_int(output_tokens)
                details = (
                    chunk_usage.get("prompt_tokens_details")
                    or chunk_usage.get("input_tokens_details")
                    or {}
                )
                if isinstance(details, dict):
                    usage_found["cache_read_input_tokens"] = _usage_int(
                        details.get("cached_tokens")
                    )

        elif provider == "gemini":
            usage_meta = data.get("usageMetadata")
            if usage_meta:
                usage_found["input_tokens"] = usage_meta.get("promptTokenCount", 0)
                usage_found["output_tokens"] = usage_meta.get("candidatesTokenCount", 0)
                usage_found["cache_read_input_tokens"] = usage_meta.get(
                    "cachedContentTokenCount", 0
                )

    return usage_found if usage_found else None


def parse_sse_to_response(sse_data: str, provider: str) -> dict[str, Any] | None:
    """Reconstruct a provider response from complete SSE text."""
    if provider != "anthropic":
        return None

    response: dict[str, Any] = {"content": [], "usage": {}}
    blocks_by_index: dict[int, dict[str, Any]] = {}
    current_block: dict[str, Any] | None = None

    for line in sse_data.split("\n"):
        if not line.startswith("data: "):
            continue
        data_str = line[6:].strip()
        if not data_str or data_str == "[DONE]":
            continue

        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        event_type = data.get("type", "")

        if event_type == "message_start":
            msg = data.get("message", {})
            response["id"] = msg.get("id")
            response["model"] = msg.get("model")
            response["role"] = msg.get("role", "assistant")
            response["stop_reason"] = msg.get("stop_reason")
            if msg.get("usage"):
                response["usage"].update(msg["usage"])

        elif event_type == "content_block_start":
            block = data.get("content_block", {})
            block_index = data.get("index", len(response["content"]))
            current_block = _start_anthropic_content_block(block, block_index)
            blocks_by_index[block_index] = current_block

        elif event_type == "content_block_delta":
            idx = data.get("index")
            target = (blocks_by_index.get(idx) if idx is not None else None) or current_block
            if target is not None:
                _apply_anthropic_content_delta(target, data.get("delta", {}))

        elif event_type == "content_block_stop":
            idx = data.get("index")
            target = (blocks_by_index.get(idx) if idx is not None else None) or current_block
            if target is not None:
                _finalize_anthropic_content_block(target)
                if target not in response["content"]:
                    response["content"].append(target)
                current_block = None

        elif event_type == "message_delta":
            delta = data.get("delta", {})
            if delta.get("stop_reason"):
                response["stop_reason"] = delta["stop_reason"]
            if data.get("usage"):
                response["usage"].update(data["usage"])

    return response if response.get("content") else None


def response_to_sse(response: dict[str, Any], provider: str) -> list[bytes]:
    """Convert a response dict back to provider SSE bytes."""
    if provider != "anthropic":
        return []

    events: list[bytes] = []
    msg_start = {
        "type": "message_start",
        "message": {
            "id": response.get("id", "msg_generated"),
            "type": "message",
            "role": response.get("role", "assistant"),
            "model": response.get("model", "unknown"),
            "content": [],
            "stop_reason": None,
            "usage": response.get("usage", {}),
        },
    }
    events.append(f"event: message_start\ndata: {json.dumps(msg_start)}\n\n".encode())

    for idx, block in enumerate(response.get("content", [])):
        if block.get("type") == "text":
            block_start = {
                "type": "content_block_start",
                "index": idx,
                "content_block": {"type": "text", "text": ""},
            }
        elif block.get("type") == "tool_use":
            block_start = {
                "type": "content_block_start",
                "index": idx,
                "content_block": {
                    "type": "tool_use",
                    "id": block.get("id", f"toolu_{idx}"),
                    "name": block.get("name", ""),
                    "input": {},
                },
            }
        else:
            continue

        events.append(f"event: content_block_start\ndata: {json.dumps(block_start)}\n\n".encode())

        if block.get("type") == "text" and block.get("text"):
            delta = {
                "type": "content_block_delta",
                "index": idx,
                "delta": {"type": "text_delta", "text": block["text"]},
            }
            events.append(f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n".encode())
        elif block.get("type") == "tool_use" and block.get("input"):
            delta = {
                "type": "content_block_delta",
                "index": idx,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(block["input"]),
                },
            }
            events.append(f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n".encode())

        block_stop = {"type": "content_block_stop", "index": idx}
        events.append(f"event: content_block_stop\ndata: {json.dumps(block_stop)}\n\n".encode())

    msg_delta = {
        "type": "message_delta",
        "delta": {"stop_reason": response.get("stop_reason", "end_turn")},
        "usage": {"output_tokens": response.get("usage", {}).get("output_tokens", 0)},
    }
    events.append(f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n".encode())
    events.append(b'event: message_stop\ndata: {"type": "message_stop"}\n\n')
    return events


def uses_upstream_token_denominator(provider: str) -> bool:
    """Return whether provider-stream usage should replace proxy token estimates."""
    return provider in {"openai", "gemini"}


def supports_stream_memory_tools(provider: str) -> bool:
    """Return whether streaming memory tool detection is supported."""
    return provider == "anthropic"


def supports_prefix_response_append(provider: str) -> bool:
    """Return whether prefix tracking can append reconstructed streamed response."""
    return provider == "anthropic"


def supports_codex_wire_debug(provider: str, url: str) -> bool:
    """Return whether Codex wire debugging should capture this stream."""
    return provider == "openai" and "/responses" in url


def _usage_int(value: Any) -> int:
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0


def _start_anthropic_content_block(block: dict[str, Any], block_index: int) -> dict[str, Any]:
    current_block: dict[str, Any] = {
        "type": block.get("type"),
        "index": block_index,
    }
    btype = block.get("type")
    if btype == "text":
        current_block["text"] = block.get("text", "")
    elif btype == "tool_use":
        current_block["id"] = block.get("id")
        current_block["name"] = block.get("name")
        current_block["input"] = {}
    elif btype == "thinking":
        current_block["thinking_buffer"] = block.get("thinking", "")
        if "signature" in block:
            current_block["signature"] = block["signature"]
    elif btype == "redacted_thinking" and "data" in block:
        current_block["data"] = block["data"]
    return current_block


def _apply_anthropic_content_delta(target: dict[str, Any], delta: dict[str, Any]) -> None:
    dtype = delta.get("type")
    if dtype == "text_delta":
        target["text"] = target.get("text", "") + delta.get("text", "")
    elif dtype == "input_json_delta":
        target["_partial_json"] = target.get("_partial_json", "") + delta.get("partial_json", "")
    elif dtype == "thinking_delta":
        target["thinking_buffer"] = target.get("thinking_buffer", "") + delta.get("thinking", "")
    elif dtype == "signature_delta":
        if "signature" in delta:
            target["signature"] = delta["signature"]
    elif dtype == "citations_delta":
        citations = target.setdefault("citations", [])
        citation = delta.get("citation")
        if citation is not None:
            citations.append(citation)


def _finalize_anthropic_content_block(target: dict[str, Any]) -> None:
    if target.get("type") == "tool_use" and "_partial_json" in target:
        try:
            target["input"] = json.loads(target["_partial_json"])
        except json.JSONDecodeError:
            target["input"] = {}
        del target["_partial_json"]
    if target.get("type") == "thinking" and "thinking_buffer" in target:
        target["thinking"] = target.pop("thinking_buffer")
