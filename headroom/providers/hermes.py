"""Hermes Studio scoped coding-agent proxy support.

Hermes owns authentication and protocol adaptation for its scoped proxy routes.
This module owns the small Headroom integration point: safely compress the chat
portion of those requests before the generic proxy forwards them upstream.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("headroom.providers.hermes")

_CHAT_ROLES = frozenset({"user", "assistant"})
_CLAUDE_TEXT_PART_TYPES = frozenset({"text"})
# Responses uses protocol-specific text part names; normalize these only while
# passing a message through the generic compressor, then restore them exactly.
_RESPONSES_TEXT_PART_TYPES = frozenset({"text", "input_text", "output_text"})
_CODEX_RESPONSES_SUFFIX = "/v1/responses"
_CLAUDE_MESSAGES_SUFFIX = "/v1/messages"


def is_scoped_coding_agent_path(path: str) -> bool:
    """Return whether *path* is a Hermes scoped coding-agent endpoint."""
    return (path.startswith("/api/codex-proxy/") and path.endswith(_CODEX_RESPONSES_SUFFIX)) or (
        path.startswith("/api/claude-code-proxy/") and path.endswith(_CLAUDE_MESSAGES_SUFFIX)
    )


def compress_scoped_passthrough_body(
    path: str,
    body: bytes,
    *,
    optimize: bool,
    bypass: bool,
) -> bytes:
    """Compress supported Hermes request bodies, otherwise return *body* unchanged.

    The adapter deliberately understands only Hermes's two scoped routes. It
    leaves system, tool, reasoning, and non-dictionary input items untouched;
    only user/assistant messages are handed to Headroom's compressor.
    """
    if not optimize or bypass or not is_scoped_coding_agent_path(path):
        return body

    try:
        payload = json.loads(body)
        if not isinstance(payload, dict):
            return body
        model = str(payload.get("model") or "").strip()
        if not model:
            return body

        if path.startswith("/api/claude-code-proxy/"):
            field_name = "messages"
            route_name = "claude-code"
        else:
            field_name = "input"
            route_name = "codex"

        raw_items = payload.get(field_name)
        if isinstance(raw_items, str) and route_name == "codex":
            compressed = _compress_messages(
                [{"role": "user", "content": raw_items}], model=model, route_name=route_name
            )
            if compressed is None:
                return body
            payload[field_name] = compressed
            return _encode_payload(payload)

        if not isinstance(raw_items, list):
            return body

        chat_indices = [
            index
            for index, item in enumerate(raw_items)
            if _is_compressible_chat_message(item, route_name=route_name)
        ]
        if not chat_indices:
            return body

        chat_messages = [raw_items[index] for index in chat_indices]
        messages_for_compression = (
            _normalize_responses_text_parts(chat_messages)
            if route_name == "codex"
            else chat_messages
        )
        compressed = _compress_messages(
            messages_for_compression, model=model, route_name=route_name
        )
        if compressed is None:
            return body
        if route_name == "codex":
            compressed = _restore_responses_text_parts(chat_messages, compressed)
        payload[field_name] = _splice_compressed_messages(raw_items, chat_indices, compressed)
        return _encode_payload(payload)
    except Exception as exc:  # Compression must never block Hermes passthrough.
        logger.info("Hermes passthrough compression skipped: %s", exc)
        return body


def _is_compressible_chat_message(item: Any, *, route_name: str) -> bool:
    """Return whether a message has only text content safe for compression."""
    if not isinstance(item, dict) or item.get("role") not in _CHAT_ROLES:
        return False
    content = item.get("content")
    if isinstance(content, str):
        return True
    if not isinstance(content, list) or not content:
        return False
    text_part_types = (
        _RESPONSES_TEXT_PART_TYPES if route_name == "codex" else _CLAUDE_TEXT_PART_TYPES
    )
    return all(
        isinstance(part, dict)
        and part.get("type") in text_part_types
        and isinstance(part.get("text"), str)
        for part in content
    )


def _normalize_responses_text_parts(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Adapt Responses text parts to the compressor's generic ``text`` shape."""
    normalized: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            normalized.append(message)
            continue
        normalized.append(
            {
                **message,
                "content": [{**part, "type": "text"} for part in content],
            }
        )
    return normalized


def _restore_responses_text_parts(
    original_messages: list[dict[str, Any]], compressed_messages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Restore Responses part types and metadata after generic compression.

    If the compressor unexpectedly changes the list/message/block structure,
    preserve the original message rather than risk producing an invalid
    Responses request.
    """
    if len(compressed_messages) != len(original_messages):
        return original_messages

    restored: list[dict[str, Any]] = []
    for original, compressed in zip(original_messages, compressed_messages, strict=True):
        if not isinstance(compressed, dict):
            return original_messages
        original_content = original.get("content")
        compressed_content = compressed.get("content")
        if isinstance(original_content, str):
            if not isinstance(compressed_content, str):
                return original_messages
            restored.append({**original, "content": compressed_content})
            continue
        if not isinstance(original_content, list) or not isinstance(compressed_content, list):
            return original_messages
        if len(compressed_content) != len(original_content):
            return original_messages

        restored_parts: list[dict[str, Any]] = []
        for original_part, compressed_part in zip(
            original_content, compressed_content, strict=True
        ):
            if not isinstance(compressed_part, dict) or not isinstance(
                compressed_part.get("text"), str
            ):
                return original_messages
            restored_parts.append({**original_part, "text": compressed_part["text"]})
        restored.append({**original, "content": restored_parts})
    return restored


def _compress_messages(
    messages: list[dict[str, Any]], *, model: str, route_name: str
) -> list[dict[str, Any]] | None:
    from headroom import compress as headroom_compress

    before_bytes = len(json.dumps(messages, ensure_ascii=False).encode("utf-8"))
    result = headroom_compress(messages=messages, model=model, optimize=True)
    compressed = list(result.messages)
    after_bytes = len(json.dumps(compressed, ensure_ascii=False).encode("utf-8"))
    logger.info(
        "Hermes %s passthrough compression: %d -> %d bytes (saved %d)",
        route_name,
        before_bytes,
        after_bytes,
        max(0, before_bytes - after_bytes),
    )
    return compressed


def _splice_compressed_messages(
    original_items: list[Any], chat_indices: list[int], compressed_items: list[dict[str, Any]]
) -> list[Any]:
    """Restore compressed chat messages to their original slots.

    A defensive fallback retains an original item if a compressor unexpectedly
    returns fewer messages than it received.
    """
    compressed_by_index = dict(zip(chat_indices, compressed_items, strict=False))
    return [compressed_by_index.get(index, item) for index, item in enumerate(original_items)]


def _encode_payload(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
