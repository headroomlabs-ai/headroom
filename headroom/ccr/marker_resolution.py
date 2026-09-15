"""Inline resolution of ``<<ccr:...>>`` markers on the response path.

Normal CCR resolution relies on the ``headroom_retrieve`` tool: a marker is
redeemed when the model calls the tool back. That path assumes there's a
subsequent turn in which the model *can* call it. Callers that never see an
injected tool at all — e.g. Headroom running as a LiteLLM guardrail/proxy hop
with no tool-call turn in between (#2509) — have no way to redeem a marker,
so it leaks through as raw text.

This module provides an explicit, opt-in fallback (``--ccr-inline-resolve``):
scan the outgoing response for markers and substitute the original content
directly, instead of leaving the marker for the model to redeem later.
"""

from __future__ import annotations

import json
import logging
import re
from copy import deepcopy
from typing import Any

from ..cache.compression_store import (
    CompressionStore,
    format_retrieval_miss_detail,
    get_compression_store,
)

logger = logging.getLogger(__name__)

# Matches the opaque-blob marker form `<<ccr:HASH,KIND,SIZE>>` (and the
# row-offload form `<<ccr:HASH N_rows_offloaded>>`) emitted by SmartCrusher.
# HASH is 12-24 hex chars; see headroom/ccr/tool_injection.py for the same
# constant used on the injection side.
_MARKER_RE = re.compile(r"<<ccr:([a-f0-9]{12,24})([^>]*)>>")


def resolve_markers_in_text(text: str, *, store: CompressionStore | None = None) -> str:
    """Replace every ``<<ccr:HASH,...>>`` marker in ``text`` with its original content.

    A miss (expired/evicted/unknown hash) can't be reported back to the
    model on this path — there's no tool-call round-trip — so the marker is
    left in place with the miss reason appended rather than raising.
    """
    if "<<ccr:" not in text:
        return text

    resolved_store = store or get_compression_store()

    def _replace(match: re.Match[str]) -> str:
        hash_key = match.group(1)
        entry = resolved_store.retrieve(hash_key)
        if entry is not None:
            original = entry.original_content
            return original if isinstance(original, str) else json.dumps(original)

        get_status = getattr(resolved_store, "get_entry_status", None)
        status = get_status(hash_key, clean_expired=True) if callable(get_status) else None
        detail = format_retrieval_miss_detail(status) if status else "entry not found"
        logger.warning(f"CCR inline-resolve: marker {hash_key} unresolvable ({detail})")
        return f"{match.group(0)} [unresolved: {detail}]"

    return _MARKER_RE.sub(_replace, text)


def resolve_markers_in_response(response: Any, *, store: CompressionStore | None = None) -> Any:
    """Recursively resolve ``<<ccr:...>>`` markers in every string field of a payload.

    Walks the full response structure rather than picking out
    provider-specific fields (``content`` blocks, ``message.content``,
    Responses-API ``output`` items, ...) so it stays correct regardless of
    where a marker ends up, and doesn't need per-provider maintenance.
    """
    resolved_store = store or get_compression_store()
    if isinstance(response, str):
        return resolve_markers_in_text(response, store=resolved_store)
    if isinstance(response, list):
        return [resolve_markers_in_response(item, store=resolved_store) for item in response]
    if isinstance(response, dict):
        return {
            key: resolve_markers_in_response(value, store=resolved_store)
            for key, value in response.items()
        }
    return response


def scrub_markers_for_client(
    value: Any,
    *,
    store: CompressionStore | None = None,
    resolve_hits: bool = True,
) -> Any:
    """Losslessly resolve CCR markers, or replace misses with a safe placeholder.

    Unlike the opt-in inline resolver, this client-bound safety primitive never
    returns an unresolved internal marker.  It walks every string field so
    markers embedded in tool arguments are covered as well as visible prose.
    """
    resolved_store = store or get_compression_store()
    if isinstance(value, str):
        if "<<ccr:" not in value:
            return value

        def _replace(match: re.Match[str]) -> str:
            hash_key = match.group(1)
            entry = resolved_store.retrieve(hash_key)
            if entry is not None:
                if resolve_hits:
                    original = entry.original_content
                    return original if isinstance(original, str) else json.dumps(original)
                descriptor = match.group(2).lstrip(" ,")
                suffix = f": {descriptor}" if descriptor else ""
                return f"[compressed content{suffix}]"
            logger.warning("CCR egress scrub: marker %s unavailable", hash_key)
            descriptor = match.group(2).lstrip(" ,")
            suffix = f": {descriptor}" if descriptor else ""
            return f"[compressed content unavailable{suffix}]"

        return _MARKER_RE.sub(_replace, value)
    if isinstance(value, list):
        return [
            scrub_markers_for_client(item, store=resolved_store, resolve_hits=resolve_hits)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: scrub_markers_for_client(item, store=resolved_store, resolve_hits=resolve_hits)
            for key, item in value.items()
        }
    return value


def scrub_client_payload(
    value: Any,
    *,
    store: CompressionStore | None = None,
    in_tool_arguments: bool = False,
) -> Any:
    """Scrub a provider payload with context-sensitive marker semantics.

    Tool arguments need the exact original bytes so a client-owned write/bash
    call remains correct. Visible model prose gets a compact descriptor instead
    of unexpectedly expanding an arbitrarily large stored payload.
    """
    resolved_store = store or get_compression_store()
    if isinstance(value, str):
        return scrub_markers_for_client(
            value,
            store=resolved_store,
            resolve_hits=in_tool_arguments,
        )
    if isinstance(value, list):
        return [
            scrub_client_payload(
                item,
                store=resolved_store,
                in_tool_arguments=in_tool_arguments,
            )
            for item in value
        ]
    if isinstance(value, dict):
        value_type = value.get("type")
        is_tool_call = value_type in {"tool_use", "function_call"}
        delta_type = value.get("type") == "input_json_delta"
        return {
            key: scrub_client_payload(
                item,
                store=resolved_store,
                in_tool_arguments=(
                    in_tool_arguments
                    or (is_tool_call and key in {"input", "arguments"})
                    or (delta_type and key == "partial_json")
                    or "function_call_arguments" in str(value_type)
                ),
            )
            for key, item in value.items()
        }
    return value


def strip_internal_retrieve_calls(response: Any, provider: str) -> Any:
    """Remove Headroom's private retrieve call while preserving client tools.

    Mixed turns cannot be continued server-side because Headroom has no result
    for the client's other tool calls.  Returning Headroom's injected call is
    still invalid: the client never declared or implemented it.  Strip only the
    private call and leave every client-owned call and stop signal intact.
    """
    if not isinstance(response, dict):
        return response
    result = deepcopy(response)
    if provider == "anthropic":
        content = result.get("content")
        if isinstance(content, list):
            result["content"] = [
                block
                for block in content
                if not (
                    isinstance(block, dict)
                    and block.get("type") == "tool_use"
                    and block.get("name") == "headroom_retrieve"
                )
            ]
        return result
    if provider == "openai_responses":
        output = result.get("output")
        if isinstance(output, list):
            result["output"] = [
                item
                for item in output
                if not (
                    isinstance(item, dict)
                    and item.get("type") == "function_call"
                    and item.get("name") == "headroom_retrieve"
                )
            ]
        return result
    if provider == "openai":
        for choice in result.get("choices") or []:
            message = choice.get("message") if isinstance(choice, dict) else None
            if not isinstance(message, dict) or not isinstance(message.get("tool_calls"), list):
                continue
            message["tool_calls"] = [
                call
                for call in message["tool_calls"]
                if not (
                    isinstance(call, dict)
                    and isinstance(call.get("function"), dict)
                    and call["function"].get("name") == "headroom_retrieve"
                )
            ]
    return result
