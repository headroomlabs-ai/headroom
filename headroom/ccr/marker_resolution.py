"""Inline resolution of CCR markers on the response path.

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
from typing import Any

from ..cache.compression_store import (
    CompressionStore,
    format_retrieval_miss_detail,
    get_compression_store,
)

logger = logging.getLogger(__name__)

# Matches the opaque-blob and row-offload forms emitted by SmartCrusher, plus
# bracketed markers emitted by the text, config, read, log, and Kompress paths.
# HASH is 12-24 hex chars; see headroom/ccr/tool_injection.py for the same
# ownership boundary used on the injection side.
_MARKER_RE = re.compile(
    r"<<ccr:([a-f0-9]{12,24})[^>]*>>"
    r"|\[[^\]]*?Retrieve (?:more|original): hash=([a-f0-9]{12,24})[^\]]*\]",
    re.IGNORECASE,
)


def resolve_markers_in_text(text: str, *, store: CompressionStore | None = None) -> str:
    """Replace supported CCR markers in ``text`` with their original content.

    A miss (expired/evicted/unknown hash) can't be reported back to the
    model on this path — there's no tool-call round-trip — so the marker is
    left in place with the miss reason appended rather than raising.
    """
    if not (
        "<<ccr:" in text or "Retrieve more: hash=" in text or "Retrieve original: hash=" in text
    ):
        return text

    resolved_store = store or get_compression_store()

    def _replace(match: re.Match[str]) -> str:
        hash_key = (match.group(1) or match.group(2)).lower()
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
    """Recursively resolve CCR markers in every string field of a payload.

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
