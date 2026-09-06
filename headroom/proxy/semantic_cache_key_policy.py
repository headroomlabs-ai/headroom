"""Pure key policy for proxy semantic response cache."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def strip_cache_control(obj: Any) -> Any:
    """Recursively drop prompt-cache annotations, preserving schema properties."""
    return _strip_cache_control(obj, preserve_key=False)


def _strip_cache_control(obj: Any, *, preserve_key: bool) -> Any:
    """Strip directives while retaining names inside JSON Schema ``properties``.

    ``cache_control`` is also a valid user-defined JSON Schema property name.
    Its value still needs normal recursion because that property's schema may
    itself contain prompt-cache annotations.
    """
    if isinstance(obj, dict):
        return {
            k: _strip_cache_control(v, preserve_key=k == "properties")
            for k, v in obj.items()
            if k != "cache_control" or preserve_key
        }
    if isinstance(obj, list):
        return [_strip_cache_control(item, preserve_key=False) for item in obj]
    return obj


def compute_semantic_cache_key(
    messages: list[dict],
    model: str,
    **key_fields: Any,
) -> str:
    """Compute a stable cache key from request content and shaping fields.

    ``cache_control`` is stripped from ``messages`` as well as the shaping
    fields: it is a prompt-caching directive for the upstream provider that
    never changes the generated completion, so a moved breakpoint must not
    fragment the key. Messages are the primary key component and, on the
    Anthropic path, the most common place a client (e.g. Claude Code) moves a
    breakpoint between turns, so leaving them un-stripped defeated the strip for
    the field that matters most.
    """
    normalized = json.dumps(
        {
            "model": model,
            "messages": strip_cache_control(messages),
            **{k: strip_cache_control(v) for k, v in key_fields.items()},
        },
        sort_keys=True,
    )
    return hashlib.sha256(normalized.encode()).hexdigest()[:32]
