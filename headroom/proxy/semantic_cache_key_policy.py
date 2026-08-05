"""Pure key policy for proxy semantic response cache."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def strip_cache_control(obj: Any) -> Any:
    """Recursively drop ``cache_control`` annotations before hashing."""
    if isinstance(obj, dict):
        return {k: strip_cache_control(v) for k, v in obj.items() if k != "cache_control"}
    if isinstance(obj, list):
        return [strip_cache_control(item) for item in obj]
    return obj


def compute_semantic_cache_key(
    messages: list[dict],
    model: str,
    **key_fields: Any,
) -> str:
    """Compute a stable cache key from request content and shaping fields."""
    normalized = json.dumps(
        {
            "model": model,
            "messages": messages,
            **{k: strip_cache_control(v) for k, v in key_fields.items()},
        },
        sort_keys=True,
    )
    return hashlib.sha256(normalized.encode()).hexdigest()[:32]
