"""xAI model metadata normalization for Grok's custom endpoint."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from headroom.providers.grok.runtime import DEFAULT_API_URL

_XAI_API_HOST = urlsplit(DEFAULT_API_URL).hostname


def is_xai_model_list_target(base_url: str) -> bool:
    """Return whether ``base_url`` is the exact xAI model-list target."""
    parsed = urlsplit(base_url)
    return (
        parsed.hostname is not None
        and _XAI_API_HOST is not None
        and parsed.hostname.lower() == _XAI_API_HOST.lower()
        and parsed.path.rstrip("/") in {"", "/v1"}
    )


def normalize_xai_model_metadata(payload: Any) -> Any:
    """Add Grok's context alias to eligible xAI model-list entries."""
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        return None

    normalized_entries: list[Any] = []
    changed = False
    for entry in payload["data"]:
        if not isinstance(entry, Mapping):
            normalized_entries.append(entry)
            continue
        context_length = entry.get("context_length")
        if (
            isinstance(context_length, int)
            and not isinstance(context_length, bool)
            and context_length > 0
            and "context_window" not in entry
            and "contextWindow" not in entry
        ):
            normalized_entry = dict(entry)
            normalized_entry["context_window"] = context_length
            normalized_entries.append(normalized_entry)
            changed = True
        else:
            normalized_entries.append(entry)

    if not changed:
        return None
    normalized_payload = dict(payload)
    normalized_payload["data"] = normalized_entries
    return normalized_payload
