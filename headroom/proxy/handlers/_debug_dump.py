"""Shared helpers for diagnostic dumps of upstream-error (>=400) requests.

The dump can contain cleartext prompt / tool / system content, so it is OFF by
default, is never written in stateless mode, and content is redacted unless the
operator explicitly opts in to full content. Used by both the Anthropic and the
OpenAI handlers so the gating stays consistent across providers.
"""

from __future__ import annotations

import os
from typing import Any


def _debug_dump_mode(config: Any) -> str:
    """Return the diagnostic-dump mode for upstream (>=400) errors.

    - "off"      : nothing written (default, and forced in stateless mode)
    - "redacted" : structure, roles, and lengths only — content elided
                   (``HEADROOM_DEBUG_DUMP=1``/``true``/``on``/``redacted``)
    - "full"     : everything including content (``HEADROOM_DEBUG_DUMP=full``)
    """
    if getattr(config, "stateless", False):
        return "off"
    raw = os.environ.get("HEADROOM_DEBUG_DUMP", "").strip().lower()
    if raw in ("full", "all", "content"):
        return "full"
    if raw in ("1", "true", "yes", "on", "redacted"):
        return "redacted"
    return "off"


def _redact_debug_value(value: Any, _max_len: int = 80) -> Any:
    """Recursively elide long strings (likely prompt/tool content) while keeping
    structure, roles, type tags, ids, and other short fields for debugging.

    Note: short strings (<= ``_max_len``) are preserved, so the redacted tier is
    not a guarantee against leaking very short sensitive values — it is a
    best-effort convenience. The default ("off") writes nothing at all.
    """
    if isinstance(value, str):
        return value if len(value) <= _max_len else f"<redacted: {len(value)} chars>"
    if isinstance(value, dict):
        return {k: _redact_debug_value(v, _max_len) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_debug_value(v, _max_len) for v in value]
    return value
