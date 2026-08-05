"""Formatting policy for opt-in proxy wire debug artifacts."""

from __future__ import annotations

import json
from typing import Any

WIRE_DEBUG_NAME_MAX_CHARS = 80


def safe_wire_debug_name(value: str) -> str:
    """Return a filename-safe wire-debug name fragment."""
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)[
        :WIRE_DEBUG_NAME_MAX_CHARS
    ]


def wire_debug_preview(value: Any, *, max_chars: int | None = None) -> str:
    """Return the compact wire payload preview used in proxy logs."""
    try:
        if isinstance(value, bytes):
            text = value.decode("utf-8", errors="replace")
        elif isinstance(value, str):
            text = value
        elif value is None:
            return ""
        else:
            text = json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
    except Exception:
        text = repr(value)

    text = " ".join(text.split())
    if max_chars is not None and len(text) > max_chars:
        return text[: max_chars - 1] + "…"
    return text
