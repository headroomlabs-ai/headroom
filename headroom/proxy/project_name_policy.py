"""Project-name normalization policy for proxy attribution."""

from __future__ import annotations

from typing import Any
from urllib.parse import unquote

PROJECT_NAME_MAX_LENGTH = 128


def sanitize_project_name(value: Any) -> str | None:
    """Normalize a client-supplied project name; ``None`` when unusable.

    Strips control characters, trims whitespace, and caps length so a
    misbehaving client cannot bloat persisted state or dashboard payloads.
    Percent-encoded values are decoded first so stored names match the original
    directory name.
    """
    if not isinstance(value, str):
        return None
    decoded = unquote(value)
    cleaned = "".join(ch for ch in decoded if ch.isprintable()).strip()
    if not cleaned:
        return None
    return cleaned[:PROJECT_NAME_MAX_LENGTH]
