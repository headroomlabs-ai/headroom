"""Case-insensitive header helpers for Codex provider adapters."""

from __future__ import annotations

from collections.abc import Mapping


def header_name(headers: Mapping[str, str], name: str) -> str | None:
    """Return the existing header key matching ``name`` case-insensitively."""
    lowered = name.lower()
    for header_name_value in headers:
        if header_name_value.lower() == lowered:
            return header_name_value
    return None


def drop_header(headers: dict[str, str], name: str) -> None:
    """Remove the header matching ``name`` case-insensitively from ``headers``."""
    existing_name = header_name(headers, name)
    if existing_name is not None:
        headers.pop(existing_name, None)
