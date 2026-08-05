"""Cloud Code Assist route classification helpers."""

from __future__ import annotations

CLOUDCODE_INTERNAL_PREFIX = "v1internal:"
CLOUDCODE_VERSIONED_INTERNAL_PREFIX = "v1/v1internal:"


def normalize_cloudcode_passthrough_path(path: str) -> str | None:
    """Return the canonical Cloud Code internal path, or ``None`` when unrelated."""
    clean_path = path.lstrip("/")
    if not clean_path.startswith((CLOUDCODE_INTERNAL_PREFIX, CLOUDCODE_VERSIONED_INTERNAL_PREFIX)):
        return None
    if clean_path.startswith("v1/"):
        clean_path = clean_path[3:]
    return f"/{clean_path}"
