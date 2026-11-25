"""Cloud Code Assist route classification helpers."""

from __future__ import annotations

CLOUDCODE_INTERNAL_PREFIX = "v1internal:"
CLOUDCODE_VERSIONED_INTERNAL_PREFIX = "v1/v1internal:"

ANTIGRAVITY_DAILY_API_URL = "https://daily-cloudcode-pa.sandbox.googleapis.com"


def normalize_cloudcode_passthrough_path(path: str) -> str | None:
    """Return the canonical Cloud Code internal path, or ``None`` when unrelated."""
    clean_path = path.lstrip("/")
    if not clean_path.startswith((CLOUDCODE_INTERNAL_PREFIX, CLOUDCODE_VERSIONED_INTERNAL_PREFIX)):
        return None
    if clean_path.startswith("v1/"):
        clean_path = clean_path[3:]
    return f"/{clean_path}"


def is_antigravity_user_agent(user_agent: str | None) -> bool:
    """Return True when a ``User-Agent`` header identifies an Antigravity client.

    This is the header-only mirror of the Gemini handler's Antigravity
    detection: the passthrough route runs before the request body is parsed,
    so the body signals the handler also accepts (``requestType`` /
    ``userAgent``) are not available there.
    """
    if not user_agent:
        return False
    return user_agent.lower().startswith("antigravity/")


def cloudcode_passthrough_target(user_agent: str | None, default_target: str) -> str:
    """Resolve the Cloud Code passthrough upstream for a request.

    Antigravity clients must reach the daily sandbox host — the same host
    the Gemini handler resolves for ``generateContent`` — while every other
    client keeps the configured Cloud Code target.
    """
    if is_antigravity_user_agent(user_agent):
        return ANTIGRAVITY_DAILY_API_URL
    return default_target
