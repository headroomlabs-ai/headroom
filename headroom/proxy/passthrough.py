"""Shared passthrough routing and telemetry helpers."""

from __future__ import annotations

from urllib.parse import urlparse

OPENCODE_ZEN_HOSTS = {"opencode.ai", "www.opencode.ai"}


def custom_base_passthrough_telemetry(method: str, path: str, base_url: str) -> tuple[str, str]:
    """Return passthrough telemetry metadata for narrow custom-base exceptions."""
    # OpenCode Zen sends provider-prefixed OpenAI-compatible traffic through
    # custom-base routing. Keep this exact to avoid labeling arbitrary
    # custom-base tool traffic as LLM provider telemetry.
    if method.upper() != "POST":
        return "", ""
    try:
        host = (urlparse(base_url.strip()).hostname or "").lower()
    except ValueError:
        return "", ""
    if host not in OPENCODE_ZEN_HOSTS:
        return "", ""
    normalized_path = path[1:] if path.startswith("/") else path
    if normalized_path == "zen/v1/chat/completions":
        return "chat/completions", "zen"
    return "", ""
