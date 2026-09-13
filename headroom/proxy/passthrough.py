"""Shared passthrough routing and telemetry helpers."""

from __future__ import annotations

from urllib.parse import urlparse

OPENCODE_ZEN_HOSTS = {"opencode.ai", "www.opencode.ai"}
XAI_HOSTS = {"api.x.ai"}


def custom_base_passthrough_telemetry(method: str, path: str, base_url: str) -> tuple[str, str]:
    """Return passthrough telemetry metadata for narrow custom-base exceptions."""
    # Known OpenAI-compatible vendors reached via custom-base routing. Kept as
    # exact host allowlists to avoid labeling arbitrary custom-base tool
    # traffic as LLM provider telemetry.
    #
    # - OpenCode Zen sends provider-prefixed traffic (zen/v1/...).
    # - Grok Build (x.ai) sends plain OpenAI chat completions; attributing it
    #   as "xai" lets savings rollups distinguish grok traffic from OpenAI's,
    #   which downstream dashboards display as separate rows.
    if method.upper() != "POST":
        return "", ""
    try:
        host = (urlparse(base_url.strip()).hostname or "").lower()
    except ValueError:
        return "", ""
    normalized_path = path[1:] if path.startswith("/") else path
    if host in OPENCODE_ZEN_HOSTS and normalized_path == "zen/v1/chat/completions":
        return "chat/completions", "zen"
    if host in XAI_HOSTS and normalized_path == "v1/chat/completions":
        return "chat/completions", "xai"
    return "", ""
