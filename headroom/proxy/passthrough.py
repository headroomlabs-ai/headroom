"""Shared passthrough routing and telemetry helpers."""

from __future__ import annotations

from urllib.parse import urlparse

OPENCODE_ZEN_HOSTS = {"opencode.ai", "www.opencode.ai"}

# Chat upstreams we can name, keyed by exact host. Bounded by design: the
# request-controlled ``x-headroom-base-url`` must never become a telemetry
# label, or a client sending a fresh hostname per request grows the
# per-provider stores and exported series without limit (review on #3759).
CUSTOM_BASE_CHAT_PROVIDERS = {
    "api.openai.com": "openai",
    "api.z.ai": "zai",
    "api.meta.ai": "meta",
}

# Fixed label for chat traffic on any other custom base.
CUSTOM_BASE_PROVIDER = "custom"


def is_opencode_zen_base(base_url: str | None) -> bool:
    """Return True when ``base_url`` targets the OpenCode Zen gateway.

    Zen validates OpenCode-client attribution on the wire, so requests that
    Headroom rewrites are rejected even though the caller *is* the OpenCode
    client. Callers that would otherwise change the upstream request shape
    should stay transparent when this returns True.
    """
    if not base_url:
        return False
    try:
        host = (urlparse(base_url.strip()).hostname or "").lower()
    except ValueError:
        return False
    return host in OPENCODE_ZEN_HOSTS


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
    normalized_path = path[1:] if path.startswith("/") else path
    if host in OPENCODE_ZEN_HOSTS:
        if normalized_path == "zen/v1/chat/completions":
            return "chat/completions", "zen"
        return "", ""
    # Known OpenAI-compatible chat hosts get a fixed name; chat-completions
    # traffic on any other custom base is the shared "custom" bucket and
    # everything else stays unnamed so tool traffic is not LLM telemetry.
    # Match whole path segments so ``/v1/notchat/completions`` stays unnamed.
    provider = CUSTOM_BASE_CHAT_PROVIDERS.get(host)
    is_chat_path = normalized_path == "chat/completions" or normalized_path.endswith(
        "/chat/completions"
    )
    if provider is not None and is_chat_path:
        return "chat/completions", provider
    return "", ""
