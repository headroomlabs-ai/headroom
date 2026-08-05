"""Provider upstream target resolution for proxy routes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from headroom.providers.codex import resolve_codex_routing
from headroom.providers.codex.endpoints import CHATGPT_BACKEND_API_URL
from headroom.providers.vertex import vertex_target_for_location as _vertex_target_for_location

LEGACY_API_TARGET_ATTRS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_URL",
    "openai": "OPENAI_API_URL",
    "gemini": "GEMINI_API_URL",
    "cloudcode": "CLOUDCODE_API_URL",
    "vertex": "VERTEX_API_URL",
}


def api_target(proxy: Any, provider_name: str) -> str:
    """Return the proxy target for a provider, honoring legacy proxy attributes."""
    legacy_attr = LEGACY_API_TARGET_ATTRS[provider_name]
    return cast(str, getattr(proxy, legacy_attr, proxy.provider_runtime.api_target(provider_name)))


def vertex_target_for_location(proxy: Any, location: str) -> str:
    """Resolve the Vertex upstream host for a request, region-aware."""
    return _vertex_target_for_location(api_target(proxy, "vertex"), location)


def select_passthrough_base_url(proxy: Any, headers: Mapping[str, str]) -> str:
    """Resolve the upstream base URL for catch-all proxy passthrough requests."""
    routing = resolve_codex_routing(headers)
    if routing.is_chatgpt_auth:
        return CHATGPT_BACKEND_API_URL
    if headers.get("x-goog-api-key"):
        return api_target(proxy, "gemini")
    if headers.get("api-key"):
        azure_base = headers.get("x-headroom-base-url", "")
        if azure_base:
            return azure_base.rstrip("/")
    provider_name = proxy.provider_runtime.model_metadata_provider(headers)
    return api_target(proxy, provider_name)
