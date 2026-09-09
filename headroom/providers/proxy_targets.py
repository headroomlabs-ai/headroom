"""Provider upstream target resolution for proxy routes."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, cast

from headroom.proxy.upstream_guard import is_safe_upstream_url

LEGACY_API_TARGET_ATTRS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_URL",
    "openai": "OPENAI_API_URL",
}


def api_target(proxy: Any, provider_name: str) -> str:
    """Return the proxy target for a provider, honoring legacy proxy attributes."""
    legacy_attr = LEGACY_API_TARGET_ATTRS[provider_name]
    return cast(str, getattr(proxy, legacy_attr, proxy.provider_runtime.api_target(provider_name)))


logger = logging.getLogger("headroom.proxy")


def select_passthrough_base_url(
    proxy: Any, headers: Mapping[str, str], path: str | None = None
) -> str:
    """Resolve the upstream base URL for catch-all proxy passthrough requests."""
    del path
    if headers.get("api-key"):
        azure_base = headers.get("x-headroom-base-url", "")
        if azure_base:
            # `api-key` is attacker-supplied too, so this branch is reachable by
            # anyone who can send a header, and it returns the destination the
            # caller named. Guard against SSRF into loopback/RFC1918/cloud-
            # metadata space (CVE-2026-77775).
            if is_safe_upstream_url(azure_base):
                return azure_base.rstrip("/")
            logger.warning("ignoring unsafe x-headroom-base-url override: %r", azure_base)
    provider_name = proxy.provider_runtime.model_metadata_provider(headers)
    return api_target(proxy, provider_name)
