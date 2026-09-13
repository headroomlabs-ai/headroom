"""Provider upstream target resolution for proxy routes."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, cast

from headroom.copilot_auth import (
    copilot_completions_base_url,
    is_copilot_completions_host,
    is_copilot_completions_path,
)
from headroom.providers.codex import resolve_codex_routing
from headroom.providers.codex.endpoints import CHATGPT_BACKEND_API_URL
from headroom.providers.codex.runtime import DEFAULT_API_URL as DEFAULT_OPENAI_API_URL
from headroom.providers.grok.runtime import DEFAULT_API_URL as XAI_API_URL
from headroom.providers.grok.runtime import is_grok_cli_request
from headroom.providers.vertex import vertex_target_for_location as _vertex_target_for_location
from headroom.proxy.upstream_guard import is_safe_upstream_url

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


logger = logging.getLogger("headroom.proxy")


def route_grok_to_xai(headers: Mapping[str, str], openai_target: str) -> bool:
    """Return True when Grok CLI traffic should be redirected to ``api.x.ai``.

    Grok CLI cannot set ``x-headroom-base-url``, so a shared proxy started for
    Claude/Codex has to recognize it from wire signals or it forwards xAI
    session tokens to ``api.openai.com``.

    Only applies while the OpenAI target is still the default. An operator who
    pointed the proxy at a gateway (LiteLLM, Azure, self-hosted vLLM) chose it
    for every OpenAI-compatible client; a client User-Agent must not silently
    bypass that.

    This gate is URL policy only. It does not keep operator-configured
    ``OPENAI_TARGET_API_HEADERS`` away from xAI — those extras are configured
    independently of the target URL, so a default-URL proxy can still redirect
    here. The direct OpenAI HTTP handlers enforce credential isolation
    separately by suppressing configured extras when their OpenAI-compatible
    upstream candidate is the xAI host. Configured backend transports retain
    their existing header policy.
    """
    if not is_grok_cli_request(headers):
        return False
    return openai_target.rstrip("/") == DEFAULT_OPENAI_API_URL


def openai_compatible_base_url(proxy: Any, headers: Mapping[str, str]) -> str:
    """Resolve upstream for OpenAI-compatible metadata/passthrough traffic.

    Routes official Grok CLI to ``api.x.ai`` so ``GET /v1/models`` and catch-all
    passthrough succeed on a shared proxy whose OpenAI target is the default.
    """
    target = api_target(proxy, "openai")
    if route_grok_to_xai(headers, target):
        return XAI_API_URL
    return target


def select_passthrough_base_url(
    proxy: Any, headers: Mapping[str, str], path: str | None = None
) -> str:
    """Resolve the upstream base URL for catch-all proxy passthrough requests."""
    routing = resolve_codex_routing(headers)
    if routing.is_chatgpt_auth:
        return CHATGPT_BACKEND_API_URL
    if headers.get("x-goog-api-key"):
        return api_target(proxy, "gemini")
    if headers.get("api-key"):
        azure_base = headers.get("x-headroom-base-url", "")
        if azure_base:
            # Validate here, not only at the routes. `api-key` is attacker-
            # supplied too, so this branch is reachable by anyone who can send
            # a header, and it returns the destination the caller named. Routes
            # that forgot to guard turned the proxy into an SSRF relay into
            # loopback/RFC1918/cloud-metadata space (CVE-2026-77775).
            if is_safe_upstream_url(azure_base):
                return azure_base.rstrip("/")
            logger.warning("ignoring unsafe x-headroom-base-url override: %r", azure_base)
    provider_name = proxy.provider_runtime.model_metadata_provider(headers)
    target = api_target(proxy, provider_name)
    if (
        path is not None
        and provider_name == "openai"
        and is_copilot_completions_path(path)
        and not is_copilot_completions_host(target)
    ):
        # Copilot's inline completions arrive here because
        # `/v1/engines/<engine>/completions` matches no built-in route. Nothing
        # above this line looks at the path, so the request fell through to the
        # OpenAI target and Headroom forwarded editor keystrokes to
        # api.openai.com — a host that has not served the Engines API for years,
        # and one many corporate networks block outright (#3076).
        #
        # Only Copilot emits this path, so sending it to Copilot is unambiguous.
        # `copilot_completions_base_url()` does no I/O: it prefers an operator
        # override, then the completions host GitHub advertised in the last
        # token exchange, then the Copilot API URL — so the destination is
        # GitHub's own answer where we have it rather than a hardcoded guess,
        # and GHE deployments keep their host.
        #
        # The guard is on the *completions* host, not "any Copilot host". A CAPI
        # host is not a completions host: `headroom wrap vscode` points the
        # OpenAI target at the resolved subscription URL, which is the chat
        # surface (`GITHUB_COPILOT_API_URL`), and leaving that alone sent
        # `/v1/engines/.../completions` to a host that does not serve it. An
        # already-correct completions host — an operator override or a per-SKU
        # `endpoints.proxy` value — is still left untouched.
        #
        # Scoped to the OpenAI fall-through, which is the branch that is wrong
        # for this path. Every other branch above reflects a deliberate choice
        # of upstream by the caller's own auth headers, and the Copilot editor
        # extension sends none of them — so a request that took one of those
        # branches is not Copilot's and keeps the upstream it asked for.
        return copilot_completions_base_url()
    if provider_name == "openai":
        # Grok CLI reaches passthrough the same way, recognized by wire signals
        # rather than path, and only while the OpenAI target is still the
        # default. Checked after Copilot because that branch keys on a path only
        # Copilot emits, so it is the narrower claim on this fall-through.
        return openai_compatible_base_url(proxy, headers)
    return target
