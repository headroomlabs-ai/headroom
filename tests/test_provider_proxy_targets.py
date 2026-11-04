from __future__ import annotations

from unittest.mock import patch

from headroom.providers.proxy_targets import (
    api_target,
    is_anthropic_hello_path,
    openai_compatible_base_url,
    select_passthrough_base_url,
    vertex_target_for_location,
)
from headroom.providers.registry import DEFAULT_VERTEX_API_URL
from headroom.proxy import upstream_guard


def _proxy(**legacy_targets: str):
    class Runtime:
        @staticmethod
        def api_target(provider: str) -> str:
            return f"https://runtime.{provider}.test"

        @staticmethod
        def model_metadata_provider(headers) -> str:  # type: ignore[no-untyped-def]
            return "anthropic" if headers.get("x-api-key") else "openai"

    return type("Proxy", (), {**legacy_targets, "provider_runtime": Runtime()})()


def test_api_target_prefers_legacy_proxy_attrs() -> None:
    proxy = _proxy(ANTHROPIC_API_URL="https://legacy.anthropic.test")

    assert api_target(proxy, "anthropic") == "https://legacy.anthropic.test"
    assert api_target(proxy, "openai") == "https://runtime.openai.test"


def test_vertex_target_for_location_derives_region_when_default_configured() -> None:
    proxy = _proxy(VERTEX_API_URL=DEFAULT_VERTEX_API_URL)

    assert vertex_target_for_location(proxy, "europe-west1") == (
        "https://europe-west1-aiplatform.googleapis.com"
    )
    assert vertex_target_for_location(proxy, "global") == "https://aiplatform.googleapis.com"


def test_vertex_target_for_location_honors_explicit_gateway() -> None:
    proxy = _proxy(VERTEX_API_URL="https://vertex-gateway.example")

    assert vertex_target_for_location(proxy, "europe-west1") == "https://vertex-gateway.example"


def test_select_passthrough_base_url_handles_special_auth_modes() -> None:
    proxy = _proxy(
        ANTHROPIC_API_URL="https://legacy.anthropic.test",
        OPENAI_API_URL="https://legacy.openai.test",
        GEMINI_API_URL="https://legacy.gemini.test",
    )

    assert select_passthrough_base_url(proxy, {"chatgpt-account-id": "acct"}) == (
        "https://chatgpt.com"
    )
    assert select_passthrough_base_url(proxy, {"x-goog-api-key": "test"}) == (
        "https://legacy.gemini.test"
    )
    # The Azure branch honours the override only after the SSRF guard clears
    # the destination (CVE-2026-77775), and `azure.example` does not resolve.
    # Pin a public answer so this stays a test of target *precedence*.
    with patch.object(
        upstream_guard.socket,
        "getaddrinfo",
        return_value=[(None, None, None, None, ("20.10.10.10", 443))],
    ):
        assert (
            select_passthrough_base_url(
                proxy,
                {"api-key": "azure", "x-headroom-base-url": "https://azure.example/base/"},
            )
            == "https://azure.example/base"
        )
    assert select_passthrough_base_url(proxy, {"x-api-key": "anthropic"}) == (
        "https://legacy.anthropic.test"
    )
    assert select_passthrough_base_url(proxy, {}) == "https://legacy.openai.test"
    assert select_passthrough_base_url(proxy, {}, "/api/hello") == "https://legacy.anthropic.test"
    assert select_passthrough_base_url(proxy, {}, "/api/hello/") == "https://legacy.anthropic.test"
    # Non-probe unauthenticated paths still fall through to OpenAI.
    assert select_passthrough_base_url(proxy, {}, "/other/endpoint") == "https://legacy.openai.test"
    # A configured OpenAI target outranks Grok wire signals (see
    # ``test_openai_compatible_base_url_respects_configured_openai_target``).
    assert (
        select_passthrough_base_url(
            proxy,
            {
                "x-xai-token-auth": "xai-grok-cli",
                "user-agent": "grok-shell/0.2.117",
            },
        )
        == "https://legacy.openai.test"
    )


def test_openai_compatible_base_url_routes_grok_to_xai_on_default_target() -> None:
    """The shared-proxy case this routing exists for: OpenAI target untouched."""
    proxy = _proxy(OPENAI_API_URL="https://api.openai.com")
    grok_headers = {"user-agent": "grok-pager/0.2.117 grok-shell/0.2.117"}

    assert openai_compatible_base_url(proxy, grok_headers) == "https://api.x.ai"
    assert openai_compatible_base_url(proxy, {}) == "https://api.openai.com"
    # Also via passthrough, which is the catch-all entry point Grok CLI hits.
    assert select_passthrough_base_url(proxy, grok_headers) == "https://api.x.ai"


def test_openai_compatible_base_url_routes_grok_build_to_xai_on_default_target() -> None:
    """Grok Build + Antigravity manifest case: the OpenAI target stays default.

    Antigravity needs api.openai.com, so the manifest cannot point the whole
    proxy at xAI; Grok Build's ``grok/`` user agent must still route there per
    request — for direct inference and for model-list/passthrough alike.
    """
    proxy = _proxy(OPENAI_API_URL="https://api.openai.com")
    grok_build_headers = {"user-agent": "grok/1.2.3"}

    assert openai_compatible_base_url(proxy, grok_build_headers) == "https://api.x.ai"
    assert select_passthrough_base_url(proxy, grok_build_headers) == "https://api.x.ai"
    # Antigravity traffic keeps the default OpenAI target.
    assert (
        openai_compatible_base_url(proxy, {"user-agent": "antigravity/1.0.0"})
        == "https://api.openai.com"
    )


def test_openai_compatible_base_url_respects_configured_openai_target() -> None:
    """An operator gateway is chosen for every OpenAI-compatible client.

    A client User-Agent must not silently bypass it, nor carry the gateway's
    ``OPENAI_TARGET_API_HEADERS`` to a different vendor.
    """
    proxy = _proxy(OPENAI_API_URL="https://legacy.openai.test")

    assert (
        openai_compatible_base_url(
            proxy,
            {"user-agent": "grok-pager/0.2.117 grok-shell/0.2.117"},
        )
        == "https://legacy.openai.test"
    )
    assert openai_compatible_base_url(proxy, {}) == "https://legacy.openai.test"


def test_select_passthrough_base_url_factory_mode_wins_over_auth_modes() -> None:
    proxy = _proxy(OPENAI_API_URL="https://legacy.openai.test")
    proxy.config = type("Config", (), {"factory_api_url": "https://api.factory.ai/"})()

    # In Factory Droid mode every non-LLM REST call is pinned to Factory,
    # ahead of the header-derived auth modes, with the trailing slash trimmed.
    assert select_passthrough_base_url(proxy, {}) == "https://api.factory.ai"
    assert (
        select_passthrough_base_url(proxy, {"x-goog-api-key": "test"}) == "https://api.factory.ai"
    )
    assert (
        select_passthrough_base_url(proxy, {"chatgpt-account-id": "a"}) == "https://api.factory.ai"
    )


def test_is_anthropic_hello_path() -> None:
    assert is_anthropic_hello_path("/api/hello")
    assert is_anthropic_hello_path("api/hello")
    assert is_anthropic_hello_path("/api/hello/")
    assert is_anthropic_hello_path("api/hello/")
    assert not is_anthropic_hello_path(None)
    assert not is_anthropic_hello_path("")
    assert not is_anthropic_hello_path("/api/hello/extra")
    assert not is_anthropic_hello_path("/v1/messages")
    assert not is_anthropic_hello_path("/hello")
