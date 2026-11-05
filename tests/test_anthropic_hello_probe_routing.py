"""Regression tests for Anthropic connectivity probe routing (#3336).

Claude Code (and other Anthropic SDK clients) sends unauthenticated
``HEAD /api/hello`` or ``GET /api/hello`` requests to test upstream
connectivity. Because the probe carries no authorization headers,
it must not fall through to the default OpenAI target.
"""

from __future__ import annotations

from typing import Any

from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from headroom.providers.proxy_targets import (
    is_anthropic_hello_path,
    select_passthrough_base_url,
)
from headroom.proxy.server import HeadroomProxy, ProxyConfig, create_app


def _mock_proxy(**legacy_targets: str) -> Any:
    class Runtime:
        @staticmethod
        def api_target(provider: str) -> str:
            return f"https://runtime.{provider}.test"

        @staticmethod
        def model_metadata_provider(headers: Any) -> str:
            return "anthropic" if headers.get("x-api-key") else "openai"

    return type("Proxy", (), {**legacy_targets, "provider_runtime": Runtime()})()


def test_is_anthropic_hello_path_matches_expected_patterns() -> None:
    assert is_anthropic_hello_path("/api/hello")
    assert is_anthropic_hello_path("api/hello")
    assert is_anthropic_hello_path("/api/hello/")
    assert is_anthropic_hello_path("api/hello/")

    assert not is_anthropic_hello_path(None)
    assert not is_anthropic_hello_path("")
    assert not is_anthropic_hello_path("/api/hello/subpath")
    assert not is_anthropic_hello_path("/v1/messages")
    assert not is_anthropic_hello_path("/api/other")


def test_select_passthrough_base_url_routes_hello_to_anthropic() -> None:
    proxy = _mock_proxy(
        ANTHROPIC_API_URL="https://anthropic.custom.test",
        OPENAI_API_URL="https://openai.custom.test",
    )

    # Keyless probe on /api/hello routes to Anthropic target.
    assert select_passthrough_base_url(proxy, {}, "/api/hello") == "https://anthropic.custom.test"
    assert select_passthrough_base_url(proxy, {}, "api/hello/") == "https://anthropic.custom.test"

    # Other keyless paths still fall through to OpenAI.
    assert select_passthrough_base_url(proxy, {}, "/other/endpoint") == "https://openai.custom.test"


def test_hello_probe_end_to_end_routing(monkeypatch: Any) -> None:
    calls: list[tuple[str, str, str, str]] = []

    async def fake_passthrough(
        self: Any, request: Any, base_url: str, sub_path: str = "", provider_name: str = ""
    ) -> JSONResponse:
        calls.append((request.method, request.url.path, base_url, provider_name))
        return JSONResponse(
            {
                "method": request.method,
                "path": request.url.path,
                "base_url": base_url,
                "sub_path": sub_path,
                "provider": provider_name,
            }
        )

    monkeypatch.setattr(HeadroomProxy, "handle_passthrough", fake_passthrough)

    app = create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            anthropic_api_url="https://api.anthropic.test",
            openai_api_url="https://api.openai.test",
        )
    )

    with TestClient(app) as client:
        # Keyless HEAD request as sent by Claude Code's Bun connectivity probe.
        head_resp = client.head("/api/hello", headers={"user-agent": "Bun/1.1.20"})
        assert head_resp.status_code == 200
        assert calls[-1] == ("HEAD", "/api/hello", "https://api.anthropic.test", "anthropic")

        # Keyless GET request.
        get_resp = client.get("/api/hello")
        assert get_resp.status_code == 200
        assert calls[-1] == ("GET", "/api/hello", "https://api.anthropic.test", "anthropic")
        assert get_resp.json()["base_url"] == "https://api.anthropic.test"
