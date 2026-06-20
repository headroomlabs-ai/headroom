"""Tests for the ``X-Headroom-Upstream`` per-request upstream override.

Enables a single Headroom proxy instance to fan out to many upstreams: the
caller tags each request with the real upstream base and the proxy forwards
there instead of its startup default (e.g. ``headroom wrap opencode`` routing
OpenCode's 75+ providers through one proxy).

Coverage:

1. ``request_upstream_override`` normalization (pure function).
2. ``/v1/messages`` threads the override through ``upstream_base_url``.
3. ``/v1/chat/completions`` builds the forward URL from the override.
4. The verbatim catch-all / passthrough routes build the URL from the
   override, and the ``x-headroom-upstream`` header is stripped before the
   upstream call (no fingerprint leakage).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from starlette.datastructures import Headers

from headroom.proxy.helpers import request_upstream_override
from headroom.proxy.server import HeadroomProxy, ProxyConfig, create_app


def _stub_request(headers: dict[str, str]) -> Any:
    """A minimal stand-in exposing a case-insensitive ``headers`` mapping."""

    class _R:
        def __init__(self, hdrs: dict[str, str]) -> None:
            self.headers = Headers(hdrs)

    return _R(headers)


def _app(**overrides: Any) -> Any:
    cfg: dict[str, Any] = {
        "optimize": False,
        "cache_enabled": False,
        "rate_limit_enabled": False,
        "anthropic_api_url": "https://api.anthropic.test",
        "openai_api_url": "https://api.openai.test",
        "gemini_api_url": "https://api.gemini.test",
        "cloudcode_api_url": "https://cloudcode.test",
        "vertex_api_url": "https://vertex.test",
    }
    cfg.update(overrides)
    return create_app(ProxyConfig(**cfg))


# ── resolver normalization ───────────────────────────────────────────


def test_override_absent_returns_none() -> None:
    assert request_upstream_override(_stub_request({})) is None


def test_override_empty_returns_none() -> None:
    assert request_upstream_override(_stub_request({"x-headroom-upstream": ""})) is None
    assert request_upstream_override(_stub_request({"x-headroom-upstream": "   "})) is None


def test_override_bare_host_is_kept() -> None:
    assert (
        request_upstream_override(_stub_request({"x-headroom-upstream": "https://api.deepseek.com"}))
        == "https://api.deepseek.com"
    )


def test_override_strips_trailing_slash() -> None:
    assert (
        request_upstream_override(_stub_request({"x-headroom-upstream": "https://api.deepseek.com/"}))
        == "https://api.deepseek.com"
    )


def test_override_strips_trailing_v1() -> None:
    assert (
        request_upstream_override(_stub_request({"x-headroom-upstream": "https://api.deepseek.com/v1"}))
        == "https://api.deepseek.com"
    )


def test_override_strips_trailing_v1_with_slash() -> None:
    assert (
        request_upstream_override(_stub_request({"x-headroom-upstream": "https://api.deepseek.com/v1/"}))
        == "https://api.deepseek.com"
    )


def test_override_preserves_path_prefix_before_v1() -> None:
    # OpenRouter / Groq style: the /v1 is the API version, the prefix is real.
    assert (
        request_upstream_override(_stub_request({"x-headroom-upstream": "https://openrouter.ai/api/v1"}))
        == "https://openrouter.ai/api"
    )
    assert (
        request_upstream_override(_stub_request({"x-headroom-upstream": "https://api.groq.com/openai/v1"}))
        == "https://api.groq.com/openai"
    )


def test_override_header_lookup_is_case_insensitive() -> None:
    assert (
        request_upstream_override(_stub_request({"X-Headroom-Upstream": "https://api.deepseek.com/v1"}))
        == "https://api.deepseek.com"
    )


# ── /v1/messages threads the override through upstream_base_url ──────


def test_v1_messages_passes_override_as_upstream_base_url(monkeypatch) -> None:
    captured: list[Any] = []

    async def fake_handle_anthropic_messages(
        self,
        request,
        upstream_base_url=None,
        provider_name="anthropic",
        model_override=None,
        force_stream=False,
    ):  # type: ignore[no-untyped-def]
        captured.append(upstream_base_url)
        return JSONResponse({"upstream_base_url": upstream_base_url})

    monkeypatch.setattr(HeadroomProxy, "handle_anthropic_messages", fake_handle_anthropic_messages)

    with TestClient(_app()) as client:
        # Without the header: no override is threaded (default behaviour).
        assert client.post("/v1/messages", json={"model": "claude", "messages": []}).json()[
            "upstream_base_url"
        ] is None
        # With the header: the normalized override is threaded through.
        assert client.post(
            "/v1/messages",
            json={"model": "claude", "messages": []},
            headers={"X-Headroom-Upstream": "https://api.anthropic.com/v1"},
        ).json()["upstream_base_url"] == "https://api.anthropic.com"

    assert captured == [None, "https://api.anthropic.com"]


# ── /v1/chat/completions builds the forward URL from the override ────


def _ok_response() -> httpx.Response:
    return httpx.Response(
        200,
        content=b'{"id":"chatcmpl-test","object":"chat.completion","choices":[]}',
        headers={"content-type": "application/json"},
    )


def test_v1_chat_completions_forwards_to_override_upstream(monkeypatch) -> None:
    forwarded: list[str] = []

    async def fake_retry(self, method, url, headers, body, stream=False, **kwargs):  # type: ignore[no-untyped-def]
        forwarded.append(url)
        return _ok_response()

    monkeypatch.setattr(HeadroomProxy, "_retry_request", fake_retry)

    with TestClient(_app(optimize=False)) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]},
            headers={
                "X-Headroom-Upstream": "https://api.deepseek.com/v1",
                "Authorization": "Bearer sk-test",
            },
        )

    assert resp.status_code == 200
    # The /v1 API-version segment comes from the hardcoded chat path; the
    # override supplies the base (trailing /v1 stripped by the resolver).
    assert forwarded == ["https://api.deepseek.com/v1/chat/completions"]


def test_v1_chat_completions_defaults_to_startup_target_without_override(monkeypatch) -> None:
    forwarded: list[str] = []

    async def fake_retry(self, method, url, headers, body, stream=False, **kwargs):  # type: ignore[no-untyped-def]
        forwarded.append(url)
        return _ok_response()

    monkeypatch.setattr(HeadroomProxy, "_retry_request", fake_retry)

    with TestClient(_app(optimize=False)) as client:
        client.post(
            "/v1/chat/completions",
            json={"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer sk-test"},
        )

    assert forwarded == ["https://api.openai.test/v1/chat/completions"]


# ── catch-all / passthrough: URL built from override + header stripped ─


def _install_fake_http_client(proxy, response: httpx.Response) -> MagicMock:
    """Replace ``proxy.http_client`` so forwarding never touches the network."""
    client = MagicMock()
    client.request = AsyncMock(return_value=response)
    client.aclose = AsyncMock()
    proxy.http_client = client
    proxy.http_client_h1 = None
    return client


def test_passthrough_route_forwards_to_override_upstream() -> None:
    """A registered passthrough route (POST /v1/embeddings) honors the header."""
    response = httpx.Response(
        200, content=b'{"data":[]}', headers={"content-type": "application/json"}
    )
    with TestClient(_app()) as client:
        proxy = client.app.state.proxy
        http = _install_fake_http_client(proxy, response)
        client.post(
            "/v1/embeddings",
            json={"model": "text-embedding-test", "input": "hi"},
            headers={
                "X-Headroom-Upstream": "https://api.openai.com/v1",
                "Authorization": "Bearer sk-test",
            },
        )

    url = http.request.call_args.kwargs["url"]
    assert url == "https://api.openai.com/v1/embeddings"


def test_catchall_forwards_to_override_upstream() -> None:
    """The verbatim catch-all (/{path:path}) honors the header."""
    response = httpx.Response(
        200, content=b'{"ok":true}', headers={"content-type": "application/json"}
    )
    with TestClient(_app()) as client:
        proxy = client.app.state.proxy
        http = _install_fake_http_client(proxy, response)
        client.get(
            "/some/custom/path",
            headers={
                "X-Headroom-Upstream": "https://api.deepseek.com",
                "Authorization": "Bearer sk-test",
            },
        )

    url = http.request.call_args.kwargs["url"]
    assert url == "https://api.deepseek.com/some/custom/path"


def test_override_header_stripped_before_upstream_call() -> None:
    """The x-headroom-upstream control flag must not leak to the upstream."""
    response = httpx.Response(
        200, content=b'{"data":[]}', headers={"content-type": "application/json"}
    )
    with TestClient(_app()) as client:
        proxy = client.app.state.proxy
        http = _install_fake_http_client(proxy, response)
        client.post(
            "/v1/embeddings",
            json={"model": "text-embedding-test", "input": "hi"},
            headers={
                "X-Headroom-Upstream": "https://api.deepseek.com/v1",
                "Authorization": "Bearer sk-test",
            },
        )

    forwarded_headers = http.request.call_args.kwargs["headers"]
    forwarded_lower = {k.lower() for k in forwarded_headers}
    assert "x-headroom-upstream" not in forwarded_lower
