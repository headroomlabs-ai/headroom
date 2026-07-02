"""Turnkey Claude Code + Vertex compression wiring.

Covers the fixes that let `headroom wrap claude` + Vertex actually deliver
compression:

- `litellm-vertex` -> `vertex_ai` provider alias (registry),
- the Vertex upstream host derived per-request from the path's `location`
  (so a europe-west1 request is not sent to a us-central1 host),
- the native `:rawPredict` route running the compression handler (not the
  verbatim passthrough) for the `anthropic` publisher.

No real GCP/Vertex is contacted — handlers and the backend class are stubbed.
"""

from __future__ import annotations

import logging
import types
from typing import Any

from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from headroom.providers import proxy_routes, registry
from headroom.providers.registry import DEFAULT_VERTEX_API_URL
from headroom.proxy.server import HeadroomProxy, ProxyConfig, create_app


# --------------------------------------------------------------------------
# Region-aware Vertex upstream host
# --------------------------------------------------------------------------
def _stub_proxy(vertex_url: str) -> Any:
    # `_api_target` reads `proxy.VERTEX_API_URL`, but its getattr default eagerly
    # evaluates `proxy.provider_runtime.api_target(...)`, so the stub needs both.
    return types.SimpleNamespace(
        VERTEX_API_URL=vertex_url,
        provider_runtime=types.SimpleNamespace(api_target=lambda name: vertex_url),
    )


def test_vertex_target_derives_region_from_location_on_default() -> None:
    proxy = _stub_proxy(DEFAULT_VERTEX_API_URL)
    assert (
        proxy_routes._vertex_target_for_location(proxy, "europe-west1")
        == "https://europe-west1-aiplatform.googleapis.com"
    )


def test_vertex_target_global_uses_unprefixed_host() -> None:
    proxy = _stub_proxy(DEFAULT_VERTEX_API_URL)
    assert (
        proxy_routes._vertex_target_for_location(proxy, "global")
        == "https://aiplatform.googleapis.com"
    )


def test_vertex_target_empty_location_uses_unprefixed_host() -> None:
    proxy = _stub_proxy(DEFAULT_VERTEX_API_URL)
    assert (
        proxy_routes._vertex_target_for_location(proxy, "") == "https://aiplatform.googleapis.com"
    )


def test_vertex_target_honors_explicit_override() -> None:
    # An operator who pinned a non-default upstream (private gateway) wins,
    # regardless of the path's location.
    proxy = _stub_proxy("https://vertex-gateway.internal")
    assert (
        proxy_routes._vertex_target_for_location(proxy, "europe-west1")
        == "https://vertex-gateway.internal"
    )


# --------------------------------------------------------------------------
# litellm-vertex -> vertex_ai provider alias
# --------------------------------------------------------------------------
def _capture_provider(backend: str) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    class FakeLiteLLM:
        def __init__(self, provider: str, region: str | None = None) -> None:
            captured["provider"] = provider
            captured["region"] = region

    registry.create_proxy_backend(
        backend=backend,
        anyllm_provider="openai",
        bedrock_region="us-east5",
        logger=logging.getLogger("test-vertex"),
        litellm_backend_cls=FakeLiteLLM,
    )
    return captured


def test_litellm_vertex_aliases_to_vertex_ai() -> None:
    # The documented `litellm-vertex` must reach the `vertex_ai` provider, not
    # a generic pass-through (which would drop the region and mangle the model).
    assert _capture_provider("litellm-vertex")["provider"] == "vertex_ai"


def test_litellm_vertex_ai_unchanged() -> None:
    assert _capture_provider("litellm-vertex_ai")["provider"] == "vertex_ai"


def test_litellm_bedrock_not_aliased() -> None:
    assert _capture_provider("litellm-bedrock")["provider"] == "bedrock"


# --------------------------------------------------------------------------
# Native :rawPredict route → compression handler with region-derived host
# --------------------------------------------------------------------------
def _default_vertex_app() -> Any:
    # No vertex_api_url override -> default us-central1 host -> route derives
    # the regional host from the request path.
    return create_app(ProxyConfig(optimize=True, cache_enabled=False, rate_limit_enabled=False))


def test_vertex_rawpredict_anthropic_runs_compression_handler(monkeypatch) -> None:
    captured: dict[str, str] = {}

    # The route calls handle_anthropic_messages(request, base_url, provider, model).
    async def fake(self, request, base_url, provider, model, *rest):  # type: ignore[no-untyped-def]
        captured.update(base_url=str(base_url), provider=str(provider), model=str(model))
        return JSONResponse({"ok": True})

    monkeypatch.setattr(HeadroomProxy, "handle_anthropic_messages", fake)

    with TestClient(_default_vertex_app()) as client:
        resp = client.post(
            "/v1/projects/p/locations/europe-west1/publishers/anthropic/models/"
            "claude-sonnet-4-6:rawPredict",
            json={"anthropic_version": "vertex-2023-10-16", "messages": []},
        )
    assert resp.status_code == 200
    # The anthropic publisher is routed to the compression handler (not the
    # verbatim passthrough), with the region-derived upstream host (exact match,
    # not a substring check).
    assert captured["provider"] == "vertex:anthropic"
    assert captured["base_url"] == "https://europe-west1-aiplatform.googleapis.com"
    assert captured["model"] == "claude-sonnet-4-6"


def test_vertex_rawpredict_versionless_anthropic_rewrites_to_v1(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    async def fake(
        self,
        request,
        base_url,
        provider,
        model,
        force_stream=False,
    ):  # type: ignore[no-untyped-def]
        captured.update(
            path=request.url.path,
            raw_path=request.scope.get("raw_path"),
            base_url=str(base_url),
            provider=str(provider),
            model=str(model),
            force_stream=force_stream,
        )
        return JSONResponse({"ok": True})

    monkeypatch.setattr(HeadroomProxy, "handle_anthropic_messages", fake)

    with TestClient(_default_vertex_app()) as client:
        resp = client.post(
            "/projects/p/locations/europe-west1/publishers/anthropic/models/"
            "claude-sonnet-4-6:rawPredict",
            json={"anthropic_version": "vertex-2023-10-16", "messages": []},
        )

    assert resp.status_code == 200
    assert captured == {
        "path": (
            "/projects/p/locations/europe-west1/publishers/anthropic/models/"
            "claude-sonnet-4-6:rawPredict"
        ),
        "raw_path": (
            b"/projects/p/locations/europe-west1/publishers/anthropic/models/"
            b"claude-sonnet-4-6:rawPredict"
        ),
        "base_url": "https://europe-west1-aiplatform.googleapis.com/v1",
        "provider": "vertex:anthropic",
        "model": "claude-sonnet-4-6",
        "force_stream": False,
    }


def test_vertex_stream_rawpredict_versionless_anthropic_forces_stream(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    async def fake(
        self,
        request,
        base_url,
        provider,
        model,
        force_stream=False,
    ):  # type: ignore[no-untyped-def]
        captured.update(
            path=request.url.path,
            base_url=str(base_url),
            provider=str(provider),
            model=str(model),
            force_stream=force_stream,
        )
        return JSONResponse({"ok": True})

    monkeypatch.setattr(HeadroomProxy, "handle_anthropic_messages", fake)

    with TestClient(_default_vertex_app()) as client:
        resp = client.post(
            "/projects/p/locations/europe-west1/publishers/anthropic/models/"
            "claude-sonnet-4-6:streamRawPredict",
            json={"anthropic_version": "vertex-2023-10-16", "messages": []},
        )

    assert resp.status_code == 200
    assert captured == {
        "path": (
            "/projects/p/locations/europe-west1/publishers/anthropic/models/"
            "claude-sonnet-4-6:streamRawPredict"
        ),
        "base_url": "https://europe-west1-aiplatform.googleapis.com/v1",
        "provider": "vertex:anthropic",
        "model": "claude-sonnet-4-6",
        "force_stream": True,
    }
