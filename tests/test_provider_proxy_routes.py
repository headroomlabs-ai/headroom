from __future__ import annotations

import importlib
from typing import Any
from unittest.mock import patch

import httpx
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from headroom.proxy.server import HeadroomProxy, ProxyConfig, create_app


def _app() -> Any:
    return create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            anthropic_api_url="https://api.anthropic.test",
            openai_api_url="https://api.openai.test",
            gemini_api_url="https://api.gemini.test",
            cloudcode_api_url="https://cloudcode.test",
        )
    )


def test_provider_passthrough_routes_forward_expected_targets(monkeypatch) -> None:
    calls: list[tuple[str, str, str, str]] = []

    async def fake_passthrough(self, request, base_url, sub_path="", provider_name=""):  # type: ignore[no-untyped-def]
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

    with TestClient(_app()) as client:
        assert client.post("/v1/messages/count_tokens").json()["base_url"] == (
            "https://api.anthropic.test"
        )
        assert client.get("/v1/models", headers={"x-goog-api-key": "test"}).json()["base_url"] == (
            "https://api.openai.test"
        )
        assert client.get("/v1/models/demo").json()["sub_path"] == "models"
        assert (
            client.get(
                "/azure/models",
                headers={
                    "api-key": "azure-key",
                    "x-headroom-base-url": "https://azure.example/openai/",
                },
            ).json()["base_url"]
            == "https://azure.example/openai"
        )
        assert client.post("/v1/embeddings").json()["provider"] == "openai"
        assert client.post("/v1/moderations").json()["sub_path"] == "moderations"
        assert client.post("/v1/images/generations").json()["sub_path"] == "images/generations"
        assert client.post("/v1/audio/transcriptions").json()["sub_path"] == "audio/transcriptions"
        assert client.post("/v1/audio/speech").json()["sub_path"] == "audio/speech"
        assert client.get("/v1beta/models").json()["provider"] == "gemini"
        assert client.get("/v1beta/models/demo").json()["sub_path"] == "models"
        assert client.post("/v1beta/models/demo:embedContent").json()["sub_path"] == "embedContent"
        assert client.post("/v1beta/cachedContents").json()["sub_path"] == "cachedContents"
        assert client.get("/v1beta/cachedContents").json()["sub_path"] == "cachedContents"
        assert client.get("/v1beta/cachedContents/cache-1").json()["sub_path"] == "cachedContents"
        assert client.delete("/v1beta/cachedContents/cache-1").json()["sub_path"] == (
            "cachedContents"
        )
        assert (
            client.get(
                "/unhandled/path",
                headers={"x-headroom-base-url": "https://custom.example/base/"},
            ).json()["base_url"]
            == "https://custom.example/base"
        )
        assert client.get("/another/path", headers={"x-goog-api-key": "test"}).json()[
            "base_url"
        ] == ("https://api.gemini.test")

    assert len(calls) >= 16


def test_proxy_route_helpers_prefer_legacy_targets_and_gemini_passthrough() -> None:
    proxy_routes = importlib.import_module("headroom.providers.proxy_routes")
    proxy = type(
        "Proxy",
        (),
        {
            "ANTHROPIC_API_URL": "https://legacy.anthropic.test",
            "OPENAI_API_URL": "https://legacy.openai.test",
            "GEMINI_API_URL": "https://legacy.gemini.test",
            "provider_runtime": type(
                "Runtime",
                (),
                {
                    "api_target": staticmethod(lambda provider: f"https://runtime.{provider}.test"),
                    "model_metadata_provider": staticmethod(lambda headers: "anthropic"),
                },
            )(),
        },
    )()

    assert proxy_routes._api_target(proxy, "anthropic") == "https://legacy.anthropic.test"
    assert proxy_routes._select_passthrough_base_url(proxy, {"x-goog-api-key": "test"}) == (
        "https://legacy.gemini.test"
    )
    assert (
        proxy_routes._select_passthrough_base_url(
            proxy, {"api-key": "azure", "x-headroom-base-url": "https://azure.example/base/"}
        )
        == "https://azure.example/base"
    )
    assert proxy_routes._select_passthrough_base_url(proxy, {"api-key": "azure"}) == (
        "https://legacy.anthropic.test"
    )
    assert proxy_routes._select_passthrough_base_url(proxy, {}) == "https://legacy.anthropic.test"


def test_provider_specific_routes_delegate_to_expected_proxy_handlers(monkeypatch) -> None:
    delegated: list[tuple[str, str, tuple[str, ...]]] = []

    def install(name: str) -> None:
        async def fake(self, request, *args):  # type: ignore[no-untyped-def]
            delegated.append((name, request.url.path, tuple(str(arg) for arg in args)))
            return JSONResponse({"handler": name, "path": request.url.path, "args": list(args)})

        monkeypatch.setattr(HeadroomProxy, name, fake)

    for handler_name in (
        "handle_anthropic_messages",
        "handle_anthropic_batch_create",
        "handle_anthropic_batch_passthrough",
        "handle_anthropic_batch_results",
        "handle_openai_chat",
        "handle_openai_responses",
        "handle_batch_create",
        "handle_batch_list",
        "handle_batch_get",
        "handle_batch_cancel",
        "handle_gemini_generate_content",
        "handle_gemini_stream_generate_content",
        "handle_gemini_count_tokens",
        "handle_google_cloudcode_stream",
        "handle_google_batch_create",
        "handle_google_batch_results",
        "handle_google_batch_passthrough",
    ):
        install(handler_name)

    with TestClient(_app()) as client:
        assert client.post("/v1/messages").json()["handler"] == "handle_anthropic_messages"
        assert (
            client.post("/v1/messages/batches").json()["handler"] == "handle_anthropic_batch_create"
        )
        assert client.get("/v1/messages/batches").json()["handler"] == (
            "handle_anthropic_batch_passthrough"
        )
        assert client.get("/v1/messages/batches/b1").json()["args"] == ["b1"]
        assert client.get("/v1/messages/batches/b1/results").json()["handler"] == (
            "handle_anthropic_batch_results"
        )
        assert client.post("/v1/messages/batches/b1/cancel").json()["handler"] == (
            "handle_anthropic_batch_passthrough"
        )
        assert client.post("/v1/chat/completions").json()["handler"] == "handle_openai_chat"
        assert client.post("/v1/responses").json()["handler"] == "handle_openai_responses"
        assert client.post("/v1/codex/responses").json()["handler"] == "handle_openai_responses"
        assert client.post("/backend-api/responses").json()["handler"] == "handle_openai_responses"
        assert client.post("/backend-api/codex/responses").json()["handler"] == (
            "handle_openai_responses"
        )
        assert client.post("/v1/batches").json()["handler"] == "handle_batch_create"
        assert client.get("/v1/batches").json()["handler"] == "handle_batch_list"
        assert client.get("/v1/batches/b1").json()["handler"] == "handle_batch_get"
        assert client.post("/v1/batches/b1/cancel").json()["handler"] == "handle_batch_cancel"
        assert client.post("/v1beta/models/demo:generateContent").json()["handler"] == (
            "handle_gemini_generate_content"
        )
        assert client.post("/v1beta/models/demo:streamGenerateContent").json()["handler"] == (
            "handle_gemini_stream_generate_content"
        )
        assert client.post("/v1beta/models/demo:countTokens").json()["handler"] == (
            "handle_gemini_count_tokens"
        )
        assert client.post("/v1internal:streamGenerateContent").json()["handler"] == (
            "handle_google_cloudcode_stream"
        )
        assert client.post("/v1/v1internal:streamGenerateContent").json()["handler"] == (
            "handle_google_cloudcode_stream"
        )
        assert client.post("/v1beta/models/demo:batchGenerateContent").json()["handler"] == (
            "handle_google_batch_create"
        )
        assert client.get("/v1beta/batches/b1").json()["handler"] == "handle_google_batch_results"
        assert client.post("/v1beta/batches/b1:cancel").json()["handler"] == (
            "handle_google_batch_passthrough"
        )
        assert client.delete("/v1beta/batches/b1").json()["handler"] == (
            "handle_google_batch_passthrough"
        )

    assert len(delegated) >= 24


def test_openai_response_websocket_aliases_delegate_to_openai_ws_handler(monkeypatch) -> None:
    seen_paths: list[str] = []

    async def fake_ws(self, websocket):  # type: ignore[no-untyped-def]
        seen_paths.append(websocket.url.path)
        await websocket.accept()
        await websocket.send_json({"path": websocket.url.path})
        await websocket.close()

    monkeypatch.setattr(HeadroomProxy, "handle_openai_responses_ws", fake_ws)

    with TestClient(_app()) as client:
        for path in (
            "/v1/responses",
            "/v1/codex/responses",
            "/backend-api/responses",
            "/backend-api/codex/responses",
        ):
            with client.websocket_connect(path) as websocket:
                assert websocket.receive_json() == {"path": path}

    assert seen_paths == [
        "/v1/responses",
        "/v1/codex/responses",
        "/backend-api/responses",
        "/backend-api/codex/responses",
    ]


def test_openai_response_subpath_passthrough_returns_502_on_http_failure() -> None:
    class FailingAsyncClient:
        async def request(self, method, url, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError(f"boom: {method} {url}")

        async def aclose(self) -> None:
            return None

    with TestClient(_app()) as client:
        client.app.state.proxy.http_client = FailingAsyncClient()
        with patch("headroom.providers.proxy_routes.logger") as logger:
            response = client.post("/v1/responses/compact?trace=1", json={"model": "gpt-4o"})

    assert response.status_code == 502
    assert "boom: POST https://api.openai.test/v1/responses/compact?trace=1" in response.text
    logger.error.assert_called_once()


def test_openai_response_subpath_passthrough_uses_openai_target() -> None:
    class FakeAsyncClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, dict[str, str]]] = []

        async def request(self, method, url, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append((method, url, dict(kwargs.get("headers", {}))))
            return httpx.Response(200, json={"url": url})

        async def aclose(self) -> None:
            return None

    with TestClient(_app()) as client:
        fake = FakeAsyncClient()
        client.app.state.proxy.http_client = fake
        response = client.delete(
            "/v1/responses/items/resp_123?trace=7",
            headers={"Authorization": "Bearer sk-proj-test"},
        )

    assert response.status_code == 200
    assert len(fake.calls) == 1
    method, url, headers = fake.calls[0]
    assert method == "DELETE"
    assert url == "https://api.openai.test/v1/responses/items/resp_123?trace=7"
    assert headers["authorization"] == "Bearer sk-proj-test"


def test_openai_response_subpath_aliases_and_chatgpt_auth_use_expected_targets(monkeypatch) -> None:
    monkeypatch.setattr(
        "headroom.providers.proxy_routes._resolve_codex_routing_headers",
        lambda headers: (headers, True),
    )

    class FakeAsyncClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        async def request(self, method, url, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append((method, url))
            return httpx.Response(200, json={"url": url})

        async def aclose(self) -> None:
            return None

    with TestClient(_app()) as client:
        fake = FakeAsyncClient()
        client.app.state.proxy.http_client = fake
        assert client.get("/v1/codex/responses/items/resp_1").status_code == 200
        assert client.post("/backend-api/responses/items/resp_2").status_code == 200
        assert client.delete("/backend-api/codex/responses/items/resp_3").status_code == 200

    assert fake.calls == [
        ("GET", "https://chatgpt.com/backend-api/codex/responses/items/resp_1"),
        ("POST", "https://chatgpt.com/backend-api/codex/responses/items/resp_2"),
        ("DELETE", "https://chatgpt.com/backend-api/codex/responses/items/resp_3"),
    ]


def test_gemini_batch_embed_contents_passthrough_uses_gemini_target(monkeypatch) -> None:
    calls: list[tuple[str, str, str]] = []

    async def fake_passthrough(self, request, base_url, sub_path="", provider_name=""):  # type: ignore[no-untyped-def]
        calls.append((request.url.path, base_url, sub_path))
        return JSONResponse({"base_url": base_url, "sub_path": sub_path, "provider": provider_name})

    monkeypatch.setattr(HeadroomProxy, "handle_passthrough", fake_passthrough)

    with TestClient(_app()) as client:
        response = client.post("/v1beta/models/demo:batchEmbedContents")

    assert response.status_code == 200
    assert response.json() == {
        "base_url": "https://api.gemini.test",
        "sub_path": "batchEmbedContents",
        "provider": "gemini",
    }
    assert calls == [
        ("/v1beta/models/demo:batchEmbedContents", "https://api.gemini.test", "batchEmbedContents")
    ]


def test_v1_models_fetches_codex_registry_under_chatgpt_auth(monkeypatch) -> None:
    proxy_routes = importlib.import_module("headroom.providers.proxy_routes")
    debug_messages: list[tuple[str, tuple[object, ...]]] = []
    monkeypatch.setattr(
        proxy_routes.logger,
        "debug",
        lambda message, *args: debug_messages.append((message, args)),
    )

    class FakeAsyncClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, dict[str, str]]] = []

        async def get(self, url, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(("GET", url, dict(kwargs.get("headers", {}))))
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"slug": "gpt-5.5"},
                        {"slug": "gpt-5.3-codex-spark"},
                    ]
                },
            )

        async def aclose(self) -> None:
            return None

    with TestClient(_app()) as client:
        fake_http_client = FakeAsyncClient()
        client.app.state.proxy.http_client = fake_http_client
        response = client.get(
            "/v1/models?client_version=0.135.0",
            headers={
                "authorization": "Bearer eyJ-chatgpt-oauth-token",
                "chatgpt-account-id": "test-account",
                "originator": "Codex Desktop",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "object": "list",
        "data": [
            {
                "id": "gpt-5.5",
                "object": "model",
                "created": 0,
                "owned_by": "openai",
            },
            {
                "id": "gpt-5.3-codex-spark",
                "object": "model",
                "created": 0,
                "owned_by": "openai",
            },
        ],
    }
    assert len(fake_http_client.calls) == 1
    method, url, headers = fake_http_client.calls[0]
    assert method == "GET"
    assert url == "https://chatgpt.com/backend-api/codex/models?client_version=0.135.0"
    assert headers["authorization"] == "Bearer eyJ-chatgpt-oauth-token"
    assert headers["chatgpt-account-id"] == "test-account"
    assert headers["originator"] == "Codex Desktop"
    assert headers["accept"] == "application/json"
    assert "Accept" not in headers
    assert debug_messages == [
        (
            "Fetched Codex model IDs from upstream model registry: %s",
            (["gpt-5.5", "gpt-5.3-codex-spark"],),
        ),
    ]


def test_v1_models_falls_back_to_synthetic_list_under_chatgpt_auth(monkeypatch) -> None:
    """Issue #478: under Codex ChatGPT-subscription OAuth, the proxy
    must NOT forward `/v1/models` to chatgpt.com/backend-api/models
    (which returns 403). If the Codex-specific registry also fails,
    synthesize an OpenAI-compatible response with the known-supported
    Codex/ChatGPT model set instead, so Codex's model-picker refresh succeeds.
    """

    class FakeAsyncClient:
        async def get(self, url, **kwargs):  # type: ignore[no-untyped-def]
            return httpx.Response(403, json={"error": "forbidden"})

        async def aclose(self) -> None:
            return None

    with TestClient(_app()) as client:
        client.app.state.proxy.http_client = FakeAsyncClient()
        # ChatGPT auth detected via Bearer + ChatGPT account header
        # (mirrors what Codex Desktop sends).
        response = client.get(
            "/v1/models",
            headers={
                "authorization": "Bearer eyJ-chatgpt-oauth-token",
                "chatgpt-account-id": "test-account",
                "originator": "Codex Desktop",
            },
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert isinstance(payload["data"], list)
    assert len(payload["data"]) > 0
    model_ids = {entry["id"] for entry in payload["data"]}
    # Spot-check: the model from issue #478's repro log must be present.
    assert "gpt-5.5" in model_ids
    for entry in payload["data"]:
        assert entry["object"] == "model"
        assert entry["owned_by"] == "openai"


def test_v1_models_get_single_dynamic_under_chatgpt_auth() -> None:
    """The single-model variant (`/v1/models/{id}`) is also called by
    Codex for some flows. It should use the Codex registry first so
    dynamically exposed model slugs validate consistently."""

    class FakeAsyncClient:
        def __init__(self) -> None:
            self.calls = 0

        async def get(self, url, **kwargs):  # type: ignore[no-untyped-def]
            self.calls += 1
            return httpx.Response(
                200,
                json={"models": [{"slug": "gpt-5.5"}, {"slug": "gpt-5.3-codex-spark"}]},
            )

        async def aclose(self) -> None:
            return None

    with TestClient(_app()) as client:
        fake_http_client = FakeAsyncClient()
        client.app.state.proxy.http_client = fake_http_client
        ok = client.get(
            "/v1/models/gpt-5.3-codex-spark",
            headers={
                "authorization": "Bearer eyJ-chatgpt-oauth-token",
                "chatgpt-account-id": "test-account",
            },
        )
        unknown = client.get(
            "/v1/models/gpt-99-future",
            headers={
                "authorization": "Bearer eyJ-chatgpt-oauth-token",
                "chatgpt-account-id": "test-account",
            },
        )
    assert ok.status_code == 200
    assert ok.json() == {
        "id": "gpt-5.3-codex-spark",
        "object": "model",
        "created": 0,
        "owned_by": "openai",
    }
    assert unknown.status_code == 404
    assert fake_http_client.calls == 2


def test_v1_models_still_forwards_under_non_chatgpt_auth() -> None:
    """Non-ChatGPT auth (regular API key, Gemini, etc.) must still
    forward to the upstream provider — only the ChatGPT-OAuth path
    short-circuits to the synthetic response."""
    calls: list[tuple[str, str, str]] = []

    async def fake_passthrough(self, request, base_url, sub_path="", provider_name=""):  # type: ignore[no-untyped-def]
        calls.append((request.url.path, base_url, provider_name))
        return JSONResponse({"base_url": base_url, "provider": provider_name})

    with patch.object(HeadroomProxy, "handle_passthrough", fake_passthrough):
        with TestClient(_app()) as client:
            response = client.get(
                "/v1/models",
                headers={"authorization": "Bearer sk-real-api-key"},
            )
    assert response.status_code == 200
    # Forwarded — not synthesized — because no chatgpt-account-id header.
    assert calls, "Non-ChatGPT-auth /v1/models must forward, not synthesize"


def test_v1_models_routes_claude_code_gateway_discovery_to_anthropic() -> None:
    """Claude Code gateway/OAuth model discovery can use a Bearer token that
    does not look like an Anthropic API key. Route those `/v1/models` requests
    to Anthropic so Claude's gateway model cache is not populated from OpenAI.
    """
    calls: list[tuple[str, str, str]] = []

    async def fake_passthrough(self, request, base_url, sub_path="", provider_name=""):  # type: ignore[no-untyped-def]
        calls.append((request.url.path, base_url, provider_name))
        return JSONResponse({"base_url": base_url, "provider": provider_name})

    with patch.object(HeadroomProxy, "handle_passthrough", fake_passthrough):
        with TestClient(_app()) as client:
            list_response = client.get(
                "/v1/models",
                headers={
                    "authorization": "Bearer claude-gateway-oauth-token",
                    "user-agent": "claude-code/1.5.0 (darwin; arm64)",
                },
            )
            get_response = client.get(
                "/v1/models/claude-opus-4-8",
                headers={
                    "authorization": "Bearer claude-gateway-oauth-token",
                    "user-agent": "claude-code/1.5.0 (darwin; arm64)",
                },
            )

    assert list_response.status_code == 200
    assert get_response.status_code == 200
    assert list_response.json() == {
        "base_url": "https://api.anthropic.test",
        "provider": "anthropic",
    }
    assert get_response.json() == {
        "base_url": "https://api.anthropic.test",
        "provider": "anthropic",
    }
    assert calls == [
        ("/v1/models", "https://api.anthropic.test", "anthropic"),
        ("/v1/models/claude-opus-4-8", "https://api.anthropic.test", "anthropic"),
    ]
