from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.testclient import TestClient

from headroom.providers.model_metadata import (
    MODEL_METADATA_LIST_ENDPOINT,
    ModelMetadataEndpoint,
    handle_model_metadata_endpoint,
    model_metadata_get_endpoint,
)


def test_model_metadata_endpoints_are_explicit() -> None:
    assert MODEL_METADATA_LIST_ENDPOINT == ModelMetadataEndpoint(
        "/v1/models",
        "/backend-api/models",
    )
    assert model_metadata_get_endpoint("gpt-5") == ModelMetadataEndpoint(
        "/v1/models/{model_id}",
        "/backend-api/models/gpt-5",
    )


def test_handle_model_metadata_endpoint_returns_chatgpt_response_when_present(monkeypatch) -> None:
    async def fake_chatgpt_metadata(
        http_client,
        request: Request,
        upstream_path: str,
    ) -> Response:
        return JSONResponse({"client": http_client, "upstream_path": upstream_path})

    monkeypatch.setattr(
        "headroom.providers.model_metadata.handle_chatgpt_model_metadata",
        fake_chatgpt_metadata,
    )
    proxy = type("Proxy", (), {"http_client": "h2"})()
    app = FastAPI()

    @app.get("/probe")
    async def probe(request: Request):
        return await handle_model_metadata_endpoint(
            proxy,
            request,
            endpoint=MODEL_METADATA_LIST_ENDPOINT,
            provider_api_base_url="https://api.openai.test",
            provider_name="openai",
        )

    with TestClient(app) as client:
        response = client.get("/probe")

    assert response.json() == {"client": "h2", "upstream_path": "/backend-api/models"}


def test_handle_model_metadata_endpoint_falls_back_to_selected_provider(monkeypatch) -> None:
    async def fake_chatgpt_metadata(http_client, request: Request, upstream_path: str) -> None:
        return None

    calls: list[tuple[str, str, str]] = []

    class Proxy:
        http_client = "h2"

        async def handle_passthrough(
            self,
            request: Request,
            base_url: str,
            sub_path: str = "",
            provider_name: str = "",
        ) -> Response:
            calls.append((base_url, sub_path, provider_name))
            return JSONResponse({"provider": provider_name, "sub_path": sub_path})

    monkeypatch.setattr(
        "headroom.providers.model_metadata.handle_chatgpt_model_metadata",
        fake_chatgpt_metadata,
    )
    app = FastAPI()

    @app.get("/probe")
    async def probe(request: Request):
        return await handle_model_metadata_endpoint(
            Proxy(),
            request,
            endpoint=model_metadata_get_endpoint("claude-opus"),
            provider_api_base_url="https://api.anthropic.test",
            provider_name="anthropic",
        )

    with TestClient(app) as client:
        response = client.get("/probe")

    assert response.json() == {"provider": "anthropic", "sub_path": "models"}
    assert calls == [("https://api.anthropic.test", "models", "anthropic")]


def test_grok_dispatch_adapts_xai_model_list(monkeypatch) -> None:
    async def no_chatgpt_metadata(http_client, request: Request, upstream_path: str) -> None:
        return None

    monkeypatch.setattr(
        "headroom.providers.model_metadata.handle_chatgpt_model_metadata",
        no_chatgpt_metadata,
    )

    class Proxy:
        http_client = "h2"

        async def handle_passthrough(self, request, base_url, sub_path="", provider_name=""):
            return Response(
                content=b'{"object":"list","data":[{"id":"grok-4.6","context_length":500000}]}',
                status_code=200,
                headers={"content-type": "application/json"},
            )

    app = FastAPI()

    @app.get("/probe")
    async def probe(request: Request):
        return await handle_model_metadata_endpoint(
            Proxy(),
            request,
            endpoint=MODEL_METADATA_LIST_ENDPOINT,
            provider_api_base_url="https://api.x.ai/v1",
            provider_name="openai",
        )

    with TestClient(app) as client:
        response = client.get("/probe")

    assert response.json()["data"] == [
        {"id": "grok-4.6", "context_length": 500000, "context_window": 500000}
    ]


def test_grok_response_preserves_status_and_safe_headers_without_stale_framing(monkeypatch) -> None:
    async def no_chatgpt_metadata(http_client, request: Request, upstream_path: str) -> None:
        return None

    monkeypatch.setattr(
        "headroom.providers.model_metadata.handle_chatgpt_model_metadata",
        no_chatgpt_metadata,
    )

    class Proxy:
        http_client = "h2"

        async def handle_passthrough(self, request, base_url, sub_path="", provider_name=""):
            return Response(
                content=b'{"data":[{"id":"grok","context_length":1}]}',
                status_code=206,
                headers={
                    "content-type": "application/json",
                    "x-upstream": "kept",
                    "etag": '"stale"',
                    "last-modified": "Thu, 01 Jan 1970 00:00:00 GMT",
                    "cache-control": "max-age=60",
                    "content-length": "44",
                    "content-encoding": "gzip",
                    "transfer-encoding": "chunked",
                },
            )

    app = FastAPI()

    @app.get("/probe")
    async def probe(request: Request):
        return await handle_model_metadata_endpoint(
            Proxy(),
            request,
            endpoint=MODEL_METADATA_LIST_ENDPOINT,
            provider_api_base_url="https://api.x.ai",
            provider_name="openai",
        )

    with TestClient(app) as client:
        response = client.get("/probe")

    assert response.status_code == 206
    assert response.headers["x-upstream"] == "kept"
    assert response.headers.get("etag") is None
    assert response.headers.get("last-modified") is None
    assert response.headers.get("cache-control") is None
    assert response.headers.get("content-encoding") is None
    assert response.headers.get("transfer-encoding") is None
    assert response.json()["data"][0]["context_window"] == 1


def test_grok_non_success_and_non_json_responses_are_unchanged(monkeypatch) -> None:
    async def no_chatgpt_metadata(http_client, request: Request, upstream_path: str) -> None:
        return None

    monkeypatch.setattr(
        "headroom.providers.model_metadata.handle_chatgpt_model_metadata",
        no_chatgpt_metadata,
    )
    responses = {
        "/error": Response(
            content=b'{"data":[{"context_length":500000}]}',
            status_code=500,
            headers={"x-upstream": "error"},
        ),
        "/non-json": Response(
            content=b"upstream text",
            status_code=200,
            headers={"x-upstream": "text"},
        ),
        "/surrogate": Response(
            content=b'{"data":[{"id":"grok","context_length":1,"label":"\\ud800"}]}',
            status_code=200,
            headers={"x-upstream": "surrogate"},
        ),
        "/non-finite": Response(
            content=b'{"data":[{"id":"grok","context_length":1}],"extra":NaN}',
            status_code=200,
            headers={"x-upstream": "non-finite"},
        ),
    }

    class Proxy:
        http_client = "h2"

        async def handle_passthrough(self, request, base_url, sub_path="", provider_name=""):
            return responses[request.url.path]

    app = FastAPI()

    @app.get("/{kind}")
    async def probe(request: Request, kind: str):
        return await handle_model_metadata_endpoint(
            Proxy(),
            request,
            endpoint=MODEL_METADATA_LIST_ENDPOINT,
            provider_api_base_url="https://api.x.ai",
            provider_name="openai",
        )

    with TestClient(app) as client:
        error = client.get("/error")
        non_json = client.get("/non-json")
        surrogate = client.get("/surrogate")
        non_finite = client.get("/non-finite")

    assert error.status_code == 500
    assert error.content == b'{"data":[{"context_length":500000}]}'
    assert error.headers["x-upstream"] == "error"
    assert non_json.status_code == 200
    assert non_json.content == b"upstream text"
    assert non_json.headers["x-upstream"] == "text"
    assert surrogate.content == b'{"data":[{"id":"grok","context_length":1,"label":"\\ud800"}]}'
    assert surrogate.headers["x-upstream"] == "surrogate"
    assert non_finite.content == b'{"data":[{"id":"grok","context_length":1}],"extra":NaN}'
    assert non_finite.headers["x-upstream"] == "non-finite"


def test_grok_response_unchanged_preserves_entity_validators(monkeypatch) -> None:
    async def no_chatgpt_metadata(http_client, request: Request, upstream_path: str) -> None:
        return None

    monkeypatch.setattr(
        "headroom.providers.model_metadata.handle_chatgpt_model_metadata",
        no_chatgpt_metadata,
    )

    class Proxy:
        http_client = "h2"

        async def handle_passthrough(self, request, base_url, sub_path="", provider_name=""):
            return Response(
                content=b'{"data":[{"id":"grok","context_length":1,"context_window":1}]}',
                status_code=200,
                headers={
                    "content-type": "application/json",
                    "etag": '"current"',
                    "last-modified": "Thu, 01 Jan 1970 00:00:00 GMT",
                    "cache-control": "max-age=60",
                },
            )

    app = FastAPI()

    @app.get("/probe")
    async def probe(request: Request):
        return await handle_model_metadata_endpoint(
            Proxy(),
            request,
            endpoint=MODEL_METADATA_LIST_ENDPOINT,
            provider_api_base_url="https://api.x.ai",
            provider_name="openai",
        )

    with TestClient(app) as client:
        response = client.get("/probe")

    assert response.headers["etag"] == '"current"'
    assert response.headers["last-modified"] == "Thu, 01 Jan 1970 00:00:00 GMT"
    assert response.headers["cache-control"] == "max-age=60"


def test_grok_negative_space_bypasses_non_xai_detail_and_aliases(monkeypatch) -> None:
    async def no_chatgpt_metadata(http_client, request: Request, upstream_path: str) -> None:
        return None

    monkeypatch.setattr(
        "headroom.providers.model_metadata.handle_chatgpt_model_metadata",
        no_chatgpt_metadata,
    )

    class Proxy:
        http_client = "h2"

        async def handle_passthrough(self, request, base_url, sub_path="", provider_name=""):
            if request.url.path == "/non-xai":
                content = b'{"data":[{"context_length":500000}]}'
            elif request.url.path == "/detail":
                content = b'{"id":"grok-4.6","context_length":500000}'
            else:
                content = b'{"data":[{"context_length":500000,"contextWindow":123}]}'
            return Response(content=content, status_code=200, media_type="application/json")

    app = FastAPI()

    @app.get("/{kind}")
    async def probe(request: Request, kind: str):
        return await handle_model_metadata_endpoint(
            Proxy(),
            request,
            endpoint=MODEL_METADATA_LIST_ENDPOINT
            if kind == "list"
            else model_metadata_get_endpoint("grok"),
            provider_api_base_url="https://api.x.ai"
            if kind != "non-xai"
            else "https://api.openai.com",
            provider_name="openai",
        )

    with TestClient(app) as client:
        non_xai = client.get("/non-xai")
        detail = client.get("/detail")
        alias = client.get("/list")

    assert non_xai.content == b'{"data":[{"context_length":500000}]}'
    assert detail.content == b'{"id":"grok-4.6","context_length":500000}'
    assert alias.json()["data"][0]["contextWindow"] == 123
