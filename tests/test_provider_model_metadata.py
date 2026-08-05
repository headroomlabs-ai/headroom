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
