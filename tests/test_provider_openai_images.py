from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.testclient import TestClient

from headroom.providers.openai_images import (
    OPENAI_IMAGE_ENDPOINTS,
    OpenAIImageEndpoint,
    codex_image_subpath,
    handle_openai_image_endpoint,
    select_codex_image_client,
)


def test_openai_image_endpoints_are_explicit() -> None:
    assert OPENAI_IMAGE_ENDPOINTS == (
        OpenAIImageEndpoint("/v1/images/generations", "images/generations"),
        OpenAIImageEndpoint("/v1/images/edits", "images/edits"),
    )


def test_codex_image_subpath_drops_openai_images_prefix() -> None:
    assert codex_image_subpath("images/generations") == "generations"
    assert codex_image_subpath("images/edits") == "edits"


def test_select_codex_image_client_prefers_h1_client() -> None:
    proxy = type("Proxy", (), {"http_client_h1": "h1", "http_client": "h2"})()
    fallback_proxy = type("Proxy", (), {"http_client": "h2"})()

    assert select_codex_image_client(proxy) == "h1"
    assert select_codex_image_client(fallback_proxy) == "h2"


def test_handle_openai_image_endpoint_returns_codex_response_when_present(monkeypatch) -> None:
    async def fake_codex_images(client: Any, request: Request, sub_path: str) -> Response:
        return JSONResponse({"client": client, "sub_path": sub_path})

    monkeypatch.setattr(
        "headroom.providers.openai_images.handle_chatgpt_codex_images",
        fake_codex_images,
    )
    proxy = type("Proxy", (), {"http_client_h1": "h1", "http_client": "h2"})()
    app = FastAPI()

    @app.post("/probe")
    async def probe(request: Request):
        return await handle_openai_image_endpoint(
            proxy,
            request,
            openai_api_base_url="https://api.openai.test",
            endpoint=OpenAIImageEndpoint("/probe", "images/generations"),
        )

    with TestClient(app) as client:
        response = client.post("/probe", json={"prompt": "test"})

    assert response.json() == {"client": "h1", "sub_path": "generations"}


def test_handle_openai_image_endpoint_falls_back_to_openai_passthrough(monkeypatch) -> None:
    async def fake_codex_images(client: Any, request: Request, sub_path: str) -> None:
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
        "headroom.providers.openai_images.handle_chatgpt_codex_images",
        fake_codex_images,
    )
    app = FastAPI()

    @app.post("/probe")
    async def probe(request: Request):
        return await handle_openai_image_endpoint(
            Proxy(),
            request,
            openai_api_base_url="https://api.openai.test",
            endpoint=OpenAIImageEndpoint("/probe", "images/edits"),
        )

    with TestClient(app) as client:
        response = client.post("/probe", json={"prompt": "test"})

    assert response.json() == {"provider": "openai", "sub_path": "images/edits"}
    assert calls == [("https://api.openai.test", "images/edits", "openai")]
