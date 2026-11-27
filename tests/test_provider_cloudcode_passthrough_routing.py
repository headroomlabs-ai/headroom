"""Regression tests for #4957: Antigravity Cloud Code passthrough routing.

Non-generateContent Cloud Code endpoints (fetchAvailableModels,
loadCodeAssist) fall through to the catch-all passthrough route instead of
the Gemini handler, so Antigravity requests must be resolved to the daily
sandbox host at that seam as well.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from headroom.providers.proxy_routes import register_provider_routes

DAILY_CLOUDCODE_TARGET = "https://daily-cloudcode-pa.sandbox.googleapis.com"
CLOUDCODE_TARGET = "https://cloudcode.example.test"
ANTHROPIC_TARGET = "https://anthropic.example.test"
ANTIGRAVITY_UA = "Antigravity/1.2.3 Darwin/arm64"


class _Runtime:
    @staticmethod
    def api_target(provider: str) -> str:
        return f"https://{provider}.example.test"

    @staticmethod
    def model_metadata_provider(headers: dict[str, str]) -> str:
        return "anthropic"


class _Proxy:
    ANTHROPIC_API_URL = ANTHROPIC_TARGET
    OPENAI_API_URL = "https://openai.example.test"
    GEMINI_API_URL = "https://gemini.example.test"
    CLOUDCODE_API_URL = CLOUDCODE_TARGET
    VERTEX_API_URL = "https://vertex.example.test"

    def __init__(self) -> None:
        self.config = SimpleNamespace(bedrock_api_url=None)
        self.provider_runtime = _Runtime()
        self.calls: list[dict[str, Any]] = []

    async def handle_passthrough(
        self,
        request: Any,
        base_url: str,
        endpoint_name: str = "",
        provider: str = "",
    ) -> JSONResponse:
        self.calls.append({"path": request.url.path, "base_url": base_url})
        return JSONResponse(self.calls[-1])


@pytest.fixture()
def proxy() -> _Proxy:
    return _Proxy()


@pytest.fixture()
def client(proxy: _Proxy) -> TestClient:
    app = FastAPI()
    register_provider_routes(app, proxy)
    return TestClient(app)


def test_antigravity_fetch_available_models_routes_to_daily_host(
    client: TestClient, proxy: _Proxy
) -> None:
    response = client.post(
        "/v1internal:fetchAvailableModels",
        headers={"user-agent": ANTIGRAVITY_UA},
        json={},
    )
    assert response.status_code == 200
    assert proxy.calls[-1]["base_url"] == DAILY_CLOUDCODE_TARGET


def test_antigravity_load_code_assist_versioned_path_routes_to_daily_host(
    client: TestClient, proxy: _Proxy
) -> None:
    response = client.post(
        "/v1/v1internal:loadCodeAssist",
        headers={"user-agent": ANTIGRAVITY_UA},
        json={},
    )
    assert response.status_code == 200
    assert proxy.calls[-1]["base_url"] == DAILY_CLOUDCODE_TARGET


def test_non_antigravity_cloudcode_passthrough_keeps_configured_target(
    client: TestClient, proxy: _Proxy
) -> None:
    response = client.post(
        "/v1internal:fetchAvailableModels",
        headers={"user-agent": "google-api-nodejs-client/9.15.1"},
        json={},
    )
    assert response.status_code == 200
    assert proxy.calls[-1]["base_url"] == CLOUDCODE_TARGET


def test_antigravity_user_agent_does_not_reroute_unrelated_paths(
    client: TestClient, proxy: _Proxy
) -> None:
    response = client.post(
        "/some/unrelated/path",
        headers={"user-agent": ANTIGRAVITY_UA},
        json={},
    )
    assert response.status_code == 200
    assert proxy.calls[-1]["base_url"] == ANTHROPIC_TARGET
