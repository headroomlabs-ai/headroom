from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from headroom.providers.proxy_routes import register_provider_routes
from headroom.proxy.handlers.openai import OpenAIHandlerMixin


class _Runtime:
    @staticmethod
    def api_target(provider: str) -> str:
        return f"https://{provider}.example.test"

    @staticmethod
    def model_metadata_provider(headers: dict[str, str]) -> str:
        return "anthropic"


class _Proxy:
    ANTHROPIC_API_URL = "https://anthropic.example.test"
    OPENAI_API_URL = "https://openai.example.test"
    GEMINI_API_URL = "https://gemini.example.test"
    CLOUDCODE_API_URL = "https://cloudcode.example.test"
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
        self.calls.append(
            {
                "path": request.url.path,
                "base_url": base_url,
                "endpoint_name": endpoint_name,
                "provider": provider,
            }
        )
        return JSONResponse(self.calls[-1])


class _ChatCompletionsRequest:
    method = "POST"
    headers = {}
    url = SimpleNamespace(path="/zen/v1/chat/completions", query="")

    async def body(self) -> bytes:
        return b'{"model":"zen"}'


class _OpenAIUsageClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def request(self, **kwargs: Any) -> httpx.Response:
        self.calls.append(kwargs)
        request = httpx.Request(kwargs["method"], kwargs["url"])
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "application/json"},
            json={
                "usage": {
                    "prompt_tokens": 21,
                    "completion_tokens": 8,
                    "prompt_tokens_details": {"cached_tokens": 5},
                }
            },
        )


def test_custom_base_provider_prefixed_chat_completions_gets_telemetry() -> None:
    app = FastAPI()
    proxy = _Proxy()
    register_provider_routes(app, proxy)

    with TestClient(app) as client:
        for base_url, expected_base_url in (
            ("https://opencode.ai/", "https://opencode.ai"),
            ("https://www.opencode.ai/", "https://www.opencode.ai"),
        ):
            response = client.post(
                "/zen/v1/chat/completions",
                headers={"x-headroom-base-url": base_url},
                json={"model": "zen"},
            )
            assert response.status_code == 200
            assert response.json() == {
                "path": "/zen/v1/chat/completions",
                "base_url": expected_base_url,
                "endpoint_name": "chat/completions",
                "provider": "zen",
            }


def test_custom_base_unrelated_passthrough_paths_stay_unclassified() -> None:
    app = FastAPI()
    proxy = _Proxy()
    register_provider_routes(app, proxy)

    with TestClient(app) as client:
        for path in (
            "/mcp",
            "/mcp/v1/chat/completions",
            "/npm/v1/chat/completions",
            "/context7/v1/chat/completions",
        ):
            response = client.post(
                path,
                headers={"x-headroom-base-url": "https://opencode.ai/"},
                json={},
            )
            assert response.status_code == 200
            assert response.json() == {
                "path": path,
                "base_url": "https://opencode.ai",
                "endpoint_name": "",
                "provider": "",
            }


def test_custom_base_chat_completions_telemetry_is_post_and_opencode_zen_only() -> None:
    app = FastAPI()
    proxy = _Proxy()
    register_provider_routes(app, proxy)

    with TestClient(app) as client:
        get_response = client.get(
            "/zen/v1/chat/completions",
            headers={"x-headroom-base-url": "https://opencode.ai/"},
        )
        other_host_response = client.post(
            "/zen/v1/chat/completions",
            headers={"x-headroom-base-url": "https://custom.example/"},
            json={"model": "zen"},
        )
        double_slash_response = client.post(
            "/zen//v1/chat/completions",
            headers={"x-headroom-base-url": "https://opencode.ai/"},
            json={"model": "zen"},
        )
        trailing_slash_response = client.post(
            "/zen/v1/chat/completions/",
            headers={"x-headroom-base-url": "https://opencode.ai/"},
            json={"model": "zen"},
        )

    for response in (
        get_response,
        other_host_response,
        double_slash_response,
        trailing_slash_response,
    ):
        assert response.status_code == 200
        assert response.json()["endpoint_name"] == ""
        assert response.json()["provider"] == ""


def test_classified_custom_base_passthrough_records_telemetry_usage() -> None:
    handler = object.__new__(OpenAIHandlerMixin)
    handler.http_client = _OpenAIUsageClient()
    outcomes = []

    async def next_request_id() -> str:
        return "req_zen"

    async def record(outcome: Any) -> None:
        outcomes.append(outcome)

    handler._next_request_id = next_request_id
    handler._record_request_outcome = record

    response = asyncio.run(
        handler.handle_passthrough(
            _ChatCompletionsRequest(),
            "https://opencode.ai",
            "chat/completions",
            "zen",
        )
    )

    assert response.status_code == 200
    assert json.loads(response.body) == {
        "usage": {
            "prompt_tokens": 21,
            "completion_tokens": 8,
            "prompt_tokens_details": {"cached_tokens": 5},
        }
    }
    assert handler.http_client.calls[0]["url"] == ("https://opencode.ai/zen/v1/chat/completions")
    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome.provider == "zen"
    assert outcome.model == "passthrough:chat/completions"
    assert outcome.optimized_tokens == 21
    assert outcome.output_tokens == 8
    assert outcome.cache_read_tokens == 5
