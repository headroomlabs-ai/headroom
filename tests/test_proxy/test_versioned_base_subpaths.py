"""Responses sub-paths and OpenAI batches on a versioned OpenAI-compatible base.

Some OpenAI-compatible vendors serve their API under a version other than
``/v1`` (Volcengine Ark at ``https://ark.cn-beijing.volces.com/api/v3``).
``build_copilot_upstream_url`` stopped stacking ``/v1`` onto such a base
(#4917, for #4915), but the Responses sub-path route and the OpenAI batch/file
helpers built ``<base>/v1/...`` by hand and still hit ``.../api/v3/v1/...``.
These tests drive the real routes and record what the upstream receives.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

httpx = pytest.importorskip("httpx")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402

ARK_BASE = "https://ark.cn-beijing.volces.com/api/v3"


def _make_proxy_client(**config_overrides: Any) -> TestClient:
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
        **config_overrides,
    )
    return TestClient(create_app(config))


def _batch_upstream(seen: list[str], bodies: dict[str, bytes] | None = None) -> httpx.MockTransport:
    """An OpenAI-compatible upstream that answers the batch and file endpoints."""
    batch_line = json.dumps(
        {
            "custom_id": "req-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url}")
        if bodies is not None:
            bodies[f"{request.method} {request.url}"] = request.content
        path = request.url.path
        if path.endswith("/content"):
            return httpx.Response(200, text=batch_line + "\n")
        if path.endswith("/files"):
            return httpx.Response(200, json={"id": "file-uploaded", "object": "file"})
        if path.endswith("/batches"):
            return httpx.Response(200, json={"id": "batch_1", "object": "batch"})
        return httpx.Response(200, json={"id": "resp_123", "object": "response"})

    return httpx.MockTransport(handler)


@pytest.mark.parametrize(
    ("method", "path", "expected_path"),
    [
        ("GET", "/v1/responses/resp_123", "/api/v3/responses/resp_123"),
        ("DELETE", "/v1/responses/resp_123", "/api/v3/responses/resp_123"),
        ("POST", "/v1/responses/resp_123/cancel", "/api/v3/responses/resp_123/cancel"),
    ],
)
def test_responses_subpath_keeps_the_base_version(
    method: str, path: str, expected_path: str
) -> None:
    seen: list[str] = []
    with _make_proxy_client(openai_api_url=ARK_BASE) as client:
        client.app.state.proxy.http_client = httpx.AsyncClient(transport=_batch_upstream(seen))
        response = client.request(method, path, headers={"authorization": "Bearer synthetic"})

    assert response.status_code == 200, response.text
    assert seen == [f"{method} https://ark.cn-beijing.volces.com{expected_path}"]


def test_responses_subpath_on_an_unversioned_base_keeps_v1() -> None:
    seen: list[str] = []
    with _make_proxy_client(openai_api_url="https://api.deepseek.com") as client:
        client.app.state.proxy.http_client = httpx.AsyncClient(transport=_batch_upstream(seen))
        response = client.get("/v1/responses/resp_123", headers={"authorization": "Bearer x"})

    assert response.status_code == 200, response.text
    assert seen == ["GET https://api.deepseek.com/v1/responses/resp_123"]


def test_responses_subpath_on_copilot_follows_the_main_route_layout() -> None:
    # Copilot serves its OpenAI surface without /v1, and the main /v1/responses
    # route already strips it there; the sub-path route now matches it.
    seen: list[str] = []
    with _make_proxy_client(openai_api_url="https://api.githubcopilot.com") as client:
        client.app.state.proxy.http_client = httpx.AsyncClient(transport=_batch_upstream(seen))
        response = client.get("/v1/responses/resp_123", headers={"authorization": "Bearer x"})

    assert response.status_code == 200, response.text
    assert seen == ["GET https://api.githubcopilot.com/responses/resp_123"]


def test_batch_create_downloads_uploads_and_creates_under_the_base_version() -> None:
    seen: list[str] = []
    bodies: dict[str, bytes] = {}
    with _make_proxy_client(openai_api_url=ARK_BASE) as client:
        client.app.state.proxy.http_client = httpx.AsyncClient(
            transport=_batch_upstream(seen, bodies)
        )
        response = client.post(
            "/v1/batches",
            headers={"authorization": "Bearer synthetic"},
            json={
                "input_file_id": "file-abc",
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            },
        )

    assert response.status_code == 200, response.text
    assert seen == [
        f"GET {ARK_BASE}/files/file-abc/content",
        f"POST {ARK_BASE}/files",
        f"POST {ARK_BASE}/batches",
    ]
    # The batch is created from the file the proxy just uploaded.
    assert json.loads(bodies[f"POST {ARK_BASE}/batches"])["input_file_id"] == "file-uploaded"


def test_batch_passthrough_creates_under_the_base_version() -> None:
    # Non-chat batch endpoints skip compression and are forwarded as-is.
    seen: list[str] = []
    with _make_proxy_client(openai_api_url=ARK_BASE) as client:
        client.app.state.proxy.http_client = httpx.AsyncClient(transport=_batch_upstream(seen))
        response = client.post(
            "/v1/batches",
            headers={"authorization": "Bearer synthetic"},
            json={
                "input_file_id": "file-abc",
                "endpoint": "/v1/embeddings",
                "completion_window": "24h",
            },
        )

    assert response.status_code == 200, response.text
    assert seen == [f"POST {ARK_BASE}/batches"]
