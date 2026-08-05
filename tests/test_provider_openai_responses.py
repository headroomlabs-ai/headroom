from __future__ import annotations

from unittest.mock import patch

import httpx
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from headroom.providers.openai_responses import (
    OPENAI_RESPONSES_ROOT_PATHS,
    OPENAI_RESPONSES_SUBPATH_ROUTES,
    OPENAI_RESPONSES_WEBSOCKET_PATHS,
    OpenAIResponsesSubpathRoute,
    handle_openai_responses_subpath,
    normalize_openai_responses_headers,
    openai_responses_subpath_url,
)


def test_openai_responses_route_aliases_are_explicit() -> None:
    assert OPENAI_RESPONSES_ROOT_PATHS == (
        "/v1/responses",
        "/v1/codex/responses",
        "/backend-api/responses",
        "/backend-api/codex/responses",
    )
    assert OPENAI_RESPONSES_WEBSOCKET_PATHS == OPENAI_RESPONSES_ROOT_PATHS
    assert OPENAI_RESPONSES_SUBPATH_ROUTES == (
        OpenAIResponsesSubpathRoute("/v1/responses/{sub_path:path}", ("GET", "POST", "DELETE")),
        OpenAIResponsesSubpathRoute(
            "/v1/codex/responses/{sub_path:path}",
            ("GET", "POST", "DELETE"),
        ),
        OpenAIResponsesSubpathRoute(
            "/backend-api/responses/{sub_path:path}",
            ("GET", "POST", "DELETE"),
        ),
        OpenAIResponsesSubpathRoute(
            "/backend-api/codex/responses/{sub_path:path}",
            ("GET", "POST", "DELETE"),
        ),
    )


def test_openai_responses_route_aliases_are_unique() -> None:
    root_keys = {("POST", path) for path in OPENAI_RESPONSES_ROOT_PATHS}
    websocket_keys = {("WS", path) for path in OPENAI_RESPONSES_WEBSOCKET_PATHS}
    subpath_keys = {
        (method, spec.path) for spec in OPENAI_RESPONSES_SUBPATH_ROUTES for method in spec.methods
    }

    assert len(root_keys) == len(OPENAI_RESPONSES_ROOT_PATHS)
    assert len(websocket_keys) == len(OPENAI_RESPONSES_WEBSOCKET_PATHS)
    assert len(subpath_keys) == sum(len(spec.methods) for spec in OPENAI_RESPONSES_SUBPATH_ROUTES)


def test_openai_responses_subpath_url_includes_optional_query() -> None:
    assert (
        openai_responses_subpath_url(
            "https://api.openai.example/",
            "items/resp_1",
            "trace=2",
        )
        == "https://api.openai.example/v1/responses/items/resp_1?trace=2"
    )
    assert (
        openai_responses_subpath_url("https://api.openai.example", "compact")
        == "https://api.openai.example/v1/responses/compact"
    )


def test_normalize_openai_responses_headers_drops_host() -> None:
    assert normalize_openai_responses_headers(
        {"host": "localhost:8000", "authorization": "Bearer test"}
    ) == {"authorization": "Bearer test"}


def test_handle_openai_responses_subpath_forwards_body_headers_and_query() -> None:
    class FakeAsyncClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, dict[str, object]]] = []

        async def request(self, method: str, url: str, **kwargs: object) -> httpx.Response:
            self.calls.append((method, url, kwargs))
            return httpx.Response(202, json={"ok": True}, headers={"x-upstream": "yes"})

    fake = FakeAsyncClient()
    app = FastAPI()

    @app.post("/probe/{sub_path:path}")
    async def probe(request: Request, sub_path: str):
        return await handle_openai_responses_subpath(
            fake,
            request,
            "https://api.openai.example",
            sub_path,
        )

    with TestClient(app) as client:
        response = client.post(
            "/probe/items/resp_1?trace=1",
            headers={"Authorization": "Bearer test"},
            json={"model": "gpt-4o"},
        )

    assert response.status_code == 202
    assert response.headers["x-upstream"] == "yes"
    assert len(fake.calls) == 1
    method, url, kwargs = fake.calls[0]
    assert method == "POST"
    assert url == "https://api.openai.example/v1/responses/items/resp_1?trace=1"
    assert kwargs["headers"]["authorization"] == "Bearer test"  # type: ignore[index]
    assert kwargs["content"] == b'{"model":"gpt-4o"}'
    assert kwargs["timeout"] == 120.0


def test_handle_openai_responses_subpath_returns_502_on_failure() -> None:
    class FailingAsyncClient:
        async def request(self, method: str, url: str, **kwargs: object) -> httpx.Response:
            raise RuntimeError(f"boom: {method} {url}")

    app = FastAPI()

    @app.delete("/probe/{sub_path:path}")
    async def probe(request: Request, sub_path: str):
        return await handle_openai_responses_subpath(
            FailingAsyncClient(),
            request,
            "https://api.openai.example",
            sub_path,
        )

    with TestClient(app) as client:
        with patch("headroom.providers.openai_responses.logger") as logger:
            response = client.delete("/probe/items/resp_1?trace=1")

    assert response.status_code == 502
    assert response.text == "Upstream request failed."
    logger.error.assert_called_once()
    assert "boom: DELETE https://api.openai.example/v1/responses/items/resp_1?trace=1" in str(
        logger.error.call_args
    )
