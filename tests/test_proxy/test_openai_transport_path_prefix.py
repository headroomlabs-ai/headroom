"""Tests for OpenAI transport path-prefix reconstruction from upstream hints."""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from headroom.proxy.server import ProxyConfig, create_app

_OPENAI_CHAT_PATH = "/v1/chat/completions"
_OPENAI_RESPONSES_PATH = "/v1/responses"


def _build_openai_client():
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
    )

    app = create_app(config)
    proxy = app.state.proxy
    captured: dict[str, object] = {}

    async def _fake_retry(
        method: str,
        url: str,
        headers: dict[str, str],
        body: dict,
        **_kwargs: object,
    ) -> httpx.Response:
        captured["method"] = method
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = body
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "object": "response",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "total_tokens": 12,
                },
            },
        )

    proxy._retry_request = _fake_retry
    proxy._record_request_outcome = AsyncMock()

    return TestClient(app), captured


def _assert_internal_header_absent(captured: dict[str, object], name: str) -> None:
    assert isinstance(captured.get("headers"), dict)
    headers = {k.lower() for k in captured["headers"].keys()}  # type: ignore[union-attr]
    assert name.lower() not in headers


def _assert_path(captured: dict[str, object], path: str) -> None:
    url = captured.get("url")
    assert isinstance(url, str)
    assert url.endswith(path)


def _assert_origin(captured: dict[str, object], origin: str) -> None:
    url = captured.get("url")
    assert isinstance(url, str)
    assert url.startswith(origin)


def test_chat_upstream_reconstruction_base_fails_head_passes() -> None:
    endpoint = _OPENAI_CHAT_PATH
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    base_fail = "://bad-base"

    for headers in [
        {
            "Authorization": "Bearer sk-test",
            "x-headroom-base-url": base_fail,
            "x-headroom-original-path": "/chat/completions",
        },
        {
            "Authorization": "Bearer sk-test",
            "x-headroom-base-url": "https://api.deepseek.com",
            "x-headroom-original-path": "/chat/completions",
        },
    ]:
        client, captured = _build_openai_client()
        response = client.post(endpoint, headers=headers, json=body)
        assert response.status_code == 200, response.text

        assert captured["method"] == "POST"
        if headers["x-headroom-base-url"] == base_fail:
            _assert_path(captured, "/v1/chat/completions")
        else:
            _assert_origin(captured, "https://api.deepseek.com")
            _assert_path(captured, "/chat/completions")


def test_responses_upstream_reconstruction_base_fails_head_passes() -> None:
    endpoint = _OPENAI_RESPONSES_PATH
    body = {"model": "gpt-4o", "input": "hi"}
    base_fail = "://bad-base"

    for headers in [
        {
            "Authorization": "Bearer sk-test",
            "x-headroom-base-url": base_fail,
            "x-headroom-original-path": "/responses",
        },
        {
            "Authorization": "Bearer sk-test",
            "x-headroom-base-url": "https://api.deepseek.com",
            "x-headroom-original-path": "/responses",
        },
    ]:
        client, captured = _build_openai_client()
        response = client.post(endpoint, headers=headers, json=body)
        assert response.status_code == 200, response.text

        assert captured["method"] == "POST"
        if headers["x-headroom-base-url"] == base_fail:
            _assert_path(captured, "/v1/responses")
        else:
            _assert_origin(captured, "https://api.deepseek.com")
            _assert_path(captured, "/responses")


def test_direct_v1_paths_are_preserved() -> None:
    cases = [
        (
            _OPENAI_CHAT_PATH,
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
            "/v1/chat/completions",
        ),
        (
            _OPENAI_RESPONSES_PATH,
            {"model": "gpt-4o", "input": "hi"},
            "/v1/responses",
        ),
    ]

    for endpoint, body, expected_path in cases:
        client, captured = _build_openai_client()
        response = client.post(
            endpoint,
            headers={"Authorization": "Bearer sk-test"},
            json=body,
        )
        assert response.status_code == 200, response.text
        assert captured["method"] == "POST"
        _assert_path(captured, expected_path)


def test_query_string_is_preserved_for_reconstructed_upstream_paths() -> None:
    cases = [
        (
            _OPENAI_CHAT_PATH,
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
            {
                "Authorization": "Bearer sk-test",
                "x-headroom-base-url": "https://api.deepseek.com",
                "x-headroom-original-path": "/base/chat/completions",
            },
            "/base/chat/completions?foo=1",
        ),
        (
            _OPENAI_RESPONSES_PATH,
            {"model": "gpt-4o", "input": "hi"},
            {
                "Authorization": "Bearer sk-test",
                "x-headroom-base-url": "https://api.deepseek.com",
                "x-headroom-original-path": "/base/responses",
            },
            "/base/responses?foo=1",
        ),
    ]

    for endpoint, body, headers, expected_path in cases:
        client, captured = _build_openai_client()
        response = client.post(f"{endpoint}?foo=1", headers=headers, json=body)
        assert response.status_code == 200, response.text
        assert captured["method"] == "POST"
        _assert_origin(captured, "https://api.deepseek.com")
        _assert_path(captured, expected_path)


def test_non_http_base_url_falls_back_to_v1() -> None:
    cases = [
        (
            _OPENAI_CHAT_PATH,
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
            {
                "Authorization": "Bearer sk-test",
                "x-headroom-base-url": "ws://api.deepseek.com",
                "x-headroom-original-path": "/chat/completions",
            },
            "/v1/chat/completions",
        ),
        (
            _OPENAI_RESPONSES_PATH,
            {"model": "gpt-4o", "input": "hi"},
            {
                "Authorization": "Bearer sk-test",
                "x-headroom-base-url": "wss://api.deepseek.com",
                "x-headroom-original-path": "/responses",
            },
            "/v1/responses",
        ),
    ]

    for endpoint, body, headers, expected_path in cases:
        client, captured = _build_openai_client()
        response = client.post(endpoint, headers=headers, json=body)
        assert response.status_code == 200, response.text
        assert captured["method"] == "POST"
        _assert_path(captured, expected_path)


def test_invalid_original_path_falls_back_to_v1() -> None:
    cases = [
        (
            _OPENAI_CHAT_PATH,
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
            {
                "Authorization": "Bearer sk-test",
                "x-headroom-base-url": "https://api.deepseek.com",
                "x-headroom-original-path": "https://evil.example/chat/completions",
            },
            "/v1/chat/completions",
        ),
        (
            _OPENAI_RESPONSES_PATH,
            {"model": "gpt-4o", "input": "hi"},
            {
                "Authorization": "Bearer sk-test",
                "x-headroom-base-url": "https://api.deepseek.com",
                "x-headroom-original-path": "/x/responses?bad",
            },
            "/v1/responses",
        ),
    ]

    for endpoint, body, headers, expected_path in cases:
        client, captured = _build_openai_client()
        response = client.post(endpoint, headers=headers, json=body)
        assert response.status_code == 200, response.text
        assert captured["method"] == "POST"
        _assert_path(captured, expected_path)


def test_original_path_header_is_not_forwarded_upstream() -> None:
    headers = {
        "Authorization": "Bearer sk-test",
        "x-headroom-base-url": "https://api.deepseek.com",
        "x-headroom-original-path": "/chat/completions",
    }
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}

    client, captured = _build_openai_client()
    response = client.post(_OPENAI_CHAT_PATH, headers=headers, json=body)
    assert response.status_code == 200, response.text

    _assert_internal_header_absent(captured, "x-headroom-original-path")
    _assert_internal_header_absent(captured, "x-headroom-base-url")
