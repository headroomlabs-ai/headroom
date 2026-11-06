"""Gemini response replay must strip upstream transport framing headers.

The Gemini generateContent / countTokens replay paths copied upstream headers
by hand and popped only content-encoding / content-length. They therefore
replayed headers the shared sanitize_forwarded_response_headers exists to drop:
transfer-encoding from a chunked upstream, connection-local connection /
keep-alive, and the upstream server banner.

A stale transfer-encoding on an already-decoded, re-framed body is the #3019
class of bug: RFC 9112 makes Transfer-Encoding override Content-Length, so the
client parses plain JSON as chunked frames.
"""

from __future__ import annotations

import json

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from headroom.proxy.server import ProxyConfig, create_app

_UPSTREAM_HEADERS = {
    "content-type": "application/json",
    "server": "upstream-marker/9.9",
    "keep-alive": "timeout=5, max=100",
    "connection": "keep-alive",
    "transfer-encoding": "chunked",
}

_FRAMING_HEADERS = ("server", "keep-alive", "connection", "transfer-encoding")

_GENERATE_BODY = json.dumps(
    {
        "candidates": [
            {
                "content": {"parts": [{"text": "hi"}], "role": "model"},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 4,
            "candidatesTokenCount": 1,
            "totalTokenCount": 5,
        },
    }
).encode()

_COUNT_TOKENS_BODY = json.dumps({"totalTokens": 4}).encode()


def _client(body: bytes) -> TestClient:
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        image_optimize=False,
    )
    app = create_app(config)
    proxy = app.state.proxy

    async def _fake_retry(*_args: object, **_kwargs: object) -> httpx.Response:
        return httpx.Response(200, headers=_UPSTREAM_HEADERS, content=body)

    proxy._retry_request = _fake_retry  # type: ignore[method-assign]
    return TestClient(app)


def _assert_upstream_framing_is_stripped(response: httpx.Response) -> None:
    lower = {name.lower() for name in response.headers}
    for header in _FRAMING_HEADERS:
        assert header not in lower, (
            f"upstream {header!r} reached the client: {dict(response.headers)}"
        )
    assert response.headers["content-type"].startswith("application/json")


def test_generate_content_replay_strips_upstream_framing_headers() -> None:
    client = _client(_GENERATE_BODY)

    response = client.post(
        "/v1beta/models/gemini-pro:generateContent?key=test-key",
        json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
    )

    assert response.status_code == 200
    assert response.json()["candidates"][0]["content"]["parts"][0]["text"] == "hi"
    _assert_upstream_framing_is_stripped(response)


def test_count_tokens_replay_strips_upstream_framing_headers() -> None:
    client = _client(_COUNT_TOKENS_BODY)

    response = client.post(
        "/v1beta/models/gemini-pro:countTokens?key=test-key",
        json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
    )

    assert response.status_code == 200
    assert response.json()["totalTokens"] == 4
    _assert_upstream_framing_is_stripped(response)
