"""The response cache must not store an error delivered as HTTP 200.

Both handlers gate ``SemanticCache.set`` on ``status_code == 200``. Anthropic- and
OpenAI-compatible gateways can answer 200 with an error object (or an empty
object), and once stored that body is replayed to every matching non-streaming
request for the full TTL without the upstream ever being asked again.
"""

from __future__ import annotations

import json

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from headroom.proxy.semantic_cache import _is_cacheable_reply
from headroom.proxy.server import ProxyConfig, create_app

MESSAGES = [{"role": "user", "content": "Say hi."}]

ANTHROPIC_ERROR = {"type": "error", "error": {"type": "api_error", "message": "boom"}}
OPENAI_ERROR = {"error": {"message": "boom", "type": "server_error", "code": None}}

ANTHROPIC_OK = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Hello"}],
    "usage": {"input_tokens": 10, "output_tokens": 3},
}
OPENAI_OK = {
    "id": "chatcmpl_1",
    "object": "chat.completion",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
}


@pytest.mark.parametrize(
    "body,cacheable",
    [
        (json.dumps(ANTHROPIC_OK).encode(), True),
        (json.dumps(OPENAI_OK).encode(), True),
        (json.dumps({**OPENAI_OK, "error": None}).encode(), True),
        (json.dumps(ANTHROPIC_ERROR).encode(), False),
        (json.dumps(OPENAI_ERROR).encode(), False),
        (b"", False),
        (b"opaque-bytes", True),
        (b"{}", False),
        (b"null", False),
        (b"[]", False),
    ],
)
def test_is_cacheable_reply(body: bytes, cacheable: bool) -> None:
    assert _is_cacheable_reply(body) is cacheable


def _client() -> TestClient:
    config = ProxyConfig(
        optimize=False,
        cache_enabled=True,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
    )
    return TestClient(create_app(config))


@pytest.mark.parametrize(
    "path,headers,model,error_body,ok_body",
    [
        (
            "/v1/messages",
            {"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            "claude-haiku-4-5",
            ANTHROPIC_ERROR,
            ANTHROPIC_OK,
        ),
        (
            "/v1/chat/completions",
            {"authorization": "Bearer test-key"},
            "gpt-4o-mini",
            OPENAI_ERROR,
            OPENAI_OK,
        ),
    ],
)
def test_error_200_is_not_replayed_from_cache(path, headers, model, error_body, ok_body) -> None:
    calls = {"n": 0}

    with _client() as client:

        async def _fake_retry(method, url, req_headers, body, stream=False, **kwargs):  # noqa: ANN001
            calls["n"] += 1
            return httpx.Response(200, json=error_body if calls["n"] == 1 else ok_body)

        client.app.state.proxy._retry_request = _fake_retry
        request = {"model": model, "max_tokens": 64, "messages": MESSAGES, "stream": False}

        first = client.post(path, headers=headers, json=request)
        assert first.json() == error_body

        # Before the fix the error body was cached: this came back from the
        # cache with the upstream never called a second time.
        second = client.post(path, headers=headers, json=request)
        assert calls["n"] == 2
        assert second.json() == ok_body

        # A real reply is still cached.
        third = client.post(path, headers=headers, json=request)
        assert calls["n"] == 2
        assert third.json() == ok_body
