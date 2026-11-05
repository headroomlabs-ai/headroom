"""Response-cache keys must survive in-place pre_compress mutations."""

from __future__ import annotations

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.hooks import CompressionHooks  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402


class _InPlaceHooks(CompressionHooks):
    def pre_compress(self, messages, ctx):
        for message in messages:
            if message.get("role") == "user":
                message["content"] = "rewritten in place"
        return messages


@pytest.mark.parametrize(
    ("provider", "path", "headers", "body", "response"),
    [
        (
            "openai",
            "/v1/chat/completions",
            {"authorization": "Bearer test-key"},
            {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "original"}]},
            {
                "id": "chatcmpl_1",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "reply"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            },
        ),
        (
            "anthropic",
            "/v1/messages",
            {"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            {
                "model": "claude-haiku-4-5",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "original"}],
            },
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "reply"}],
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 1,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            },
        ),
    ],
)
def test_in_place_pre_compress_mutation_keeps_response_cache_hit(
    provider: str, path: str, headers: dict, body: dict, response: dict
) -> None:
    calls = {"n": 0}
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
        hooks=_InPlaceHooks(),
    )

    with TestClient(create_app(config)) as client:
        proxy = client.app.state.proxy

        async def _fake_retry(method, url, request_headers, request_body, stream=False, **kwargs):
            calls["n"] += 1
            result = dict(response)
            result["id"] = f"{provider}_{calls['n']}"
            return httpx.Response(200, json=result)

        proxy._retry_request = _fake_retry
        first = client.post(path, headers=headers, json=body)
        second = client.post(path, headers=headers, json=body)

    assert first.status_code == second.status_code == 200
    assert calls["n"] == 1
    assert first.json() == second.json()
