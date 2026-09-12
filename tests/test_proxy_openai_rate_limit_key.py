"""Regression coverage for OpenAI credential-scoped rate limiting (#3364)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.handlers.openai import _openai_rate_limit_key  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402


def test_openai_rate_limit_key_accepts_api_key_header() -> None:
    assert _openai_rate_limit_key({"api-key": "gateway-key-A"}) != _openai_rate_limit_key(
        {"api-key": "gateway-key-B"}
    )


def test_openai_rate_limit_key_uses_complete_credential() -> None:
    shared_prefix = "a" * 20
    assert _openai_rate_limit_key({"api-key": f"{shared_prefix}-A"}) != _openai_rate_limit_key(
        {"api-key": f"{shared_prefix}-B"}
    )


def test_openai_rate_limit_key_prefers_authorization() -> None:
    headers = {"authorization": "Bearer primary", "api-key": "secondary"}
    assert _openai_rate_limit_key(headers) == _openai_rate_limit_key(
        {"authorization": "Bearer primary"}
    )


@pytest.mark.parametrize("endpoint", ["chat", "responses"])
def test_distinct_api_keys_do_not_share_openai_rate_limit_bucket(monkeypatch, endpoint) -> None:
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=True,
        rate_limit_requests_per_minute=1,
        rate_limit_tokens_per_minute=100_000,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
        retry_enabled=False,
    )

    if endpoint == "chat":
        path = "/v1/chat/completions"
        body = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        }
        upstream_body = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        }
    else:
        path = "/v1/responses"
        body = {"model": "gpt-4o-mini", "input": "hello", "stream": False}
        upstream_body = {
            "id": "resp-test",
            "object": "response",
            "status": "completed",
            "output": [],
            "usage": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
        }

    with TestClient(create_app(config)) as client:
        proxy = client.app.state.proxy
        proxy._retry_request = AsyncMock(
            side_effect=lambda *args, **kwargs: httpx.Response(200, json=upstream_body)
        )

        first = client.post(path, headers={"api-key": "gateway-key-A"}, json=body)
        second = client.post(path, headers={"api-key": "gateway-key-B"}, json=body)

        assert first.status_code == 200
        assert second.status_code == 200
        assert proxy._retry_request.await_count == 2
