"""The configured token-per-minute bucket must be consumed by handlers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402


def test_anthropic_request_over_tpm_is_rejected_before_upstream(monkeypatch) -> None:
    # The first request finds a full bucket and is admitted even though it is
    # larger than the whole bucket (otherwise it could never pass); it is
    # charged in full, so the second one is refused before upstream.
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=True,
        rate_limit_requests_per_minute=1_000,
        rate_limit_tokens_per_minute=1,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
    )

    with TestClient(create_app(config), raise_server_exceptions=False) as client:
        proxy = client.app.state.proxy
        tokenizer = MagicMock()
        proxy._count_tokens_offloaded = AsyncMock(return_value=(tokenizer, 5_000))
        proxy._retry_request = AsyncMock(side_effect=RuntimeError("upstream stub"))

        def post():
            return client.post(
                "/v1/messages",
                headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
                json={
                    "model": "claude-haiku-4-5",
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": "large request"}],
                    "stream": False,
                },
            )

        assert post().status_code != 429
        assert proxy._retry_request.await_count == 1

        response = post()

        assert response.status_code == 429
        assert response.json()["detail"].startswith("Token rate limited.")
        assert proxy._retry_request.await_count == 1
        assert len(proxy.rate_limiter._token_buckets) == 1


@pytest.mark.parametrize(
    ("path", "headers", "body"),
    [
        (
            "/v1/chat/completions",
            {"authorization": "Bearer test-key"},
            {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "large request"}],
                "stream": False,
            },
        ),
        (
            "/v1/responses",
            {"authorization": "Bearer test-key"},
            {
                "model": "gpt-4o-mini",
                "input": "large request",
                "stream": False,
            },
        ),
        (
            "/v1beta/models/gemini-2.5-flash:generateContent",
            {"x-goog-api-key": "test-key"},
            {
                "contents": [{"role": "user", "parts": [{"text": "large request"}]}],
            },
        ),
    ],
)
def test_other_handlers_reject_request_over_tpm_before_upstream(
    monkeypatch, path, headers, body
) -> None:
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=True,
        rate_limit_requests_per_minute=1_000,
        rate_limit_tokens_per_minute=1,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
    )

    with TestClient(create_app(config), raise_server_exceptions=False) as client:
        proxy = client.app.state.proxy
        tokenizer = MagicMock()
        proxy._count_tokens_offloaded = AsyncMock(return_value=(tokenizer, 5_000))
        proxy._retry_request = AsyncMock(side_effect=RuntimeError("upstream stub"))

        assert client.post(path, headers=headers, json=body).status_code != 429
        (bucket,) = proxy.rate_limiter._token_buckets.values()
        assert bucket.tokens < 0  # admitted and charged in full
        upstream_calls = proxy._retry_request.await_count

        response = client.post(path, headers=headers, json=body)

        assert response.status_code == 429
        assert response.json()["detail"].startswith("Token rate limited.")
        assert proxy._retry_request.await_count == upstream_calls
        assert len(proxy.rate_limiter._token_buckets) == 1
