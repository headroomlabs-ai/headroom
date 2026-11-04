"""Anthropic CCR exception path must re-raise — NOT silently return the raw tool call.

Regression test for the silent-fallback bug where ``handle_anthropic_messages``
caught CCR ``handle_response`` exceptions with ``logger.warning + continue``
instead of ``logger.error + raise`` (matching the OpenAI backend path).

When the CCR handler fails, the proxy must return a 500 rather than forwarding
the raw ``headroom_retrieve`` tool-call body to the client.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.cache.compression_store import get_compression_store  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402


def _make_config() -> ProxyConfig:
    return ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=True,
        ccr_handle_responses=True,
        ccr_context_tracking=False,
        image_optimize=False,
    )


def _tool_call_response(hash_key: str = "badhash") -> dict:
    """Anthropic-shaped response containing a headroom_retrieve tool use block."""
    return {
        "id": "msg_ccr_test",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_bad",
                "name": "headroom_retrieve",
                "input": {"hash": hash_key},
            }
        ],
        "stop_reason": "tool_use",
        "usage": {
            "input_tokens": 50,
            "output_tokens": 5,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    }


def test_anthropic_ccr_exception_reraises_not_swallowed():
    """When CCR handle_response raises, the proxy must 500 — not silently return
    the raw headroom_retrieve tool-call body to the client."""
    config = _make_config()
    tool_resp = _tool_call_response()

    with patch("headroom.proxy.server.AnyLLMBackend"):
        app = create_app(config)
        with TestClient(app) as client:
            proxy = client.app.state.proxy

            async def _fake_retry(method, url, headers, body, stream=False, **kwargs):
                return httpx.Response(200, json=tool_resp)

            proxy._retry_request = _fake_retry

            failing_handler = MagicMock()
            failing_handler.has_ccr_tool_calls = MagicMock(return_value=True)
            failing_handler.handle_response = AsyncMock(
                side_effect=RuntimeError("ccr-store-blew-up")
            )
            proxy.ccr_response_handler = failing_handler

            resp = client.post(
                "/v1/messages",
                headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

    # CCR error must propagate as an error response (502), NOT silently forward
    # the raw headroom_retrieve tool-call body to the client as a 200.
    failing_handler.handle_response.assert_awaited_once()
    assert resp.status_code != 200, (
        f"expected non-200 (CCR error re-raised), got 200: {resp.text[:200]}"
    )
    body = resp.json()
    # The outer handler sanitises uncaught exceptions into a 502 generic error.
    assert resp.status_code == 502, (
        f"expected 502 (CCR re-raise caught by outer handler), got {resp.status_code}: {resp.text[:200]}"
    )
    # Confirm the raw headroom_retrieve tool_use block was NOT returned.
    for block in body.get("content", []):
        assert block.get("name") != "headroom_retrieve", (
            "proxy silently forwarded the raw CCR tool call — silent fallback not fixed"
        )
    assert "error" in body


@pytest.mark.parametrize("status", [401, 429, 500])
def test_anthropic_ccr_non_2xx_continuation_keeps_original_response(status: int):
    """A failed CCR continuation must not become a synthetic HTTP 200 error."""
    config = _make_config()
    hash_key = get_compression_store().store(
        original="retrieved content",
        compressed="compressed content",
        original_item_count=1,
    )
    initial_response = _tool_call_response(hash_key)
    initial_bytes = json.dumps(initial_response, separators=(",", ":")).encode()
    continuation_error = {
        "type": "error",
        "error": {"type": "authentication_error", "message": "invalid api key"},
    }

    class _ContinuationClient:
        def __init__(self) -> None:
            self.post_calls: list[dict] = []

        async def post(self, url, *, content=None, headers=None, timeout=None):  # noqa: ANN001
            self.post_calls.append({"url": url, "content": content, "headers": headers})
            return httpx.Response(status, json=continuation_error)

        async def aclose(self) -> None:
            return None

    with patch("headroom.proxy.server.AnyLLMBackend"):
        app = create_app(config)
        with TestClient(app) as client:
            proxy = client.app.state.proxy

            async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
                return httpx.Response(
                    200,
                    content=initial_bytes,
                    headers={"content-type": "application/json"},
                )

            continuation_client = _ContinuationClient()
            proxy._retry_request = _fake_retry
            proxy.http_client = continuation_client

            response = client.post(
                "/v1/messages",
                headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": "retrieve it"}],
                },
            )

    assert len(continuation_client.post_calls) == 1
    assert response.status_code == 200
    assert response.json() == initial_response
    assert response.content == initial_bytes
