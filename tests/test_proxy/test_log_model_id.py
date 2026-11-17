"""A client-supplied model id is logged as one token, never as raw text.

``sanitize_anthropic_model_id`` keeps an interior newline, so the handlers must
render the model with ``safe_id`` wherever they log it.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.models import CacheEntry  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402

MULTILINE_MODEL = "claude-x\n2026-01-01 00:00:00 INFO headroom.other second-line"


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.fixture
def proxy_logs():
    target = logging.getLogger("headroom.proxy")
    handler = _Capture()
    previous = target.level
    target.addHandler(handler)
    target.setLevel(logging.INFO)
    try:
        yield handler
    finally:
        target.removeHandler(handler)
        target.setLevel(previous)


def _config(**overrides) -> ProxyConfig:
    base = {
        "optimize": False,
        "cache_enabled": False,
        "rate_limit_enabled": False,
        "cost_tracking_enabled": False,
        "log_requests": False,
        "image_optimize": False,
    }
    base.update(overrides)
    return ProxyConfig(**base)


def _error_response() -> httpx.Response:
    return httpx.Response(
        400,
        content=json.dumps({"error": {"type": "invalid_request_error", "message": "bad"}}).encode(),
        headers={"content-type": "application/json"},
    )


def _assert_single_line(handler: _Capture, marker: str) -> None:
    lines = [m for m in handler.messages if marker in m]
    assert lines, f"{marker} was not logged"
    for line in lines:
        assert "\n" not in line
        assert "claude-x\\n2026" in line


@pytest.mark.parametrize(
    ("path", "headers"),
    [
        ("/v1/messages", {"x-api-key": "test-key", "anthropic-version": "2023-06-01"}),
        ("/v1/chat/completions", {"authorization": "Bearer test-key"}),
    ],
)
def test_upstream_error_log_keeps_model_on_one_line(proxy_logs, path, headers):
    body = {
        "model": MULTILINE_MODEL,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
    }
    with patch("headroom.proxy.server.AnyLLMBackend"):
        app = create_app(_config())
        with TestClient(app) as client:
            client.app.state.proxy._retry_request = AsyncMock(return_value=_error_response())
            client.post(path, headers=headers, json=body)

    _assert_single_line(proxy_logs, "UPSTREAM_ERROR")


def test_anthropic_cache_hit_log_keeps_model_on_one_line(proxy_logs):
    entry = CacheEntry(
        response_body=json.dumps(
            {
                "id": "msg_cached",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": "cached"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        ).encode(),
        response_headers={},
        created_at=datetime.now(),
        ttl_seconds=3600,
    )
    body = {
        "model": MULTILINE_MODEL,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
    }
    with patch("headroom.proxy.server.AnyLLMBackend"):
        app = create_app(_config(cache_enabled=True))
        with TestClient(app) as client:
            proxy = client.app.state.proxy
            proxy.cache.get = AsyncMock(return_value=entry)
            proxy._retry_request = AsyncMock(side_effect=AssertionError("no upstream on a hit"))
            client.post(
                "/v1/messages",
                headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
                json=body,
            )

    _assert_single_line(proxy_logs, "RESPONSE-CACHE-HIT")


def test_unknown_anthropic_model_warning_logs_model_as_one_token() -> None:
    from headroom.providers.anthropic import AnthropicProvider

    target = logging.getLogger("headroom.providers.anthropic")
    handler = _Capture()
    target.addHandler(handler)
    try:
        AnthropicProvider().get_context_limit(MULTILINE_MODEL + " provider-warning")
    finally:
        target.removeHandler(handler)

    [message] = [m for m in handler.messages if "Unknown Anthropic model" in m]
    assert "\n" not in message
    assert repr(MULTILINE_MODEL + " provider-warning") in message
