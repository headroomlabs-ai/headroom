"""System-prompt immutability tests for PR-A2 (P0-1 fix).

After PR-A2, memory context never mutates the system prompt or the
Responses API ``instructions`` field. The cache hot zone is sacrosanct
(invariant I2). Memory routes exclusively to the live-zone tail (the
first text block of the latest non-frozen user turn).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from headroom.proxy.server import ProxyConfig, create_app


class _FakePrefixTracker:
    def __init__(self, frozen_count: int):
        self._frozen_count = frozen_count
        self._cached_token_count = 0
        self._last_original_messages: list[dict[str, object]] = []
        self._last_forwarded_messages: list[dict[str, object]] = []

    def get_frozen_message_count(self) -> int:
        return self._frozen_count

    def get_last_original_messages(self):  # noqa: ANN201
        return self._last_original_messages.copy()

    def get_last_forwarded_messages(self):  # noqa: ANN201
        return self._last_forwarded_messages.copy()

    def update_from_response(self, **kwargs):  # noqa: ANN003
        self._cached_token_count = kwargs.get("cache_read_tokens", 0) + kwargs.get(
            "cache_write_tokens", 0
        )
        self._last_original_messages = kwargs.get(
            "original_messages", kwargs.get("messages", [])
        ).copy()
        self._last_forwarded_messages = kwargs.get("messages", []).copy()
        return None


def _make_proxy_client() -> TestClient:
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
    return TestClient(app)


def _install_memory_handler(proxy: object) -> None:
    """Wire a deterministic in-process memory handler that returns 'MEMCTX'."""
    proxy.memory_handler = SimpleNamespace(  # type: ignore[attr-defined]
        config=SimpleNamespace(inject_context=True, inject_tools=False),
        search_and_format_context=AsyncMock(return_value="MEMCTX"),
        has_memory_tool_calls=lambda resp, provider: False,
    )


def _install_session_tracker(proxy: object, frozen_count: int) -> None:
    fake_tracker = _FakePrefixTracker(frozen_count=frozen_count)
    proxy.session_tracker_store.compute_session_id = (  # type: ignore[attr-defined]
        lambda request, model, messages: "stable-session"
    )
    proxy.session_tracker_store.get_or_create = (  # type: ignore[attr-defined]
        lambda session_id, provider: fake_tracker
    )


def _install_capture_retry(proxy: object, captured: dict[str, object]) -> None:
    async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
        captured["body"] = body
        captured["headers"] = dict(headers)
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 3,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            },
        )

    proxy._retry_request = _fake_retry  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Anthropic /v1/messages tests
# ---------------------------------------------------------------------------


def test_memory_enabled_does_not_mutate_system() -> None:
    """Memory injection must not mutate the top-level ``system`` field."""
    captured: dict[str, object] = {}
    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = False
        proxy.config.image_optimize = False
        proxy.config.ccr_proactive_expansion = False

        _install_session_tracker(proxy, frozen_count=1)
        _install_memory_handler(proxy)
        _install_capture_retry(proxy, captured)

        original_system = "You are a helpful assistant."
        response = client.post(
            "/v1/messages",
            headers={
                "x-api-key": "test-key",
                "anthropic-version": "2023-06-01",
                "x-headroom-user-id": "u1",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "system": original_system,
                "messages": [
                    {"role": "user", "content": "frozen prefix"},
                    {"role": "assistant", "content": "ack"},
                    {"role": "user", "content": "latest user"},
                ],
            },
        )

        assert response.status_code == 200
        sent = captured["body"]
        assert isinstance(sent, dict)
        assert sent["system"] == original_system, (
            "system prompt must be byte-equal to client-sent value"
        )


def test_memory_context_appears_in_latest_user_message_tail() -> None:
    """Memory context appends to the latest non-frozen user turn's text."""
    captured: dict[str, object] = {}
    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = False
        proxy.config.image_optimize = False
        proxy.config.ccr_proactive_expansion = False

        _install_session_tracker(proxy, frozen_count=1)
        _install_memory_handler(proxy)
        _install_capture_retry(proxy, captured)

        response = client.post(
            "/v1/messages",
            headers={
                "x-api-key": "test-key",
                "anthropic-version": "2023-06-01",
                "x-headroom-user-id": "u1",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "system": "base system",
                "messages": [
                    {"role": "user", "content": "frozen prefix"},
                    {"role": "assistant", "content": "ack"},
                    {"role": "user", "content": "latest user"},
                ],
            },
        )

        assert response.status_code == 200
        sent = captured["body"]
        assert isinstance(sent, dict)
        assert sent["system"] == "base system"
        # Memory context appended to the *latest* user turn only.
        assert sent["messages"][0]["content"] == "frozen prefix"
        assert sent["messages"][2]["content"].endswith("MEMCTX")


def test_memory_context_byte_deterministic_for_same_query() -> None:
    """Two identical inbound requests produce identical outbound bytes."""
    capture_a: dict[str, object] = {}
    capture_b: dict[str, object] = {}

    def _send(capture: dict[str, object]) -> None:
        with _make_proxy_client() as client:
            proxy = client.app.state.proxy
            proxy.config.optimize = False
            proxy.config.image_optimize = False
            proxy.config.ccr_proactive_expansion = False

            _install_session_tracker(proxy, frozen_count=1)
            _install_memory_handler(proxy)
            _install_capture_retry(proxy, capture)

            response = client.post(
                "/v1/messages",
                headers={
                    "x-api-key": "test-key",
                    "anthropic-version": "2023-06-01",
                    "x-headroom-user-id": "u1",
                },
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 64,
                    "system": "base system",
                    "messages": [
                        {"role": "user", "content": "frozen prefix"},
                        {"role": "assistant", "content": "ack"},
                        {"role": "user", "content": "latest user"},
                    ],
                },
            )
            assert response.status_code == 200

    _send(capture_a)
    _send(capture_b)
    assert capture_a["body"] == capture_b["body"]


def test_memory_disabled_is_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """``HEADROOM_MEMORY_INJECTION_MODE=disabled`` skips injection."""
    monkeypatch.setenv("HEADROOM_MEMORY_INJECTION_MODE", "disabled")
    captured: dict[str, object] = {}
    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = False
        proxy.config.image_optimize = False
        proxy.config.ccr_proactive_expansion = False

        _install_session_tracker(proxy, frozen_count=1)
        _install_memory_handler(proxy)
        _install_capture_retry(proxy, captured)

        response = client.post(
            "/v1/messages",
            headers={
                "x-api-key": "test-key",
                "anthropic-version": "2023-06-01",
                "x-headroom-user-id": "u1",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "system": "base system",
                "messages": [{"role": "user", "content": "latest user"}],
            },
        )

        assert response.status_code == 200
        sent = captured["body"]
        assert isinstance(sent, dict)
        assert sent["system"] == "base system"
        assert sent["messages"][0]["content"] == "latest user"


def test_invalid_injection_mode_raises() -> None:
    """Unknown values for the env var must fail loudly — no silent fallback."""
    import os

    from headroom.proxy.helpers import get_memory_injection_mode

    prev = os.environ.get("HEADROOM_MEMORY_INJECTION_MODE")
    os.environ["HEADROOM_MEMORY_INJECTION_MODE"] = "system_prompt"
    try:
        with pytest.raises(ValueError, match="Invalid HEADROOM_MEMORY_INJECTION_MODE"):
            get_memory_injection_mode()
    finally:
        if prev is None:
            os.environ.pop("HEADROOM_MEMORY_INJECTION_MODE", None)
        else:
            os.environ["HEADROOM_MEMORY_INJECTION_MODE"] = prev


# ---------------------------------------------------------------------------
# OpenAI /v1/responses tests
# ---------------------------------------------------------------------------


def _make_responses_proxy_client() -> TestClient:
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
    return TestClient(app)


