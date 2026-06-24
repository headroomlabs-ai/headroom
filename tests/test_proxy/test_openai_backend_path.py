"""Tests for the OpenAI chat-completions backend (LiteLLM/Bedrock) path.

Covers Fix #1 (PrefixCacheTracker.update_from_response on backend path)
and Fix #2 (CCR response intercept for the OpenAI provider shape) on the
non-streaming backend path of ``handle_openai_chat``.

All three scenarios mock ``anthropic_backend.send_openai_message`` so we
don't need a real provider:

1. Backend response with cache_read_input_tokens > 0 → tracker.update_from_response
   is called with the right cache_read_tokens and cache_write_tokens.
2. Backend response with headroom_retrieve tool call → ccr_response_handler.handle_response
   is awaited with provider="openai", and the final body returned.
3. CCR intercept exception path → re-raises (NOT swallowed).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.backends.base import BackendResponse  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402


class _RecordingTracker:
    """Stub PrefixCacheTracker that records ``update_from_response`` calls."""

    def __init__(self, provider: str = "openai") -> None:
        self.provider = provider
        self.calls: list[dict] = []
        self._frozen = 0
        self._last_original: list[dict] = []
        self._last_forwarded: list[dict] = []

    def update_from_response(
        self,
        cache_read_tokens: int,
        cache_write_tokens: int,
        messages: list[dict],
        message_token_counts: list[int] | None = None,
        original_messages: list[dict] | None = None,
    ) -> None:
        self.calls.append(
            {
                "cache_read_tokens": cache_read_tokens,
                "cache_write_tokens": cache_write_tokens,
                "messages": messages,
            }
        )
        self._last_original = list(original_messages or messages)
        self._last_forwarded = list(messages)

    # Minimal surface used by handle_openai_chat — return 0 so we never freeze.
    def get_frozen_message_count(self) -> int:
        return self._frozen

    def get_last_original_messages(self) -> list[dict]:
        return list(self._last_original)

    def get_last_forwarded_messages(self) -> list[dict]:
        return list(self._last_forwarded)


def _make_config() -> ProxyConfig:
    return ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        backend="anyllm",
        anyllm_provider="openai",
    )


def _make_mock_backend(response_body: dict, status_code: int = 200) -> MagicMock:
    backend = MagicMock()
    backend.name = "anyllm-openai"
    backend.send_openai_message = AsyncMock(
        return_value=BackendResponse(
            body=response_body,
            status_code=status_code,
            headers={"content-type": "application/json"},
        )
    )
    return backend


def _install_tracker_stub(client: TestClient) -> _RecordingTracker:
    """Force the session_tracker_store to hand out our recording tracker."""
    tracker = _RecordingTracker(provider="openai")
    # Find the proxy instance behind the app — it's stored as app.state.proxy.
    proxy = client.app.state.proxy
    proxy.session_tracker_store.get_or_create = MagicMock(return_value=tracker)
    return tracker


def test_backend_response_updates_prefix_tracker_with_bedrock_cache_fields():
    """Bedrock/Anthropic-shape cache fields → tracker sees authoritative read/write counts."""
    config = _make_config()
    response_body = {
        "id": "chatcmpl-bedrock-1",
        "object": "chat.completion",
        "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hi!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 20,
            "total_tokens": 1020,
            # Bedrock/Anthropic top-level keys
            "cache_read_input_tokens": 700,
            "cache_creation_input_tokens": 100,
            # OpenAI shape (always populated by the LiteLLM normalizer)
            "prompt_tokens_details": {"cached_tokens": 700},
        },
    }

    mock_backend = _make_mock_backend(response_body)
    with patch("headroom.proxy.server.AnyLLMBackend", return_value=mock_backend):
        app = create_app(config)
        with TestClient(app) as client:
            tracker = _install_tracker_stub(client)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                },
                headers={"Authorization": "Bearer test-key"},
            )

    assert resp.status_code == 200, resp.text
    assert mock_backend.send_openai_message.await_count == 1
    assert len(tracker.calls) == 1, tracker.calls
    call = tracker.calls[0]
    # Prefer the Bedrock authoritative top-level read/write counts.
    assert call["cache_read_tokens"] == 700
    assert call["cache_write_tokens"] == 100


def test_backend_response_falls_back_to_openai_cached_tokens_when_bedrock_keys_absent():
    """Pure OpenAI shape (no top-level Anthropic keys) → fall back to prompt_tokens_details + infer write."""
    config = _make_config()
    response_body = {
        "id": "chatcmpl-openai-1",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hi!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 500,
            "completion_tokens": 10,
            "total_tokens": 510,
            # No top-level Anthropic keys, only OpenAI shape
            "prompt_tokens_details": {"cached_tokens": 200},
        },
    }

    mock_backend = _make_mock_backend(response_body)
    with patch("headroom.proxy.server.AnyLLMBackend", return_value=mock_backend):
        app = create_app(config)
        with TestClient(app) as client:
            tracker = _install_tracker_stub(client)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                },
                headers={"Authorization": "Bearer test-key"},
            )

    assert resp.status_code == 200, resp.text
    assert len(tracker.calls) == 1
    call = tracker.calls[0]
    assert call["cache_read_tokens"] == 200
    # No cache_creation_input_tokens → inferred = prompt_tokens - cache_read = 500 - 200 = 300
    assert call["cache_write_tokens"] == 300


def test_backend_response_with_ccr_tool_call_is_intercepted_and_resolved():
    """OpenAI-shape response carrying headroom_retrieve → CCR handler resolves it."""
    config = _make_config()
    # First response: tool_call for headroom_retrieve
    tool_call_response = {
        "id": "chatcmpl-ccr-1",
        "object": "chat.completion",
        "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": "headroom_retrieve",
                                "arguments": '{"hash": "deadbeef"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 110,
        },
    }
    final_resp_json = {
        "id": "chatcmpl-ccr-final",
        "object": "chat.completion",
        "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Resolved!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "total_tokens": 105,
        },
    }

    mock_backend = _make_mock_backend(tool_call_response)
    with patch("headroom.proxy.server.AnyLLMBackend", return_value=mock_backend):
        app = create_app(config)
        with TestClient(app) as client:
            _install_tracker_stub(client)
            proxy = client.app.state.proxy
            # Replace the response handler with a recording mock.
            recording_handler = MagicMock()
            recording_handler.has_ccr_tool_calls = MagicMock(return_value=True)
            recording_handler.handle_response = AsyncMock(return_value=final_resp_json)
            proxy.ccr_response_handler = recording_handler

            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                },
                headers={"Authorization": "Bearer test-key"},
            )

    assert resp.status_code == 200, resp.text
    # handle_response was awaited with provider="openai"
    recording_handler.handle_response.assert_awaited_once()
    _args, kwargs = recording_handler.handle_response.call_args
    assert kwargs.get("provider") == "openai"
    # Resolved body propagated back to the client
    assert resp.json()["choices"][0]["message"]["content"] == "Resolved!"


def test_backend_ccr_intercept_exception_is_reraised_not_swallowed():
    """CCR resolution failure on the backend path → 500, NOT silent fallback to original body."""
    config = _make_config()
    tool_call_response = {
        "id": "chatcmpl-ccr-fail",
        "object": "chat.completion",
        "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_bad",
                            "type": "function",
                            "function": {
                                "name": "headroom_retrieve",
                                "arguments": '{"hash": "badhash"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 5, "total_tokens": 55},
    }

    mock_backend = _make_mock_backend(tool_call_response)
    with patch("headroom.proxy.server.AnyLLMBackend", return_value=mock_backend):
        app = create_app(config)
        with TestClient(app) as client:
            _install_tracker_stub(client)
            proxy = client.app.state.proxy
            failing_handler = MagicMock()
            failing_handler.has_ccr_tool_calls = MagicMock(return_value=True)
            failing_handler.handle_response = AsyncMock(
                side_effect=RuntimeError("ccr-store-blew-up")
            )
            proxy.ccr_response_handler = failing_handler

            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                },
                headers={"Authorization": "Bearer test-key"},
            )

    # The outer `try/except Exception` on the backend block converts the
    # re-raise into a 500 response. The critical assertion is that the
    # original tool_call body is NOT returned to the client — which is
    # what a silent fallback would do.
    failing_handler.handle_response.assert_awaited_once()
    assert resp.status_code == 500, (
        f"expected 500 (CCR error re-raised), got {resp.status_code}: {resp.text[:200]}"
    )
    body = resp.json()
    # Confirm we didn't propagate the original tool_call body.
    assert (
        "choices" not in body
        or body.get("choices", [{}])[0].get("message", {}).get("tool_calls") is None
    )
    assert "error" in body
    assert "ccr-store-blew-up" in body["error"]["message"]


def test_backend_streaming_passes_prefix_tracker_through():
    """Streaming backend path should accept and use prefix_tracker — non-regression smoke."""
    # The wiring contract is structural — just confirm the parameter exists.
    import inspect

    from headroom.proxy.handlers.streaming import StreamingMixin

    sig = inspect.signature(StreamingMixin._stream_openai_via_backend)
    assert "prefix_tracker" in sig.parameters, (
        "_stream_openai_via_backend must accept prefix_tracker to match the direct path"
    )
    assert "optimized_messages" in sig.parameters, (
        "_stream_openai_via_backend must accept optimized_messages so the "
        "tracker can record the messages that were sent"
    )
