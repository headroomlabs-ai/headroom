"""Regression test for #1534.

The live OpenAI `/v1/chat/completions` compression path must thread the proxy
savings-profile kwargs (``proxy_pipeline_kwargs(config)``) into
``openai_pipeline.apply`` — the same way ``handlers/anthropic.py`` and the
dedicated OpenAI compress endpoint do. Before the fix the chat path only passed
``model_limit``/``context``/``frozen_message_count``/``biases``/
``compression_policy``, so ``HEADROOM_SAVINGS_PROFILE=agent-90`` (and other
profile knobs) were silently dropped on the real chat path.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.backends.base import BackendResponse  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402


def _make_mock_backend() -> MagicMock:
    backend = MagicMock()
    backend.name = "anyllm-openai"
    backend.send_openai_message = AsyncMock(
        return_value=BackendResponse(
            body={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 2, "total_tokens": 102},
            },
            status_code=200,
            headers={"content-type": "application/json"},
        )
    )
    return backend


def _make_mock_backend_with_usage(usage: dict) -> MagicMock:
    backend = MagicMock()
    backend.name = "anyllm-openai"
    backend.send_openai_message = AsyncMock(
        return_value=BackendResponse(
            body={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": usage,
            },
            status_code=200,
            headers={"content-type": "application/json"},
        )
    )
    return backend


def test_chat_completions_survives_null_usage_token_counts():
    """A backend that reports present-but-null token counts must not 500.

    `.get(key, default)` returns None for a null value, and the chat path
    feeds those counts into `max(...)`/int-typed metrics. Without coercion a
    single such response crashes the request and its outcome recording
    (same class as the gemini fix in #2347).
    """
    config = ProxyConfig(
        optimize=True,
        cache_enabled=False,
        rate_limit_enabled=False,
        backend="anyllm",
        anyllm_provider="openai",
    )

    # prompt_tokens / completion_tokens present but explicitly null.
    mock_backend = _make_mock_backend_with_usage(
        {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    )
    with patch("headroom.proxy.server.AnyLLMBackend", return_value=mock_backend):
        app = create_app(config)
        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False,
                },
                headers={"Authorization": "Bearer test-key"},
            )

    assert resp.status_code == 200, resp.text


def test_chat_completions_threads_savings_profile_kwargs_into_apply():
    """With HEADROOM_SAVINGS_PROFILE=agent-90, the chat path must pass the
    profile knobs (compress_user_messages, target_ratio, ...) to apply()."""
    config = ProxyConfig(
        optimize=True,
        cache_enabled=False,
        rate_limit_enabled=False,
        backend="anyllm",
        anyllm_provider="openai",
        savings_profile="agent-90",
    )

    captured: dict[str, object] = {}

    def recording_apply(**kwargs):
        captured.update(kwargs)
        sent = kwargs["messages"]
        return SimpleNamespace(
            messages=sent,
            transforms_applied=[],
            timing={},
            tokens_before=4000,
            tokens_after=400,
            waste_signals=None,
        )

    # A large user message so the compression decision actually fires.
    big = "word " * 4000

    mock_backend = _make_mock_backend()
    with patch("headroom.proxy.server.AnyLLMBackend", return_value=mock_backend):
        app = create_app(config)
        with TestClient(app) as client:
            proxy = client.app.state.proxy
            proxy.openai_pipeline.apply = MagicMock(side_effect=recording_apply)

            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": big}],
                    "stream": False,
                },
                headers={"Authorization": "Bearer test-key"},
            )

    assert resp.status_code == 200, resp.text
    assert proxy.openai_pipeline.apply.call_count >= 1, "compression apply() never ran"

    # The agent-90 profile knobs must be present on the apply() call.
    assert captured.get("compress_user_messages") is True
    assert captured.get("target_ratio") == 0.10
    assert captured.get("min_tokens_to_compress") == 120
    assert captured.get("compress_system_messages") is True
