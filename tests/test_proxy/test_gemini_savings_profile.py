"""Regression test: the native Gemini generateContent compression path must
thread the proxy savings-profile kwargs (``proxy_pipeline_kwargs(config)``) into
``openai_pipeline.apply`` — the same way ``handlers/openai.py`` (#1534) and
``handlers/anthropic.py`` already do.

Before the fix the three Gemini/Vertex ``openai_pipeline.apply(...)`` call sites
passed only ``messages``/``model``/``model_limit``/``context``/``waste_messages``,
so ``HEADROOM_SAVINGS_PROFILE`` and the ProxyConfig compression knobs
(``target_ratio``/``min_tokens_to_compress``/``protect_recent``/...) were
silently dropped on the Gemini path.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402


def _make_fake_gemini_response() -> MagicMock:
    """A minimal stand-in for the httpx response returned by _retry_request."""
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"content-type": "application/json"}
    resp.content = b'{"candidates":[{"content":{"parts":[{"text":"ok"}]}}],"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":2}}'
    resp.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
        "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 2},
    }
    return resp


def test_gemini_generate_content_threads_savings_profile_kwargs_into_apply():
    """With HEADROOM_SAVINGS_PROFILE=agent-90, the native Gemini path must pass
    the profile knobs (compress_user_messages, target_ratio, ...) to apply()."""
    config = ProxyConfig(
        optimize=True,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
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

    app = create_app(config)
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        proxy.openai_pipeline.apply = MagicMock(side_effect=recording_apply)
        proxy._retry_request = AsyncMock(return_value=_make_fake_gemini_response())

        resp = client.post(
            "/v1beta/models/gemini-2.0-flash:generateContent?key=test-key",
            json={"contents": [{"parts": [{"text": big}]}]},
        )

    assert resp.status_code == 200, resp.text
    assert proxy.openai_pipeline.apply.call_count >= 1, "compression apply() never ran"

    # The agent-90 profile knobs must be present on the apply() call.
    assert captured.get("compress_user_messages") is True
    assert captured.get("target_ratio") == 0.10
    assert captured.get("min_tokens_to_compress") == 120
    assert captured.get("compress_system_messages") is True


def test_gemini_null_usage_counts_do_not_crash():
    """A Gemini response whose usageMetadata carries a null token count (e.g. a
    safety-blocked turn with no candidates) must not crash outcome recording:
    the counts are coerced to int, not left as None."""
    config = ProxyConfig(
        optimize=True,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )

    def passthrough_apply(**kwargs):
        return SimpleNamespace(
            messages=kwargs["messages"],
            transforms_applied=[],
            timing={},
            tokens_before=10,
            tokens_after=10,
            waste_signals=None,
        )

    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"content-type": "application/json"}
    resp.content = (
        b'{"candidates":[{"content":{"parts":[{"text":"ok"}]}}],'
        b'"usageMetadata":{"promptTokenCount":20,"candidatesTokenCount":null}}'
    )
    resp.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
        "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": None},
    }

    captured: dict[str, object] = {}

    async def recording_outcome(outcome):  # noqa: ANN001
        captured["outcome"] = outcome

    big = "word " * 4000
    app = create_app(config)
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        proxy.openai_pipeline.apply = MagicMock(side_effect=passthrough_apply)
        proxy._retry_request = AsyncMock(return_value=resp)
        proxy._record_request_outcome = AsyncMock(side_effect=recording_outcome)

        r = client.post(
            "/v1beta/models/gemini-2.0-flash:generateContent?key=test-key",
            json={"contents": [{"parts": [{"text": big}]}]},
        )

    assert r.status_code == 200, r.text
    outcome = captured["outcome"]
    assert outcome.output_tokens == 0
    assert isinstance(outcome.output_tokens, int)
    # max(0, promptTokenCount - cache_read) with a null candidate count must not raise.
    assert outcome.uncached_input_tokens == 20


def test_gemini_zero_usage_prompt_count_is_preserved():
    """A real zero promptTokenCount must stay zero, not fall back to estimates."""
    config = ProxyConfig(
        optimize=True,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )

    def passthrough_apply(**kwargs):
        return SimpleNamespace(
            messages=kwargs["messages"],
            transforms_applied=[],
            timing={},
            tokens_before=10,
            tokens_after=10,
            waste_signals=None,
        )

    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"content-type": "application/json"}
    resp.content = (
        b'{"candidates":[{"content":{"parts":[{"text":"ok"}]}}],'
        b'"usageMetadata":{"promptTokenCount":0,"candidatesTokenCount":0}}'
    )
    resp.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
        "usageMetadata": {"promptTokenCount": 0, "candidatesTokenCount": 0},
    }

    captured: dict[str, object] = {}

    async def recording_outcome(outcome):  # noqa: ANN001
        captured["outcome"] = outcome

    big = "word " * 4000
    app = create_app(config)
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        proxy.openai_pipeline.apply = MagicMock(side_effect=passthrough_apply)
        proxy._retry_request = AsyncMock(return_value=resp)
        proxy._record_request_outcome = AsyncMock(side_effect=recording_outcome)

        r = client.post(
            "/v1beta/models/gemini-2.0-flash:generateContent?key=test-key",
            json={"contents": [{"parts": [{"text": big}]}]},
        )

    assert r.status_code == 200, r.text
    outcome = captured["outcome"]
    assert outcome.optimized_tokens == 0
    assert outcome.uncached_input_tokens == 0
