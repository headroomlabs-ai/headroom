"""Tests for best-effort Copilot model-name normalization.

Covers a repeatedly-observed live failure: an orchestrating agent asked to
review a PR "with Claude Opus 4.8, Sonnet 5, GPT Sol, GPT 5.5, Gemini" (or
similar casual phrasing) passes the literal display-style string as the
Task-tool ``model`` override instead of the canonical Copilot API model ID
(e.g. ``"Claude Opus 4.8"`` instead of ``"claude-opus-4.8"``). GitHub's
hosted API rejects these labels outright with `400 The requested model is
not supported`, confirmed live via `~/.headroom/logs/proxy.log` PERF lines
showing the exact literal strings reaching headroom unmodified.

`normalize_copilot_model_id()` is a best-effort correction, not a source of
truth for account entitlements: it only substitutes a normalized/aliased
form when that form matches a known Copilot model ID, and leaves anything
ambiguous or unrecognized untouched so it still fails loudly upstream
rather than silently routing to the wrong model.
"""

from __future__ import annotations

import httpx
import pytest

pytest.importorskip("fastapi")

from headroom.proxy.handlers.openai import normalize_copilot_model_id

_COPILOT_BASE = "https://api.githubcopilot.com"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Claude Opus 4.8", "claude-opus-4.8"),
        ("Claude Sonnet 5", "claude-sonnet-5"),
        ("GPT 5.5", "gpt-5.5"),
        ("Gemini 3.1 Pro (Preview)", "gemini-3.1-pro-preview"),
        ("GPT-5.4", "gpt-5.4"),
        ("GPT Sol", "gpt-5.6-sol"),
        ("Sol", "gpt-5.6-sol"),
        ("gpt sol", "gpt-5.6-sol"),
        # Already-canonical IDs pass through unchanged.
        ("gpt-5.4", "gpt-5.4"),
        ("claude-sonnet-4.6", "claude-sonnet-4.6"),
    ],
)
def test_normalize_copilot_model_id_corrects_known_labels(raw: str, expected: str) -> None:
    assert normalize_copilot_model_id(raw) == expected


@pytest.mark.parametrize("raw", ["Gemini", "unknown-model-xyz", "", None])
def test_normalize_copilot_model_id_leaves_ambiguous_names_untouched(raw: str | None) -> None:
    # Bare "Gemini" is ambiguous (multiple real Gemini model IDs exist) and
    # must not be guessed; unrecognized/empty input passes through as-is.
    assert normalize_copilot_model_id(raw) == raw


def test_bridge_normalizes_model_before_routing_and_forwarding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: a casually-typed model label is normalized before the
    routing decision (Copilot /chat/completions bridge) and before the
    outbound wire body is built, so the corrected ID -- not the original
    unrecognized label -- reaches GitHub's API."""
    pytest.importorskip("litellm")
    from fastapi.testclient import TestClient

    from headroom.proxy.server import ProxyConfig, create_app

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
    proxy = app.state.proxy
    captured: dict[str, object] = {}

    async def _fake_apply_copilot_api_auth(headers: dict[str, str], *, url: str) -> dict[str, str]:
        return {**headers, "Authorization": "******"}

    monkeypatch.setattr(
        "headroom.proxy.handlers.openai.apply_copilot_api_auth",
        _fake_apply_copilot_api_auth,
    )

    async def _fake_retry(
        method: str,
        url: str,
        headers: dict[str, str],
        body: dict,
        **_kwargs: object,
    ) -> httpx.Response:
        captured["url"] = url
        captured["body"] = body
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1,
                "model": "claude-opus-4.8",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    proxy._retry_request = _fake_retry

    async def _record_request_outcome(outcome: object) -> None:
        captured["outcome"] = outcome

    proxy._record_request_outcome = _record_request_outcome

    client = TestClient(app)
    response = client.post(
        "/v1/responses",
        headers={
            "Authorization": "******",
            "x-headroom-base-url": _COPILOT_BASE,
            "x-headroom-original-path": "/responses",
        },
        # Note the casual label, matching the exact live failure.
        json={"model": "Claude Opus 4.8", "input": "Hi there", "stream": False},
    )

    assert response.status_code == 200, response.text
    # Routed via the bridge (non-reasoning model) with the corrected ID.
    assert captured["url"].endswith("/chat/completions")
    assert captured["body"]["model"] == "claude-opus-4.8"
    assert response.json()["model"] == "claude-opus-4.8"
