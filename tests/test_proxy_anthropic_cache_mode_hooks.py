"""Compression hooks on the Anthropic proxy path, in cache mode and token mode.

Cache mode skips the hooks that mutate or steer messages (``pre_compress``,
``compute_biases``, ``protect_messages``), because rewriting a cached prefix
busts the provider cache. ``post_compress`` only observes the result, so it
must still fire, with the turn's user query, in both modes.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from headroom.hooks import CompressContext, CompressEvent, CompressionHooks
from headroom.proxy.server import ProxyConfig, create_app

USER_QUERY = "which rows failed the import?"


class _RecordingHooks(CompressionHooks):
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.events: list[CompressEvent] = []
        self.contexts: list[CompressContext] = []

    def pre_compress(self, messages, ctx):  # noqa: ANN001, ANN201
        self.calls.append("pre_compress")
        self.contexts.append(ctx)
        return messages

    def compute_biases(self, messages, ctx):  # noqa: ANN001, ANN201
        self.calls.append("compute_biases")
        return {}

    def post_compress(self, event: CompressEvent) -> None:
        self.calls.append("post_compress")
        self.events.append(event)


class _ColdStartTracker:
    """A session with nothing frozen and nothing forwarded yet."""

    _cached_token_count = 0

    def get_frozen_message_count(self) -> int:
        return 0

    def get_last_original_messages(self):  # noqa: ANN201
        return []

    def get_last_forwarded_messages(self):  # noqa: ANN201
        return []

    def update_from_response(self, **kwargs):  # noqa: ANN003
        return None


def _force_compression(monkeypatch: pytest.MonkeyPatch) -> None:
    decision = SimpleNamespace(should_compress=True, passthrough_reason=None)
    decision.apply_to_tags = lambda tags: None
    monkeypatch.setattr(
        "headroom.proxy.handlers.anthropic.CompressionDecision.decide",
        lambda **kwargs: decision,
    )


def _send_compressible_turn(hooks: CompressionHooks, mode: str) -> dict:
    captured: dict = {}
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
        hooks=hooks,
    )
    with TestClient(create_app(config)) as client:
        proxy = client.app.state.proxy
        # Same order as the cache-stability tests: switch modes on the live
        # proxy, after startup has built its pipelines.
        proxy.config.optimize = True
        proxy.config.mode = mode
        tracker = _ColdStartTracker()
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stable-session"
        )
        proxy.session_tracker_store.get_or_create = (
            lambda session_id, provider, cache_ttl_seconds=None: tracker
        )

        def _shrinking_apply(**kwargs):  # noqa: ANN003, ANN202
            captured["applied"] = True
            shrunk = list(kwargs["messages"])
            shrunk[0] = {"role": "user", "content": "rows 3 and 9"}
            return SimpleNamespace(
                messages=shrunk,
                transforms_applied=["fake:shrink"],
                timing={},
                # Cache mode's cold start takes these as the turn's counts.
                tokens_before=900,
                tokens_after=40,
                waste_signals=None,
            )

        proxy.anthropic_pipeline.apply = _shrinking_apply

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001, ANN202
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "msg_hooks",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {"input_tokens": 10, "output_tokens": 1},
                },
            )

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 16,
                "messages": [
                    {"role": "user", "content": "import log: " + "row ok\n" * 800},
                    {"role": "assistant", "content": "Read it."},
                    {"role": "user", "content": USER_QUERY},
                ],
            },
        )
        assert response.status_code == 200, response.text
    return captured


def test_cache_mode_fires_post_compress_with_the_user_query(monkeypatch, caplog) -> None:  # noqa: ANN001
    _force_compression(monkeypatch)
    hooks = _RecordingHooks()

    with caplog.at_level(logging.WARNING, logger="headroom.proxy"):
        captured = _send_compressible_turn(hooks, mode="cache")

    assert captured.get("applied"), "the pipeline never ran"
    assert not [r for r in caplog.records if "] hook " in r.getMessage()]

    assert captured["body"]["messages"][0]["content"] == "rows 3 and 9"
    assert hooks.calls == ["post_compress"]
    (event,) = hooks.events
    assert event.user_query == USER_QUERY
    assert event.provider == "anthropic"
    assert event.tokens_saved > 0
    assert event.tokens_before - event.tokens_after == event.tokens_saved
    assert "fake:shrink" in event.transforms_applied


def test_token_mode_runs_every_hook_with_one_user_query(monkeypatch) -> None:  # noqa: ANN001
    _force_compression(monkeypatch)
    hooks = _RecordingHooks()

    _send_compressible_turn(hooks, mode="token")

    assert hooks.calls == ["pre_compress", "compute_biases", "post_compress"]
    (ctx,) = hooks.contexts
    (event,) = hooks.events
    assert ctx.user_query == USER_QUERY
    assert event.user_query == USER_QUERY
    assert event.tokens_saved > 0
