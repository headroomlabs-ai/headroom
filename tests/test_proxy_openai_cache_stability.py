"""Regression tests for OpenAI cache-mode stability in proxy mode."""

from __future__ import annotations

import copy
import json
import pathlib
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from headroom.backends.base import BackendResponse
from headroom.observability import HeadroomOtelMetrics, reset_otel_metrics, set_otel_metrics
from headroom.proxy.server import ProxyConfig, create_app
from tests.test_observability_metrics import _collect_metrics


class _FakePrefixTracker:
    def __init__(
        self,
        frozen_count: int,
        previous_original: list[dict] | None = None,
        previous_forwarded: list[dict] | None = None,
    ):
        self._frozen_count = frozen_count
        self._previous_original = previous_original or []
        self._previous_forwarded = previous_forwarded or []
        self.update_calls: list[dict] = []

    def get_frozen_message_count(self) -> int:
        return self._frozen_count

    # Empty history → overlay_cached_prefix() is a no-op here, so these tests
    # keep asserting the cache-freeze behavior they always have. The cross-turn
    # overlay itself is exercised in test_cross_turn_cache_safety.py against the
    # real tracker; these stubs just satisfy the handler's overlay call.
    def get_last_original_messages(self):  # noqa: ANN201
        return copy.deepcopy(self._previous_original)

    def get_last_forwarded_messages(self):  # noqa: ANN201
        return copy.deepcopy(self._previous_forwarded)

    def update_from_response(self, **kwargs):  # noqa: ANN003
        self.update_calls.append(kwargs)
        return None


def _make_proxy_client(**config_overrides) -> TestClient:  # noqa: ANN003
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
        **config_overrides,
    )
    app = create_app(config)
    return TestClient(app)


def test_openai_cache_mode_freezes_previous_turns() -> None:
    captured = {}
    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = True
        proxy.config.mode = "cache"

        fake_tracker = _FakePrefixTracker(frozen_count=0)
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stable-session"
        )
        proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker

        def _fake_apply(**kwargs):
            captured["frozen_message_count"] = kwargs.get("frozen_message_count")
            return SimpleNamespace(
                messages=kwargs["messages"],
                transforms_applied=[],
                timing={},
                tokens_before=60,
                tokens_after=60,
                waste_signals=None,
            )

        proxy.openai_pipeline.apply = _fake_apply

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl_1",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 60, "completion_tokens": 3, "total_tokens": 63},
                },
            )

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer test-key"},
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "user", "content": "turn1"},
                    {"role": "assistant", "content": "turn1-assistant"},
                    {"role": "user", "content": "current turn"},
                ],
            },
        )

        assert response.status_code == 200
        assert captured["frozen_message_count"] == 2


def test_openai_handler_replays_nonempty_cached_prefix() -> None:
    captured = {}
    previous_original = [{"role": "user", "content": "original prefix"}]
    previous_forwarded = [{"role": "user", "content": "comp"}]
    fake_tracker = _FakePrefixTracker(0, previous_original, previous_forwarded)
    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stable-session"
        )
        proxy.session_tracker_store.resolve_tracker = lambda *args, **kwargs: fake_tracker

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl_overlay",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23},
                },
            )

        proxy._retry_request = _fake_retry
        response = client.post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer test-key"},
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "user", "content": "original prefix"},
                    {"role": "user", "content": "new suffix"},
                ],
            },
        )

    assert response.status_code == 200
    assert captured["body"]["messages"] == [
        previous_forwarded[0],
        {"role": "user", "content": "new suffix"},
    ]


@pytest.mark.parametrize("tail_role", ["tool", "function"])
def test_openai_cache_mode_keeps_final_tool_observation_mutable(tail_role: str) -> None:
    captured = {}
    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = True
        proxy.config.mode = "cache"

        fake_tracker = _FakePrefixTracker(frozen_count=0)
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stable-session"
        )
        proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker

        def _fake_apply(**kwargs):
            captured.setdefault("calls", []).append(
                {
                    "frozen_message_count": kwargs.get("frozen_message_count"),
                    "roles": [msg.get("role") for msg in kwargs["messages"]],
                    "mode": proxy.config.mode,
                }
            )
            return SimpleNamespace(
                messages=kwargs["messages"],
                transforms_applied=["test:compress-tail"],
                timing={},
                tokens_before=120,
                tokens_after=80,
                waste_signals=None,
            )

        proxy.openai_pipeline.apply = _fake_apply

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl_tool_tail",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 80, "completion_tokens": 3, "total_tokens": 83},
                },
            )

        proxy._retry_request = _fake_retry

        tail = {
            "role": tail_role,
            "content": "large command observation " * 200,
        }
        if tail_role == "tool":
            tail["tool_call_id"] = "call_1"
        else:
            tail["name"] = "bash"

        response = client.post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer test-key"},
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "user", "content": "turn1"},
                    {"role": "assistant", "content": "run command"},
                    tail,
                ],
            },
        )

        assert response.status_code == 200
        assert any(call["frozen_message_count"] == 2 for call in captured["calls"]), captured[
            "calls"
        ]


def test_openai_cache_mode_restores_mutated_frozen_prefix() -> None:
    captured = {}
    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = True
        proxy.config.mode = "cache"

        fake_tracker = _FakePrefixTracker(frozen_count=0)
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stable-session"
        )
        proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker

        original_messages = [
            {"role": "user", "content": "turn1"},
            {"role": "assistant", "content": "turn1-assistant"},
            {"role": "user", "content": "current turn"},
        ]

        def _fake_apply(**kwargs):
            mutated = list(kwargs["messages"])
            mutated[0] = {**mutated[0], "content": "MUTATED_PREFIX"}
            return SimpleNamespace(
                messages=mutated,
                transforms_applied=["fake:mutated"],
                timing={},
                tokens_before=70,
                tokens_after=65,
                waste_signals=None,
            )

        proxy.openai_pipeline.apply = _fake_apply

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl_2",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 65, "completion_tokens": 3, "total_tokens": 68},
                },
            )

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer test-key"},
            json={
                "model": "gpt-4o-mini",
                "messages": original_messages,
            },
        )

        assert response.status_code == 200
        sent_messages = captured["body"]["messages"]
        assert sent_messages[0] == original_messages[0]
        assert sent_messages[1] == original_messages[1]


# ─── Issue #327 cross-handler regression ────────────────────────────────
#
# The OpenAI handler was never affected by issue #327's content-keyed walker
# bug — it has only ever used `compute_frozen_count` (positional). This test
# locks that property by spying on the OpenAI traffic path and asserting that
# the buggy walker functions (`should_defer_compression`, `mark_stable`) are
# never called from the production handler. If a future refactor accidentally
# adds the same walker to OpenAI, this test fails immediately.


def test_issue_327_openai_handler_does_not_call_walker_functions() -> None:
    calls: list[tuple[str, tuple, dict]] = []

    class _SpyCompCache:
        def apply_cached(self, messages):  # noqa: ANN001
            calls.append(("apply_cached", (), {}))
            return list(messages)

        def compute_frozen_count(self, messages):  # noqa: ANN001
            calls.append(("compute_frozen_count", (), {}))
            return 0

        def update_from_result(self, originals, compressed):  # noqa: ANN001
            calls.append(("update_from_result", (), {}))

        def mark_stable_from_messages(self, messages, up_to):  # noqa: ANN001
            calls.append(("mark_stable_from_messages", (up_to,), {}))

        # Methods below MUST NOT be called from OpenAI handler.
        def should_defer_compression(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            calls.append(("should_defer_compression", args, kwargs))
            return False

        def mark_stable(self, content_hash):  # noqa: ANN001
            calls.append(("mark_stable", (content_hash,), {}))

        @staticmethod
        def content_hash(content):  # noqa: ANN001
            return f"H({content[:40] if isinstance(content, str) else 'list'})"

    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = True
        proxy.config.mode = "token"  # token mode is where Anthropic had the bug

        fake_tracker = _FakePrefixTracker(frozen_count=0)
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "openai-spy-session"
        )
        proxy.session_tracker_store.get_or_create = lambda s, p: fake_tracker
        proxy._get_compression_cache = lambda s: _SpyCompCache()

        def _fake_apply(**kwargs):  # noqa: ANN003
            return SimpleNamespace(
                messages=list(kwargs["messages"]),
                transforms_applied=[],
                timing={},
                tokens_before=60,
                tokens_after=60,
                waste_signals=None,
            )

        proxy.openai_pipeline.apply = _fake_apply

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            return httpx.Response(
                200,
                json={
                    "id": "cmpl",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 60, "completion_tokens": 3, "total_tokens": 63},
                },
            )

        proxy._retry_request = _fake_retry

        # Drive 5 turns so any walker bug would have time to fire repeatedly.
        for turn in range(5):
            r = client.post(
                "/v1/chat/completions",
                headers={"authorization": "Bearer test-key"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "user", "content": f"turn-{turn}-q"},
                        {"role": "assistant", "content": f"turn-{turn}-a"},
                        {"role": "tool", "tool_call_id": "t1", "content": "x" * 600},
                        {"role": "user", "content": f"continue-{turn}"},
                    ],
                },
            )
            assert r.status_code == 200

    method_names = [c[0] for c in calls]
    assert "should_defer_compression" not in method_names, (
        f"OpenAI handler unexpectedly called should_defer_compression. "
        f"Calls observed: {method_names}"
    )
    assert "mark_stable" not in method_names, (
        f"OpenAI handler unexpectedly called mark_stable (the walker side-effect). "
        f"Calls observed: {method_names}"
    )
    # Sanity: the safe positional methods DID fire.
    assert "compute_frozen_count" in method_names
    assert "apply_cached" in method_names


def test_openai_chat_completions_compacts_tools_when_profile_enabled() -> None:
    captured = {}
    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = True
        proxy.config.mode = "token"
        proxy.config.savings_profile = "agent-90"

        def _fake_apply(**kwargs):  # noqa: ANN003
            return SimpleNamespace(
                messages=kwargs["messages"],
                transforms_applied=[],
                timing={},
                tokens_before=10,
                tokens_after=10,
                waste_signals=None,
            )

        proxy.openai_pipeline.apply = _fake_apply

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl_tools",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 500, "completion_tokens": 3, "total_tokens": 503},
                },
            )

        proxy._retry_request = _fake_retry
        verbose_schema_note = "schema annotation repeated for opencode tool definitions " * 50
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "read file helper",
                    "parameters": {
                        "$schema": "https://json-schema.org/draft/2020-12/schema",
                        "title": "ReadFileParameters",
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "title": "Path",
                                "description": verbose_schema_note,
                                "examples": [verbose_schema_note],
                            }
                        },
                        "required": ["path"],
                    },
                },
            }
        ]

        response = client.post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer test-key"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "inspect this file"}],
                "tools": tools,
            },
        )

        assert response.status_code == 200
        assert "openai:chat:tool_schema_compaction" in response.headers["x-headroom-transforms"]
        assert int(response.headers["x-headroom-tokens-saved"]) > 0
        sent_params = captured["body"]["tools"][0]["function"]["parameters"]
        assert "$schema" not in sent_params
        assert "title" not in sent_params
        assert "title" not in sent_params["properties"]["path"]
        assert "examples" not in sent_params["properties"]["path"]


def test_openai_handler_replays_the_provider_confirmed_prefix_even_when_it_inflates() -> None:
    """#3379's twin on the OpenAI path: inside the provider-confirmed prefix the
    replay source is exactly what OpenAI hashed, so a byte-larger replay must
    still go out. Declining it forwards recompressed history and busts the
    prompt cache from the first changed byte (gpt-5: $1.25/M vs $0.125/M
    cached), which is strictly worse than replaying at the cache-read rate."""
    captured = {}
    # Same client original both turns, but last turn's forwarded form is LARGER
    # than what the pipeline produces now -- i.e. the replay inflates and the
    # size bound alone would decline it.
    previous_original = [{"role": "user", "content": "original prefix"}]
    previous_forwarded = [{"role": "user", "content": "previously forwarded " * 20}]
    fake_tracker = _FakePrefixTracker(1, previous_original, previous_forwarded)
    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stable-session"
        )
        proxy.session_tracker_store.resolve_tracker = lambda *args, **kwargs: fake_tracker

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl_confirmed_floor",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23},
                },
            )

        proxy._retry_request = _fake_retry
        response = client.post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer test-key"},
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "user", "content": "original prefix"},
                    {"role": "user", "content": "new suffix"},
                ],
            },
        )

    assert response.status_code == 200
    assert captured["body"]["messages"][0] == previous_forwarded[0]
    assert captured["body"]["messages"][1] == {"role": "user", "content": "new suffix"}


def _tool_history() -> list[dict]:
    """A history whose big tool result the pipeline can compress."""
    return [
        {"role": "user", "content": "run the build and show me the log"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "build log line 0001 module m0 status OK\n" * 200,
        },
    ]


def _compress_first_tool_result(**kwargs):
    """Fake pipeline: compress the big tool result so forwarded ≠ client bytes."""
    mutated = [dict(message) for message in kwargs["messages"]]
    for i, message in enumerate(mutated):
        if message.get("role") == "tool":
            mutated[i] = {**message, "content": "[compressed build log]"}
            break
    return SimpleNamespace(
        messages=mutated,
        transforms_applied=["fake:mutated"],
        timing={},
        tokens_before=2000,
        tokens_after=30,
        waste_signals=None,
    )


def _install_tracker(proxy, tracker) -> None:
    proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
        "stable-session"
    )
    proxy.session_tracker_store.get_or_create = lambda session_id, provider: tracker
    proxy.session_tracker_store.resolve_tracker = lambda *args, **kwargs: tracker


def _chat_usage(prompt_tokens: int, cached_tokens: int) -> dict:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 3,
        "total_tokens": prompt_tokens + 3,
        "prompt_tokens_details": {"cached_tokens": cached_tokens},
    }


def test_openai_streaming_chat_hands_client_originals_to_prefix_tracker() -> None:
    """Direct streaming chat must give the stream finalizer the client's originals.

    Without them the prefix tracker stores the forwarded (compressed) messages
    as "original", so next turn's overlay never matches the client prefix and
    cannot replay the cached bytes.
    """
    captured = {}
    client_messages = _tool_history() + [{"role": "user", "content": "current turn"}]
    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = True
        proxy.config.mode = "cache"
        _install_tracker(proxy, _FakePrefixTracker(frozen_count=0))
        proxy.openai_pipeline.apply = _compress_first_tool_result

        async def _fake_stream_response(*args, **kwargs):  # noqa: ANN002, ANN003
            captured.update(kwargs)
            return httpx.Response(200, text="data: [DONE]\n\n")

        proxy._stream_response = _fake_stream_response
        client.post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer test-key"},
            json={"model": "gpt-4o-mini", "stream": True, "messages": client_messages},
        )

    assert captured.get("original_messages") == client_messages


def test_openai_buffered_chat_hands_client_originals_to_prefix_tracker() -> None:
    """Direct buffered chat must record the client's originals, not its own output.

    Token mode isolates the recording behavior: in cache mode the frozen-prefix
    restore reverts prefix bytes before they are forwarded, which would make the
    forwarded list coincidentally equal the originals. The originals plumbing
    under test here is mode-independent.
    """
    client_messages = _tool_history() + [{"role": "user", "content": "current turn"}]
    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = True
        proxy.config.mode = "token"
        tracker = _FakePrefixTracker(frozen_count=0)
        _install_tracker(proxy, tracker)
        proxy.openai_pipeline.apply = _compress_first_tool_result

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl_originals_buffered",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": _chat_usage(30, 0),
                },
            )

        proxy._retry_request = _fake_retry
        response = client.post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer test-key"},
            json={"model": "gpt-4o-mini", "messages": client_messages},
        )

    assert response.status_code == 200
    assert len(tracker.update_calls) == 1
    call = tracker.update_calls[0]
    # The tracker must store what the client SENT as "original", and the
    # compressed form separately as what was forwarded.
    assert call["original_messages"] == client_messages
    assert call["messages"][2]["content"] == "[compressed build log]"


def _mock_openai_backend_response() -> dict:
    return {
        "id": "chatcmpl-backend-originals",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": _chat_usage(30, 0),
    }


def _sse_chunks(prompt_tokens: int, cached_tokens: int) -> list[str]:
    return [
        'data: {"id":"c1","object":"chat.completion.chunk",'
        '"choices":[{"index":0,"delta":{"role":"assistant","content":"ok"}}]}\n\n',
        'data: {"id":"c1","object":"chat.completion.chunk",'
        '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
        'data: {"id":"c1","object":"chat.completion.chunk","choices":[],'
        f'"usage":{{"prompt_tokens":{prompt_tokens},"completion_tokens":3,'
        f'"total_tokens":{prompt_tokens + 3},'
        f'"prompt_tokens_details":{{"cached_tokens":{cached_tokens}}}}}}}\n\n',
        "data: [DONE]\n\n",
    ]


def _mock_streaming_backend(chunks: list[str]) -> MagicMock:
    async def fake_stream(body, headers):  # noqa: ANN001
        for chunk in chunks:
            yield chunk

    backend = MagicMock()
    backend.name = "anyllm-openai"
    backend.stream_openai_message = fake_stream
    return backend


def test_openai_backend_buffered_chat_hands_client_originals_to_prefix_tracker() -> None:
    """Backend-routed buffered chat must record the client's originals too."""
    client_messages = _tool_history() + [{"role": "user", "content": "current turn"}]
    backend = MagicMock()
    backend.name = "anyllm-openai"
    backend.send_openai_message = AsyncMock(
        return_value=BackendResponse(
            body=_mock_openai_backend_response(),
            status_code=200,
            headers={"content-type": "application/json"},
        )
    )
    with patch("headroom.proxy.server.AnyLLMBackend", return_value=backend):
        with _make_proxy_client(backend="anyllm", anyllm_provider="openai") as client:
            proxy = client.app.state.proxy
            proxy.config.optimize = True
            proxy.config.mode = "token"
            tracker = _FakePrefixTracker(frozen_count=0)
            _install_tracker(proxy, tracker)
            proxy.openai_pipeline.apply = _compress_first_tool_result
            response = client.post(
                "/v1/chat/completions",
                headers={"authorization": "Bearer test-key"},
                json={"model": "gpt-4o-mini", "stream": False, "messages": client_messages},
            )

    assert response.status_code == 200
    assert backend.send_openai_message.await_count == 1
    assert len(tracker.update_calls) == 1
    call = tracker.update_calls[0]
    assert call["original_messages"] == client_messages
    assert call["messages"][2]["content"] == "[compressed build log]"


def test_openai_backend_streaming_chat_hands_client_originals_to_prefix_tracker() -> None:
    """Backend-routed streaming chat must record the client's originals too."""
    client_messages = _tool_history() + [{"role": "user", "content": "current turn"}]
    backend = _mock_streaming_backend(_sse_chunks(prompt_tokens=30, cached_tokens=0))
    with patch("headroom.proxy.server.AnyLLMBackend", return_value=backend):
        with _make_proxy_client(backend="anyllm", anyllm_provider="openai") as client:
            proxy = client.app.state.proxy
            proxy.config.optimize = True
            proxy.config.mode = "token"
            tracker = _FakePrefixTracker(frozen_count=0)
            _install_tracker(proxy, tracker)
            proxy.openai_pipeline.apply = _compress_first_tool_result
            response = client.post(
                "/v1/chat/completions",
                headers={"authorization": "Bearer test-key"},
                json={"model": "gpt-4o-mini", "stream": True, "messages": client_messages},
            )

    assert response.status_code == 200
    assert "[DONE]" in response.text
    assert len(tracker.update_calls) == 1
    call = tracker.update_calls[0]
    assert call["original_messages"] == client_messages
    assert call["messages"][2]["content"] == "[compressed build log]"


def test_openai_chat_real_tracker_stores_client_originals_across_turns() -> None:
    """Real PrefixCacheTracker, two buffered turns (token mode).

    Turn 2's stored "original" history must be the client's own messages
    (including the new turn), not the compressed bytes that were forwarded —
    otherwise next turn's overlay can never match the client prefix and
    cannot replay the previously forwarded bytes.
    """
    from headroom.cache.prefix_tracker import PrefixCacheTracker

    turn1_messages = _tool_history()
    turn2_messages = turn1_messages + [
        {"role": "assistant", "content": "the build passed"},
        {"role": "user", "content": "and now?"},
    ]
    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = True
        proxy.config.mode = "token"
        tracker = PrefixCacheTracker(provider="openai")
        _install_tracker(proxy, tracker)
        proxy.openai_pipeline.apply = _compress_first_tool_result

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl_real_tracker",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": _chat_usage(prompt_tokens=5000, cached_tokens=0),
                },
            )

        proxy._retry_request = _fake_retry
        for messages in (turn1_messages, turn2_messages):
            response = client.post(
                "/v1/chat/completions",
                headers={"authorization": "Bearer test-key"},
                json={"model": "gpt-4o-mini", "messages": messages},
            )
            assert response.status_code == 200

    # The forwarded history is the compressed tool result...
    forwarded = tracker.get_last_forwarded_messages()
    assert forwarded[2]["content"] == "[compressed build log]"
    # ...but the recorded "original" history must be exactly what the client
    # sent on turn 2, byte-for-byte, new turns included.
    assert tracker.get_last_original_messages() == turn2_messages


def test_openai_cache_mode_keeps_replayed_prefix_over_frozen_restore() -> None:
    """Cache mode: the frozen-prefix restore must not undo a replayed prefix.

    The overlay replays last turn's forwarded bytes, which the provider cached.
    Restoring the frozen prefix to the raw client original afterwards
    re-forwards different bytes and busts the cache from message 0.
    """
    captured = {}
    previous_original = [{"role": "user", "content": "original prefix"}]
    previous_forwarded = [{"role": "user", "content": "comp"}]
    fake_tracker = _FakePrefixTracker(1, previous_original, previous_forwarded)
    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = True
        proxy.config.mode = "cache"
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stable-session"
        )
        proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker
        proxy.session_tracker_store.resolve_tracker = lambda *args, **kwargs: fake_tracker

        def _fake_apply(**kwargs):
            return SimpleNamespace(
                messages=kwargs["messages"],
                transforms_applied=[],
                timing={},
                tokens_before=20,
                tokens_after=20,
                waste_signals=None,
            )

        proxy.openai_pipeline.apply = _fake_apply

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl_replay_restore",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23},
                },
            )

        proxy._retry_request = _fake_retry
        response = client.post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer test-key"},
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "user", "content": "original prefix"},
                    {"role": "user", "content": "new suffix"},
                ],
            },
        )

    assert response.status_code == 200
    assert captured["body"]["messages"] == [
        previous_forwarded[0],
        {"role": "user", "content": "new suffix"},
    ]


def _forward_cache_mode_chat(
    tracker: _FakePrefixTracker,
    messages: list[dict],
    *,
    mutate_index: int | None = None,
) -> list[dict]:
    """Send one cache-mode chat turn and return the messages forwarded upstream.

    ``mutate_index`` makes the fake pipeline rewrite that message, standing in
    for a transform that touched a frozen position."""
    captured: dict = {}
    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = True
        proxy.config.mode = "cache"
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stable-session"
        )
        proxy.session_tracker_store.get_or_create = lambda session_id, provider: tracker
        proxy.session_tracker_store.resolve_tracker = lambda *args, **kwargs: tracker

        def _fake_apply(**kwargs):
            out = list(kwargs["messages"])
            if mutate_index is not None:
                out[mutate_index] = {**out[mutate_index], "content": "MUTATED_BY_PIPELINE"}
            return SimpleNamespace(
                messages=out,
                transforms_applied=[],
                timing={},
                tokens_before=20,
                tokens_after=20,
                waste_signals=None,
            )

        proxy.openai_pipeline.apply = _fake_apply

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl_restore_replay",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23},
                },
            )

        proxy._retry_request = _fake_retry
        response = client.post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer test-key"},
            json={"model": "gpt-4o-mini", "messages": messages},
        )
    assert response.status_code == 200
    return captured["body"]["messages"]


def test_openai_cache_mode_replays_only_up_to_the_first_changed_message() -> None:
    """Replay stops where the client's history diverges. Past that point the
    frozen messages are restored to the client's bytes, so a pipeline rewrite
    of a frozen message never reaches the provider, even on a replayed turn."""
    previous_original = [
        {"role": "user", "content": "first original " * 20},
        {"role": "user", "content": "second original " * 20},
    ]
    previous_forwarded = [
        {"role": "user", "content": "first comp"},
        {"role": "user", "content": "second comp"},
    ]
    current = [
        previous_original[0],
        {"role": "user", "content": "second edited by the client"},
        {"role": "user", "content": "new suffix"},
    ]

    forwarded = _forward_cache_mode_chat(
        _FakePrefixTracker(2, previous_original, previous_forwarded),
        current,
        mutate_index=1,
    )

    assert forwarded == [previous_forwarded[0], current[1], current[2]]


def test_openai_cache_mode_keeps_restored_originals_when_history_diverges() -> None:
    """A rewritten history head has no safe replay: the restored originals go out."""
    previous_original = [{"role": "user", "content": "original prefix"}]
    previous_forwarded = [{"role": "user", "content": "comp"}]
    current = [
        {"role": "user", "content": "client rewrote the head"},
        {"role": "user", "content": "new suffix"},
    ]

    forwarded = _forward_cache_mode_chat(
        _FakePrefixTracker(1, previous_original, previous_forwarded),
        current,
        mutate_index=0,
    )

    assert forwarded == current


def test_openai_cache_mode_block_append_replay_survives_the_restore() -> None:
    """Blocks appended inside the last message: forwarded blocks, then new ones."""
    previous_original = [{"role": "user", "content": [{"type": "text", "text": "block one " * 30}]}]
    previous_forwarded = [{"role": "user", "content": [{"type": "text", "text": "[compacted]"}]}]
    current = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "block one " * 30},
                {"type": "text", "text": "block two"},
            ],
        }
    ]

    forwarded = _forward_cache_mode_chat(
        _FakePrefixTracker(1, previous_original, previous_forwarded), current
    )

    assert forwarded == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "[compacted]"},
                {"type": "text", "text": "block two"},
            ],
        }
    ]


def _compress_unfrozen_tool_results(**kwargs):
    """Fake pipeline that, like the real one, only rewrites past the frozen count."""
    frozen = kwargs.get("frozen_message_count") or 0
    out = [
        {**message, "content": f"[compressed {message['tool_call_id']}]"}
        if i >= frozen and message.get("role") == "tool"
        else message
        for i, message in enumerate(kwargs["messages"])
    ]
    return SimpleNamespace(
        messages=out,
        transforms_applied=["fake:compressed"],
        timing={},
        tokens_before=2000,
        tokens_after=30,
        waste_signals=None,
    )


def _provider_reply(request: httpx.Request) -> httpx.Response:
    usage = _chat_usage(prompt_tokens=4000, cached_tokens=2048)
    if json.loads(request.content).get("stream"):
        return httpx.Response(
            200,
            content="".join(_sse_chunks(4000, 2048)).encode(),
            headers={"content-type": "text/event-stream"},
        )
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl_lifecycle",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        },
    )


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "stream"])
def test_openai_cache_mode_forwards_last_turns_bytes_unchanged(stream: bool) -> None:
    """Cache mode, real tracker, three turns: every turn re-forwards the previous
    turn's forwarded messages byte-for-byte, so the provider's prefix cache holds.

    Needs both halves: the tracker must record the client's originals (else the
    overlay cannot match the client prefix), and the frozen-prefix restore must
    not undo the replay (else the compressed tool results revert to raw bytes).
    """
    from headroom.cache.prefix_tracker import PrefixCacheTracker

    def tool_turn(n: int) -> list[dict]:
        call_id = f"call_{n}"
        return [
            {"role": "user", "content": f"turn {n}: run the build"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": f"build {n} log line module m0 status OK\n" * 200,
            },
        ]

    sent: list[list[dict]] = []

    def provider(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content)["messages"])
        return _provider_reply(request)

    history: list[dict] = []
    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = True
        proxy.config.mode = "cache"
        _install_tracker(proxy, PrefixCacheTracker(provider="openai"))
        proxy.openai_pipeline.apply = _compress_unfrozen_tool_results
        proxy.http_client = httpx.AsyncClient(transport=httpx.MockTransport(provider))
        for n in (1, 2, 3):
            history += tool_turn(n)
            response = client.post(
                "/v1/chat/completions",
                headers={"authorization": "Bearer test-key"},
                json={"model": "gpt-4o-mini", "stream": stream, "messages": history},
            )
            assert response.status_code == 200
            response.read()
            history.append({"role": "assistant", "content": "ok"})

    assert len(sent) == 3
    assert sent[0][2]["content"] == "[compressed call_1]"
    assert sent[1][: len(sent[0])] == sent[0]
    assert sent[2][: len(sent[1])] == sent[1]


def test_openai_chat_custom_base_reports_zai_as_outcome_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An OpenAI-compatible ``x-headroom-base-url`` upstream is not ``openai``."""
    # Allowlist the host so the SSRF guard does not need live DNS.
    monkeypatch.setenv("HEADROOM_ALLOWED_BASE_URLS", "api.z.ai")
    captured = {}
    with _make_proxy_client() as client:
        proxy = client.app.state.proxy

        async def _fake_stream_response(*args, **kwargs):  # noqa: ANN002, ANN003
            captured.update(kwargs)
            return httpx.Response(200, text="data: [DONE]\n\n")

        proxy._stream_response = _fake_stream_response
        client.post(
            "/v1/chat/completions",
            headers={
                "authorization": "Bearer test-key",
                "x-headroom-base-url": "https://api.z.ai/api/coding/paas/v4",
            },
            json={
                "model": "glm-5",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert captured.get("outcome_provider") == "zai"


def test_openai_chat_outcome_provider_uses_the_fixed_taxonomy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Buffered chat labels each custom base from the fixed set, never the host."""
    # Keep the SSRF guard out of the way: these hosts are deliberately
    # unresolvable, and a DNS miss makes the handler ignore the override.
    monkeypatch.setattr("headroom.proxy.handlers.openai.is_safe_upstream_url", lambda url: True)
    # Keep the lifetime savings store out of the user's real on-disk state.
    monkeypatch.setenv("HEADROOM_SAVINGS_PATH", str(tmp_path / "savings.json"))
    with _make_proxy_client() as client:
        proxy = client.app.state.proxy

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001, ANN002, ANN003
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl_1",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
                },
            )

        proxy._retry_request = _fake_retry

        def _chat(base: str | None) -> None:
            headers = {"authorization": "Bearer test-key"}
            if base is not None:
                headers["x-headroom-base-url"] = base
            response = client.post(
                "/v1/chat/completions",
                headers=headers,
                json={"model": "glm-5", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert response.status_code == 200

        _chat("https://api.meta.ai/v1")
        _chat("https://api.meta.ai/v1/sub")
        _chat("https://llm.example.internal/v1")
        # No header: the plain OpenAI path keeps its own label (control).
        _chat(None)

        by_provider = proxy.metrics.requests_by_provider
        assert by_provider["meta"] == 2
        assert by_provider["custom"] == 1
        assert by_provider["openai"] == 1


def test_openai_chat_custom_base_flood_cannot_grow_the_provider_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Regression for the #3759 review: a host flood stays inside the fixed set.

    The old code derived the outcome provider from the request-controlled
    ``x-headroom-base-url`` hostname, so one fresh hostname per request grew
    the provider stores and exported series without bound. 980 distinct flood
    hosts plus 20 z.ai requests cover success, upstream failure, local 429 and
    upstream 429, and every sink must count exactly ``zai`` and ``custom``.
    """
    fixed_providers = {"openai", "zen", "zai", "meta", "custom"}
    zai_base = "https://api.z.ai/api/coding/paas/v4"
    limited_auth = "Bearer rate-limited"
    # The flood hosts are unresolvable on purpose; without this stub the SSRF
    # guard would drop the override and relabel the flood as "openai".
    monkeypatch.setattr("headroom.proxy.handlers.openai.is_safe_upstream_url", lambda url: True)
    # Isolated sinks: a fresh lifetime store and workspace, so the exact
    # counts below are this test's traffic and nothing reaches ~/.headroom.
    monkeypatch.setenv("HEADROOM_SAVINGS_PATH", str(tmp_path / "savings.json"))
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "workspace"))
    reader = InMemoryMetricReader()
    set_otel_metrics(HeadroomOtelMetrics(meter_provider=MeterProvider(metric_readers=[reader])))

    class _LimitOneKey:
        """Headroom's own limiter, refusing only ``limited_auth``."""

        async def check_request(self, key: str = "default") -> tuple[bool, float]:
            return key != limited_auth, 1.0

        async def check_tokens(self, key: str, tokens: int) -> tuple[bool, float]:
            return True, 0.0

    async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001, ANN002, ANN003
        if "flood-down-" in url:
            raise httpx.ConnectError("flood host down")
        if "flood-upstream429-" in url:
            return httpx.Response(429, json={"error": {"message": "slow down"}})
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_flood",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            },
        )

    expected_status = {"ok": 200, "down": 502, "limited": 429, "upstream429": 429}
    sent: dict[tuple[str, str], int] = {}
    flood_hosts: set[str] = set()
    try:
        with _make_proxy_client() as client:
            proxy = client.app.state.proxy
            proxy._retry_request = _fake_retry
            proxy.rate_limiter = _LimitOneKey()

            for i in range(1000):
                if i % 50 == 0:
                    provider, kind = "zai", ("ok" if i % 100 == 0 else "limited")
                    base = zai_base
                else:
                    provider, kind = "custom", ("limited", "upstream429", "down", "ok")[i % 4]
                    host = f"flood-{kind}-{i}.example.invalid"
                    flood_hosts.add(host)
                    base = f"https://{host}/v1"
                sent[provider, kind] = sent.get((provider, kind), 0) + 1
                auth = limited_auth if kind == "limited" else "Bearer test-key"
                response = client.post(
                    "/v1/chat/completions",
                    headers={"authorization": auth, "x-headroom-base-url": base},
                    json={"model": "glm-5", "messages": [{"role": "user", "content": f"turn {i}"}]},
                )
                assert response.status_code == expected_status[kind], (i, base)

            # The traffic this test sends, pinned so the claim stays exact.
            assert len(flood_hosts) == 980
            assert sent == {
                ("zai", "ok"): 10,
                ("zai", "limited"): 10,
                ("custom", "ok"): 250,
                ("custom", "down"): 240,
                ("custom", "limited"): 240,
                ("custom", "upstream429"): 250,
            }

            metrics = proxy.metrics
            # In-memory stores: exact counts; "unknown" is the seeded zero bucket.
            assert dict(metrics.requests_by_provider) == {"zai": 10, "custom": 250}
            assert dict(metrics.requests_failed_by_provider) == {"unknown": 0, "custom": 240}
            assert metrics.requests_rate_limited_by_source == {"headroom": 250, "upstream": 250}

            # Lifetime store, isolated under tmp_path.
            lifetime = metrics.savings_tracker.lifetime_response()["requests"]
            assert lifetime["by_provider"] == {"zai": 10, "custom": 250}
            assert lifetime["failed_by_provider"] == {"custom": 240}
            assert lifetime["rate_limited_by_source"] == {"headroom": 250, "upstream": 250}

            # Prometheus export: every provider series is from the fixed set.
            export = client.get("/metrics").text
            assert set(re.findall(r'provider="([^"]+)"', export)) <= fixed_providers | {"unknown"}
            assert 'headroom_requests_by_provider{provider="zai"} 10' in export
            assert 'headroom_requests_by_provider{provider="custom"} 250' in export

        # OTel: every proxy data point carries a fixed provider attribute, and
        # z.ai stays its own series on the success and local-429 paths.
        otel = _collect_metrics(reader)
        otel_points: dict[tuple[str, str, str], float] = {}
        for name, metric in otel.items():
            if not name.startswith("headroom.proxy."):
                continue
            for point in metric.data.data_points:
                label = point.attributes.get("provider")
                assert label in fixed_providers, (name, label)
                if name.startswith("headroom.proxy.requests"):
                    key = (name, label, point.attributes.get("source", ""))
                    otel_points[key] = otel_points.get(key, 0) + point.value
        assert otel_points == {
            ("headroom.proxy.requests", "zai", ""): 10,
            ("headroom.proxy.requests", "custom", ""): 250,
            ("headroom.proxy.requests.failed", "custom", ""): 240,
            ("headroom.proxy.requests.rate_limited", "zai", "headroom"): 10,
            ("headroom.proxy.requests.rate_limited", "custom", "headroom"): 240,
            ("headroom.proxy.requests.rate_limited", "custom", "upstream"): 250,
        }
    finally:
        reset_otel_metrics()
