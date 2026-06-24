"""Regression tests for OpenAI cache-mode stability in proxy mode."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from headroom.proxy.server import ProxyConfig, create_app


class _FakePrefixTracker:
    def __init__(self, frozen_count: int):
        self._frozen_count = frozen_count

    def get_frozen_message_count(self) -> int:
        return self._frozen_count

    def update_from_response(self, **kwargs):  # noqa: ANN003
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
