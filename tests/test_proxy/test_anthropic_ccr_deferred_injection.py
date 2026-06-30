from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from headroom.proxy.server import ProxyConfig, create_app

_RAW_TRANSCRIPT = "\n".join(f"row {idx}: payload payload payload" for idx in range(80))


class _FakePrefixTracker:
    def __init__(self, frozen_count: int):
        self._frozen_count = frozen_count
        self._cached_token_count = 0
        self._last_original_messages: list[dict] = []
        self._last_forwarded_messages: list[dict] = []

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


class _FakeCompressionCache:
    def __init__(
        self,
        frozen_count: int,
        cached_messages: list[dict] | None = None,
    ):
        self._frozen_count = frozen_count
        self._cached_messages = cached_messages

    def apply_cached(self, messages):  # noqa: ANN201
        if self._cached_messages is not None:
            return self._cached_messages
        return messages

    def compute_frozen_count(self, messages) -> int:  # noqa: ARG002
        return self._frozen_count

    def mark_stable_from_messages(self, messages, frozen_count) -> None:  # noqa: ARG002
        return None

    def update_from_result(self, originals, compressed) -> None:  # noqa: ARG002
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


def _force_compression(monkeypatch) -> None:  # noqa: ANN001
    decision = SimpleNamespace(should_compress=True, passthrough_reason=None)
    decision.apply_to_tags = lambda tags: None
    monkeypatch.setattr(
        "headroom.proxy.handlers.anthropic.CompressionDecision.decide",
        lambda **kwargs: decision,
    )


def _disable_pipeline_extensions(proxy) -> None:  # noqa: ANN001
    proxy.pipeline_extensions.emit = lambda *args, **kwargs: SimpleNamespace(
        messages=kwargs.get("messages"),
        tools=kwargs.get("tools"),
        headers=kwargs.get("headers"),
        metadata=kwargs.get("metadata"),
    )


def test_frozen_prefix_skips_marker_emission_when_tool_injection_is_deferred(monkeypatch) -> None:
    captured: dict[str, object] = {}
    original_messages = [{"role": "user", "content": _RAW_TRANSCRIPT}]
    cached_marker_messages = [
        {
            "role": "user",
            "content": "[100 items compressed to 10. Retrieve more: hash=abc123def456abc123def456]",
        }
    ]
    _force_compression(monkeypatch)

    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = True
        proxy.config.image_optimize = False
        proxy.config.ccr_inject_tool = True
        proxy.config.mode = "cache"
        _disable_pipeline_extensions(proxy)

        fake_tracker = _FakePrefixTracker(frozen_count=1)
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stable-session"
        )
        proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker
        proxy._get_compression_cache = lambda session_id: _FakeCompressionCache(
            frozen_count=1,
            cached_messages=cached_marker_messages,
        )

        def _fake_apply(**kwargs):
            captured.setdefault("compression_calls", []).append(kwargs["messages"])
            return SimpleNamespace(
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "[100 items compressed to 10. "
                            "Retrieve more: hash=abc123def456abc123def456]"
                        ),
                    }
                ],
                transforms_applied=["fake:ccr"],
                timing={},
                tokens_before=40,
                tokens_after=10,
                waste_signals=None,
            )

        proxy.anthropic_pipeline.apply = _fake_apply

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "msg_ccr_frozen",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 20,
                        "output_tokens": 3,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "messages": original_messages,
            },
        )

        assert response.status_code == 200
        assert captured.get("compression_calls", []) == []
        forwarded = captured["body"]
        assert forwarded["messages"] == original_messages
        assert "tools" not in forwarded


def test_unfrozen_prefix_keeps_reversible_ccr_path(monkeypatch) -> None:
    captured: dict[str, object] = {}
    marker_message = {
        "role": "user",
        "content": "[100 items compressed to 10. Retrieve more: hash=abc123def456abc123def456]",
    }
    _force_compression(monkeypatch)

    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = True
        proxy.config.image_optimize = False
        proxy.config.ccr_inject_tool = True
        _disable_pipeline_extensions(proxy)

        fake_tracker = _FakePrefixTracker(frozen_count=0)
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stable-session"
        )
        proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker

        def _fake_apply(**kwargs):
            captured.setdefault("compression_calls", []).append(kwargs["messages"])
            return SimpleNamespace(
                messages=[marker_message],
                transforms_applied=["fake:ccr"],
                timing={},
                tokens_before=40,
                tokens_after=10,
                waste_signals=None,
            )

        proxy.anthropic_pipeline.apply = _fake_apply

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "msg_ccr_unfrozen",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 20,
                        "output_tokens": 3,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": _RAW_TRANSCRIPT}],
            },
        )

        assert response.status_code == 200
        assert len(captured.get("compression_calls", [])) == 1
        forwarded = captured["body"]
        assert forwarded["messages"] == [marker_message]
        assert any(tool.get("name") == "headroom_retrieve" for tool in forwarded["tools"])


def test_token_mode_reclamp_keeps_reversible_ccr_path_when_effective_prefix_drops_to_zero(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    marker_message = {
        "role": "user",
        "content": "[100 items compressed to 10. Retrieve more: hash=abc123def456abc123def456]",
    }
    _force_compression(monkeypatch)

    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = True
        proxy.config.image_optimize = False
        proxy.config.ccr_inject_tool = True
        proxy.config.mode = "token"
        _disable_pipeline_extensions(proxy)

        fake_tracker = _FakePrefixTracker(frozen_count=1)
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stable-session"
        )
        proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker
        proxy._get_compression_cache = lambda session_id: _FakeCompressionCache(frozen_count=0)

        def _fake_apply(**kwargs):
            captured.setdefault("compression_calls", []).append(kwargs["messages"])
            captured["frozen_message_count"] = kwargs["frozen_message_count"]
            return SimpleNamespace(
                messages=[marker_message],
                transforms_applied=["fake:ccr"],
                timing={},
                tokens_before=40,
                tokens_after=10,
                waste_signals=None,
            )

        proxy.anthropic_pipeline.apply = _fake_apply

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "msg_ccr_reclamp",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 20,
                        "output_tokens": 3,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": _RAW_TRANSCRIPT}],
            },
        )

        assert response.status_code == 200
        assert captured.get("frozen_message_count") == 0
        assert len(captured.get("compression_calls", [])) == 1
        forwarded = captured["body"]
        assert forwarded["messages"] == [marker_message]
        assert any(tool.get("name") == "headroom_retrieve" for tool in forwarded["tools"])


def test_token_mode_compresses_frozen_prefix_turns_when_tool_is_not_already_present(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    marker_message = {
        "role": "user",
        "content": "[100 items compressed to 10. Retrieve more: hash=abc123def456abc123def456]",
    }
    _force_compression(monkeypatch)

    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = True
        proxy.config.image_optimize = False
        proxy.config.ccr_inject_tool = True
        proxy.config.mode = "token"
        _disable_pipeline_extensions(proxy)

        fake_tracker = _FakePrefixTracker(frozen_count=1)
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stable-session"
        )
        proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker
        proxy._get_compression_cache = lambda session_id: _FakeCompressionCache(frozen_count=1)

        def _fake_apply(**kwargs):
            captured.setdefault("compression_calls", []).append(kwargs["messages"])
            captured["frozen_message_count"] = kwargs["frozen_message_count"]
            return SimpleNamespace(
                messages=[marker_message],
                transforms_applied=["fake:ccr"],
                timing={},
                tokens_before=40,
                tokens_after=10,
                waste_signals=None,
            )

        proxy.anthropic_pipeline.apply = _fake_apply

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "msg_ccr_token_frozen",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 20,
                        "output_tokens": 3,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": _RAW_TRANSCRIPT}],
            },
        )

        assert response.status_code == 200
        assert captured.get("frozen_message_count") == 1
        assert len(captured.get("compression_calls", [])) == 1
        forwarded = captured["body"]
        assert forwarded["messages"] == [marker_message]
        assert any(tool.get("name") == "headroom_retrieve" for tool in forwarded["tools"])


def test_existing_retrieve_tool_keeps_reversible_ccr_path_when_prefix_is_frozen(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    marker_message = {
        "role": "user",
        "content": "[100 items compressed to 10. Retrieve more: hash=abc123def456abc123def456]",
    }
    _force_compression(monkeypatch)

    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = True
        proxy.config.image_optimize = False
        proxy.config.ccr_inject_tool = True
        _disable_pipeline_extensions(proxy)

        fake_tracker = _FakePrefixTracker(frozen_count=1)
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stable-session"
        )
        proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker

        def _fake_apply(**kwargs):
            captured.setdefault("compression_calls", []).append(kwargs["messages"])
            return SimpleNamespace(
                messages=[marker_message],
                transforms_applied=["fake:ccr"],
                timing={},
                tokens_before=40,
                tokens_after=10,
                waste_signals=None,
            )

        proxy.anthropic_pipeline.apply = _fake_apply

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "msg_ccr_frozen_existing_tool",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 20,
                        "output_tokens": 3,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )

        proxy._retry_request = _fake_retry

        existing_tool = {
            "name": "headroom_retrieve",
            "description": "Retrieve compressed content",
            "input_schema": {"type": "object", "properties": {}},
        }
        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "tools": [existing_tool],
                "messages": [{"role": "user", "content": _RAW_TRANSCRIPT}],
            },
        )

        assert response.status_code == 200
        assert len(captured.get("compression_calls", [])) == 1
        forwarded = captured["body"]
        assert forwarded["messages"] == [marker_message]
        assert [tool["name"] for tool in forwarded["tools"]] == ["headroom_retrieve"]


def test_cache_mode_skip_forwards_original_prefix_when_tool_injection_is_deferred(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    original_messages = [
        {"role": "user", "content": "prefix raw content"},
        {"role": "user", "content": _RAW_TRANSCRIPT},
    ]
    previous_forwarded_messages = [
        {
            "role": "user",
            "content": "[100 items compressed to 10. Retrieve more: hash=abc123def456abc123def456]",
        }
    ]
    _force_compression(monkeypatch)

    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = True
        proxy.config.image_optimize = False
        proxy.config.ccr_inject_tool = True
        proxy.config.mode = "cache"
        _disable_pipeline_extensions(proxy)

        fake_tracker = _FakePrefixTracker(frozen_count=1)
        fake_tracker._last_original_messages = [original_messages[0]]
        fake_tracker._last_forwarded_messages = previous_forwarded_messages
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stable-session"
        )
        proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker

        def _fake_apply(**kwargs):
            captured.setdefault("compression_calls", []).append(kwargs["messages"])
            return SimpleNamespace(
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "[100 items compressed to 10. "
                            "Retrieve more: hash=abc123def456abc123def456]"
                        ),
                    }
                ],
                transforms_applied=["fake:ccr"],
                timing={},
                tokens_before=40,
                tokens_after=10,
                waste_signals=None,
            )

        proxy.anthropic_pipeline.apply = _fake_apply

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "msg_ccr_cache_mode_skip",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 20,
                        "output_tokens": 3,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "messages": original_messages,
            },
        )

        assert response.status_code == 200
        assert captured.get("compression_calls", []) == []
        forwarded = captured["body"]
        assert forwarded["messages"] == original_messages
        assert "tools" not in forwarded


def test_cache_mode_exact_prefix_replay_forwards_original_messages_when_tool_injection_is_deferred(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    original_messages = [{"role": "user", "content": _RAW_TRANSCRIPT}]
    previous_forwarded_messages = [
        {
            "role": "user",
            "content": "[100 items compressed to 10. Retrieve more: hash=abc123def456abc123def456]",
        }
    ]
    _force_compression(monkeypatch)

    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = True
        proxy.config.image_optimize = False
        proxy.config.ccr_inject_tool = True
        proxy.config.mode = "cache"
        _disable_pipeline_extensions(proxy)

        fake_tracker = _FakePrefixTracker(frozen_count=1)
        fake_tracker._last_original_messages = original_messages.copy()
        fake_tracker._last_forwarded_messages = previous_forwarded_messages
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stable-session"
        )
        proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker

        def _fake_apply(**kwargs):
            captured.setdefault("compression_calls", []).append(kwargs["messages"])
            return SimpleNamespace(
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "[100 items compressed to 10. "
                            "Retrieve more: hash=abc123def456abc123def456]"
                        ),
                    }
                ],
                transforms_applied=["fake:ccr"],
                timing={},
                tokens_before=40,
                tokens_after=10,
                waste_signals=None,
            )

        proxy.anthropic_pipeline.apply = _fake_apply

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "msg_ccr_cache_mode_exact_prefix",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 20,
                        "output_tokens": 3,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "messages": original_messages,
            },
        )

        assert response.status_code == 200
        assert captured.get("compression_calls", []) == []
        forwarded = captured["body"]
        assert forwarded["messages"] == original_messages
        assert "tools" not in forwarded


def test_token_mode_cached_messages_skip_cache_update_when_pipeline_result_is_unchanged(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    marker_messages = [
        {
            "role": "user",
            "content": "[100 items compressed to 10. Retrieve more: hash=abc123def456abc123def456]",
        }
    ]
    _force_compression(monkeypatch)

    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = True
        proxy.config.image_optimize = False
        proxy.config.ccr_inject_tool = True
        _disable_pipeline_extensions(proxy)

        fake_tracker = _FakePrefixTracker(frozen_count=0)
        cache = _FakeCompressionCache(frozen_count=0, cached_messages=marker_messages)
        cache_updates: list[tuple[list[dict], list[dict]]] = []
        cache.update_from_result = lambda originals, compressed: cache_updates.append(  # type: ignore[method-assign]
            (originals, compressed)
        )
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stable-session"
        )
        proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker
        proxy._get_compression_cache = lambda session_id: cache

        def _fake_apply(**kwargs):
            captured.setdefault("compression_calls", []).append(kwargs["messages"])
            return SimpleNamespace(
                messages=marker_messages,
                transforms_applied=["fake:ccr"],
                timing={},
                tokens_before=40,
                tokens_after=10,
                waste_signals=None,
            )

        proxy.anthropic_pipeline.apply = _fake_apply

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "msg_ccr_token_cache_hit",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 20,
                        "output_tokens": 3,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": _RAW_TRANSCRIPT}],
            },
        )

        assert response.status_code == 200
        assert len(captured.get("compression_calls", [])) == 1
        assert cache_updates == []
        forwarded = captured["body"]
        assert forwarded["messages"] == marker_messages


def test_non_token_non_cache_mode_still_skips_marker_emission_when_tool_is_unavailable(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    original_messages = [{"role": "user", "content": _RAW_TRANSCRIPT}]
    _force_compression(monkeypatch)
    monkeypatch.setattr("headroom.proxy.modes.is_token_mode", lambda mode: False)
    monkeypatch.setattr("headroom.proxy.modes.is_cache_mode", lambda mode: False)

    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = True
        proxy.config.image_optimize = False
        proxy.config.ccr_inject_tool = True
        _disable_pipeline_extensions(proxy)

        fake_tracker = _FakePrefixTracker(frozen_count=1)
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stable-session"
        )
        proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker

        def _fake_apply(**kwargs):
            captured.setdefault("compression_calls", []).append(kwargs["messages"])
            return SimpleNamespace(
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "[100 items compressed to 10. "
                            "Retrieve more: hash=abc123def456abc123def456]"
                        ),
                    }
                ],
                transforms_applied=["fake:ccr"],
                timing={},
                tokens_before=40,
                tokens_after=10,
                waste_signals=None,
            )

        proxy.anthropic_pipeline.apply = _fake_apply

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "msg_ccr_non_token_skip",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 20,
                        "output_tokens": 3,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "messages": original_messages,
            },
        )

        assert response.status_code == 200
        assert captured.get("compression_calls", []) == []
        forwarded = captured["body"]
        assert forwarded["messages"] == original_messages
        assert "tools" not in forwarded


def test_non_token_non_cache_mode_keeps_reversible_path_and_records_waste_signals(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    marker_message = {
        "role": "user",
        "content": "[100 items compressed to 10. Retrieve more: hash=abc123def456abc123def456]",
    }
    existing_tool = {
        "name": "headroom_retrieve",
        "description": "Retrieve compressed content",
        "input_schema": {"type": "object", "properties": {}},
    }
    _force_compression(monkeypatch)
    monkeypatch.setattr("headroom.proxy.modes.is_token_mode", lambda mode: False)
    monkeypatch.setattr("headroom.proxy.modes.is_cache_mode", lambda mode: False)

    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = True
        proxy.config.image_optimize = False
        proxy.config.ccr_inject_tool = True
        _disable_pipeline_extensions(proxy)

        fake_tracker = _FakePrefixTracker(frozen_count=1)
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stable-session"
        )
        proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker

        class _FakeWasteSignals:
            def to_dict(self) -> dict[str, bool]:
                return {"oversized_tool_result": True}

        def _fake_apply(**kwargs):
            captured.setdefault("compression_calls", []).append(kwargs["messages"])
            return SimpleNamespace(
                messages=[marker_message],
                transforms_applied=["fake:ccr"],
                timing={},
                tokens_before=40,
                tokens_after=10,
                waste_signals=_FakeWasteSignals(),
            )

        proxy.anthropic_pipeline.apply = _fake_apply

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "msg_ccr_non_token_reversible",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 20,
                        "output_tokens": 3,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "tools": [existing_tool],
                "messages": [{"role": "user", "content": _RAW_TRANSCRIPT}],
            },
        )

        assert response.status_code == 200
        assert len(captured.get("compression_calls", [])) == 1
        forwarded = captured["body"]
        assert forwarded["messages"] == [marker_message]
        assert [tool["name"] for tool in forwarded["tools"]] == ["headroom_retrieve"]


def test_cache_mode_existing_retrieve_tool_keeps_exact_prefix_replay(monkeypatch) -> None:
    captured: dict[str, object] = {}
    original_messages = [{"role": "user", "content": _RAW_TRANSCRIPT}]
    previous_forwarded_messages = [
        {
            "role": "user",
            "content": "[100 items compressed to 10. Retrieve more: hash=abc123def456abc123def456]",
        }
    ]
    existing_tool = {
        "name": "headroom_retrieve",
        "description": "Retrieve compressed content",
        "input_schema": {"type": "object", "properties": {}},
    }
    _force_compression(monkeypatch)

    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = True
        proxy.config.image_optimize = False
        proxy.config.ccr_inject_tool = True
        proxy.config.mode = "cache"
        _disable_pipeline_extensions(proxy)

        fake_tracker = _FakePrefixTracker(frozen_count=1)
        fake_tracker._last_original_messages = original_messages.copy()
        fake_tracker._last_forwarded_messages = previous_forwarded_messages
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stable-session"
        )
        proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker

        def _fake_apply(**kwargs):
            captured.setdefault("compression_calls", []).append(kwargs["messages"])
            return SimpleNamespace(
                messages=[previous_forwarded_messages[0]],
                transforms_applied=["fake:ccr"],
                timing={},
                tokens_before=40,
                tokens_after=10,
                waste_signals=None,
            )

        proxy.anthropic_pipeline.apply = _fake_apply

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "msg_ccr_cache_mode_exact_prefix_existing_tool",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 20,
                        "output_tokens": 3,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "tools": [existing_tool],
                "messages": original_messages,
            },
        )

        assert response.status_code == 200
        assert captured.get("compression_calls", []) == []
        forwarded = captured["body"]
        assert forwarded["messages"] == previous_forwarded_messages
        assert [tool["name"] for tool in forwarded["tools"]] == ["headroom_retrieve"]


def test_cache_mode_existing_retrieve_tool_compresses_only_the_unfrozen_delta(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    original_messages = [
        {"role": "user", "content": "prefix raw content"},
        {"role": "user", "content": _RAW_TRANSCRIPT},
    ]
    previous_forwarded_messages = [
        {
            "role": "user",
            "content": "[100 items compressed to 10. Retrieve more: hash=prefixprefixprefixprefix]",
        }
    ]
    existing_tool = {
        "name": "headroom_retrieve",
        "description": "Retrieve compressed content",
        "input_schema": {"type": "object", "properties": {}},
    }
    _force_compression(monkeypatch)

    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = True
        proxy.config.image_optimize = False
        proxy.config.ccr_inject_tool = True
        proxy.config.mode = "cache"
        _disable_pipeline_extensions(proxy)

        fake_tracker = _FakePrefixTracker(frozen_count=1)
        fake_tracker._last_original_messages = [original_messages[0]]
        fake_tracker._last_forwarded_messages = previous_forwarded_messages
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stable-session"
        )
        proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker

        def _fake_apply(**kwargs):
            captured.setdefault("compression_calls", []).append(kwargs["messages"])
            return SimpleNamespace(
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "[100 items compressed to 10. "
                            "Retrieve more: hash=deltaabcd1234deltaabcd1234]"
                        ),
                    }
                ],
                transforms_applied=["fake:ccr"],
                timing={},
                tokens_before=40,
                tokens_after=10,
                waste_signals=None,
            )

        proxy.anthropic_pipeline.apply = _fake_apply

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "msg_ccr_cache_mode_delta_existing_tool",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 20,
                        "output_tokens": 3,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "tools": [existing_tool],
                "messages": original_messages,
            },
        )

        assert response.status_code == 200
        assert len(captured.get("compression_calls", [])) == 1
        assert captured["compression_calls"][0] == [original_messages[1]]
        forwarded = captured["body"]
        assert forwarded["messages"] == [
            previous_forwarded_messages[0],
            {
                "role": "user",
                "content": (
                    "[100 items compressed to 10. Retrieve more: hash=deltaabcd1234deltaabcd1234]"
                ),
            },
        ]
        assert [tool["name"] for tool in forwarded["tools"]] == ["headroom_retrieve"]


def test_non_token_non_cache_mode_preserves_original_messages_when_result_is_unchanged(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    original_messages = [{"role": "user", "content": _RAW_TRANSCRIPT}]
    existing_tool = {
        "name": "headroom_retrieve",
        "description": "Retrieve compressed content",
        "input_schema": {"type": "object", "properties": {}},
    }
    _force_compression(monkeypatch)
    monkeypatch.setattr("headroom.proxy.modes.is_token_mode", lambda mode: False)
    monkeypatch.setattr("headroom.proxy.modes.is_cache_mode", lambda mode: False)

    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = True
        proxy.config.image_optimize = False
        proxy.config.ccr_inject_tool = True
        _disable_pipeline_extensions(proxy)

        fake_tracker = _FakePrefixTracker(frozen_count=1)
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stable-session"
        )
        proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker

        def _fake_apply(**kwargs):
            captured.setdefault("compression_calls", []).append(kwargs["messages"])
            return SimpleNamespace(
                messages=original_messages,
                transforms_applied=[],
                timing={},
                tokens_before=40,
                tokens_after=40,
                waste_signals=None,
            )

        proxy.anthropic_pipeline.apply = _fake_apply

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "msg_ccr_non_token_unchanged",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 20,
                        "output_tokens": 3,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "tools": [existing_tool],
                "messages": original_messages,
            },
        )

        assert response.status_code == 200
        assert len(captured.get("compression_calls", [])) == 1
        forwarded = captured["body"]
        assert forwarded["messages"] == original_messages
        assert [tool["name"] for tool in forwarded["tools"]] == ["headroom_retrieve"]


def test_non_token_non_cache_mode_recovers_from_compression_errors(monkeypatch) -> None:
    captured: dict[str, object] = {}
    original_messages = [{"role": "user", "content": _RAW_TRANSCRIPT}]
    existing_tool = {
        "name": "headroom_retrieve",
        "description": "Retrieve compressed content",
        "input_schema": {"type": "object", "properties": {}},
    }
    _force_compression(monkeypatch)
    monkeypatch.setattr("headroom.proxy.modes.is_token_mode", lambda mode: False)
    monkeypatch.setattr("headroom.proxy.modes.is_cache_mode", lambda mode: False)

    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = True
        proxy.config.image_optimize = False
        proxy.config.ccr_inject_tool = True
        _disable_pipeline_extensions(proxy)

        fake_tracker = _FakePrefixTracker(frozen_count=1)
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stable-session"
        )
        proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker
        proxy.anthropic_pipeline.apply = lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic compression failure")
        )

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "msg_ccr_non_token_error_recovery",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 20,
                        "output_tokens": 3,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "tools": [existing_tool],
                "messages": original_messages,
            },
        )

        assert response.status_code == 200
        forwarded = captured["body"]
        assert forwarded["messages"] == original_messages
        assert [tool["name"] for tool in forwarded["tools"]] == ["headroom_retrieve"]


def test_cache_mode_without_stable_delta_keeps_original_messages(monkeypatch) -> None:
    captured: dict[str, object] = {}
    original_messages = [{"role": "user", "content": _RAW_TRANSCRIPT}]
    existing_tool = {
        "name": "headroom_retrieve",
        "description": "Retrieve compressed content",
        "input_schema": {"type": "object", "properties": {}},
    }
    _force_compression(monkeypatch)

    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = True
        proxy.config.image_optimize = False
        proxy.config.ccr_inject_tool = True
        proxy.config.mode = "cache"
        _disable_pipeline_extensions(proxy)

        fake_tracker = _FakePrefixTracker(frozen_count=1)
        fake_tracker._last_original_messages = [{"role": "user", "content": "different prefix"}]
        fake_tracker._last_forwarded_messages = [
            {
                "role": "user",
                "content": "[100 items compressed to 10. Retrieve more: hash=unrelatedhashunrelatedhash]",
            }
        ]
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stable-session"
        )
        proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker

        def _fake_apply(**kwargs):
            captured.setdefault("compression_calls", []).append(kwargs["messages"])
            return SimpleNamespace(
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "[100 items compressed to 10. "
                            "Retrieve more: hash=deltaabcd1234deltaabcd1234]"
                        ),
                    }
                ],
                transforms_applied=["fake:ccr"],
                timing={},
                tokens_before=40,
                tokens_after=10,
                waste_signals=None,
            )

        proxy.anthropic_pipeline.apply = _fake_apply

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "msg_ccr_cache_mode_no_delta",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 20,
                        "output_tokens": 3,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "tools": [existing_tool],
                "messages": original_messages,
            },
        )

        assert response.status_code == 200
        assert captured.get("compression_calls", []) == []
        forwarded = captured["body"]
        assert forwarded["messages"] == original_messages
        assert [tool["name"] for tool in forwarded["tools"]] == ["headroom_retrieve"]
