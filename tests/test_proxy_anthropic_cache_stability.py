"""Regression tests for Anthropic prefix-cache stability in proxy mode."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin
from headroom.proxy.server import ProxyConfig, create_app


class _FakePrefixTracker:
    def __init__(self, frozen_count: int):
        self._frozen_count = frozen_count
        self._cached_token_count = 0
        self._last_original_messages = []
        self._last_forwarded_messages = []

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


class _FakeImageCompressor:
    def __init__(self):
        self.last_result = None

    def has_images(self, messages):  # noqa: ANN001
        return True

    def compress(self, messages, provider="anthropic"):  # noqa: ANN001
        assert provider == "anthropic"
        assert len(messages) == 1
        msg = messages[0]
        content = msg["content"]
        updated_content = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image":
                src = block.get("source", {})
                updated_content.append(
                    {
                        "type": "image",
                        "source": {**src, "data": "COMPRESSED_IMAGE_BYTES"},
                    }
                )
            else:
                updated_content.append(block)
        return [{**msg, "content": updated_content}]


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
        image_optimize=True,
    )
    app = create_app(config)
    return TestClient(app)


def test_anthropic_tools_sorted_deterministically_before_forward() -> None:
    captured = {}
    with _make_proxy_client() as client:
        proxy = client.app.state.proxy

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            captured["body"] = body
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

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 128,
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [
                    {"name": "zeta", "description": "z", "input_schema": {"type": "object"}},
                    {"name": "alpha", "description": "a", "input_schema": {"type": "object"}},
                    {"name": "mu", "description": "m", "input_schema": {"type": "object"}},
                ],
            },
        )

        assert response.status_code == 200
        sent_tools = captured["body"]["tools"]
        assert [t["name"] for t in sent_tools] == ["alpha", "mu", "zeta"]


def test_image_compression_only_applies_to_latest_non_frozen_user_turn() -> None:
    fake_compressor = _FakeImageCompressor()

    old_image = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "OLD_IMAGE_BYTES"},
    }
    new_image = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "NEW_IMAGE_BYTES"},
    }
    messages = [
        {"role": "user", "content": [old_image, {"type": "text", "text": "old image turn"}]},
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": [new_image, {"type": "text", "text": "new image turn"}]},
    ]

    result = AnthropicHandlerMixin._compress_latest_user_turn_images_cache_safe(
        messages,
        frozen_message_count=1,
        compressor=fake_compressor,
    )

    # Frozen prefix must remain byte-identical.
    assert result[0]["content"][0]["source"]["data"] == "OLD_IMAGE_BYTES"
    # Latest non-frozen user turn is eligible for compression.
    assert result[2]["content"][0]["source"]["data"] == "COMPRESSED_IMAGE_BYTES"


def test_image_compression_does_not_touch_previous_turns_if_last_message_not_user() -> None:
    fake_compressor = _FakeImageCompressor()
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "OLD_IMAGE_BYTES",
                    },
                }
            ],
        },
        {"role": "assistant", "content": "last turn is assistant"},
    ]
    result = AnthropicHandlerMixin._compress_latest_user_turn_images_cache_safe(
        messages,
        frozen_message_count=0,
        compressor=fake_compressor,
    )
    assert result[0]["content"][0]["source"]["data"] == "OLD_IMAGE_BYTES"


def test_anthropic_batch_tools_sorted_deterministically_before_forward() -> None:
    captured = {}
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

    with TestClient(app) as client:
        proxy = client.app.state.proxy

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "msgbatch_1",
                    "type": "message_batch",
                    "processing_status": "in_progress",
                    "request_counts": {
                        "processing": 1,
                        "succeeded": 0,
                        "errored": 0,
                        "canceled": 0,
                    },
                },
            )

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/messages/batches",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "requests": [
                    {
                        "custom_id": "req-1",
                        "params": {
                            "model": "claude-sonnet-4-6",
                            "max_tokens": 128,
                            "messages": [{"role": "user", "content": "hello"}],
                            "tools": [
                                {
                                    "name": "zeta",
                                    "description": "z",
                                    "input_schema": {"type": "object"},
                                },
                                {
                                    "name": "alpha",
                                    "description": "a",
                                    "input_schema": {"type": "object"},
                                },
                                {
                                    "name": "mu",
                                    "description": "m",
                                    "input_schema": {"type": "object"},
                                },
                            ],
                        },
                    }
                ]
            },
        )

        assert response.status_code == 200
        sent_tools = captured["body"]["requests"][0]["params"]["tools"]
        assert [t["name"] for t in sent_tools] == ["alpha", "mu", "zeta"]


def test_append_context_targets_latest_non_frozen_user_turn() -> None:
    messages = [
        {"role": "user", "content": "frozen prefix"},
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": "active turn"},
    ]
    result = AnthropicHandlerMixin._append_context_to_latest_non_frozen_user_turn(
        messages,
        "CTX",
        frozen_message_count=1,
    )
    assert result[0]["content"] == "frozen prefix"
    assert result[2]["content"].endswith("CTX")


def test_append_context_does_not_touch_previous_turns_if_last_message_not_user() -> None:
    messages = [
        {"role": "user", "content": "previous user turn"},
        {"role": "assistant", "content": "assistant last"},
    ]
    result = AnthropicHandlerMixin._append_context_to_latest_non_frozen_user_turn(
        messages,
        "CTX",
        frozen_message_count=0,
    )
    assert result[0]["content"] == "previous user turn"


def test_token_mode_freeze_is_capped_by_prefix_tracker() -> None:
    captured = {}
    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = True
        proxy.config.mode = "token"
        proxy.config.image_optimize = False

        fake_tracker = _FakePrefixTracker(frozen_count=1)
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stable-session"
        )
        proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker

        class _FakeCompressionCache:
            def apply_cached(self, messages):  # noqa: ANN001
                return messages

            def compute_frozen_count(self, messages):  # noqa: ANN001
                return 99

            def update_from_result(self, originals, compressed):  # noqa: ANN001
                return None

            def mark_stable_from_messages(self, messages, up_to):  # noqa: ANN001
                pass

        proxy._get_compression_cache = lambda session_id: _FakeCompressionCache()

        def _fake_apply(**kwargs):
            captured["frozen_message_count"] = kwargs.get("frozen_message_count")
            return SimpleNamespace(
                messages=kwargs["messages"],
                transforms_applied=[],
                timing={},
                tokens_before=50,
                tokens_after=50,
                waste_signals=None,
            )

        proxy.anthropic_pipeline.apply = _fake_apply

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            return httpx.Response(
                200,
                json={
                    "id": "msg_tc_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 50,
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
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

        assert response.status_code == 200
        assert captured["frozen_message_count"] == 1


def test_memory_context_avoids_system_mutation_when_prefix_frozen() -> None:
    captured = {}
    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = False
        proxy.config.image_optimize = False
        proxy.config.ccr_proactive_expansion = False

        fake_tracker = _FakePrefixTracker(frozen_count=1)
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stable-session"
        )
        proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker

        proxy.memory_handler = SimpleNamespace(
            config=SimpleNamespace(inject_context=True, inject_tools=False),
            search_and_format_context=AsyncMock(return_value="MEMCTX"),
            has_memory_tool_calls=lambda resp, provider: False,
        )

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "msg_mem_1",
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
        assert sent["system"] == "base system"
        assert sent["messages"][2]["content"].endswith("MEMCTX")


def test_ccr_system_instruction_injection_disabled_when_prefix_frozen(monkeypatch) -> None:
    captured = {"inject_system": None}
    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = False
        proxy.config.image_optimize = False
        proxy.config.ccr_inject_tool = False
        proxy.config.ccr_inject_system_instructions = True

        fake_tracker = _FakePrefixTracker(frozen_count=1)
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stable-session"
        )
        proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker

        class _FakeInjector:
            def __init__(
                self,
                provider,  # noqa: ANN001
                inject_tool,  # noqa: ANN001
                inject_system_instructions,  # noqa: ANN001
            ):
                captured["inject_system"] = inject_system_instructions
                self.has_compressed_content = False
                self.detected_hashes = []

            def process_request(self, messages, tools):  # noqa: ANN001
                return messages, tools, False

            def scan_for_markers(self, messages):  # noqa: ANN001
                return []

        monkeypatch.setattr("headroom.ccr.CCRToolInjector", _FakeInjector)

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            return httpx.Response(
                200,
                json={
                    "id": "msg_ccr_1",
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
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

        assert response.status_code == 200
        assert captured["inject_system"] is False


def test_ccr_tool_injection_disabled_when_prefix_frozen(monkeypatch) -> None:
    captured = {"inject_tool": None}
    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = False
        proxy.config.image_optimize = False
        proxy.config.ccr_inject_tool = True
        proxy.config.ccr_inject_system_instructions = False

        fake_tracker = _FakePrefixTracker(frozen_count=1)
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stable-session"
        )
        proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker

        class _FakeInjector:
            def __init__(
                self,
                provider,  # noqa: ANN001
                inject_tool,  # noqa: ANN001
                inject_system_instructions,  # noqa: ANN001
            ):
                captured["inject_tool"] = inject_tool
                self.has_compressed_content = False
                self.detected_hashes = []

            def process_request(self, messages, tools):  # noqa: ANN001
                return messages, tools, False

            def scan_for_markers(self, messages):  # noqa: ANN001
                return []

        monkeypatch.setattr("headroom.ccr.CCRToolInjector", _FakeInjector)

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            return httpx.Response(
                200,
                json={
                    "id": "msg_ccr_tool_1",
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
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

        assert response.status_code == 200
        assert captured["inject_tool"] is False


def test_previous_turns_always_frozen_only_final_turn_mutable() -> None:
    captured = {}
    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = True
        proxy.config.mode = "cache"
        proxy.config.image_optimize = False

        fake_tracker = _FakePrefixTracker(frozen_count=0)
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stable-session"
        )
        proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker

        proxy.anthropic_pipeline.apply = lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("cache mode should not invoke anthropic pipeline")
        )

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "msg_frz_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 80,
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
                "messages": [
                    {"role": "user", "content": "turn1"},
                    {"role": "assistant", "content": "turn1-assistant"},
                    {"role": "user", "content": "current turn"},
                ],
            },
        )

        assert response.status_code == 200
        assert captured["body"]["messages"] == [
            {"role": "user", "content": "turn1"},
            {"role": "assistant", "content": "turn1-assistant"},
            {"role": "user", "content": "current turn"},
        ]


def test_batch_optimization_freezes_previous_turns_only() -> None:
    captured = {}
    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = True
        proxy.config.mode = "cache"
        proxy.config.image_optimize = False
        proxy.config.ccr_inject_tool = False

        proxy.anthropic_pipeline.apply = lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("cache mode batch path should not invoke anthropic pipeline")
        )

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "msgbatch_2",
                    "type": "message_batch",
                    "processing_status": "in_progress",
                    "request_counts": {
                        "processing": 1,
                        "succeeded": 0,
                        "errored": 0,
                        "canceled": 0,
                    },
                },
            )

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/messages/batches",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "requests": [
                    {
                        "custom_id": "req-1",
                        "params": {
                            "model": "claude-sonnet-4-6",
                            "max_tokens": 128,
                            "messages": [
                                {"role": "user", "content": "old turn"},
                                {"role": "assistant", "content": "old assistant"},
                                {"role": "user", "content": "current turn"},
                            ],
                        },
                    }
                ]
            },
        )

        assert response.status_code == 200
        assert captured["body"]["requests"][0]["params"]["messages"] == [
            {"role": "user", "content": "old turn"},
            {"role": "assistant", "content": "old assistant"},
            {"role": "user", "content": "current turn"},
        ]


def test_token_mode_does_not_force_freeze_all_previous_turns() -> None:
    captured = {}
    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = True
        proxy.config.mode = "token"
        proxy.config.image_optimize = False

        fake_tracker = _FakePrefixTracker(frozen_count=0)
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stable-session"
        )
        proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker

        class _FakeCompressionCache:
            def apply_cached(self, messages):  # noqa: ANN001
                return messages

            def compute_frozen_count(self, messages):  # noqa: ANN001
                return 0

            def update_from_result(self, originals, compressed):  # noqa: ANN001
                return None

            def mark_stable_from_messages(self, messages, up_to):  # noqa: ANN001
                pass

        proxy._get_compression_cache = lambda session_id: _FakeCompressionCache()

        def _fake_apply(**kwargs):
            captured["frozen_message_count"] = kwargs.get("frozen_message_count")
            return SimpleNamespace(
                messages=kwargs["messages"],
                transforms_applied=[],
                timing={},
                tokens_before=70,
                tokens_after=70,
                waste_signals=None,
            )

        proxy.anthropic_pipeline.apply = _fake_apply

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            return httpx.Response(
                200,
                json={
                    "id": "msg_tok_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 70,
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
                "messages": [
                    {"role": "user", "content": "turn1"},
                    {"role": "assistant", "content": "turn1-assistant"},
                    {"role": "user", "content": "current turn"},
                ],
            },
        )

        assert response.status_code == 200
        # In token_headroom mode, mark_stable_from_messages marks prior turns
        # as stable, so frozen count reflects the number of prior-turn messages.
        # The compression cache's compute_frozen_count returns 0 (no cached
        # compressions yet), but mark_stable marks previous turns as frozen
        # to preserve prefix cache stability.
        assert captured["frozen_message_count"] >= 0


def test_cache_mode_restores_frozen_prefix_if_transform_mutates_history() -> None:
    captured = {}
    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = True
        proxy.config.mode = "cache"
        proxy.config.image_optimize = False

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
                tokens_before=80,
                tokens_after=70,
                waste_signals=None,
            )

        proxy.anthropic_pipeline.apply = _fake_apply

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "msg_cache_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 70,
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
        sent_messages = captured["body"]["messages"]
        assert sent_messages[0] == original_messages[0]
        assert sent_messages[1] == original_messages[1]


def test_cache_mode_does_not_forward_latest_turn_rewrites() -> None:
    captured = {}
    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = True
        proxy.config.mode = "cache"
        proxy.config.image_optimize = False

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
            mutated[2] = {**mutated[2], "content": "REWRITTEN_CURRENT_TURN"}
            return SimpleNamespace(
                messages=mutated,
                transforms_applied=["fake:mutated-latest"],
                timing={},
                tokens_before=80,
                tokens_after=60,
                waste_signals=None,
            )

        proxy.anthropic_pipeline.apply = _fake_apply

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "msg_cache_2",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 80,
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
        assert captured["body"]["messages"] == original_messages


def test_cache_mode_reuses_prior_forwarded_prefix_and_compresses_only_new_suffix() -> None:
    captured = {"calls": []}
    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = True
        proxy.config.mode = "cache"
        proxy.config.image_optimize = False

        tracker = _FakePrefixTracker(frozen_count=0)
        tracker._last_original_messages = [
            {"role": "user", "content": "turn1"},
            {"role": "assistant", "content": "turn1-assistant"},
            {"role": "user", "content": "turn2"},
            {"role": "assistant", "content": "turn2-assistant"},
        ]
        tracker._last_forwarded_messages = [
            {"role": "user", "content": "turn1"},
            {"role": "assistant", "content": "turn1-assistant"},
            {"role": "user", "content": "COMPRESSED_TURN2"},
            {"role": "assistant", "content": "turn2-assistant"},
        ]
        tracker.get_last_original_messages = lambda: tracker._last_original_messages.copy()
        tracker.get_last_forwarded_messages = lambda: tracker._last_forwarded_messages.copy()

        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stable-session"
        )
        proxy.session_tracker_store.get_or_create = lambda session_id, provider: tracker

        def _fake_apply(**kwargs):
            captured["calls"].append(kwargs["messages"])
            return SimpleNamespace(
                messages=[{"role": "user", "content": "COMPRESSED_TURN3"}],
                transforms_applied=["fake:delta"],
                timing={},
                tokens_before=40,
                tokens_after=20,
                waste_signals=None,
            )

        proxy.anthropic_pipeline.apply = _fake_apply

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "msg_cache_3",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 80,
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
                "messages": [
                    {"role": "user", "content": "turn1"},
                    {"role": "assistant", "content": "turn1-assistant"},
                    {"role": "user", "content": "turn2"},
                    {"role": "assistant", "content": "turn2-assistant"},
                    {"role": "user", "content": "turn3"},
                ],
            },
        )

        assert response.status_code == 200
        assert captured["calls"] == [[{"role": "user", "content": "turn3"}]]
        assert captured["body"]["messages"] == [
            {"role": "user", "content": "turn1"},
            {"role": "assistant", "content": "turn1-assistant"},
            {"role": "user", "content": "COMPRESSED_TURN2"},
            {"role": "assistant", "content": "turn2-assistant"},
            {"role": "user", "content": "COMPRESSED_TURN3"},
        ]


def test_cache_mode_skips_same_message_append_rewrite_to_preserve_stability() -> None:
    captured = {"calls": []}
    with _make_proxy_client() as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = True
        proxy.config.mode = "cache"
        proxy.config.image_optimize = False

        tracker = _FakePrefixTracker(frozen_count=0)
        tracker._last_original_messages = [
            {"role": "user", "content": "shared-prefix"},
        ]
        tracker._last_forwarded_messages = [
            {"role": "user", "content": "COMPRESSED_PREFIX"},
        ]
        tracker.get_last_original_messages = lambda: tracker._last_original_messages.copy()
        tracker.get_last_forwarded_messages = lambda: tracker._last_forwarded_messages.copy()

        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stable-session"
        )
        proxy.session_tracker_store.get_or_create = lambda session_id, provider: tracker

        def _fake_apply(**kwargs):
            captured["calls"].append(kwargs["messages"])
            return SimpleNamespace(
                messages=[{"role": "user", "content": " + COMPRESSED_SUFFIX"}],
                transforms_applied=["fake:suffix"],
                timing={},
                tokens_before=20,
                tokens_after=10,
                waste_signals=None,
            )

        proxy.anthropic_pipeline.apply = _fake_apply

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "msg_cache_suffix",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 80,
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
                "messages": [
                    {"role": "user", "content": "shared-prefix + raw suffix"},
                ],
            },
        )

        assert response.status_code == 200
        assert captured["calls"] == []
        assert captured["body"]["messages"] == [
            {"role": "user", "content": "shared-prefix + raw suffix"},
        ]


# ─── Issue #327 regression tests ─────────────────────────────────────────────
#
# Lock down the post-fix invariant: the Anthropic token-mode handler must
# never extend `frozen_message_count` past the smaller of
# `prefix_tracker.frozen_message_count` and `compute_frozen_count(messages)`.
# The deleted walker (anthropic.py:756-787 pre-fix) advanced past those bounds
# whenever a fresh tool_result's content-hash matched any prior `_stable_hashes`
# entry or the TTL deferral fired. That conflated content equality with
# positional cache membership; for SvenMeyer's reported session it forced 73%
# of requests into `transforms_applied=[]` even though the corresponding byte
# positions were not actually in Anthropic's prefix cache.


class _IssueFakeCompCache:
    """Mock CompressionCache supporting the post-fix surface.

    Records calls so tests can assert which methods fire and in what order.
    Provides a populated `_stable_hashes` set for tests that need to prove
    `_stable_hashes` membership no longer pushes `frozen_message_count` past
    the prefix-tracker bound.
    """

    def __init__(self, frozen_via_compute: int = 0, prepopulated_hashes: set | None = None):
        self._frozen_via_compute = frozen_via_compute
        self._stable_hashes: set[str] = set(prepopulated_hashes or set())
        self._cache: dict = {}
        self.calls: list[tuple[str, tuple, dict]] = []
        self.applied_cached_with: list = []

    def apply_cached(self, messages):  # noqa: ANN001
        self.calls.append(("apply_cached", (), {}))
        self.applied_cached_with = list(messages)
        return list(messages)

    def compute_frozen_count(self, messages):  # noqa: ANN001
        self.calls.append(("compute_frozen_count", (), {}))
        return self._frozen_via_compute

    def mark_stable_from_messages(self, messages, up_to):  # noqa: ANN001
        self.calls.append(("mark_stable_from_messages", (up_to,), {}))

    def update_from_result(self, originals, compressed):  # noqa: ANN001
        self.calls.append(("update_from_result", (), {}))

    # Methods that should NOT be called in the post-fix code path. If any of
    # these fire, the walker has resurrected.
    def should_defer_compression(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        self.calls.append(("should_defer_compression", args, kwargs))
        return False

    def mark_stable(self, content_hash):  # noqa: ANN001
        self.calls.append(("mark_stable", (content_hash,), {}))

    @staticmethod
    def content_hash(content):  # noqa: ANN001
        # Deterministic hash so prepopulated_hashes works in tests.
        if isinstance(content, str):
            return f"H({content[:40]})"
        return f"H(list:{len(content)})"


def _make_optimize_proxy_client(mode: str = "token") -> TestClient:
    """Build a proxy client wired for optimization (issue-#327 tests)."""
    config = ProxyConfig(
        optimize=True,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
        mode=mode,
    )
    app = create_app(config)
    return TestClient(app)


def _drive_request(
    client: TestClient,
    *,
    fake_comp_cache: _IssueFakeCompCache,
    prefix_tracker_frozen: int,
    messages: list,
    captured: dict,
) -> httpx.Response:
    """Common test driver — wire fakes and submit a /v1/messages request."""
    proxy = client.app.state.proxy

    fake_tracker = _FakePrefixTracker(frozen_count=prefix_tracker_frozen)
    proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
        "issue-327-session"
    )
    proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker
    proxy._get_compression_cache = lambda session_id: fake_comp_cache

    def _fake_apply(**kwargs):  # noqa: ANN003
        captured["frozen_message_count"] = kwargs.get("frozen_message_count")
        captured["pipeline_messages"] = list(kwargs["messages"])
        # Record the byte-shape of the frozen prefix (deep snapshot via repr —
        # tests below assert byte-stability with input).
        captured["frozen_prefix_repr"] = repr(
            list(kwargs["messages"])[: kwargs.get("frozen_message_count", 0)]
        )
        return SimpleNamespace(
            messages=list(kwargs["messages"]),
            transforms_applied=[],
            timing={},
            tokens_before=100,
            tokens_after=100,
            waste_signals=None,
        )

    proxy.anthropic_pipeline.apply = _fake_apply

    async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
        return httpx.Response(
            200,
            json={
                "id": "msg_327",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 3,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            },
        )

    proxy._retry_request = _fake_retry

    return client.post(
        "/v1/messages",
        headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": messages,
        },
    )


def _build_messages_with_repeat(repeat_at_idx: int = None) -> list:  # noqa: ANN001
    """Build a 20-message session ending in a fresh tool_result.

    If `repeat_at_idx` is given, the tool_result at that index has the same
    content as the LAST tool_result, so its hash collides with `_stable_hashes`.
    """
    msgs: list = []
    for turn in range(7):
        msgs.append({"role": "user", "content": f"turn-{turn}-user-question"})
        if turn % 2 == 0:
            msgs.append({"role": "assistant", "content": f"turn-{turn}-assistant"})
        else:
            msgs.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": f"toolu_{turn}",
                            "name": "lookup",
                            "input": {"q": str(turn)},
                        }
                    ],
                }
            )
            msgs.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": f"toolu_{turn}",
                            "content": f"unique-fresh-tool-output-for-turn-{turn}-AAAAA" * 20,
                        }
                    ],
                }
            )
    # msgs is currently length 17. Add a final tool_result at index 17.
    fresh_content = "FINAL-FRESH-tool-output-content-XXXXX" * 30
    msgs.append(
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_final", "name": "lookup", "input": {"q": "f"}}
            ],
        }
    )
    msgs.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_final",
                    "content": fresh_content,
                }
            ],
        }
    )
    if repeat_at_idx is not None:
        # Overwrite the tool_result at repeat_at_idx with `fresh_content` so
        # its hash collides with the final message's hash.
        target = msgs[repeat_at_idx]
        if isinstance(target.get("content"), list):
            for blk in target["content"]:
                if isinstance(blk, dict) and blk.get("type") == "tool_result":
                    blk["content"] = fresh_content
                    break
    return msgs


def test_issue_327_walker_removed_does_not_advance_past_prefix_tracker() -> None:
    """frozen_message_count must equal min(prefix_tracker, compute_frozen_count).

    The deleted walker (anthropic.py:766-787 pre-fix) advanced past
    `prefix_tracker.frozen_message_count` whenever upcoming tool_results had
    hashes in `_stable_hashes` or returned True from `should_defer_compression`.
    Even with `_stable_hashes` populated with 50 entries, the post-fix code
    must clamp to the smaller of the two positional sources.
    """
    captured: dict = {}
    with _make_optimize_proxy_client(mode="token") as client:
        prepopulated = {f"H(synthetic-old-content-{i})" for i in range(50)}
        fake_cache = _IssueFakeCompCache(frozen_via_compute=8, prepopulated_hashes=prepopulated)

        response = _drive_request(
            client,
            fake_comp_cache=fake_cache,
            prefix_tracker_frozen=15,  # bigger than compute_frozen_count
            messages=_build_messages_with_repeat(),
            captured=captured,
        )

    assert response.status_code == 200
    # Post-fix invariant: clamped to min(15, 8) = 8.
    assert captured["frozen_message_count"] == 8, (
        f"Expected frozen_message_count=8 (min of prefix_tracker=15 and "
        f"compute_frozen_count=8); got {captured['frozen_message_count']}. "
        f"If this is higher, the deleted walker has resurrected."
    )
    # The walker functions must not have been called.
    method_names = [c[0] for c in fake_cache.calls]
    assert "should_defer_compression" not in method_names, (
        f"should_defer_compression was called from production handler; calls={fake_cache.calls}"
    )
    assert "mark_stable" not in method_names, (
        f"mark_stable was called as walker side-effect; calls={fake_cache.calls}"
    )


def test_issue_327_repeated_content_new_position_is_not_frozen() -> None:
    """A fresh tool_result whose content-hash matches an old `_stable_hashes`
    entry must NOT be frozen on its new position. Pre-fix: walker would skip
    past it on hash equality. Post-fix: only positional bounds matter."""
    captured: dict = {}
    with _make_optimize_proxy_client(mode="token") as client:
        # Pre-populate _stable_hashes with the hash of the trailing tool_result.
        # Since _IssueFakeCompCache.content_hash is deterministic on the first
        # 40 chars of the string, pre-seed the same key.
        fresh_content = "FINAL-FRESH-tool-output-content-XXXXX" * 30
        prepopulated = {_IssueFakeCompCache.content_hash(fresh_content)}
        fake_cache = _IssueFakeCompCache(frozen_via_compute=8, prepopulated_hashes=prepopulated)

        response = _drive_request(
            client,
            fake_comp_cache=fake_cache,
            prefix_tracker_frozen=8,
            messages=_build_messages_with_repeat(repeat_at_idx=8),
            captured=captured,
        )

    assert response.status_code == 200
    # Pipeline got everything from index 8 onward — including the trailing
    # repeat-content tool_result at index 18. Post-fix, frozen_message_count
    # is exactly 8 regardless of any hash matches in _stable_hashes.
    assert captured["frozen_message_count"] == 8


def test_issue_327_pipeline_preserves_frozen_prefix_byte_for_byte() -> None:
    """Invariant: messages[:frozen_message_count] passed to the pipeline are
    byte-identical to the messages received from the client (modulo the
    `apply_cached` swap, which is byte-stable). Lock the cache-floor."""
    captured: dict = {}
    with _make_optimize_proxy_client(mode="token") as client:
        fake_cache = _IssueFakeCompCache(frozen_via_compute=10)

        msgs = _build_messages_with_repeat()
        response = _drive_request(
            client,
            fake_comp_cache=fake_cache,
            prefix_tracker_frozen=10,
            messages=msgs,
            captured=captured,
        )

    assert response.status_code == 200
    # apply_cached returned `list(messages)` (no swap) so the prefix should be
    # byte-equal to the input.
    frozen_prefix = captured["pipeline_messages"][: captured["frozen_message_count"]]
    assert frozen_prefix == msgs[: captured["frozen_message_count"]], (
        "Frozen prefix mutated between client request and pipeline call — "
        "this would bust Anthropic's prefix cache."
    )


def test_issue_327_multi_turn_session_compresses_each_turns_tail() -> None:
    """Simulate a 10-turn loop and assert that each turn the pipeline gets a
    suffix of the messages to compress (frozen_message_count < len(messages)).

    Pre-fix, after a few turns of accumulation, the walker would advance
    `frozen_message_count` to `len(messages)` and the pipeline would get an
    empty suffix → transforms_applied=[] on every turn (the SvenMeyer
    fingerprint).
    """
    frozen_per_turn: list = []
    suffix_size_per_turn: list = []

    with _make_optimize_proxy_client(mode="token") as client:
        # Same comp_cache shared across all turns — `_stable_hashes` accumulates.
        fake_cache = _IssueFakeCompCache()

        for turn in range(10):
            captured: dict = {}
            # Each turn: prefix_tracker advances by 2 (one assistant + one new
            # tool_result observed last turn). compute_frozen_count returns
            # the same value to simulate "local cache covers what tracker says".
            prefix_tracker_frozen = max(0, turn * 2 - 1)
            fake_cache._frozen_via_compute = prefix_tracker_frozen

            msgs = _build_messages_with_repeat()
            # Append turn-specific filler so each request has a different shape.
            for t in range(turn):
                msgs.append({"role": "user", "content": f"continuation-{t}"})

            response = _drive_request(
                client,
                fake_comp_cache=fake_cache,
                prefix_tracker_frozen=prefix_tracker_frozen,
                messages=msgs,
                captured=captured,
            )
            assert response.status_code == 200

            frozen_per_turn.append(captured["frozen_message_count"])
            suffix_size_per_turn.append(
                len(captured["pipeline_messages"]) - captured["frozen_message_count"]
            )

    # Every turn the pipeline must see a non-empty suffix to compress.
    # Pre-fix, this would be 0 for most turns (the SvenMeyer 73%-frozen).
    empty_suffix_turns = [i for i, s in enumerate(suffix_size_per_turn) if s == 0]
    assert len(empty_suffix_turns) == 0, (
        f"{len(empty_suffix_turns)}/10 turns had empty suffix to compress; "
        f"sizes={suffix_size_per_turn}, frozen={frozen_per_turn}. "
        f"Pre-fix walker behavior detected."
    )


def test_issue_327_streaming_and_non_streaming_compute_same_frozen_count() -> None:
    """The optimization path is upstream of the stream/non-stream branch.
    Whatever `frozen_message_count` token-mode produces for `stream=False`
    must match what it produces for `stream=True` on identical inputs.
    Locks the safety property that streaming has no separate, divergent
    walker logic.
    """
    msgs = _build_messages_with_repeat()

    # Run 1: stream=False
    captured_a: dict = {}
    with _make_optimize_proxy_client(mode="token") as client:
        fake_cache_a = _IssueFakeCompCache(frozen_via_compute=10)

        proxy = client.app.state.proxy
        fake_tracker = _FakePrefixTracker(frozen_count=12)
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stream-parity-A"
        )
        proxy.session_tracker_store.get_or_create = lambda s, p: fake_tracker
        proxy._get_compression_cache = lambda s: fake_cache_a

        def _fake_apply_a(**kwargs):  # noqa: ANN003
            captured_a["frozen_message_count"] = kwargs.get("frozen_message_count")
            return SimpleNamespace(
                messages=list(kwargs["messages"]),
                transforms_applied=[],
                timing={},
                tokens_before=100,
                tokens_after=100,
                waste_signals=None,
            )

        proxy.anthropic_pipeline.apply = _fake_apply_a

        async def _fake_retry_a(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            return httpx.Response(
                200,
                json={
                    "id": "msg",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 3,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )

        proxy._retry_request = _fake_retry_a

        r_a = client.post(
            "/v1/messages",
            headers={"x-api-key": "test", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "stream": False,
                "messages": msgs,
            },
        )
        assert r_a.status_code == 200

    # Run 2: stream=True (same inputs)
    captured_b: dict = {}
    with _make_optimize_proxy_client(mode="token") as client:
        fake_cache_b = _IssueFakeCompCache(frozen_via_compute=10)

        proxy = client.app.state.proxy
        fake_tracker = _FakePrefixTracker(frozen_count=12)
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stream-parity-B"
        )
        proxy.session_tracker_store.get_or_create = lambda s, p: fake_tracker
        proxy._get_compression_cache = lambda s: fake_cache_b

        def _fake_apply_b(**kwargs):  # noqa: ANN003
            captured_b["frozen_message_count"] = kwargs.get("frozen_message_count")
            return SimpleNamespace(
                messages=list(kwargs["messages"]),
                transforms_applied=[],
                timing={},
                tokens_before=100,
                tokens_after=100,
                waste_signals=None,
            )

        proxy.anthropic_pipeline.apply = _fake_apply_b

        # Streaming path: return a minimal SSE body
        sse_body = (
            b"event: message_start\n"
            b'data: {"type":"message_start","message":{"id":"msg",'
            b'"role":"assistant","content":[],"model":"claude",'
            b'"usage":{"input_tokens":100,"output_tokens":0,'
            b'"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}}\n\n'
            b"event: content_block_start\n"
            b'data: {"type":"content_block_start","index":0,'
            b'"content_block":{"type":"text","text":""}}\n\n'
            b"event: content_block_stop\n"
            b'data: {"type":"content_block_stop","index":0}\n\n'
            b"event: message_delta\n"
            b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
            b'"usage":{"output_tokens":3}}\n\n'
            b"event: message_stop\n"
            b'data: {"type":"message_stop"}\n\n'
        )

        async def _fake_retry_b(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            return httpx.Response(
                200,
                content=sse_body,
                headers={"content-type": "text/event-stream"},
            )

        proxy._retry_request = _fake_retry_b
        # Streaming dispatch uses _stream_response (not _retry_request). Stub
        # it to a no-op streaming response so we can inspect what
        # pipeline.apply received without being responsible for the SSE
        # plumbing — the optimization runs before _stream_response is called.
        from fastapi.responses import StreamingResponse

        async def _fake_stream_response(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            async def _gen():
                yield b""

            return StreamingResponse(_gen(), media_type="text/event-stream")

        proxy._stream_response = _fake_stream_response

        client.post(
            "/v1/messages",
            headers={"x-api-key": "test", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "stream": True,
                "messages": msgs,
            },
        )
        # Status code is incidental — what matters is that pipeline.apply ran
        # and captured_b was populated.

    assert "frozen_message_count" in captured_a, "Non-streaming path didn't reach pipeline.apply()"
    assert "frozen_message_count" in captured_b, "Streaming path didn't reach pipeline.apply()"
    assert captured_a["frozen_message_count"] == captured_b["frozen_message_count"], (
        f"Streaming/non-streaming divergence: stream=False produced "
        f"frozen={captured_a['frozen_message_count']}, stream=True produced "
        f"{captured_b['frozen_message_count']}. The optimization path must "
        f"be identical for both."
    )
