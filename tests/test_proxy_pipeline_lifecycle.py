from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, call, patch

import httpx
from fastapi.testclient import TestClient

from headroom.pipeline import PipelineStage
from headroom.proxy.server import ProxyConfig, create_app


class _RecordingExtension:
    def __init__(self) -> None:
        self.stages: list[PipelineStage] = []

    def on_pipeline_event(self, event):
        self.stages.append(event.stage)
        return None


class _DummyTokenizer:
    def count_messages(self, messages):
        return len(messages)


def _assert_stage_order(stages: list[PipelineStage]) -> None:
    expected = [
        PipelineStage.SETUP,
        PipelineStage.PRE_START,
        PipelineStage.POST_START,
        PipelineStage.INPUT_RECEIVED,
        PipelineStage.INPUT_ROUTED,
        PipelineStage.INPUT_COMPRESSED,
        PipelineStage.INPUT_REMEMBERED,
        PipelineStage.PRE_SEND,
        PipelineStage.POST_SEND,
        PipelineStage.RESPONSE_RECEIVED,
    ]
    positions = [stages.index(stage) for stage in expected]
    assert positions == sorted(positions)


def test_proxy_shutdown_unloads_image_models() -> None:
    config = ProxyConfig(
        optimize=False,
        image_optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
    )
    app = create_app(config)
    proxy = app.state.proxy
    proxy.http_client = None
    proxy.memory_handler = None

    quota_registry = SimpleNamespace(stop_all=AsyncMock())
    with (
        patch("headroom.proxy.server.get_quota_registry", return_value=quota_registry),
        patch("headroom.models.ml_models.MLModelRegistry.unload_prefix") as unload_prefix,
    ):
        asyncio.run(proxy.shutdown())

    assert unload_prefix.call_args_list == [
        call("technique_router:"),
        call("siglip:"),
    ]
    quota_registry.stop_all.assert_awaited_once()


def test_openai_chat_pipeline_events_cover_proxy_lifecycle(monkeypatch) -> None:
    recorder = _RecordingExtension()
    config = ProxyConfig(
        optimize=True,
        image_optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        pipeline_extensions=[recorder],
        discover_pipeline_extensions=False,
    )
    app = create_app(config)

    with TestClient(app) as client:
        proxy = client.app.state.proxy
        proxy.openai_pipeline = SimpleNamespace(
            apply=lambda messages, model, **kwargs: SimpleNamespace(
                messages=[
                    {"role": "system", "content": "memory"},
                    {"role": "user", "content": "hello"},
                ],
                transforms_applied=["router:text:kompress"],
                tokens_before=10,
                tokens_after=6,
            )
        )
        proxy.memory_handler = SimpleNamespace(
            config=SimpleNamespace(inject_context=True, inject_tools=False),
            search_and_format_context=AsyncMock(return_value="memory"),
            has_memory_tool_calls=lambda response, provider: False,
        )

        monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl_1",
                    "object": "chat.completion",
                    "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
                },
            )

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-test", "x-headroom-user-id": "user-1"},
            json={
                "model": "gpt-5.4",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [{"type": "function", "function": {"name": "tool_a"}}],
            },
        )

    assert response.status_code == 200
    _assert_stage_order(recorder.stages)


def test_anthropic_messages_pipeline_events_cover_proxy_lifecycle(monkeypatch) -> None:
    recorder = _RecordingExtension()
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
        pipeline_extensions=[recorder],
        discover_pipeline_extensions=False,
    )
    app = create_app(config)

    with TestClient(app) as client:
        proxy = client.app.state.proxy
        proxy.anthropic_pipeline = SimpleNamespace(
            apply=lambda messages, model, **kwargs: SimpleNamespace(
                messages=[
                    {"role": "system", "content": "memory"},
                    {"role": "user", "content": "hello"},
                ],
                transforms_applied=["router:text:kompress"],
                tokens_before=10,
                tokens_after=6,
            )
        )
        proxy.memory_handler = SimpleNamespace(
            config=SimpleNamespace(inject_context=True, inject_tools=False),
            search_and_format_context=AsyncMock(return_value="memory"),
            has_memory_tool_calls=lambda response, provider: False,
        )

        monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
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
            headers={
                "x-api-key": "test-key",
                "anthropic-version": "2023-06-01",
                "x-headroom-user-id": "user-1",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 128,
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [
                    {"name": "tool_a", "description": "a", "input_schema": {"type": "object"}}
                ],
            },
        )

    assert response.status_code == 200
    _assert_stage_order(recorder.stages)
