from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from headroom.proxy.gateway.config import CapabilityConfig, GatewayConfigSnapshot
from headroom.proxy.gateway.errors import GatewayPublicError
from headroom.proxy.gateway.protocols.events import (
    StreamEvent,
    translate_event,
    translate_sse_stream,
)
from headroom.proxy.models import ProxyConfig
from headroom.proxy.server import create_app

EXAMPLE = (
    Path(__file__).parents[2]
    / "docs"
    / "proposals"
    / "unified-api-gateway"
    / "examples"
    / "gateway.api-keys.json"
)

ANTHROPIC_START = (
    b'data: {"type":"message_start","message":{"content":[]}}\n\n'
    b'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
)


@pytest.mark.asyncio
async def test_translation_accepts_crlf_multiline_and_many_bounded_events():
    async def upstream():
        yield (
            b": ping\r\n\r\n" * 100000
            + ANTHROPIC_START
            + b'data: {"type":"content_block_delta","index":0,\r\ndata: "delta":{"type":"text_delta","text":"hello"}}\r\n\r\ndata: {"type":"content_block_stop","index":0}\r\n\r\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}\r\n\r\ndata: {"type":"message_stop"}\r\n\r\n'
        )

    received = b"".join(
        [
            part
            async for part in translate_sse_stream(
                "anthropic-messages", "openai-chat", upstream(), public_model="alias"
            )
        ]
    )
    assert b'"content":"hello"' in received
    assert received.endswith(b"data: [DONE]\n\n")


def test_standalone_tool_argument_translation_is_unavailable() -> None:
    source = StreamEvent(
        kind="tool_argument_delta",
        index=1,
        call_id="call_2",
        tool_name="lookup",
        data='{"city":"Mon',
    )

    with pytest.raises(GatewayPublicError):
        translate_event("openai-chat", "anthropic-messages", source)


@pytest.mark.asyncio
async def test_anthropic_text_delta_becomes_openai_chunk_before_stream_completion() -> None:
    completion_released = False

    async def source():
        yield ANTHROPIC_START
        yield b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
        yield b'"delta":{"type":"text_delta","text":"Hel"}}\n\n'
        assert completion_released
        yield b'data: {"type":"content_block_stop","index":0}\n\nevent: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}\n\nevent: message_stop\ndata: {"type":"message_stop"}\n\n'

    translated = translate_sse_stream(
        "anthropic-messages",
        "openai-chat",
        source(),
        public_model="public-claude",
    )
    iterator = translated.__aiter__()

    assert b'"role":"assistant"' in await anext(iterator)
    assert await anext(iterator) == (
        b'data: {"id":"gateway-translated","object":"chat.completion.chunk","created":0,"model":"public-claude",'
        b'"choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}]}\n\n'
    )
    completion_released = True
    assert await anext(iterator) == (
        b'data: {"id":"gateway-translated","object":"chat.completion.chunk","created":0,"model":"public-claude",'
        b'"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    )
    assert await anext(iterator) == b"data: [DONE]\n\n"


def test_gateway_dispatch_uses_incremental_translated_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEADROOM_GATEWAY_CLIENT_TOKEN", "client-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-secret")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")

    async def upstream(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                ANTHROPIC_START
                + b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
                b'"delta":{"type":"text_delta","text":"Hi"}}\n\n'
                b'data: {"type":"content_block_stop","index":0}\n\nevent: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}\n\nevent: message_stop\ndata: {"type":"message_stop"}\n\n'
            ),
        )

    snapshot = GatewayConfigSnapshot.load(EXAMPLE)
    anthropic = next(route for route in snapshot.routes if route.id == "anthropic-native")
    translated_route = anthropic.model_copy(
        update={
            "public_model": "public-claude",
            "upstream_model": "claude-provider",
            "ingress_protocols": ("openai-chat", "anthropic-messages"),
            "capabilities": {
                **anthropic.capabilities,
                "openai-chat": {
                    "http-json": CapabilityConfig(features=("text",)),
                    "http-stream": CapabilityConfig(features=("text",)),
                },
            },
            "translation": "qualified",
            "body_contract": "routed-native",
        }
    )
    snapshot = snapshot.model_copy(
        update={
            "routes": tuple(
                translated_route if route.id == anthropic.id else route for route in snapshot.routes
            )
        }
    )
    app = create_app(ProxyConfig(gateway=snapshot))
    app.state.gateway_runtime.dependencies.http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(upstream)
    )

    response = TestClient(app).post(
        "/v1/chat/completions",
        headers={"host": "127.0.0.1:8787", "authorization": "Bearer client-secret"},
        json={
            "model": "public-claude",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 16,
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert response.content == (
        b'data: {"id":"gateway-translated","object":"chat.completion.chunk","created":0,"model":"public-claude",'
        b'"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
        b'data: {"id":"gateway-translated","object":"chat.completion.chunk","created":0,"model":"public-claude",'
        b'"choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}\n\n'
        b'data: {"id":"gateway-translated","object":"chat.completion.chunk","created":0,"model":"public-claude",'
        b'"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        b"data: [DONE]\n\n"
    )
