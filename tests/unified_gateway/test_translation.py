from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from headroom.proxy.gateway.config import CapabilityConfig, GatewayConfigSnapshot
from headroom.proxy.gateway.errors import GatewayAuthorizationError
from headroom.proxy.gateway.protocols import translate, translate_response
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


@pytest.mark.parametrize(
    ("source_protocol", "target_protocol", "source", "expected"),
    [
        (
            "openai-chat",
            "anthropic-messages",
            {
                "model": "public-model",
                "messages": [
                    {"role": "system", "content": "Be exact."},
                    {"role": "user", "content": "Hello"},
                ],
                "max_tokens": 64,
            },
            {
                "model": "public-model",
                "system": [{"type": "text", "text": "Be exact."}],
                "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
                "max_tokens": 64,
            },
        ),
        (
            "anthropic-messages",
            "openai-chat",
            {
                "model": "public-model",
                "system": "Be exact.",
                "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
                "max_tokens": 64,
            },
            {
                "model": "public-model",
                "messages": [
                    {"role": "system", "content": "Be exact."},
                    {"role": "user", "content": "Hello"},
                ],
                "max_tokens": 64,
            },
        ),
        (
            "gemini-generate",
            "openai-chat",
            {
                "systemInstruction": {"parts": [{"text": "Be exact."}]},
                "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
                "generationConfig": {"maxOutputTokens": 64},
            },
            {
                "messages": [
                    {"role": "system", "content": "Be exact."},
                    {"role": "user", "content": "Hello"},
                ],
                "max_tokens": 64,
            },
        ),
    ],
)
def test_directed_translation_matches_independent_literal_oracle(
    source_protocol: str,
    target_protocol: str,
    source: dict[str, object],
    expected: dict[str, object],
) -> None:
    assert translate(source_protocol, target_protocol, source) == expected


def test_public_parallel_tool_translation_is_unavailable() -> None:
    source = {
        "model": "public-model",
        "messages": [
            {"role": "user", "content": "Compare weather"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "weather", "arguments": '{"city":"Montréal"}'},
                    },
                    {
                        "id": "call_b",
                        "type": "function",
                        "function": {"name": "weather", "arguments": '{"city":"Tokyo"}'},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_a", "content": "3 C"},
            {"role": "tool", "tool_call_id": "call_b", "content": "20 C"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
    }

    with pytest.raises(GatewayAuthorizationError):
        translate("openai-chat", "anthropic-messages", source)


def test_inline_image_and_text_order_survives_openai_to_anthropic() -> None:
    source = {
        "model": "public-model",
        "max_tokens": 8,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Inspect this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AQID"},
                    },
                    {"type": "text", "text": "Describe it"},
                ],
            }
        ],
    }

    translated = translate("openai-chat", "anthropic-messages", source)

    assert translated["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Inspect this"},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": "AQID"},
                },
                {"type": "text", "text": "Describe it"},
            ],
        }
    ]


def test_public_tool_use_and_result_translation_is_unavailable() -> None:
    source = {
        "model": "public-model",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_a",
                        "name": "weather",
                        "input": {"city": "Montréal"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "call_a", "content": "3 C"}],
            },
        ],
        "tools": [
            {
                "name": "weather",
                "description": "Get weather",
                "input_schema": {"type": "object"},
            }
        ],
    }

    with pytest.raises(GatewayAuthorizationError):
        translate("anthropic-messages", "openai-chat", source)


@pytest.mark.parametrize(
    ("source_protocol", "target_protocol", "source", "expected"),
    [
        (
            "openai-chat",
            "anthropic-messages",
            {
                "id": "chat_1",
                "model": "provider-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Done"},
                        "finish_reason": "length",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            },
            {
                "id": "chat_1",
                "type": "message",
                "role": "assistant",
                "model": "public-model",
                "content": [{"type": "text", "text": "Done"}],
                "stop_reason": "max_tokens",
                "stop_sequence": None,
                "usage": {"input_tokens": 5, "output_tokens": 3},
            },
        ),
        (
            "anthropic-messages",
            "openai-chat",
            {
                "content": [{"type": "text", "text": "Done"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 3},
            },
            {
                "id": None,
                "object": "chat.completion",
                "model": "public-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Done"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            },
        ),
    ],
)
def test_response_translation_matches_literal_finish_and_usage_oracle(
    source_protocol: str,
    target_protocol: str,
    source: dict[str, object],
    expected: dict[str, object],
) -> None:
    assert (
        translate_response(
            source_protocol,
            target_protocol,
            source,
            public_model="public-model",
        )
        == expected
    )


@pytest.mark.parametrize("unsupported_field", ["thinking", "mcp_servers", "unknown_extension"])
def test_unrepresentable_fields_are_rejected(unsupported_field: str) -> None:
    source = {
        "model": "public-model",
        "messages": [{"role": "user", "content": "Hello"}],
        unsupported_field: {"enabled": True},
    }

    with pytest.raises(GatewayAuthorizationError, match="unsupported"):
        translate("openai-chat", "anthropic-messages", source)


def test_openai_chat_to_anthropic_route_translates_request_and_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEADROOM_GATEWAY_CLIENT_TOKEN", "client-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-secret")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
    captured: list[httpx.Request] = []

    async def upstream(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-provider",
                "content": [{"type": "text", "text": "Hello back"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 7, "output_tokens": 2},
            },
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
    real_broker = app.state.gateway_runtime.broker

    class CountingBroker:
        acquisitions = 0

        async def acquire(self, route, account_ref=None):
            self.acquisitions += 1
            return await real_broker.acquire(route, account_ref=account_ref)

    broker = CountingBroker()
    app.state.gateway_runtime.dependencies.broker = broker

    response = TestClient(app).post(
        "/v1/chat/completions",
        headers={"host": "127.0.0.1:8787", "authorization": "Bearer client-secret"},
        json={
            "model": "public-claude",
            "messages": [
                {"role": "system", "content": "Be exact."},
                {"role": "user", "content": "Hello"},
            ],
            "max_tokens": 64,
        },
    )

    assert len(captured) == 1
    assert str(captured[0].url) == "https://api.anthropic.com/v1/messages"
    assert captured[0].headers["x-api-key"] == "anthropic-secret"
    assert json.loads(captured[0].content) == {
        "model": "claude-provider",
        "system": [{"type": "text", "text": "Be exact."}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
        "max_tokens": 64,
    }
    assert response.json() == {
        "id": "msg_1",
        "object": "chat.completion",
        "model": "public-claude",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello back"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
    }

    rejected = TestClient(app).post(
        "/v1/chat/completions",
        headers={"host": "127.0.0.1:8787", "authorization": "Bearer client-secret"},
        json={
            "model": "public-claude",
            "messages": [{"role": "user", "content": "Hello"}],
            "thinking": {"type": "enabled", "budget_tokens": 1000},
        },
    )
    assert rejected.status_code == 400
    assert broker.acquisitions == 1
    assert len(captured) == 1
