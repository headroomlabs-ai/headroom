"""Directed independent oracles for the admitted translation boundary."""

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from headroom.proxy.gateway.capabilities import implemented_features
from headroom.proxy.gateway.config import CapabilityConfig, GatewayConfigSnapshot
from headroom.proxy.gateway.dispatch import rewrite_routed_native_model
from headroom.proxy.gateway.errors import GatewayPublicError
from headroom.proxy.gateway.protocols import translate, translate_response
from headroom.proxy.gateway.protocols.events import translate_sse_stream
from headroom.proxy.models import ProxyConfig
from headroom.proxy.server import create_app


def test_gemini_ingress_receives_openai_response_with_length_and_usage():
    assert translate_response(
        "openai-chat",
        "gemini-generate",
        {
            "id": "chat-1",
            "choices": [
                {"message": {"role": "assistant", "content": "雪"}, "finish_reason": "length"}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        },
        public_model="public",
    ) == {
        "responseId": "chat-1",
        "modelVersion": "public",
        "candidates": [
            {
                "index": 0,
                "content": {"role": "model", "parts": [{"text": "雪"}]},
                "finishReason": "MAX_TOKENS",
            }
        ],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2, "totalTokenCount": 7},
    }


@pytest.mark.parametrize(
    "source,target,payload,expected",
    [
        (
            "openai-chat",
            "anthropic-messages",
            {
                "choices": [
                    {"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 5},
            },
            {"input_tokens": 5},
        ),
        (
            "anthropic-messages",
            "openai-chat",
            {
                "content": [{"type": "text", "text": "hi"}],
                "stop_reason": "end_turn",
                "usage": {"output_tokens": 2},
            },
            {"completion_tokens": 2},
        ),
    ],
)
def test_partial_response_usage_preserves_known_units_without_inventing_missing_counts(
    source, target, payload, expected
):
    assert translate_response(source, target, payload, public_model="public")["usage"] == expected


@pytest.mark.parametrize("target", ["anthropic-messages", "gemini-generate"])
def test_unrequested_semantic_response_fields_fail_closed(target):
    with pytest.raises(GatewayPublicError):
        translate_response(
            "openai-chat",
            target,
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                        "logprobs": {"content": [{"token": "hi", "logprob": -1}]},
                    }
                ]
            },
            public_model="public",
        )


def test_anthropic_plain_string_is_not_an_unsupported_content_form():
    assert translate(
        "anthropic-messages",
        "openai-chat",
        {
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 8,
        },
    ) == {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 8}


@pytest.mark.parametrize(
    "payload",
    [
        {"messages": [{"role": "user", "content": "first"}, {"role": "system", "content": "late"}]},
        {"messages": [{"role": "user", "content": "hello"}], "stream": "false"},
    ],
)
def test_openai_unrepresentable_request_rejects_without_reordering(payload):
    with pytest.raises(GatewayPublicError):
        translate("openai-chat", "anthropic-messages", payload)


@pytest.mark.parametrize(
    "blocks",
    [
        [
            {"type": "tool_result", "tool_use_id": "a", "content": "result"},
            {"type": "text", "text": "tail"},
        ],
        [
            {"type": "tool_use", "id": "a", "name": "f", "input": {}},
            {"type": "text", "text": "tail"},
        ],
    ],
)
def test_anthropic_interleaved_tool_and_text_rejects_instead_of_dropping_or_reordering(blocks):
    with pytest.raises(GatewayPublicError):
        translate(
            "anthropic-messages",
            "openai-chat",
            {"messages": [{"role": "assistant", "content": blocks}]},
        )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": "!"},
                        }
                    ],
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "audio/wav", "data": "YQ=="},
                        }
                    ],
                }
            ]
        },
    ],
)
def test_anthropic_invalid_image_rejects(payload):
    with pytest.raises(GatewayPublicError):
        translate("anthropic-messages", "openai-chat", payload)


def test_gemini_ordered_inline_image_request_has_literal_openai_oracle():
    assert translate(
        "gemini-generate",
        "openai-chat",
        {
            "systemInstruction": {"parts": [{"text": "Be exact"}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": "before"},
                        {"inlineData": {"mimeType": "image/png", "data": "YQ=="}},
                        {"text": "after"},
                    ],
                }
            ],
        },
    ) == {
        "messages": [
            {"role": "system", "content": "Be exact"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,YQ=="}},
                    {"type": "text", "text": "after"},
                ],
            },
        ]
    }


@pytest.mark.parametrize(
    "protocol,target",
    [
        ("openai-chat", "anthropic-messages"),
        ("anthropic-messages", "openai-chat"),
        ("gemini-generate", "openai-chat"),
    ],
)
def test_admitted_translation_features_have_both_transport_halves(protocol, target):
    assert implemented_features(protocol, "http-json", False, target) == frozenset(
        {"text", "inline_images"}
    )
    assert implemented_features(protocol, "http-stream", False, target) == frozenset(
        {"text", "inline_images"}
    )
    assert not implemented_features("openai-responses", "http-json", False, target)


async def _chunks(data):
    for byte in data:
        yield bytes([byte])


def _data(raw):
    return [
        json.loads(line[6:])
        for line in raw.decode().splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


CHAT_STREAM = (
    'data: {"id":"chat-1","choices":[{"index":0,"delta":{"role":"assistant","content":"雪"},"finish_reason":null}]}\n\n'
    'data: {"id":"chat-1","choices":[{"index":0,"delta":{},"finish_reason":"length"}]}\n\n'
    'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}\n\n'
    "data: [DONE]\n\n"
).encode()


@pytest.mark.asyncio
async def test_openai_stream_to_gemini_preserves_text_finish_and_usage():
    raw = b"".join(
        [
            chunk
            async for chunk in translate_sse_stream(
                "openai-chat", "gemini-generate", _chunks(CHAT_STREAM), public_model="public"
            )
        ]
    )
    assert _data(raw) == [
        {
            "responseId": "chat-1",
            "modelVersion": "public",
            "candidates": [{"index": 0, "content": {"role": "model", "parts": [{"text": "雪"}]}}],
        },
        {
            "responseId": "chat-1",
            "modelVersion": "public",
            "candidates": [{"index": 0, "finishReason": "MAX_TOKENS"}],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 2,
                "totalTokenCount": 7,
            },
        },
    ]


@pytest.mark.asyncio
async def test_truncated_chat_stream_never_publishes_gemini_success_finish():
    emitted = []
    with pytest.raises(GatewayPublicError):
        async for chunk in translate_sse_stream(
            "openai-chat",
            "gemini-generate",
            _chunks(CHAT_STREAM.removesuffix(b"data: [DONE]\n\n")),
            public_model="public",
        ):
            emitted.append(chunk)
    assert b"finishReason" not in b"".join(emitted)


@pytest.mark.asyncio
async def test_openai_stream_to_anthropic_has_complete_lifecycle_and_usage():
    raw = b"".join(
        [
            chunk
            async for chunk in translate_sse_stream(
                "openai-chat", "anthropic-messages", _chunks(CHAT_STREAM), public_model="public"
            )
        ]
    )
    events = _data(raw)
    assert [event["type"] for event in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[2] == {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "雪"},
    }
    assert events[4] == {
        "type": "message_delta",
        "delta": {"stop_reason": "max_tokens", "stop_sequence": None},
        "usage": {"input_tokens": 5, "output_tokens": 2},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data",
    [
        b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"x"}}\n\n',
        b'data: {"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"secret"}}\n\n',
        b'data: {"type":"error","error":{"message":"secret"}}\n\n',
    ],
)
async def test_translated_stream_does_not_accept_truncated_or_unknown_semantics(data):
    with pytest.raises(GatewayPublicError):
        async for _ in translate_sse_stream(
            "anthropic-messages", "openai-chat", _chunks(data), public_model="public"
        ):
            pass


def test_routed_native_preserves_all_unrelated_lexical_bytes():
    original = (
        b'{ "model" : "public", "unknown":1.00, "nested":{"model":"public"}, "signed":"a\\u0062" }'
    )
    assert (
        rewrite_routed_native_model(original, public_model="public", upstream_model="provider")
        == b'{ "model" : "provider", "unknown":1.00, "nested":{"model":"public"}, "signed":"a\\u0062" }'
    )


def test_unqualified_request_direction_cannot_use_lossy_gemini_encoder():
    with pytest.raises(GatewayPublicError):
        translate(
            "openai-chat",
            "gemini-generate",
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,YQ=="},
                            }
                        ],
                    }
                ]
            },
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "wire",
    [
        b'data: {"type":"message_stop"}\n\n',
        b'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":8}}\n\ndata: {"type":"message_stop"}\n\n',
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"\\ud800"}}\n\n',
    ],
)
async def test_malformed_anthropic_lifecycle_has_bounded_gateway_error(wire):
    with pytest.raises(GatewayPublicError):
        async for _ in translate_sse_stream(
            "anthropic-messages", "openai-chat", _chunks(wire), public_model="public"
        ):
            pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source,target",
    [
        ("anthropic-messages", "openai-chat"),
        ("openai-chat", "anthropic-messages"),
        ("openai-chat", "gemini-generate"),
    ],
)
@pytest.mark.parametrize(
    "wire",
    [
        b"data: {invalid}\n\n",
        b'data: {"text":"\xf0\x80"}\n\n',
        b"data: " + b"x" * 1_048_577,
        b'event: error\ndata: {"error":{"message":"PROVIDER_SECRET"}}\n\n',
        b'data: {"error":{"message":"PROVIDER_SECRET"}}\n\n',
        b"data: {",
    ],
    ids=[
        "malformed-json",
        "fragmented-invalid-utf8",
        "oversized",
        "named-provider-error",
        "provider-error",
        "partial-event",
    ],
)
async def test_every_advertised_stream_direction_rejects_bounded_malformed_provider_events(
    source, target, wire
):
    async def fragments():
        yield wire[:13]
        yield wire[13:]

    with pytest.raises(GatewayPublicError) as caught:
        async for _ in translate_sse_stream(source, target, fragments(), public_model="public"):
            pass
    assert "PROVIDER_SECRET" not in caught.value.message
    assert len(caught.value.message) < 100


@pytest.mark.parametrize(
    "protocol,body",
    [
        ("openai-chat", {"messages": [{"role": "user", "content": "hello"}]}),
        (
            "openai-chat",
            {"messages": [], "max_tokens": 8, "tools": [{"type": "web_search_preview"}]},
        ),
        (
            "openai-chat",
            {
                "messages": [],
                "max_tokens": 8,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "x", "schema": {"type": "object"}},
                },
            },
        ),
        (
            "openai-chat",
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "a",
                                "type": "function",
                                "function": {"name": "f", "arguments": "{}"},
                            }
                        ],
                    }
                ],
                "max_tokens": 8,
            },
        ),
        (
            "anthropic-messages",
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "SECRET", "signature": "SIGNED"}
                        ],
                    }
                ],
                "max_tokens": 8,
            },
        ),
        (
            "anthropic-messages",
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "server_tool_use", "id": "a", "name": "search", "input": {}}
                        ],
                    }
                ],
                "max_tokens": 8,
            },
        ),
        (
            "gemini-generate",
            {
                "contents": [
                    {"role": "model", "parts": [{"text": "SECRET", "thoughtSignature": "SIGNED"}]}
                ]
            },
        ),
        (
            "gemini-generate",
            {
                "contents": [
                    {"role": "model", "parts": [{"functionCall": {"name": "f", "args": {}}}]}
                ]
            },
        ),
        (
            "gemini-generate",
            {
                "contents": [],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseSchema": {"type": "OBJECT"},
                },
            },
        ),
    ],
)
@pytest.mark.parametrize("stream", [False, True])
def test_unsupported_translation_has_zero_broker_and_upstream_calls(
    monkeypatch, protocol, body, stream
):
    for name in (
        "HEADROOM_GATEWAY_CLIENT_TOKEN",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
    ):
        monkeypatch.setenv(name, "client-secret")
    example = (
        Path(__file__).parents[2]
        / "docs/proposals/unified-api-gateway/examples/gateway.api-keys.json"
    )
    snapshot = GatewayConfigSnapshot.load(example)
    base = next(
        route
        for route in snapshot.routes
        if route.id == ("anthropic-native" if protocol == "openai-chat" else "openai-native")
    )
    route = base.model_copy(
        update={
            "public_model": "public",
            "upstream_model": "provider",
            "ingress_protocols": (protocol,),
            "native_protocols": (
                "anthropic-messages" if protocol == "openai-chat" else "openai-chat",
            ),
            "translation": "qualified",
            "body_contract": "routed-native",
            "capabilities": {
                protocol: {
                    transport: CapabilityConfig(features=("text", "inline_images"))
                    for transport in ("http-json", "http-stream")
                }
            },
        }
    )
    snapshot = snapshot.model_copy(
        update={"routes": tuple(route if r.id == base.id else r for r in snapshot.routes)}
    )
    app = create_app(ProxyConfig(gateway=snapshot))
    counts = {"broker": 0, "upstream": 0}

    class Broker:
        async def acquire(self, *args, **kwargs):
            counts["broker"] += 1
            raise AssertionError("Unexpected broker acquisition")

    async def upstream(request):
        counts["upstream"] += 1
        raise AssertionError("Unexpected upstream request")

    app.state.gateway_runtime.dependencies.broker = Broker()
    app.state.gateway_runtime.dependencies.http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(upstream)
    )
    paths = {
        "openai-chat": "/v1/chat/completions",
        "anthropic-messages": "/v1/messages",
        "gemini-generate": f"/v1beta/models/public:{'streamGenerateContent' if stream else 'generateContent'}",
    }
    sent = dict(body)
    if protocol != "gemini-generate":
        sent.update(model="public", stream=stream)
    response = TestClient(app).post(
        paths[protocol],
        headers={"host": "127.0.0.1:8787", "authorization": "Bearer client-secret"},
        json=sent,
    )
    assert response.status_code == 400
    assert counts == {"broker": 0, "upstream": 0}
    assert "SECRET" not in response.text and "SIGNED" not in response.text
    if protocol == "gemini-generate":
        assert response.json()["error"]["code"] == 400
        assert response.json()["error"]["status"] == "INVALID_ARGUMENT"
    elif protocol == "anthropic-messages":
        assert response.json()["type"] == "error"


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["anthropic-messages", "gemini-generate"])
@pytest.mark.parametrize("extra", [{"logprobs": {"content": []}}, {"audio": {"id": "secret"}}])
async def test_translated_chat_stream_rejects_unmapped_choice_semantics(target, extra):
    choice = {"index": 0, "delta": {"content": "hello"}, "finish_reason": "stop", **extra}
    wire = b"data: " + json.dumps({"choices": [choice]}).encode() + b"\n\ndata: [DONE]\n\n"
    with pytest.raises(GatewayPublicError):
        async for _ in translate_sse_stream("openai-chat", target, _chunks(wire), public_model="p"):
            pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        "missing-message-start",
        "missing-block-start",
        "wrong-index",
        "missing-block-stop",
        "late-content",
        "duplicate-finish",
        "nonzero-first-index",
    ],
)
async def test_translated_anthropic_stream_rejects_invalid_event_order(mutation):
    events = [
        {"type": "message_start", "message": {"id": "m", "content": [], "usage": {}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
        {"type": "message_stop"},
    ]
    if mutation == "missing-message-start":
        del events[0]
    elif mutation == "missing-block-start":
        del events[1]
    elif mutation == "wrong-index":
        events[2]["index"] = 1
    elif mutation == "missing-block-stop":
        del events[3]
    elif mutation == "late-content":
        events.insert(-1, events[2])
    elif mutation == "duplicate-finish":
        events.insert(-1, events[4])
    else:
        for event in events:
            if "index" in event:
                event["index"] = 1
    wire = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
    received = []
    with pytest.raises(GatewayPublicError):
        async for chunk in translate_sse_stream(
            "anthropic-messages", "openai-chat", _chunks(wire), public_model="p"
        ):
            received.append(chunk)
    assert b"[DONE]" not in b"".join(received)


@pytest.mark.asyncio
async def test_translated_anthropic_text_blocks_keep_order_and_usage():
    wire = (
        b'data: {"type":"message_start","message":{"id":"m","content":[],"usage":{"input_tokens":3}}}\n\n'
        b'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":"first"}}\n\n'
        b'data: {"type":"content_block_stop","index":0}\n\n'
        b'data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}\n\n'
        b'data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"second"}}\n\n'
        b'data: {"type":"content_block_stop","index":1}\n\n'
        b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}\n\n'
        b'data: {"type":"message_stop"}\n\n'
    )
    output = b"".join(
        [
            chunk
            async for chunk in translate_sse_stream(
                "anthropic-messages", "openai-chat", _chunks(wire), public_model="p"
            )
        ]
    )
    events = _data(output)
    assert [event["choices"][0]["delta"].get("content") for event in events[:-2]] == [
        None,
        "first",
        "second",
    ]
    assert events[-2]["choices"][0]["finish_reason"] == "stop"
    assert events[-1]["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    assert output.endswith(b"data: [DONE]\n\n")
