"""Independent regression oracles for protocol completeness review findings."""

import json
import time
from itertools import product
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from headroom.proxy.gateway.config import CapabilityConfig, GatewayConfigSnapshot, LimitsConfig
from headroom.proxy.gateway.errors import GatewayPublicError
from headroom.proxy.gateway.protocols import translate, translate_response
from headroom.proxy.gateway.protocols.events import (
    StreamEvent,
    translate_event,
    translate_sse_stream,
)
from headroom.proxy.gateway.streaming import StreamObserver
from headroom.proxy.models import ProxyConfig
from headroom.proxy.server import create_app


async def chunks(wire):
    yield wire


@pytest.mark.parametrize(
    "source,target,payload",
    [
        (
            "openai-chat",
            "anthropic-messages",
            {
                "messages": [],
                "max_tokens": 8,
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "f", "parameters": {"type": "object"}},
                    }
                ],
            },
        ),
        (
            "openai-chat",
            "anthropic-messages",
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
                            },
                            {
                                "id": "b",
                                "type": "function",
                                "function": {"name": "f", "arguments": "{}"},
                            },
                        ],
                    }
                ],
                "max_tokens": 8,
            },
        ),
        (
            "openai-chat",
            "anthropic-messages",
            {"messages": [{"role": "tool", "tool_call_id": "a", "content": "ok"}], "max_tokens": 8},
        ),
        ("openai-chat", "anthropic-messages", {"messages": []}),
        (
            "anthropic-messages",
            "openai-chat",
            {"messages": [], "tools": [{"name": "f", "input_schema": {"type": "object"}}]},
        ),
        (
            "anthropic-messages",
            "openai-chat",
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [{"type": "tool_use", "id": "a", "name": "f", "input": {}}],
                    }
                ]
            },
        ),
        (
            "anthropic-messages",
            "openai-chat",
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": "a", "content": "ok"}],
                    }
                ]
            },
        ),
        (
            "gemini-generate",
            "openai-chat",
            {
                "contents": [
                    {"role": "model", "parts": [{"functionCall": {"name": "f", "args": {}}}]}
                ]
            },
        ),
    ],
    ids=[
        "tools",
        "parallel-calls",
        "chat-result",
        "missing-limit",
        "anthropic-tools",
        "anthropic-call",
        "anthropic-result",
        "gemini-call",
    ],
)
def test_public_request_helper_rejects_unqualified_features(source, target, payload):
    with pytest.raises(GatewayPublicError) as error:
        translate(source, target, payload)
    assert error.value.status_code == 400


PROTOCOLS = ["openai-chat", "anthropic-messages", "gemini-generate", "openai-responses"]
REVERSE = {
    ("anthropic-messages", "openai-chat"),
    ("openai-chat", "anthropic-messages"),
    ("openai-chat", "gemini-generate"),
}


@pytest.mark.parametrize(
    "source,target",
    [pair for pair in product(PROTOCOLS, repeat=2) if pair[::-1] not in REVERSE],
)
def test_public_request_helper_rejects_every_unqualified_direction(source, target):
    payload = (
        {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
        if source == "gemini-generate"
        else {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}
    )
    with pytest.raises(GatewayPublicError) as caught:
        translate(source, target, payload)
    assert caught.value.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source,target", [pair for pair in product(PROTOCOLS, repeat=2) if pair not in REVERSE]
)
async def test_public_stream_helper_rejects_every_unqualified_direction(source, target):
    with pytest.raises(GatewayPublicError) as caught:
        async for _ in translate_sse_stream(
            source, target, chunks(b"data: [DONE]\n\n"), public_model="p"
        ):
            pass
    assert caught.value.status_code == 400


@pytest.mark.parametrize(
    "source,target", [pair for pair in product(PROTOCOLS, repeat=2) if pair not in REVERSE]
)
def test_public_response_helper_rejects_every_unqualified_direction(source, target):
    payload = {
        "candidates": [
            {"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}
        ]
    }
    with pytest.raises(GatewayPublicError):
        translate_response(source, target, payload, public_model="p")


def test_public_fragment_helper_cannot_admit_excluded_tool_streams():
    with pytest.raises(GatewayPublicError):
        translate_event(
            "openai-chat",
            "anthropic-messages",
            StreamEvent(kind="tool_argument_delta", index=0, data="{}"),
        )


@pytest.mark.parametrize(
    "source,target,payload",
    [
        ("anthropic-messages", "openai-chat", {"content": [], "stop_reason": "tool_use"}),
        (
            "anthropic-messages",
            "openai-chat",
            {"content": [{"type": "text", "text": "REFUSAL_SECRET"}], "stop_reason": "refusal"},
        ),
        (
            "openai-chat",
            "anthropic-messages",
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "refusal": "REFUSAL_SECRET",
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
        ),
    ],
)
def test_public_response_helper_rejects_unqualified_or_refused_output(source, target, payload):
    with pytest.raises(GatewayPublicError):
        translate_response(source, target, payload, public_model="p")


@pytest.fixture
def translated_client(monkeypatch):
    for name in (
        "HEADROOM_GATEWAY_CLIENT_TOKEN",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
    ):
        monkeypatch.setenv(name, "client-secret")

    def make(protocol):
        snapshot = GatewayConfigSnapshot.load(
            Path(__file__).parents[2]
            / "docs/proposals/unified-api-gateway/examples/gateway.api-keys.json"
        )
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
                raise AssertionError("Unexpected acquisition")

        async def upstream(request):
            counts["upstream"] += 1
            raise AssertionError("Unexpected upstream")

        app.state.gateway_runtime.dependencies.broker = Broker()
        app.state.gateway_runtime.dependencies.http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(upstream)
        )
        return TestClient(
            app, headers={"host": "127.0.0.1:8787", "authorization": "Bearer client-secret"}
        ), counts

    return make


@pytest.mark.parametrize(
    "protocol,path,payload",
    [
        (
            "openai-chat",
            "/v1/chat/completions",
            {"model": "public", "messages": [], "max_tokens": 8},
        ),
        (
            "anthropic-messages",
            "/v1/messages",
            {"model": "public", "messages": [], "max_tokens": 8},
        ),
        ("gemini-generate", "/v1beta/models/public:generateContent", {"contents": []}),
    ],
)
def test_unknown_field_errors_are_bounded_and_nonreflecting(
    translated_client, protocol, path, payload
):
    unknown = "SECRET_FIELD_" + "x" * 200_000
    body = {**payload, unknown: True}
    with pytest.raises(GatewayPublicError) as caught:
        translate(
            protocol,
            "anthropic-messages" if protocol == "openai-chat" else "openai-chat",
            {
                key: value
                for key, value in body.items()
                if protocol != "gemini-generate" or key != "model"
            },
        )
    client, counts = translated_client(protocol)
    response = client.post(path, json=body)
    assert response.status_code == 400
    assert len(response.content) < 512
    assert b"SECRET_FIELD" not in response.content
    assert len(caught.value.message) < 160
    assert "SECRET_FIELD" not in caught.value.message
    assert counts == {"broker": 0, "upstream": 0}
    if protocol == "anthropic-messages":
        assert response.json()["type"] == "error"
    elif protocol == "gemini-generate":
        assert response.json()["error"]["code"] == 400


@pytest.mark.parametrize(
    "settings",
    [
        {"maxOutputTokens": "12"},
        {"maxOutputTokens": True},
        {"maxOutputTokens": 0},
        {"maxOutputTokens": -1},
        {"maxOutputTokens": 1.5},
        {"maxOutputTokens": None},
        {"temperature": "0.5"},
        {"temperature": True},
        {"temperature": -0.1},
        {"temperature": 2.1},
        {"temperature": None},
    ],
)
def test_invalid_gemini_generation_settings_reject_before_broker(translated_client, settings):
    body = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}], "generationConfig": settings}
    client, counts = translated_client("gemini-generate")
    response = client.post("/v1beta/models/public:generateContent", json=body)
    assert response.status_code == 400
    assert counts == {"broker": 0, "upstream": 0}
    assert len(response.content) < 512


@pytest.mark.parametrize("temperature", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_gemini_temperature_rejected_by_public_helper(temperature):
    with pytest.raises(GatewayPublicError):
        translate(
            "gemini-generate",
            "openai-chat",
            {"contents": [], "generationConfig": {"temperature": temperature}},
        )


def response_with_usage(source, usage):
    if source == "openai-chat":
        return {
            "choices": [
                {"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
            ],
            "usage": usage,
        }
    return {"content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn", "usage": usage}


USAGE_ORACLES = [
    (
        "openai-chat",
        "gemini-generate",
        {
            "prompt_tokens": 3,
            "completion_tokens": 5,
            "total_tokens": 8,
            "completion_tokens_details": {"reasoning_tokens": 4},
        },
        {
            "promptTokenCount": 3,
            "candidatesTokenCount": 1,
            "thoughtsTokenCount": 4,
            "totalTokenCount": 8,
        },
    ),
    (
        "openai-chat",
        "gemini-generate",
        {"completion_tokens_details": {"reasoning_tokens": 4}},
        {"thoughtsTokenCount": 4},
    ),
    (
        "openai-chat",
        "gemini-generate",
        {"prompt_tokens_details": {"cached_tokens": 2}},
        {"cachedContentTokenCount": 2},
    ),
    (
        "openai-chat",
        "anthropic-messages",
        {"prompt_tokens_details": {"cached_tokens": 2}},
        {"cache_read_input_tokens": 2},
    ),
    (
        "openai-chat",
        "anthropic-messages",
        {
            "prompt_tokens": 5,
            "completion_tokens": 2,
            "total_tokens": 7,
            "prompt_tokens_details": {"cached_tokens": 3},
        },
        {"input_tokens": 2, "output_tokens": 2, "cache_read_input_tokens": 3},
    ),
    (
        "anthropic-messages",
        "openai-chat",
        {"cache_read_input_tokens": 3},
        {"prompt_tokens_details": {"cached_tokens": 3}},
    ),
    (
        "anthropic-messages",
        "openai-chat",
        {"input_tokens": 2, "output_tokens": 4, "cache_read_input_tokens": 3},
        {
            "prompt_tokens": 5,
            "completion_tokens": 4,
            "total_tokens": 9,
            "prompt_tokens_details": {"cached_tokens": 3},
        },
    ),
]


@pytest.mark.parametrize("source,target,usage,expected", USAGE_ORACLES)
def test_available_usage_units_have_literal_response_mapping(source, target, usage, expected):
    result = translate_response(
        source, target, response_with_usage(source, usage), public_model="p"
    )
    assert result["usageMetadata" if target == "gemini-generate" else "usage"] == expected


def usage_stream(source, usage):
    if source == "openai-chat":
        events = [
            {"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}]},
            {"choices": [], "usage": usage},
        ]
    else:
        events = [
            {"type": "message_start", "message": {"content": [], "usage": {}}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": usage},
            {"type": "message_stop"},
        ]
    return b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events) + (
        b"data: [DONE]\n\n" if source == "openai-chat" else b""
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("source,target,usage,expected", USAGE_ORACLES)
async def test_available_usage_units_have_literal_stream_mapping(source, target, usage, expected):
    output = b"".join(
        [
            part
            async for part in translate_sse_stream(
                source, target, chunks(usage_stream(source, usage)), public_model="p"
            )
        ]
    )
    events = [
        json.loads(line[6:])
        for line in output.decode().splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    key = "usageMetadata" if target == "gemini-generate" else "usage"
    assert [event[key] for event in events if event.get(key)] == [expected]


UNMAPPABLE_USAGE = [
    (
        "openai-chat",
        "anthropic-messages",
        {"completion_tokens": 5, "completion_tokens_details": {"reasoning_tokens": 4}},
    ),
    ("openai-chat", "anthropic-messages", {"total_tokens": 8}),
    ("anthropic-messages", "openai-chat", {"cache_creation_input_tokens": 3}),
    ("anthropic-messages", "openai-chat", {"input_tokens": 2, "cache_creation_input_tokens": 3}),
    ("anthropic-messages", "openai-chat", {"cache_creation": {"ephemeral_5m_input_tokens": 3}}),
    ("openai-chat", "gemini-generate", {"completion_tokens_details": {"audio_tokens": 2}}),
    ("openai-chat", "gemini-generate", {"prompt_tokens_details": {"audio_tokens": 2}}),
    (
        "openai-chat",
        "gemini-generate",
        {"completion_tokens_details": {"accepted_prediction_tokens": 2}},
    ),
    (
        "openai-chat",
        "gemini-generate",
        {"completion_tokens_details": {"rejected_prediction_tokens": 2}},
    ),
    ("openai-chat", "gemini-generate", {"SECRET_UNKNOWN_UNIT": 3}),
    ("openai-chat", "gemini-generate", {"completion_tokens_details": {"SECRET_UNKNOWN_UNIT": 3}}),
    (
        "openai-chat",
        "gemini-generate",
        {"completion_tokens": 2, "completion_tokens_details": {"reasoning_tokens": 3}},
    ),
    (
        "openai-chat",
        "anthropic-messages",
        {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 99},
    ),
]


@pytest.mark.parametrize("source,target,usage", UNMAPPABLE_USAGE)
@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.asyncio
async def test_unrepresentable_usage_fails_bounded_without_semantic_loss(
    source, target, usage, streamed
):
    with pytest.raises(GatewayPublicError) as caught:
        if streamed:
            async for _ in translate_sse_stream(
                source, target, chunks(usage_stream(source, usage)), public_model="p"
            ):
                pass
        else:
            translate_response(source, target, response_with_usage(source, usage), public_model="p")
    assert len(caught.value.message) < 160 and "SECRET" not in caught.value.message


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["anthropic-messages", "gemini-generate"])
@pytest.mark.parametrize("composed", [False, True])
@pytest.mark.parametrize(
    "late_choice",
    [
        {"index": 0, "delta": {}, "finish_reason": "stop"},
        {"index": 0, "delta": {}, "finish_reason": "length"},
        {"index": 0, "delta": {}, "finish_reason": None},
        {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None},
    ],
)
async def test_chat_stream_rejects_duplicate_finish_and_postfinish_choices(
    target, composed, late_choice
):
    wire = (
        b'data: {"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}\n\n'
        + b"data: "
        + json.dumps({"choices": [late_choice]}).encode()
        + b"\n\ndata: [DONE]\n\n"
    )
    observer = StreamObserver("openai-chat", LimitsConfig(), time.monotonic() + 5)
    source = observer.observe(chunks(wire)) if composed else chunks(wire)
    received = []
    with pytest.raises(GatewayPublicError):
        async for part in translate_sse_stream("openai-chat", target, source, public_model="p"):
            received.append(part)
    assert b"message_stop" not in b"".join(received)
    assert b"finishReason" not in b"".join(received)
    assert observer.terminal != "success"


@pytest.mark.parametrize("source,target", list(REVERSE))
@pytest.mark.parametrize("usage", [[], "SECRET", 5, False])
def test_malformed_usage_envelope_is_not_silently_omitted(source, target, usage):
    with pytest.raises(GatewayPublicError):
        translate_response(source, target, response_with_usage(source, usage), public_model="p")


@pytest.mark.asyncio
async def test_bad_anthropic_usage_never_publishes_success_finish():
    received = []
    with pytest.raises(GatewayPublicError):
        async for part in translate_sse_stream(
            "anthropic-messages",
            "openai-chat",
            chunks(usage_stream("anthropic-messages", {"SECRET_UNKNOWN_UNIT": 3})),
            public_model="p",
        ):
            received.append(part)
    assert b'"finish_reason":"stop"' not in b"".join(received)


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["anthropic-messages", "gemini-generate"])
async def test_chat_usage_snapshots_preserve_previously_observed_categories(target):
    wire = (
        b'data: {"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}],"usage":{"prompt_tokens":5,"prompt_tokens_details":{"cached_tokens":3}}}\n\n'
        b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        b'data: {"choices":[],"usage":{"completion_tokens":2,"total_tokens":7}}\n\ndata: [DONE]\n\n'
    )
    output = b"".join(
        [
            part
            async for part in translate_sse_stream(
                "openai-chat", target, chunks(wire), public_model="p"
            )
        ]
    )
    events = [
        json.loads(line[6:]) for line in output.decode().splitlines() if line.startswith("data: ")
    ]
    if target == "gemini-generate":
        assert events[-1]["usageMetadata"] == {
            "promptTokenCount": 5,
            "candidatesTokenCount": 2,
            "totalTokenCount": 7,
            "cachedContentTokenCount": 3,
        }
    else:
        assert events[-2]["usage"] == {
            "input_tokens": 2,
            "output_tokens": 2,
            "cache_read_input_tokens": 3,
        }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source,target,wire",
    [
        (
            "openai-chat",
            "anthropic-messages",
            b'data: {"choices":[{"delta":{"refusal":"REFUSAL_SECRET"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
        ),
        (
            "anthropic-messages",
            "openai-chat",
            b'data: {"type":"message_start","message":{"content":[]}}\n\ndata: {"type":"message_delta","delta":{"stop_reason":"refusal"}}\n\ndata: {"type":"message_stop"}\n\n',
        ),
    ],
)
async def test_public_stream_helper_rejects_refusal_without_content_leak(source, target, wire):
    output = []
    with pytest.raises(GatewayPublicError):
        async for part in translate_sse_stream(source, target, chunks(wire), public_model="p"):
            output.append(part)
    assert b"REFUSAL_SECRET" not in b"".join(output)


def snapshot_stream(source, initial, later):
    if source == "openai-chat":
        events = [
            {
                "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
                "usage": initial,
            },
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": later},
        ]
    else:
        events = [
            {"type": "message_start", "message": {"content": [], "usage": initial}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": later},
            {"type": "message_stop"},
        ]
    return b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events) + (
        b"data: [DONE]\n\n" if source == "openai-chat" else b""
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("composed", [False, True])
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("nullable", ["absent", "detail", "unit"])
@pytest.mark.parametrize("target", ["anthropic-messages", "gemini-generate"])
async def test_nullable_cached_snapshots_retain_known_units(target, nullable, partial, composed):
    initial = {"prompt_tokens_details": {"cached_tokens": 3}}
    later = {}
    if not partial:
        initial["prompt_tokens"] = 5
        later.update(prompt_tokens=5, completion_tokens=2, total_tokens=7)
    if nullable != "absent":
        later["prompt_tokens_details"] = None if nullable == "detail" else {"cached_tokens": None}
    observer = StreamObserver("openai-chat", LimitsConfig(), time.monotonic() + 5)
    source = chunks(snapshot_stream("openai-chat", initial, later))
    output = b"".join(
        [
            part
            async for part in translate_sse_stream(
                "openai-chat",
                target,
                observer.observe(source) if composed else source,
                public_model="p",
            )
        ]
    )
    events = [json.loads(line[6:]) for line in output.splitlines() if line.startswith(b"data: ")]
    if target == "gemini-generate":
        assert events[-1]["candidates"] == [{"index": 0, "finishReason": "STOP"}]
        assert events[-1]["usageMetadata"] == (
            {"cachedContentTokenCount": 3}
            if partial
            else {
                "promptTokenCount": 5,
                "candidatesTokenCount": 2,
                "totalTokenCount": 7,
                "cachedContentTokenCount": 3,
            }
        )
    else:
        assert events[-1]["type"] == "message_stop"
        assert events[-2]["usage"] == (
            {"cache_read_input_tokens": 3}
            if partial
            else {"input_tokens": 2, "output_tokens": 2, "cache_read_input_tokens": 3}
        )
    if composed:
        assert observer.terminal == "success"


@pytest.mark.asyncio
@pytest.mark.parametrize("composed", [False, True])
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("nullable", ["absent", "detail", "unit"])
async def test_nullable_reasoning_snapshots_retain_known_units(nullable, partial, composed):
    initial = {"completion_tokens_details": {"reasoning_tokens": 3}}
    later = {}
    if not partial:
        initial["completion_tokens"] = 5
        later.update(prompt_tokens=2, completion_tokens=5, total_tokens=7)
    if nullable != "absent":
        later["completion_tokens_details"] = (
            None if nullable == "detail" else {"reasoning_tokens": None}
        )
    observer = StreamObserver("openai-chat", LimitsConfig(), time.monotonic() + 5)
    source = chunks(snapshot_stream("openai-chat", initial, later))
    output = b"".join(
        [
            part
            async for part in translate_sse_stream(
                "openai-chat",
                "gemini-generate",
                observer.observe(source) if composed else source,
                public_model="p",
            )
        ]
    )
    event = json.loads(output.splitlines()[-2][6:])
    assert event["candidates"] == [{"index": 0, "finishReason": "STOP"}]
    assert event["usageMetadata"] == (
        {"thoughtsTokenCount": 3}
        if partial
        else {
            "promptTokenCount": 2,
            "candidatesTokenCount": 2,
            "totalTokenCount": 7,
            "thoughtsTokenCount": 3,
        }
    )
    if composed:
        assert observer.terminal == "success"


@pytest.mark.asyncio
@pytest.mark.parametrize("composed", [False, True])
@pytest.mark.parametrize(
    "source,target,initial,later,expected",
    [
        (
            "openai-chat",
            "gemini-generate",
            {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
            {"promptTokenCount": 5, "candidatesTokenCount": 2, "totalTokenCount": 7},
        ),
        (
            "openai-chat",
            "anthropic-messages",
            {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
            {"input_tokens": 5, "output_tokens": 2},
        ),
        (
            "anthropic-messages",
            "openai-chat",
            {"input_tokens": 2, "cache_read_input_tokens": 3},
            {"input_tokens": None, "cache_read_input_tokens": None, "output_tokens": 2},
            {
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7,
                "prompt_tokens_details": {"cached_tokens": 3},
            },
        ),
        (
            "anthropic-messages",
            "openai-chat",
            {"cache_read_input_tokens": 3},
            {"cache_read_input_tokens": None},
            {"prompt_tokens_details": {"cached_tokens": 3}},
        ),
    ],
)
async def test_nullable_root_snapshots_retain_known_units(
    source, target, initial, later, expected, composed
):
    observer = StreamObserver(source, LimitsConfig(), time.monotonic() + 5)
    stream = chunks(snapshot_stream(source, initial, later))
    output = b"".join(
        [
            part
            async for part in translate_sse_stream(
                source, target, observer.observe(stream) if composed else stream, public_model="p"
            )
        ]
    )
    events = [
        json.loads(line[6:])
        for line in output.splitlines()
        if line.startswith(b"data: ") and line != b"data: [DONE]"
    ]
    key = "usageMetadata" if target == "gemini-generate" else "usage"
    assert [event[key] for event in events if event.get(key)][-1] == expected
    if composed:
        assert observer.terminal == "success"


@pytest.mark.asyncio
@pytest.mark.parametrize("composed", [False, True])
@pytest.mark.parametrize(
    "source,target,initial,later",
    [
        ("openai-chat", "gemini-generate", {"prompt_tokens": 5}, {"prompt_tokens": 4}),
        ("openai-chat", "anthropic-messages", {"completion_tokens": 3}, {"completion_tokens": 2}),
        (
            "openai-chat",
            "gemini-generate",
            {"prompt_tokens_details": {"cached_tokens": 3}},
            {"prompt_tokens_details": {"cached_tokens": 2}},
        ),
        (
            "openai-chat",
            "anthropic-messages",
            {"prompt_tokens_details": {"cached_tokens": 3}},
            {"prompt_tokens_details": {"cached_tokens": 0}},
        ),
        (
            "openai-chat",
            "gemini-generate",
            {"completion_tokens_details": {"reasoning_tokens": 3}},
            {"completion_tokens_details": {"reasoning_tokens": 2}},
        ),
        (
            "anthropic-messages",
            "openai-chat",
            {"cache_read_input_tokens": 3},
            {"cache_read_input_tokens": 2},
        ),
    ],
)
async def test_decreasing_usage_snapshots_fail_bounded_before_success(
    source, target, initial, later, composed
):
    observer = StreamObserver(source, LimitsConfig(), time.monotonic() + 5)
    stream = chunks(snapshot_stream(source, initial, later))
    output = []
    with pytest.raises(GatewayPublicError) as caught:
        async for part in translate_sse_stream(
            source, target, observer.observe(stream) if composed else stream, public_model="p"
        ):
            output.append(part)
    assert len(caught.value.message) < 160
    wire = b"".join(output)
    assert b"message_stop" not in wire and b"finishReason" not in wire
    assert b'"finish_reason":"stop"' not in wire
    assert observer.terminal != "success"


@pytest.mark.asyncio
@pytest.mark.parametrize("composed", [False, True])
@pytest.mark.parametrize("target", ["anthropic-messages", "gemini-generate"])
async def test_increasing_usage_snapshots_use_latest_cumulative_counts(target, composed):
    initial = {
        "prompt_tokens": 4,
        "completion_tokens": 1,
        "total_tokens": 5,
        "prompt_tokens_details": {"cached_tokens": 2},
    }
    later = {
        "prompt_tokens": 5,
        "completion_tokens": 2,
        "total_tokens": 7,
        "prompt_tokens_details": {"cached_tokens": 3},
    }
    observer = StreamObserver("openai-chat", LimitsConfig(), time.monotonic() + 5)
    stream = chunks(snapshot_stream("openai-chat", initial, later))
    output = b"".join(
        [
            part
            async for part in translate_sse_stream(
                "openai-chat",
                target,
                observer.observe(stream) if composed else stream,
                public_model="p",
            )
        ]
    )
    events = [json.loads(line[6:]) for line in output.splitlines() if line.startswith(b"data: ")]
    if target == "gemini-generate":
        assert events[-1]["usageMetadata"] == {
            "promptTokenCount": 5,
            "candidatesTokenCount": 2,
            "totalTokenCount": 7,
            "cachedContentTokenCount": 3,
        }
    else:
        assert events[-2]["usage"] == {
            "input_tokens": 2,
            "output_tokens": 2,
            "cache_read_input_tokens": 3,
        }
    if composed:
        assert observer.terminal == "success"


@pytest.mark.parametrize("served", [False, True])
@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("protocol", ["openai-chat", "anthropic-messages", "gemini-generate"])
def test_enormous_temperature_is_bounded_request_error(translated_client, protocol, sign, served):
    temperature = sign * 10**400
    if protocol == "gemini-generate":
        path = "/v1beta/models/public:generateContent"
        payload = {"contents": [], "generationConfig": {"temperature": temperature}}
    else:
        path = "/v1/chat/completions" if protocol == "openai-chat" else "/v1/messages"
        payload = {"model": "public", "messages": [], "max_tokens": 8, "temperature": temperature}
    if not served:
        with pytest.raises(GatewayPublicError) as caught:
            translate(
                protocol,
                "anthropic-messages" if protocol == "openai-chat" else "openai-chat",
                payload,
            )
        assert caught.value.status_code == 400 and len(caught.value.message) < 160
        assert str(temperature) not in caught.value.message
        return
    client, counts = translated_client(protocol)
    response = client.post(path, json=payload)
    assert response.status_code == 400
    assert len(response.content) < 512 and str(temperature) not in response.text
    assert counts == {"broker": 0, "upstream": 0}
    error = response.json()["error"]
    if protocol == "gemini-generate":
        assert error["code"] == 400 and error["status"] == "INVALID_ARGUMENT"
    else:
        assert error["code"] == "gateway_unsupported_capability"
        if protocol == "anthropic-messages":
            assert response.json()["type"] == "error"
            assert error["type"] == "invalid_request_error"
        else:
            assert error["type"] == "gateway_error"


@pytest.mark.parametrize("temperature", [0, 0.5, 2])
def test_supported_gemini_temperature_range_is_preserved(temperature):
    result = translate(
        "gemini-generate",
        "openai-chat",
        {"contents": [], "generationConfig": {"temperature": temperature}},
    )
    assert result["temperature"] == temperature
