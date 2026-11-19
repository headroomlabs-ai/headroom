"""A Claude-format request served by an OpenAI model (route advice -> LiteLLMBackend).

Contract pinned here:
- the thinking effort a Claude request asks for reaches OpenAI as
  ``reasoning_effort``; a request that asks for none sends none;
- Anthropic server-side tools (tool search, web search) never reach OpenAI as
  parameterless functions, and Headroom does not inject its own tool-search tool
  into a request that route advice sends to another vendor;
- a tool-call round trip and a streamed tool call come back in Anthropic shape.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from tests._dotenv import importorskip_no_env_leak

importorskip_no_env_leak("litellm")

from headroom.backends.base import BackendResponse  # noqa: E402
from headroom.backends.litellm import LiteLLMBackend  # noqa: E402
from headroom.proxy.route_advice import BackendResolver, RouteAdvice  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402


def _tool(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "description": f"Tool {name}.",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
    }


USER_TURN = {"role": "user", "content": "What is the weather in Paris?"}
SERVER_TOOLS = [
    {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"},
    {"type": "web_search_20250305", "name": "web_search"},
]


def _usage() -> SimpleNamespace:
    return SimpleNamespace(prompt_tokens=20, completion_tokens=7)


def _completion(*, content: str | None = None, tool_call: tuple[str, str, str] | None = None):
    tool_calls = None
    if tool_call:
        call_id, name, arguments = tool_call
        tool_calls = [
            SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))
        ]
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    finish = "tool_calls" if tool_call else "stop"
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish)], usage=_usage()
    )


@pytest.mark.asyncio
async def test_tool_call_round_trip_on_an_openai_model() -> None:
    backend = LiteLLMBackend(provider="openai")
    body: dict[str, Any] = {
        "model": "gpt-6.1-sol",
        "max_tokens": 4096,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "xhigh"},
        "tools": [*SERVER_TOOLS, _tool("get_weather")],
        "messages": [USER_TURN],
    }
    upstream = AsyncMock(
        side_effect=[
            _completion(tool_call=("call_1", "get_weather", '{"city": "Paris"}')),
            _completion(content="18C, light rain."),
        ]
    )

    with patch("headroom.backends.litellm.acompletion", upstream):
        first = await backend.send_message(body, {"x-api-key": "sk-ant-not-for-openai"})
        tool_use = first.body["content"][0]
        body["messages"] = [
            USER_TURN,
            {"role": "assistant", "content": first.body["content"]},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": tool_use["id"], "content": "18C rain"}
                ],
            },
        ]
        second = await backend.send_message(body, {"x-api-key": "sk-ant-not-for-openai"})

    assert first.body["stop_reason"] == "tool_use"
    assert tool_use == {
        "type": "tool_use",
        "id": "call_1",
        "name": "get_weather",
        "input": {"city": "Paris"},
    }
    assert second.body["content"] == [{"type": "text", "text": "18C, light rain."}]
    assert second.body["stop_reason"] == "end_turn"

    first_call, second_call = (call.kwargs for call in upstream.call_args_list)
    assert first_call["model"] == "openai/gpt-6.1-sol"
    assert first_call["reasoning_effort"] == "high"  # xhigh clamps to what OpenAI takes
    assert [t["function"]["name"] for t in first_call["tools"]] == ["get_weather"]
    assert "api_key" not in first_call  # an sk-ant- key cannot authenticate OpenAI
    assert second_call["messages"][1:] == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "18C rain"},
    ]


@pytest.mark.parametrize(
    ("request_fields", "expected"),
    [
        ({}, None),
        ({"thinking": {"type": "disabled"}}, None),
        ({"thinking": {"type": "adaptive"}}, None),
        ({"thinking": {"type": "enabled", "budget_tokens": 1024}}, "low"),
        ({"thinking": {"type": "enabled", "budget_tokens": 2048}}, "medium"),
        ({"thinking": {"type": "enabled", "budget_tokens": 31999}}, "high"),
        ({"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}}, "low"),
    ],
)
@pytest.mark.asyncio
async def test_reasoning_effort_follows_the_claude_thinking_config(
    request_fields: dict[str, Any], expected: str | None
) -> None:
    backend = LiteLLMBackend(provider="openai")
    body = {"model": "gpt-6-luna", "messages": [USER_TURN], **request_fields}
    upstream = AsyncMock(return_value=_completion(content="ok"))

    with patch("headroom.backends.litellm.acompletion", upstream):
        await backend.send_message(body, {})

    assert upstream.call_args.kwargs.get("reasoning_effort") == expected
    assert "thinking" not in upstream.call_args.kwargs


@pytest.mark.asyncio
async def test_streamed_tool_call_arrives_as_anthropic_events() -> None:
    backend = LiteLLMBackend(provider="openai")

    def chunk(*, tool_call: Any = None, finish: str | None = None, usage: Any = None):
        delta = SimpleNamespace(content=None, tool_calls=[tool_call] if tool_call else None)
        return SimpleNamespace(
            choices=[SimpleNamespace(delta=delta, finish_reason=finish)], usage=usage
        )

    async def chunks():
        opening = SimpleNamespace(name="get_weather", arguments="")
        yield chunk(tool_call=SimpleNamespace(index=0, id="call_1", function=opening))
        for piece in ('{"city":', ' "Paris"}'):
            function = SimpleNamespace(name=None, arguments=piece)
            yield chunk(tool_call=SimpleNamespace(index=0, id=None, function=function))
        yield chunk(finish="tool_calls", usage=_usage())

    upstream = AsyncMock(return_value=chunks())
    body = {
        "model": "gpt-6-luna",
        "stream": True,
        "thinking": {"type": "enabled", "budget_tokens": 8000},
        "tools": [*SERVER_TOOLS, _tool("get_weather")],
        "messages": [USER_TURN],
    }

    with patch("headroom.backends.litellm.acompletion", upstream):
        events = [event.data async for event in backend.stream_message(body, {})]

    assert upstream.call_args.kwargs["reasoning_effort"] == "high"
    assert [t["function"]["name"] for t in upstream.call_args.kwargs["tools"]] == ["get_weather"]
    assert [event["type"] for event in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[1]["content_block"] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "get_weather",
        "input": {},
    }
    assert "".join(e["delta"]["partial_json"] for e in events[2:4]) == '{"city": "Paris"}'
    assert events[5]["delta"]["stop_reason"] == "tool_use"
    assert events[5]["usage"] == {"output_tokens": 7, "input_tokens": 20}


def test_headroom_does_not_inject_tool_search_into_a_cross_vendor_request(monkeypatch) -> None:  # noqa: ANN001
    received: list[dict[str, Any]] = []

    class _Backend:
        name = "stub-openai"

        async def send_message(self, body: dict[str, Any], headers: dict[str, str]):
            received.append(body)
            return BackendResponse(
                body={
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "model": body["model"],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
                status_code=200,
                headers={"content-type": "application/json"},
            )

    monkeypatch.setenv("HEADROOM_TOOL_SEARCH", "1")
    monkeypatch.setattr(BackendResolver, "_build", lambda self, provider: _Backend())
    app = create_app(ProxyConfig(optimize=False, cache_enabled=False, rate_limit_enabled=False))

    @app.middleware("http")
    async def publish_advice(request, call_next):  # noqa: ANN001, ANN202
        request.state.headroom_route = RouteAdvice(model="gpt-6-luna", provider="openai")
        return await call_next(request)

    # Enough tools that the injector would defer them on the direct Anthropic path.
    tools = [_tool(f"mcp__srv__tool_{i}") for i in range(40)]
    with TestClient(app) as client:
        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-5",
                "max_tokens": 64,
                "tools": tools,
                "messages": [USER_TURN],
            },
            headers={"x-api-key": "sk-ant-test", "anthropic-version": "2023-06-01"},
        )

    assert response.status_code == 200, response.text
    assert received[0]["model"] == "gpt-6-luna"
    assert received[0]["tools"] == tools
