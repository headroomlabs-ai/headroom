"""``TurnContext.conversation``: one opaque key per client conversation.

Hooks that pin wording per conversation (headroom-condense) key on it, so two
sessions that open with the same prompt must get different keys, and every
turn of one conversation, retries and incremental Responses bodies included,
the same key.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import respx

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import headroom.proxy.handlers.openai as openai_module  # noqa: E402
from headroom.proxy.conversation_savings import savings_conversation_key  # noqa: E402
from headroom.proxy.loopback_guard import require_loopback  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402
from headroom.proxy.turn_hooks import clear_turn_hooks, register_turn_hook  # noqa: E402
from tests.test_openai_codex_ws_lifecycle import (  # noqa: E402
    _DummyOpenAIHandler,
    _FakeUpstream,
    _FakeWebSocket,
    _make_fake_websockets_module,
)


class _Recorder:
    name = "conversation_recorder"
    stream_safe = True

    def __init__(self) -> None:
        self.seen: list[str | None] = []

    def on_request(self, ctx: Any) -> None:
        self.seen.append(ctx.conversation)


@pytest.fixture
def recorder(monkeypatch):
    # litellm fetches its price map on import unless told not to; respx would
    # reject that request.
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    clear_turn_hooks()
    hook = _Recorder()
    register_turn_hook(hook)
    yield hook
    clear_turn_hooks()


def _app(*, optimize: bool):
    app = create_app(
        ProxyConfig(
            optimize=optimize,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
        )
    )
    app.dependency_overrides[require_loopback] = lambda: None
    return app


PROMPT = [{"role": "user", "content": "fix the bug"}]


@respx.mock
def test_claude_code_sessions_with_the_same_prompt_get_different_conversations(recorder):
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-4-5",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
        )
    )
    body = {"model": "claude-haiku-4-5", "max_tokens": 16, "messages": PROMPT}
    sessions = [
        {"x-claude-code-session-id": "s-a"},
        {"x-claude-code-session-id": "s-a"},  # a retry, or the next turn
        {"x-claude-code-session-id": "s-b"},
        {"x-headroom-session-id": "s-a"},
        {},
    ]
    with TestClient(_app(optimize=False)) as client:
        for headers in sessions:
            r = client.post(
                "/v1/messages", json=body, headers={"x-api-key": "sk-ant-test", **headers}
            )
            assert r.status_code == 200, r.text
    a, a_again, b, headroom_a, none = recorder.seen
    assert a == a_again and a is not None
    assert b not in (None, a)
    # Namespaced by source, and never the raw client id.
    assert headroom_a not in (None, a)
    assert "s-a" not in (a, headroom_a)
    assert none is None


@respx.mock
def test_responses_http_turns_of_one_thread_share_a_conversation(recorder):
    respx.post("https://api.openai.com/v1/responses").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "resp_1",
                "output": [{"type": "message", "role": "assistant", "content": []}],
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
        )
    )

    def body(thread: str, **extra: Any) -> dict[str, Any]:
        return {
            "model": "gpt-4o",
            "input": [{"type": "message", "role": "user", "content": "fix the bug"}],
            "stream": False,
            "client_metadata": {"thread_id": thread},
            **extra,
        }

    with TestClient(_app(optimize=True)) as client:
        for sent in (body("t-1"), body("t-1", previous_response_id="resp_0"), body("t-2")):
            r = client.post("/v1/responses", json=sent, headers={"authorization": "Bearer sk-t"})
            assert r.status_code == 200, r.text
    first, incremental, other = recorder.seen
    assert first == incremental and first is not None
    assert other not in (None, first)


@respx.mock
def test_responses_server_side_conversation_names_the_conversation(recorder):
    respx.post("https://api.openai.com/v1/responses").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "resp_1",
                "output": [{"type": "message", "role": "assistant", "content": []}],
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
        )
    )

    def body(conversation: Any) -> dict[str, Any]:
        return {
            "model": "gpt-4o",
            "input": [{"type": "message", "role": "user", "content": "fix the bug"}],
            "stream": False,
            "conversation": conversation,
        }

    sent = [body("conv_1"), body("conv_1"), body({"id": "conv_1"}), body("conv_2")]
    with TestClient(_app(optimize=True)) as client:
        for one in sent:
            r = client.post("/v1/responses", json=one, headers={"authorization": "Bearer sk-t"})
            assert r.status_code == 200, r.text
    first, again, as_object, other = recorder.seen
    assert first == again == as_object and first is not None
    assert other not in (None, first)
    # The savings funnel still books these per request: the provider holds the history.
    assert all(savings_conversation_key(one) is None for one in sent)


def _ws_frame(thread: str, **extra: Any) -> str:
    response = {
        "model": "gpt-5.4",
        "input": "fix the bug",
        "client_metadata": {"thread_id": thread},
        **extra,
    }
    return json.dumps({"type": "response.create", "response": response})


async def _ws_conversations(frames: list[str]) -> list[str | None]:
    """The ``conversation`` each WS frame's compression pass was handed."""
    seen: list[str | None] = []

    def _compress(payload, *, model, request_id, conversation=None, **_kw):
        seen.append(conversation)
        return payload, False, 0, [], "router_no_compression", 10, 10, 0

    client_ws = _FakeWebSocket(frames=frames, hold_after_initial=True)
    handler = _DummyOpenAIHandler()
    handler.config.optimize = True
    handler._compress_openai_responses_payload = _compress  # type: ignore[method-assign]
    upstream = _FakeUpstream([], hold_after_events=True)

    async def _disconnect() -> None:
        await asyncio.sleep(0.05)
        client_ws.trigger_disconnect()

    with (
        patch.object(openai_module, "COMPRESSION_TIMEOUT_SECONDS", 30.0),
        patch.dict(sys.modules, {"websockets": _make_fake_websockets_module(upstream)}),
    ):
        trigger = asyncio.create_task(_disconnect())
        try:
            await asyncio.wait_for(handler.handle_openai_responses_ws(client_ws), timeout=2.0)
        finally:
            trigger.cancel()
    return seen


@pytest.mark.asyncio
async def test_responses_ws_frames_of_one_thread_share_a_conversation():
    first, later = await _ws_conversations(
        [_ws_frame("t-1"), _ws_frame("t-1", previous_response_id="resp_1")]
    )
    (other,) = await _ws_conversations([_ws_frame("t-2")])
    assert first == later and first is not None
    assert other not in (None, first)


@respx.mock
def test_chat_completions_take_the_headroom_session_header(recorder):
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "c",
                "choices": [
                    {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1},
            },
        )
    )
    body = {"model": "gpt-4o", "messages": PROMPT}
    with TestClient(_app(optimize=False)) as client:
        for headers in ({"x-headroom-session-id": "h-1"}, {"x-headroom-session-id": "h-2"}, {}):
            r = client.post(
                "/v1/chat/completions",
                json=body,
                headers={"authorization": "Bearer sk-t", **headers},
            )
            assert r.status_code == 200, r.text
    one, two, none = recorder.seen
    assert one is not None and two not in (None, one)
    assert none is None
