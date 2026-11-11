"""web_search_call history on a Responses request that declares no web_search tool.

Codex on a custom provider compacts locally: it resends the whole history with
``tools: []``. When that history holds a hosted ``web_search_call`` item, the
ChatGPT backend answers 200 and then the in-band error "response protection is
unavailable", and the Codex session never answers again. The proxy drops those
items from such a request (HTTP and WebSocket) and leaves every other body
byte-identical.
"""

from __future__ import annotations

import asyncio
import copy
import json
import sys
from typing import Any
from unittest.mock import patch

import pytest

from headroom.proxy.handlers.openai import (
    _drop_undeclared_web_search_calls,
    _drop_undeclared_web_search_calls_from_frame,
)
from tests.test_openai_codex_ws_lifecycle import (
    _codex_lite_headers,
    _DummyOpenAIHandler,
    _FakeUpstream,
    _FakeWebSocket,
    _make_fake_websockets_module,
)
from tests.test_proxy_byte_faithful_forwarding import (
    _make_no_optimize_app,
    _start_proxy_log_capture,
    _stop_proxy_log_capture,
)


def _user(text: str) -> dict[str, Any]:
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def _search(n: int) -> dict[str, Any]:
    return {
        "type": "web_search_call",
        "id": f"ws_{n}",
        "status": "completed",
        "action": {"type": "search", "query": f"query {n}"},
    }


def _answer(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }


def _compaction_body(**overrides: Any) -> dict[str, Any]:
    """A Codex local-compaction request: full history, ``tools: []``."""
    body: dict[str, Any] = {
        "model": "gpt-5.5",
        "instructions": "Summarize the conversation.",
        "input": [
            _user("look this up"),
            {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "enc"},
            _search(1),
            _search(2),
            _answer("found it"),
            _user("Summarize the conversation so far."),
        ],
        "tools": [],
        "stream": True,
        "store": False,
    }
    body.update(overrides)
    return body


def _types(body: dict[str, Any]) -> list[str]:
    return [item["type"] for item in body["input"]]


# --- the decision ----------------------------------------------------------


@pytest.mark.parametrize(
    "tools",
    [
        pytest.param([], id="empty"),
        pytest.param(None, id="absent"),
        pytest.param([{"type": "function", "name": "shell"}], id="function-only"),
        pytest.param([{"type": "function", "name": "web_search"}], id="function-named-web_search"),
    ],
)
def test_drops_web_search_calls_when_no_web_search_tool_is_declared(tools) -> None:
    body = _compaction_body()
    if tools is None:
        del body["tools"]
    else:
        body["tools"] = tools

    assert _drop_undeclared_web_search_calls(body, request_id="r1") == 2
    # Only the search actions leave; the answer and reasoning stay in order.
    assert _types(body) == ["message", "reasoning", "message", "message"]
    assert body["input"][2] == _answer("found it")


@pytest.mark.parametrize(
    "tool_type",
    ["web_search", "web_search_preview", "web_search_preview_2025_03_11", "web_search_2025_08_26"],
)
def test_keeps_web_search_calls_when_a_web_search_tool_is_declared(tool_type: str) -> None:
    body = _compaction_body(tools=[{"type": "function", "name": "shell"}, {"type": tool_type}])
    before = copy.deepcopy(body)

    assert _drop_undeclared_web_search_calls(body) == 0
    assert body == before


def test_keeps_web_search_calls_declared_through_codex_additional_tools() -> None:
    # Codex >= 0.149 sends tools as an ``additional_tools`` input item and no
    # top-level ``tools``; a normal turn there must not lose its history.
    body = _compaction_body()
    del body["tools"]
    body["input"].insert(0, {"type": "additional_tools", "tools": [{"type": "web_search"}]})
    before = copy.deepcopy(body)

    assert _drop_undeclared_web_search_calls(body) == 0
    assert body == before


def test_leaves_unaffected_bodies_alone() -> None:
    no_search = _compaction_body()
    no_search["input"] = [item for item in no_search["input"] if item["type"] != "web_search_call"]
    continued = _compaction_body(previous_response_id="resp_1")
    string_input = _compaction_body(input="hello")
    for body in (no_search, continued, string_input):
        before = copy.deepcopy(body)
        assert _drop_undeclared_web_search_calls(body) == 0
        assert body == before


def test_kill_switch_disables_the_drop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_WEB_SEARCH_HISTORY", "0")
    body = _compaction_body()
    before = copy.deepcopy(body)

    assert _drop_undeclared_web_search_calls(body) == 0
    assert body == before


def test_frame_helper_returns_the_same_string_unless_it_drops() -> None:
    wrapped = json.dumps({"type": "response.create", "response": _compaction_body()})
    out = _drop_undeclared_web_search_calls_from_frame(wrapped)
    assert _types(json.loads(out)["response"]) == ["message", "reasoning", "message", "message"]

    flat = json.dumps({"type": "response.create", **_compaction_body()})
    assert "web_search_call" not in _drop_undeclared_web_search_calls_from_frame(flat)

    for raw in (
        json.dumps(
            {
                "type": "response.create",
                "response": _compaction_body(tools=[{"type": "web_search"}]),
            }
        ),
        json.dumps({"type": "response.cancel", "response": _compaction_body()}),
        json.dumps({"type": "response.create", "response": {"model": "m", "input": "hi"}}),
        "not json",
    ):
        assert _drop_undeclared_web_search_calls_from_frame(raw) is raw


def test_frame_helper_decodes_escaped_type_spellings() -> None:
    # json.loads reads "web_search_call" as web_search_call, and so does
    # the backend, so the prescreen must not skip it.
    escaped = (
        '{"type":"response.create","response":{"input":'
        '[{"type":"web_search\\u005fcall","id":"ws_1"}],"tools":[]}}'
    )
    assert json.loads(escaped)["response"]["input"][0]["type"] == "web_search_call"
    assert json.loads(_drop_undeclared_web_search_calls_from_frame(escaped))["response"] == {
        "input": [],
        "tools": [],
    }

    # Escapes alone change nothing: declared tool, no search history.
    for raw in (
        '{"type":"response.create","response":{"input":'
        '[{"type":"web_search\\u005fcall","id":"ws_1"}],"tools":[{"type":"web_search"}]}}',
        '{"type":"response.create","response":{"input":'
        '[{"type":"message","role":"user","content":"caf\\u00e9"}],"tools":[]}}',
    ):
        assert _drop_undeclared_web_search_calls_from_frame(raw) is raw


# --- HTTP /v1/responses ----------------------------------------------------

_CODEX_HEADERS = {
    "authorization": "Bearer test-token",
    "chatgpt-account-id": "acct_test",
    "originator": "codex_cli_rs",
    "content-type": "application/json",
}


def _post(
    body_bytes: bytes, *, stream: bool, extra_headers: dict[str, str] | None = None
) -> tuple[bytes, list[str]]:
    headers = {**_CODEX_HEADERS, **(extra_headers or {})}
    client, transport = _make_no_optimize_app()
    proxy_logger, handler, prev_level, records = _start_proxy_log_capture()
    try:
        if stream:
            with client.stream(
                "POST", "/v1/responses", headers=headers, content=body_bytes
            ) as response:
                assert response.status_code == 200
                for _ in response.iter_bytes():
                    pass
        else:
            response = client.post("/v1/responses", headers=headers, content=body_bytes)
            assert response.status_code == 200, response.text
    finally:
        _stop_proxy_log_capture(proxy_logger, handler, prev_level)
    assert transport.captured_body is not None
    return transport.captured_body, [record.getMessage() for record in records]


@pytest.mark.parametrize("stream", [True, False], ids=["stream", "buffered"])
def test_http_compaction_request_reaches_upstream_without_web_search_calls(stream: bool) -> None:
    body = _compaction_body(stream=stream)
    sent, logs = _post(json.dumps(body, indent=2).encode(), stream=stream)

    wire = json.loads(sent)
    assert _types(wire) == ["message", "reasoning", "message", "message"]
    expected = copy.deepcopy(body)
    expected["input"] = [item for item in body["input"] if item["type"] != "web_search_call"]
    assert wire == expected
    assert any("event=web_search_history" in m and "dropped=2" in m for m in logs), logs
    assert any(
        "event=outbound_request" in m and "responses_web_search_history" in m for m in logs
    ), logs


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(_compaction_body(tools=[{"type": "web_search"}]), id="tool-declared"),
        pytest.param(
            {**_compaction_body(), "input": [_user("hi"), _answer("hello")]}, id="no-search-history"
        ),
    ],
)
def test_http_unaffected_request_is_forwarded_byte_identical(body: dict[str, Any]) -> None:
    body_bytes = json.dumps(body, indent=2).encode()
    sent, logs = _post(body_bytes, stream=True)

    assert sent == body_bytes
    assert not any("event=web_search_history" in m for m in logs)


@pytest.mark.parametrize("stream", [True, False], ids=["stream", "buffered"])
def test_http_bypass_request_is_forwarded_byte_identical(stream: bool) -> None:
    # A bypass client asked Headroom to stay out of its bytes, so even a body
    # the backend will refuse goes out exactly as sent.
    body_bytes = json.dumps(_compaction_body(stream=stream), indent=2).encode()
    sent, logs = _post(body_bytes, stream=stream, extra_headers={"x-headroom-bypass": "true"})

    assert sent == body_bytes
    assert not any("event=web_search_history" in m for m in logs)


# --- Codex WebSocket /v1/responses ------------------------------------------


def _frame(body: dict[str, Any]) -> str:
    return json.dumps({"type": "response.create", "response": body})


def _upstream_events() -> list[str]:
    return [
        json.dumps({"type": "response.created", "response": {"id": "r_1"}}),
        json.dumps({"type": "response.completed", "response": {"id": "r_1"}}),
    ]


@pytest.mark.asyncio
async def test_ws_first_frame_reaches_upstream_without_web_search_calls() -> None:
    upstream = _FakeUpstream(_upstream_events())
    client_ws = _FakeWebSocket(frames=[_frame(_compaction_body())])

    with patch.dict(sys.modules, {"websockets": _make_fake_websockets_module(upstream)}):
        await _DummyOpenAIHandler().handle_openai_responses_ws(client_ws)

    assert len(upstream.sent) == 1
    assert _types(json.loads(upstream.sent[0])["response"]) == [
        "message",
        "reasoning",
        "message",
        "message",
    ]


@pytest.mark.asyncio
async def test_ws_bypass_session_frames_reach_upstream_unchanged() -> None:
    compaction = _frame(_compaction_body())
    upstream = _FakeUpstream([], hold_after_events=True)
    client_ws = _FakeWebSocket(
        frames=[compaction, compaction],
        headers={"authorization": "Bearer test", "x-headroom-bypass": "true"},
        hold_after_initial=True,
    )

    async def _disconnect_soon() -> None:
        await asyncio.sleep(0.05)
        client_ws.trigger_disconnect()

    with patch.dict(sys.modules, {"websockets": _make_fake_websockets_module(upstream)}):
        trigger = asyncio.create_task(_disconnect_soon())
        try:
            await asyncio.wait_for(
                _DummyOpenAIHandler().handle_openai_responses_ws(client_ws), timeout=2.0
            )
        finally:
            trigger.cancel()

    # First frame and a later frame both go out as the client sent them.
    assert upstream.sent == [compaction, compaction]


@pytest.mark.asyncio
@pytest.mark.parametrize("bypass", [False, True], ids=["normal", "bypass"])
async def test_ws_escaped_web_search_call_frame(bypass: bool) -> None:
    escaped = (
        '{"type":"response.create","response":{"model":"gpt-5-codex","input":'
        '[{"type":"web_search\\u005fcall","id":"ws_1"}],"tools":[]}}'
    )
    headers = {"authorization": "Bearer test"}
    if bypass:
        headers["x-headroom-bypass"] = "true"
    upstream = _FakeUpstream([], hold_after_events=True)
    client_ws = _FakeWebSocket(frames=[escaped], headers=headers, hold_after_initial=True)

    async def _disconnect_soon() -> None:
        await asyncio.sleep(0.05)
        client_ws.trigger_disconnect()

    with patch.dict(sys.modules, {"websockets": _make_fake_websockets_module(upstream)}):
        trigger = asyncio.create_task(_disconnect_soon())
        try:
            await asyncio.wait_for(
                _DummyOpenAIHandler().handle_openai_responses_ws(client_ws), timeout=2.0
            )
        finally:
            trigger.cancel()

    assert len(upstream.sent) == 1
    if bypass:
        assert upstream.sent[0] == escaped
    else:
        assert json.loads(upstream.sent[0])["response"]["input"] == []


@pytest.mark.asyncio
async def test_ws_later_frame_reaches_upstream_without_web_search_calls() -> None:
    normal_turn = _frame(_compaction_body(tools=[{"type": "web_search"}]))
    compaction = _frame(_compaction_body())
    upstream = _FakeUpstream([], hold_after_events=True)
    client_ws = _FakeWebSocket(frames=[normal_turn, compaction], hold_after_initial=True)

    async def _disconnect_soon() -> None:
        await asyncio.sleep(0.05)
        client_ws.trigger_disconnect()

    with patch.dict(sys.modules, {"websockets": _make_fake_websockets_module(upstream)}):
        trigger = asyncio.create_task(_disconnect_soon())
        try:
            await asyncio.wait_for(
                _DummyOpenAIHandler().handle_openai_responses_ws(client_ws), timeout=2.0
            )
        finally:
            trigger.cancel()

    assert len(upstream.sent) == 2
    # The turn that declares the tool goes out as the client sent it.
    assert upstream.sent[0] == normal_turn
    assert "web_search_call" not in upstream.sent[1]


@pytest.mark.asyncio
async def test_ws_http_fallback_gets_the_frame_without_web_search_calls() -> None:
    fake_ws_mod = _make_fake_websockets_module(None, connect_error=RuntimeError("HTTP 500"))
    client_ws = _FakeWebSocket(
        frames=[_frame(_compaction_body())], headers=_codex_lite_headers(chatgpt=True)
    )
    handler = _DummyOpenAIHandler()
    fallback_frames: list[str] = []

    async def _fallback(websocket, body, first_msg_raw, upstream_headers, request_id):
        fallback_frames.append(first_msg_raw)
        return 0, 0, 0, 0, 0

    handler._ws_http_fallback = _fallback  # type: ignore[assignment]

    with patch.dict(sys.modules, {"websockets": fake_ws_mod}):
        await handler.handle_openai_responses_ws(client_ws)

    assert len(fallback_frames) == 1
    assert "web_search_call" not in fallback_frames[0]
