"""Buffered CCR on streaming /v1/chat/completions (#3815).

Compression still runs on a streaming chat turn, so content carries retrieval
markers, but this path had no way to answer ``headroom_retrieve``. With
``ccr_buffered_chat_streaming`` on, a turn whose outgoing body carries a
redeemable marker runs as ``stream: false`` upstream through the same code as a
non-streaming turn, then goes back to the client as ``chat.completion.chunk``
frames. Everything else keeps its real stream.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.responses import Response  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from headroom.cache.compression_store import (  # noqa: E402
    get_compression_store,
    reset_compression_store,
)
from headroom.ccr.response_handler import CCRResponseHandler, StreamingCCRHandler  # noqa: E402
from headroom.ccr.tool_injection import CCR_TOOL_NAME  # noqa: E402
from headroom.proxy.handlers.openai import (  # noqa: E402
    _buffered_chat_reply_as_sse,
    _is_openai_chat_stream_ccr_eligible,
    _take_stream_options_for_buffered_chat,
)
from headroom.proxy.helpers import _reset_session_ccr_tracker_for_test  # noqa: E402
from headroom.proxy.server import ProxyConfig, _proxy_config_from_env, create_app  # noqa: E402

ORIGINAL = "row 1: the original uncompressed rows\nrow 2: more of them"
CLIENT_TOOL = {"type": "function", "function": {"name": "Read", "description": "read a file"}}
USAGE_FIRST = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
USAGE_FINAL = {"prompt_tokens": 400, "completion_tokens": 10, "total_tokens": 410}


@pytest.fixture(autouse=True)
def reset_store():
    reset_compression_store()
    _reset_session_ccr_tracker_for_test()
    yield
    reset_compression_store()
    _reset_session_ccr_tracker_for_test()


def _stored_hash() -> str:
    return get_compression_store().store(
        original=ORIGINAL,
        compressed="[2 items compressed to 0]",
        original_item_count=2,
        compressed_item_count=0,
    )


def _completion(message: dict[str, Any], finish_reason: str, usage: dict[str, int]) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": usage,
    }


def _tool_call(name: str, arguments: dict[str, Any], call_id: str = "call_1") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _retrieve_reply(hash_key: str) -> dict:
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [_tool_call(CCR_TOOL_NAME, {"hash": hash_key})],
    }
    return _completion(message, "tool_calls", USAGE_FIRST)


def _final_reply() -> dict:
    return _completion(
        {"role": "assistant", "content": "there are 2 rows"},
        "stop",
        USAGE_FINAL,
    )


def _messages(hash_key: str | None) -> list[dict[str, Any]]:
    content = "how many rows?"
    if hash_key is not None:
        content += f" [2 items compressed to 0. Retrieve more: hash={hash_key}]"
    return [{"role": "user", "content": content}]


def _request(hash_key: str | None, **overrides: Any) -> dict[str, Any]:
    return {
        "model": "gpt-4o",
        "messages": _messages(hash_key),
        "stream": True,
        "tools": [CLIENT_TOOL],
        **overrides,
    }


def _events(text: str) -> list[Any]:
    """Parse an SSE body into its JSON payloads, keeping the [DONE] sentinel."""
    events: list[Any] = []
    for line in text.splitlines():
        if line.startswith("data: "):
            data = line[len("data: ") :]
            events.append(data if data == "[DONE]" else json.loads(data))
    return events


def _chunks(text: str) -> list[dict[str, Any]]:
    return [event for event in _events(text) if event != "[DONE]"]


class _Proxy:
    """A live app whose upstream calls are served from a script."""

    def __init__(self, upstream: list[Any], **config: Any) -> None:
        self._remaining = list(upstream)
        self.sent: list[dict[str, Any]] = []
        self.sent_headers: list[dict[str, str]] = []
        self.outcomes: list[Any] = []
        self.plain_streams = 0
        settings = {
            "optimize": False,
            "cache_enabled": False,
            "rate_limit_enabled": False,
            "ccr_handle_responses": True,
            "ccr_inject_tool": True,
            "ccr_buffered_chat_streaming": True,
            **config,
        }
        self._app = create_app(ProxyConfig(**settings))

    async def _retry(self, method, url, headers, body, *args, **kwargs):  # noqa: ANN001
        self.sent.append(json.loads(json.dumps(body)))
        self.sent_headers.append(dict(headers))
        entry = self._remaining.pop(0)
        if isinstance(entry, Exception):
            raise entry
        status, payload = entry if isinstance(entry, tuple) else (200, entry)
        return httpx.Response(status, json=payload, headers={"content-type": "application/json"})

    def __enter__(self) -> _Proxy:
        self._client = TestClient(self._app)
        self._client.__enter__()
        proxy = self._client.app.state.proxy
        proxy._retry_request = self._retry
        self.proxy = proxy

        async def _plain_stream(*args: Any, **kwargs: Any) -> Response:
            self.plain_streams += 1
            return Response(b"data: plain\n\n", media_type="text/event-stream")

        proxy._stream_response = _plain_stream

        real_record = proxy._record_request_outcome

        async def _capture(outcome, *args, **kwargs):  # noqa: ANN001
            self.outcomes.append(outcome)
            return await real_record(outcome, *args, **kwargs)

        proxy._record_request_outcome = _capture
        return self

    def __exit__(self, *exc: Any) -> None:
        self._client.__exit__(*exc)

    def post(self, body: dict[str, Any], **extra_headers: str) -> httpx.Response:
        return self._client.post(
            "/v1/chat/completions",
            json=body,
            headers={
                "Authorization": "Bearer test-key",
                "x-api-key": "test-key",
                "Accept": "text/event-stream",
                **{name.replace("_", "-"): value for name, value in extra_headers.items()},
            },
        )


def test_flag_off_keeps_the_plain_stream_and_withholds_the_tool():
    hash_key = _stored_hash()
    with _Proxy([], ccr_buffered_chat_streaming=False) as proxy:
        resp = proxy.post(_request(hash_key))

    assert resp.text == "data: plain\n\n"
    assert proxy.plain_streams == 1
    assert proxy.sent == [], "flag off must not take the buffered path"


def test_flag_defaults_off():
    assert ProxyConfig().ccr_buffered_chat_streaming is False


def test_env_var_turns_the_flag_on(monkeypatch):
    monkeypatch.setenv("HEADROOM_CCR_BUFFERED_CHAT_STREAMING", "1")
    assert _proxy_config_from_env().ccr_buffered_chat_streaming is True


def test_retrieval_is_answered_and_the_client_reads_a_real_stream():
    hash_key = _stored_hash()
    with _Proxy([_retrieve_reply(hash_key), _final_reply()]) as proxy:
        resp = proxy.post(_request(hash_key))

    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert "content-length" not in resp.headers or int(resp.headers["content-length"]) == len(
        resp.content
    )
    events = _events(resp.text)
    assert events[-1] == "[DONE]"

    chunks = _chunks(resp.text)
    assert {chunk["object"] for chunk in chunks} == {"chat.completion.chunk"}
    choices = [chunk for chunk in chunks if chunk["choices"]]
    assert choices[0]["choices"][0]["delta"]["role"] == "assistant"
    text = "".join(chunk["choices"][0]["delta"].get("content") or "" for chunk in choices)
    assert text == "there are 2 rows"
    assert choices[-1]["choices"][0]["finish_reason"] == "stop"
    assert CCR_TOOL_NAME not in resp.text

    first, continuation = proxy.sent
    assert first["stream"] is False
    assert "stream_options" not in first, "OpenAI answers 400 to stream_options without a stream"
    assert CCR_TOOL_NAME in {tool["function"]["name"] for tool in first["tools"]}
    assert ORIGINAL.splitlines()[0] in json.dumps(continuation)
    assert {k.lower(): v for k, v in proxy.sent_headers[0].items()}["accept"] == "application/json"


def test_a_turn_with_no_marker_keeps_its_real_stream():
    """Buffering a turn retrieval cannot help only costs time to first byte (#3071)."""
    with _Proxy([]) as proxy:
        resp = proxy.post(_request(None))

    assert resp.text == "data: plain\n\n"
    assert proxy.plain_streams == 1
    assert proxy.sent == []


def test_a_foreign_marker_does_not_buffer():
    """Another context tool can use the same marker shape; only owned hashes count (#2836)."""
    with _Proxy([]) as proxy:
        resp = proxy.post(_request("0123456789abcdef0123456789abcdef"))

    assert resp.text == "data: plain\n\n"
    assert proxy.sent == []


def test_a_client_tool_call_still_ends_in_tool_calls():
    hash_key = _stored_hash()
    client_call = _tool_call("Read", {"file_path": "a.py"})
    reply = _completion(
        {"role": "assistant", "content": None, "tool_calls": [client_call]},
        "tool_calls",
        USAGE_FIRST,
    )
    with _Proxy([reply]) as proxy:
        resp = proxy.post(_request(hash_key))

    choices = [chunk["choices"][0] for chunk in _chunks(resp.text) if chunk["choices"]]
    assert choices[-1]["finish_reason"] == "tool_calls"
    tool_deltas = [
        choice["delta"]["tool_calls"][0] for choice in choices if "tool_calls" in choice["delta"]
    ]
    assert [delta["function"]["name"] for delta in tool_deltas] == ["Read"]
    assert tool_deltas[0]["index"] == 0


@pytest.mark.parametrize(
    ("stream_options", "expect_usage"),
    [
        (None, True),
        ({"include_usage": True}, True),
        ({"include_usage": False}, False),
        ({}, True),
    ],
)
def test_usage_chunk_follows_the_plain_stream_rule(stream_options, expect_usage):
    hash_key = _stored_hash()
    overrides = {} if stream_options is None else {"stream_options": stream_options}
    with _Proxy([_final_reply()]) as proxy:
        resp = proxy.post(_request(hash_key, **overrides))

    usage_chunks = [chunk for chunk in _chunks(resp.text) if chunk.get("usage")]
    if expect_usage:
        assert len(usage_chunks) == 1
        assert usage_chunks[0]["choices"] == []
        assert usage_chunks[0]["usage"] == USAGE_FINAL
        assert _chunks(resp.text)[-1] is not None and _chunks(resp.text)[-1]["usage"]
    else:
        assert usage_chunks == []
    assert "stream_options" not in proxy.sent[0]


def test_upstream_error_inside_the_grace_window_keeps_its_status():
    """A 429 must reach the client as a 429, not a 200 with no frames (#2997)."""
    hash_key = _stored_hash()
    error = {"error": {"message": "slow down", "type": "rate_limit_error"}}
    with _Proxy([(429, error)]) as proxy:
        resp = proxy.post(_request(hash_key))

    assert resp.status_code == 429
    assert resp.json()["error"]["message"] == "slow down"


class _Slow(_Proxy):
    async def _retry(self, method, url, headers, body, *args, **kwargs):  # noqa: ANN001
        import asyncio

        await asyncio.sleep(0.05)
        return await super()._retry(method, url, headers, body, *args, **kwargs)


def test_a_slow_turn_gets_a_first_byte_before_the_answer():
    """Past the grace window the stream commits and a heartbeat runs (#2465, #3079)."""
    hash_key = _stored_hash()
    with _Slow([_final_reply()], buffered_ccr_grace_seconds=0.01) as proxy:
        resp = proxy.post(_request(hash_key))

    assert resp.status_code == 200
    assert resp.text.startswith(": ping\n\n")
    assert _events(resp.text)[-1] == "[DONE]"


def test_an_error_after_the_commit_is_a_typed_stream_error():
    hash_key = _stored_hash()
    error = {"error": {"message": "slow down", "type": "rate_limit_error"}}
    with _Slow([(429, error)], buffered_ccr_grace_seconds=0.01) as proxy:
        resp = proxy.post(_request(hash_key))

    assert resp.status_code == 200, "the status was already committed"
    events = _events(resp.text)
    assert events[-1] == {"error": {"message": "slow down", "type": "rate_limit_error"}}


def test_a_transport_failure_inside_the_grace_window_is_a_clean_502():
    hash_key = _stored_hash()
    with _Proxy([httpx.ConnectError("boom")], ccr_buffered_chat_streaming=True) as proxy:
        before = proxy.proxy.metrics.requests_failed
        resp = proxy.post(_request(hash_key))
        after = proxy.proxy.metrics.requests_failed

    assert resp.status_code == 502
    assert resp.json()["error"]["type"] == "server_error"
    assert after == before + 1


def test_buffered_and_non_streaming_turns_account_identically():
    """Same turn both ways: the recorded numbers have to match (#3615, #3699)."""
    fields = (
        "status_code",
        "provider_input_tokens",
        "output_tokens",
        "tokens_saved",
        "cache_read_tokens",
        "cache_write_tokens",
        "uncached_input_tokens",
    )

    def run(stream: bool) -> dict[str, Any]:
        reset_compression_store()
        hash_key = _stored_hash()
        with _Proxy([_retrieve_reply(hash_key), _final_reply()]) as proxy:
            resp = proxy.post(_request(hash_key, stream=stream))
        assert resp.status_code == 200
        assert len(proxy.outcomes) == 1
        return {field: getattr(proxy.outcomes[0], field) for field in fields}

    buffered = run(stream=True)
    assert buffered == run(stream=False)
    assert buffered["provider_input_tokens"] == 500, "both upstream calls must be counted"


def test_tools_stay_byte_stable_across_a_session_that_mixes_modes():
    """The toggle from #3815: stream and non-stream turns must send the same tools."""
    hash_key = _stored_hash()
    upstream = [_final_reply(), _final_reply(), _final_reply()]
    messages = _messages(hash_key)
    with _Proxy(upstream) as proxy:
        for stream in (False, True, False):
            resp = proxy.post(_request(hash_key, messages=messages, stream=stream))
            assert resp.status_code == 200, resp.text

    assert len(proxy.sent) == 3
    tools = [json.dumps(body["tools"]) for body in proxy.sent]
    assert tools[0] == tools[1] == tools[2]
    assert CCR_TOOL_NAME in tools[0]


@pytest.mark.parametrize(
    ("status", "error_type"),
    [(429, "rate_limit_error"), (400, "invalid_request_error"), (500, "server_error")],
)
def test_failed_continuation_fails_closed_with_its_status(status, error_type):
    """The client must not get a tool it never declared, and backoff still has to fire."""
    hash_key = _stored_hash()
    upstream = [_retrieve_reply(hash_key), (status, {"error": {"message": "nope"}})]
    with _Proxy(upstream) as proxy:
        resp = proxy.post(_request(hash_key))

    assert resp.status_code == status
    assert CCR_TOOL_NAME not in resp.text
    assert resp.json()["error"]["type"] == error_type
    assert "nope" not in resp.text, "the upstream error body is untrusted and stays in the proxy"


def test_failed_continuation_after_the_commit_is_a_typed_stream_error():
    hash_key = _stored_hash()
    upstream = [_retrieve_reply(hash_key), (429, {"error": {"message": "nope"}})]
    with _Slow(upstream, buffered_ccr_grace_seconds=0.01) as proxy:
        resp = proxy.post(_request(hash_key))

    assert resp.status_code == 200, "the status was already committed"
    assert CCR_TOOL_NAME not in resp.text
    assert _events(resp.text)[-1]["error"]["type"] == "rate_limit_error"


def test_an_expired_marker_is_still_answered_while_the_tool_is_resident():
    """The hash is gone but the model can still ask for it, and the client cannot answer."""
    hash_key = _stored_hash()
    messages = _messages(hash_key)
    upstream = [_final_reply(), _retrieve_reply(hash_key), _final_reply()]
    with _Proxy(upstream) as proxy:
        first = proxy.post(_request(hash_key, messages=messages, stream=False))
        assert first.status_code == 200, first.text

        reset_compression_store()
        resp = proxy.post(_request(hash_key, messages=messages))

    assert resp.status_code == 200, resp.text
    assert proxy.plain_streams == 0
    assert CCR_TOOL_NAME not in resp.text
    assert "there are 2 rows" in resp.text


def test_a_bypassed_turn_keeps_its_stream():
    """Bypass skips injection, so there is nothing to answer and no reason to buffer."""
    hash_key = _stored_hash()
    with _Proxy([]) as proxy:
        resp = proxy.post(_request(hash_key), x_headroom_bypass="true")

    assert resp.text == "data: plain\n\n"
    assert proxy.sent == []


def test_response_hooks_do_not_run_on_a_buffered_turn(monkeypatch):
    """Their on_request half never ran; it is decided before dispatch, like a stream."""
    import headroom.proxy.turn_hooks as turn_hooks

    calls: list[str] = []

    class _Hook:
        name = "buffered-only"

        def on_request(self, ctx):  # noqa: ANN001
            calls.append("on_request")

        async def on_response(self, ctx, response, call_model):  # noqa: ANN001
            calls.append("on_response")
            return None

    hash_key = _stored_hash()
    with _Proxy([_final_reply()]) as proxy:
        monkeypatch.setattr(turn_hooks, "_hooks", [_Hook()], raising=False)
        monkeypatch.setattr(turn_hooks, "registered_turn_hooks", lambda: [_Hook()])
        resp = proxy.post(_request(hash_key))

    assert resp.status_code == 200, resp.text
    assert calls == []


def _mixed_reply(hash_key: str) -> dict:
    return _completion(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                _tool_call(CCR_TOOL_NAME, {"hash": hash_key}),
                _tool_call("Read", {"file_path": "a.py"}, call_id="call_2"),
            ],
        },
        "tool_calls",
        USAGE_FIRST,
    )


def test_retrieval_next_to_a_client_tool_lets_the_model_decide_again():
    """The unrun client call is dropped, and the continuation replaces the turn."""
    hash_key = _stored_hash()
    with _Proxy([_mixed_reply(hash_key), _final_reply()]) as proxy:
        resp = proxy.post(_request(hash_key))

    assert resp.status_code == 200, resp.text
    assert CCR_TOOL_NAME not in resp.text
    assert "there are 2 rows" in resp.text
    replayed = proxy.sent[1]["messages"][-2]
    assert [call["function"]["name"] for call in replayed["tool_calls"]] == [CCR_TOOL_NAME]


def test_mixed_turn_goes_back_whole_when_its_continuation_fails():
    """The model made no newer decision, so the client gets the turn it did make."""
    hash_key = _stored_hash()
    upstream = [_mixed_reply(hash_key), (500, {"error": {"message": "nope"}})]
    with _Proxy(upstream) as proxy:
        resp = proxy.post(_request(hash_key))

    assert resp.status_code == 200, resp.text
    choices = [chunk["choices"][0] for chunk in _chunks(resp.text) if chunk["choices"]]
    names = [
        choice["delta"]["tool_calls"][0]["function"]["name"]
        for choice in choices
        if "tool_calls" in choice["delta"]
    ]
    assert names == [CCR_TOOL_NAME, "Read"]
    assert choices[-1]["finish_reason"] == "tool_calls"


def test_continuation_replays_what_the_provider_put_on_the_assistant_message():
    """DeepSeek rejects a tool-call turn replayed without its reasoning_content."""
    hash_key = _stored_hash()
    reply = _retrieve_reply(hash_key)
    message = reply["choices"][0]["message"]
    message["reasoning_content"] = "thinking it over"
    message["reasoning_details"] = [{"text": "t"}]
    message["refusal"] = None
    message["annotations"] = []
    with _Proxy([reply, _final_reply()]) as proxy:
        resp = proxy.post(_request(hash_key))

    assert resp.status_code == 200, resp.text
    replayed = proxy.sent[1]["messages"][-2]
    assert replayed["role"] == "assistant"
    assert replayed["reasoning_content"] == "thinking it over"
    assert replayed["reasoning_details"] == [{"text": "t"}]
    assert "refusal" not in replayed, "empty fields stay out of the replayed message"
    assert "annotations" not in replayed


class _RequiresReasoning(_Proxy):
    """An upstream that rejects a tool-call turn replayed without its reasoning."""

    async def _retry(self, method, url, headers, body, *args, **kwargs):  # noqa: ANN001
        for message in body["messages"]:
            if message.get("tool_calls") and "reasoning_content" not in message:
                self.sent.append(json.loads(json.dumps(body)))
                return httpx.Response(400, json={"error": {"message": "reasoning_content missing"}})
        return await super()._retry(method, url, headers, body, *args, **kwargs)


def test_a_provider_that_requires_reasoning_on_replay_gets_it():
    hash_key = _stored_hash()
    reply = _retrieve_reply(hash_key)
    reply["choices"][0]["message"]["reasoning_content"] = "required-replay"
    with _RequiresReasoning([reply, _final_reply()]) as proxy:
        resp = proxy.post(_request(hash_key))

    assert resp.status_code == 200, resp.text
    assert "there are 2 rows" in resp.text


@pytest.mark.parametrize(
    "payload",
    [
        {"error": {"message": "gateway said no"}},
        {"choices": []},
        {"choices": None},
        {"choices": "nope"},
        {"choices": [None]},
        {"choices": [{}]},
        {"choices": [{"finish_reason": "stop"}]},
        {"choices": [{"message": None, "finish_reason": "stop"}]},
    ],
)
def test_a_200_with_no_usable_choice_is_not_replayed_as_an_empty_stream(payload):
    hash_key = _stored_hash()
    with _Proxy([payload]) as proxy:
        resp = proxy.post(_request(hash_key))

    assert resp.status_code == 502
    assert resp.json()["error"]["type"] == "upstream_protocol_error"


def _as_sse(payload: Any) -> Response:
    return _buffered_chat_reply_as_sse(
        Response(json.dumps(payload).encode(), status_code=200),
        ccr_response_handler=CCRResponseHandler(),
        include_usage=False,
        continuation_status=None,
        request_id="test",
    )


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {"choices": [None]},
        {"choices": [{}]},
        {"choices": [{"message": "text"}]},
    ],
)
def test_the_helper_rejects_a_reply_with_no_usable_choice(payload):
    result = _as_sse(payload)

    assert result.status_code == 502
    assert json.loads(result.body)["error"]["type"] == "upstream_protocol_error"


def test_a_refusal_is_a_valid_reply():
    hash_key = _stored_hash()
    reply = _completion(
        {"role": "assistant", "content": None, "refusal": "no"}, "stop", USAGE_FIRST
    )
    with _Proxy([reply]) as proxy:
        resp = proxy.post(_request(hash_key))

    assert resp.status_code == 200, resp.text
    deltas = [chunk["choices"][0]["delta"] for chunk in _chunks(resp.text) if chunk["choices"]]
    assert {"refusal": "no"} in deltas


def test_a_buffered_reply_never_lands_in_the_response_cache():
    """It was built for a stream:true request; a later caller must not get it (#3019)."""
    hash_key = _stored_hash()
    with _Proxy([_final_reply()], cache_enabled=True) as proxy:
        calls: list[Any] = []

        async def _set(*args: Any, **kwargs: Any) -> None:
            calls.append(args)

        proxy.proxy.cache.set = _set
        resp = proxy.post(_request(hash_key))

    assert resp.status_code == 200
    assert calls == []


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, True),
        ({"body": {"stream": False, "tools": [CLIENT_TOOL]}}, False),
        ({"body": {"stream": True, "tools": []}}, False),
        ({"body": {"stream": True}}, False),
        ({"body": {"stream": True, "tools": [CLIENT_TOOL], "n": 2}}, False),
        ({"body": {"stream": True, "tools": [CLIENT_TOOL], "n": 1}}, True),
        ({"body": {"stream": True, "tools": [CLIENT_TOOL], "logprobs": True}}, False),
        ({"enabled": False}, False),
        ({"ccr_response_handler_enabled": False}, False),
        ({"ccr_inject_tool": False}, False),
        ({"routed_to_backend": True}, False),
        ({"upstream_base_url": "https://opencode.ai/zen/v1"}, False),
    ],
)
def test_eligibility(overrides, expected):
    args: dict[str, Any] = {
        "body": {"stream": True, "tools": [CLIENT_TOOL]},
        "enabled": True,
        "ccr_response_handler_enabled": True,
        "ccr_inject_tool": True,
        "routed_to_backend": False,
        "upstream_base_url": "https://api.openai.com",
        **overrides,
    }
    assert _is_openai_chat_stream_ccr_eligible(**args) is expected


def test_taking_stream_options_leaves_other_fields_alone():
    body = {"stream": True, "stream_options": {"include_usage": True}, "temperature": 0}
    assert _take_stream_options_for_buffered_chat(body) is True
    assert body == {"stream": True, "temperature": 0}


def test_frames_keep_the_fields_a_client_replays_on_its_next_turn():
    call = {
        **_tool_call("Read", {"file_path": "a.py"}),
        "extra_content": {"google": {"thought_signature": "sig"}},
    }
    reply = _completion(
        {
            "role": "assistant",
            "reasoning_content": "thinking it over",
            "reasoning_details": [{"text": "t"}],
            "refusal": None,
            "content": "reading",
            "tool_calls": [call],
        },
        "tool_calls",
        USAGE_FIRST,
    )
    frames = StreamingCCRHandler.openai_chat_sse_frames(reply, include_usage=False)
    deltas = [
        chunk["choices"][0]["delta"]
        for chunk in _chunks(b"".join(frames).decode())
        if chunk["choices"]
    ]

    assert {"reasoning_content": "thinking it over", "reasoning_details": [{"text": "t"}]} in deltas
    tool_delta = next(delta["tool_calls"][0] for delta in deltas if "tool_calls" in delta)
    assert tool_delta["extra_content"] == {"google": {"thought_signature": "sig"}}
    assert tool_delta["index"] == 0
    assert frames[-1] == b"data: [DONE]\n\n"
