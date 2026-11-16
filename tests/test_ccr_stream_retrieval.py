"""``headroom_retrieve`` on a streamed turn is served by the proxy (#3587).

A turn whose redeemable markers were all forwarded on earlier turns streams
instead of buffering, so the client gets the first token right away. The model
can still ask for one of those originals. The client never declared
``headroom_retrieve`` (a coding agent answers "No such tool available"), so
the call must not reach it: the stream withholds it, the proxy serves it from
the compression store, and the continuation round streams into the same client
message.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.ccr.tool_injection import create_ccr_tool_definition  # noqa: E402
from headroom.proxy.memory_tool_stream import MemoryToolStreamFilter  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402

RETRIEVE_ONLY = frozenset({"headroom_retrieve"})
ORIGINAL = "line 1 of the original build log\nline 2: the error the model needs"


def _frame(payload: dict[str, Any]) -> bytes:
    return f"event: {payload['type']}\ndata: {json.dumps(payload)}\n\n".encode()


def _sse(blocks: list[dict[str, Any]], stop_reason: str) -> bytes:
    """An Anthropic SSE response carrying ``blocks``."""
    frames = [
        _frame(
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "sonnet-test",
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 10, "output_tokens": 1},
                },
            }
        )
    ]
    for index, block in enumerate(blocks):
        if block["type"] == "text":
            start = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": block["text"]}
        else:
            start = {**block, "input": {}}
            delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
        frames.append(
            _frame({"type": "content_block_start", "index": index, "content_block": start})
        )
        frames.append(_frame({"type": "content_block_delta", "index": index, "delta": delta}))
        frames.append(_frame({"type": "content_block_stop", "index": index}))
    frames.append(
        _frame(
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": 5},
            }
        )
    )
    frames.append(_frame({"type": "message_stop"}))
    return b"".join(frames)


def _events(raw: bytes) -> list[dict[str, Any]]:
    return [
        json.loads(line[len("data: ") :])
        for line in raw.decode().splitlines()
        if line.startswith("data: ")
    ]


def _tool_names(events: list[dict[str, Any]]) -> list[str]:
    return [
        e["content_block"]["name"]
        for e in events
        if e["type"] == "content_block_start" and e["content_block"].get("type") == "tool_use"
    ]


TEXT = {"type": "text", "text": "Let me look at the earlier output."}
BASH = {"type": "tool_use", "id": "toolu_bash", "name": "Bash", "input": {"command": "ls"}}


def _retrieve(hash_key: str) -> dict[str, Any]:
    return {
        "type": "tool_use",
        "id": "toolu_ccr",
        "name": "headroom_retrieve",
        "input": {"hash": hash_key},
    }


class TestHoldingFilter:
    def test_text_before_the_first_tool_call_streams_live(self) -> None:
        flt = MemoryToolStreamFilter(RETRIEVE_ONLY, hold_tool_use=True)
        forwarded = _events(b"".join(flt.feed(_sse([TEXT, BASH], "tool_use"))))
        assert [e["type"] for e in forwarded] == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
        ]
        assert flt.holds_client_tool_use

    def test_without_a_retrieval_the_round_is_byte_identical(self) -> None:
        raw = _sse([TEXT, BASH], "tool_use")
        flt = MemoryToolStreamFilter(RETRIEVE_ONLY, hold_tool_use=True)
        out = b"".join(flt.feed(raw)) + b"".join(flt.closing_frames())
        assert out == raw

    def test_dropping_held_client_calls(self) -> None:
        flt = MemoryToolStreamFilter(RETRIEVE_ONLY, hold_tool_use=True)
        flt.feed(_sse([BASH, _retrieve("abc")], "tool_use"))
        assert flt.hidden_tool_names == ["headroom_retrieve"]
        assert flt.release_held(drop_tool_use=True) == []
        assert flt.next_index == 0
        assert not flt.visible_tool_use


def _sse_response(raw: bytes) -> MagicMock:
    response = MagicMock()
    response.headers = httpx.Headers({"content-type": "text/event-stream"})
    response.status_code = 200

    async def aiter_bytes(raw: bytes = raw):
        # Split mid-frame so the filter has to reassemble events.
        yield raw[:41]
        yield raw[41:]

    response.aiter_bytes = aiter_bytes
    response.aclose = AsyncMock()
    return response


@pytest.fixture
def owned_marker():
    """A marker the proxy owns, with its original in the compression store."""
    from headroom.cache.backends import InMemoryBackend
    from headroom.cache.compression_store import get_compression_store, reset_compression_store

    reset_compression_store()
    store = get_compression_store(backend=InMemoryBackend())
    hash_key = store.store(
        ORIGINAL,
        "<<ccr:placeholder>>",
        original_tokens=100,
        compressed_tokens=5,
        tool_name="Bash",
    )
    try:
        yield hash_key
    finally:
        reset_compression_store()


def _two_turn_session(
    marker: str, second_turn_rounds: list[bytes]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Turn 1 introduces ``marker``; turn 2 only carries it in history.

    Returns the client's SSE events for turn 2, the bodies the proxy sent
    upstream on turn 2's stream (first round, then continuations), and how
    many buffered upstream requests turn 2 made.
    """
    first_turn = [
        {"role": "user", "content": "build it"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "tu_1", "name": "Bash", "input": {}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tu_1",
                    "content": f"build log compressed at <<ccr:{marker}>>",
                }
            ],
        },
    ]
    second_turn = [
        *first_turn,
        {"role": "assistant", "content": [{"type": "text", "text": "The build failed."}]},
        {"role": "user", "content": "what was the exact error?"},
    ]
    buffered_calls: list[int] = []

    async def fake_retry(method, url, headers, req_body, *args, **kwargs):  # noqa: ANN001
        buffered_calls.append(1)
        return httpx.Response(
            200,
            json={
                "id": "msg_t1",
                "type": "message",
                "role": "assistant",
                "model": "sonnet-test",
                "content": [{"type": "text", "text": "The build failed."}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
            headers={"content-type": "application/json"},
        )

    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        memory_enabled=False,
    )
    headers = {"x-api-key": "test-key", "anthropic-version": "2023-06-01"}
    tools = [
        create_ccr_tool_definition("anthropic"),
        {"name": "Bash", "description": "run", "input_schema": {"type": "object"}},
    ]

    def body(messages: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "model": "sonnet-test",
            "max_tokens": 64,
            "stream": True,
            "tools": tools,
            "messages": messages,
        }

    app = create_app(config)
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        proxy._retry_request = fake_retry
        first = client.post("/v1/messages", json=body(first_turn), headers=headers)
        assert first.status_code == 200, first.text
        assert len(buffered_calls) == 1, "the turn that introduces the marker buffers"

        upstream = MagicMock(spec=httpx.AsyncClient)
        upstream.build_request = MagicMock(return_value=MagicMock())
        upstream.send = AsyncMock(side_effect=[_sse_response(r) for r in second_turn_rounds])
        proxy.http_client = upstream
        second = client.post("/v1/messages", json=body(second_turn), headers=headers)
        assert second.status_code == 200, second.text

    sent = [json.loads(c.kwargs["content"]) for c in upstream.build_request.call_args_list]
    return _events(second.content), sent, len(buffered_calls) - 1


def test_later_turn_retrieval_is_served_by_the_proxy(owned_marker: str) -> None:
    events, sent, buffered = _two_turn_session(
        owned_marker,
        [
            _sse([TEXT, _retrieve(owned_marker)], "tool_use"),
            _sse([{"type": "text", "text": "The error is on line 2."}], "end_turn"),
        ],
    )

    assert buffered == 0, "a turn with only already-forwarded markers keeps streaming"
    assert "headroom_retrieve" not in _tool_names(events)
    texts = [e["delta"]["text"] for e in events if e["type"] == "content_block_delta"]
    assert texts == ["Let me look at the earlier output.", "The error is on line 2."]
    stops = [e["delta"]["stop_reason"] for e in events if e["type"] == "message_delta"]
    assert stops == ["end_turn"]
    assert [e["type"] for e in events].count("message_start") == 1

    assert len(sent) == 2
    continuation = sent[1]["messages"]
    assert continuation[-2]["content"][-1]["name"] == "headroom_retrieve"
    served = continuation[-1]["content"][0]
    assert served["type"] == "tool_result" and served["tool_use_id"] == "toolu_ccr"
    assert json.loads(served["content"])["original_content"] == ORIGINAL


@pytest.mark.parametrize("client_call_first", [True, False])
def test_retrieval_mixed_with_a_client_tool(owned_marker: str, client_call_first: bool) -> None:
    calls = [BASH, _retrieve(owned_marker)]
    if not client_call_first:
        calls.reverse()
    reissued = {**BASH, "id": "toolu_bash_2"}
    events, sent, buffered = _two_turn_session(
        owned_marker,
        [
            _sse(calls, "tool_use"),
            _sse([{"type": "text", "text": "Now rerun it."}, reissued], "tool_use"),
        ],
    )

    assert buffered == 0
    # Only the model's decision made with the content in hand reaches the client.
    assert _tool_names(events) == ["Bash"]
    started = [e["content_block"] for e in events if e["type"] == "content_block_start"]
    assert [b.get("id") for b in started if b.get("type") == "tool_use"] == ["toolu_bash_2"]
    stops = [e["delta"]["stop_reason"] for e in events if e["type"] == "message_delta"]
    assert stops == ["tool_use"]
    assert [e["index"] for e in events if e["type"] == "content_block_start"] == [0, 1]

    continuation = sent[1]["messages"]
    assert [b["name"] for b in continuation[-2]["content"]] == ["headroom_retrieve"]
    assert json.loads(continuation[-1]["content"][0]["content"])["original_content"] == ORIGINAL


def test_old_marker_turn_without_retrieval_streams_unchanged(owned_marker: str) -> None:
    raw = _sse([{"type": "text", "text": "It failed on line 2."}, BASH], "tool_use")
    events, sent, buffered = _two_turn_session(owned_marker, [raw])

    assert buffered == 0
    assert len(sent) == 1, "no continuation round without a retrieval"
    assert events == _events(raw)
