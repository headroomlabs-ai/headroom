"""Proxy-injected memory tool calls never reach a streaming client (GH #2195).

Claude Code streams every request. Before this fix the SSE path forwarded a
``memory_save`` / ``memory_search`` tool_use to the client, which never
declared those tools and answered ``No such tool available``, so saves looked
failed and searches never returned.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from headroom.proxy.memory_tool_stream import MemoryToolStreamFilter
from headroom.proxy.server import HeadroomProxy

MEMORY_TOOLS = frozenset({"memory_save", "memory_search"})


def _frame(payload: dict[str, Any]) -> bytes:
    return f"event: {payload['type']}\ndata: {json.dumps(payload)}\n\n".encode()


def _sse(blocks: list[dict[str, Any]], stop_reason: str, *, input_tokens: int = 10) -> bytes:
    """An Anthropic SSE response carrying ``blocks``."""
    frames = [
        _frame(
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-test",
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": input_tokens, "output_tokens": 1},
                },
            }
        )
    ]
    for index, block in enumerate(blocks):
        if block["type"] == "text":
            frames.append(
                _frame(
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {"type": "text", "text": ""},
                    }
                )
            )
            delta = {"type": "text_delta", "text": block["text"]}
        else:
            frames.append(
                _frame(
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {**block, "input": {}},
                    }
                )
            )
            delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
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


TEXT = {"type": "text", "text": "Saving that."}
SAVE = {
    "type": "tool_use",
    "id": "toolu_mem",
    "name": "memory_save",
    "input": {"content": "deploy region is ap-southeast-1"},
}
BASH = {"type": "tool_use", "id": "toolu_bash", "name": "Bash", "input": {"command": "ls"}}


class TestMemoryToolStreamFilter:
    def test_stream_without_memory_calls_is_byte_identical(self) -> None:
        raw = _sse([TEXT, BASH], "tool_use")
        flt = MemoryToolStreamFilter(MEMORY_TOOLS)
        out = b"".join(flt.feed(raw)) + b"".join(flt.closing_frames())
        assert out == raw
        assert not flt.hid_tool_calls

    def test_byte_at_a_time_feed_matches_one_shot(self) -> None:
        raw = _sse([TEXT, SAVE], "tool_use")
        one_shot = MemoryToolStreamFilter(MEMORY_TOOLS)
        expected = b"".join(one_shot.feed(raw)) + b"".join(one_shot.closing_frames())
        trickle = MemoryToolStreamFilter(MEMORY_TOOLS)
        got = b"".join(b"".join(trickle.feed(raw[i : i + 1])) for i in range(len(raw)))
        got += b"".join(trickle.closing_frames())
        assert got == expected

    def test_memory_tool_use_is_withheld_and_turn_ends(self) -> None:
        flt = MemoryToolStreamFilter(MEMORY_TOOLS)
        forwarded = _events(b"".join(flt.feed(_sse([TEXT, SAVE], "tool_use"))))
        assert not any(e.get("content_block", {}).get("type") == "tool_use" for e in forwarded)
        assert not any(e["type"] in ("message_delta", "message_stop") for e in forwarded)
        assert flt.hidden_tool_names == ["memory_save"]
        assert flt.stop_reason == "tool_use"

        tail = _events(b"".join(flt.closing_frames()))
        assert [e["type"] for e in tail] == ["message_delta", "message_stop"]
        assert tail[0]["delta"]["stop_reason"] == "end_turn"

    def test_blocks_after_a_hidden_call_are_reindexed(self) -> None:
        flt = MemoryToolStreamFilter(MEMORY_TOOLS)
        forwarded = _events(b"".join(flt.feed(_sse([SAVE, TEXT], "end_turn"))))
        assert {e["index"] for e in forwarded if "index" in e} == {0}
        assert flt.next_index == 1

    def test_continuation_round_extends_the_open_message(self) -> None:
        flt = MemoryToolStreamFilter(MEMORY_TOOLS, index_offset=1, forward_message_start=False)
        forwarded = _events(b"".join(flt.feed(_sse([TEXT], "end_turn"))))
        assert forwarded[0]["type"] == "content_block_start"
        assert {e["index"] for e in forwarded if "index" in e} == {1}

    def test_client_tool_alongside_memory_call_keeps_tool_use_stop(self) -> None:
        flt = MemoryToolStreamFilter(MEMORY_TOOLS)
        forwarded = _events(b"".join(flt.feed(_sse([SAVE, BASH], "tool_use"))))
        names = [e["content_block"].get("name") for e in forwarded if "content_block" in e]
        assert names == ["Bash"]
        assert flt.visible_tool_use
        tail = _events(b"".join(flt.closing_frames()))
        assert tail[0]["delta"]["stop_reason"] == "tool_use"

    def test_client_declared_memory_tool_is_not_withheld(self) -> None:
        flt = MemoryToolStreamFilter(frozenset({"memory_search"}))
        forwarded = _events(b"".join(flt.feed(_sse([SAVE], "tool_use"))))
        assert any(e.get("content_block", {}).get("name") == "memory_save" for e in forwarded)
        assert not flt.hid_tool_calls


def _proxy(upstream_bodies: list[bytes], tool_results: list[dict[str, Any]]) -> HeadroomProxy:
    proxy = object.__new__(HeadroomProxy)
    proxy.http_client = MagicMock(spec=httpx.AsyncClient)
    proxy.metrics = MagicMock()
    proxy.metrics.record_request = AsyncMock(return_value=None)
    proxy.metrics.record_failed = AsyncMock(return_value=None)
    proxy.cost_tracker = MagicMock()
    proxy.cost_tracker.estimate_cost.return_value = 0.0
    proxy.stats = {
        "requests_total": 0,
        "requests_optimized": 0,
        "tokens": {"original": 0, "optimized": 0, "saved": 0},
        "cost": {"total_usd": 0, "savings_usd": 0},
        "errors": 0,
        "active_requests": 0,
        "requests_per_model": {},
    }
    proxy._config = MagicMock()
    proxy._config.ccr_inject_tool = False
    proxy._config.retry_max_attempts = 1
    proxy.config = proxy._config
    proxy.memory_handler = MagicMock()
    proxy.memory_handler.handle_memory_tool_calls = AsyncMock(return_value=tool_results)

    responses = []
    for raw in upstream_bodies:
        response = MagicMock()
        response.headers = httpx.Headers({"content-type": "text/event-stream"})
        response.status_code = 200

        async def aiter_bytes(raw: bytes = raw):
            # Split mid-frame so the filter has to reassemble events.
            yield raw[:37]
            yield raw[37:]

        response.aiter_bytes = aiter_bytes
        response.aclose = AsyncMock()
        responses.append(response)
    proxy.http_client.build_request = MagicMock(return_value=MagicMock())
    proxy.http_client.send = AsyncMock(side_effect=responses)
    return proxy


async def _client_view(proxy: HeadroomProxy, **kwargs: Any) -> list[dict[str, Any]]:
    result = await proxy._stream_response(
        url="https://api.anthropic.com/v1/messages",
        headers={"x-api-key": "sk-test"},
        body={
            "model": "claude-test",
            "max_tokens": 100,
            "stream": True,
            "messages": [{"role": "user", "content": "remember the deploy region"}],
        },
        provider="anthropic",
        model="claude-test",
        request_id="test-mem",
        original_tokens=10,
        optimized_tokens=10,
        tokens_saved=0,
        transforms_applied=[],
        tags={},
        optimization_latency=0.0,
        memory_user_id="user-1",
        **kwargs,
    )
    raw = b"".join([chunk async for chunk in result.body_iterator])
    return _events(raw)


SAVE_RESULT = [{"type": "tool_result", "tool_use_id": "toolu_mem", "content": '{"status":"saved"}'}]


class TestStreamingMemoryContinuation:
    @pytest.mark.asyncio
    async def test_memory_call_runs_server_side_and_answer_streams_on(self) -> None:
        proxy = _proxy(
            [
                _sse([TEXT, SAVE], "tool_use"),
                _sse([{"type": "text", "text": "Saved."}], "end_turn"),
            ],
            SAVE_RESULT,
        )
        events = await _client_view(proxy, server_memory_tool_names=MEMORY_TOOLS)

        assert not any(e.get("content_block", {}).get("type") == "tool_use" for e in events)
        assert [e["type"] for e in events].count("message_start") == 1
        texts = [e["delta"]["text"] for e in events if e["type"] == "content_block_delta"]
        assert texts == ["Saving that.", "Saved."]
        starts = [e["index"] for e in events if e["type"] == "content_block_start"]
        assert starts == [0, 1]
        deltas = [e for e in events if e["type"] == "message_delta"]
        assert len(deltas) == 1 and deltas[0]["delta"]["stop_reason"] == "end_turn"
        assert events[-1]["type"] == "message_stop"

        proxy.memory_handler.handle_memory_tool_calls.assert_awaited_once()
        continuation = json.loads(
            proxy.http_client.build_request.call_args_list[1].kwargs["content"]
        )
        assert [m["role"] for m in continuation["messages"]] == ["user", "assistant", "user"]
        assert continuation["messages"][1]["content"][1]["name"] == "memory_save"
        assert continuation["messages"][2]["content"] == SAVE_RESULT

    @pytest.mark.asyncio
    async def test_finalizer_sees_the_message_the_client_received(self) -> None:
        proxy = _proxy(
            [
                _sse([TEXT, SAVE], "tool_use"),
                _sse([{"type": "text", "text": "Saved."}], "end_turn"),
            ],
            SAVE_RESULT,
        )
        proxy._finalize_stream_response = AsyncMock(return_value=None)
        await _client_view(proxy, server_memory_tool_names=MEMORY_TOOLS)

        finalized = proxy._finalize_stream_response.await_args.kwargs["parsed_response"]
        assert [b.get("text") for b in finalized["content"]] == ["Saving that.", "Saved."]

    @pytest.mark.asyncio
    async def test_client_tool_in_same_round_returns_turn_to_client(self) -> None:
        proxy = _proxy([_sse([SAVE, BASH], "tool_use")], SAVE_RESULT)
        events = await _client_view(proxy, server_memory_tool_names=MEMORY_TOOLS)

        names = [e["content_block"].get("name") for e in events if "content_block" in e]
        assert names == ["Bash"]
        assert [e["delta"]["stop_reason"] for e in events if e["type"] == "message_delta"] == [
            "tool_use"
        ]
        assert proxy.http_client.send.await_count == 1
        proxy.memory_handler.handle_memory_tool_calls.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_without_injected_tools_stream_passes_through(self) -> None:
        raw = _sse([TEXT, SAVE], "tool_use")
        proxy = _proxy([raw], SAVE_RESULT)
        proxy.memory_handler.has_memory_tool_calls = MagicMock(return_value=False)
        events = await _client_view(proxy)
        assert events == _events(raw)
