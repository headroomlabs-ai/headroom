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

from headroom.proxy.memory_tool_stream import (
    MemoryToolStreamFilter,
    MemoryToolStreamOverflowError,
)
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


class TestRecordedMemoryCalls:
    def test_hidden_call_input_is_rebuilt_from_stream(self) -> None:
        flt = MemoryToolStreamFilter(MEMORY_TOOLS)
        flt.feed(_sse([TEXT, SAVE], "tool_use"))
        assert flt.hidden_calls() == [SAVE]

    def test_call_with_truncated_input_is_not_run(self) -> None:
        raw = _sse([SAVE], "max_tokens").replace(b'\\"deploy region', b'\\"deploy', 1)
        raw = raw.replace(b'ap-southeast-1\\"}', b"ap-southeast-1", 1)
        flt = MemoryToolStreamFilter(MEMORY_TOOLS)
        flt.feed(raw)
        assert flt.hid_tool_calls
        assert flt.hidden_calls() == []

    def test_call_whose_block_never_stopped_is_not_run(self) -> None:
        # Complete, valid input, but the stream ends before content_block_stop.
        raw = _sse([SAVE], "tool_use")
        raw = raw[: raw.index(b"event: content_block_stop")]
        flt = MemoryToolStreamFilter(MEMORY_TOOLS)
        flt.feed(raw)
        assert flt.hid_tool_calls
        assert flt.hidden_calls() == []

    def test_prior_rounds_usage_is_added_to_the_final_delta(self) -> None:
        flt = MemoryToolStreamFilter(MEMORY_TOOLS)
        flt.feed(_sse([TEXT], "end_turn"))
        tail = _events(b"".join(flt.closing_frames(prior_usage={"output_tokens": 7})))
        assert tail[0]["usage"]["output_tokens"] == 12


async def _drain(gen: Any) -> list[dict[str, Any]]:
    return _events(b"".join([frame async for frame in gen]))


def _continue(proxy: HeadroomProxy, flt: MemoryToolStreamFilter, response: Any) -> Any:
    body = {"model": "claude-test", "messages": [{"role": "user", "content": "hi"}]}
    return proxy._continue_memory_tool_stream(
        flt,
        response,
        url="https://api.anthropic.com/v1/messages",
        outbound_headers={"x-api-key": "sk-test", "content-length": "1"},
        outbound_bytes=json.dumps(body).encode(),
        memory_user_id="user-1",
        memory_request_ctx=None,
        server_memory_tool_names=MEMORY_TOOLS,
        stream_state={},
        request_id="test-mem",
    )


class TestContinuationEdgeCases:
    @pytest.mark.asyncio
    async def test_unrebuilt_round_still_runs_its_memory_calls(self) -> None:
        proxy = _proxy([], SAVE_RESULT)
        flt = MemoryToolStreamFilter(MEMORY_TOOLS)
        flt.feed(_sse([TEXT, SAVE], "tool_use"))

        events = await _drain(_continue(proxy, flt, None))

        proxy.memory_handler.handle_memory_tool_calls.assert_awaited_once()
        ran = proxy.memory_handler.handle_memory_tool_calls.await_args.args[0]["content"]
        assert ran == [SAVE]
        assert proxy.http_client.send.await_count == 0
        assert events[0]["delta"]["stop_reason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_continuation_keeps_streaming_past_the_buffer_cap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import headroom.proxy.helpers as helpers

        round_one = _sse([TEXT, SAVE], "tool_use")
        reply = _sse([{"type": "text", "text": "Saved."}], "end_turn")
        proxy = _proxy([reply], SAVE_RESULT)
        flt = MemoryToolStreamFilter(MEMORY_TOOLS)
        flt.feed(round_one)
        response = proxy._parse_sse_to_response(round_one.decode(), "anthropic")
        # _proxy splits each body in two; the second chunk arrives past the cap.
        monkeypatch.setattr(helpers, "MAX_SSE_BUFFER_SIZE", 16)

        events = await _drain(_continue(proxy, flt, response))

        texts = [e["delta"]["text"] for e in events if e["type"] == "content_block_delta"]
        assert texts == ["Saved."]
        assert [e["type"] for e in events][-2:] == ["message_delta", "message_stop"]

    @pytest.mark.asyncio
    async def test_interrupted_call_is_not_run(self) -> None:
        raw = _sse([TEXT, SAVE], "tool_use")
        raw = raw[: raw.rindex(b"event: content_block_stop")]
        proxy = _proxy([], SAVE_RESULT)
        flt = MemoryToolStreamFilter(MEMORY_TOOLS)
        flt.feed(raw)

        await _drain(_continue(proxy, flt, None))

        proxy.memory_handler.handle_memory_tool_calls.assert_not_awaited()
        assert proxy.http_client.send.await_count == 0

    @pytest.mark.asyncio
    async def test_continuation_over_the_buffer_cap_is_not_continued_again(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import headroom.proxy.helpers as helpers

        round_one = _sse([TEXT, SAVE], "tool_use")
        proxy = _proxy([_sse([SAVE], "tool_use")], SAVE_RESULT)
        flt = MemoryToolStreamFilter(MEMORY_TOOLS)
        flt.feed(round_one)
        response = proxy._parse_sse_to_response(round_one.decode(), "anthropic")
        monkeypatch.setattr(helpers, "MAX_SSE_BUFFER_SIZE", 64)

        events = await _drain(_continue(proxy, flt, response))

        assert proxy.http_client.send.await_count == 1
        assert proxy.memory_handler.handle_memory_tool_calls.await_count == 2
        assert [e["type"] for e in events][-2:] == ["message_delta", "message_stop"]
        assert events[-2]["delta"]["stop_reason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_client_sees_output_usage_of_every_round(self) -> None:
        proxy = _proxy(
            [
                _sse([TEXT, SAVE], "tool_use"),
                _sse([{"type": "text", "text": "Saved."}], "end_turn"),
            ],
            SAVE_RESULT,
        )
        events = await _client_view(proxy, server_memory_tool_names=MEMORY_TOOLS)
        deltas = [e for e in events if e["type"] == "message_delta"]
        assert deltas[0]["usage"]["output_tokens"] == 10


class TestFrameParsingEdgeCases:
    """Frames the filter cannot interpret pass through to the client untouched."""

    @pytest.mark.parametrize(
        "frame",
        [
            b": keep-alive\n\n",
            b"event: ping\n\n",
            b"event: content_block_delta\ndata: {not json\n\n",
            b"event: content_block_delta\ndata: [1, 2]\n\n",
            b'event: ping\ndata: {"type": "ping"}\n\n',
            b'event: content_block_delta\ndata: {"type": "content_block_delta", "index": "0"}\n\n',
        ],
    )
    def test_uninterpretable_frame_is_forwarded_verbatim(self, frame: bytes) -> None:
        flt = MemoryToolStreamFilter(MEMORY_TOOLS)
        assert flt.feed(frame) == [frame]

    def test_hidden_call_with_non_object_input_is_not_run(self) -> None:
        raw = _sse([SAVE], "tool_use").replace(
            json.dumps(json.dumps(SAVE["input"])).encode(), json.dumps("[1]").encode(), 1
        )
        flt = MemoryToolStreamFilter(MEMORY_TOOLS)
        flt.feed(raw)
        assert flt.hid_tool_calls
        assert flt.hidden_calls() == []

    def test_hidden_call_with_input_in_its_start_frame(self) -> None:
        raw = b"".join(
            [
                _frame({"type": "content_block_start", "index": 0, "content_block": SAVE}),
                _frame({"type": "content_block_stop", "index": 0}),
            ]
        )
        flt = MemoryToolStreamFilter(MEMORY_TOOLS)
        assert flt.feed(raw) == []
        assert flt.hidden_calls() == [SAVE]

    def test_other_delta_on_hidden_block_is_withheld(self) -> None:
        raw = b"".join(
            [
                _frame({"type": "content_block_start", "index": 0, "content_block": SAVE}),
                _frame({"type": "content_block_delta", "index": 0, "delta": {"type": "x"}}),
                _frame({"type": "content_block_stop", "index": 0}),
            ]
        )
        flt = MemoryToolStreamFilter(MEMORY_TOOLS)
        assert flt.feed(raw) == []
        assert flt.hidden_calls() == [SAVE]

    def test_message_delta_without_stop_reason_keeps_none(self) -> None:
        flt = MemoryToolStreamFilter(MEMORY_TOOLS)
        flt.feed(_frame({"type": "message_delta", "delta": {}, "usage": {"output_tokens": 1}}))
        assert flt.stop_reason is None

    def test_unterminated_trailing_frame_is_dropped(self) -> None:
        flt = MemoryToolStreamFilter(MEMORY_TOOLS)
        flt.feed(_sse([TEXT], "end_turn"))
        flt.feed(b"event: ping\ndata: {")
        assert b"ping" not in b"".join(flt.closing_frames())

    def test_truncated_hidden_delta_never_reaches_the_client(self) -> None:
        secret = "my private deploy key"
        start = _frame({"type": "content_block_start", "index": 0, "content_block": SAVE})
        delta = _frame(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": f'{{"content": "{secret}'},
            }
        )
        flt = MemoryToolStreamFilter(MEMORY_TOOLS)
        emitted = flt.feed(start + delta[:-2])  # the stream ends before the terminator
        emitted += flt.closing_frames()
        assert secret.encode() not in b"".join(emitted)
        assert flt.hidden_calls() == []


class TestRetainedByteLimit:
    def test_unterminated_event_stops_growing_at_the_limit(self) -> None:
        flt = MemoryToolStreamFilter(MEMORY_TOOLS, max_retained_bytes=4096)
        chunk = b"event: content_block_delta\ndata: " + b"x" * 1024
        with pytest.raises(MemoryToolStreamOverflowError):
            for _ in range(12):
                flt.feed(chunk)
        assert flt.closing_frames() == []

    def test_oversized_hidden_input_is_dropped_not_run(self) -> None:
        flt = MemoryToolStreamFilter(MEMORY_TOOLS, max_retained_bytes=4096)
        flt.feed(_frame({"type": "content_block_start", "index": 0, "content_block": SAVE}))
        fragment = {"type": "input_json_delta", "partial_json": "y" * 1024}
        with pytest.raises(MemoryToolStreamOverflowError):
            for _ in range(12):
                flt.feed(_frame({"type": "content_block_delta", "index": 0, "delta": fragment}))
        assert not flt.hid_tool_calls
        assert flt.hidden_calls() == []
        assert flt.closing_frames() == []

    def test_inline_hidden_input_counts_against_the_limit(self) -> None:
        flt = MemoryToolStreamFilter(MEMORY_TOOLS, max_retained_bytes=4096)
        block = {**SAVE, "input": {"content": "z" * 20000}}
        with pytest.raises(MemoryToolStreamOverflowError):
            flt.feed(_frame({"type": "content_block_start", "index": 0, "content_block": block}))
        assert not flt.hid_tool_calls

    def test_limit_is_in_bytes_at_the_boundary(self) -> None:
        start = _frame({"type": "content_block_start", "index": 0, "content_block": SAVE})
        snowmen = {"type": "input_json_delta", "partial_json": "\u2603" * 24}
        payload = {"type": "content_block_delta", "index": 0, "delta": snowmen}
        # Raw UTF-8 on the wire: 24 characters, 72 bytes.
        delta = f"event: {payload['type']}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        frames = start + delta.encode()

        MemoryToolStreamFilter(MEMORY_TOOLS, max_retained_bytes=len(frames)).feed(frames)
        with pytest.raises(MemoryToolStreamOverflowError):
            MemoryToolStreamFilter(MEMORY_TOOLS, max_retained_bytes=len(frames) - 1).feed(frames)

    def test_lone_surrogate_in_hidden_input_is_counted(self) -> None:
        flt = MemoryToolStreamFilter(MEMORY_TOOLS)
        flt.feed(_frame({"type": "content_block_start", "index": 0, "content_block": SAVE}))
        fragment = {"type": "input_json_delta", "partial_json": '{"content": "\ud800"}'}
        frame = _frame({"type": "content_block_delta", "index": 0, "delta": fragment})
        assert flt.feed(frame) == []
        assert flt.feed(_frame({"type": "content_block_stop", "index": 0})) == []
        assert flt.hidden_tool_names == ["memory_save"]

    def test_repeated_hidden_start_replaces_its_charge(self) -> None:
        first = {**SAVE, "input": {"content": "a" * 300}}
        second = {**SAVE, "input": {"content": "b" * 300}}
        start_one = _frame({"type": "content_block_start", "index": 0, "content_block": first})
        start_two = _frame({"type": "content_block_start", "index": 0, "content_block": second})
        stop = _frame({"type": "content_block_stop", "index": 0})
        flt = MemoryToolStreamFilter(MEMORY_TOOLS, max_retained_bytes=len(start_two) + 1)
        flt.feed(start_one)
        flt.feed(stop)
        flt.feed(start_two)
        assert flt.hidden_calls() == []  # the replacement has not stopped yet
        flt.feed(stop)
        assert flt.hidden_calls() == [second]

    def test_frames_within_the_limit_pass(self) -> None:
        raw = _sse([TEXT, SAVE], "tool_use")
        flt = MemoryToolStreamFilter(MEMORY_TOOLS, max_retained_bytes=len(raw))
        flt.feed(raw)
        assert flt.hidden_calls() == [SAVE]


class TestUpstreamErrorFrame:
    def test_anthropic_error_envelope_passes_through(self) -> None:
        from headroom.proxy.handlers.streaming import _upstream_error_frame

        envelope = {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}
        [event] = _events(_upstream_error_frame(json.dumps(envelope).encode(), "req-1"))
        assert event == envelope

    @pytest.mark.parametrize(
        "body", [b"<html>Bad Gateway</html>", b"\xff\xfe", b'{"detail": "nope"}']
    )
    def test_other_bodies_become_the_public_error(self, body: bytes) -> None:
        from headroom.proxy.handlers.streaming import _upstream_error_frame

        raw = _upstream_error_frame(body, "req-1")
        assert raw.startswith(b"event: error\n")
        [event] = _events(raw)
        assert event["type"] == "error"
        assert body.decode("utf-8", "replace") not in json.dumps(event)


class TestContinuationFallbacks:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "outbound", [b"not json", json.dumps({"model": "claude-test", "messages": "hi"}).encode()]
    )
    async def test_unreadable_request_body_ends_the_turn_after_running_calls(
        self, outbound: bytes
    ) -> None:
        round_one = _sse([TEXT, SAVE], "tool_use")
        proxy = _proxy([], SAVE_RESULT)
        flt = MemoryToolStreamFilter(MEMORY_TOOLS)
        flt.feed(round_one)
        response = proxy._parse_sse_to_response(round_one.decode(), "anthropic")

        events = await _drain(
            proxy._continue_memory_tool_stream(
                flt,
                response,
                url="https://api.anthropic.com/v1/messages",
                outbound_headers={"x-api-key": "sk-test"},
                outbound_bytes=outbound,
                memory_user_id="user-1",
                memory_request_ctx=None,
                server_memory_tool_names=MEMORY_TOOLS,
                stream_state={},
                request_id="test-mem",
            )
        )

        proxy.memory_handler.handle_memory_tool_calls.assert_awaited_once()
        assert proxy.http_client.send.await_count == 0
        assert events[0]["delta"]["stop_reason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_round_limit_stops_continuing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        round_one = _sse([TEXT, SAVE], "tool_use")
        proxy = _proxy([], SAVE_RESULT)
        monkeypatch.setattr(type(proxy), "_MEMORY_CONTINUATION_MAX_ROUNDS", 0)
        flt = MemoryToolStreamFilter(MEMORY_TOOLS)
        flt.feed(round_one)
        response = proxy._parse_sse_to_response(round_one.decode(), "anthropic")

        events = await _drain(_continue(proxy, flt, response))

        proxy.memory_handler.handle_memory_tool_calls.assert_awaited_once()
        assert proxy.http_client.send.await_count == 0
        assert events[0]["delta"]["stop_reason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_failed_continuation_round_sends_an_error_event(self) -> None:
        round_one = _sse([TEXT, SAVE], "tool_use")
        proxy = _proxy([], SAVE_RESULT)
        failed = MagicMock()
        failed.status_code = 529
        envelope = {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}
        failed.aread = AsyncMock(return_value=json.dumps(envelope).encode())
        failed.aclose = AsyncMock()
        proxy.http_client.send = AsyncMock(return_value=failed)
        flt = MemoryToolStreamFilter(MEMORY_TOOLS)
        flt.feed(round_one)
        response = proxy._parse_sse_to_response(round_one.decode(), "anthropic")

        events = await _drain(_continue(proxy, flt, response))

        assert proxy.http_client.send.await_count == 1
        assert events == [envelope]
        failed.aclose.assert_awaited()

    @pytest.mark.asyncio
    async def test_calls_are_not_run_without_a_memory_user(self) -> None:
        proxy = _proxy([], SAVE_RESULT)
        flt = MemoryToolStreamFilter(MEMORY_TOOLS)
        flt.feed(_sse([TEXT, SAVE], "tool_use"))

        events = await _drain(
            proxy._continue_memory_tool_stream(
                flt,
                None,
                url="https://api.anthropic.com/v1/messages",
                outbound_headers={"x-api-key": "sk-test"},
                outbound_bytes=b"{}",
                memory_user_id=None,
                memory_request_ctx=None,
                server_memory_tool_names=MEMORY_TOOLS,
                stream_state={},
                request_id="test-mem",
            )
        )

        proxy.memory_handler.handle_memory_tool_calls.assert_not_awaited()
        assert events[0]["delta"]["stop_reason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_round_without_usage_still_continues(self) -> None:
        round_one = _sse([TEXT, SAVE], "tool_use")
        proxy = _proxy([_sse([{"type": "text", "text": "Saved."}], "end_turn")], SAVE_RESULT)
        flt = MemoryToolStreamFilter(MEMORY_TOOLS)
        flt.feed(round_one)
        response = proxy._parse_sse_to_response(round_one.decode(), "anthropic")
        response.pop("usage", None)

        events = await _drain(_continue(proxy, flt, response))

        assert proxy.http_client.send.await_count == 1
        texts = [e["delta"]["text"] for e in events if e["type"] == "content_block_delta"]
        assert texts == ["Saved."]
        assert events[-2]["delta"]["stop_reason"] == "end_turn"


class TestStreamingMemoryFallbacks:
    @pytest.mark.asyncio
    async def test_client_handled_memory_calls_still_run_without_server_tools(self) -> None:
        raw = _sse([TEXT, SAVE], "tool_use")
        proxy = _proxy([raw], SAVE_RESULT)
        proxy.memory_handler.has_memory_tool_calls = MagicMock(return_value=True)
        events = await _client_view(proxy)

        assert events == _events(raw)
        proxy.memory_handler.handle_memory_tool_calls.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("server_tools", [MEMORY_TOOLS, frozenset()])
    async def test_subscription_credential_error_ends_the_message(
        self, server_tools: frozenset[str]
    ) -> None:
        refusal = {
            "type": "text",
            "text": "This credential is only authorized for use with Claude Code.",
        }
        proxy = _proxy([_sse([refusal], "end_turn")], SAVE_RESULT)
        events = await _client_view(proxy, server_memory_tool_names=server_tools)

        assert proxy.http_client.send.await_count == 1
        proxy.memory_handler.handle_memory_tool_calls.assert_not_awaited()
        assert [e["type"] for e in events][-2:] == ["message_delta", "message_stop"]

    @pytest.mark.asyncio
    async def test_buffer_cap_still_runs_withheld_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import headroom.proxy.helpers as helpers

        proxy = _proxy([_sse([TEXT, SAVE], "tool_use")], SAVE_RESULT)
        monkeypatch.setattr(helpers, "MAX_SSE_BUFFER_SIZE", 64)
        events = await _client_view(proxy, server_memory_tool_names=MEMORY_TOOLS)

        assert not any(e.get("content_block", {}).get("type") == "tool_use" for e in events)
        proxy.memory_handler.handle_memory_tool_calls.assert_awaited_once()
        assert proxy.http_client.send.await_count == 1
        deltas = [e for e in events if e["type"] == "message_delta"]
        assert deltas[-1]["delta"]["stop_reason"] == "end_turn"


class TestServerMemoryToolNames:
    def test_injected_memory_tools_are_server_side(self) -> None:
        from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin

        tools = [
            {"name": "Bash"},
            {"name": "memory_save"},
            {"name": "memory_search"},
            {"type": "memory_20250818", "name": "memory"},
            {"type": "web_search"},
            "not-a-tool",
        ]
        names = AnthropicHandlerMixin._server_memory_tool_names(tools, [{"name": "Bash"}])
        assert names == frozenset({"memory_save", "memory_search", "memory"})

    def test_client_declared_memory_tool_stays_with_the_client(self) -> None:
        from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin

        tools = [{"name": "memory_save"}, {"name": "memory_search"}]
        names = AnthropicHandlerMixin._server_memory_tool_names(
            tools, [{"name": "memory_save"}, "junk"]
        )
        assert names == frozenset({"memory_search"})
        assert AnthropicHandlerMixin._server_memory_tool_names(None, None) == frozenset()


class TestHandlerPassesServerMemoryTools:
    def test_injected_memory_tools_reach_the_stream(self) -> None:
        from types import SimpleNamespace

        from fastapi.responses import StreamingResponse
        from fastapi.testclient import TestClient

        from headroom.proxy.server import ProxyConfig, create_app

        config = ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            ccr_context_tracking=False,
            image_optimize=False,
        )
        captured: dict[str, Any] = {}

        async def fake_stream_response(*args: Any, **kwargs: Any) -> StreamingResponse:
            captured.update(kwargs)

            async def gen() -> Any:
                yield b""

            return StreamingResponse(gen(), media_type="text/event-stream")

        memory_tools = [
            {"name": name, "description": name, "input_schema": {"type": "object"}}
            for name in sorted(MEMORY_TOOLS)
        ]
        with TestClient(create_app(config)) as client:
            proxy = client.app.state.proxy
            proxy.memory_handler = SimpleNamespace(
                config=SimpleNamespace(inject_context=False, inject_tools=True),
                compute_memory_tool_definitions=lambda provider: memory_tools,
                get_beta_headers=lambda: {},
                has_memory_tool_calls=lambda resp, provider: False,
            )
            proxy._stream_response = fake_stream_response
            client.post(
                "/v1/messages",
                headers={
                    "x-api-key": "test-key",
                    "anthropic-version": "2023-06-01",
                    "x-headroom-user-id": "u1",
                },
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 64,
                    "stream": True,
                    "tools": [{"name": "Bash", "input_schema": {"type": "object"}}],
                    "messages": [{"role": "user", "content": "remember the deploy region"}],
                },
            )

        assert captured["server_memory_tool_names"] == MEMORY_TOOLS


class TestStreamingRetainedByteLimit:
    SECRET = "s3cr3t-" * 256

    def _oversized_save(self) -> bytes:
        save = {**SAVE, "input": {"content": self.SECRET}}
        return _sse([TEXT, save], "tool_use")

    @pytest.mark.asyncio
    async def test_overflow_ends_the_stream_with_an_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import headroom.proxy.memory_tool_stream as mts

        monkeypatch.setattr(mts, "DEFAULT_MAX_RETAINED_BYTES", 256)
        proxy = _proxy([self._oversized_save()], SAVE_RESULT)
        events = await _client_view(proxy, server_memory_tool_names=MEMORY_TOOLS)

        assert events[-1]["type"] == "error"
        assert self.SECRET not in json.dumps(events)
        proxy.memory_handler.handle_memory_tool_calls.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_continuation_overflow_ends_the_stream_with_an_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import headroom.proxy.memory_tool_stream as mts

        monkeypatch.setattr(mts, "DEFAULT_MAX_RETAINED_BYTES", 1024)
        proxy = _proxy([_sse([TEXT, SAVE], "tool_use"), self._oversized_save()], SAVE_RESULT)
        events = await _client_view(proxy, server_memory_tool_names=MEMORY_TOOLS)

        assert proxy.http_client.send.await_count == 2
        proxy.memory_handler.handle_memory_tool_calls.assert_awaited_once()
        assert events[-1]["type"] == "error"
        assert self.SECRET not in json.dumps(events)
