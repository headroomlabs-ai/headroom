from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

import headroom.ccr.stream_splice as stream_splice_module
from headroom.cache.compression_store import get_compression_store, reset_compression_store
from headroom.ccr.response_handler import CCRResponseHandler
from headroom.ccr.stream_splice import EventLevelCCRInterceptor
from headroom.proxy.handlers.openai import _openai_responses_to_sse
from headroom.proxy.handlers.streaming import StreamingMixin
from headroom.proxy.server import HeadroomProxy


@pytest.fixture(autouse=True)
def reset_store():
    reset_compression_store()
    yield
    reset_compression_store()


def _event(event: dict) -> bytes:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


async def _chunks(chunks: list[bytes]):
    for chunk in chunks:
        yield chunk


def _stored_hash() -> str:
    return get_compression_store().store(original="full source", compressed="summary")


@pytest.mark.asyncio
async def test_anthropic_streams_long_text_then_splices_retrieval_continuation():
    hash_key = _stored_hash()
    upstream = [
        _event(
            {
                "type": "message_start",
                "message": {"id": "msg-1", "role": "assistant", "model": "claude"},
            }
        ),
        _event(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }
        ),
        _event(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "x" * 12_000},
            }
        ),
        _event({"type": "content_block_stop", "index": 0}),
        _event(
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "tool-ccr",
                    "name": "headroom_retrieve",
                    "input": {},
                },
            }
        ),
        _event(
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps({"hash": hash_key}),
                },
            }
        ),
        _event({"type": "content_block_stop", "index": 1}),
        _event(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 10},
            }
        ),
        _event({"type": "message_stop"}),
    ]
    calls = 0

    async def continuation(messages, tools):  # noqa: ANN001
        nonlocal calls
        calls += 1
        return {
            "id": "msg-2",
            "role": "assistant",
            "model": "claude",
            "content": [{"type": "text", "text": "continued"}],
            "stop_reason": "end_turn",
            "usage": {"output_tokens": 3},
        }

    interceptor = EventLevelCCRInterceptor(
        CCRResponseHandler(),
        provider="anthropic",
        render_response=lambda response: StreamingMixin()._response_to_sse(response, "anthropic"),
    )
    output = b"".join(
        [chunk async for chunk in interceptor.process(_chunks(upstream), [], None, continuation)]
    )

    assert calls == 1
    assert b"x" * 12_000 in output
    assert b"continued" in output
    assert b"headroom_retrieve" not in output
    assert output.count(b"event: message_start") == 1
    assert output.count(b"event: message_stop") == 1
    assert b'"output_tokens": 13' in output


@pytest.mark.asyncio
async def test_responses_splices_continuation_and_combines_terminal_output():
    hash_key = _stored_hash()
    visible = {
        "type": "message",
        "id": "msg-visible",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "before"}],
    }
    private = {
        "type": "function_call",
        "id": "fc-ccr",
        "call_id": "call-ccr",
        "name": "headroom_retrieve",
        "arguments": json.dumps({"hash": hash_key}),
    }
    initial = {
        "id": "resp-1",
        "status": "completed",
        "output": [visible, private],
        "usage": {"input_tokens": 7, "output_tokens": 5},
    }
    upstream = [
        _event({"type": "response.created", "sequence_number": 0, "response": initial}),
        _event(
            {
                "type": "response.output_item.added",
                "sequence_number": 1,
                "output_index": 0,
                "item": visible,
            }
        ),
        _event(
            {
                "type": "response.output_item.done",
                "sequence_number": 2,
                "output_index": 0,
                "item": visible,
            }
        ),
        _event(
            {
                "type": "response.output_item.added",
                "sequence_number": 3,
                "output_index": 1,
                "item": private,
            }
        ),
        _event(
            {
                "type": "response.output_item.done",
                "sequence_number": 4,
                "output_index": 1,
                "item": private,
            }
        ),
        _event(
            {
                "type": "response.completed",
                "sequence_number": 5,
                "response": initial,
            }
        ),
        b"data: [DONE]\n\n",
    ]
    final = {
        "id": "resp-2",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "msg-final",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "after"}],
            }
        ],
        "usage": {"input_tokens": 11, "output_tokens": 3},
    }

    async def continuation(messages, tools):  # noqa: ANN001
        return final

    interceptor = EventLevelCCRInterceptor(
        CCRResponseHandler(),
        provider="openai_responses",
        render_response=_openai_responses_to_sse,
    )
    output_chunks = [
        chunk
        async for chunk in interceptor.process(
            _chunks([b"".join(upstream)[:97], b"".join(upstream)[97:]]),
            [],
            None,
            continuation,
        )
    ]
    output = b"".join(output_chunks)

    assert b"headroom_retrieve" not in output
    assert b"before" in output
    assert b"after" in output
    assert output.count(b"event: response.created") == 1
    completed = [
        json.loads(line[6:])
        for line in output.splitlines()
        if line.startswith(b"data: {") and json.loads(line[6:]).get("type") == "response.completed"
    ][0]
    assert [item["id"] for item in completed["response"]["output"]] == [
        "msg-visible",
        "msg-final",
    ]
    assert completed["response"]["usage"] == {"input_tokens": 18, "output_tokens": 8}


@pytest.mark.asyncio
async def test_anthropic_mixed_private_and_client_tools_preserves_only_client_tool():
    hash_key = _stored_hash()
    upstream = [
        _event(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "tool-ccr",
                    "name": "headroom_retrieve",
                    "input": {},
                },
            }
        ),
        _event(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps({"hash": hash_key}),
                },
            }
        ),
        _event({"type": "content_block_stop", "index": 0}),
        _event(
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "tool-client",
                    "name": "write_file",
                    "input": {},
                },
            }
        ),
        _event(
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '{"path":"a"}'},
            }
        ),
        _event({"type": "content_block_stop", "index": 1}),
        _event(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 4},
            }
        ),
        _event({"type": "message_stop"}),
    ]

    async def unexpected_continuation(messages, tools):  # noqa: ANN001
        raise AssertionError("mixed client/private tools must return control to the client")

    interceptor = EventLevelCCRInterceptor(
        CCRResponseHandler(),
        provider="anthropic",
        render_response=lambda response: [],
    )
    output = b"".join(
        [
            chunk
            async for chunk in interceptor.process(
                _chunks(upstream), [], None, unexpected_continuation
            )
        ]
    )

    assert b"headroom_retrieve" not in output
    assert b"tool-ccr" not in output
    assert b"write_file" in output
    assert b"tool-client" in output
    assert b'"stop_reason": "tool_use"' in output
    assert output.count(b"event: message_stop") == 1


@pytest.mark.asyncio
async def test_shared_http_stream_loop_runs_anthropic_interceptor_and_continuation():
    hash_key = _stored_hash()
    upstream_events = [
        _event(
            {
                "type": "message_start",
                "message": {"id": "msg-1", "role": "assistant", "model": "claude"},
            }
        ),
        _event(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "tool-ccr",
                    "name": "headroom_retrieve",
                    "input": {},
                },
            }
        ),
        _event(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps({"hash": hash_key}),
                },
            }
        ),
        _event({"type": "content_block_stop", "index": 0}),
        _event(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 2},
            }
        ),
        _event({"type": "message_stop"}),
    ]
    upstream_response = AsyncMock()
    upstream_response.headers = httpx.Headers({"content-type": "text/event-stream"})
    upstream_response.status_code = 200
    upstream_response.aiter_bytes = lambda: _chunks(upstream_events)
    upstream_response.aclose = AsyncMock()

    proxy = object.__new__(HeadroomProxy)
    proxy.http_client = MagicMock(spec=httpx.AsyncClient)
    proxy.http_client.build_request = MagicMock(return_value=MagicMock())
    proxy.http_client.send = AsyncMock(return_value=upstream_response)
    proxy.config = MagicMock(
        ccr_inject_tool=True,
        retry_enabled=True,
        retry_max_attempts=1,
        retry_base_delay_ms=0,
        retry_max_delay_ms=0,
        memory_enabled=False,
    )
    proxy._config = proxy.config
    proxy.memory_handler = None
    proxy.ccr_response_handler = CCRResponseHandler()
    proxy._parse_sse_usage_from_buffer = MagicMock(return_value=None)
    proxy._finalize_stream_response = AsyncMock(return_value=None)
    proxy._record_ccr_feedback_from_response = MagicMock()
    proxy._retry_request = AsyncMock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg-2",
                "role": "assistant",
                "model": "claude",
                "content": [{"type": "text", "text": "integrated continuation"}],
                "stop_reason": "end_turn",
                "usage": {"output_tokens": 3},
            },
        )
    )

    result = await proxy._stream_response(
        url="https://api.anthropic.com/v1/messages",
        headers={"x-api-key": "test"},
        body={
            "model": "claude",
            "stream": True,
            "messages": [{"role": "user", "content": "retrieve"}],
            "tools": [
                {
                    "name": "headroom_retrieve",
                    "input_schema": {"type": "object"},
                }
            ],
        },
        provider="anthropic",
        model="claude",
        request_id="stream-integration",
        original_tokens=10,
        optimized_tokens=10,
        tokens_saved=0,
        transforms_applied=[],
        tags={},
        optimization_latency=0.0,
        session_key="stream-integration",
        ccr_stream_provider="anthropic",
    )
    output = b"".join([chunk async for chunk in result.body_iterator])

    assert b"integrated continuation" in output
    assert b"headroom_retrieve" not in output
    assert proxy._retry_request.await_count == 1
    continuation_body = proxy._retry_request.await_args.args[3]
    assert continuation_body["stream"] is False
    assert continuation_body["messages"][-1]["content"][0]["type"] == "tool_result"


@pytest.mark.asyncio
async def test_large_ordinary_stream_stays_live_after_reconstruction_window_closes(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(stream_splice_module, "MAX_CCR_STREAM_RECONSTRUCTION_BYTES", 300)
    upstream = [
        _event(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "x" * 150},
            }
        ),
        _event(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "y" * 150},
            }
        ),
        _event({"type": "message_stop"}),
    ]

    async def unexpected_continuation(messages, tools):  # noqa: ANN001
        raise AssertionError("ordinary stream must not start a continuation")

    interceptor = EventLevelCCRInterceptor(
        CCRResponseHandler(),
        provider="anthropic",
        render_response=lambda response: [],
    )
    output = b"".join(
        [
            chunk
            async for chunk in interceptor.process(
                _chunks(upstream), [], None, unexpected_continuation
            )
        ]
    )

    assert b"x" * 150 in output
    assert b"y" * 150 in output
    assert b"event: error" not in output
