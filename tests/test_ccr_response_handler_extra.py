from __future__ import annotations

import json
from typing import Any

import pytest

from headroom.ccr.response_handler import (
    CCRResponseHandler,
    CCRToolCall,
    CCRToolResult,
    StreamingCCRBuffer,
    StreamingCCRHandler,
)
from headroom.ccr.tool_injection import CCR_TOOL_NAME


class FakeStore:
    def __init__(
        self, *, search_error: Exception | None = None, retrieve_error: Exception | None = None
    ) -> None:
        self.search_error = search_error
        self.retrieve_error = retrieve_error

    def search(self, hash_key: str, query: str) -> list[dict[str, str]]:
        if self.search_error:
            raise self.search_error
        return [{"id": "1", "text": query}]

    def retrieve(self, hash_key: str):
        if self.retrieve_error:
            raise self.retrieve_error
        return {"unexpected": True}


async def _async_iter(items: list[bytes]):
    for item in items:
        yield item


def test_extract_tool_calls_google_and_invalid_shapes() -> None:
    handler = CCRResponseHandler()
    google_response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "hello"},
                        {"functionCall": {"name": CCR_TOOL_NAME, "args": {"hash": "abc"}}},
                    ]
                }
            }
        ]
    }
    assert handler._extract_tool_calls(google_response, "google") == [
        {"functionCall": {"name": CCR_TOOL_NAME, "args": {"hash": "abc"}}}
    ]
    assert handler._extract_tool_calls({"content": "bad"}, "anthropic") == []
    with pytest.raises(IndexError):
        handler._extract_tool_calls({"choices": []}, "openai")
    assert handler._extract_tool_calls({"candidates": []}, "google") == []


def test_parse_ccr_tool_calls_google_and_other_calls() -> None:
    handler = CCRResponseHandler()
    response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "name": CCR_TOOL_NAME,
                                "args": {
                                    "hash": "aaaaaaaaaaaaaaaaaaaaaaaa",
                                    "query": "pizza",
                                },
                            }
                        },
                        {"functionCall": {"name": "other_tool", "args": {}}},
                    ]
                }
            }
        ]
    }
    ccr_calls, other_calls = handler._parse_ccr_tool_calls(response, "google")
    assert ccr_calls == [
        CCRToolCall(
            tool_call_id=CCR_TOOL_NAME,
            hash_key="aaaaaaaaaaaaaaaaaaaaaaaa",
            query="pizza",
        )
    ]
    assert other_calls == [{"functionCall": {"name": "other_tool", "args": {}}}]


def test_execute_retrieval_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    handler = CCRResponseHandler()
    monkeypatch.setattr(
        "headroom.ccr.response_handler.get_compression_store",
        lambda: FakeStore(search_error=RuntimeError("search boom")),
    )
    search_result = handler._execute_retrieval(
        CCRToolCall(tool_call_id="t1", hash_key="abc", query="find")
    )
    assert search_result.success is False
    assert "Retrieval failed: search boom" in search_result.content

    monkeypatch.setattr(
        "headroom.ccr.response_handler.get_compression_store",
        lambda: FakeStore(retrieve_error=RuntimeError("retrieve boom")),
    )
    retrieve_result = handler._execute_retrieval(CCRToolCall(tool_call_id="t2", hash_key="abc"))
    assert retrieve_result.success is False
    assert "Retrieval failed: retrieve boom" in retrieve_result.content


def test_create_tool_result_message_google_and_generic_formats() -> None:
    handler = CCRResponseHandler()
    results = [
        CCRToolResult(tool_call_id="headroom_retrieve", content='{"count": 1}', success=True)
    ]
    google_message = handler._create_tool_result_message(results, "google")
    assert google_message == {
        "role": "user",
        "parts": [{"functionResponse": {"name": "headroom_retrieve", "response": {"count": 1}}}],
    }

    generic_message = handler._create_tool_result_message(
        [CCRToolResult(tool_call_id="tool-1", content="not-json", success=False)],
        "other",
    )
    assert generic_message["role"] == "tool"
    assert json.loads(generic_message["content"]) == [
        {"tool_call_id": "tool-1", "result": "not-json"}
    ]

    invalid_google = handler._create_tool_result_message(
        [CCRToolResult(tool_call_id="headroom_retrieve", content="not-json", success=True)],
        "google",
    )
    assert invalid_google["parts"][0]["functionResponse"]["response"] == {"content": "not-json"}


def test_extract_assistant_message_google_and_generic() -> None:
    handler = CCRResponseHandler()
    google_message = handler._extract_assistant_message(
        {"candidates": [{"content": {"parts": [{"text": "hello"}]}}]},
        "google",
    )
    assert google_message == {"role": "model", "parts": [{"text": "hello"}]}

    assert handler._extract_assistant_message({}, "google") == {"role": "model", "parts": []}
    assert handler._extract_assistant_message({"content": "plain"}, "other") == {
        "role": "assistant",
        "content": "plain",
    }


@pytest.mark.asyncio
async def test_handle_response_openai_success_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    handler = CCRResponseHandler()
    initial_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": CCR_TOOL_NAME,
                                "arguments": '{"hash":"aaaaaaaaaaaaaaaaaaaaaaaa"}',
                            },
                        }
                    ],
                }
            }
        ]
    }
    monkeypatch.setattr(
        handler,
        "_execute_retrieval",
        lambda call: CCRToolResult(
            tool_call_id=call.tool_call_id,
            content='{"hash":"aaaaaaaaaaaaaaaaaaaaaaaa"}',
            success=True,
        ),
    )

    captured_messages: list[list[dict[str, Any]]] = []

    async def success_api_call(messages, tools):
        captured_messages.append(messages)
        return {"choices": [{"message": {"role": "assistant", "content": "done"}}]}

    result = await handler.handle_response(
        initial_response, [{"role": "user", "content": "hi"}], [], success_api_call, "openai"
    )
    assert result == {"choices": [{"message": {"role": "assistant", "content": "done"}}]}
    assert captured_messages[0][1]["role"] == "assistant"
    assert captured_messages[0][2]["role"] == "tool"
    assert handler.get_stats()["total_retrievals"] == 1

    async def failing_api_call(messages, tools):
        raise RuntimeError("continuation failed")

    failed = await handler.handle_response(initial_response, [], [], failing_api_call, "openai")
    assert failed == initial_response


def test_streaming_buffer_and_parse_sse_helpers() -> None:
    buffer = StreamingCCRBuffer()
    assert buffer.add_chunk(b"plain") is False
    assert buffer.get_accumulated() == b"plain"

    handler = StreamingCCRHandler(CCRResponseHandler(), provider="anthropic")
    # Per SSE spec each event is terminated by `\n\n`. The byte-buffer
    # parser introduced in PR-A8 requires the spec terminator so partial
    # multi-byte UTF-8 reads don't corrupt event boundaries.
    anthropic_data = b"\n\n".join(
        [
            b'data: {"type":"content_block_start","content_block":{"type":"text","text":"Hel"}}',
            b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"lo"}}',
            b'data: {"type":"content_block_stop"}',
            b'data: {"type":"content_block_start","content_block":{"type":"tool_use","id":"tool_1","name":"headroom_retrieve"}}',
            b'data: {"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"{\\"hash\\":\\"abc\\"}"}}',
            b'data: {"type":"content_block_stop"}',
            b'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"}}',
            b"data: [DONE]\n\n",
        ]
    )
    parsed = handler._parse_sse_stream(anthropic_data)
    assert parsed["content"][0] == {"type": "text", "text": "Hello"}
    assert parsed["content"][1]["name"] == "headroom_retrieve"
    assert parsed["content"][1]["input"] == {"hash": "abc"}
    assert parsed["stop_reason"] == "tool_use"

    openai_handler = StreamingCCRHandler(CCRResponseHandler(), provider="openai")
    parsed_openai = openai_handler._reconstruct_openai_response(
        [
            {"choices": [{"delta": {"content": "Hi"}}]},
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {
                                        "name": "headroom_retrieve",
                                        "arguments": '{"hash":"aaaaaaaaaaaa',
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": 'aaaaaaaaaaaa"}'},
                                }
                            ]
                        }
                    }
                ]
            },
        ]
    )
    message = parsed_openai["choices"][0]["message"]
    assert message["content"] == "Hi"
    assert message["tool_calls"][0]["id"] == "call_1"
    assert message["tool_calls"][0]["function"]["arguments"] == (
        '{"hash":"aaaaaaaaaaaaaaaaaaaaaaaa"}'
    )


@pytest.mark.asyncio
async def test_streaming_handler_process_stream_pass_through_and_ccr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response_handler = CCRResponseHandler()
    handler = StreamingCCRHandler(response_handler, provider="anthropic")

    passthrough_chunks = [
        b'data: {"type":"content_block_delta","delta":{"text":"hello"}}',
        b'data: {"stop_reason":"end_turn"}',
    ]
    yielded = [
        chunk
        async for chunk in handler.process_stream(
            _async_iter(passthrough_chunks), [], None, lambda m, t: None
        )
    ]
    assert yielded == passthrough_chunks

    ccr_handler = StreamingCCRHandler(response_handler, provider="anthropic")
    monkeypatch.setattr(
        ccr_handler,
        "_parse_sse_stream",
        lambda data: {
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_1",
                    "name": CCR_TOOL_NAME,
                    "input": {"hash": "abc"},
                }
            ]
        },
    )

    async def fake_handle_response(response, messages, tools, api_call_fn, provider):  # noqa: ANN001
        return {"content": [{"type": "text", "text": "done"}]}

    async def fake_response_to_sse(response):  # noqa: ANN001
        yield b"event: message_start\n"
        yield b"event: message_stop\n"

    monkeypatch.setattr(response_handler, "handle_response", fake_handle_response)
    monkeypatch.setattr(ccr_handler, "_response_to_sse", fake_response_to_sse)

    ccr_chunks = [
        b'{"type":"tool_use","name":"headroom_retrieve"',
        b',"stop_reason":"tool_use"}',
        b"tail",
    ]
    streamed = [
        chunk
        async for chunk in ccr_handler.process_stream(
            _async_iter(ccr_chunks), [], None, lambda m, t: None
        )
    ]
    assert streamed == [b"event: message_start\n", b"event: message_stop\n"]


@pytest.mark.asyncio
async def test_streaming_handler_falls_back_to_buffer_on_processing_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response_handler = CCRResponseHandler()
    handler = StreamingCCRHandler(response_handler, provider="openai")
    monkeypatch.setattr(
        handler,
        "_parse_sse_stream",
        lambda data: (_ for _ in ()).throw(RuntimeError("parse failed")),
    )

    chunks = [b'{"type":"tool_use","name":"headroom_retrieve"', b',"stop_reason":"tool_use"}']
    streamed = [
        chunk
        async for chunk in handler.process_stream(_async_iter(chunks), [], None, lambda m, t: None)
    ]
    assert streamed == [b"".join(chunks)]


@pytest.mark.asyncio
async def test_response_to_sse_formats() -> None:
    anthropic = StreamingCCRHandler(CCRResponseHandler(), provider="anthropic")
    anthropic_chunks = [chunk async for chunk in anthropic._response_to_sse({"content": []})]
    assert anthropic_chunks[0] == b"event: message_start\n"
    assert anthropic_chunks[-1] == b'data: {"type": "message_stop"}\n\n'

    openai = StreamingCCRHandler(CCRResponseHandler(), provider="openai")
    openai_chunks = [chunk async for chunk in openai._response_to_sse({"choices": []})]
    assert openai_chunks == [b'data: {"choices": []}\n\n', b"data: [DONE]\n\n"]
