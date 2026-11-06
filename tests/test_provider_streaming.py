import json

from headroom.providers.streaming import (
    parse_sse_to_response,
    parse_sse_usage_buffer,
    response_to_sse,
    supports_codex_wire_debug,
    supports_prefix_response_append,
    supports_stream_memory_tools,
    uses_upstream_token_denominator,
)


def test_streaming_usage_parser_handles_provider_native_usage_shapes() -> None:
    anthropic_state = {
        "sse_buffer": bytearray(
            b'event: message_start\ndata: {"type":"message_start","message":{"usage":'
            b'{"input_tokens":10,"cache_read_input_tokens":2,'
            b'"cache_creation_input_tokens":3}}}\n\n'
        )
    }
    openai_state = {
        "sse_buffer": bytearray(
            b'data: {"usage":{"prompt_tokens":11,"completion_tokens":7,'
            b'"prompt_tokens_details":{"cached_tokens":4}}}\n\n'
        )
    }
    gemini_state = {
        "sse_buffer": bytearray(
            b'data: {"usageMetadata":{"promptTokenCount":12,'
            b'"candidatesTokenCount":8,"cachedContentTokenCount":5}}\n\n'
        )
    }

    assert parse_sse_usage_buffer(anthropic_state, "anthropic") == {
        "input_tokens": 10,
        "cache_read_input_tokens": 2,
        "cache_creation_input_tokens": 3,
        "cache_creation_ephemeral_5m_input_tokens": 0,
        "cache_creation_ephemeral_1h_input_tokens": 0,
    }
    assert parse_sse_usage_buffer(openai_state, "openai") == {
        "input_tokens": 11,
        "output_tokens": 7,
        "cache_read_input_tokens": 4,
    }
    assert parse_sse_usage_buffer(gemini_state, "gemini") == {
        "input_tokens": 12,
        "output_tokens": 8,
        "cache_read_input_tokens": 5,
    }


def test_streaming_response_parser_reconstructs_anthropic_tool_use() -> None:
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "model": "claude",
                "role": "assistant",
                "usage": {"input_tokens": 3},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "tool_1", "name": "headroom_retrieve"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"hash":"abc123"}'},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
    ]
    sse = "".join(f"data: {json.dumps(event)}\n\n" for event in events)

    response = parse_sse_to_response(sse, "anthropic")

    assert response is not None
    assert response["content"][0]["type"] == "tool_use"
    assert response["content"][0]["input"] == {"hash": "abc123"}
    assert response["stop_reason"] == "tool_use"
    assert response_to_sse(response, "anthropic")


def test_streaming_provider_policy_helpers_keep_core_branch_free() -> None:
    assert supports_stream_memory_tools("anthropic") is True
    assert supports_stream_memory_tools("openai") is False
    assert supports_prefix_response_append("anthropic") is True
    assert uses_upstream_token_denominator("openai") is True
    assert uses_upstream_token_denominator("gemini") is True
    assert uses_upstream_token_denominator("anthropic") is False
    assert supports_codex_wire_debug("openai", "https://api.openai.com/v1/responses") is True
    assert supports_codex_wire_debug("anthropic", "https://api.anthropic.com/v1/messages") is False
