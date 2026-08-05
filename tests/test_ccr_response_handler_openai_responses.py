"""Tests for CCR response handling of the OpenAI Responses API shape.

Covers #1877: `CCRResponseHandler` previously only understood "anthropic",
"openai" (chat completions), and "google" response shapes. Responses API
function calls are flat `function_call` items in a top-level `output[]`
array (not nested under `choices[].message.tool_calls`), and results are
`function_call_output` items appended to `input[]` rather than a single
role/content message — these tests exercise the new "openai_responses"
provider branch end to end.
"""

from __future__ import annotations

import json

import pytest

from headroom.cache.compression_store import (
    get_compression_store,
    reset_compression_store,
)
from headroom.ccr.response_handler import CCRResponseHandler, CCRToolResult
from headroom.ccr.tool_injection import CCR_TOOL_NAME, parse_tool_call


@pytest.fixture(autouse=True)
def reset_store():
    reset_compression_store()
    yield
    reset_compression_store()


def _function_call_response(hash_key: str, call_id: str = "call_abc") -> dict:
    return {
        "id": "resp_1",
        "object": "response",
        "status": "completed",
        "output": [
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [],
            },
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": call_id,
                "name": CCR_TOOL_NAME,
                "arguments": json.dumps({"hash": hash_key}),
            },
        ],
        "usage": {"input_tokens": 50, "output_tokens": 10},
    }


class TestOpenAIResponsesDetection:
    def test_detect_function_call_tool_call(self) -> None:
        handler = CCRResponseHandler()
        response = _function_call_response("abc123def456abc123def456")

        assert handler.has_ccr_tool_calls(response, "openai_responses")

    def test_no_false_positive_for_other_function_call(self) -> None:
        handler = CCRResponseHandler()
        response = {
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "some_other_tool",
                    "arguments": "{}",
                }
            ]
        }

        assert not handler.has_ccr_tool_calls(response, "openai_responses")

    def test_no_false_positive_for_message_only_output(self) -> None:
        handler = CCRResponseHandler()
        response = {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hi"}],
                }
            ]
        }

        assert not handler.has_ccr_tool_calls(response, "openai_responses")

    def test_empty_output(self) -> None:
        handler = CCRResponseHandler()
        assert not handler.has_ccr_tool_calls({"output": []}, "openai_responses")
        assert not handler.has_ccr_tool_calls({}, "openai_responses")


class TestOpenAIResponsesParsing:
    def test_parse_extracts_call_id_not_item_id(self) -> None:
        """`call_id` (not the function_call item's own `id`) matches the
        `function_call_output.call_id` the continuation must echo back."""
        handler = CCRResponseHandler()
        response = _function_call_response("abc123def456abc123def456", call_id="call_xyz")

        ccr_calls, other_calls = handler._parse_ccr_tool_calls(response, "openai_responses")

        assert len(ccr_calls) == 1
        assert ccr_calls[0].tool_call_id == "call_xyz"
        assert ccr_calls[0].hash_key == "abc123def456abc123def456"
        assert not other_calls

    def test_parse_tool_call_direct(self) -> None:
        """`parse_tool_call` reads flat name/arguments, not a nested `function` key."""
        tool_call = {
            "type": "function_call",
            "call_id": "call_1",
            "name": CCR_TOOL_NAME,
            "arguments": '{"hash": "abc123def456abc123def456"}',
        }

        assert parse_tool_call(tool_call, "openai_responses") == "abc123def456abc123def456"

    def test_parse_tool_call_rejects_other_names(self) -> None:
        tool_call = {
            "type": "function_call",
            "call_id": "call_1",
            "name": "read_file",
            "arguments": '{"path": "/etc/config"}',
        }

        assert parse_tool_call(tool_call, "openai_responses") is None

    def test_parse_tool_call_malformed_arguments(self) -> None:
        tool_call = {
            "type": "function_call",
            "call_id": "call_1",
            "name": CCR_TOOL_NAME,
            "arguments": "not json",
        }

        assert parse_tool_call(tool_call, "openai_responses") is None


class TestOpenAIResponsesMessageShaping:
    def test_extract_assistant_message_echoes_full_output_array(self) -> None:
        handler = CCRResponseHandler()
        response = _function_call_response("abc123def456abc123def456")

        result = handler._extract_assistant_message(response, "openai_responses")

        assert result == {"_openai_responses_output_items": response["output"]}

    def test_create_tool_result_message_uses_call_id(self) -> None:
        handler = CCRResponseHandler()
        results = [
            CCRToolResult(tool_call_id="call_xyz", content='{"data": "x"}', success=True),
            CCRToolResult(tool_call_id="call_abc", content='{"data": "y"}', success=True),
        ]

        message = handler._create_tool_result_message(results, "openai_responses")

        assert "_openai_responses_tool_results" in message
        items = message["_openai_responses_tool_results"]
        assert len(items) == 2
        assert items[0] == {
            "type": "function_call_output",
            "call_id": "call_xyz",
            "output": '{"data": "x"}',
        }


class TestOpenAIResponsesHandleResponse:
    @pytest.mark.asyncio
    async def test_handle_response_resolves_retrieve_and_extends_input(self) -> None:
        store = get_compression_store()
        original = json.dumps([{"id": i} for i in range(30)])
        hash_key = store.store(original=original, compressed="[]", original_item_count=30)

        handler = CCRResponseHandler()
        initial_response = _function_call_response(hash_key, call_id="call_1")
        final_response = {
            "id": "resp_2",
            "object": "response",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Here are all 30 items."}],
                }
            ],
        }

        captured_calls: list[list[dict]] = []

        async def mock_api_call(items, tools):
            captured_calls.append(items)
            return final_response

        result = await handler.handle_response(
            initial_response,
            [{"role": "user", "content": "get the data"}],
            None,
            mock_api_call,
            "openai_responses",
        )

        assert result == final_response
        assert len(captured_calls) == 1
        # Original input item + the two echoed output items (reasoning +
        # function_call) + the function_call_output — extended, not
        # appended as a single blob.
        sent_items = captured_calls[0]
        assert sent_items[0] == {"role": "user", "content": "get the data"}
        assert {"type": "function_call", "name": CCR_TOOL_NAME} in [
            {"type": i.get("type"), "name": i.get("name")}
            for i in sent_items
            if i.get("type") == "function_call"
        ]
        tool_outputs = [i for i in sent_items if i.get("type") == "function_call_output"]
        assert len(tool_outputs) == 1
        assert tool_outputs[0]["call_id"] == "call_1"

    @pytest.mark.asyncio
    async def test_handle_response_no_ccr_passthrough(self) -> None:
        handler = CCRResponseHandler()
        response = {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "no tool call here"}],
                }
            ]
        }

        async def mock_api_call(items, tools):
            raise AssertionError("should not be called")

        result = await handler.handle_response(
            response, [], None, mock_api_call, "openai_responses"
        )

        assert result == response
