"""OpenAI CCR adapter."""

from __future__ import annotations

import json
from typing import Any

from .base import CCR_TOOL_DESCRIPTION, CCR_TOOL_NAME, CCR_TOOL_PARAMETERS, GenericCcrAdapter


class OpenAICcrAdapter(GenericCcrAdapter):
    """OpenAI CCR message, tool, streaming, and batch formats."""

    provider = "openai"

    def tool_definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": CCR_TOOL_NAME,
                "description": CCR_TOOL_DESCRIPTION,
                "parameters": CCR_TOOL_PARAMETERS,
            },
        }

    def parse_tool_call(self, tool_call: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
        function = tool_call.get("function", {})
        args_str = function.get("arguments", "{}")
        try:
            input_data = json.loads(args_str)
        except json.JSONDecodeError:
            input_data = {}
        return function.get("name"), input_data

    def extract_tool_calls(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        message = response.get("choices", [{}])[0].get("message", {})
        tool_calls = message.get("tool_calls", [])
        return list(tool_calls) if tool_calls else []

    def tool_call_id(self, tool_call: dict[str, Any]) -> str:
        return str(tool_call.get("id", ""))

    def tool_result_message(self, results: list[Any]) -> dict[str, Any]:
        return {
            "_openai_tool_results": [
                {
                    "role": "tool",
                    "tool_call_id": result.tool_call_id,
                    "content": result.content,
                }
                for result in results
            ]
        }

    def retrieval_tool_result(self, tool_call_id: str, content: str) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }

    def append_tool_result_messages(
        self,
        messages: list[dict[str, Any]],
        tool_result_message: dict[str, Any],
    ) -> None:
        messages.extend(tool_result_message.get("_openai_tool_results", []))

    def assistant_message(self, response: dict[str, Any]) -> dict[str, Any]:
        message = response.get("choices", [{}])[0].get("message", {})
        return {
            "role": "assistant",
            "content": message.get("content"),
            "tool_calls": message.get("tool_calls"),
        }

    def reconstruct_stream_response(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        message: dict[str, Any] = {
            "role": "assistant",
            "content": "",
            "tool_calls": [],
        }
        tool_calls_map: dict[int, dict[str, Any]] = {}

        for event in events:
            choices = event.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})

            if "content" in delta and delta["content"]:
                message["content"] = (message.get("content") or "") + delta["content"]

            if "tool_calls" in delta:
                for tc_delta in delta["tool_calls"]:
                    idx = tc_delta.get("index", 0)
                    if idx not in tool_calls_map:
                        tool_calls_map[idx] = {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }

                    tc = tool_calls_map[idx]
                    if "id" in tc_delta:
                        tc["id"] = tc_delta["id"]
                    if "function" in tc_delta:
                        fn = tc_delta["function"]
                        if "name" in fn:
                            tc["function"]["name"] = fn["name"]
                        if "arguments" in fn:
                            tc["function"]["arguments"] += fn["arguments"]

        message["tool_calls"] = [tool_calls_map[i] for i in sorted(tool_calls_map.keys())]
        if not message["tool_calls"]:
            del message["tool_calls"]
        if not message["content"]:
            message["content"] = None

        return {"choices": [{"message": message, "finish_reason": "stop"}]}

    def batch_custom_id(self, result: dict[str, Any]) -> str:
        return str(result.get("custom_id", ""))

    def batch_response(self, result: dict[str, Any]) -> dict[str, Any] | None:
        inner = result.get("response", {})
        response = inner.get("body") if isinstance(inner, dict) else None
        return response if isinstance(response, dict) else None

    def update_batch_result(
        self, original_result: dict[str, Any], final_response: dict[str, Any]
    ) -> dict[str, Any]:
        result = dict(original_result)
        if "response" not in result:
            result["response"] = {}
        result["response"]["body"] = final_response
        return result

    def continuation_url(
        self, api_urls: dict[str, str], request_context: Any, batch_context: Any
    ) -> str:
        del request_context, batch_context
        return f"{api_urls['openai']}/v1/chat/completions"

    def continuation_headers(self, batch_context: Any) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if batch_context.api_key:
            headers["Authorization"] = f"Bearer {batch_context.api_key}"
        return headers

    def continuation_body(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        request_context: Any,
    ) -> dict[str, Any]:
        body = {"model": request_context.model, "messages": messages}
        if tools:
            body["tools"] = tools
        return body
