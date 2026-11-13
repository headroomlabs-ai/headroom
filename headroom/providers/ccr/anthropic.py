"""Anthropic CCR adapter."""

from __future__ import annotations

import json
from typing import Any

from .base import CCR_TOOL_DESCRIPTION, CCR_TOOL_NAME, CCR_TOOL_PARAMETERS, GenericCcrAdapter


def _tool_sort_key(tool: dict[str, Any]) -> tuple[str, str]:
    name = (
        str(tool.get("name", ""))
        or str(tool.get("function", {}).get("name", ""))
        or str(tool.get("type", ""))
    )
    try:
        canonical = json.dumps(tool, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        canonical = str(tool)
    return (name, canonical)


class AnthropicCcrAdapter(GenericCcrAdapter):
    """Anthropic CCR message, tool, streaming, and batch formats."""

    provider = "anthropic"

    def tool_definition(self) -> dict[str, Any]:
        return {
            "name": CCR_TOOL_NAME,
            "description": CCR_TOOL_DESCRIPTION,
            "input_schema": CCR_TOOL_PARAMETERS,
        }

    def parse_tool_call(self, tool_call: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
        return tool_call.get("name"), tool_call.get("input", {})

    def extract_tool_calls(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        content = response.get("content", [])
        if isinstance(content, list):
            return [block for block in content if block.get("type") == "tool_use"]
        return []

    def tool_call_id(self, tool_call: dict[str, Any]) -> str:
        return str(tool_call.get("id", ""))

    def tool_result_message(self, results: list[Any]) -> dict[str, Any]:
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": result.tool_call_id,
                    "content": result.content,
                }
                for result in results
            ],
        }

    def retrieval_tool_result(self, tool_call_id: str, content: str) -> dict[str, Any]:
        return {
            "type": "tool_result",
            "tool_use_id": tool_call_id,
            "content": content,
        }

    def assistant_message(self, response: dict[str, Any]) -> dict[str, Any]:
        return {"role": "assistant", "content": response.get("content", [])}

    def reconstruct_stream_response(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        response: dict[str, Any] = {"content": [], "stop_reason": None, "usage": {}}
        current_text = ""
        current_tool: dict[str, Any] | None = None

        for event in events:
            event_type = event.get("type", "")

            if event_type == "content_block_start":
                block = event.get("content_block", {})
                if block.get("type") == "text":
                    current_text = block.get("text", "")
                elif block.get("type") == "tool_use":
                    current_tool = {
                        "type": "tool_use",
                        "id": block.get("id", ""),
                        "name": block.get("name", ""),
                        "input": {},
                    }

            elif event_type == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    current_text += delta.get("text", "")
                elif delta.get("type") == "input_json_delta" and current_tool is not None:
                    current_tool["_partial_json"] = current_tool.get("_partial_json", "") + (
                        delta.get("partial_json", "")
                    )

            elif event_type == "content_block_stop":
                if current_text:
                    response["content"].append({"type": "text", "text": current_text})
                    current_text = ""
                if current_tool:
                    partial = current_tool.pop("_partial_json", "")
                    if partial:
                        try:
                            current_tool["input"] = json.loads(partial)
                        except json.JSONDecodeError:
                            current_tool["input"] = {}
                    response["content"].append(current_tool)
                    current_tool = None

            elif event_type == "message_delta":
                delta = event.get("delta", {})
                if "stop_reason" in delta:
                    response["stop_reason"] = delta["stop_reason"]

        return response

    def response_to_sse_chunks(self, response: dict[str, Any]) -> list[bytes]:
        return [
            b"event: message_start\n",
            f"data: {json.dumps({'type': 'message_start', 'message': response})}\n\n".encode(),
            b"event: message_stop\n",
            b'data: {"type": "message_stop"}\n\n',
        ]

    def batch_custom_id(self, result: dict[str, Any]) -> str:
        return str(result.get("custom_id", ""))

    def batch_response(self, result: dict[str, Any]) -> dict[str, Any] | None:
        inner = result.get("result", {})
        response = inner.get("message") if isinstance(inner, dict) else None
        return response if isinstance(response, dict) else None

    def update_batch_result(
        self, original_result: dict[str, Any], final_response: dict[str, Any]
    ) -> dict[str, Any]:
        result = dict(original_result)
        if "result" not in result:
            result["result"] = {}
        result["result"]["message"] = final_response
        result["result"]["type"] = "succeeded"
        return result

    def continuation_url(
        self, api_urls: dict[str, str], request_context: Any, batch_context: Any
    ) -> str:
        del request_context, batch_context
        return f"{api_urls['anthropic']}/v1/messages"

    def continuation_headers(self, batch_context: Any) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
        if batch_context.api_key:
            headers["x-api-key"] = batch_context.api_key
        return headers

    def continuation_body(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        request_context: Any,
    ) -> dict[str, Any]:
        body = {
            "model": request_context.model,
            "messages": messages,
            "max_tokens": request_context.extras.get("max_tokens", 4096),
        }
        if tools:
            body["tools"] = sorted(tools, key=_tool_sort_key)
        return body
