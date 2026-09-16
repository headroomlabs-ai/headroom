"""Google/Gemini CCR adapter."""

from __future__ import annotations

import json
from typing import Any

from .base import CCR_TOOL_NAME, GenericCcrAdapter


class GoogleCcrAdapter(GenericCcrAdapter):
    """Google/Gemini CCR message, tool, and batch formats."""

    provider = "google"

    def tool_definition(self) -> dict[str, Any]:
        return {
            "name": CCR_TOOL_NAME,
            "description": (
                "Retrieve original uncompressed content that was compressed to save tokens. "
                "Use this when you need more data than what's shown in compressed tool results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "hash": {
                        "type": "string",
                        "description": "Hash key from the compression marker",
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional search query to filter results",
                    },
                },
                "required": ["hash"],
            },
        }

    def parse_tool_call(self, tool_call: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
        function_call = tool_call.get("functionCall", {})
        return function_call.get("name"), function_call.get("args", {})

    def extract_tool_calls(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        candidates = response.get("candidates", [])
        if not candidates:
            return []
        parts = candidates[0].get("content", {}).get("parts", [])
        return [part for part in parts if "functionCall" in part]

    def tool_call_id(self, tool_call: dict[str, Any]) -> str:
        return str(tool_call.get("functionCall", {}).get("name", CCR_TOOL_NAME))

    def tool_result_message(self, results: list[Any]) -> dict[str, Any]:
        parts = []
        for result in results:
            try:
                response_data = json.loads(result.content)
            except json.JSONDecodeError:
                response_data = {"content": result.content}
            parts.append(
                {
                    "functionResponse": {
                        "name": result.tool_call_id,
                        "response": response_data,
                    }
                }
            )
        return {"role": "user", "parts": parts}

    def assistant_message(self, response: dict[str, Any]) -> dict[str, Any]:
        candidates = response.get("candidates", [])
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        return {"role": "model", "parts": parts}

    def reconstruct_stream_response(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        del events
        return {"candidates": [{"content": {"parts": []}}]}

    def batch_custom_id(self, result: dict[str, Any]) -> str:
        metadata = result.get("metadata", {})
        return str(metadata.get("key", "") if isinstance(metadata, dict) else "")

    def continuation_url(
        self, api_urls: dict[str, str], request_context: Any, batch_context: Any
    ) -> str:
        model = request_context.model
        url = f"{api_urls['google']}/v1beta/models/{model}:generateContent"
        if batch_context.api_key:
            url = f"{url}?key={batch_context.api_key}"
        return url

    def continuation_body(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        request_context: Any,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"contents": self.messages_to_contents(messages)}
        if request_context.system_instruction:
            body["systemInstruction"] = {"parts": [{"text": request_context.system_instruction}]}
        if tools:
            body["tools"] = [{"functionDeclarations": tools}]
        return body

    def messages_to_contents(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert standard messages to Google contents format."""
        contents = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content")

            if "parts" in msg:
                google_role = "model" if role in ("assistant", "model") else "user"
                contents.append({"role": google_role, "parts": msg["parts"]})
                continue

            if role == "system":
                continue
            if role == "assistant":
                google_role = "model"
            else:
                google_role = "user"

            if isinstance(content, str):
                contents.append({"role": google_role, "parts": [{"text": content}]})
            elif isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            parts.append({"text": block.get("text", "")})
                        elif block.get("type") == "tool_result":
                            parts.append(
                                {
                                    "functionResponse": {
                                        "name": block.get("tool_use_id", CCR_TOOL_NAME),
                                        "response": {"content": block.get("content", "")},
                                    }
                                }
                            )
                        elif block.get("type") == "tool_use":
                            parts.append(
                                {
                                    "functionCall": {
                                        "name": block.get("name", ""),
                                        "args": block.get("input", {}),
                                    }
                                }
                            )
                if parts:
                    contents.append({"role": google_role, "parts": parts})

        return contents
