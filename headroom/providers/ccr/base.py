"""Shared CCR provider adapter contract."""

from __future__ import annotations

import json
from typing import Any, Protocol

CCR_TOOL_NAME = "headroom_retrieve"

CCR_TOOL_DESCRIPTION = (
    "Retrieve original uncompressed content that was compressed to save tokens. "
    "Use this when you need more data than what's shown in compressed tool results. "
    "The hash is provided in compression markers like [N items compressed... hash=abc123]."
)

CCR_TOOL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "hash": {
            "type": "string",
            "description": "Hash key from the compression marker (e.g., 'abc123' from hash=abc123)",
        },
        "query": {
            "type": "string",
            "description": (
                "Optional search query to filter results. "
                "If provided, only returns items matching the query. "
                "If omitted, returns all original items."
            ),
        },
    },
    "required": ["hash"],
}


class ProviderCcrAdapter(Protocol):
    """Provider-specific CCR formatting behavior."""

    provider: str

    def tool_definition(self) -> dict[str, Any]:
        """Return this provider's CCR tool definition."""
        ...

    def parse_tool_call(self, tool_call: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
        """Return the tool name and parsed input payload for a provider tool call."""
        ...

    def extract_tool_calls(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract provider-native tool calls from a response payload."""
        ...

    def tool_call_id(self, tool_call: dict[str, Any]) -> str:
        """Return the provider-native identifier used for a tool result."""
        ...

    def tool_result_message(self, results: list[Any]) -> dict[str, Any]:
        """Build provider-native tool-result message data."""
        ...

    def retrieval_tool_result(self, tool_call_id: str, content: str) -> dict[str, Any]:
        """Build provider-native single retrieval tool-result data."""
        ...

    def assistant_message(self, response: dict[str, Any]) -> dict[str, Any]:
        """Extract the assistant/model message for continuation."""
        ...

    def append_tool_result_messages(
        self,
        messages: list[dict[str, Any]],
        tool_result_message: dict[str, Any],
    ) -> None:
        """Append provider-native tool result messages to the continuation."""
        messages.append(tool_result_message)

    def reconstruct_stream_response(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        """Reconstruct a non-streaming response from provider SSE events."""
        ...

    def response_to_sse_chunks(self, response: dict[str, Any]) -> list[bytes]:
        """Convert a response payload to provider SSE chunks."""
        ...

    def batch_custom_id(self, result: dict[str, Any]) -> str:
        """Extract a provider batch result custom id."""
        return str(result.get("custom_id", result.get("id", "")))

    def batch_response(self, result: dict[str, Any]) -> dict[str, Any] | None:
        """Extract a provider response payload from a batch result."""
        response = result.get("response")
        return response if isinstance(response, dict) else None

    def update_batch_result(
        self,
        original_result: dict[str, Any],
        final_response: dict[str, Any],
    ) -> dict[str, Any]:
        """Return a batch result updated with the final provider response."""
        result = dict(original_result)
        result["response"] = final_response
        return result

    def continuation_url(
        self,
        api_urls: dict[str, str],
        request_context: Any,
        batch_context: Any,
    ) -> str:
        """Build the provider continuation URL."""
        raise ValueError(f"Unknown provider: {self.provider}")

    def continuation_headers(self, batch_context: Any) -> dict[str, str]:
        """Build provider continuation request headers."""
        return {"Content-Type": "application/json"}

    def continuation_body(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        request_context: Any,
    ) -> dict[str, Any]:
        """Build the provider continuation request body."""
        return {"model": request_context.model, "messages": messages, "tools": tools or []}

    def messages_to_contents(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert standard messages into Google-style contents when supported."""
        return messages


class GenericCcrAdapter:
    """Fallback CCR adapter for unknown provider strings."""

    provider = "generic"

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
        return tool_call.get("name"), tool_call.get("input", tool_call.get("args", {}))

    def extract_tool_calls(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        return []

    def tool_call_id(self, tool_call: dict[str, Any]) -> str:
        return str(tool_call.get("id", ""))

    def tool_result_message(self, results: list[Any]) -> dict[str, Any]:
        return {
            "role": "tool",
            "content": json.dumps(
                [{"tool_call_id": r.tool_call_id, "result": r.content} for r in results]
            ),
        }

    def retrieval_tool_result(self, tool_call_id: str, content: str) -> dict[str, Any]:
        return {
            "tool_call_id": tool_call_id,
            "content": content,
        }

    def assistant_message(self, response: dict[str, Any]) -> dict[str, Any]:
        return {"role": "assistant", "content": response.get("content", "")}

    def append_tool_result_messages(
        self,
        messages: list[dict[str, Any]],
        tool_result_message: dict[str, Any],
    ) -> None:
        """Append provider-native tool result messages to the continuation."""
        messages.append(tool_result_message)

    def reconstruct_stream_response(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        del events
        return {"content": ""}

    def response_to_sse_chunks(self, response: dict[str, Any]) -> list[bytes]:
        """Convert a response payload to generic OpenAI-style SSE chunks."""
        return [f"data: {json.dumps(response)}\n\n".encode(), b"data: [DONE]\n\n"]

    def batch_custom_id(self, result: dict[str, Any]) -> str:
        """Extract a provider batch result custom id."""
        return str(result.get("custom_id", result.get("id", "")))

    def batch_response(self, result: dict[str, Any]) -> dict[str, Any] | None:
        """Extract a provider response payload from a batch result."""
        response = result.get("response")
        return response if isinstance(response, dict) else None

    def update_batch_result(
        self,
        original_result: dict[str, Any],
        final_response: dict[str, Any],
    ) -> dict[str, Any]:
        """Return a batch result updated with the final provider response."""
        result = dict(original_result)
        result["response"] = final_response
        return result

    def continuation_url(
        self,
        api_urls: dict[str, str],
        request_context: Any,
        batch_context: Any,
    ) -> str:
        """Build the provider continuation URL."""
        del api_urls, request_context, batch_context
        raise ValueError(f"Unknown provider: {self.provider}")

    def continuation_headers(self, batch_context: Any) -> dict[str, str]:
        """Build provider continuation request headers."""
        del batch_context
        return {"Content-Type": "application/json"}

    def continuation_body(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        request_context: Any,
    ) -> dict[str, Any]:
        """Build the provider continuation request body."""
        return {"model": request_context.model, "messages": messages, "tools": tools or []}

    def messages_to_contents(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert standard messages into Google-style contents when supported."""
        return messages
