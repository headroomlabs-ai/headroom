"""Qualified Anthropic Messages request adapter."""

from __future__ import annotations

import json
from typing import Any

from headroom.proxy.gateway.protocols.content import (
    ContentBlock,
    Conversation,
    Message,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from headroom.proxy.gateway.protocols.openai import (
    _inline_image,
    _optional_int,
    _optional_number,
    _reject_unknown,
    _unsupported,
)

_FIELDS = frozenset({"model", "system", "messages", "max_tokens", "temperature", "stream", "tools"})


def _decode_anthropic(payload: dict[str, Any]) -> Conversation:
    _reject_unknown(payload, _FIELDS)
    if "stream" in payload and not isinstance(payload["stream"], bool):
        _unsupported("stream")
    system = _blocks(payload.get("system", ""))
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        _unsupported("messages")
    messages: list[Message] = []
    for item in raw_messages:
        if not isinstance(item, dict) or set(item) != {"role", "content"}:
            _unsupported("message")
        role = item["role"]
        if role not in ("user", "assistant"):
            _unsupported("message role")
        content, calls, results = _message_blocks(item["content"])
        messages.append(Message(role=role, content=content, tool_calls=calls, tool_results=results))
    return Conversation(
        model=payload.get("model") if isinstance(payload.get("model"), str) else None,
        system=system,
        messages=tuple(messages),
        tools=_decode_tools(payload.get("tools", [])),
        max_tokens=_optional_int(payload, "max_tokens"),
        temperature=_optional_number(payload, "temperature"),
        stream=payload.get("stream", False) is True,
    )


def _encode_anthropic(conversation: Conversation) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    for message in conversation.messages:
        blocks: list[dict[str, Any]] = []
        for block in message.content:
            if block.kind == "text":
                blocks.append({"type": "text", "text": block.text})
            elif block.kind == "image":
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": block.media_type,
                            "data": block.data,
                        },
                    }
                )
        blocks.extend(
            {
                "type": "tool_use",
                "id": call.id,
                "name": call.name,
                "input": json.loads(call.arguments),
            }
            for call in message.tool_calls
        )
        result_blocks = [
            {
                "type": "tool_result",
                "tool_use_id": tool_result.call_id,
                "content": "".join(block.text or "" for block in tool_result.content),
            }
            for tool_result in message.tool_results
        ]
        if result_blocks and messages and messages[-1]["role"] == "user":
            messages[-1]["content"].extend(result_blocks)
        else:
            messages.append({"role": message.role, "content": blocks + result_blocks})
    result: dict[str, Any] = {"messages": messages}
    if conversation.model is not None:
        result["model"] = conversation.model
    if conversation.system:
        result["system"] = [{"type": "text", "text": block.text} for block in conversation.system]
    if conversation.max_tokens is not None:
        result["max_tokens"] = conversation.max_tokens
    if conversation.temperature is not None:
        result["temperature"] = conversation.temperature
    if conversation.stream:
        result["stream"] = True
    if conversation.tools:
        result["tools"] = [
            {
                "name": tool.name,
                **({"description": tool.description} if tool.description is not None else {}),
                "input_schema": tool.input_schema,
            }
            for tool in conversation.tools
        ]
    return result


def _blocks(value: object) -> tuple[ContentBlock, ...]:
    if isinstance(value, str):
        return (ContentBlock(kind="text", text=value),) if value else ()
    if isinstance(value, list):
        blocks: list[ContentBlock] = []
        for block in value:
            if not isinstance(block, dict) or set(block) != {"type", "text"}:
                _unsupported("content block")
            if block["type"] != "text" or not isinstance(block["text"], str):
                _unsupported("content block")
            blocks.append(ContentBlock(kind="text", text=block["text"]))
        return tuple(blocks)
    _unsupported("content")


def _message_blocks(
    value: object,
) -> tuple[tuple[ContentBlock, ...], tuple[ToolCall, ...], tuple[ToolResult, ...]]:
    if isinstance(value, str):
        return _blocks(value), (), ()
    if not isinstance(value, list):
        _unsupported("content")
    content: list[ContentBlock] = []
    calls: list[ToolCall] = []
    results: list[ToolResult] = []
    for block in value:
        if not isinstance(block, dict) or not isinstance(block.get("type"), str):
            _unsupported("content block")
        if block["type"] == "text" and set(block) == {"type", "text"}:
            if calls or results:
                _unsupported("interleaved tool content")
            if not isinstance(block["text"], str):
                _unsupported("text")
            content.append(ContentBlock(kind="text", text=block["text"]))
        elif block["type"] == "image" and set(block) == {"type", "source"}:
            if calls or results:
                _unsupported("interleaved tool content")
            source = block["source"]
            if not isinstance(source, dict) or set(source) != {
                "type",
                "media_type",
                "data",
            }:
                _unsupported("image")
            if source["type"] != "base64" or not all(
                isinstance(source.get(key), str) for key in ("media_type", "data")
            ):
                _unsupported("image")
            content.append(_inline_image(source["media_type"], source["data"]))
        elif block["type"] == "tool_use" and set(block) == {
            "type",
            "id",
            "name",
            "input",
        }:
            if not isinstance(block["id"], str) or not isinstance(block["name"], str):
                _unsupported("tool use")
            if not isinstance(block["input"], dict):
                _unsupported("tool input")
            calls.append(
                ToolCall(
                    id=block["id"],
                    name=block["name"],
                    arguments=json.dumps(block["input"], ensure_ascii=False, separators=(",", ":")),
                )
            )
        elif block["type"] == "tool_result" and set(block) == {
            "type",
            "tool_use_id",
            "content",
        }:
            if content or calls:
                _unsupported("mixed tool result content")
            if not isinstance(block["tool_use_id"], str) or not isinstance(block["content"], str):
                _unsupported("tool result")
            results.append(
                ToolResult(
                    call_id=block["tool_use_id"],
                    content=(ContentBlock(kind="text", text=block["content"]),),
                )
            )
        else:
            _unsupported("content block")
    return tuple(content), tuple(calls), tuple(results)


def _decode_tools(value: object) -> tuple[ToolDefinition, ...]:
    if not isinstance(value, list):
        _unsupported("tools")
    tools: list[ToolDefinition] = []
    for tool in value:
        if not isinstance(tool, dict) or set(tool) - {
            "name",
            "description",
            "input_schema",
        }:
            _unsupported("tool")
        name = tool.get("name")
        description = tool.get("description")
        schema = tool.get("input_schema")
        if not isinstance(name, str) or not isinstance(schema, dict):
            _unsupported("tool")
        if description is not None and not isinstance(description, str):
            _unsupported("tool")
        tools.append(ToolDefinition(name=name, description=description, input_schema=schema))
    return tuple(tools)
