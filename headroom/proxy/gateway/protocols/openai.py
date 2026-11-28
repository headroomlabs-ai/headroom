"""Qualified OpenAI Chat request adapter."""

from __future__ import annotations

import base64
import binascii
import json
import math
from typing import Any, NoReturn, cast

from headroom.proxy.gateway.errors import GatewayAuthorizationError
from headroom.proxy.gateway.protocols.content import (
    ContentBlock,
    Conversation,
    Message,
    ToolCall,
    ToolDefinition,
    ToolResult,
)

_FIELDS = frozenset({"model", "messages", "max_tokens", "temperature", "stream", "tools"})


def _decode_openai(payload: dict[str, Any]) -> Conversation:
    _reject_unknown(payload, _FIELDS)
    if "stream" in payload and not isinstance(payload["stream"], bool):
        _unsupported("stream")
    system: list[ContentBlock] = []
    messages: list[Message] = []
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        _unsupported("messages")
    for item in raw_messages:
        if not isinstance(item, dict) or "role" not in item:
            _unsupported("message")
        role = item["role"]
        allowed = {"role", "content"}
        if role == "assistant":
            allowed.add("tool_calls")
        elif role == "tool":
            allowed.add("tool_call_id")
        if set(item) - allowed:
            _unsupported("message")
        content = _decode_content(item.get("content"))
        if role == "system":
            if messages:
                _unsupported("non-leading system message")
            _plain_text(content)
            system.extend(content)
        elif role in ("user", "assistant"):
            calls = _decode_tool_calls(item.get("tool_calls", []))
            messages.append(Message(role=role, content=content, tool_calls=calls))
        elif role == "tool":
            call_id = item.get("tool_call_id")
            if not isinstance(call_id, str):
                _unsupported("tool_call_id")
            messages.append(
                Message(
                    role="user",
                    content=(),
                    tool_results=(ToolResult(call_id=call_id, content=content),),
                )
            )
        else:
            _unsupported("message role")
    tools = _decode_tools(payload.get("tools", []))
    return Conversation(
        model=payload.get("model") if isinstance(payload.get("model"), str) else None,
        system=tuple(system),
        messages=tuple(messages),
        tools=tools,
        max_tokens=_optional_int(payload, "max_tokens"),
        temperature=_optional_number(payload, "temperature"),
        stream=payload.get("stream", False) is True,
    )


def _encode_openai(conversation: Conversation) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    if conversation.system:
        messages.append({"role": "system", "content": _plain_text(conversation.system)})
    for message in conversation.messages:
        if message.tool_results:
            if message.content or message.tool_calls:
                _unsupported("mixed tool result content")
            messages.extend(
                {
                    "role": "tool",
                    "tool_call_id": result.call_id,
                    "content": _plain_text(result.content),
                }
                for result in message.tool_results
            )
            continue
        encoded: dict[str, Any] = {
            "role": message.role,
            "content": _openai_content(message.content),
        }
        if message.tool_calls:
            encoded["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in message.tool_calls
            ]
            if not message.content:
                encoded["content"] = None
        messages.append(encoded)
    result: dict[str, Any] = {"messages": messages}
    if conversation.model is not None:
        result["model"] = conversation.model
    if conversation.max_tokens is not None:
        result["max_tokens"] = conversation.max_tokens
    if conversation.temperature is not None:
        result["temperature"] = conversation.temperature
    if conversation.stream:
        result["stream"] = True
    if conversation.tools:
        result["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    **({"description": tool.description} if tool.description is not None else {}),
                    "parameters": tool.input_schema,
                },
            }
            for tool in conversation.tools
        ]
    return result


def _decode_content(value: object) -> tuple[ContentBlock, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (ContentBlock(kind="text", text=value),)
    if isinstance(value, list):
        blocks: list[ContentBlock] = []
        for block in value:
            if not isinstance(block, dict):
                _unsupported("content block")
            if set(block) == {"type", "text"} and block["type"] == "text":
                if not isinstance(block["text"], str):
                    _unsupported("content block")
                blocks.append(ContentBlock(kind="text", text=block["text"]))
                continue
            if set(block) == {"type", "image_url"} and block["type"] == "image_url":
                image = block["image_url"]
                if not isinstance(image, dict) or set(image) != {"url"}:
                    _unsupported("image")
                url = image["url"]
                if not isinstance(url, str) or not url.startswith("data:image/"):
                    _unsupported("remote image")
                header, separator, data = url.partition(",")
                media_type = header.removeprefix("data:").removesuffix(";base64")
                if (
                    not separator
                    or not header.endswith(";base64")
                    or media_type
                    not in {
                        "image/png",
                        "image/jpeg",
                        "image/gif",
                        "image/webp",
                    }
                ):
                    _unsupported("image")
                try:
                    base64.b64decode(data, validate=True)
                except (ValueError, binascii.Error) as exc:
                    raise GatewayAuthorizationError(
                        status_code=400,
                        code="gateway_request_invalid",
                        message="Inline image is not valid base64",
                    ) from exc
                blocks.append(ContentBlock(kind="image", media_type=media_type, data=data))
                continue
            _unsupported("content block")
        return tuple(blocks)
    _unsupported("content")


def _decode_tool_calls(value: object) -> tuple[ToolCall, ...]:
    if not isinstance(value, list):
        _unsupported("tool_calls")
    calls: list[ToolCall] = []
    for call in value:
        if not isinstance(call, dict) or set(call) != {"id", "type", "function"}:
            _unsupported("tool call")
        function = call["function"]
        if call["type"] != "function" or not isinstance(function, dict):
            _unsupported("tool call")
        if set(function) != {"name", "arguments"}:
            _unsupported("tool call")
        if not all(isinstance(function.get(key), str) for key in ("name", "arguments")):
            _unsupported("tool call")
        try:
            arguments = json.loads(function["arguments"])
        except json.JSONDecodeError as exc:
            raise GatewayAuthorizationError(
                status_code=400,
                code="gateway_request_invalid",
                message="Tool arguments are not valid JSON",
            ) from exc
        if not isinstance(arguments, dict):
            _unsupported("tool arguments")
        calls.append(
            ToolCall(id=call["id"], name=function["name"], arguments=function["arguments"])
        )
    return tuple(calls)


def _inline_image(media_type: object, data: object) -> ContentBlock:
    if media_type not in {"image/png", "image/jpeg", "image/gif", "image/webp"} or not isinstance(
        data, str
    ):
        _unsupported("image")
    if len(data) > 5_592_408:
        _unsupported("image size")
    try:
        base64.b64decode(data, validate=True)
    except (ValueError, binascii.Error):
        _unsupported("image encoding")
    return ContentBlock(kind="image", media_type=str(media_type), data=data)


def _decode_tools(value: object) -> tuple[ToolDefinition, ...]:
    if not isinstance(value, list):
        _unsupported("tools")
    tools: list[ToolDefinition] = []
    for tool in value:
        if not isinstance(tool, dict) or set(tool) != {"type", "function"}:
            _unsupported("tool")
        function = tool["function"]
        if tool["type"] != "function" or not isinstance(function, dict):
            _unsupported("tool")
        if set(function) - {"name", "description", "parameters"}:
            _unsupported("tool")
        name = function.get("name")
        schema = function.get("parameters")
        description = function.get("description")
        if not isinstance(name, str) or not isinstance(schema, dict):
            _unsupported("tool")
        if description is not None and not isinstance(description, str):
            _unsupported("tool")
        tools.append(ToolDefinition(name=name, description=description, input_schema=schema))
    return tuple(tools)


def _plain_text(blocks: tuple[ContentBlock, ...]) -> str:
    if any(block.kind != "text" or block.text is None for block in blocks):
        _unsupported("non-text content")
    return "".join(block.text or "" for block in blocks)


def _openai_content(blocks: tuple[ContentBlock, ...]) -> str | list[dict[str, Any]]:
    if all(block.kind == "text" for block in blocks):
        return _plain_text(blocks)
    result: list[dict[str, Any]] = []
    for block in blocks:
        if block.kind == "text":
            result.append({"type": "text", "text": block.text})
        elif block.kind == "image":
            result.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{block.media_type};base64,{block.data}",
                    },
                }
            )
    return result


def _reject_unknown(payload: dict[str, Any], fields: frozenset[str]) -> None:
    if set(payload) - fields:
        _unsupported("unknown field")


def _optional_int(payload: dict[str, Any], name: str) -> int | None:
    value = payload.get(name)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        _unsupported(name)
    return cast(int, value)


def _optional_number(payload: dict[str, Any], name: str) -> float | None:
    value = payload.get(name)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        _unsupported(name)
    try:
        number = float(value)
    except OverflowError:
        _unsupported(name)
    if not math.isfinite(number):
        _unsupported(name)
    return number


def _unsupported(feature: str) -> NoReturn:
    raise GatewayAuthorizationError(
        status_code=400,
        code="gateway_unsupported_capability",
        message=f"Translation feature is unsupported: {feature}",
    )
