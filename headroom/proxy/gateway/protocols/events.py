"""Bounded, incremental mappings for admitted text streams.

Tool, signed and hosted-tool streams are not admitted by the translation
contract. Unexpected semantic events fail closed even after admission.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, AsyncIterable
from dataclasses import dataclass
from typing import Any, Literal

from headroom.proxy.gateway.capabilities import implemented_features
from headroom.proxy.gateway.errors import GatewayAuthorizationError
from headroom.proxy.gateway.protocols import mapped_usage
from headroom.proxy.gateway.streaming import SSEFrames, parse_event, stream_error


@dataclass(frozen=True, slots=True)
class StreamEvent:
    kind: Literal["text_delta", "tool_argument_delta", "finish"]
    index: int
    data: str
    call_id: str | None = None
    tool_name: str | None = None


def translate_event(
    source_protocol: str, target_protocol: str, event: StreamEvent
) -> tuple[dict[str, object], ...]:
    """Standalone tool fragments lack a qualified lifecycle and are unavailable."""
    raise GatewayAuthorizationError(
        status_code=400,
        code="gateway_unsupported_capability",
        message="Standalone event translation is unsupported",
    )


def _sse(payload: dict[str, Any], *, named: bool = False) -> bytes:
    prefix = f"event: {payload['type']}\n" if named else ""
    try:
        return (
            prefix
            + "data: "
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            + "\n\n"
        ).encode()
    except UnicodeEncodeError:
        raise stream_error("malformed") from None


async def translate_sse_stream(
    source_protocol: str, target_protocol: str, chunks: AsyncIterable[bytes], *, public_model: str
) -> AsyncGenerator[bytes, None]:
    if not implemented_features(target_protocol, "http-stream", False, source_protocol):
        raise GatewayAuthorizationError(
            status_code=400,
            code="gateway_unsupported_capability",
            message="Streaming translation direction is unsupported",
        )
    frames = SSEFrames(1_048_576)
    identifier = "gateway-translated"
    usage: dict[str, Any] = {}
    source_usage: dict[str, Any] = {}
    finish: str | None = None
    started = False
    text_started = False
    terminal = False
    open_block: int | None = None
    next_block = 0

    def merge_usage(previous: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
        merged = dict(previous)
        for key, value in snapshot.items():
            prior = previous.get(key)
            # Null is no new knowledge, not a retraction of an observed unit.
            if value is None and key in previous:
                continue
            if isinstance(value, dict) and isinstance(prior, dict):
                value = merge_usage(prior, value)
            elif type(value) is int and type(prior) is int and value < prior:
                # Counts are cumulative snapshots, never deltas or corrections.
                raise stream_error("malformed")
            merged[key] = value
        return merged

    def observe_usage(raw: Any) -> None:
        nonlocal source_usage, usage
        if raw is None:
            return
        if not isinstance(raw, dict):
            raise stream_error("malformed")
        merged = merge_usage(source_usage, raw)
        usage = mapped_usage(source_protocol, target_protocol, merged)
        source_usage = merged

    def chat(delta: dict[str, Any], reason: str | None = None) -> dict[str, Any]:
        return {
            "id": identifier,
            "object": "chat.completion.chunk",
            "created": 0,
            "model": public_model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": reason}],
        }

    async for chunk in chunks:
        for frame in frames.feed(chunk):
            parsed = parse_event(frame)
            if parsed.name == "error":
                raise stream_error("upstream_error")
            if not parsed.data:
                continue
            if terminal:
                raise stream_error("malformed")
            if parsed.data == "[DONE]":
                if source_protocol != "openai-chat" or finish is None:
                    raise stream_error("truncated")
                terminal = True
                if target_protocol == "anthropic-messages":
                    if text_started:
                        yield _sse({"type": "content_block_stop", "index": 0}, named=True)
                    yield _sse(
                        {
                            "type": "message_delta",
                            "delta": {"stop_reason": finish, "stop_sequence": None},
                            "usage": usage,
                        },
                        named=True,
                    )
                    yield _sse({"type": "message_stop"}, named=True)
                else:
                    terminal_payload: dict[str, Any] = {
                        "responseId": identifier,
                        "modelVersion": public_model,
                        "candidates": [{"index": 0, "finishReason": finish}],
                    }
                    if usage:
                        terminal_payload["usageMetadata"] = usage
                    yield _sse(terminal_payload)
                continue
            try:
                event = json.loads(parsed.data)
            except (ValueError, RecursionError):
                raise stream_error("malformed") from None
            if not isinstance(event, dict):
                raise stream_error("malformed")
            if event.get("error") or event.get("type") == "error":
                raise stream_error("upstream_error")
            if source_protocol == "anthropic-messages":
                kind = event.get("type")
                if kind == "ping":
                    continue
                if kind == "message_start":
                    if started:
                        raise stream_error("malformed")
                    message = event.get("message", {})
                    if not isinstance(message, dict) or message.get("content", []):
                        raise stream_error("malformed")
                    identifier = message.get("id", identifier)
                    observe_usage(message.get("usage"))
                    started = True
                    yield _sse(chat({"role": "assistant"}))
                elif kind == "content_block_start":
                    if (
                        not started
                        or finish is not None
                        or open_block is not None
                        or type(event.get("index")) is not int
                        or event["index"] != next_block
                    ):
                        raise stream_error("malformed")
                    block = event.get("content_block", {})
                    if (
                        not isinstance(block, dict)
                        or set(block) != {"type", "text"}
                        or block.get("type") != "text"
                        or not isinstance(block.get("text"), str)
                    ):
                        raise stream_error("malformed")
                    open_block = next_block
                    if block["text"]:
                        yield _sse(chat({"content": block["text"]}))
                elif kind == "content_block_delta":
                    if (
                        open_block is None
                        or type(event.get("index")) is not int
                        or event["index"] != open_block
                    ):
                        raise stream_error("malformed")
                    delta = event.get("delta", {})
                    if (
                        not isinstance(delta, dict)
                        or set(delta) != {"type", "text"}
                        or delta.get("type") != "text_delta"
                        or not isinstance(delta["text"], str)
                    ):
                        raise stream_error("malformed")
                    yield _sse(chat({"content": delta["text"]}))
                elif kind == "content_block_stop":
                    if (
                        open_block is None
                        or type(event.get("index")) is not int
                        or event["index"] != open_block
                    ):
                        raise stream_error("malformed")
                    open_block = None
                    next_block += 1
                    continue
                elif kind == "message_delta":
                    if not started or open_block is not None or finish is not None:
                        raise stream_error("malformed")
                    delta = event.get("delta", {})
                    if (
                        not isinstance(delta, dict)
                        or set(delta) - {"stop_reason", "stop_sequence"}
                        or delta.get("stop_sequence") is not None
                    ):
                        raise stream_error("malformed")
                    raw_finish = delta.get("stop_reason")
                    if not isinstance(raw_finish, str):
                        raise stream_error("malformed")
                    finish = {
                        "end_turn": "stop",
                        "max_tokens": "length",
                    }.get(raw_finish)
                    if finish is None:
                        raise stream_error("malformed")
                    observe_usage(event.get("usage"))
                elif kind == "message_stop":
                    if finish is None:
                        raise stream_error("truncated")
                    yield _sse(chat({}, finish))
                    if usage:
                        tail = chat({})
                        tail["choices"] = []
                        tail["usage"] = usage
                        yield _sse(tail)
                    yield b"data: [DONE]\n\n"
                    terminal = True
                else:
                    raise stream_error("malformed")
                continue

            if set(event) - {
                "id",
                "object",
                "created",
                "model",
                "choices",
                "usage",
                "system_fingerprint",
                "service_tier",
            }:
                raise stream_error("malformed")
            identifier = event.get("id", identifier)
            choices = event.get("choices", [])
            if not isinstance(choices, list) or len(choices) > 1:
                raise stream_error("malformed")
            observe_usage(event.get("usage"))
            for choice in choices:
                if (
                    finish is not None
                    or not isinstance(choice, dict)
                    or choice.get("index", 0) != 0
                    or set(choice) - {"index", "delta", "finish_reason", "logprobs"}
                    or choice.get("logprobs") is not None
                ):
                    raise stream_error("malformed")
                delta = choice.get("delta", {})
                if (
                    not isinstance(delta, dict)
                    or set(delta) - {"role", "content", "refusal"}
                    or delta.get("role", "assistant") != "assistant"
                ):
                    raise stream_error("malformed")
                refusal = delta.get("refusal")
                text = delta.get("content")
                if refusal:
                    raise stream_error("upstream_error")
                if text is not None and not isinstance(text, str):
                    raise stream_error("malformed")
                if not started and target_protocol == "anthropic-messages":
                    yield _sse(
                        {
                            "type": "message_start",
                            "message": {
                                "id": identifier,
                                "type": "message",
                                "role": "assistant",
                                "model": public_model,
                                "content": [],
                                "stop_reason": None,
                                "stop_sequence": None,
                                "usage": {},
                            },
                        },
                        named=True,
                    )
                started = True
                if text:
                    if finish is not None and not refusal:
                        raise stream_error("malformed")
                    if target_protocol == "anthropic-messages":
                        if not text_started:
                            yield _sse(
                                {
                                    "type": "content_block_start",
                                    "index": 0,
                                    "content_block": {"type": "text", "text": ""},
                                },
                                named=True,
                            )
                            text_started = True
                        yield _sse(
                            {
                                "type": "content_block_delta",
                                "index": 0,
                                "delta": {"type": "text_delta", "text": text},
                            },
                            named=True,
                        )
                    else:
                        yield _sse(
                            {
                                "responseId": identifier,
                                "modelVersion": public_model,
                                "candidates": [
                                    {
                                        "index": 0,
                                        "content": {"role": "model", "parts": [{"text": text}]},
                                    }
                                ],
                            }
                        )
                reason = choice.get("finish_reason")
                if reason is not None:
                    mapping = (
                        {"stop": "end_turn", "length": "max_tokens"}
                        if target_protocol == "anthropic-messages"
                        else {"stop": "STOP", "length": "MAX_TOKENS"}
                    )
                    if reason not in mapping:
                        raise stream_error("malformed")
                    finish = mapping[reason]
    if frames.pending or not terminal:
        raise stream_error("truncated")
