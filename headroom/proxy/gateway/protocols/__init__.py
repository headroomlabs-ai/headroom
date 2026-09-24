"""Directed, fail-closed protocol translation registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NoReturn

from headroom.proxy.gateway.capabilities import implemented_features, requested_features
from headroom.proxy.gateway.errors import GatewayAuthorizationError
from headroom.proxy.gateway.protocols.anthropic import _decode_anthropic, _encode_anthropic
from headroom.proxy.gateway.protocols.gemini import _decode_gemini, _encode_gemini
from headroom.proxy.gateway.protocols.openai import _decode_openai, _encode_openai

Decoder = Callable[[dict[str, Any]], object]
Encoder = Callable[[Any], dict[str, Any]]

_DECODERS: dict[str, Decoder] = {
    "openai-chat": _decode_openai,
    "anthropic-messages": _decode_anthropic,
    "gemini-generate": _decode_gemini,
}
_ENCODERS: dict[str, Encoder] = {
    "openai-chat": _encode_openai,
    "anthropic-messages": _encode_anthropic,
    "gemini-generate": _encode_gemini,
}


def translate(
    source_protocol: str,
    target_protocol: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    admitted = implemented_features(source_protocol, "http-json", False, target_protocol)
    if not admitted or not requested_features(source_protocol, payload) <= admitted:
        raise GatewayAuthorizationError(
            status_code=400,
            code="gateway_unsupported_capability",
            message="Translation direction or feature is unsupported",
        )
    try:
        decoder = _DECODERS[source_protocol]
        encoder = _ENCODERS[target_protocol]
    except KeyError as exc:
        raise GatewayAuthorizationError(
            status_code=400,
            code="gateway_unsupported_capability",
            message="Translation direction is unsupported",
        ) from exc
    result = encoder(decoder(payload))
    if target_protocol == "anthropic-messages" and result.get("max_tokens") is None:
        raise GatewayAuthorizationError(
            status_code=400,
            code="gateway_unsupported_capability",
            message="This route requires an explicit output token limit",
        )
    return result


def translate_response(
    source_protocol: str,
    target_protocol: str,
    payload: dict[str, Any],
    *,
    public_model: str,
) -> dict[str, Any]:
    """Translate a qualified non-streaming response without inventing usage."""

    if not implemented_features(target_protocol, "http-json", False, source_protocol):
        _unsupported_response()

    if source_protocol == "openai-chat" and target_protocol == "gemini-generate":
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            _unsupported_response()
        choice = choices[0]
        if (
            set(choice) - {"index", "message", "finish_reason", "logprobs"}
            or choice.get("logprobs") is not None
        ):
            _unsupported_response()
        message = choice.get("message")
        if not isinstance(message, dict) or set(message) - {"role", "content", "refusal"}:
            _unsupported_response()
        text = message.get("content")
        if message.get("role") != "assistant" or not isinstance(text, str):
            _unsupported_response()
        raw_finish = choice.get("finish_reason")
        if not isinstance(raw_finish, str):
            _unsupported_response()
        finish = {"stop": "STOP", "length": "MAX_TOKENS"}.get(raw_finish)
        if finish is None:
            _unsupported_response()
        if message.get("refusal"):
            _unsupported_response()
        result: dict[str, Any] = {
            "responseId": payload.get("id"),
            "modelVersion": public_model,
            "candidates": [
                {
                    "index": 0,
                    "content": {"role": "model", "parts": [{"text": text}]},
                    "finishReason": finish,
                }
            ],
        }
        usage = payload.get("usage")
        if usage is not None:
            result["usageMetadata"] = mapped_usage("openai-chat", "gemini-generate", usage)
        return result

    if source_protocol == "openai-chat" and target_protocol == "anthropic-messages":
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            _unsupported_response()
        choice = choices[0]
        if (
            set(choice) - {"index", "message", "finish_reason", "logprobs"}
            or choice.get("logprobs") is not None
        ):
            _unsupported_response()
        message = choice.get("message")
        if not isinstance(message, dict) or set(message) - {"role", "content", "refusal"}:
            _unsupported_response()
        content = message.get("content")
        if message.get("refusal"):
            _unsupported_response()
        if message.get("role") != "assistant" or not isinstance(content, str):
            _unsupported_response()
        raw_finish = choice.get("finish_reason")
        if raw_finish is not None and not isinstance(raw_finish, str):
            _unsupported_response()
        stop_reason = (
            None
            if raw_finish is None
            else {"stop": "end_turn", "length": "max_tokens"}.get(raw_finish)
        )
        if raw_finish is not None and stop_reason is None:
            _unsupported_response()
        openai_result: dict[str, Any] = {
            "id": payload.get("id"),
            "type": "message",
            "role": "assistant",
            "model": public_model,
            "content": [{"type": "text", "text": content}],
            "stop_reason": stop_reason,
            "stop_sequence": None,
        }
        usage = payload.get("usage")
        if usage is not None:
            openai_result["usage"] = mapped_usage(source_protocol, target_protocol, usage)
        return openai_result

    if source_protocol == "anthropic-messages" and target_protocol == "openai-chat":
        content = payload.get("content")
        if not isinstance(content, list):
            _unsupported_response()
        anthropic_text_parts: list[str] = []
        for block in content:
            if not isinstance(block, dict) or set(block) != {"type", "text"}:
                _unsupported_response()
            if block["type"] != "text" or not isinstance(block["text"], str):
                _unsupported_response()
            anthropic_text_parts.append(block["text"])
        raw_stop_reason = payload.get("stop_reason")
        if raw_stop_reason is not None and not isinstance(raw_stop_reason, str):
            _unsupported_response()
        stop_reason = (
            None
            if raw_stop_reason is None
            else {
                "end_turn": "stop",
                "max_tokens": "length",
            }.get(raw_stop_reason)
        )
        if raw_stop_reason is not None and stop_reason is None:
            _unsupported_response()
        anthropic_result: dict[str, Any] = {
            "id": payload.get("id"),
            "object": "chat.completion",
            "model": public_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "".join(anthropic_text_parts)},
                    "finish_reason": stop_reason,
                }
            ],
        }
        usage = payload.get("usage")
        if usage is not None:
            anthropic_result["usage"] = mapped_usage(source_protocol, target_protocol, usage)
        return anthropic_result
    _unsupported_response()


def mapped_usage(source: str, target: str, usage: dict[str, Any]) -> dict[str, Any]:
    """Preserve available token units or reject nonrepresentable observations."""
    if not isinstance(usage, dict):
        _unsupported_response()
    result: dict[str, Any] = {}
    if source == "openai-chat":
        _usage_fields(
            usage,
            {
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "prompt_tokens_details",
                "completion_tokens_details",
            },
        )
        counts = _usage_counts(
            {
                key: usage[key]
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                if key in usage
            }
        )
        prompt = _usage_details(
            usage.get("prompt_tokens_details"), {"cached_tokens", "audio_tokens"}
        )
        completion = _usage_details(
            usage.get("completion_tokens_details"),
            {
                "reasoning_tokens",
                "audio_tokens",
                "accepted_prediction_tokens",
                "rejected_prediction_tokens",
            },
        )
        if prompt.get("audio_tokens", 0) or any(
            completion.get(key, 0)
            for key in ("audio_tokens", "accepted_prediction_tokens", "rejected_prediction_tokens")
        ):
            _unsupported_response()
        cached = prompt.get("cached_tokens")
        reasoning = completion.get("reasoning_tokens")
        if cached is not None and "prompt_tokens" in counts and cached > counts["prompt_tokens"]:
            _unsupported_response()
        if (
            reasoning is not None
            and "completion_tokens" in counts
            and reasoning > counts["completion_tokens"]
        ):
            _unsupported_response()
        if (
            all(key in counts for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
            and counts["total_tokens"] != counts["prompt_tokens"] + counts["completion_tokens"]
        ):
            _unsupported_response()
        if target == "gemini-generate":
            for source_key, target_key in {
                "prompt_tokens": "promptTokenCount",
                "completion_tokens": "candidatesTokenCount",
                "total_tokens": "totalTokenCount",
            }.items():
                if source_key in counts:
                    result[target_key] = counts[source_key]
            if cached is not None:
                result["cachedContentTokenCount"] = cached
            if reasoning is not None:
                result["thoughtsTokenCount"] = reasoning
                if "candidatesTokenCount" in result:
                    result["candidatesTokenCount"] -= reasoning
        elif target == "anthropic-messages":
            # Anthropic has no reasoning/prediction usage breakdown or separate
            # total counter. A complete total is derivable; a lone total is not.
            if reasoning or (
                "total_tokens" in counts
                and not {"prompt_tokens", "completion_tokens"} <= counts.keys()
            ):
                _unsupported_response()
            if "prompt_tokens" in counts:
                result["input_tokens"] = counts["prompt_tokens"] - (cached or 0)
            if "completion_tokens" in counts:
                result["output_tokens"] = counts["completion_tokens"]
            if cached is not None:
                result["cache_read_input_tokens"] = cached
        else:
            _unsupported_response()
    elif source == "anthropic-messages" and target == "openai-chat":
        _usage_fields(
            usage,
            {
                "input_tokens",
                "output_tokens",
                "cache_read_input_tokens",
                "cache_creation_input_tokens",
                "cache_creation",
            },
        )
        counts = _usage_counts(
            {key: value for key, value in usage.items() if key != "cache_creation"}
        )
        creation = _usage_details(
            usage.get("cache_creation"), {"ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"}
        )
        if counts.get("cache_creation_input_tokens", 0) or any(creation.values()):
            # Chat cannot retain the distinct cache-write unit; folding it into
            # inclusive prompt tokens would silently discard its meaning.
            _unsupported_response()
        cached = counts.get("cache_read_input_tokens")
        if cached is not None:
            result["prompt_tokens_details"] = {"cached_tokens": cached}
        if "input_tokens" in counts:
            result["prompt_tokens"] = counts["input_tokens"] + (cached or 0)
        if "output_tokens" in counts:
            result["completion_tokens"] = counts["output_tokens"]
        if {"prompt_tokens", "completion_tokens"} <= result.keys():
            result["total_tokens"] = result["prompt_tokens"] + result["completion_tokens"]
    else:
        _unsupported_response()
    return result


def _usage_fields(value: dict[str, Any], allowed: set[str]) -> None:
    if set(value) - allowed:
        _unsupported_response()


def _usage_counts(value: dict[str, Any]) -> dict[str, int]:
    result = {}
    for key, count in value.items():
        if count is None:
            continue
        if type(count) is not int or count < 0:
            _unsupported_response()
        result[key] = count
    return result


def _usage_details(value: Any, allowed: set[str]) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        _unsupported_response()
    _usage_fields(value, allowed)
    return _usage_counts(value)


def _unsupported_response() -> NoReturn:
    raise GatewayAuthorizationError(
        status_code=502,
        code="gateway_upstream_semantics_unsupported",
        message="Upstream response cannot be represented in the ingress protocol",
    )


__all__ = [
    "translate",
    "translate_response",
]
