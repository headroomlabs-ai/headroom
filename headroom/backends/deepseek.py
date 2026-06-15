"""Native Deepseek backend for Headroom proxy.

Uses Deepseek's Anthropic-compatible and OpenAI-compatible APIs directly,
bypassing LiteLLM. No message format conversion needed since Deepseek
natively supports both API formats.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections.abc import AsyncIterator
from dataclasses import asdict
from typing import Any

from .base import Backend, BackendResponse, StreamEvent

logger = logging.getLogger(__name__)

DEEPSEEK_ANTHROPIC_BASE_URL = "https://api.deepseek.com/anthropic"
DEEPSEEK_OPENAI_BASE_URL = "https://api.deepseek.com"

# Map Claude model names to Deepseek models (Deepseek's Anthropic API supports this)
_DEEPMODEL_MAP: dict[str, str] = {
    # Claude 4 Opus -> Deepseek V4 Pro
    "claude-opus-4-7": "deepseek-v4-pro",
    "claude-opus-4-6": "deepseek-v4-pro",
    "claude-opus-4-5-20251101": "deepseek-v4-pro",
    "claude-opus-4-5": "deepseek-v4-pro",
    # Claude 4 Sonnet/Haiku -> Deepseek V4 Flash
    "claude-sonnet-4-20250514": "deepseek-v4-flash",
    "claude-sonnet-4": "deepseek-v4-flash",
    "claude-haiku-4-5-20251001": "deepseek-v4-flash",
    "claude-haiku-4-5": "deepseek-v4-flash",
    # Claude 3.5 -> Deepseek V3
    "claude-3-5-sonnet-20241022": "deepseek-chat",
    "claude-3-5-sonnet-latest": "deepseek-chat",
    "claude-3-5-haiku-20241022": "deepseek-chat",
    "claude-3-5-haiku-latest": "deepseek-chat",
    # Claude 3 -> Deepseek V2
    "claude-3-opus-20240229": "deepseek-chat",
    "claude-3-opus-latest": "deepseek-chat",
    "claude-3-sonnet-20240229": "deepseek-chat",
    "claude-3-haiku-20240307": "deepseek-chat",
}


class DeepseekBackend(Backend):
    """Native Deepseek backend using Deepseek's own API endpoints.

    Supports both Anthropic Messages API and OpenAI Chat Completions API
    natively without format conversion.
    """

    def __init__(
        self,
        api_key: str | None = None,
        anthropic_client: Any | None = None,
        openai_client: Any | None = None,
    ) -> None:
        self._api_key = api_key
        self._anthropic_client = anthropic_client
        self._openai_client = openai_client

    @property
    def name(self) -> str:
        return "deepseek"

    def _resolve_api_key(self, headers: dict[str, str]) -> str | None:
        """Resolve API key from constructor, headers, or environment."""
        if self._api_key:
            return self._api_key
        header_key = headers.get("x-api-key") or headers.get("X-Api-Key")
        if header_key:
            return header_key
        return os.environ.get("DEEPSEEK_API_KEY")

    def _get_anthropic_client(self, api_key: str | None = None) -> Any:
        if self._anthropic_client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as err:
                raise RuntimeError(
                    "anthropic SDK is required for native Deepseek backend. "
                    "Install with: pip install anthropic"
                ) from err
            kwargs: dict[str, Any] = {"base_url": DEEPSEEK_ANTHROPIC_BASE_URL}
            if api_key:
                kwargs["api_key"] = api_key
            self._anthropic_client = AsyncAnthropic(**kwargs)
        return self._anthropic_client

    def _get_openai_client(self, api_key: str | None = None) -> Any:
        if self._openai_client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as err:
                raise RuntimeError(
                    "openai SDK is required for native Deepseek backend. "
                    "Install with: pip install openai"
                ) from err
            kwargs: dict[str, Any] = {"base_url": DEEPSEEK_OPENAI_BASE_URL}
            if api_key:
                kwargs["api_key"] = api_key
            self._openai_client = AsyncOpenAI(**kwargs)
        return self._openai_client

    def map_model_id(self, anthropic_model: str) -> str:
        if anthropic_model in _DEEPMODEL_MAP:
            return _DEEPMODEL_MAP[anthropic_model]
        if anthropic_model.startswith("deepseek-"):
            return anthropic_model
        return anthropic_model

    def supports_model(self, model: str) -> bool:
        return model.startswith("claude-") or model.startswith("deepseek-")

    async def send_message(
        self, body: dict[str, Any], headers: dict[str, str]
    ) -> BackendResponse:
        api_key = self._resolve_api_key(headers)
        client = self._get_anthropic_client(api_key)
        model = self.map_model_id(body.get("model", "deepseek-chat"))

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": body.get("max_tokens", 4096),
            "messages": body.get("messages", []),
        }
        if "system" in body:
            kwargs["system"] = body["system"]
        if "temperature" in body:
            kwargs["temperature"] = body["temperature"]
        if "top_p" in body:
            kwargs["top_p"] = body["top_p"]
        if "stop_sequences" in body:
            kwargs["stop_sequences"] = body["stop_sequences"]
        if "tools" in body:
            kwargs["tools"] = body["tools"]
        if "tool_choice" in body:
            kwargs["tool_choice"] = body["tool_choice"]
        if "thinking" in body:
            kwargs["thinking"] = body["thinking"]

        try:
            response = await client.messages.create(**kwargs)
            return BackendResponse(
                body=response.model_dump(),
                status_code=200,
            )
        except Exception as e:
            return self._handle_anthropic_error(e)

    async def stream_message(
        self, body: dict[str, Any], headers: dict[str, str]
    ) -> AsyncIterator[StreamEvent]:
        api_key = self._resolve_api_key(headers)
        client = self._get_anthropic_client(api_key)
        model = self.map_model_id(body.get("model", "deepseek-chat"))

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": body.get("max_tokens", 4096),
            "messages": body.get("messages", []),
        }
        if "system" in body:
            kwargs["system"] = body["system"]
        if "temperature" in body:
            kwargs["temperature"] = body["temperature"]
        if "top_p" in body:
            kwargs["top_p"] = body["top_p"]
        if "stop_sequences" in body:
            kwargs["stop_sequences"] = body["stop_sequences"]
        if "tools" in body:
            kwargs["tools"] = body["tools"]
        if "tool_choice" in body:
            kwargs["tool_choice"] = body["tool_choice"]
        if "thinking" in body:
            kwargs["thinking"] = body["thinking"]

        msg_id = f"msg_{uuid.uuid4().hex}"

        try:
            async with client.messages.stream(**kwargs) as stream:
                yield StreamEvent(
                    event_type="message_start",
                    data={
                        "type": "message_start",
                        "message": {
                            "id": msg_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": model,
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": 0, "output_tokens": 0},
                        },
                    },
                )

                block_index = 0

                async for event in stream:
                    if event.type == "content_block_start":
                        content_block = event.content_block
                        if hasattr(content_block, "__dataclass_fields__"):
                            block_data = asdict(content_block)
                        else:
                            block_data = {
                                "type": content_block.type,
                                "text": getattr(content_block, "text", ""),
                            }
                        yield StreamEvent(
                            event_type="content_block_start",
                            data={
                                "type": "content_block_start",
                                "index": block_index,
                                "content_block": block_data,
                            },
                        )
                    elif event.type == "content_block_delta":
                        delta = event.delta
                        if hasattr(delta, "__dataclass_fields__"):
                            delta_data = asdict(delta)
                        else:
                            delta_data = {
                                "type": delta.type,
                                "text": getattr(delta, "text", ""),
                            }
                        yield StreamEvent(
                            event_type="content_block_delta",
                            data={
                                "type": "content_block_delta",
                                "index": block_index,
                                "delta": delta_data,
                            },
                        )
                    elif event.type == "content_block_stop":
                        yield StreamEvent(
                            event_type="content_block_stop",
                            data={"type": "content_block_stop", "index": block_index},
                        )
                        block_index += 1

                final_message = await stream.get_final_message()
                yield StreamEvent(
                    event_type="message_delta",
                    data={
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": final_message.stop_reason or "end_turn",
                            "stop_sequence": None,
                        },
                        "usage": {"output_tokens": final_message.usage.output_tokens},
                    },
                )
                yield StreamEvent(
                    event_type="message_stop",
                    data={"type": "message_stop"},
                )

        except Exception as e:
            error_response = self._handle_anthropic_error(e)
            yield StreamEvent(
                event_type="error",
                data={
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": error_response.error or "Unknown error",
                    },
                },
            )

    async def send_openai_message(
        self, body: dict[str, Any], headers: dict[str, str]
    ) -> BackendResponse:
        api_key = self._resolve_api_key(headers)
        client = self._get_openai_client(api_key)
        model = body.get("model", "deepseek-chat")

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": body.get("messages", []),
        }
        for param in [
            "max_tokens",
            "temperature",
            "top_p",
            "stop",
            "tools",
            "tool_choice",
            "response_format",
            "seed",
            "n",
            "logprobs",
            "top_logprobs",
            "user_id",
        ]:
            if param in body:
                kwargs[param] = body[param]

        extra_body = self._build_openai_extra_body(body)
        if extra_body:
            kwargs["extra_body"] = extra_body

        try:
            response = await client.chat.completions.create(**kwargs)
            return BackendResponse(
                body=response.model_dump(),
                status_code=200,
            )
        except Exception as e:
            return self._handle_openai_error(e)

    async def stream_openai_message(
        self, body: dict[str, Any], headers: dict[str, str]
    ) -> AsyncIterator[str]:
        api_key = self._resolve_api_key(headers)
        client = self._get_openai_client(api_key)
        model = body.get("model", "deepseek-chat")

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": body.get("messages", []),
            "stream": True,
        }
        for param in [
            "max_tokens",
            "temperature",
            "top_p",
            "stop",
            "tools",
            "tool_choice",
            "response_format",
            "seed",
            "n",
            "stream_options",
            "logprobs",
            "top_logprobs",
            "user_id",
        ]:
            if param in body:
                kwargs[param] = body[param]

        extra_body = self._build_openai_extra_body(body)
        if extra_body:
            kwargs["extra_body"] = extra_body

        try:
            stream = await client.chat.completions.create(**kwargs)
            async for chunk in stream:
                yield f"data: {json.dumps(chunk.model_dump())}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            error_response = self._handle_openai_error(e)
            yield f"data: {json.dumps(error_response.body)}\n\n"
            yield "data: [DONE]\n\n"

    def _build_openai_extra_body(self, body: dict[str, Any]) -> dict[str, Any] | None:
        """Build extra_body for Deepseek-specific OpenAI params."""
        extra: dict[str, Any] = {}

        thinking = body.get("thinking")
        reasoning_effort = body.get("reasoning_effort")

        if thinking is not None:
            extra["thinking"] = thinking
        if reasoning_effort is not None:
            extra["reasoning_effort"] = reasoning_effort

        return extra if extra else None

    def _handle_anthropic_error(self, e: Exception) -> BackendResponse:
        status_code = 500
        error_msg = str(e)

        if hasattr(e, "status_code"):
            status_code = e.status_code
        elif "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
            status_code = 401
        elif "rate" in error_msg.lower() and "limit" in error_msg.lower():
            status_code = 429
        elif "not found" in error_msg.lower():
            status_code = 404
        elif "invalid" in error_msg.lower():
            status_code = 400

        return BackendResponse(
            body={"type": "error", "error": {"type": "api_error", "message": error_msg}},
            status_code=status_code,
            error=error_msg,
        )

    def _handle_openai_error(self, e: Exception) -> BackendResponse:
        status_code = 500
        error_msg = str(e)

        if hasattr(e, "status_code"):
            status_code = e.status_code
        elif "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
            status_code = 401
        elif "rate" in error_msg.lower() and "limit" in error_msg.lower():
            status_code = 429

        return BackendResponse(
            body={"error": {"message": error_msg, "type": "api_error"}},
            status_code=status_code,
            error=error_msg,
        )

    async def close(self) -> None:
        if self._anthropic_client is not None:
            await self._anthropic_client.close()
        if self._openai_client is not None:
            await self._openai_client.close()
