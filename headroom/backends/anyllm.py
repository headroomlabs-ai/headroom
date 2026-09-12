"""any-llm backend for Headroom.

Talk to 38+ LLM providers (OpenAI, Mistral, Groq, Ollama, Bedrock, etc.)
through a single interface. Auth and format translation handled automatically.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any, cast

from headroom.message_contract import preserve_message_fields

from .base import Backend, BackendResponse, StreamEvent

logger = logging.getLogger(__name__)

try:
    from any_llm import AnyLLM

    ANYLLM_AVAILABLE = True
except ImportError:
    ANYLLM_AVAILABLE = False
    AnyLLM = None  # type: ignore


def _convert_anthropic_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Convert an Anthropic tool definition to the OpenAI function shape.

    any-llm speaks OpenAI, so an Anthropic ``{name, description, input_schema}``
    tool must become ``{type: function, function: {name, description,
    parameters}}`` before it is forwarded, or the provider ignores/rejects the
    tools array and the model never calls a tool. Mirrors the LiteLLM backend's
    converter so both OpenAI-compatible backends send the same shape.
    """
    func: dict[str, Any] = {"name": tool.get("name", "")}
    if "description" in tool:
        func["description"] = tool["description"]
    if "input_schema" in tool:
        func["parameters"] = tool["input_schema"]
    return {"type": "function", "function": func}


def _convert_tool_choice(choice: Any) -> Any:
    """Convert an Anthropic ``tool_choice`` to the OpenAI shape (mirrors LiteLLM).

    Anthropic: ``{"type": "auto"}``, ``{"type": "any"}``, ``{"type": "tool",
    "name": ...}``. OpenAI: ``"auto"``, ``"required"``, ``{"type": "function",
    "function": {"name": ...}}``. Passing the raw Anthropic dict through makes
    the provider reject or ignore it.
    """
    if isinstance(choice, str):
        return choice
    if isinstance(choice, dict):
        choice_type = choice.get("type", "auto")
        if choice_type == "auto":
            return "auto"
        if choice_type == "any":
            return "required"
        if choice_type == "tool":
            return {"type": "function", "function": {"name": choice.get("name", "")}}
    return "auto"


class AnyLLMBackend(Backend):
    """Backend using any-llm for multi-provider support."""

    def __init__(
        self,
        provider: str = "openai",
        api_key: str | None = None,
        api_base: str | None = None,
    ):
        if not ANYLLM_AVAILABLE:
            raise ImportError(
                "any-llm-sdk is required for AnyLLMBackend. "
                "Install with: pip install 'any-llm-sdk[all]'"
            )

        self.provider = provider.lower()
        # Normalize empty-string overrides (e.g. an env var set to "") to None
        # so provider defaults stay active instead of forwarding a blank value.
        self.api_key = api_key or None
        self.api_base = api_base or None

        # Create the AnyLLM instance once and reuse. api_key/api_base are only
        # forwarded when set so providers keep their own env-var defaults
        # (e.g. OPENAI_API_KEY / OPENAI_BASE_URL) otherwise.
        create_kwargs: dict[str, Any] = {}
        if self.api_key is not None:
            create_kwargs["api_key"] = self.api_key
        if self.api_base is not None:
            create_kwargs["api_base"] = self.api_base
        self.llm = AnyLLM.create(self.provider, **create_kwargs)

        logger.info(
            f"any-llm backend initialized (provider={provider}, "
            f"api_base={self.api_base or 'default'})"
        )

    @property
    def name(self) -> str:
        return f"anyllm-{self.provider}"

    def map_model_id(self, model: str) -> str:
        """Pass through model name - any-llm handles provider-specific naming."""
        return model

    def supports_model(self, model: str) -> bool:
        """any-llm supports any model the provider supports."""
        return True

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert Anthropic message format to OpenAI/any-llm format."""
        converted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, str):
                converted.append(preserve_message_fields(msg, {"role": role, "content": content}))
                continue

            if isinstance(content, list):
                text_parts = []
                has_complex_content = False

                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") in ("tool_use", "tool_result", "image"):
                            has_complex_content = True
                            break

                if not has_complex_content and text_parts:
                    converted.append(
                        preserve_message_fields(
                            msg,
                            {"role": role, "content": "\n".join(text_parts)},
                        )
                    )
                else:
                    openai_content = self._convert_content_blocks(content)
                    converted.append(
                        preserve_message_fields(msg, {"role": role, "content": openai_content})
                    )

        return converted

    def _convert_content_blocks(self, blocks: list[dict[str, Any]]) -> list[dict[str, Any]] | str:
        """Convert Anthropic content blocks to OpenAI format."""
        openai_blocks = []

        for block in blocks:
            block_type = block.get("type")

            if block_type == "text":
                openai_blocks.append({"type": "text", "text": block.get("text", "")})

            elif block_type == "image":
                source = block.get("source", {})
                if source.get("type") == "base64":
                    media_type = source.get("media_type", "image/png")
                    data = source.get("data", "")
                    openai_blocks.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{data}"},
                        }
                    )
                elif source.get("type") == "url":
                    openai_blocks.append(
                        {"type": "image_url", "image_url": {"url": source.get("url", "")}}
                    )

        if len(openai_blocks) == 1 and openai_blocks[0].get("type") == "text":
            return str(openai_blocks[0]["text"])

        return openai_blocks if openai_blocks else ""

    def _to_anthropic_response(
        self,
        response: Any,
        original_model: str,
    ) -> dict[str, Any]:
        """Convert any-llm/OpenAI response to Anthropic format."""
        msg_id = f"msg_{uuid.uuid4().hex[:24]}"

        # A non-streaming upstream response can be HTTP 200 with an empty
        # ``choices`` list (e.g. Azure OpenAI content filtering, or any
        # OpenAI-compatible gateway on a usage-only / filtered turn). The
        # streaming sibling already skips this (`if ... and chunk.choices`);
        # indexing ``choices[0]`` here would instead raise IndexError and 500
        # the request. Return a valid empty assistant turn.
        if not getattr(response, "choices", None):
            usage = {"input_tokens": 0, "output_tokens": 0}
            if getattr(response, "usage", None):
                usage = {
                    "input_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
                    "output_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
                }
            return {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": original_model,
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": usage,
            }

        choice = response.choices[0]
        message = choice.message

        content = []
        if message.content:
            content.append({"type": "text", "text": message.content})

        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        pass
                content.append(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.function.name,
                        "input": args,
                    }
                )

        stop_reason_map = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "content_filter": "end_turn",
        }
        finish_reason = getattr(choice, "finish_reason", "stop") or "stop"
        stop_reason = stop_reason_map.get(finish_reason, "end_turn")

        usage = {"input_tokens": 0, "output_tokens": 0}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "input_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
                "output_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
            }

        return {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": content,
            "model": original_model,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": usage,
        }

    async def send_message(
        self,
        body: dict[str, Any],
        headers: dict[str, str],
    ) -> BackendResponse:
        """Send message via any-llm."""
        original_model = body.get("model", "gpt-4o")

        try:
            messages = self._convert_messages(body.get("messages", []))

            # Add system message if present
            if "system" in body:
                system = body["system"]
                if isinstance(system, str):
                    messages.insert(0, {"role": "system", "content": system})
                elif isinstance(system, list):
                    system_text = " ".join(
                        s.get("text", "") if isinstance(s, dict) else str(s) for s in system
                    )
                    messages.insert(0, {"role": "system", "content": system_text})

            kwargs: dict[str, Any] = {"model": original_model, "messages": messages}

            if "max_tokens" in body:
                kwargs["max_tokens"] = body["max_tokens"]
            if "temperature" in body:
                kwargs["temperature"] = body["temperature"]
            if "top_p" in body:
                kwargs["top_p"] = body["top_p"]
            if "stop_sequences" in body:
                kwargs["stop"] = body["stop_sequences"]
            if "tools" in body:
                kwargs["tools"] = [_convert_anthropic_tool(t) for t in body["tools"]]
            if "tool_choice" in body:
                kwargs["tool_choice"] = _convert_tool_choice(body["tool_choice"])

            logger.debug(f"any-llm request: provider={self.provider}, model={original_model}")

            response = await self.llm.acompletion(**kwargs)
            anthropic_response = self._to_anthropic_response(response, original_model)

            return BackendResponse(
                body=anthropic_response,
                status_code=200,
                headers={"content-type": "application/json"},
            )

        except Exception as e:
            logger.error(f"any-llm error: {e}")
            return self._error_response(e)

    async def stream_message(
        self,
        body: dict[str, Any],
        headers: dict[str, str],
    ) -> AsyncIterator[StreamEvent]:
        """Stream message via any-llm."""
        original_model = body.get("model", "gpt-4o")

        try:
            messages = self._convert_messages(body.get("messages", []))

            if "system" in body:
                system = body["system"]
                if isinstance(system, str):
                    messages.insert(0, {"role": "system", "content": system})

            kwargs: dict[str, Any] = {
                "model": original_model,
                "messages": messages,
                "stream": True,
            }

            if "max_tokens" in body:
                kwargs["max_tokens"] = body["max_tokens"]
            if "temperature" in body:
                kwargs["temperature"] = body["temperature"]
            if "top_p" in body:
                kwargs["top_p"] = body["top_p"]
            if "stop_sequences" in body:
                kwargs["stop"] = body["stop_sequences"]
            if "tools" in body:
                kwargs["tools"] = [_convert_anthropic_tool(t) for t in body["tools"]]
            if "tool_choice" in body:
                kwargs["tool_choice"] = _convert_tool_choice(body["tool_choice"])

            msg_id = f"msg_{uuid.uuid4().hex[:24]}"

            yield StreamEvent(
                event_type="message_start",
                data={
                    "type": "message_start",
                    "message": {
                        "id": msg_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": original_model,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                },
            )

            stream_response = await self.llm.acompletion(**kwargs)
            output_tokens = 0
            # Stream text immediately in a single text block, but BUFFER tool
            # calls and emit them as complete blocks at the end. OpenAI streams
            # parallel tool calls interleaved by index (index 0 and 1 introduced
            # together, then a fragment for 0, then for 1), while Anthropic
            # requires each content block to be fully emitted — start, deltas,
            # stop — before the next opens. Reassembling per index and flushing
            # complete blocks keeps every delta inside its own block's start/stop
            # for any interleaving. (The previous version pre-opened one text
            # block and dropped tool calls entirely; a naive open-on-new-index
            # instead mis-sequenced parallel calls, emitting a fragment for an
            # already-stopped block.)
            current_block_index = -1
            text_block_open = False
            # provider tool index -> {"id", "name", "arguments"}, first-seen order
            tool_calls: dict[int, dict[str, Any]] = {}
            tool_order: list[int] = []
            stop_reason = "end_turn"

            async for chunk in cast(AsyncIterator[Any], stream_response):
                if not (hasattr(chunk, "choices") and chunk.choices):
                    continue
                choice = chunk.choices[0]
                delta = choice.delta

                # Map OpenAI finish_reason to the Anthropic stop_reason so a tool
                # call or a length truncation is not reported as end_turn.
                finish_reason = getattr(choice, "finish_reason", None)
                if finish_reason == "tool_calls":
                    stop_reason = "tool_use"
                elif finish_reason == "length":
                    stop_reason = "max_tokens"
                elif finish_reason == "stop":
                    stop_reason = "end_turn"

                if getattr(delta, "tool_calls", None):
                    for tc in delta.tool_calls:
                        idx = tc.index if getattr(tc, "index", None) is not None else 0
                        buf = tool_calls.get(idx)
                        if buf is None:
                            buf = {"id": None, "name": "", "arguments": ""}
                            tool_calls[idx] = buf
                            tool_order.append(idx)
                        if getattr(tc, "id", None):
                            buf["id"] = tc.id
                        func = getattr(tc, "function", None)
                        if func is not None:
                            if getattr(func, "name", None):
                                buf["name"] = func.name
                            if getattr(func, "arguments", None):
                                buf["arguments"] += func.arguments

                elif getattr(delta, "content", None):
                    if not text_block_open:
                        current_block_index += 1
                        text_block_open = True
                        yield StreamEvent(
                            event_type="content_block_start",
                            data={
                                "type": "content_block_start",
                                "index": current_block_index,
                                "content_block": {"type": "text", "text": ""},
                            },
                        )
                    yield StreamEvent(
                        event_type="content_block_delta",
                        data={
                            "type": "content_block_delta",
                            "index": current_block_index,
                            "delta": {"type": "text_delta", "text": delta.content},
                        },
                    )
                    output_tokens += 1

            # Close the text block before any tool blocks (Anthropic orders
            # content blocks sequentially, text then tool_use).
            if text_block_open:
                yield StreamEvent(
                    event_type="content_block_stop",
                    data={"type": "content_block_stop", "index": current_block_index},
                )

            # Flush each buffered tool call as a complete, self-contained block:
            # start, one input_json_delta with the reassembled arguments, stop.
            for idx in tool_order:
                buf = tool_calls[idx]
                current_block_index += 1
                tool_id = buf["id"] or f"toolu_{uuid.uuid4().hex[:24]}"
                yield StreamEvent(
                    event_type="content_block_start",
                    data={
                        "type": "content_block_start",
                        "index": current_block_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": tool_id,
                            "name": buf["name"],
                            "input": {},
                        },
                    },
                )
                if buf["arguments"]:
                    yield StreamEvent(
                        event_type="content_block_delta",
                        data={
                            "type": "content_block_delta",
                            "index": current_block_index,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": buf["arguments"],
                            },
                        },
                    )
                    output_tokens += 1
                yield StreamEvent(
                    event_type="content_block_stop",
                    data={"type": "content_block_stop", "index": current_block_index},
                )

            yield StreamEvent(
                event_type="message_delta",
                data={
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": {"output_tokens": output_tokens},
                },
            )

            yield StreamEvent(
                event_type="message_stop",
                data={"type": "message_stop"},
            )

        except Exception as e:
            logger.error(f"any-llm streaming error: {e}")
            yield StreamEvent(
                event_type="error",
                data={
                    "type": "error",
                    "error": {"type": "api_error", "message": str(e)},
                },
            )

    async def send_openai_message(
        self,
        body: dict[str, Any],
        headers: dict[str, str],
    ) -> BackendResponse:
        """Send OpenAI-format message via any-llm (no conversion)."""
        original_model = body.get("model", "gpt-4o")

        try:
            kwargs: dict[str, Any] = {"model": original_model, "messages": body.get("messages", [])}

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
            ]:
                if param in body:
                    kwargs[param] = body[param]

            logger.debug(
                f"any-llm OpenAI request: provider={self.provider}, model={original_model}"
            )

            response: Any = await self.llm.acompletion(**kwargs)

            choices = []
            for c in response.choices:
                msg: dict[str, Any] = {
                    "role": c.message.role,
                    "content": c.message.content,
                }
                if c.message.tool_calls:
                    msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            **(
                                {
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    }
                                }
                                if hasattr(tc, "function") and tc.function
                                else {}
                            ),
                        }
                        for tc in c.message.tool_calls
                    ]
                choices.append(
                    {
                        "index": c.index,
                        "message": msg,
                        "finish_reason": c.finish_reason,
                    }
                )

            usage = response.usage
            response_dict: dict[str, Any] = {
                "id": response.id,
                "object": "chat.completion",
                "created": response.created,
                "model": original_model,
                "choices": choices,
                "usage": {
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "total_tokens": usage.total_tokens if usage else 0,
                },
            }

            return BackendResponse(
                body=response_dict,
                status_code=200,
                headers={"content-type": "application/json"},
            )

        except Exception as e:
            logger.error(f"any-llm OpenAI error: {e}")
            return self._error_response(e, openai_format=True)

    def _error_response(self, e: Exception, openai_format: bool = False) -> BackendResponse:
        """Build error response from exception."""
        error_type = "api_error"
        status_code = 500

        error_str = str(e).lower()
        if "authentication" in error_str or "api_key" in error_str or "api key" in error_str:
            error_type = "invalid_api_key" if openai_format else "authentication_error"
            status_code = 401
        elif "rate" in error_str or "limit" in error_str:
            error_type = "rate_limit_exceeded" if openai_format else "rate_limit_error"
            status_code = 429
        elif "not found" in error_str or "model" in error_str:
            error_type = "model_not_found" if openai_format else "not_found_error"
            status_code = 404

        body: dict[str, Any]
        if openai_format:
            body = {"error": {"message": str(e), "type": error_type, "code": error_type}}
        else:
            body = {"type": "error", "error": {"type": error_type, "message": str(e)}}

        return BackendResponse(body=body, status_code=status_code, error=str(e))

    async def stream_openai_message(
        self,
        body: dict[str, Any],
        headers: dict[str, str],
    ) -> AsyncIterator[str]:
        """Stream OpenAI-format chat completion via any-llm.

        Yields SSE-formatted strings ready to send to the client.
        """
        original_model = body.get("model", "gpt-4o")

        try:
            kwargs: dict[str, Any] = {
                "model": original_model,
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
            ]:
                if param in body:
                    kwargs[param] = body[param]

            if "stream_options" in body:
                kwargs["stream_options"] = body["stream_options"]

            stream_response = await self.llm.acompletion(**kwargs)

            async for chunk in cast(AsyncIterator[Any], stream_response):
                chunk_dict = chunk.model_dump(exclude_none=True, exclude_unset=True)
                yield f"data: {json.dumps(chunk_dict)}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"any-llm OpenAI streaming error: {e}")
            error_data = {
                "error": {
                    "message": str(e),
                    "type": "api_error",
                    "code": "backend_error",
                }
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"

    async def close(self) -> None:
        """Clean up (no-op for any-llm)."""
        pass
