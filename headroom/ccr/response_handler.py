"""Response handling for CCR (Compress-Cache-Retrieve).

This module provides response interception and CCR tool call handling.
When the LLM calls headroom_retrieve, this handler:
1. Detects the tool call in the response
2. Retrieves content from the compression store
3. Continues the conversation with the tool result
4. Returns the final response to the client

This solves the critical gap where the proxy injects the tool but
can't handle the LLM's tool calls.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from ..cache.compression_store import get_compression_store
from .tool_injection import CCR_TOOL_NAME, parse_tool_call

logger = logging.getLogger(__name__)


@dataclass
class CCRToolCall:
    """Represents a detected CCR tool call."""

    tool_call_id: str
    hash_key: str
    query: str | None = None


@dataclass
class CCRToolResult:
    """Result of handling a CCR tool call."""

    tool_call_id: str
    content: str
    success: bool
    items_retrieved: int = 0
    was_search: bool = False


@dataclass
class ResponseHandlerConfig:
    """Configuration for CCR response handling."""

    # Whether to handle CCR tool calls automatically
    enabled: bool = True

    # Maximum number of CCR retrieval rounds (prevent infinite loops)
    max_retrieval_rounds: int = 3

    # Whether to strip CCR tool calls from final response
    strip_ccr_from_response: bool = True

    # Timeout for continuation requests (ms)
    continuation_timeout_ms: int = 120000


class CCRResponseHandler:
    """Handles CCR tool calls in LLM responses.

    This handler intercepts responses, detects CCR tool calls,
    retrieves content, and continues the conversation until
    the LLM produces a response without CCR tool calls.

    Example flow:
    1. LLM response contains: tool_use(headroom_retrieve, hash=abc123)
    2. Handler detects this, retrieves original content
    3. Handler makes another API call with tool result
    4. LLM responds with actual content (no CCR tool call)
    5. Handler returns this final response

    Usage:
        handler = CCRResponseHandler(config)

        # Check if response needs handling
        if handler.has_ccr_tool_calls(response_json):
            # Handle the tool calls
            final_response = await handler.handle_response(
                response_json,
                messages,
                tools,
                api_call_fn,
                provider="anthropic"
            )
        else:
            final_response = response_json
    """

    def __init__(self, config: ResponseHandlerConfig | None = None):
        self.config = config or ResponseHandlerConfig()
        self._retrieval_count = 0
        self._retrieval_count_lock = __import__("threading").Lock()

    def has_ccr_tool_calls(
        self,
        response: dict[str, Any],
        provider: str = "anthropic",
    ) -> bool:
        """Check if response contains CCR tool calls.

        Args:
            response: The API response JSON.
            provider: The provider type.

        Returns:
            True if response contains headroom_retrieve tool calls.
        """
        tool_calls = self._extract_tool_calls(response, provider)
        return any(
            tc.get("name") == CCR_TOOL_NAME
            or tc.get("function", {}).get("name") == CCR_TOOL_NAME
            or tc.get("functionCall", {}).get("name") == CCR_TOOL_NAME  # Google format
            for tc in tool_calls
        )

    def _extract_tool_calls(
        self,
        response: dict[str, Any],
        provider: str,
    ) -> list[dict[str, Any]]:
        """Extract tool calls from response based on provider format."""
        if provider == "anthropic":
            # Anthropic format: content blocks with type=tool_use
            content = response.get("content", [])
            if isinstance(content, list):
                return [block for block in content if block.get("type") == "tool_use"]
            return []

        elif provider == "openai":
            # OpenAI format: message.tool_calls array
            message = response.get("choices", [{}])[0].get("message", {})
            tool_calls = message.get("tool_calls", [])
            return list(tool_calls) if tool_calls else []

        elif provider == "google":
            # Google/Gemini format: candidates[0].content.parts contains functionCall objects
            # Each part with a functionCall has: {"functionCall": {"name": "...", "args": {...}}}
            candidates = response.get("candidates", [])
            if not candidates:
                return []
            parts = candidates[0].get("content", {}).get("parts", [])
            return [part for part in parts if "functionCall" in part]

        return []

    def _parse_ccr_tool_calls(
        self,
        response: dict[str, Any],
        provider: str,
    ) -> tuple[list[CCRToolCall], list[dict[str, Any]]]:
        """Parse CCR tool calls from response, separate from other tool calls.

        Returns:
            Tuple of (ccr_tool_calls, other_tool_calls)
        """
        all_tool_calls = self._extract_tool_calls(response, provider)

        ccr_calls = []
        other_calls = []

        for tc in all_tool_calls:
            hash_key, query = parse_tool_call(tc, provider)

            if hash_key is not None:
                # This is a CCR tool call - extract tool_call_id based on provider
                if provider == "google":
                    # Google uses function name as identifier for matching responses
                    # The functionResponse.name must match the functionCall.name
                    tool_call_id = tc.get("functionCall", {}).get("name", CCR_TOOL_NAME)
                else:
                    # Anthropic and OpenAI use explicit IDs
                    tool_call_id = tc.get("id", "")
                ccr_calls.append(
                    CCRToolCall(
                        tool_call_id=tool_call_id,
                        hash_key=hash_key,
                        query=query,
                    )
                )
            else:
                # Not a CCR tool call
                other_calls.append(tc)

        return ccr_calls, other_calls

    def _execute_retrieval(self, ccr_call: CCRToolCall) -> CCRToolResult:
        """Execute a CCR retrieval.

        Args:
            ccr_call: The CCR tool call to execute.

        Returns:
            CCRToolResult with the retrieved content.
        """
        store = get_compression_store()

        try:
            if ccr_call.query:
                # Search within compressed content
                results = store.search(ccr_call.hash_key, ccr_call.query)
                content = json.dumps(
                    {
                        "hash": ccr_call.hash_key,
                        "query": ccr_call.query,
                        "results": results,
                        "count": len(results),
                    },
                    indent=2,
                )
                return CCRToolResult(
                    tool_call_id=ccr_call.tool_call_id,
                    content=content,
                    success=True,
                    items_retrieved=len(results),
                    was_search=True,
                )
            else:
                # Full retrieval
                entry = store.retrieve(ccr_call.hash_key)
                if entry:
                    content = json.dumps(
                        {
                            "hash": ccr_call.hash_key,
                            "original_content": entry.original_content,
                            "original_item_count": entry.original_item_count,
                        },
                        indent=2,
                    )
                    return CCRToolResult(
                        tool_call_id=ccr_call.tool_call_id,
                        content=content,
                        success=True,
                        items_retrieved=entry.original_item_count,
                        was_search=False,
                    )
                else:
                    content = json.dumps(
                        {
                            "error": "Entry not found or expired (TTL: 5 minutes)",
                            "hash": ccr_call.hash_key,
                        },
                        indent=2,
                    )
                    return CCRToolResult(
                        tool_call_id=ccr_call.tool_call_id,
                        content=content,
                        success=False,
                    )

        except Exception as e:
            logger.error(f"CCR retrieval failed for {ccr_call.hash_key}: {e}")
            content = json.dumps(
                {
                    "error": f"Retrieval failed: {str(e)}",
                    "hash": ccr_call.hash_key,
                },
                indent=2,
            )
            return CCRToolResult(
                tool_call_id=ccr_call.tool_call_id,
                content=content,
                success=False,
            )

    def _create_tool_result_message(
        self,
        results: list[CCRToolResult],
        provider: str,
    ) -> dict[str, Any]:
        """Create a tool result message from CCR results.

        Args:
            results: List of CCR tool results.
            provider: The provider type.

        Returns:
            Message dict in the appropriate format.
        """
        if provider == "anthropic":
            # Anthropic: user message with tool_result content blocks
            content_blocks = []
            for result in results:
                content_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": result.tool_call_id,
                        "content": result.content,
                    }
                )
            return {
                "role": "user",
                "content": content_blocks,
            }

        elif provider == "openai":
            # OpenAI: multiple tool messages
            # Actually for OpenAI we return a list of messages
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

        elif provider == "google":
            # Google/Gemini: user message with functionResponse parts
            # Format: {"role": "user", "parts": [{"functionResponse": {"name": "...", "response": {...}}}]}
            parts = []
            for result in results:
                # Parse the content JSON to include as response object
                try:
                    response_data = json.loads(result.content)
                except json.JSONDecodeError:
                    response_data = {"content": result.content}
                parts.append(
                    {
                        "functionResponse": {
                            "name": result.tool_call_id,  # tool_call_id contains the function name for Google
                            "response": response_data,
                        }
                    }
                )
            return {
                "role": "user",
                "parts": parts,
            }

        else:
            # Generic format
            return {
                "role": "tool",
                "content": json.dumps(
                    [{"tool_call_id": r.tool_call_id, "result": r.content} for r in results]
                ),
            }

    def _extract_assistant_message(
        self,
        response: dict[str, Any],
        provider: str,
    ) -> dict[str, Any]:
        """Extract the assistant message from an API response.

        Args:
            response: The API response.
            provider: The provider type.

        Returns:
            The assistant message dict.
        """
        if provider == "anthropic":
            return {
                "role": "assistant",
                "content": response.get("content", []),
            }
        elif provider == "openai":
            message = response.get("choices", [{}])[0].get("message", {})
            return {
                "role": "assistant",
                "content": message.get("content"),
                "tool_calls": message.get("tool_calls"),
            }
        elif provider == "google":
            # Google/Gemini format: role is "model", content is in candidates[0].content.parts
            candidates = response.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
            else:
                parts = []
            return {
                "role": "model",
                "parts": parts,
            }
        else:
            return {
                "role": "assistant",
                "content": response.get("content", ""),
            }

    async def handle_response(
        self,
        response: dict[str, Any],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        api_call_fn: Callable[
            [list[dict[str, Any]], list[dict[str, Any]] | None], Awaitable[dict[str, Any]]
        ],
        provider: str = "anthropic",
    ) -> dict[str, Any]:
        """Handle CCR tool calls in a response.

        This method:
        1. Detects CCR tool calls
        2. Executes retrievals
        3. Continues conversation with tool results
        4. Repeats until no CCR tool calls remain

        Args:
            response: The initial API response.
            messages: The conversation messages.
            tools: The tools list (should include CCR tool).
            api_call_fn: Async function to make API calls.
                         Signature: (messages, tools) -> response
            provider: The provider type.

        Returns:
            The final response (with no CCR tool calls).
        """
        if not self.config.enabled:
            return response

        current_response = response
        current_messages = list(messages)  # Copy to avoid mutation
        rounds = 0

        while rounds < self.config.max_retrieval_rounds:
            # Check for CCR tool calls
            ccr_calls, other_calls = self._parse_ccr_tool_calls(current_response, provider)

            if not ccr_calls:
                # No CCR tool calls, we're done
                break

            rounds += 1
            with self._retrieval_count_lock:
                self._retrieval_count += len(ccr_calls)

            logger.info(f"CCR: Handling {len(ccr_calls)} retrieval(s) in round {rounds}")

            # Execute all CCR retrievals
            results = [self._execute_retrieval(call) for call in ccr_calls]

            # Log retrieval stats
            total_items = sum(r.items_retrieved for r in results)
            searches = sum(1 for r in results if r.was_search)
            logger.debug(
                f"CCR: Retrieved {total_items} items "
                f"({searches} searches, {len(results) - searches} full)"
            )

            # Build continuation messages
            # Add assistant message (the response that had tool calls)
            assistant_msg = self._extract_assistant_message(current_response, provider)
            current_messages.append(assistant_msg)

            # Add tool results
            tool_result_msg = self._create_tool_result_message(results, provider)

            if provider == "openai" and "_openai_tool_results" in tool_result_msg:
                # OpenAI uses multiple messages for tool results
                current_messages.extend(tool_result_msg["_openai_tool_results"])
            else:
                current_messages.append(tool_result_msg)

            # Make continuation API call
            try:
                current_response = await api_call_fn(current_messages, tools)
            except Exception as e:
                logger.error(f"CCR: Continuation API call failed: {e}")
                # Return the response we had (with unhandled CCR calls)
                # The client will see the tool_use and might handle it differently
                break

        if rounds >= self.config.max_retrieval_rounds:
            logger.warning(
                f"CCR: Hit max retrieval rounds ({self.config.max_retrieval_rounds}), "
                f"returning response with possible unhandled CCR calls"
            )

        return current_response

    def get_stats(self) -> dict[str, Any]:
        """Get handler statistics."""
        return {
            "total_retrievals": self._retrieval_count,
            "config": {
                "enabled": self.config.enabled,
                "max_rounds": self.config.max_retrieval_rounds,
            },
        }


@dataclass
class StreamingCCRBuffer:
    """Buffer for detecting CCR tool calls in streaming responses.

    Since streaming responses come in chunks, we need to buffer
    until we can detect whether there's a CCR tool call.

    Strategy:
    1. Buffer chunks until we see a complete tool_use block
    2. If it's a CCR call, switch to buffered mode
    3. Handle CCR and then stream the continuation
    """

    chunks: list[bytes] = field(default_factory=list)
    detected_ccr: bool = False
    complete_response: dict[str, Any] | None = None

    # Patterns to detect tool_use in stream
    _tool_use_start: bytes = b'"type":"tool_use"'
    _ccr_tool_pattern: bytes = f'"{CCR_TOOL_NAME}"'.encode()

    def add_chunk(self, chunk: bytes) -> bool:
        """Add a chunk and check for CCR tool calls.

        Returns:
            True if CCR tool call detected (should switch to buffered mode).
        """
        self.chunks.append(chunk)

        # Quick check: does accumulated content contain CCR tool?
        accumulated = b"".join(self.chunks)

        if self._tool_use_start in accumulated and self._ccr_tool_pattern in accumulated:
            self.detected_ccr = True
            return True

        return False

    def get_accumulated(self) -> bytes:
        """Get all accumulated chunks."""
        return b"".join(self.chunks)

    def clear(self) -> None:
        """Clear the buffer."""
        self.chunks.clear()
        self.detected_ccr = False
        self.complete_response = None


class StreamingCCRHandler:
    """Handle CCR tool calls in streaming responses.

    For streaming, we have two modes:
    1. Pass-through: No CCR detected, stream chunks directly
    2. Buffered: CCR detected, buffer response, handle, then stream result

    The challenge is we can't know if there's a CCR call until we see
    enough of the response. So we buffer initially, then decide.
    """

    def __init__(
        self,
        response_handler: CCRResponseHandler,
        provider: str = "anthropic",
    ) -> None:
        self.response_handler = response_handler
        self.provider = provider
        self.buffer = StreamingCCRBuffer()

    async def process_stream(
        self,
        stream_iterator: Any,  # AsyncIterator[bytes]
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        api_call_fn: Callable[
            [list[dict[str, Any]], list[dict[str, Any]] | None], Awaitable[dict[str, Any]]
        ],
    ) -> Any:  # AsyncGenerator[bytes, None]
        """Process a streaming response, handling CCR if needed.

        This is an async generator that yields chunks.
        If CCR is detected, it buffers, handles, and re-streams.

        Args:
            stream_iterator: Async iterator of response chunks.
            messages: The conversation messages.
            tools: The tools list.
            api_call_fn: Function to make API calls for continuation.

        Yields:
            Response chunks (possibly from continuation response).
        """
        # Phase 1: Initial detection
        # Buffer chunks until we can determine if there's a CCR call
        detection_complete = False

        async for chunk in stream_iterator:
            self.buffer.add_chunk(chunk)

            # Check if we can determine CCR presence
            # For Anthropic, tool_use blocks come after text content
            # We need to see the stop_reason to know if there's a tool call
            accumulated = self.buffer.get_accumulated()

            # Look for stream end markers
            if b'"stop_reason"' in accumulated:
                detection_complete = True

                if self.buffer.detected_ccr:
                    # CCR detected - need to handle
                    break
                else:
                    # No CCR - yield all buffered chunks
                    for buffered_chunk in self.buffer.chunks:
                        yield buffered_chunk
                    self.buffer.clear()

            # If we haven't detected anything yet and buffer is large,
            # start yielding (response is probably just text)
            elif len(accumulated) > 10000 and not self.buffer.detected_ccr:
                for buffered_chunk in self.buffer.chunks:
                    yield buffered_chunk
                self.buffer.clear()

        # Continue streaming rest of response
        if not detection_complete and not self.buffer.detected_ccr:
            async for chunk in stream_iterator:
                if self.buffer.detected_ccr:
                    self.buffer.add_chunk(chunk)
                else:
                    yield chunk

        # Phase 2: Handle CCR if detected
        if self.buffer.detected_ccr:
            logger.info("CCR: Detected tool call in stream, switching to buffered mode")

            # Collect rest of stream with timeout to prevent indefinite blocking
            import asyncio

            try:
                async for chunk in stream_iterator:
                    self.buffer.add_chunk(chunk)
            except asyncio.TimeoutError:
                logger.warning("CCR: Timed out collecting rest of stream")
            except Exception as e:
                logger.error(f"CCR: Error collecting rest of stream: {e}")

            # Parse the complete response
            try:
                # For SSE streams, we need to parse the accumulated data
                complete_data = self._parse_sse_stream(self.buffer.get_accumulated())

                # Handle CCR
                final_response = await self.response_handler.handle_response(
                    complete_data,
                    messages,
                    tools,
                    api_call_fn,
                    self.provider,
                )

                # Re-stream the final response
                # Convert back to SSE format
                async for chunk in self._response_to_sse(final_response):
                    yield chunk

            except Exception as e:
                logger.error(f"CCR: Failed to handle streamed CCR: {e}")
                # Fall back to yielding original buffered content
                yield self.buffer.get_accumulated()

    def _parse_sse_stream(self, data: bytes) -> dict[str, Any]:
        """Parse SSE stream data into a response dict.

        SSE format: data: {...}\\n\\n

        PR-A8 / P1-8: bytes-level event splitter; each complete event
        decodes as UTF-8 only AFTER the ``\\n\\n`` boundary has been
        located in bytes. Multi-byte characters split across upstream
        TCP reads are preserved intact. Invalid UTF-8 in a *complete*
        event is an upstream protocol bug — surfaced loudly, not
        silently corrupted.
        """
        from headroom.proxy.helpers import parse_sse_events_from_byte_buffer

        # Accumulate all event data via the canonical bytes-buffer
        # splitter. ``data`` is a closed payload here, so any partial
        # tail bytes left in ``buf`` indicate the upstream truncated
        # mid-event — log and ignore (already handled at the streaming
        # layer above).
        buf = bytearray(data)
        events: list[dict[str, Any]] = []
        for _event_name, data_str in parse_sse_events_from_byte_buffer(buf):
            stripped = data_str.strip()
            if not stripped or stripped == "[DONE]":
                continue
            try:
                events.append(json.loads(stripped))
            except json.JSONDecodeError:
                # Per-event JSON garbage from upstream — skip the
                # event but keep the rest of the stream parseable.
                continue
        if buf:
            logger.debug(
                "CCR: %d trailing bytes left in SSE buffer after parse "
                "(upstream truncated mid-event)",
                len(buf),
            )

        # Reconstruct response from events
        # This is provider-specific
        if self.provider == "anthropic":
            return self._reconstruct_anthropic_response(events)
        else:
            return self._reconstruct_openai_response(events)

    def _reconstruct_anthropic_response(
        self,
        events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Reconstruct Anthropic response from stream events."""
        response: dict[str, Any] = {
            "content": [],
            "stop_reason": None,
            "usage": {},
        }

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
                elif delta.get("type") == "input_json_delta":
                    # Accumulate JSON for tool input
                    if current_tool is not None:
                        partial = delta.get("partial_json", "")
                        # This is tricky - partial JSON needs accumulation
                        # For simplicity, we'll try to parse when complete
                        current_tool["_partial_json"] = (
                            current_tool.get("_partial_json", "") + partial
                        )

            elif event_type == "content_block_stop":
                if current_text:
                    response["content"].append(
                        {
                            "type": "text",
                            "text": current_text,
                        }
                    )
                    current_text = ""
                if current_tool:
                    # Parse accumulated JSON
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

            elif event_type == "message_stop":
                pass

        return response

    def _reconstruct_openai_response(
        self,
        events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Reconstruct OpenAI response from stream events."""
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

        return {
            "choices": [{"message": message, "finish_reason": "stop"}],
        }

    async def _response_to_sse(
        self,
        response: dict[str, Any],
    ) -> Any:  # AsyncGenerator[bytes, None]
        """Convert a response back to SSE format for streaming.

        This is a simplified version - in practice you might want
        to chunk the response more granularly.
        """
        if self.provider == "anthropic":
            # Anthropic SSE format
            yield b"event: message_start\n"
            yield f"data: {json.dumps({'type': 'message_start', 'message': response})}\n\n".encode()
            yield b"event: message_stop\n"
            yield b'data: {"type": "message_stop"}\n\n'
        else:
            # OpenAI SSE format
            yield f"data: {json.dumps(response)}\n\n".encode()
            yield b"data: [DONE]\n\n"
