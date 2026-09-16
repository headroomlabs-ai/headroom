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

from ..cache.compression_store import format_retrieval_miss_detail, get_compression_store
from .tool_calls import (
    CCRToolCall,
    extract_tool_calls,
    has_ccr_tool_calls,
    parse_ccr_tool_calls,
)
from .tool_injection import CCR_TOOL_NAME

logger = logging.getLogger(__name__)

# Residual-CCR status signals (provider-generic).
#
# ``handle_response`` may return a response that still contains
# ``headroom_retrieve`` tool calls. Callers need to know *why* so they can
# decide whether that is a safe passthrough or a genuine failure:
#
# - RESIDUAL_CCR_RESOLVED:       no CCR tool calls remain — fully handled.
# - RESIDUAL_CCR_SKIPPED_MIXED:  CCR was intentionally skipped because the model
#                                emitted headroom_retrieve alongside a non-CCR
#                                client tool (#839). The client must resolve both
#                                tool calls; the proxy must pass the turn through
#                                unchanged (200), not fail closed.
# - RESIDUAL_CCR_ERROR:          CCR tool calls remain with no accompanying client
#                                tool — i.e. a real conversion/handling failure the
#                                proxy could not resolve. Callers should fail closed.
RESIDUAL_CCR_RESOLVED = "resolved"
RESIDUAL_CCR_SKIPPED_MIXED = "skipped_mixed_tools"
RESIDUAL_CCR_ERROR = "error"


@dataclass
class CCRToolResult:
    """Result of handling a CCR tool call."""

    tool_call_id: str
    content: str
    success: bool
    items_retrieved: int = 0
    tool_name: str | None = None


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
        return has_ccr_tool_calls(response, provider)

    def residual_ccr_status(
        self,
        response: dict[str, Any],
        provider: str = "anthropic",
    ) -> str:
        """Classify why (if at all) CCR tool calls remain in a handled response.

        This is a stateless, provider-generic signal derived from the same
        parsing ``handle_response`` uses, so it stays correct under concurrency
        and works identically for every provider/harness.

        Returns one of:
        - ``RESIDUAL_CCR_RESOLVED``: no headroom_retrieve tool calls remain.
        - ``RESIDUAL_CCR_SKIPPED_MIXED``: headroom_retrieve remains *alongside*
          a non-CCR client tool call. This is an intentional skip (#839) — the
          proxy cannot synthesize the client tool_result, so the turn must be
          handed back to the client unchanged rather than failed closed.
        - ``RESIDUAL_CCR_ERROR``: headroom_retrieve remains with no accompanying
          client tool call — a genuine handling/conversion failure.
        """
        ccr_calls, other_calls = self._parse_ccr_tool_calls(response, provider)
        if not ccr_calls:
            return RESIDUAL_CCR_RESOLVED
        if other_calls:
            return RESIDUAL_CCR_SKIPPED_MIXED
        return RESIDUAL_CCR_ERROR

    def _extract_tool_calls(
        self,
        response: dict[str, Any],
        provider: str,
    ) -> list[dict[str, Any]]:
        """Extract tool calls from response based on provider format."""
        if provider == "openai" and response.get("choices") == []:
            # Preserve legacy private-method behavior for existing tests/callers.
            raise IndexError("list index out of range")
        return extract_tool_calls(response, provider)

    def _parse_ccr_tool_calls(
        self,
        response: dict[str, Any],
        provider: str,
    ) -> tuple[list[CCRToolCall], list[dict[str, Any]]]:
        """Parse CCR tool calls from response, separate from other tool calls.

        Returns:
            Tuple of (ccr_tool_calls, other_tool_calls)
        """
        return parse_ccr_tool_calls(response, provider)

    def _execute_retrieval(self, ccr_call: CCRToolCall) -> CCRToolResult:
        """Execute a CCR retrieval.

        Args:
            ccr_call: The CCR tool call to execute.

        Returns:
            CCRToolResult with the retrieved content.
        """
        store = get_compression_store()

        try:
            get_status = getattr(store, "get_entry_status", None)
            entry_status = (
                get_status(ccr_call.hash_key, clean_expired=True) if callable(get_status) else None
            )
            if entry_status is not None and entry_status["status"] != "available":
                content = json.dumps(
                    {
                        "error": format_retrieval_miss_detail(entry_status),
                        "hash": ccr_call.hash_key,
                        "status": entry_status["status"],
                        "ttl_seconds": entry_status.get(
                            "ttl_seconds", entry_status["default_ttl_seconds"]
                        ),
                    },
                    indent=2,
                )
                return CCRToolResult(
                    tool_call_id=ccr_call.tool_call_id,
                    content=content,
                    success=False,
                    tool_name=ccr_call.tool_name,
                )

            # Retrieval is by hash: always return the full original content.
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
                    tool_name=ccr_call.tool_name,
                )

            miss_status = (
                get_status(ccr_call.hash_key, clean_expired=True)
                if callable(get_status)
                else {"hash": ccr_call.hash_key, "status": "missing"}
            )
            content = json.dumps(
                {
                    "error": format_retrieval_miss_detail(miss_status),
                    "hash": ccr_call.hash_key,
                    "status": miss_status["status"],
                    "ttl_seconds": miss_status.get("ttl_seconds"),
                },
                indent=2,
            )
            return CCRToolResult(
                tool_call_id=ccr_call.tool_call_id,
                content=content,
                success=False,
                tool_name=ccr_call.tool_name,
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
                tool_name=ccr_call.tool_name,
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

        elif provider == "openai_responses":
            # Responses API: `function_call_output` items, echoed back into
            # `input[]` alongside (not nested under) the preceding
            # function_call items. Sentinel key mirrors the "openai"
            # multi-message pattern above — handle_response() extends
            # rather than appends when it sees this key.
            return {
                "_openai_responses_tool_results": [
                    {
                        "type": "function_call_output",
                        "call_id": result.tool_call_id,
                        "output": result.content,
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
                function_response = {
                    "name": result.tool_name or result.tool_call_id,
                    "response": response_data,
                }
                if result.tool_name and result.tool_call_id != result.tool_name:
                    function_response["id"] = result.tool_call_id
                parts.append({"functionResponse": function_response})
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
            # Guard an empty/malformed ``choices`` the same way the Google branch
            # below (and ccr/tool_calls.py) already do: ``response.get("choices",
            # [{}])`` only falls back when the key is absent, so a present-but-
            # empty ``choices: []`` (or ``[null]``) — which OpenAI-compatible
            # gateways can send on a content-filtered/usage-only response — made
            # ``[0]`` raise IndexError (or ``.get`` raise on a non-dict).
            choices = response.get("choices")
            first = choices[0] if isinstance(choices, list) and choices else {}
            message = first.get("message", {}) if isinstance(first, dict) else {}
            return {
                "role": "assistant",
                "content": message.get("content"),
                "tool_calls": message.get("tool_calls"),
            }
        elif provider == "openai_responses":
            # Responses API: the model's turn is the full `output[]` array
            # (function_call items, message items, reasoning items, ...),
            # echoed back verbatim as `input[]` items — not a single
            # role/content dict like chat completions. Sentinel key mirrors
            # `_openai_tool_results`; handle_response() extends on it.
            # `.get("output", [])` only falls back when the key is absent, so a
            # present-but-null `output` would return None and make the
            # `current_messages.extend(...)` in handle_response raise TypeError;
            # coerce to a list like the choices branch above.
            output_items = response.get("output")
            return {
                "_openai_responses_output_items": output_items
                if isinstance(output_items, list)
                else []
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

            # If the model called CCR alongside non-CCR tools, we cannot build
            # a valid continuation — every tool_use in the assistant message
            # requires a matching tool_result, but we only have CCR results.
            # Skip CCR handling and let the client resolve all tool calls.
            if other_calls:
                logger.warning(
                    "CCR: Skipping CCR handling — model called %d non-CCR tool(s) "
                    "alongside headroom_retrieve. Cannot create a valid continuation "
                    "without results for the other tools. Client must handle all tool calls.",
                    len(other_calls),
                )
                break

            rounds += 1
            with self._retrieval_count_lock:
                self._retrieval_count += len(ccr_calls)

            logger.info(f"CCR: Handling {len(ccr_calls)} retrieval(s) in round {rounds}")

            # Execute all CCR retrievals
            results = [self._execute_retrieval(call) for call in ccr_calls]

            # Log retrieval stats
            total_items = sum(r.items_retrieved for r in results)
            logger.debug(
                f"CCR: Retrieved {total_items} items across {len(results)} full retrieval(s)"
            )

            # Build continuation messages
            # Add assistant message (the response that had tool calls).
            # Responses API turns are a list of output items rather than a
            # single role/content dict, so extend on that sentinel instead
            # of appending it as one entry.
            assistant_msg = self._extract_assistant_message(current_response, provider)
            if (
                isinstance(assistant_msg, dict)
                and "_openai_responses_output_items" in assistant_msg
            ):
                current_messages.extend(assistant_msg["_openai_responses_output_items"])
            else:
                current_messages.append(assistant_msg)

            # Add tool results
            tool_result_msg = self._create_tool_result_message(results, provider)

            if provider == "openai" and "_openai_tool_results" in tool_result_msg:
                # OpenAI uses multiple messages for tool results
                current_messages.extend(tool_result_msg["_openai_tool_results"])
            elif "_openai_responses_tool_results" in tool_result_msg:
                current_messages.extend(tool_result_msg["_openai_responses_tool_results"])
            else:
                current_messages.append(tool_result_msg)

            # Make continuation API call
            try:
                current_response = await api_call_fn(current_messages, tools)
            except Exception as e:
                # Log the type and repr, not just str(e): many exceptions
                # (httpx.TimeoutException(''), a bare Exception(), a connect
                # error with no message) have an empty str(), which produced a
                # log line with nothing after the colon and lost the cause
                # entirely (#3129).
                logger.error("CCR: Continuation API call failed: %s: %r", type(e).__name__, e)
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
    provider: str = "anthropic"

    # Wire markers for the start of a tool call. Anthropic streams
    # `"type":"tool_use"` content blocks; OpenAI-compatible streams carry a
    # `"tool_calls"` array inside `choices[].delta` and never emit the
    # Anthropic marker, so scanning only for the latter meant CCR was never
    # detected on an OpenAI stream.
    _tool_use_start: bytes = b'"type":"tool_use"'
    _openai_tool_use_start: bytes = b'"tool_calls"'
    _ccr_tool_pattern: bytes = f'"{CCR_TOOL_NAME}"'.encode()

    def _tool_call_marker(self) -> bytes:
        """The provider's on-the-wire marker for the start of a tool call."""
        if self.provider == "anthropic":
            return self._tool_use_start
        return self._openai_tool_use_start

    def add_chunk(self, chunk: bytes) -> bool:
        """Add a chunk and check for CCR tool calls.

        Returns:
            True if CCR tool call detected (should switch to buffered mode).
        """
        self.chunks.append(chunk)

        # Quick check: does accumulated content contain CCR tool?
        accumulated = b"".join(self.chunks)

        if self._tool_call_marker() in accumulated and self._ccr_tool_pattern in accumulated:
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
        self.buffer = StreamingCCRBuffer(provider=provider)

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
        # Buffer chunks until we can determine if there's a CCR call.
        #
        # The end-of-stream marker is provider-specific. Anthropic signals the
        # terminal state with `stop_reason` in `message_delta`; OpenAI-compatible
        # streams have no such field and terminate with the `[DONE]` sentinel.
        end_marker = b'"stop_reason"' if self.provider == "anthropic" else b"data: [DONE]"

        async for chunk in stream_iterator:
            self.buffer.add_chunk(chunk)

            # Check if we can determine CCR presence
            # For Anthropic, tool_use blocks come after text content
            # We need to see the stop_reason to know if there's a tool call
            accumulated = self.buffer.get_accumulated()

            # Look for stream end markers
            if end_marker in accumulated:
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

        # The end marker is not guaranteed to arrive: upstream can truncate, a
        # provider can omit the sentinel, or the stream can be a shape this
        # detector does not recognise. Anything still buffered once the source
        # iterator is exhausted is real response data the client has never
        # seen, so flush it instead of dropping it.
        if not self.buffer.detected_ccr and self.buffer.chunks:
            for buffered_chunk in self.buffer.chunks:
                yield buffered_chunk
            self.buffer.clear()

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

        blocks_by_index: dict[int, dict[str, Any]] = {}
        current_block: dict[str, Any] | None = None

        for event in events:
            event_type = event.get("type", "")

            if event_type == "content_block_start":
                block = event.get("content_block", {})
                block_index = event.get("index", len(blocks_by_index))
                btype = block.get("type")
                current_block = {"type": btype}
                if btype == "text":
                    current_block["text"] = block.get("text", "")
                elif btype == "tool_use":
                    current_block.update(
                        {
                            "id": block.get("id", ""),
                            "name": block.get("name", ""),
                            "input": {},
                        }
                    )
                elif btype == "thinking":
                    current_block["thinking_buffer"] = block.get("thinking", "")
                    if "signature" in block:
                        current_block["signature"] = block["signature"]
                elif btype == "redacted_thinking":
                    if "data" in block:
                        current_block["data"] = block["data"]
                elif btype:
                    current_block = dict(block)
                blocks_by_index[block_index] = current_block

            elif event_type == "content_block_delta":
                idx = event.get("index")
                target = (blocks_by_index.get(idx) if idx is not None else None) or current_block
                if target is None:
                    continue
                delta = event.get("delta", {})
                dtype = delta.get("type")
                if dtype == "text_delta":
                    target["text"] = target.get("text", "") + delta.get("text", "")
                elif dtype == "input_json_delta":
                    # Accumulate for any block streaming input (tool_use AND
                    # server_tool_use); the stop handler parses it into `input`
                    # (#2438).
                    partial = delta.get("partial_json", "")
                    target["_partial_json"] = target.get("_partial_json", "") + partial
                elif dtype == "thinking_delta":
                    target["thinking_buffer"] = target.get("thinking_buffer", "") + delta.get(
                        "thinking", ""
                    )
                elif dtype == "signature_delta":
                    if "signature" in delta:
                        target["signature"] = delta["signature"]
                elif dtype == "citations_delta":
                    citation = delta.get("citation")
                    if citation is not None:
                        target.setdefault("citations", []).append(citation)

            elif event_type == "content_block_stop":
                idx = event.get("index")
                target = (blocks_by_index.get(idx) if idx is not None else None) or current_block
                if target is not None:
                    # Parse streamed `_partial_json` into `input` for any block
                    # that carried input_json_delta — tool_use AND
                    # server_tool_use — not just tool_use. The narrow type gate
                    # left server_tool_use.input malformed and leaked the scratch
                    # key into replayed history (#2438). Always strip the key.
                    if "_partial_json" in target:
                        partial = target.pop("_partial_json")
                        try:
                            target["input"] = json.loads(partial) if partial else {}
                        except json.JSONDecodeError:
                            target["input"] = {}
                    if target.get("type") == "thinking" and "thinking_buffer" in target:
                        target["thinking"] = target.pop("thinking_buffer")
                    if target not in response["content"]:
                        response["content"].append(target)
                    current_block = None

            elif event_type == "message_start":
                msg = event.get("message", {})
                if "id" in msg:
                    response["id"] = msg["id"]
                if "model" in msg:
                    response["model"] = msg["model"]
                if "role" in msg:
                    response["role"] = msg["role"]
                if "stop_reason" in msg:
                    response["stop_reason"] = msg["stop_reason"]
                if "stop_details" in msg:
                    response["stop_details"] = msg["stop_details"]
                if msg.get("usage"):
                    response["usage"].update(msg["usage"])

            elif event_type == "message_delta":
                delta = event.get("delta", {})
                if "stop_reason" in delta:
                    response["stop_reason"] = delta["stop_reason"]
                if "stop_details" in delta:
                    response["stop_details"] = delta["stop_details"]
                if event.get("usage"):
                    response["usage"].update(event["usage"])

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
        finish_reason: str | None = None
        envelope: dict[str, Any] = {}
        usage: Any = None

        for event in events:
            # Carry the chunk envelope through. Dropping it left the
            # reconstructed body without `id`, `model`, `created` or `usage`,
            # which downstream middleware reads for routing and metering.
            for key in ("id", "created", "model", "system_fingerprint"):
                value = event.get(key)
                if value is not None:
                    envelope[key] = value
            if event.get("usage") is not None:
                usage = event["usage"]

            choices = event.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0]
            if not isinstance(choice, dict):
                continue

            # `finish_reason` is null on every chunk but the last, so keep the
            # most recent non-null value rather than the first one seen.
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]

            # Some OpenAI-compatible providers send `"delta": null` on the
            # terminal chunk instead of an empty object.
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                delta = {}

            if "content" in delta and delta["content"]:
                message["content"] = (message.get("content") or "") + delta["content"]

            # Guard the value, not just the key: some OpenAI-compatible
            # providers include ``"tool_calls": null`` (and ``"function": null``)
            # in a delta rather than omitting the key, which would make the
            # iteration below raise ``TypeError: 'NoneType' object is not
            # iterable`` and abort the whole reconstruction. Mirrors the
            # ``and delta["content"]`` value-guard above.
            tool_calls = delta.get("tool_calls")
            if isinstance(tool_calls, list):
                for tc_delta in tool_calls:
                    if not isinstance(tc_delta, dict):
                        continue
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
                    fn = tc_delta.get("function")
                    if isinstance(fn, dict):
                        if "name" in fn:
                            tc["function"]["name"] = fn["name"]
                        if "arguments" in fn:
                            tc["function"]["arguments"] += fn["arguments"]

        message["tool_calls"] = [tool_calls_map[i] for i in sorted(tool_calls_map.keys())]
        has_tool_calls = bool(message["tool_calls"])
        if not has_tool_calls:
            del message["tool_calls"]
        if not message["content"]:
            message["content"] = None

        # OpenAI requires `finish_reason: "tool_calls"` whenever the message
        # carries tool calls. This was hardcoded to "stop", which tells any
        # client that drives its agent loop off `finish_reason` that the turn
        # is over, so the reconstructed tool calls were never executed.
        if has_tool_calls:
            finish_reason = "tool_calls"
        elif finish_reason is None:
            finish_reason = "stop"

        response: dict[str, Any] = {
            "object": "chat.completion",
            **envelope,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        }
        if usage is not None:
            response["usage"] = usage
        return response

    def _openai_response_to_chunks(self, response: dict[str, Any]) -> list[bytes]:
        """Split a non-streaming ``chat.completion`` body into SSE chunk frames.

        A streaming client reads ``choices[].delta``, not ``choices[].message``.
        Serialising the reconstructed non-streaming body into a single SSE frame
        produced a stream in which both the text and the tool calls were
        invisible to the client.
        """
        choices = response.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else {}
        if not isinstance(choice, dict):
            choice = {}
        message = choice.get("message")
        if not isinstance(message, dict):
            message = {}
        finish_reason = choice.get("finish_reason") or "stop"

        base: dict[str, Any] = {"object": "chat.completion.chunk"}
        for key in ("id", "created", "model", "system_fingerprint"):
            if response.get(key) is not None:
                base[key] = response[key]

        def frame(delta: dict[str, Any], reason: str | None) -> bytes:
            payload = {
                **base,
                "choices": [{"index": 0, "delta": delta, "finish_reason": reason}],
            }
            return f"data: {json.dumps(payload)}\n\n".encode()

        frames = [frame({"role": message.get("role") or "assistant"}, None)]

        content = message.get("content")
        if content:
            frames.append(frame({"content": content}, None))

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for index, tool_call in enumerate(tool_calls):
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function")
                if not isinstance(function, dict):
                    function = {}
                frames.append(
                    frame(
                        {
                            "tool_calls": [
                                {
                                    "index": index,
                                    "id": tool_call.get("id", ""),
                                    "type": tool_call.get("type", "function"),
                                    "function": {
                                        "name": function.get("name", ""),
                                        "arguments": function.get("arguments", ""),
                                    },
                                }
                            ]
                        },
                        None,
                    )
                )

        frames.append(frame({}, finish_reason))
        return frames

    async def _response_to_sse(
        self,
        response: dict[str, Any],
    ) -> Any:  # AsyncGenerator[bytes, None]
        """Convert a response back to SSE format for streaming.

        This is a simplified version - in practice you might want
        to chunk the response more granularly.
        """
        if self.provider == "anthropic":
            from headroom.proxy.handlers.streaming import StreamingMixin

            for chunk in StreamingMixin()._response_to_sse(response, "anthropic"):
                yield chunk
        else:
            # OpenAI SSE format: `chat.completion.chunk` frames, then [DONE].
            for chunk in self._openai_response_to_chunks(response):
                yield chunk
            yield b"data: [DONE]\n\n"
