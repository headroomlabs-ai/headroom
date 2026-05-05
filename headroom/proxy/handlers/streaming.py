"""Streaming handler mixin for HeadroomProxy.

Contains SSE parsing, streaming response generation, and related utilities.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

from headroom.proxy.helpers import compute_turn_id, jitter_delay_ms

if TYPE_CHECKING:
    from fastapi.responses import Response, StreamingResponse


import httpx

from headroom.copilot_auth import apply_copilot_api_auth

logger = logging.getLogger("headroom.proxy")


class StreamingMixin:
    """Mixin providing streaming response methods for HeadroomProxy."""

    @staticmethod
    def _extract_anthropic_cache_ttl_metrics(usage: dict[str, Any] | None) -> tuple[int, int]:
        """Extract observed Anthropic cache-write TTL bucket usage."""
        if not isinstance(usage, dict):
            return (0, 0)
        cache_creation = usage.get("cache_creation")
        if not isinstance(cache_creation, dict):
            return (0, 0)
        return (
            int(cache_creation.get("ephemeral_5m_input_tokens", 0) or 0),
            int(cache_creation.get("ephemeral_1h_input_tokens", 0) or 0),
        )

    def _parse_sse_usage(self, chunk: bytes, provider: str) -> dict[str, int] | None:
        """Parse usage information from SSE chunk.

        For Anthropic: Looks for message_start (input tokens) and message_delta (output tokens)
        For OpenAI: Looks for final chunk with usage object (requires stream_options.include_usage=true)
        For Gemini: Looks for usageMetadata in each chunk

        Returns dict with keys: input_tokens, output_tokens, cache_read_input_tokens,
        cache_creation_input_tokens, cache_creation_ephemeral_5m_input_tokens,
        cache_creation_ephemeral_1h_input_tokens
        Returns None if no usage found in this chunk.

        PR-A8 / P1-8: Decoded via the bytes-buffer SSE splitter so multi-byte
        characters split across TCP reads do not corrupt downstream parsing.
        Only complete events (terminated by ``\\n\\n``) are decoded; partial
        bytes are dropped (this method is single-chunk only — the buffered
        path is in ``_parse_sse_usage_from_buffer``).
        """
        from headroom.proxy.helpers import parse_sse_events_from_byte_buffer

        try:
            buf = bytearray(chunk)
            events = parse_sse_events_from_byte_buffer(buf)
            for _event_name, data_str in events:
                if not data_str or data_str == "[DONE]":
                    continue

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                usage = {}

                if provider == "anthropic":
                    # Anthropic sends message_start with input tokens
                    # and message_delta with output tokens
                    event_type = data.get("type", "")

                    if event_type == "message_start":
                        msg = data.get("message", {})
                        msg_usage = msg.get("usage", {})
                        if msg_usage:
                            usage["input_tokens"] = msg_usage.get("input_tokens", 0)
                            usage["cache_read_input_tokens"] = msg_usage.get(
                                "cache_read_input_tokens", 0
                            )
                            usage["cache_creation_input_tokens"] = msg_usage.get(
                                "cache_creation_input_tokens", 0
                            )
                            cache_write_5m, cache_write_1h = (
                                self._extract_anthropic_cache_ttl_metrics(msg_usage)
                            )
                            usage["cache_creation_ephemeral_5m_input_tokens"] = cache_write_5m
                            usage["cache_creation_ephemeral_1h_input_tokens"] = cache_write_1h

                    elif event_type == "message_delta":
                        delta_usage = data.get("usage", {})
                        if delta_usage:
                            usage["output_tokens"] = delta_usage.get("output_tokens", 0)

                elif provider == "openai":
                    # OpenAI sends usage in final chunk (when stream_options.include_usage=true)
                    chunk_usage = data.get("usage")
                    if chunk_usage:
                        usage["input_tokens"] = chunk_usage.get("prompt_tokens", 0)
                        usage["output_tokens"] = chunk_usage.get("completion_tokens", 0)
                        # OpenAI has cached tokens in prompt_tokens_details
                        details = chunk_usage.get("prompt_tokens_details") or {}
                        usage["cache_read_input_tokens"] = details.get("cached_tokens", 0)

                elif provider == "gemini":
                    # Gemini sends usageMetadata in each streaming chunk
                    # Format: {"usageMetadata": {"promptTokenCount": N, "candidatesTokenCount": M}}
                    usage_meta = data.get("usageMetadata")
                    if usage_meta:
                        usage["input_tokens"] = usage_meta.get("promptTokenCount", 0)
                        usage["output_tokens"] = usage_meta.get("candidatesTokenCount", 0)
                        # Gemini also has cachedContentTokenCount for context caching
                        usage["cache_read_input_tokens"] = usage_meta.get(
                            "cachedContentTokenCount", 0
                        )

                if usage:
                    return usage

        except (UnicodeDecodeError, KeyError, TypeError) as e:
            # Don't fail streaming on parse errors
            logger.debug(f"SSE usage parsing error for {provider}: {e}")

        return None

    def _parse_sse_usage_from_buffer(
        self, stream_state: dict[str, Any], provider: str
    ) -> dict[str, int] | None:
        """Parse usage from buffered SSE data, handling split chunks.

        Processes complete SSE events (terminated by ``\\n\\n``) from the
        bytes buffer and removes them from the buffer. Incomplete events
        are kept in the buffer for the next chunk.

        PR-A8 / P1-8: ``stream_state["sse_buffer"]`` is a ``bytearray``
        (not ``str``); event boundaries are found in bytes so a multi-byte
        UTF-8 character split across TCP reads is preserved intact. Each
        complete event is decoded as UTF-8 only AFTER the boundary is
        located. Invalid UTF-8 in a *complete* event raises (operator-
        visible diagnostic, not silent corruption).
        """
        from headroom.proxy.helpers import parse_sse_events_from_byte_buffer

        buffer = stream_state["sse_buffer"]
        usage_found: dict[str, int] = {}

        # Process complete SSE events (separated by double newlines).
        # ``parse_sse_events_from_byte_buffer`` mutates ``buffer`` in
        # place, leaving partial-event tail bytes for the next chunk —
        # since ``buffer`` is the same ``bytearray`` object held by
        # ``stream_state``, no reassignment is needed.
        events = parse_sse_events_from_byte_buffer(buffer)
        for _event_name, data_str in events:
            if not data_str or data_str == "[DONE]":
                continue

            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            if provider == "anthropic":
                event_type = data.get("type", "")
                if event_type == "message_start":
                    msg = data.get("message", {})
                    msg_usage = msg.get("usage", {})
                    if msg_usage:
                        usage_found["input_tokens"] = msg_usage.get("input_tokens", 0)
                        usage_found["cache_read_input_tokens"] = msg_usage.get(
                            "cache_read_input_tokens", 0
                        )
                        usage_found["cache_creation_input_tokens"] = msg_usage.get(
                            "cache_creation_input_tokens", 0
                        )
                        cache_write_5m, cache_write_1h = self._extract_anthropic_cache_ttl_metrics(
                            msg_usage
                        )
                        usage_found["cache_creation_ephemeral_5m_input_tokens"] = cache_write_5m
                        usage_found["cache_creation_ephemeral_1h_input_tokens"] = cache_write_1h
                        logger.debug(
                            f"[CACHE] Anthropic usage: input={usage_found.get('input_tokens')}, "
                            f"cache_read={usage_found.get('cache_read_input_tokens')}, "
                            f"cache_write={usage_found.get('cache_creation_input_tokens')}"
                        )
                elif event_type == "message_delta":
                    delta_usage = data.get("usage", {})
                    if delta_usage:
                        usage_found["output_tokens"] = delta_usage.get("output_tokens", 0)

            elif provider == "openai":
                chunk_usage = data.get("usage")
                if chunk_usage:
                    usage_found["input_tokens"] = chunk_usage.get("prompt_tokens", 0)
                    usage_found["output_tokens"] = chunk_usage.get("completion_tokens", 0)
                    details = chunk_usage.get("prompt_tokens_details") or {}
                    usage_found["cache_read_input_tokens"] = details.get("cached_tokens", 0)

            elif provider == "gemini":
                usage_meta = data.get("usageMetadata")
                if usage_meta:
                    usage_found["input_tokens"] = usage_meta.get("promptTokenCount", 0)
                    usage_found["output_tokens"] = usage_meta.get("candidatesTokenCount", 0)
                    usage_found["cache_read_input_tokens"] = usage_meta.get(
                        "cachedContentTokenCount", 0
                    )

        return usage_found if usage_found else None

    def _parse_sse_to_response(self, sse_data: str, provider: str) -> dict[str, Any] | None:
        """Parse SSE data to reconstruct the API response JSON.

        Args:
            sse_data: Raw SSE data string. Must already be UTF-8 decoded
                from a complete-events bytes buffer (see
                ``parse_sse_events_from_byte_buffer``).
            provider: Provider type for parsing.

        Returns:
            Reconstructed response dict or None if parsing fails.

        PR-A8 / P1-9: handles all Anthropic delta types per guide §5.1:
        ``text_delta``, ``input_json_delta``, ``thinking_delta``,
        ``signature_delta``, ``citations_delta``. Also preserves
        ``redacted_thinking.data`` and accumulates citations as a list.
        """
        if provider != "anthropic":
            return None  # Only implemented for Anthropic

        response: dict[str, Any] = {"content": [], "usage": {}}
        # Track blocks by their `index` field so out-of-order events
        # don't corrupt the reconstruction. The current block pointer
        # remains for backward-compat with code that walks this dict
        # sequentially, but the index map is the source of truth.
        blocks_by_index: dict[int, dict[str, Any]] = {}
        current_block: dict[str, Any] | None = None

        for line in sse_data.split("\n"):
            if not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if not data_str or data_str == "[DONE]":
                continue

            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            event_type = data.get("type", "")

            if event_type == "message_start":
                msg = data.get("message", {})
                response["id"] = msg.get("id")
                response["model"] = msg.get("model")
                response["role"] = msg.get("role", "assistant")
                response["stop_reason"] = msg.get("stop_reason")
                if msg.get("usage"):
                    response["usage"].update(msg["usage"])

            elif event_type == "content_block_start":
                block = data.get("content_block", {})
                block_index = data.get("index", len(response["content"]))
                btype = block.get("type")
                current_block = {
                    "type": btype,
                    "index": block_index,
                }
                if btype == "text":
                    current_block["text"] = block.get("text", "")
                elif btype == "tool_use":
                    current_block["id"] = block.get("id")
                    current_block["name"] = block.get("name")
                    current_block["input"] = {}
                elif btype == "thinking":
                    # Thinking block — accumulate text via
                    # `thinking_delta`; signature arrives via
                    # `signature_delta` (single value, not accumulated).
                    current_block["thinking_buffer"] = block.get("thinking", "")
                    if "signature" in block:
                        current_block["signature"] = block["signature"]
                elif btype == "redacted_thinking":
                    # Per Anthropic spec §2.7: opaque encrypted reasoning
                    # block. The `data` field is preserved as-is and
                    # MUST be replayed unchanged on the next turn for
                    # signature validation to pass.
                    if "data" in block:
                        current_block["data"] = block["data"]
                blocks_by_index[block_index] = current_block

            elif event_type == "content_block_delta":
                # Resolve the target block by index (preferred) or fall
                # back to current_block for legacy linear streams.
                idx = data.get("index")
                target = (blocks_by_index.get(idx) if idx is not None else None) or current_block
                if target is not None:
                    delta = data.get("delta", {})
                    dtype = delta.get("type")
                    if dtype == "text_delta":
                        target["text"] = target.get("text", "") + delta.get("text", "")
                    elif dtype == "input_json_delta":
                        # Accumulate partial JSON for tool input.
                        partial = delta.get("partial_json", "")
                        target["_partial_json"] = target.get("_partial_json", "") + partial
                    elif dtype == "thinking_delta":
                        # Accumulate thinking text into the dedicated
                        # buffer so it never collides with `text` on
                        # text blocks (separate field per guide §2.7).
                        target["thinking_buffer"] = target.get("thinking_buffer", "") + delta.get(
                            "thinking", ""
                        )
                    elif dtype == "signature_delta":
                        # Single value, not accumulated. Last-write
                        # wins per Anthropic spec.
                        if "signature" in delta:
                            target["signature"] = delta["signature"]
                    elif dtype == "citations_delta":
                        # Append the citation object to the citations
                        # list so multi-citation blocks reconstruct
                        # correctly. Per guide §2.5: each delta carries
                        # one full citation object under `citation`.
                        citations = target.setdefault("citations", [])
                        citation = delta.get("citation")
                        if citation is not None:
                            citations.append(citation)

            elif event_type == "content_block_stop":
                idx = data.get("index")
                target = (blocks_by_index.get(idx) if idx is not None else None) or current_block
                if target is not None:
                    # Parse accumulated JSON for tool_use blocks.
                    if target.get("type") == "tool_use" and "_partial_json" in target:
                        try:
                            target["input"] = json.loads(target["_partial_json"])
                        except json.JSONDecodeError:
                            target["input"] = {}
                        del target["_partial_json"]
                    # Materialize the thinking buffer into the
                    # canonical `thinking` field expected by the
                    # Anthropic API.
                    if target.get("type") == "thinking" and "thinking_buffer" in target:
                        target["thinking"] = target.pop("thinking_buffer")
                    # Append the block exactly once. `current_block`
                    # may not match the indexed target if the stream
                    # interleaved multiple blocks; index-keyed map is
                    # authoritative.
                    if target not in response["content"]:
                        response["content"].append(target)
                    current_block = None

            elif event_type == "message_delta":
                delta = data.get("delta", {})
                if delta.get("stop_reason"):
                    response["stop_reason"] = delta["stop_reason"]
                if data.get("usage"):
                    response["usage"].update(data["usage"])

        return response if response.get("content") else None

    def _response_to_sse(self, response: dict[str, Any], provider: str) -> list[bytes]:
        """Convert a response dict back to SSE format.

        Args:
            response: API response dict.
            provider: Provider type for formatting.

        Returns:
            List of SSE event bytes.
        """
        if provider != "anthropic":
            return []

        events: list[bytes] = []

        # message_start
        msg_start = {
            "type": "message_start",
            "message": {
                "id": response.get("id", "msg_generated"),
                "type": "message",
                "role": response.get("role", "assistant"),
                "model": response.get("model", "unknown"),
                "content": [],
                "stop_reason": None,
                "usage": response.get("usage", {}),
            },
        }
        events.append(f"event: message_start\ndata: {json.dumps(msg_start)}\n\n".encode())

        # Content blocks
        for idx, block in enumerate(response.get("content", [])):
            # content_block_start
            if block.get("type") == "text":
                block_start = {
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": {"type": "text", "text": ""},
                }
            elif block.get("type") == "tool_use":
                block_start = {
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": {
                        "type": "tool_use",
                        "id": block.get("id", f"toolu_{idx}"),
                        "name": block.get("name", ""),
                        "input": {},
                    },
                }
            else:
                continue

            events.append(
                f"event: content_block_start\ndata: {json.dumps(block_start)}\n\n".encode()
            )

            # content_block_delta(s)
            if block.get("type") == "text" and block.get("text"):
                delta = {
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {"type": "text_delta", "text": block["text"]},
                }
                events.append(f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n".encode())
            elif block.get("type") == "tool_use" and block.get("input"):
                delta = {
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(block["input"]),
                    },
                }
                events.append(f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n".encode())

            # content_block_stop
            block_stop = {"type": "content_block_stop", "index": idx}
            events.append(f"event: content_block_stop\ndata: {json.dumps(block_stop)}\n\n".encode())

        # message_delta
        msg_delta = {
            "type": "message_delta",
            "delta": {"stop_reason": response.get("stop_reason", "end_turn")},
            "usage": {"output_tokens": response.get("usage", {}).get("output_tokens", 0)},
        }
        events.append(f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n".encode())

        # message_stop
        events.append(b'event: message_stop\ndata: {"type": "message_stop"}\n\n')

        return events

    def _record_ccr_feedback_from_response(
        self, response: dict, provider: str, request_id: str
    ) -> None:
        """Extract headroom_retrieve tool calls from a response and record feedback.

        This closes the TOIN feedback loop for streaming responses where
        the proxy can't intercept and handle retrieval calls inline.
        """
        from headroom.cache.compression_store import get_compression_store

        content = response.get("content", [])
        if not isinstance(content, list):
            return

        store = get_compression_store()

        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_use":
                continue
            if block.get("name") != "headroom_retrieve":
                continue

            input_data = block.get("input", {})
            hash_key = input_data.get("hash")
            query = input_data.get("query")

            if not hash_key:
                continue

            logger.info(
                f"[{request_id}] CCR Feedback: Recording retrieval "
                f"hash={hash_key[:8]}... query={query!r}"
            )

            # Call store.retrieve()/search() for the side effect of triggering
            # the feedback chain: _log_retrieval -> process_pending_feedback
            # -> toin.record_retrieval(). We discard the returned content.
            try:
                if query:
                    store.search(hash_key, query)
                else:
                    store.retrieve(hash_key, query=None)
            except Exception as e:
                logger.debug(f"[{request_id}] CCR Feedback recording failed: {e}")

    async def _finalize_stream_response(
        self,
        *,
        body: dict,
        provider: str,
        model: str,
        request_id: str,
        original_tokens: int,
        optimized_tokens: int,
        tokens_saved: int,
        transforms_applied: list[str],
        optimization_latency: float,
        stream_state: dict[str, Any],
        start_time: float,
        tags: dict[str, str] | None = None,
        pipeline_timing: dict[str, float] | None = None,
        prefix_tracker: Any | None = None,
        original_messages: list[dict] | None = None,
        full_sse_data: str = "",
        parsed_response: dict[str, Any] | None = None,
    ) -> None:
        from headroom.proxy.cost import _summarize_transforms

        total_latency = (time.time() - start_time) * 1000
        output_tokens = stream_state["output_tokens"]
        if output_tokens is None:
            output_tokens = stream_state["total_bytes"] // 40
            logger.warning(
                f"[{request_id}] Could not parse output_tokens from SSE, "
                f"estimating {output_tokens} from {stream_state['total_bytes']} bytes"
            )

        cache_read_tokens = stream_state["cache_read_input_tokens"] or 0
        cache_write_tokens = stream_state["cache_creation_input_tokens"] or 0
        cache_write_5m_tokens = stream_state["cache_creation_ephemeral_5m_input_tokens"] or 0
        cache_write_1h_tokens = stream_state["cache_creation_ephemeral_1h_input_tokens"] or 0
        uncached_input_tokens = max(optimized_tokens - cache_read_tokens - cache_write_tokens, 0)

        num_msgs = len(body.get("messages", []))
        cache_hit_pct = (
            round(cache_read_tokens / (cache_read_tokens + cache_write_tokens) * 100)
            if (cache_read_tokens + cache_write_tokens) > 0
            else 0
        )
        logger.info(
            f"[{request_id}] PERF "
            f"model={model} msgs={num_msgs} "
            f"tok_before={original_tokens} tok_after={optimized_tokens} "
            f"tok_saved={tokens_saved} "
            f"cache_read={cache_read_tokens} cache_write={cache_write_tokens} "
            f"cache_hit_pct={cache_hit_pct} "
            f"opt_ms={optimization_latency:.0f} "
            f"transforms={_summarize_transforms(transforms_applied)}"
        )

        if prefix_tracker is not None:
            import copy as _copy

            forwarded_messages = body.get("messages", [])
            next_forwarded = _copy.deepcopy(forwarded_messages)
            next_original = _copy.deepcopy(original_messages or forwarded_messages)

            if full_sse_data and provider == "anthropic":
                _parsed = (
                    parsed_response
                    if parsed_response is not None
                    else self._parse_sse_to_response(full_sse_data, provider)
                )
                if _parsed:
                    asst_msg = self._assistant_message_from_response_json(_parsed)
                    if asst_msg is not None:
                        next_forwarded.append(_copy.deepcopy(asst_msg))
                        next_original.append(_copy.deepcopy(asst_msg))

            prefix_tracker.update_from_response(
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
                messages=next_forwarded,
                original_messages=next_original,
            )

        if self.cost_tracker:
            self.cost_tracker.record_tokens(
                model,
                tokens_saved,
                optimized_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
                cache_write_5m_tokens=cache_write_5m_tokens,
                cache_write_1h_tokens=cache_write_1h_tokens,
                uncached_tokens=uncached_input_tokens,
            )

        if getattr(self, "metrics", None) is not None:
            await self.metrics.record_request(
                provider=provider,
                model=model,
                input_tokens=optimized_tokens,
                output_tokens=output_tokens,
                tokens_saved=tokens_saved,
                latency_ms=total_latency,
                overhead_ms=optimization_latency,
                ttfb_ms=stream_state["ttfb_ms"] or total_latency,
                pipeline_timing=pipeline_timing,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
                cache_write_5m_tokens=cache_write_5m_tokens,
                cache_write_1h_tokens=cache_write_1h_tokens,
                uncached_input_tokens=uncached_input_tokens,
            )

        # Log the request to the in-memory request logger so it shows up in
        # /stats `recent_requests` and `/transformations/feed`. Without this
        # the streaming Anthropic path (which is what Claude Code uses) is
        # invisible to both surfaces — only Bedrock streaming and the
        # non-streaming Anthropic path were logged previously.
        if getattr(self, "logger", None) is not None:
            from headroom.proxy.models import RequestLog

            self.logger.log(
                RequestLog(
                    request_id=request_id,
                    timestamp=datetime.now().isoformat(),
                    provider=provider,
                    model=model,
                    input_tokens_original=original_tokens,
                    input_tokens_optimized=optimized_tokens,
                    output_tokens=output_tokens,
                    tokens_saved=tokens_saved,
                    savings_percent=(tokens_saved / original_tokens * 100)
                    if original_tokens > 0
                    else 0,
                    optimization_latency_ms=optimization_latency,
                    total_latency_ms=total_latency,
                    tags=tags or {},
                    cache_hit=False,
                    transforms_applied=transforms_applied,
                    request_messages=body.get("messages")
                    if getattr(self.config, "log_full_messages", False)
                    else None,
                )
            )

    async def _stream_response(
        self,
        url: str,
        headers: dict,
        body: dict,
        provider: str,
        model: str,
        request_id: str,
        original_tokens: int,
        optimized_tokens: int,
        tokens_saved: int,
        transforms_applied: list[str],
        tags: dict[str, str],
        optimization_latency: float,
        memory_user_id: str | None = None,
        pipeline_timing: dict[str, float] | None = None,
        prefix_tracker: Any | None = None,
        original_messages: list[dict] | None = None,
        *,
        original_body_bytes: bytes | None = None,
        body_mutated: bool = True,
        mutation_reasons: list[str] | None = None,
    ) -> Response | StreamingResponse:
        """Stream response with metrics tracking and memory tool handling.

        Parses SSE events to extract actual usage information from the API response
        for accurate token counting and cost calculation.

        When memory is enabled (memory_user_id provided), this method:
        1. Buffers the response to detect memory tool calls
        2. Executes memory tools if found
        3. Makes continuation requests until no memory tools remain
        4. Streams the final response to the client
        """
        from fastapi.responses import Response, StreamingResponse

        from headroom.proxy.helpers import MAX_SSE_BUFFER_SIZE

        headers = await apply_copilot_api_auth(headers, url=url)
        start_time = time.time()

        # Byte-faithful forwarding (PR-A3, fixes P0-2). Resolve outbound
        # bytes once before entering the connection-retry loop. When a
        # transform mutated the body we re-serialize canonically; otherwise
        # we forward the original client bytes verbatim.
        from headroom.proxy.helpers import (
            log_outbound_request,
            prepare_outbound_body_bytes,
        )

        outbound_bytes, outbound_source = prepare_outbound_body_bytes(
            body=body,
            original_body_bytes=original_body_bytes,
            body_mutated=body_mutated,
        )
        outbound_headers = {**headers, "content-type": "application/json"}
        log_outbound_request(
            forwarder="streaming",
            method="POST",
            path=url,
            body_bytes_count=len(outbound_bytes),
            body_mutated=body_mutated,
            mutation_reasons=list(mutation_reasons or []),
            request_id=request_id,
            source=outbound_source,
        )

        # Mutable state for the generator to update
        stream_state: dict[str, Any] = {
            "input_tokens": None,
            "output_tokens": None,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_creation_ephemeral_5m_input_tokens": 0,
            "cache_creation_ephemeral_1h_input_tokens": 0,
            "total_bytes": 0,
            # Buffer for incomplete SSE events (bytes, per PR-A8 / P1-8).
            # We split events on the ``\n\n`` byte sequence and decode
            # each complete event as UTF-8 only after the boundary is
            # found, so multi-byte characters split across TCP reads do
            # not corrupt downstream parsing.
            "sse_buffer": bytearray(),
            "ttfb_ms": None,  # Time to first byte from upstream
        }

        # Track if we need to handle memory tools
        memory_enabled = (
            memory_user_id is not None
            and self.memory_handler is not None
            and provider == "anthropic"
        )

        # Open connection before generator to capture upstream response headers
        # (needed to forward ratelimit headers to the client via StreamingResponse)
        assert self.http_client is not None, "http_client must be initialized before streaming"
        try:
            retry_attempts = max(1, getattr(self.config, "retry_max_attempts", 3))
            upstream_response = None
            last_connect_error = None

            for attempt in range(retry_attempts):
                try:
                    _upstream_req = self.http_client.build_request(
                        "POST", url, content=outbound_bytes, headers=outbound_headers
                    )
                    upstream_response = await self.http_client.send(_upstream_req, stream=True)
                    break
                except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as e:
                    last_connect_error = e
                    if attempt >= retry_attempts - 1:
                        raise

                    delay_with_jitter = jitter_delay_ms(
                        self.config.retry_base_delay_ms,
                        self.config.retry_max_delay_ms,
                        attempt,
                    )
                    logger.warning(
                        f"[{request_id}] Connection error to upstream API "
                        f"(attempt {attempt + 1}/{retry_attempts}): {e!r}; "
                        f"retrying in {delay_with_jitter:.0f}ms"
                    )
                    await asyncio.sleep(delay_with_jitter / 1000)

            if upstream_response is None:
                raise last_connect_error or RuntimeError("upstream connection did not start")
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as e:
            error_msg = str(e) or repr(e)
            logger.error(f"[{request_id}] Connection error to upstream API: {error_msg}")

            async def _error_gen():
                error_event = {
                    "type": "error",
                    "error": {
                        "type": "connection_error",
                        "message": f"Failed to connect to upstream API: {error_msg}",
                    },
                }
                yield f"event: error\ndata: {json.dumps(error_event)}\n\n".encode()

            return StreamingResponse(_error_gen(), media_type="text/event-stream")

        if upstream_response.status_code >= 400:
            logger.warning(
                "[%s] Forwarding upstream streaming error status=%s url=%s",
                request_id,
                upstream_response.status_code,
                url,
            )
            response_headers = dict(upstream_response.headers)
            response_headers.pop("content-length", None)
            response_headers.pop("transfer-encoding", None)
            response_headers.pop("connection", None)
            response_headers.pop("content-encoding", None)

            try:
                error_content = await upstream_response.aread()
            except Exception as read_error:
                logger.warning(
                    "[%s] Failed reading upstream error body status=%s url=%s error=%s",
                    request_id,
                    upstream_response.status_code,
                    url,
                    read_error,
                )
                error_content = json.dumps(
                    {
                        "error": {
                            "message": "Failed to read upstream error response body",
                            "details": str(read_error),
                        }
                    }
                ).encode("utf-8")
                response_headers["content-type"] = "application/json"
            finally:
                await upstream_response.aclose()

            stream_state["total_bytes"] = len(error_content)
            await self._finalize_stream_response(
                body=body,
                provider=provider,
                model=model,
                request_id=request_id,
                original_tokens=original_tokens,
                optimized_tokens=optimized_tokens,
                tokens_saved=tokens_saved,
                transforms_applied=transforms_applied,
                optimization_latency=optimization_latency,
                stream_state=stream_state,
                start_time=start_time,
                tags=tags,
                pipeline_timing=pipeline_timing,
                prefix_tracker=prefix_tracker,
                original_messages=original_messages,
            )
            return Response(
                content=error_content,
                status_code=upstream_response.status_code,
                headers=response_headers,
            )

        # Forward upstream ratelimit headers to the client
        forwarded_headers = {
            k: v for k, v in upstream_response.headers.items() if "ratelimit" in k.lower()
        }

        async def generate():
            nonlocal body, memory_enabled  # May need to modify for continuation requests

            # For memory mode, we buffer the response to check for tool calls
            buffered_chunks: list[bytes] = []
            # Bytes-level mirror of the SSE stream for memory/prefix
            # tracking. PR-A8 / P1-8: keep this as bytes too — we
            # decode only after a complete `\n\n`-terminated event has
            # been collected, so split UTF-8 bytes never produce
            # corrupted strings.
            full_sse_bytes = bytearray()
            parsed_response = None  # Set by memory block; used by CCR + prefix tracker

            try:
                async with contextlib.aclosing(upstream_response) as response:
                    async for chunk in response.aiter_bytes():
                        # Record TTFB on first chunk
                        if stream_state["ttfb_ms"] is None:
                            stream_state["ttfb_ms"] = (time.time() - start_time) * 1000

                        stream_state["total_bytes"] += len(chunk)

                        # PR-A8 / P1-8: append bytes verbatim. The
                        # buffer is a ``bytearray`` and event boundaries
                        # are located in bytes; decoding happens per
                        # complete event in the SSE splitter helper.
                        stream_state["sse_buffer"].extend(chunk)

                        # Safety: prevent unbounded buffer growth.
                        if len(stream_state["sse_buffer"]) > MAX_SSE_BUFFER_SIZE:
                            logger.error(
                                "SSE buffer exceeded maximum size (%d bytes), "
                                "truncating to prevent memory exhaustion",
                                MAX_SSE_BUFFER_SIZE,
                            )
                            # Keep the most recent half so an in-flight
                            # event is more likely to survive.
                            tail = bytes(stream_state["sse_buffer"][-MAX_SSE_BUFFER_SIZE // 2 :])
                            stream_state["sse_buffer"] = bytearray(tail)

                        # Always stream immediately — buffering breaks
                        # real-time clients (LangGraph, LangChain, etc.)
                        yield chunk

                        # Buffer SSE data for memory processing and/or prefix tracker
                        _track_sse = memory_enabled or (
                            prefix_tracker is not None and provider == "anthropic"
                        )
                        if _track_sse:
                            if memory_enabled:
                                buffered_chunks.append(chunk)
                            full_sse_bytes.extend(chunk)
                            if len(full_sse_bytes) > MAX_SSE_BUFFER_SIZE:
                                logger.warning(
                                    "Memory-mode SSE buffer exceeded maximum size, "
                                    "disabling memory detection for this request"
                                )
                                memory_enabled = False

                        # Parse complete SSE events from buffer
                        usage = self._parse_sse_usage_from_buffer(stream_state, provider)
                        if usage:
                            if "input_tokens" in usage:
                                stream_state["input_tokens"] = usage["input_tokens"]
                            if "output_tokens" in usage:
                                stream_state["output_tokens"] = usage["output_tokens"]
                            if "cache_read_input_tokens" in usage:
                                stream_state["cache_read_input_tokens"] = usage[
                                    "cache_read_input_tokens"
                                ]
                            if "cache_creation_input_tokens" in usage:
                                stream_state["cache_creation_input_tokens"] = usage[
                                    "cache_creation_input_tokens"
                                ]
                            if "cache_creation_ephemeral_5m_input_tokens" in usage:
                                stream_state["cache_creation_ephemeral_5m_input_tokens"] = usage[
                                    "cache_creation_ephemeral_5m_input_tokens"
                                ]
                            if "cache_creation_ephemeral_1h_input_tokens" in usage:
                                stream_state["cache_creation_ephemeral_1h_input_tokens"] = usage[
                                    "cache_creation_ephemeral_1h_input_tokens"
                                ]

                # Memory tool handling after stream completes
                # Chunks were already yielded in real-time above, so we only
                # do silent background processing here — no yielding.
                #
                # PR-A8 / P1-8: full_sse_bytes accumulated as bytes; we
                # decode here in one shot now that the stream is
                # complete (the entire payload is a closed sequence of
                # complete events). Invalid UTF-8 at this point would
                # be an upstream protocol violation — surface loudly.
                full_sse_data: str = full_sse_bytes.decode("utf-8") if full_sse_bytes else ""

                if memory_enabled and full_sse_data:
                    # Check for Claude Code credential error
                    if "only authorized for use with Claude Code" in full_sse_data:
                        logger.warning(
                            f"[{request_id}] Memory: Claude Code subscription credentials "
                            "do not support custom tool injection. Set ANTHROPIC_API_KEY "
                            "environment variable or use --no-memory-tools flag."
                        )
                        return

                    # Parse SSE to get response JSON
                    parsed_response = self._parse_sse_to_response(full_sse_data, provider)

                    if parsed_response and self.memory_handler.has_memory_tool_calls(
                        parsed_response, provider
                    ):
                        logger.info(
                            f"[{request_id}] Memory: Detected tool calls in streaming response"
                        )

                        # Execute memory tool calls — response already streamed
                        # so results are saved but continuation is not possible
                        # in SSE streaming mode. The WS and non-streaming paths
                        # handle continuation properly.
                        tool_results = await self.memory_handler.handle_memory_tool_calls(
                            parsed_response, memory_user_id, provider
                        )
                        if tool_results:
                            logger.info(
                                f"[{request_id}] Memory: Tool calls executed "
                                f"({len(tool_results)} results saved, SSE streaming — "
                                "continuation handled by client)"
                            )

                # CCR Feedback: Record headroom_retrieve tool calls for TOIN learning.
                # In streaming mode, the client handles actual retrieval, but we
                # still need to record the event so TOIN learns which fields matter.
                if self.config.ccr_inject_tool and full_sse_data:
                    ccr_parsed = (
                        parsed_response
                        if parsed_response
                        else self._parse_sse_to_response(full_sse_data, provider)
                    )
                    if ccr_parsed:
                        self._record_ccr_feedback_from_response(ccr_parsed, provider, request_id)

            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as e:
                logger.error(f"[{request_id}] Connection error to upstream API: {e}")
                error_event = {
                    "type": "error",
                    "error": {
                        "type": "connection_error",
                        "message": f"Failed to connect to upstream API: {e}",
                    },
                }
                yield f"event: error\ndata: {json.dumps(error_event)}\n\n".encode()
            except httpx.HTTPStatusError as e:
                logger.error(f"[{request_id}] HTTP error from upstream API: {e}")
                # Forward the upstream error response
                yield e.response.content
            except Exception as e:
                logger.error(f"[{request_id}] Unexpected streaming error: {e}")
                error_event = {
                    "type": "error",
                    "error": {"type": "api_error", "message": str(e)},
                }
                yield f"event: error\ndata: {json.dumps(error_event)}\n\n".encode()
            finally:
                # PR-A8 / P1-8: best-effort decode for downstream
                # finalization. This runs in `finally` so it must not
                # raise — if the upstream sent invalid bytes mid-stream
                # we surface them as `errors="strict"` would in the
                # success path above, but here we accept the
                # diagnostic-grade fallback so the finalization log
                # line still emits.
                try:
                    _final_full_sse_data: str = (
                        full_sse_bytes.decode("utf-8") if full_sse_bytes else ""
                    )
                except UnicodeDecodeError:
                    logger.warning(
                        f"[{request_id}] Final SSE buffer contained invalid UTF-8; "
                        "downstream finalization will see only the well-formed prefix."
                    )
                    # Find the longest valid UTF-8 prefix via the
                    # incremental decoder; the lossy decoder kwargs
                    # are forbidden in this module per PR-A8 / P1-8.
                    decoder = __import__("codecs").getincrementaldecoder("utf-8")()
                    _final_full_sse_data = decoder.decode(bytes(full_sse_bytes), final=False)
                await self._finalize_stream_response(
                    body=body,
                    provider=provider,
                    model=model,
                    request_id=request_id,
                    original_tokens=original_tokens,
                    optimized_tokens=optimized_tokens,
                    tokens_saved=tokens_saved,
                    transforms_applied=transforms_applied,
                    optimization_latency=optimization_latency,
                    stream_state=stream_state,
                    start_time=start_time,
                    tags=tags,
                    pipeline_timing=pipeline_timing,
                    prefix_tracker=prefix_tracker,
                    original_messages=original_messages,
                    full_sse_data=_final_full_sse_data,
                    parsed_response=parsed_response,
                )

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers=forwarded_headers,
        )

    async def _stream_response_bedrock(
        self,
        body: dict,
        headers: dict,
        provider: str,
        model: str,
        request_id: str,
        original_tokens: int,
        optimized_tokens: int,
        tokens_saved: int,
        transforms_applied: list[str],
        tags: dict[str, str],
        optimization_latency: float,
        pipeline_timing: dict[str, float] | None = None,
    ) -> StreamingResponse:
        """Stream response from Bedrock backend with metrics tracking.

        Translates Bedrock streaming events to Anthropic SSE format.
        """
        from fastapi.responses import StreamingResponse

        from headroom.proxy.cost import _summarize_transforms
        from headroom.proxy.models import RequestLog

        start_time = time.time()

        # Mutable state for the generator
        stream_state: dict[str, Any] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "ttfb_ms": None,
        }

        async def generate():
            try:
                assert self.anthropic_backend is not None

                async for event in self.anthropic_backend.stream_message(body, headers):
                    # Record TTFB on first event
                    if stream_state["ttfb_ms"] is None:
                        stream_state["ttfb_ms"] = (time.time() - start_time) * 1000

                    # Format as SSE
                    if event.raw_sse:
                        yield event.raw_sse.encode()
                    else:
                        sse_line = f"event: {event.event_type}\ndata: {json.dumps(event.data)}\n\n"
                        yield sse_line.encode()

                    # Track usage from message_start event
                    if event.event_type == "message_start":
                        msg = event.data.get("message", {})
                        usage = msg.get("usage", {})
                        if "input_tokens" in usage:
                            stream_state["input_tokens"] = usage["input_tokens"]

                    # Track output tokens from message_delta
                    if event.event_type == "message_delta":
                        usage = event.data.get("usage", {})
                        if "output_tokens" in usage:
                            stream_state["output_tokens"] = usage["output_tokens"]

                    # Handle errors
                    if event.event_type == "error":
                        logger.error(f"[{request_id}] Bedrock stream error: {event.data}")

            except Exception as e:
                logger.error(f"[{request_id}] Bedrock streaming error: {e}")
                error_event = {
                    "type": "error",
                    "error": {"type": "api_error", "message": str(e)},
                }
                yield f"event: error\ndata: {json.dumps(error_event)}\n\n".encode()

            finally:
                # Record metrics
                total_latency = (time.time() - start_time) * 1000
                output_tokens = stream_state["output_tokens"]

                _backend_name = (
                    self.anthropic_backend.name if self.anthropic_backend else "anthropic"
                )
                await self.metrics.record_request(
                    provider=_backend_name,
                    model=model,
                    input_tokens=optimized_tokens,
                    output_tokens=output_tokens,
                    tokens_saved=tokens_saved,
                    latency_ms=total_latency,
                    cached=False,
                    overhead_ms=optimization_latency,
                    ttfb_ms=stream_state["ttfb_ms"] or 0,
                    pipeline_timing=pipeline_timing,
                )

                if self.cost_tracker:
                    self.cost_tracker.record_tokens(model, tokens_saved, optimized_tokens)

                # Log request
                if self.logger:
                    self.logger.log(
                        RequestLog(
                            request_id=request_id,
                            timestamp=datetime.now().isoformat(),
                            provider=_backend_name,
                            model=model,
                            input_tokens_original=original_tokens,
                            input_tokens_optimized=optimized_tokens,
                            output_tokens=output_tokens,
                            tokens_saved=tokens_saved,
                            savings_percent=(tokens_saved / original_tokens * 100)
                            if original_tokens > 0
                            else 0,
                            optimization_latency_ms=optimization_latency,
                            total_latency_ms=total_latency,
                            tags=tags,
                            cache_hit=False,
                            transforms_applied=transforms_applied,
                            request_messages=body.get("messages")
                            if self.config.log_full_messages
                            else None,
                            turn_id=compute_turn_id(
                                model, body.get("system"), body.get("messages")
                            ),
                        )
                    )

                # Structured perf log line for `headroom perf` analysis
                num_msgs = len(body.get("messages", []))
                logger.info(
                    f"[{request_id}] PERF "
                    f"model={model} msgs={num_msgs} "
                    f"tok_before={original_tokens} tok_after={optimized_tokens} "
                    f"tok_saved={tokens_saved} "
                    f"cache_read=0 cache_write=0 cache_hit_pct=0 "
                    f"opt_ms={optimization_latency:.0f} "
                    f"transforms={_summarize_transforms(transforms_applied)}"
                )

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
        )

    async def _stream_openai_via_backend(
        self,
        body: dict,
        headers: dict,
        model: str,
        request_id: str,
        start_time: float,
        original_tokens: int,
        optimized_tokens: int,
        tokens_saved: int,
        transforms_applied: list[str],
        tags: dict[str, str],
        optimization_latency: float,
        pipeline_timing: dict[str, float] | None = None,
    ) -> StreamingResponse:
        """Stream OpenAI chat completion response from backend.

        Routes stream:true requests through the backend's stream_openai_message(),
        yielding SSE events to the client.
        """
        from fastapi.responses import StreamingResponse

        assert self.anthropic_backend is not None

        async def generate():
            try:
                async for sse_chunk in self.anthropic_backend.stream_openai_message(body, headers):
                    yield sse_chunk.encode() if isinstance(sse_chunk, str) else sse_chunk
            except Exception as e:
                logger.error(f"[{request_id}] Backend streaming error: {e}")
                error_data = {
                    "error": {
                        "message": str(e),
                        "type": "api_error",
                        "code": "backend_error",
                    }
                }
                yield f"data: {json.dumps(error_data)}\n\n".encode()
                yield b"data: [DONE]\n\n"
            finally:
                total_latency = (time.time() - start_time) * 1000
                await self.metrics.record_request(
                    provider=self.anthropic_backend.name,
                    model=model,
                    input_tokens=optimized_tokens,
                    output_tokens=0,  # Unknown in streaming
                    tokens_saved=tokens_saved,
                    latency_ms=total_latency,
                    cached=False,
                    overhead_ms=optimization_latency,
                    pipeline_timing=pipeline_timing,
                )
                if tokens_saved > 0:
                    logger.info(
                        f"[{request_id}] {model}: {original_tokens:,} → {optimized_tokens:,} "
                        f"(saved {tokens_saved:,} tokens) via {self.anthropic_backend.name} [stream]"
                    )

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
        )
