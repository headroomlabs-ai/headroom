"""Streaming handler mixin for HeadroomProxy.

Contains SSE parsing, streaming response generation, and related utilities.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from headroom.proxy.auth_mode import classify_client, supports_mid_turn_coalescing
from headroom.proxy.handlers._debug_dump import write_upstream_error_dump
from headroom.proxy.helpers import (
    RETRYABLE_OVERLOAD_STATUSES,
    jitter_delay_ms,
    overload_retry_is_futile,
    retry_after_ms,
)
from headroom.proxy.provider_usage import (
    anthropic_cache_ttl_buckets,
    billed_input_for_provider,
    usage_int,
)
from headroom.proxy.token_counting import gemini_output_tokens

if TYPE_CHECKING:
    from fastapi.responses import Response, StreamingResponse


import httpx

from headroom.copilot_auth import apply_copilot_api_auth
from headroom.proxy import public_errors
from headroom.proxy.memory_tool_stream import (
    MemoryToolStreamFilter,
    MemoryToolStreamOverflowError,
)
from headroom.proxy.stream_output_tokens import estimate_output_tokens
from headroom.utils import format_exception_message

logger = logging.getLogger("headroom.proxy")
STREAM_FAILURE_STATUS = 502


def _stream_outcome_status(status_code: int, completed_normally: bool) -> int:
    """Keep post-start stream failures out of the success counter."""
    return status_code if completed_normally else STREAM_FAILURE_STATUS


def _gemini_response_data(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    response = data.get("response")
    return response if isinstance(response, dict) else data


def _gemini_finished_candidates(data: Any) -> set[int]:
    candidates = _gemini_response_data(data).get("candidates")
    finished: set[int] = set()
    if not isinstance(candidates, list):
        return finished
    for position, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            continue
        reason = candidate.get("finishReason")
        index = candidate.get("index", position)
        if (
            isinstance(reason, str)
            and reason.strip()
            and reason != "FINISH_REASON_UNSPECIFIED"
            and isinstance(index, int)
            and not isinstance(index, bool)
            and index >= 0
        ):
            finished.add(index)
    return finished


def _gemini_requested_candidates(body: dict[str, Any]) -> int:
    request = body.get("request")
    request = request if isinstance(request, dict) else body
    config = request.get("generationConfig") or request.get("generation_config")
    if not isinstance(config, dict):
        return 1
    count = config.get("candidateCount", config.get("candidate_count", 1))
    return count if isinstance(count, int) and not isinstance(count, bool) and count > 0 else 1


def _sse_event_outcome(event_name: str | None, data_str: str, dialect: str) -> tuple[bool, bool]:
    """Classify a complete protocol event, never text inside a model delta."""
    try:
        data = json.loads(data_str)
    except (TypeError, ValueError):
        data = None
    event_type = data.get("type") if isinstance(data, dict) else None
    if not isinstance(event_type, str):
        event_type = None
    failed = event_name in {"error", "response.failed"} or event_type in {
        "error",
        "response.failed",
    }
    if isinstance(data, dict) and data.get("error") is not None:
        failed = True
    if dialect == "gemini":
        response = _gemini_response_data(data)
        feedback = response.get("promptFeedback")
        block_reason = feedback.get("blockReason") if isinstance(feedback, dict) else None
        failed = (
            failed
            or response.get("error") is not None
            or (
                isinstance(block_reason, str)
                and bool(block_reason.strip())
                and block_reason != "BLOCK_REASON_UNSPECIFIED"
            )
        )
        terminal = bool(_gemini_finished_candidates(data))
    elif dialect == "anthropic":
        terminal = event_name == "message_stop" or event_type == "message_stop"
    elif dialect == "responses":
        # Incomplete is a normal terminal result (e.g. the output-token limit),
        # unlike response.failed. It still incurred a successful model request.
        terminal = event_name in {"response.completed", "response.incomplete"} or event_type in {
            "response.completed",
            "response.incomplete",
        }
    else:
        terminal = data_str.strip() == "[DONE]"
    return failed, terminal


_ROUND_USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


def _add_round_usage(stream_state: dict[str, Any], response: dict[str, Any] | None) -> None:
    """Add a server-side continuation round's usage to the turn's totals."""
    usage = response.get("usage") if isinstance(response, dict) else None
    if not isinstance(usage, dict):
        return
    for key in _ROUND_USAGE_KEYS:
        value = usage.get(key)
        if isinstance(value, int):
            stream_state[key] = (stream_state.get(key) or 0) + value


def _upstream_error_frame(error_bytes: bytes, request_id: str) -> bytes:
    """Render an upstream error response as an Anthropic SSE ``error`` event.

    An Anthropic error envelope is addressed to the caller and passes through
    as is; anything else becomes the generic public error.
    """
    try:
        payload = json.loads(error_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError):
        payload = None
    if not (
        isinstance(payload, dict)
        and payload.get("type") == "error"
        and isinstance(payload.get("error"), dict)
    ):
        payload = public_errors.anthropic_error_body(
            public_errors.INTERNAL_ERROR, request_id=request_id
        )
    return f"event: error\ndata: {json.dumps(payload)}\n\n".encode()


def _sse_dict(value: Any) -> dict[str, Any]:
    """Return an upstream SSE object, or an empty object for a wrong-shaped value."""
    return value if isinstance(value, dict) else {}


def _parse_completion_tokens_from_sse_chunk(chunk_bytes: bytes) -> int | None:
    """Extract `usage.completion_tokens` from a single SSE chunk if present.

    Returns the integer count when the chunk carries a usage frame (LiteLLM
    emits this only when the request included
    ``stream_options.include_usage=true``), or None when no usage data is
    present (the typical content-only chunk path) or when the chunk fails
    to parse. Used by the OpenAI-via-backend stream path to track
    completion tokens online instead of buffering the entire response.
    """
    try:
        decoded = chunk_bytes.decode("utf-8", errors="replace")
    except (UnicodeDecodeError, AttributeError):
        return None
    for line in decoded.split("\n"):
        line = line.strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        try:
            data = json.loads(line[6:])
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        chunk_usage = data.get("usage")
        if isinstance(chunk_usage, dict):
            return int(chunk_usage.get("completion_tokens", 0) or 0)
    return None


class StreamingMixin:
    """Mixin providing streaming response methods for HeadroomProxy."""

    _mid_turn_queues: dict[str, asyncio.Queue] = {}
    _active_streams: set[str] = set()

    @staticmethod
    def _get_session_key(body: dict, session_header: str | None = None) -> str:
        """Return session identity from an explicit header or a body-derived hash.

        The fallback is a coarse, self-contained hash (md5 of
        ``model:system[:500]``) used ONLY to key the mid-turn steering state
        below. It is NOT the same derivation as
        ``prefix_tracker.compute_session_id`` (which json-encodes the full
        leading system-text run) — do not key cross-subsystem state on the
        assumption that the two fallbacks agree.
        """
        if session_header:
            return session_header
        import hashlib

        system = body.get("system", "")
        if isinstance(system, list):
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    system = block.get("text", "")
                    break
            else:
                system = ""
        system_content = str(system)[:500]
        key = f"{body.get('model', '')}:{system_content}"
        return hashlib.md5(key.encode()).hexdigest()[:16]

    def _queue_mid_turn_message(self, session_key: str, body: dict) -> dict:
        """Queue a mid-turn message and return a 202 response."""
        if session_key not in self._mid_turn_queues:
            self._mid_turn_queues[session_key] = asyncio.Queue()
        self._mid_turn_queues[session_key].put_nowait(body)
        return {"status": 202, "event": "headroom_queued"}

    def _should_queue_mid_turn(self, session_key: str, explicit_session_header: str | None) -> bool:
        """Return True only when a follow-up should be queued as a mid-turn message.

        Mid-turn steering is a private Headroom protocol: a queued message is
        only ever drained back to the client via the custom
        ``headroom_pending_messages`` SSE event, which a standard Anthropic SDK
        does not understand. We therefore only engage it for clients that
        *opt in* by sending an explicit ``x-headroom-session-id`` header.

        Without that header the session identity falls back to
        ``md5(model + system[:500])`` (see ``_get_session_key`` /
        ``prefix_tracker.compute_session_id``). That fallback is intentionally
        coarse and cannot distinguish genuinely concurrent, independent streams
        that happen to share a model + system prompt (e.g. a main conversation
        plus its background/parallel requests). Queuing those as mid-turn
        messages wrongly returns a 202 + JSON body to a caller that issued a
        *streaming* request, whose stream parser then sees an empty (non-SSE)
        stream and fails. So only opt-in (header-bearing) callers get queued.
        """
        return bool(explicit_session_header) and session_key in self._active_streams

    def _cleanup_mid_turn_stream(
        self, session_key: str, *, drain_pending_messages: bool = False
    ) -> list[dict]:
        """Clear active mid-turn state, optionally returning queued messages."""
        self._active_streams.discard(session_key)
        queue = self._mid_turn_queues.pop(session_key, None)
        if not drain_pending_messages or queue is None or queue.empty():
            return []
        pending_messages: list[dict] = []
        while not queue.empty():
            pending_messages.append(queue.get_nowait())
        return pending_messages

    @staticmethod
    def _extract_anthropic_cache_ttl_metrics(usage: dict[str, Any] | None) -> tuple[int, int]:
        """Extract observed Anthropic cache-write TTL bucket usage."""
        return anthropic_cache_ttl_buckets(usage)

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
            failed, terminal = _sse_event_outcome(
                _event_name,
                data_str,
                stream_state.get(
                    "sse_dialect",
                    provider if provider in {"anthropic", "gemini"} else "chat",
                ),
            )
            stream_state["stream_failed"] = stream_state.get("stream_failed", False) or failed
            stream_state["stream_terminal"] = stream_state.get("stream_terminal", False) or terminal
            if not data_str or data_str == "[DONE]":
                continue

            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue

            if provider == "gemini":
                finished = stream_state.setdefault("gemini_finished_candidates", set())
                finished.update(_gemini_finished_candidates(data))
                expected = stream_state.get("gemini_candidate_count", 1)
                stream_state["stream_terminal"] = set(range(expected)).issubset(finished)

            if provider == "anthropic":
                event_type = data.get("type", "")
                if event_type == "message_start":
                    msg_usage = _sse_dict(_sse_dict(data.get("message")).get("usage"))
                    if msg_usage:
                        # Absent stays absent: the finalizer falls back to the
                        # tokenizer derivation only when input_tokens is None.
                        if "input_tokens" in msg_usage:
                            usage_found["input_tokens"] = msg_usage["input_tokens"]
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
                    delta_usage = _sse_dict(data.get("usage"))
                    if delta_usage:
                        usage_found["output_tokens"] = delta_usage.get("output_tokens", 0)

            elif provider == "openai":
                chunk_usage = data.get("usage")
                if not isinstance(chunk_usage, dict):
                    response = data.get("response")
                    if isinstance(response, dict):
                        chunk_usage = response.get("usage")
                if isinstance(chunk_usage, dict):
                    # Chat Completions streams report prompt/completion tokens.
                    # Responses streams report input/output tokens under
                    # response.usage on response.completed.
                    input_tokens = chunk_usage.get("prompt_tokens")
                    if input_tokens is None:
                        input_tokens = chunk_usage.get("input_tokens", 0)
                    output_tokens = chunk_usage.get("completion_tokens")
                    if output_tokens is None:
                        output_tokens = chunk_usage.get("output_tokens", 0)
                    usage_found["input_tokens"] = usage_int(input_tokens)
                    usage_found["output_tokens"] = usage_int(output_tokens)
                    details = (
                        chunk_usage.get("prompt_tokens_details")
                        or chunk_usage.get("input_tokens_details")
                        or {}
                    )
                    if isinstance(details, dict):
                        usage_found["cache_read_input_tokens"] = usage_int(
                            details.get("cached_tokens")
                        )

            elif provider == "gemini":
                usage_meta = _sse_dict(data.get("usageMetadata"))
                if usage_meta:
                    usage_found["input_tokens"] = usage_meta.get("promptTokenCount", 0)
                    usage_found["output_tokens"] = gemini_output_tokens(usage_meta)
                    usage_found["cache_read_input_tokens"] = usage_meta.get(
                        "cachedContentTokenCount", 0
                    )

        return usage_found if usage_found else None

    def _parse_sse_to_response(
        self,
        sse_data: str,
        provider: str,
        *,
        require_complete: bool = False,
    ) -> dict[str, Any] | None:
        """Parse SSE data to reconstruct the API response JSON.

        Args:
            sse_data: Raw SSE data string. Must already be UTF-8 decoded
                from a complete-events bytes buffer (see
                ``parse_sse_events_from_byte_buffer``).
            provider: Provider type for parsing.
            require_complete: Reject anything short of a whole, replayable
                message — see the strictness note below. Off by default so
                streaming callers keep the lenient reconstruction they were
                written against.

        Returns:
            Reconstructed response dict or None if parsing fails.

        PR-A8 / P1-9: handles all Anthropic delta types per guide §5.1:
        ``text_delta``, ``input_json_delta``, ``thinking_delta``,
        ``signature_delta``, ``citations_delta``. Also preserves
        ``redacted_thinking.data`` and accumulates citations as a list.

        Permissive mode answers with whatever blocks it managed to
        accumulate. That is right for a streaming caller salvaging a
        partial stream, and wrong for #3130, where the reconstruction is
        handed to the client *as* the turn: a truncated stream would become
        a successful — and silently short — message, and an ``error`` event
        would vanish behind an HTTP 200. ``require_complete`` demands
        ``message_start``, a terminal ``message_stop``, every opened block
        closed, no ``error`` event, and no delta type this reconstructor
        cannot replay; anything else returns None so the caller can fail
        loudly instead.
        """
        if provider == "anthropic":
            # Keep the complete wire envelope alongside the mutable message
            # view. Callers that need to render a changed response own this
            # envelope in their request-local flow; never attach it to this
            # shared handler instance.
            from headroom.proxy.anthropic_wire import AnthropicSSEEnvelope

            envelope = AnthropicSSEEnvelope.parse(sse_data.encode("utf-8"))
            if require_complete and not envelope.is_message_reconstructable():
                return None
            return (
                envelope.message
                if envelope.message.get("content") or envelope.saw_message_start
                else None
            )

    def _response_to_sse(self, response: dict[str, Any], provider: str) -> list[bytes]:
        """Convert a response dictionary back to provider SSE frames."""

        if provider != "anthropic":
            return []

        from headroom.proxy.anthropic_wire import render_anthropic_sse_response

        return render_anthropic_sse_response(response)

    async def _continue_memory_tool_stream(
        self,
        memory_filter: MemoryToolStreamFilter,
        response: dict[str, Any] | None,
        *,
        url: str,
        outbound_headers: dict[str, str],
        outbound_bytes: bytes,
        memory_user_id: str | None,
        memory_request_ctx: Any | None,
        server_memory_tool_names: frozenset[str],
        stream_state: dict[str, Any],
        request_id: str,
    ) -> AsyncIterator[bytes]:
        """Execute withheld memory tool calls and stream continuation rounds.

        ``memory_filter`` has forwarded one upstream round to the client minus
        the proxy-owned memory tool calls and the message tail. When the round
        stopped on memory tool calls alone, run them, send the results upstream
        as the next turn, and stream that round's blocks into the same client
        message. Otherwise end the client message. A round that also called a
        client tool still has its memory calls executed, but the turn goes back
        to the client, which cannot carry the memory results.
        """
        from headroom.proxy.helpers import MAX_SSE_BUFFER_SIZE

        try:
            base_body = json.loads(outbound_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError):
            base_body = None
        messages = base_body.get("messages") if isinstance(base_body, dict) else None
        if not isinstance(messages, list):
            messages = None
        headers = {k: v for k, v in outbound_headers.items() if k.lower() != "content-length"}

        def is_memory_call(block: Any) -> bool:
            return (
                isinstance(block, dict)
                and block.get("type") == "tool_use"
                and block.get("name") in server_memory_tool_names
            )

        rounds = 0
        # What the client received across rounds; the prefix tracker must see
        # this, not round one, because it is what the client sends back.
        visible_content: list[Any] = []
        # Usage of rounds whose message_delta the client never receives.
        prior_usage: dict[str, int] = {}
        while True:
            content: list[Any] = []
            if response is not None:
                round_content = response.get("content")
                content = round_content if isinstance(round_content, list) else []
                visible_content.extend(block for block in content if not is_memory_call(block))
                stream_state["memory_client_message"] = {
                    **response,
                    "content": list(visible_content),
                }

            if not memory_filter.hid_tool_calls:
                for frame in memory_filter.closing_frames(prior_usage=prior_usage):
                    yield frame
                return

            # Run the calls the filter recorded, so they are not lost when the
            # round could not be reconstructed. Only the proxy's own calls: a
            # memory-named tool the client declared is the client's to run.
            memory_calls = memory_filter.hidden_calls()
            tool_results: list[dict[str, Any]] = []
            if memory_calls and memory_user_id is not None and self.memory_handler is not None:
                tool_results = await self.memory_handler.handle_memory_tool_calls(
                    {"content": memory_calls},
                    memory_user_id,
                    "anthropic",
                    request_context=memory_request_ctx,
                )
            logger.info(
                f"[{request_id}] Memory: Executed {len(tool_results)}/"
                f"{len(memory_filter.hidden_tool_names)} proxy-handled tool call(s) "
                "withheld from the SSE stream"
            )

            replayable = response is not None and len(tool_results) == sum(
                1 for block in content if is_memory_call(block)
            )
            can_continue = (
                memory_filter.stop_reason == "tool_use"
                and not memory_filter.visible_tool_use
                and messages is not None
                and replayable
                and rounds < self._MEMORY_CONTINUATION_MAX_ROUNDS
            )
            if not can_continue:
                if memory_filter.visible_tool_use:
                    logger.info(
                        f"[{request_id}] Memory: Round also called a client tool; "
                        "returning the turn to the client"
                    )
                elif rounds >= self._MEMORY_CONTINUATION_MAX_ROUNDS:
                    logger.warning(
                        f"[{request_id}] Memory: Stopped after {rounds} continuation "
                        "round(s) that only called memory tools"
                    )
                elif not replayable:
                    logger.warning(
                        f"[{request_id}] Memory: Could not rebuild the round for a "
                        "continuation; ending the turn after running its memory calls"
                    )
                for frame in memory_filter.closing_frames(prior_usage=prior_usage):
                    yield frame
                return

            assert response is not None and messages is not None and isinstance(base_body, dict)
            round_usage = response.get("usage")
            if isinstance(round_usage, dict):
                for key in _ROUND_USAGE_KEYS:
                    if isinstance(value := round_usage.get(key), int):
                        prior_usage[key] = prior_usage.get(key, 0) + value
            messages = [
                *messages,
                {"role": "assistant", "content": content},
                {"role": "user", "content": tool_results},
            ]
            continuation_bytes = json.dumps({**base_body, "messages": messages}).encode("utf-8")
            assert self.http_client is not None
            continuation_request = self.http_client.build_request(
                "POST", url, content=continuation_bytes, headers=headers
            )
            continuation = await self.http_client.send(continuation_request, stream=True)
            rounds += 1
            async with contextlib.aclosing(continuation) as upstream:
                if upstream.status_code >= 400:
                    error_bytes = await upstream.aread()
                    logger.warning(
                        f"[{request_id}] Memory: Continuation round {rounds} failed "
                        f"with upstream status {upstream.status_code}"
                    )
                    yield _upstream_error_frame(error_bytes, request_id)
                    return
                memory_filter = MemoryToolStreamFilter(
                    server_memory_tool_names,
                    index_offset=memory_filter.next_index,
                    forward_message_start=False,
                )
                # Same cap as the first round's buffer; past it the round is
                # still streamed, just not rebuilt for a further continuation.
                round_bytes: bytearray | None = bytearray()
                async for chunk in upstream.aiter_bytes():
                    if round_bytes is not None:
                        round_bytes.extend(chunk)
                        if len(round_bytes) > MAX_SSE_BUFFER_SIZE:
                            logger.warning(
                                f"[{request_id}] Memory: Continuation round {rounds} "
                                "exceeded the SSE buffer cap"
                            )
                            round_bytes = None
                    for frame in memory_filter.feed(chunk):
                        yield frame
            response = (
                self._parse_sse_to_response(round_bytes.decode("utf-8"), "anthropic")
                if round_bytes is not None
                else None
            )
            _add_round_usage(stream_state, response)
            logger.info(f"[{request_id}] Memory: Continuation round {rounds} streamed")

    _MEMORY_CONTINUATION_MAX_ROUNDS = 4

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

            # The payload is parsed straight from model output, so `hash` may be
            # missing or a non-string (e.g. {"hash": 123}). Skip anything that
            # isn't a usable string rather than letting `.lower()` raise.
            if not isinstance(hash_key, str) or not hash_key:
                continue

            # The store keys every entry by a lowercase hash (see
            # compression_store: sha256 hexdigest / explicit_hash.lower()), so a
            # model that echoes the marker hash uppercase would miss the lookup
            # and the retrieval feedback would never reach TOIN. Normalise here,
            # matching parse_tool_call.
            hash_key = hash_key.lower()

            logger.info(f"[{request_id}] CCR Feedback: Recording retrieval hash={hash_key[:8]}...")

            # Call store.retrieve() for the side effect of triggering the
            # feedback chain: _log_retrieval -> process_pending_feedback
            # -> toin.record_retrieval(). We discard the returned content.
            try:
                store.retrieve(hash_key)
            except Exception as e:
                logger.debug(f"[{request_id}] CCR Feedback recording failed: {e}")

    def _record_ccr_feedback_from_openai_sse(self, full_sse_data: str, request_id: str) -> None:
        """Record headroom_retrieve feedback from OpenAI Chat Completions SSE.

        OpenAI streams tool_calls incrementally via
        ``choices[0].delta.tool_calls[*].function.arguments`` (chunked
        JSON string). We accumulate per-call-index and finalize on
        stream completion. The accumulator records each completed
        ``headroom_retrieve`` invocation as a no-op store call for the
        TOIN feedback side effect (matches the Anthropic streaming
        feedback path).
        """
        from headroom.cache.compression_store import get_compression_store

        # tool_call_index -> {"name": str, "args_buf": str}
        tool_calls: dict[int, dict[str, str]] = {}

        for raw_line in full_sse_data.split("\n"):
            line = raw_line.strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if not payload or payload == "[DONE]":
                continue
            try:
                data = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(data, dict):
                continue

            choices = data.get("choices") or []
            if not choices:
                continue
            delta = (choices[0] or {}).get("delta") or {}
            for tc in delta.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                idx = tc.get("index", 0)
                fn = tc.get("function") or {}
                slot = tool_calls.setdefault(idx, {"name": "", "args_buf": ""})
                fn_name = fn.get("name")
                if fn_name:
                    slot["name"] = fn_name
                fn_args = fn.get("arguments")
                if fn_args:
                    slot["args_buf"] = slot["args_buf"] + fn_args

        if not tool_calls:
            return

        store = get_compression_store()
        for slot in tool_calls.values():
            if slot["name"] != "headroom_retrieve":
                continue
            try:
                input_data = json.loads(slot["args_buf"]) if slot["args_buf"] else {}
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(input_data, dict):
                continue
            hash_key = input_data.get("hash")
            # Skip a missing or non-string hash (malformed model output) instead
            # of raising in `.lower()`.
            if not isinstance(hash_key, str) or not hash_key:
                continue

            # Normalise to the store's lowercase key form (see
            # _record_ccr_feedback_from_response).
            hash_key = hash_key.lower()

            logger.info(
                f"[{request_id}] CCR Feedback (openai stream): Recording retrieval "
                f"hash={hash_key[:8]}..."
            )
            try:
                store.retrieve(hash_key)
            except Exception as e:
                logger.debug(f"[{request_id}] CCR Feedback (openai stream) failed: {e}")

    async def _finalize_stream_response(
        self,
        *,
        body: dict,
        provider: str,
        outcome_provider: str | None = None,
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
        client: str | None = None,
        waste_signals: dict[str, int] | None = None,
        status_code: int = 200,
        # Set only by paths whose ``tokens_saved`` is the CONVERSATION's
        # running total rather than this turn's -- OpenAI ``/v1/responses``,
        # which re-sends and recompresses the whole transcript every turn.
        # See ``conversation_savings``.
        conversation_key: str | None = None,
        conversation_tokens_saved: int | None = None,
    ) -> None:
        from headroom.proxy.outcome import RequestOutcome

        outcome_provider = outcome_provider or provider
        total_latency = (time.time() - start_time) * 1000

        # Per-chunk SSE parsing only flushes events terminated by ``\n\n``.
        # When upstream truncates mid-event (client disconnect, network
        # drop, connection reset), the residual bytes are not a complete
        # SSE event. Do not append a synthetic terminator here: doing so
        # turns a partial UTF-8 sequence or partial event into input for
        # the strict decoder and can raise while finalizing the stream.
        # Complete usage events have already been parsed in the read loop;
        # discard the incomplete tail instead of treating it as authoritative.
        sse_buffer = stream_state.get("sse_buffer")
        if isinstance(sse_buffer, bytearray) and len(sse_buffer) > 0:
            sse_buffer.clear()

        output_tokens = stream_state["output_tokens"]
        output_tokens_source = "provider"
        if output_tokens is None:
            # No usage chunk from the upstream. Count the stream's OWN TEXT
            # rather than dividing the raw wire by a constant: `total_bytes`
            # includes every `data:` prefix, JSON envelope and framing newline,
            # so its error tracked how chattily the answer was chunked instead
            # of how long the answer was. Falls back to the byte heuristic only
            # when no text could be recovered at all.
            output_tokens, output_tokens_source = estimate_output_tokens(
                sse_text=full_sse_data,
                total_bytes=stream_state["total_bytes"],
                # The provider body as sent, so a turn stopped by its output
                # ceiling can be counted exactly instead of estimated. Matters
                # most for a tool call truncated mid-arguments, whose dropped
                # JSON leaves almost no text to count.
                body=body,
            )
            # Name the actual basis. The old message always said "from N bytes"
            # even though that is now only true for the fallback rung, and an
            # operator reading it needs to know which estimate they are looking
            # at before trusting the number.
            basis = (
                "exact: turn hit its output-token ceiling"
                if output_tokens_source == "exact_ceiling"
                else "counted from stream text"
                if output_tokens_source == "estimated_text"
                else f"estimated from {stream_state['total_bytes']} raw SSE bytes"
            )
            logger.warning(
                f"[{request_id}] No usage chunk in SSE; output_tokens={output_tokens} ({basis})"
            )

        outcome_tags = dict(tags or {})
        outcome_tags["output_tokens_source"] = output_tokens_source

        provider_input_tokens = stream_state.get("input_tokens")
        effective_optimized_tokens = optimized_tokens
        effective_original_tokens = original_tokens
        if (
            provider in {"openai", "gemini"}
            and isinstance(provider_input_tokens, int)
            and provider_input_tokens > 0
        ):
            effective_optimized_tokens = provider_input_tokens
            effective_original_tokens = max(original_tokens, provider_input_tokens + tokens_saved)

        cache_read_tokens = stream_state["cache_read_input_tokens"] or 0
        cache_write_tokens = stream_state["cache_creation_input_tokens"] or 0
        cache_write_5m_tokens = stream_state["cache_creation_ephemeral_5m_input_tokens"] or 0
        cache_write_1h_tokens = stream_state["cache_creation_ephemeral_1h_input_tokens"] or 0
        if provider == "anthropic" and isinstance(provider_input_tokens, int):
            # Anthropic's usage.input_tokens already IS the uncached count
            # (tokens after the last cache breakpoint), the field the
            # non-streaming path records. Deriving it from Headroom's own
            # tokenizer count books the estimate's error as uncached input:
            # where the estimate runs over the provider's count (long
            # screenshot-heavy sessions, ~10%) every turn reported tens of
            # thousands of phantom uncached tokens, and where it runs under,
            # real uncached input clamped to 0.
            uncached_input_tokens = max(provider_input_tokens, 0)
        else:
            uncached_input_tokens = max(
                effective_optimized_tokens - cache_read_tokens - cache_write_tokens, 0
            )

        # The provider's own billed input for this turn. Without it the outcome
        # falls back to ``optimized_tokens`` — Headroom's tokenizer estimate of
        # the forwarded messages only (no system prompt, no tool definitions) —
        # and every volume/cost total downstream (dashboard, licence usage
        # report) silently reports the estimate as if it were the bill. Every
        # Claude Code turn is a streamed Anthropic turn, so this was the bulk of
        # Claude traffic.
        billed_provider_input = (
            billed_input_for_provider(
                provider,
                provider_input_tokens,
                cache_read=cache_read_tokens,
                cache_write=cache_write_tokens,
            )
            if isinstance(provider_input_tokens, int)
            else 0
        )

        # Prefix-tracker mutation is provider-specific state that lives
        # outside the metric funnel. Run it before the funnel so the next
        # request inherits correct prefix state regardless of metric path.
        if 200 <= status_code < 300 and prefix_tracker is not None:
            import copy as _copy

            forwarded_messages = body.get("messages", [])
            if not forwarded_messages and provider == "gemini":
                # Gemini bodies carry contents[] rather than messages[] (Cloud
                # Code Assist nests the payload under body["request"]).
                # Convert with the same helper the Gemini handler used so the
                # tracker walks the same message shape (#3394).
                _gemini_payload = body["request"] if isinstance(body.get("request"), dict) else body
                forwarded_messages, _ = self._gemini_contents_to_messages(
                    _gemini_payload.get("contents", []),
                    _gemini_payload.get("systemInstruction"),
                )
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

            # Cache-miss attribution (#1313), streaming Anthropic path. Mirror
            # the non-streaming handler: classify BEFORE update_from_response
            # overwrites the last-turn state the classifier reads. Compare the
            # prefix we forwarded this turn (`forwarded_messages`, pre-assistant
            # append) against last turn's.
            # `hasattr` guard: stub trackers in tests may implement only the
            # freeze API, not the full PrefixCacheTracker surface.
            if provider == "anthropic" and hasattr(prefix_tracker, "classify_cache_miss"):
                miss = prefix_tracker.classify_cache_miss(
                    cache_read_tokens=cache_read_tokens,
                    current_forwarded_messages=forwarded_messages,
                )
                if miss.is_miss:
                    logger.info(
                        f"[{request_id}] CACHE-MISS-ATTRIBUTION: reason={miss.reason} "
                        f"idle={miss.idle_seconds:.0f}s ttl={miss.cache_ttl_seconds}s "
                        f"expected_cached={miss.expected_cached_tokens:,} "
                        f"prefix_changed={miss.prefix_changed} ttl_exceeded={miss.ttl_exceeded}"
                    )
                    await self.metrics.record_cache_miss_attribution(provider, miss.reason)

            tracker_cache_write = cache_write_tokens
            if provider == "gemini" and tracker_cache_write == 0:
                # Gemini's stream usage reports cache reads only
                # (cachedContentTokenCount); implicit caching has no write
                # counter, so the uncached input portion is the write proxy
                # (same inference as the OpenAI buffered path). Kept as a
                # tracker-local value: Gemini outcomes intentionally report no
                # cache-write concept (#3394).
                tracker_cache_write = max(effective_optimized_tokens - cache_read_tokens, 0)

            prefix_tracker.update_from_response(
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=tracker_cache_write,
                messages=next_forwarded,
                original_messages=next_original,
            )

        # Active-compression denominator (``attempted_input_tokens``) is
        # derived inside ``RequestOutcome.from_stream`` as
        # ``optimized_tokens + tokens_saved``. No frozen_message_count
        # propagates to the streaming finalizer yet — per-message
        # live-zone tracking is a follow-up. Without this fallback the
        # dashboard headline collapses to 0% even when compression is
        # happening (issue #455).
        # Thinking split and stop_reason for the streaming path. Streaming is
        # where the agentic traffic is, so without this the split — and the one
        # feedback signal an adaptive ceiling can learn from — would exist only
        # for non-streaming requests, which is close to nowhere in practice.
        #
        # ``parsed_response`` is the reassembled SSE body: it carries the content
        # blocks (thinking among them) and stop_reason, which the raw usage chunk
        # does not. Same helper as the non-streaming path, so a streamed and an
        # unstreamed response of the same shape count the same.
        from headroom.proxy.handlers.anthropic import _thinking_tokens_for

        _stream_thinking = _thinking_tokens_for(parsed_response)
        _stream_stop_reason = (
            parsed_response.get("stop_reason") if isinstance(parsed_response, dict) else None
        )
        outcome = RequestOutcome.from_stream(
            thinking_tokens=_stream_thinking.tokens,
            thinking_inferred=_stream_thinking.inferred,
            stop_reason=_stream_stop_reason,
            body=body,
            provider=outcome_provider,
            model=model,
            request_id=request_id,
            original_tokens=effective_original_tokens,
            optimized_tokens=effective_optimized_tokens,
            provider_input_tokens=billed_provider_input,
            output_tokens=output_tokens,
            tokens_saved=tokens_saved,
            transforms_applied=transforms_applied,
            total_latency_ms=total_latency,
            overhead_ms=optimization_latency,
            tags=outcome_tags,
            client=client,
            status_code=status_code,
            log_full_messages=getattr(self.config, "log_full_messages", False),
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            cache_write_5m_tokens=cache_write_5m_tokens,
            cache_write_1h_tokens=cache_write_1h_tokens,
            uncached_input_tokens=uncached_input_tokens,
            ttfb_ms=stream_state["ttfb_ms"] or total_latency,
            pipeline_timing=pipeline_timing,
            original_messages=original_messages,
            waste_signals=waste_signals,
            conversation_key=conversation_key,
            conversation_tokens_saved=conversation_tokens_saved,
        )
        await self._record_request_outcome(outcome)

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
        memory_request_ctx: Any | None = None,
        outcome_provider: str | None = None,
        waste_signals: dict[str, int] | None = None,
        session_key: str | None = None,
        conversation_key: str | None = None,
        conversation_tokens_saved: int | None = None,
        server_memory_tool_names: frozenset[str] | None = None,
        client_beta: str | None = None,
    ) -> Response | StreamingResponse:
        """Stream response with metrics tracking and memory tool handling.

        Parses SSE events to extract actual usage information from the API response
        for accurate token counting and cost calculation.

        When the proxy injected memory tools (``server_memory_tool_names``),
        this method streams live but withholds those tool calls from the
        client, executes them, and streams server-side continuation rounds
        into the same client message until no memory tool call remains.
        """
        session_key = session_key or self._get_session_key(body)

        # Guard everything up to the generator's own try/finally (which owns
        # cleanup once streaming starts): any exception here — including
        # asyncio.CancelledError from a client disconnect mid-setup — must
        # still release session_key, or it wedges in _active_streams forever
        # and every later request on this session gets stuck 202-queued.
        try:
            return await self._stream_response_inner(
                url=url,
                headers=headers,
                body=body,
                provider=provider,
                model=model,
                request_id=request_id,
                original_tokens=original_tokens,
                optimized_tokens=optimized_tokens,
                tokens_saved=tokens_saved,
                transforms_applied=transforms_applied,
                tags=tags,
                optimization_latency=optimization_latency,
                memory_user_id=memory_user_id,
                pipeline_timing=pipeline_timing,
                prefix_tracker=prefix_tracker,
                original_messages=original_messages,
                original_body_bytes=original_body_bytes,
                body_mutated=body_mutated,
                mutation_reasons=mutation_reasons,
                memory_request_ctx=memory_request_ctx,
                outcome_provider=outcome_provider,
                waste_signals=waste_signals,
                session_key=session_key,
                conversation_key=conversation_key,
                conversation_tokens_saved=conversation_tokens_saved,
                server_memory_tool_names=server_memory_tool_names,
                client_beta=client_beta,
            )
        except (Exception, asyncio.CancelledError):
            self._cleanup_mid_turn_stream(session_key)
            raise

    async def _stream_response_inner(
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
        memory_user_id: str | None,
        pipeline_timing: dict[str, float] | None,
        prefix_tracker: Any | None,
        original_messages: list[dict] | None,
        original_body_bytes: bytes | None,
        body_mutated: bool,
        mutation_reasons: list[str] | None,
        memory_request_ctx: Any | None,
        outcome_provider: str | None,
        waste_signals: dict[str, int] | None,
        session_key: str,
        conversation_key: str | None = None,
        conversation_tokens_saved: int | None = None,
        server_memory_tool_names: frozenset[str] | None = None,
        client_beta: str | None = None,
    ) -> Response | StreamingResponse:
        """Actual streaming implementation, guarded by _stream_response's cleanup wrapper."""
        from fastapi.responses import Response, StreamingResponse
        from starlette.background import BackgroundTask

        from headroom.proxy.helpers import MAX_SSE_BUFFER_SIZE

        # Identify the harness (codex / claude-code / aider / cursor /
        # ...) from the *client's* User-Agent before copilot-auth
        # potentially rewrites headers for upstream.
        client = classify_client(headers)
        # Mid-turn message coalescing (queueing a concurrent same-session
        # request and later replaying it via a `headroom_pending_messages`
        # SSE event) is a Claude Code-only protocol. Only register the stream
        # as active for coalescing when the client can consume that protocol,
        # so concurrent requests from other harnesses (e.g. OpenCode subagents
        # that share a body-derived session key) are streamed normally instead
        # of being swallowed. (#1608)
        if supports_mid_turn_coalescing(client):
            self._active_streams.add(session_key)

        # Before the guard: its learned-limit lookup is keyed on the credential
        # actually sent, the same one the 400 learning below keys on.
        headers = await apply_copilot_api_auth(headers, url=url)

        # Context-limit guard (see headroom/proxy/context_guard.py): rewrites
        # only the client-bound message_start bytes when the forwarded request
        # is near the model's real window, so clients whose auto-compaction
        # keys off reported usage compact gracefully instead of looping on
        # prompt-too-long 400s. Metrics below parse the original upstream
        # bytes and are unaffected.
        context_guard = None
        if provider == "anthropic":
            from headroom.proxy.context_guard import (
                StreamUsageGuard,
                believed_context_limit,
                context_guard_enabled,
                credential_scope_from_headers,
                effective_context_limit,
            )

            try:
                if context_guard_enabled():
                    _guard_model_limit = self.anthropic_provider.get_context_limit(model)
                    _guard_beta = headers.get("anthropic-beta")
                    _guard_scope = credential_scope_from_headers(headers)
                    context_guard = StreamUsageGuard(
                        # The client's gauge follows the beta the client
                        # sent; session-sticky merging can add context-1m to
                        # the outbound header after the client dropped it.
                        # None: the caller did not say, use the outbound one.
                        believed_limit=believed_context_limit(
                            _guard_model_limit,
                            _guard_beta if client_beta is None else client_beta,
                        ),
                        effective_limit=effective_context_limit(
                            model, _guard_model_limit, _guard_beta, scope=_guard_scope
                        ),
                        request_id=request_id,
                    )
            except Exception:
                logger.debug("context_guard setup skipped", exc_info=True)

        def _guard_client_bytes(data: bytes) -> bytes:
            """Client-bound bytes through the context guard, when there is one."""
            return context_guard.feed(data) if context_guard is not None else data

        def _drain_context_guard() -> bytes:
            """Held-back guard bytes, once. Safe to call on any exit path."""
            if context_guard is None:
                return b""
            try:
                return context_guard.flush()
            except Exception:
                logger.debug("context_guard flush skipped", exc_info=True)
                return b""

        start_time = time.time()

        # Byte-faithful forwarding (PR-A3, fixes P0-2). Resolve outbound
        # bytes once before entering the connection-retry loop. When a
        # transform mutated the body we re-serialize canonically; otherwise
        # we forward the original client bytes verbatim.
        from headroom.proxy.anthropic_wire import is_safeguard_capable_request
        from headroom.proxy.body_forwarding import select_outbound_body
        from headroom.proxy.helpers import (
            capture_codex_wire_debug,
            codex_wire_debug_enabled,
            log_outbound_request,
        )

        outbound = select_outbound_body(
            body=body,
            original_body_bytes=original_body_bytes,
            body_mutated=body_mutated,
            mutation_reasons=list(mutation_reasons or []),
        )
        outbound_bytes, outbound_source = outbound.content, outbound.source
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
            dropped_mutation_reasons=outbound.dropped_mutation_reasons,
        )
        safeguard_capable = provider == "anthropic" and is_safeguard_capable_request(
            body, headers.get("anthropic-beta")
        )
        _codex_wire_debug = (
            codex_wire_debug_enabled()
            and not safeguard_capable
            and provider == "openai"
            and "/responses" in url
        )
        if _codex_wire_debug:
            capture_codex_wire_debug(
                "http_stream_upstream_request",
                request_id=request_id,
                transport="http_sse",
                direction="headroom_to_upstream",
                method="POST",
                url=url,
                headers=outbound_headers,
                body=body,
                metadata={
                    "body_bytes": len(outbound_bytes),
                    "body_mutated": body_mutated,
                    "mutation_reasons": list(mutation_reasons or []),
                    "tokens_saved": tokens_saved,
                    "transforms_applied": transforms_applied,
                },
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
            "stream_failed": False,
            "stream_terminal": False,
            "gemini_candidate_count": _gemini_requested_candidates(body),
            "sse_dialect": (
                "anthropic"
                if provider == "anthropic"
                else "gemini"
                if provider == "gemini"
                else "responses"
                if urlsplit(url).path.rstrip("/").endswith("/responses")
                else "chat"
            ),
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
                    if _codex_wire_debug:
                        capture_codex_wire_debug(
                            "http_stream_upstream_response_headers",
                            request_id=request_id,
                            transport="http_sse",
                            direction="upstream_to_headroom",
                            method="POST",
                            url=url,
                            headers=dict(upstream_response.headers),
                            status_code=upstream_response.status_code,
                        )
                    # Retry transient overloads (429 rate-limit, 529 overloaded)
                    # honoring Retry-After — the streaming sibling of the
                    # _retry_request path (#1221); on exhaustion, fall through to
                    # forward the status to the client.
                    if (
                        upstream_response.status_code in RETRYABLE_OVERLOAD_STATUSES
                        and self.config.retry_enabled
                        and attempt < retry_attempts - 1
                        and not overload_retry_is_futile(
                            upstream_response,
                            self.config.retry_max_delay_ms,
                            retries_left=retry_attempts - attempt - 1,
                        )
                    ):
                        delay_with_jitter = retry_after_ms(
                            upstream_response, self.config.retry_max_delay_ms
                        ) or jitter_delay_ms(
                            self.config.retry_base_delay_ms,
                            self.config.retry_max_delay_ms,
                            attempt,
                        )
                        await upstream_response.aclose()
                        logger.warning(
                            f"[{request_id}] Upstream {upstream_response.status_code} "
                            f"(attempt {attempt + 1}/{retry_attempts}), "
                            f"retrying in {delay_with_jitter:.0f}ms"
                        )
                        await asyncio.sleep(delay_with_jitter / 1000)
                        continue
                    break
                # Retry any transport-level failure while opening the upstream
                # stream — including HTTP/2 protocol errors (Local/RemoteProtocol
                # `StreamReset`) from a poisoned shared h2 connection. This runs
                # before any body byte is forwarded to the client, so re-sending
                # on a fresh connection is safe and avoids a 502. (#1639)
                except httpx.TransportError as e:
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
        # Retries exhausted (or a transport failure escaped the loop): emit a
        # clean SSE error instead of letting an h2 StreamReset bubble up as an
        # unhandled 502. Covers ConnectError/timeouts and Local/RemoteProtocol-
        # Error. (#1639)
        #
        # The status MUST stay non-2xx. No body byte has been forwarded yet, so
        # the status line is still ours to set, and a 200 here is
        # indistinguishable — to every Anthropic/OpenAI SDK — from a successful
        # stream that produced no events: the client reports "empty or malformed
        # response (HTTP 200)" and cannot retry, because 200 is not retryable.
        # 502 keeps the structured SSE body for clients that read it while
        # letting SDK retry logic treat a transient upstream transport failure
        # as what it is.
        except httpx.TransportError as e:
            error_msg = str(e) or repr(e)
            logger.error(f"[{request_id}] Connection error to upstream API: {error_msg}")
            self.metrics.record_upstream_connection_error(provider)
            # A corporate TLS-inspection root the proxy does not trust shows up
            # here as CERTIFICATE_VERIFY_FAILED. Name the issuer and the fix so
            # the agent's error line is actionable instead of a bare 502.
            from headroom.proxy.tls_diagnostics import describe_upstream_failure_async

            tls_hint = await describe_upstream_failure_async(e, url)
            # The exception text itself stays in the log above: it names the
            # resolved upstream host and address (public_errors module doc).
            error_body = public_errors.anthropic_error_body(
                public_errors.classify_or_internal(e),
                request_id=request_id,
                error_type="connection_error",
                hint=tls_hint,
            )

            async def _error_gen():
                yield f"event: error\ndata: {json.dumps(error_body)}\n\n".encode()

            self._cleanup_mid_turn_stream(session_key)
            return StreamingResponse(
                _error_gen(),
                status_code=502,
                media_type="text/event-stream",
            )

        # Capture Codex rate-limit window data from the upstream response
        # headers, for *every* status. Codex (gpt-5.x) almost always streams, so
        # without this the session/weekly windows surfaced in ``/stats`` and the
        # dashboard would only refresh on the rare non-streaming reply. We do this
        # *before* the error early-return below so a streaming 429/5xx — the moment
        # usage is most relevant — still refreshes the windows, matching the
        # non-streaming HTTP handlers which capture on all statuses.
        # ``update_from_headers`` is a no-op when the response carries no
        # ``x-codex-*`` headers (e.g. the Anthropic streaming path), so this is
        # safe to call unconditionally.
        from headroom.subscription.codex_rate_limits import (
            get_codex_rate_limit_state,
        )

        get_codex_rate_limit_state().update_from_headers(dict(upstream_response.headers))

        if upstream_response.status_code >= 400:
            logger.warning(
                "[%s] Forwarding upstream streaming error status=%s url=%s",
                request_id,
                upstream_response.status_code,
                url,
            )
            # Diagnostic dump of the erroring request — parity with the
            # non-streaming handlers, which dump on >=400 but never fire for a
            # streaming turn (Claude Code streams every request, so the most
            # common 400s were invisible). Same gating: OFF by default, never
            # in stateless mode, content redacted unless HEADROOM_DEBUG_DUMP=full.
            # Dump the bytes that went on the wire, not ``body``: when the edits
            # are dropped (source="passthrough") ``body`` shows a request that
            # never left the proxy.
            write_upstream_error_dump(
                getattr(self, "config", None),
                request_id=request_id,
                url=url,
                status=upstream_response.status_code,
                provider=provider,
                model=model,
                body=outbound_bytes,
                body_source=outbound_source,
                transforms=transforms_applied,
                stream=True,
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

            if provider == "anthropic" and upstream_response.status_code == 400:
                # Diagnostic only: whatever it makes of the body, the client
                # still gets the upstream error below.
                try:
                    from headroom.proxy.context_guard import (
                        credential_scope_from_headers,
                        note_prompt_too_long,
                    )

                    note_prompt_too_long(
                        model,
                        headers.get("anthropic-beta"),
                        error_content,
                        scope=credential_scope_from_headers(headers),
                    )
                except Exception:
                    logger.debug("context_guard: limit learning skipped", exc_info=True)

            if _codex_wire_debug:
                _error_text: str | None = None
                _error_body: Any = None
                try:
                    _error_text = error_content.decode("utf-8")
                    _error_body = json.loads(_error_text)
                    _error_text = None
                except Exception:
                    with contextlib.suppress(Exception):
                        _error_text = error_content.decode("utf-8", errors="replace")
                capture_codex_wire_debug(
                    "http_stream_upstream_error_response",
                    request_id=request_id,
                    transport="http_sse",
                    direction="upstream_to_headroom",
                    method="POST",
                    url=url,
                    headers=response_headers,
                    body=_error_body,
                    raw_text=_error_text,
                    status_code=upstream_response.status_code,
                )

            stream_state["total_bytes"] = len(error_content)
            await self._finalize_stream_response(
                body=body,
                provider=provider,
                outcome_provider=outcome_provider,
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
                client=client,
                waste_signals=waste_signals,
                status_code=upstream_response.status_code,
                conversation_key=conversation_key,
                conversation_tokens_saved=conversation_tokens_saved,
            )
            self._cleanup_mid_turn_stream(session_key)
            return Response(
                content=error_content,
                status_code=upstream_response.status_code,
                headers=response_headers,
            )

        # Forward upstream rate-limit headers to the client. We pass both the
        # generic ``*ratelimit*`` headers (Anthropic) and Codex's ``x-codex-*``
        # window/credit headers — the latter do not contain the ``ratelimit``
        # substring, so without the second clause the Codex CLI's own
        # session/weekly display would stop updating on the streaming path.
        # We also forward the ``request-id`` family: clients such as Claude Code
        # record it per transcript turn, and downstream usage/cost tools dedup by
        # message id + request id. ``x-litellm-key-*`` carries the per-key
        # spend/budget envelope a LiteLLM virtual key reports per request;
        # downstream status widgets read it from the streaming response the
        # same way they already do from the buffered path. The buffered
        # (non-streaming) path already forwards every upstream header, so
        # this keeps the two paths symmetric.
        forwarded_headers = {
            k: v
            for k, v in upstream_response.headers.items()
            if "ratelimit" in k.lower()
            or k.lower().startswith("x-codex")
            or k.lower().startswith("x-litellm-key-")
            or k.lower() in ("request-id", "anthropic-request-id", "x-request-id")
        }
        # Headroom's own compression metrics, as the buffered path stamps them.
        # Every counted value is known before the first byte, and streaming is
        # how real coding agents talk to the proxy, so without these a client
        # (or a metering layer in front of it) never learns what its request
        # saved.
        forwarded_headers["x-headroom-tokens-before"] = str(original_tokens)
        forwarded_headers["x-headroom-tokens-after"] = str(optimized_tokens)
        forwarded_headers["x-headroom-tokens-saved"] = str(tokens_saved)
        forwarded_headers["x-headroom-model"] = model
        if transforms_applied:
            from headroom.proxy.cost import header_safe_transforms

            forwarded_headers["x-headroom-transforms"] = ",".join(
                header_safe_transforms(transforms_applied)
            )

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
            completed_normally = False
            pending_messages: list[dict] = []
            # Proxy-injected memory tools are invisible to the client, so their
            # tool_use blocks must never reach it (GH #2195).
            memory_filter = (
                MemoryToolStreamFilter(server_memory_tool_names)
                if memory_enabled and server_memory_tool_names
                else None
            )

            try:
                async with contextlib.aclosing(upstream_response) as response:
                    sse_chunk_index = 0
                    async for chunk in response.aiter_bytes():
                        sse_chunk_index += 1
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
                        # The context guard runs last, on exactly what the
                        # client receives (after the memory filter), and may
                        # hold back bytes only until the first complete event.
                        client_frames = (
                            [chunk] if memory_filter is None else memory_filter.feed(chunk)
                        )
                        for frame in client_frames:
                            guarded_frame = _guard_client_bytes(frame)
                            if guarded_frame:
                                yield guarded_frame

                        if _codex_wire_debug:
                            capture_codex_wire_debug(
                                "http_stream_upstream_chunk",
                                request_id=request_id,
                                transport="http_sse",
                                direction="upstream_to_headroom",
                                method="POST",
                                url=url,
                                raw_text=chunk.decode("utf-8", errors="replace"),
                                metadata={
                                    "chunk": sse_chunk_index,
                                    "byte_count": len(chunk),
                                },
                            )

                        # Buffer SSE data for memory processing and/or prefix tracker
                        _track_sse = (
                            _codex_wire_debug
                            or memory_enabled
                            or (prefix_tracker is not None and provider == "anthropic")
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
                        if memory_filter is not None:
                            for frame in memory_filter.closing_frames():
                                guarded_frame = _guard_client_bytes(frame)
                                if guarded_frame:
                                    yield guarded_frame
                        guarded_tail = _drain_context_guard()
                        if guarded_tail:
                            yield guarded_tail
                        return

                    # Parse SSE to get response JSON
                    parsed_response = self._parse_sse_to_response(full_sse_data, provider)

                    if memory_filter is not None:
                        async for frame in self._continue_memory_tool_stream(
                            memory_filter,
                            parsed_response,
                            url=url,
                            outbound_headers=outbound_headers,
                            outbound_bytes=outbound_bytes,
                            memory_user_id=memory_user_id,
                            memory_request_ctx=memory_request_ctx,
                            server_memory_tool_names=server_memory_tool_names or frozenset(),
                            stream_state=stream_state,
                            request_id=request_id,
                        ):
                            guarded_frame = _guard_client_bytes(frame)
                            if guarded_frame:
                                yield guarded_frame
                        memory_filter = None
                        parsed_response = (
                            stream_state.pop("memory_client_message", None) or parsed_response
                        )
                    elif parsed_response and self.memory_handler.has_memory_tool_calls(
                        parsed_response, provider
                    ):
                        logger.info(
                            f"[{request_id}] Memory: Detected tool calls in streaming response"
                        )

                        # The client declared these memory tools itself, so the
                        # tool_use already reached it and the client continues
                        # the turn; only the server-side save happens here.
                        tool_results = await self.memory_handler.handle_memory_tool_calls(
                            parsed_response,
                            memory_user_id,
                            provider,
                            request_context=memory_request_ctx,
                        )
                        if tool_results:
                            logger.info(
                                f"[{request_id}] Memory: Tool calls executed "
                                f"({len(tool_results)} results saved, SSE streaming — "
                                "continuation handled by client)"
                            )

                if memory_filter is not None:
                    # Memory detection was skipped (empty stream or buffer cap):
                    # still run the withheld calls, then end the message.
                    async for frame in self._continue_memory_tool_stream(
                        memory_filter,
                        None,
                        url=url,
                        outbound_headers=outbound_headers,
                        outbound_bytes=outbound_bytes,
                        memory_user_id=memory_user_id,
                        memory_request_ctx=memory_request_ctx,
                        server_memory_tool_names=server_memory_tool_names or frozenset(),
                        stream_state=stream_state,
                        request_id=request_id,
                    ):
                        guarded_frame = _guard_client_bytes(frame)
                        if guarded_frame:
                            yield guarded_frame

                # A stream that ended before its first complete event may
                # leave held-back bytes in the context guard; release them,
                # after the memory filter's held tail so the final
                # message_delta is still nudged. Every exit does this,
                # including the error paths below: bytes the guard is holding
                # are bytes the upstream already sent, and dropping them
                # silently truncates the client's stream.
                guarded_tail = _drain_context_guard()
                if guarded_tail:
                    yield guarded_tail

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
                if _codex_wire_debug:
                    _debug_parsed_response = (
                        parsed_response
                        if parsed_response
                        else self._parse_sse_to_response(full_sse_data, provider)
                        if full_sse_data
                        else None
                    )
                    capture_codex_wire_debug(
                        "http_stream_upstream_complete",
                        request_id=request_id,
                        transport="http_sse",
                        direction="upstream_to_headroom",
                        method="POST",
                        url=url,
                        headers=dict(upstream_response.headers),
                        body=_debug_parsed_response,
                        raw_text=full_sse_data,
                        status_code=upstream_response.status_code,
                        metadata={"total_bytes": stream_state["total_bytes"]},
                    )
                completed_normally = (
                    stream_state["stream_terminal"] and not stream_state["stream_failed"]
                )

            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as e:
                logger.error(f"[{request_id}] Connection error to upstream API: {e}")
                guarded_tail = _drain_context_guard()
                if guarded_tail:
                    yield guarded_tail
                error_event = public_errors.anthropic_error_body(
                    public_errors.classify_or_internal(e),
                    request_id=request_id,
                    error_type="connection_error",
                )
                yield f"event: error\ndata: {json.dumps(error_event)}\n\n".encode()
            except httpx.HTTPStatusError as e:
                logger.error(f"[{request_id}] HTTP error from upstream API: {e}")
                guarded_tail = _drain_context_guard()
                if guarded_tail:
                    yield guarded_tail
                # Forward the upstream error response
                yield e.response.content
            except MemoryToolStreamOverflowError as e:
                # The filter dropped what it withheld; end the stream rather
                # than forward it.
                logger.error(f"[{request_id}] Memory: {e}; ending the stream")
                error_event = public_errors.anthropic_error_body(
                    public_errors.INTERNAL_ERROR, request_id=request_id
                )
                yield f"event: error\ndata: {json.dumps(error_event)}\n\n".encode()
            except Exception as e:
                logger.error(f"[{request_id}] Unexpected streaming error: {e}")
                guarded_tail = _drain_context_guard()
                if guarded_tail:
                    yield guarded_tail
                error_event = public_errors.anthropic_error_body(
                    public_errors.classify_or_internal(e),
                    request_id=request_id,
                    error_type="api_error",
                )
                yield f"event: error\ndata: {json.dumps(error_event)}\n\n".encode()
            finally:
                pending_messages = self._cleanup_mid_turn_stream(
                    session_key,
                    drain_pending_messages=completed_normally,
                )
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
                    outcome_provider=outcome_provider,
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
                    client=client,
                    waste_signals=waste_signals,
                    status_code=_stream_outcome_status(
                        upstream_response.status_code, completed_normally
                    ),
                    conversation_key=conversation_key,
                    conversation_tokens_saved=conversation_tokens_saved,
                )
                if supports_mid_turn_coalescing(client) and pending_messages:
                    pending_event = json.dumps(
                        {"type": "headroom_pending_messages", "messages": pending_messages}
                    )
                    yield f"event: headroom_pending_messages\ndata: {pending_event}\n\n".encode()

        async def _release_upstream_stream() -> None:
            # Guarantee the upstream HTTP/2 stream is released even when the
            # body generator above is never iterated — the client disconnected
            # before Starlette started sending the response body (routine when a
            # harness like Claude Code cancels or supersedes an in-flight turn),
            # so ``generate()`` never entered its own ``aclosing`` and nothing
            # else closes ``upstream_response``. Each such request otherwise
            # leaks one open h2 stream; they accumulate on the pooled upstream
            # connection until it reaches SETTINGS_MAX_CONCURRENT_STREAMS (100)
            # and no new stream can open ("Max outbound streams is 100, 100
            # open"), and the proxy goes unhealthy until restart (#2797).
            # Starlette runs a response's ``background`` task after the body
            # finishes *and* after an early client disconnect, so this fires in
            # both cases. ``aclose()`` is idempotent, so on the normal path —
            # where the generator already closed the stream — this is a no-op.
            with contextlib.suppress(Exception):
                await upstream_response.aclose()

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers=forwarded_headers,
            background=BackgroundTask(_release_upstream_stream),
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
        original_messages: list[dict] | None = None,
        prefix_tracker: Any | None = None,
        optimized_messages: list[dict] | None = None,
        backend: Any | None = None,
    ) -> StreamingResponse:
        """Stream response from Bedrock backend with metrics tracking.

        Translates Bedrock streaming events to Anthropic SSE format.

        ``prefix_tracker``/``optimized_messages`` carry the
        :class:`PrefixCacheTracker` for the session so cache stats from
        this turn update the tracker for the next one — mirrors the
        direct streaming path (``_finalize_stream_response``) and the
        OpenAI-via-backend sibling (``_stream_openai_via_backend``).
        Without this, ``extract_cache_stable_delta()`` always sees no
        previous turn on the Bedrock path and cache mode never compresses
        anything past the first request in a session.
        """
        from fastapi.responses import StreamingResponse

        from headroom.proxy.outcome import RequestOutcome

        # ``backend`` lets the caller serve this one request from somewhere
        # other than the configured backend (see proxy/route_advice.py). None
        # means "the configured one", i.e. what this method always did.
        backend = backend if backend is not None else self.anthropic_backend

        client = classify_client(headers)

        start_time = time.time()

        # Mutable state for the generator. Cache fields mirror the
        # native ``_finalize_stream_response`` shape so the PERF log
        # values match between paths (issue #327).
        stream_state: dict[str, Any] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "ttfb_ms": None,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_creation_ephemeral_5m_input_tokens": 0,
            "cache_creation_ephemeral_1h_input_tokens": 0,
        }
        # Bytes-level mirror of the SSE stream, used only to reconstruct
        # the final assistant message for the prefix tracker once the
        # stream closes (see finally: block below). Not on the hot path
        # for anything the client sees.
        full_sse_bytes = bytearray()
        completed_normally = False
        stream_failed = False
        stream_terminal = False

        async def generate():
            nonlocal completed_normally, stream_failed, stream_terminal
            try:
                assert backend is not None

                # Emit a synthetic ping before the first message_start so that
                # downstream clients (e.g. Claude Code) arm their mid-turn
                # steering / interruptible state.  The Bedrock-to-Anthropic
                # translation layer never produces SSE-level keepalives; we
                # synthesise one here to match the real Anthropic wire format
                # (issue #902).
                yield b"event: ping\ndata: {}\n\n"

                async for event in backend.stream_message(body, headers):
                    # Record TTFB on first event
                    if stream_state["ttfb_ms"] is None:
                        stream_state["ttfb_ms"] = (time.time() - start_time) * 1000

                    # Backfill input_tokens on message_start (issue #1132).
                    # LiteLLM/Bedrock streaming never surfaces prompt tokens
                    # (only output_tokens, at the end), so the backend emits
                    # message_start with usage.input_tokens=0. Anthropic clients
                    # (e.g. Claude Code) read input_tokens from message_start and
                    # would otherwise report ~0 input for every request. Inject
                    # the token count Headroom actually sent upstream
                    # (optimized_tokens) when the backend left it unset/zero, so
                    # downstream metrics reflect real usage. A non-zero value
                    # already reported by the backend is preserved untouched.
                    if event.event_type == "message_start" and not event.raw_sse:
                        msg_usage = event.data.setdefault("message", {}).setdefault("usage", {})
                        # A zero with cache usage beside it is a real count: a
                        # fully cached prompt bills only the cache buckets. Only
                        # a bare zero is LiteLLM's unknown placeholder.
                        _cached = any(
                            isinstance(msg_usage.get(k), int) and msg_usage.get(k) > 0
                            for k in ("cache_read_input_tokens", "cache_creation_input_tokens")
                        )
                        if (
                            not msg_usage.get("input_tokens")
                            and not _cached
                            and optimized_tokens > 0
                        ):
                            msg_usage["input_tokens"] = optimized_tokens
                            # The client-bound backfill is Headroom's estimate,
                            # not backend usage: never report it as billed.
                            stream_state["input_tokens_backfilled"] = True

                    # Format as SSE
                    if event.raw_sse:
                        chunk_bytes = event.raw_sse.encode()
                    else:
                        sse_line = f"event: {event.event_type}\ndata: {json.dumps(event.data)}\n\n"
                        chunk_bytes = sse_line.encode()
                    if prefix_tracker is not None:
                        full_sse_bytes.extend(chunk_bytes)
                    yield chunk_bytes

                    # Track usage from message_start event
                    if event.event_type == "message_start":
                        msg = event.data.get("message", {})
                        usage = msg.get("usage", {})
                        if "input_tokens" in usage:
                            stream_state["input_tokens"] = usage["input_tokens"]
                            if not stream_state.get("input_tokens_backfilled"):
                                stream_state["backend_input_tokens"] = usage["input_tokens"]
                        stream_state["cache_read_input_tokens"] = usage.get(
                            "cache_read_input_tokens", 0
                        )
                        stream_state["cache_creation_input_tokens"] = usage.get(
                            "cache_creation_input_tokens", 0
                        )
                        cw_5m, cw_1h = self._extract_anthropic_cache_ttl_metrics(usage)
                        stream_state["cache_creation_ephemeral_5m_input_tokens"] = cw_5m
                        stream_state["cache_creation_ephemeral_1h_input_tokens"] = cw_1h

                    # Track output tokens from message_delta
                    if event.event_type == "message_delta":
                        usage = event.data.get("usage", {})
                        if "output_tokens" in usage:
                            stream_state["output_tokens"] = usage["output_tokens"]
                        if "input_tokens" in usage:
                            stream_state["input_tokens"] = usage["input_tokens"]
                            stream_state["backend_input_tokens"] = usage["input_tokens"]
                        if "cache_read_input_tokens" in usage:
                            stream_state["cache_read_input_tokens"] = usage[
                                "cache_read_input_tokens"
                            ]
                        if "cache_creation_input_tokens" in usage:
                            stream_state["cache_creation_input_tokens"] = usage[
                                "cache_creation_input_tokens"
                            ]

                    # Handle errors
                    failed, terminal = _sse_event_outcome(
                        event.event_type, json.dumps(event.data), "anthropic"
                    )
                    stream_failed = stream_failed or failed
                    stream_terminal = stream_terminal or terminal
                    if failed:
                        logger.error(f"[{request_id}] Bedrock stream error: {event.data}")

                completed_normally = stream_terminal and not stream_failed

            except Exception as e:
                error_message = format_exception_message(e)
                logger.error(f"[{request_id}] Bedrock streaming error: {error_message}")
                error_event = public_errors.anthropic_error_body(
                    public_errors.classify_or_internal(e),
                    request_id=request_id,
                    error_type="api_error",
                )
                yield f"event: error\ndata: {json.dumps(error_event)}\n\n".encode()

            finally:
                total_latency = (time.time() - start_time) * 1000
                _backend_name = backend.name if backend else "anthropic"

                # Update prefix cache tracker for the next turn — mirrors
                # _finalize_stream_response (direct-API streaming path)
                # and _stream_openai_via_backend (OpenAI-via-backend
                # sibling). Run before the outcome funnel so prefix state
                # is consistent regardless of metric path.
                if completed_normally and prefix_tracker is not None:
                    import copy as _copy

                    tracker_messages = (
                        optimized_messages
                        if optimized_messages is not None
                        else body.get("messages", [])
                    )
                    next_forwarded = _copy.deepcopy(tracker_messages)
                    next_original = _copy.deepcopy(original_messages or tracker_messages)
                    if full_sse_bytes:
                        parsed = self._parse_sse_to_response(
                            full_sse_bytes.decode("utf-8", errors="replace"), provider
                        )
                        asst_msg = self._assistant_message_from_response_json(parsed)
                        if asst_msg is not None:
                            next_forwarded.append(_copy.deepcopy(asst_msg))
                            next_original.append(_copy.deepcopy(asst_msg))
                    cache_read_tokens = stream_state["cache_read_input_tokens"] or 0
                    cache_write_tokens = stream_state["cache_creation_input_tokens"] or 0
                    if provider == "anthropic" and hasattr(prefix_tracker, "classify_cache_miss"):
                        miss = prefix_tracker.classify_cache_miss(
                            cache_read_tokens=cache_read_tokens,
                            current_forwarded_messages=tracker_messages,
                        )
                        if miss.is_miss:
                            logger.info(
                                f"[{request_id}] CACHE-MISS-ATTRIBUTION: reason={miss.reason} "
                                f"idle={miss.idle_seconds:.0f}s ttl={miss.cache_ttl_seconds}s "
                                f"expected_cached={miss.expected_cached_tokens:,} "
                                f"prefix_changed={miss.prefix_changed} "
                                f"ttl_exceeded={miss.ttl_exceeded}"
                            )
                            await self.metrics.record_cache_miss_attribution(provider, miss.reason)
                    prefix_tracker.update_from_response(
                        cache_read_tokens=cache_read_tokens,
                        cache_write_tokens=cache_write_tokens,
                        messages=next_forwarded,
                        original_messages=next_original,
                    )

                # Active-compression denominator derived inside
                # ``from_stream`` as ``optimized + saved``. Bedrock
                # doesn't propagate frozen_message_count either — same
                # fallback as the SSE finalizer (#455).
                # Bedrock forwards Anthropic-shape usage: input_tokens is the
                # uncached tail only, cache buckets are disjoint. Only a count
                # the backend itself reported qualifies; the message_start
                # backfill above is Headroom's estimate.
                _bedrock_input = stream_state.get("backend_input_tokens")
                outcome = RequestOutcome.from_stream(
                    body=body,
                    provider=_backend_name,
                    model=model,
                    request_id=request_id,
                    original_tokens=original_tokens,
                    optimized_tokens=optimized_tokens,
                    provider_input_tokens=(
                        billed_input_for_provider(
                            "anthropic",
                            _bedrock_input,
                            cache_read=stream_state["cache_read_input_tokens"],
                            cache_write=stream_state["cache_creation_input_tokens"],
                        )
                        if isinstance(_bedrock_input, int)
                        else 0
                    ),
                    output_tokens=stream_state["output_tokens"],
                    tokens_saved=tokens_saved,
                    transforms_applied=transforms_applied,
                    total_latency_ms=total_latency,
                    overhead_ms=optimization_latency,
                    tags=tags,
                    client=client,
                    status_code=_stream_outcome_status(200, completed_normally),
                    log_full_messages=getattr(self.config, "log_full_messages", False),
                    cache_read_tokens=stream_state["cache_read_input_tokens"],
                    cache_write_tokens=stream_state["cache_creation_input_tokens"],
                    cache_write_5m_tokens=stream_state["cache_creation_ephemeral_5m_input_tokens"],
                    cache_write_1h_tokens=stream_state["cache_creation_ephemeral_1h_input_tokens"],
                    ttfb_ms=stream_state["ttfb_ms"] or 0,
                    pipeline_timing=pipeline_timing,
                    original_messages=original_messages,
                )
                await self._record_request_outcome(outcome)

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
        waste_signals: dict[str, int] | None = None,
        prefix_tracker: Any | None = None,
        optimized_messages: list[dict] | None = None,
        original_messages: list[dict] | None = None,
        backend: Any | None = None,
    ) -> StreamingResponse:
        """Stream OpenAI chat completion response from backend.

        Routes stream:true requests through the backend's
        ``stream_openai_message()``, yielding SSE events to the client.
        Buffers chunk bytes into ``stream_state["sse_buffer"]`` and
        incrementally drains complete events via
        :meth:`_parse_sse_usage_from_buffer` so the final usage frame
        (LiteLLM/OpenAI emits this only when the request included
        ``stream_options.include_usage=true``) yields ``prompt_tokens``,
        ``completion_tokens``, and
        ``prompt_tokens_details.cached_tokens``. OpenAI exposes no
        separate cache-write counter, so the write portion is inferred
        via :func:`_infer_openai_cache_write_tokens`. Memory stays O(1)
        because the buffer-parser consumes whole events as they arrive.

        ``prefix_tracker``/``optimized_messages`` carry the
        :class:`PrefixCacheTracker` for the session so cache stats from
        the FINAL usage frame can update the tracker for the next turn
        — mirroring the direct streaming path
        (``_stream_response``/``_finalize_stream_response``).
        ``original_messages`` is the immutable pre-transform client
        snapshot for the same update: the tracker must record what the
        client SENT as its ``last_original_messages``, not what we
        forwarded, or next turn's overlay never matches the client
        prefix and cannot replay the cached bytes.

        NOTE: CCR request-level intercept on the streaming path is
        intentionally OUT OF SCOPE. Mirrors the Anthropic streaming
        path, which also does not buffer-and-rewrite mid-stream — doing
        so would require buffering the full response and would kill the
        streaming benefit. We do still record CCR retrieval feedback
        (cheap) for TOIN learning.
        """
        from fastapi.responses import StreamingResponse

        from headroom.proxy.handlers.openai import _infer_openai_cache_write_tokens
        from headroom.proxy.outcome import RequestOutcome

        # ``backend`` lets the caller serve this one request from somewhere
        # other than the configured backend (see proxy/route_advice.py). None
        # means "the configured one", i.e. what this method always did.
        backend = backend if backend is not None else self.anthropic_backend
        assert backend is not None
        client = classify_client(headers)

        async def generate():
            stream_state: dict[str, Any] = {
                "sse_buffer": bytearray(),
                "input_tokens": None,
                "output_tokens": None,
                "cache_read_input_tokens": None,
                "cache_creation_input_tokens": None,
            }
            # Bytes-level mirror of the SSE stream so we can parse the
            # final response shape for CCR feedback after the stream
            # closes (cheap, no buffering of in-flight chunks back to
            # the client).
            full_sse_bytes = bytearray()
            completed_normally = False
            stream_failed = False

            def _absorb(usage: dict[str, int] | None) -> None:
                if not usage:
                    return
                for key in (
                    "input_tokens",
                    "output_tokens",
                    "cache_read_input_tokens",
                    "cache_creation_input_tokens",
                ):
                    if key in usage and not stream_state.get(key):
                        stream_state[key] = usage[key]

            try:
                async for sse_chunk in backend.stream_openai_message(body, headers):
                    chunk_bytes = sse_chunk.encode() if isinstance(sse_chunk, str) else sse_chunk
                    stream_state["sse_buffer"].extend(chunk_bytes)
                    full_sse_bytes.extend(chunk_bytes)
                    _absorb(self._parse_sse_usage_from_buffer(stream_state, "openai"))
                    # Per-chunk fallback for upstreams that emit only
                    # ``completion_tokens`` and not a full usage frame.
                    parsed = _parse_completion_tokens_from_sse_chunk(chunk_bytes)
                    if parsed is not None and not stream_state["output_tokens"]:
                        stream_state["output_tokens"] = parsed
                    yield chunk_bytes
                stream_failed = stream_state.get("stream_failed", False)
                completed_normally = (
                    stream_state.get("stream_terminal", False) and not stream_failed
                )
            except Exception as e:
                logger.error(f"[{request_id}] Backend streaming error: {e}")
                error_data = public_errors.openai_error_body(
                    public_errors.classify_or_internal(e),
                    request_id=request_id,
                    error_type="api_error",
                )
                yield f"data: {json.dumps(error_data)}\n\n".encode()
                yield b"data: [DONE]\n\n"
            finally:
                # A residual buffer without an SSE terminator is an
                # incomplete event. Complete usage frames were parsed in
                # the read loop; never synthesize ``\n\n`` here, because
                # that can make partial UTF-8 look like a complete event.
                buf = stream_state["sse_buffer"]
                if len(buf) > 0:
                    buf.clear()

                # Mirror the non-streaming sibling (``_extract_responses_usage``
                # in handlers/openai.py): only infer cache metrics when
                # upstream actually reported a usage frame. Otherwise the
                # proxy-side ``optimized_tokens`` would masquerade as a
                # cache write — wrong, and indistinguishable from a real
                # hit-rate-zero call in the dashboard.
                upstream_input = stream_state["input_tokens"]
                output_tokens = stream_state["output_tokens"] or 0
                cache_read_tokens = stream_state["cache_read_input_tokens"] or 0
                # Prefer authoritative cache_creation_input_tokens from
                # Bedrock/Anthropic shape when present. Fall back to
                # inferring write count from total - read for OpenAI
                # shape (which has no separate write counter).
                cache_creation_input_tokens = stream_state.get("cache_creation_input_tokens") or 0
                if upstream_input is None:
                    cache_write_tokens = 0
                    uncached_input_tokens = 0
                    cache_inferred = False
                elif cache_creation_input_tokens > 0:
                    cache_write_tokens = cache_creation_input_tokens
                    uncached_input_tokens = max(
                        upstream_input - cache_read_tokens - cache_write_tokens, 0
                    )
                    cache_inferred = False
                else:
                    cache_write_tokens = _infer_openai_cache_write_tokens(
                        upstream_input, cache_read_tokens
                    )
                    uncached_input_tokens = max(upstream_input - cache_read_tokens, 0)
                    cache_inferred = True

                # Update prefix cache tracker for the next turn — mirrors
                # the non-streaming sibling. Done before outcome funnel
                # so prefix state is consistent regardless of metric
                # path.
                if completed_normally and prefix_tracker is not None:
                    tracker_messages = (
                        optimized_messages
                        if optimized_messages is not None
                        else body.get("messages", [])
                    )
                    prefix_tracker.update_from_response(
                        cache_read_tokens=cache_read_tokens,
                        cache_write_tokens=cache_write_tokens,
                        messages=tracker_messages,
                        original_messages=original_messages,
                    )

                # CCR Feedback: record headroom_retrieve tool calls so
                # TOIN learns which fields matter. Streaming path can't
                # do request-level intercept (would require buffering
                # the full stream), so we just close the feedback loop.
                if self.config.ccr_inject_tool and len(full_sse_bytes) > 0:
                    try:
                        full_sse_data = full_sse_bytes.decode("utf-8", errors="replace")
                        self._record_ccr_feedback_from_openai_sse(full_sse_data, request_id)
                    except Exception as e:
                        logger.debug(
                            f"[{request_id}] CCR feedback recording (openai stream) failed: {e}"
                        )

                total_latency = (time.time() - start_time) * 1000
                # Active-compression denominator for backend-routed
                # streaming. No per-message live-zone tracking is wired
                # for this path yet (see the non-streaming sibling in
                # openai.py for the same caveat), so use the full pre-
                # comp request size. This keeps active_savings_percent
                # in sync with proxy_savings_percent for this provider
                # instead of collapsing the dashboard headline to 0%.
                outcome = RequestOutcome.from_stream(
                    body=body,
                    provider=backend.name,
                    model=model,
                    request_id=request_id,
                    original_tokens=original_tokens,
                    optimized_tokens=optimized_tokens,
                    # OpenAI-shape usage from the backend: the headline figure
                    # already includes cached tokens (cache_write above is
                    # subtracted from it, not added).
                    provider_input_tokens=(
                        upstream_input if isinstance(upstream_input, int) else 0
                    ),
                    output_tokens=output_tokens,
                    tokens_saved=tokens_saved,
                    transforms_applied=transforms_applied,
                    total_latency_ms=total_latency,
                    overhead_ms=optimization_latency,
                    tags=tags,
                    client=client,
                    status_code=_stream_outcome_status(200, completed_normally),
                    log_full_messages=getattr(self.config, "log_full_messages", False),
                    cache_read_tokens=cache_read_tokens,
                    cache_write_tokens=cache_write_tokens,
                    uncached_input_tokens=uncached_input_tokens,
                    cache_inferred=cache_inferred,
                    pipeline_timing=pipeline_timing,
                    waste_signals=waste_signals,
                )
                await self._record_request_outcome(outcome)

                if tokens_saved > 0:
                    logger.info(
                        f"[{request_id}] {model}: {original_tokens:,} → {optimized_tokens:,} "
                        f"(saved {tokens_saved:,} tokens) via {backend.name} [stream]"
                    )

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
        )
