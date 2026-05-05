"""OpenAI handler mixin for HeadroomProxy.

Contains all OpenAI Chat Completions, Responses API, and passthrough handlers.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import copy
import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from headroom.proxy.helpers import jitter_delay_ms
from headroom.proxy.stage_timer import StageTimer, emit_stage_timings_log
from headroom.proxy.ws_session_registry import (
    TerminationCause,
    WebSocketSessionRegistry,
    WSSessionHandle,
)

if TYPE_CHECKING:
    from fastapi import Request, WebSocket
    from fastapi.responses import JSONResponse, Response, StreamingResponse

import httpx

from headroom.copilot_auth import apply_copilot_api_auth, build_copilot_upstream_url
from headroom.pipeline import PipelineStage, summarize_routing_markers
from headroom.proxy.auth_mode import classify_auth_mode

logger = logging.getLogger("headroom.proxy")


# Interactive Responses turns are latency-sensitive. Fail open quickly rather
# than holding the session hostage on memory lookup.
RESPONSES_CONTEXT_SEARCH_TIMEOUT_SECONDS = 2.0

# Cap the wait for the first client frame after the WS handshake completes.
# A zombie or malicious client that accepts the upgrade but never sends the
# first response.create frame would otherwise hold a slot indefinitely and
# starve the session registry. 60 s is generous for real clients (Codex
# typically sends the first frame within a few hundred milliseconds of the
# accept) but short enough to bound the damage from a hung peer.
WS_FIRST_FRAME_TIMEOUT_SECONDS = 60.0


def _decode_openai_bearer_payload(headers: dict[str, str]) -> dict[str, Any] | None:
    """Best-effort decode of an OpenAI OAuth bearer token payload.

    OpenClaw's Codex OAuth flow may forward only the bearer token after the
    provider base URL is overridden. In that case the explicit
    ``ChatGPT-Account-ID`` header can be missing even though the JWT still
    carries the account id we need to route to the ChatGPT Codex backend.
    """
    auth = headers.get("authorization") or headers.get("Authorization")
    if not auth:
        return None

    scheme, _, token = auth.partition(" ")
    if scheme.lower() != "bearer" or token.count(".") < 2:
        return None

    payload = token.split(".", 2)[1]
    payload += "=" * (-len(payload) % 4)
    # Intentionally no signature verification here: this is only a best-effort
    # routing hint extractor. Upstream still performs the actual auth/authz checks.
    try:
        decoded = base64.urlsafe_b64decode(payload.encode("ascii"))
        data = json.loads(decoded.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None

    return data if isinstance(data, dict) else None


def _resolve_codex_routing_headers(headers: dict[str, str]) -> tuple[dict[str, str], bool]:
    """Resolve ChatGPT Codex routing hints from explicit headers or OAuth JWT."""
    resolved = dict(headers)
    lower_lookup = {k.lower(): k for k in resolved}

    if "chatgpt-account-id" in lower_lookup:
        return resolved, True

    payload = _decode_openai_bearer_payload(resolved)
    auth_claims = payload.get("https://api.openai.com/auth") if isinstance(payload, dict) else None
    account_id = auth_claims.get("chatgpt_account_id") if isinstance(auth_claims, dict) else None
    if isinstance(account_id, str) and account_id.strip():
        resolved["ChatGPT-Account-ID"] = account_id.strip()
        return resolved, True

    return resolved, False


class OpenAIHandlerMixin:
    """Mixin providing OpenAI API handler methods for HeadroomProxy."""

    @staticmethod
    def _strict_previous_turn_frozen_count(
        messages: list[dict[str, Any]],
        base_frozen_count: int,
    ) -> int:
        """Freeze all prior turns in cache mode; only final user turn is mutable."""
        if not messages:
            return base_frozen_count
        final_idx = len(messages) - 1
        if messages[final_idx].get("role") == "user":
            return max(base_frozen_count, final_idx)
        return len(messages)

    @staticmethod
    def _restore_frozen_prefix(
        original_messages: list[dict[str, Any]],
        candidate_messages: list[dict[str, Any]],
        *,
        frozen_message_count: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """Force frozen prefix bytes to match original request exactly."""
        if frozen_message_count <= 0 or not original_messages:
            return candidate_messages, 0

        frozen = min(frozen_message_count, len(original_messages))
        restored = list(candidate_messages)

        if len(restored) < frozen:
            return list(original_messages[:frozen]) + restored, frozen

        changed = 0
        for idx in range(frozen):
            if restored[idx] != original_messages[idx]:
                restored[idx] = original_messages[idx]
                changed += 1
        return restored, changed

    async def handle_openai_chat(
        self,
        request: Request,
    ) -> Response | StreamingResponse:
        """Handle OpenAI /v1/chat/completions endpoint."""
        if not hasattr(self, "pipeline_extensions"):
            from headroom.pipeline import PipelineExtensionManager

            self.pipeline_extensions = PipelineExtensionManager(discover=False)

        from fastapi import HTTPException
        from fastapi.responses import JSONResponse, Response

        from headroom.ccr import CCRToolInjector
        from headroom.proxy.helpers import (
            COMPRESSION_TIMEOUT_SECONDS,
            MAX_MESSAGE_ARRAY_LENGTH,
            MAX_REQUEST_BODY_SIZE,
            _read_request_json,
        )
        from headroom.proxy.modes import is_cache_mode, is_token_mode
        from headroom.tokenizers import get_tokenizer
        from headroom.utils import extract_user_query

        start_time = time.time()
        request_id = await self._next_request_id()

        # Phase F PR-F1: classify auth mode at request entry. The result
        # is stored on `request.state` so downstream handlers (cache
        # gates, header injection, lossy-compressor gates) read it
        # without re-classifying. Pure function, well under 10us.
        auth_mode = classify_auth_mode(request.headers)
        request.state.auth_mode = auth_mode
        logger.debug(f"[{request_id}] auth_mode_classified mode={auth_mode.value}")

        # Check request body size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_REQUEST_BODY_SIZE:
            return JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "message": f"Request body too large. Maximum size is {MAX_REQUEST_BODY_SIZE // (1024 * 1024)}MB",
                        "type": "invalid_request_error",
                        "code": "request_too_large",
                    }
                },
            )

        # Parse request
        try:
            body = await _read_request_json(request)
        except (json.JSONDecodeError, ValueError) as e:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": f"Invalid request body: {e!s}",
                        "type": "invalid_request_error",
                        "code": "invalid_json",
                    }
                },
            )
        model = body.get("model", "unknown")
        messages = body.get("messages", [])
        original_client_messages = copy.deepcopy(messages)
        input_event = self.pipeline_extensions.emit(
            PipelineStage.INPUT_RECEIVED,
            operation="proxy.request",
            request_id=request_id,
            provider="openai",
            model=model,
            messages=messages,
            tools=body.get("tools"),
            metadata={"path": "/v1/chat/completions", "stream": body.get("stream", False)},
        )
        if input_event.messages is not None:
            messages = input_event.messages
            original_client_messages = copy.deepcopy(messages)
        if input_event.tools is not None:
            body["tools"] = input_event.tools

        # Validate message array size
        if len(messages) > MAX_MESSAGE_ARRAY_LENGTH:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": f"Message array too large ({len(messages)} messages). "
                        f"Maximum is {MAX_MESSAGE_ARRAY_LENGTH}.",
                        "type": "invalid_request_error",
                        "code": "invalid_request",
                    }
                },
            )

        stream = body.get("stream", False)

        # Bypass: skip ALL compression for explicit opt-out
        _bypass = (
            request.headers.get("x-headroom-bypass", "").lower() == "true"
            or request.headers.get("x-headroom-mode", "").lower() == "passthrough"
        )
        if _bypass:
            logger.info(f"[{request_id}] Bypass: skipping compression (header)")

        # Image compression: tile alignment + ML-based technique routing
        if self.config.image_optimize and messages and not _bypass:
            from headroom.proxy.helpers import _get_image_compressor

            compressor = None
            try:
                compressor = _get_image_compressor()
                if compressor and compressor.has_images(messages):
                    messages = compressor.compress(messages, provider="openai")
                    if compressor.last_result:
                        logger.info(
                            f"[{request_id}] Image: {compressor.last_result.technique.value} "
                            f"({compressor.last_result.savings_percent:.0f}% saved, "
                            f"{compressor.last_result.original_tokens} → "
                            f"{compressor.last_result.compressed_tokens} tokens)"
                        )
            finally:
                if compressor and hasattr(compressor, "close"):
                    compressor.close()

        headers = dict(request.headers.items())
        headers.pop("host", None)
        headers.pop("content-length", None)
        # Strip accept-encoding so httpx negotiates its own encoding.
        # Cloudflare Workers forward "br, zstd" which OpenAI may honor;
        # if httpx lacks brotli support the response body is undecipherable → 502.
        headers.pop("accept-encoding", None)
        tags = self._extract_tags(headers)
        # PR-A5 (P5-49): strip internal x-headroom-* from upstream-bound
        # headers AFTER `_extract_tags` reads them. Inbound bypass gating
        # uses `request.headers.get(...)` above; memory user-id reads
        # `request.headers` below. From this point on, `headers` is the
        # upstream-bound copy.
        from headroom.proxy.helpers import _strip_internal_headers, log_outbound_headers

        _pre_strip_count_chat = sum(1 for k in headers if k.lower().startswith("x-headroom-"))
        headers = _strip_internal_headers(headers)
        log_outbound_headers(
            forwarder="openai_chat_completions",
            stripped_count=_pre_strip_count_chat,
            request_id=request_id,
        )

        # Memory: Get user ID when memory is enabled. Reads `request.headers`
        # directly because `headers` was stripped of `x-headroom-*` for the
        # upstream-bound copy (PR-A5).
        memory_user_id: str | None = None
        if self.memory_handler:
            memory_user_id = request.headers.get(
                "x-headroom-user-id",
                os.environ.get("USER", os.environ.get("USERNAME", "default")),
            )

        # Rate limiting
        if self.rate_limiter:
            rate_key = headers.get("authorization", "default")[:20]
            allowed, wait_seconds = await self.rate_limiter.check_request(rate_key)
            if not allowed:
                await self.metrics.record_rate_limited(provider="openai")
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limited. Retry after {wait_seconds:.1f}s",
                )

        # Check cache
        if self.cache and not stream:
            cached = await self.cache.get(messages, model)
            if cached:
                self.pipeline_extensions.emit(
                    PipelineStage.INPUT_CACHED,
                    operation="proxy.request",
                    request_id=request_id,
                    provider="openai",
                    model=model,
                    messages=messages,
                    metadata={"cache_hit": True, "path": "/v1/chat/completions"},
                )
                await self.metrics.record_request(
                    provider="openai",
                    model=model,
                    input_tokens=0,
                    output_tokens=0,
                    tokens_saved=0,  # Savings already counted when response was cached
                    latency_ms=(time.time() - start_time) * 1000,
                    cached=True,
                )

                # Remove compression headers from cached response
                response_headers = dict(cached.response_headers)
                response_headers.pop("content-encoding", None)
                response_headers.pop("content-length", None)

                return Response(content=cached.response_body, headers=response_headers)

        # Token counting
        tokenizer = get_tokenizer(model)
        original_tokens = tokenizer.count_messages(messages)

        # Hook: pre_compress
        _hook_biases = None
        if self.config.hooks:
            from headroom.hooks import CompressContext

            _hook_ctx = CompressContext(model=model, provider="openai")
            try:
                messages = self.config.hooks.pre_compress(messages, _hook_ctx)
                _hook_biases = self.config.hooks.compute_biases(messages, _hook_ctx)
            except Exception as e:
                logger.debug(f"[{request_id}] Hook error: {e}")

        # Optimization
        transforms_applied = []
        pipeline_timing: dict[str, float] = {}
        waste_signals_dict: dict[str, int] | None = None
        optimized_messages = messages
        optimized_tokens = original_tokens

        # Get prefix cache tracker for this session
        openai_session_id = self.session_tracker_store.compute_session_id(request, model, messages)
        openai_prefix_tracker = self.session_tracker_store.get_or_create(
            openai_session_id, "openai"
        )

        # PR-A6 (P5-50, preps P0-6): session-sticky `OpenAI-Beta` merge.
        # Same pattern as anthropic.py — read client value, union with
        # session-seen tokens, update tracker. WS auto-injection of
        # `responses_websockets=2026-02-06` lives on the WS handler;
        # chat-completions has no Headroom-required tokens today, so the
        # merge effectively just makes the client value byte-stable
        # across turns.
        from headroom.proxy.helpers import (
            get_session_beta_tracker as _get_session_beta_tracker_chat,
        )
        from headroom.proxy.helpers import (
            log_beta_header_merge as _log_beta_header_merge_chat,
        )

        _client_openai_beta = headers.get("openai-beta")
        _client_openai_beta_count = (
            len([t for t in (_client_openai_beta or "").split(",") if t.strip()])
            if _client_openai_beta
            else 0
        )
        _sticky_openai_beta = _get_session_beta_tracker_chat().record_and_get_sticky_betas(
            provider="openai",
            session_id=openai_session_id,
            client_value=_client_openai_beta,
        )
        _sticky_openai_beta_count = (
            len([t for t in _sticky_openai_beta.split(",") if t.strip()])
            if _sticky_openai_beta
            else 0
        )
        if _sticky_openai_beta and _sticky_openai_beta != (_client_openai_beta or ""):
            headers["openai-beta"] = _sticky_openai_beta
        _log_beta_header_merge_chat(
            provider="openai",
            session_id=openai_session_id,
            client_betas_count=_client_openai_beta_count,
            sticky_betas_count=_sticky_openai_beta_count,
            headroom_added=[],
            request_id=request_id,
        )

        openai_frozen_count = openai_prefix_tracker.get_frozen_message_count()
        if is_cache_mode(self.config.mode):
            openai_frozen_count = self._strict_previous_turn_frozen_count(
                original_client_messages,
                openai_frozen_count,
            )

        _compression_failed = False
        original_messages = messages  # Preserve for 400-retry fallback
        _license_ok = self.usage_reporter.should_compress if self.usage_reporter else True
        if self.config.optimize and messages and not _bypass and _license_ok:
            try:
                context_limit = self.openai_provider.get_context_limit(model)

                if is_token_mode(self.config.mode):
                    comp_cache = self._get_compression_cache(openai_session_id)

                    # Zone 1: Swap cached compressed versions
                    working_messages = comp_cache.apply_cached(messages)

                    # Re-freeze boundary
                    openai_frozen_count = comp_cache.compute_frozen_count(messages)

                    result = await self._run_compression_in_executor(
                        lambda: self.openai_pipeline.apply(
                            messages=working_messages,
                            model=model,
                            model_limit=context_limit,
                            context=extract_user_query(working_messages),
                            frozen_message_count=openai_frozen_count,
                            biases=_hook_biases,
                        ),
                        timeout=COMPRESSION_TIMEOUT_SECONDS,
                    )

                    if result.messages != working_messages:
                        comp_cache.update_from_result(messages, result.messages)

                    # Always use pipeline result in token mode
                    optimized_messages = result.messages
                    transforms_applied = result.transforms_applied
                    pipeline_timing = result.timing
                    # Keep original_tokens as the REAL original (pre-Zone-1-swap)
                    # so tokens_saved captures both Zone 1 + Zone 2 savings.
                    optimized_tokens = result.tokens_after
                else:
                    result = await self._run_compression_in_executor(
                        lambda: self.openai_pipeline.apply(
                            messages=messages,
                            model=model,
                            model_limit=context_limit,
                            context=extract_user_query(messages),
                            frozen_message_count=openai_frozen_count,
                            biases=_hook_biases,
                        ),
                        timeout=COMPRESSION_TIMEOUT_SECONDS,
                    )

                    if result.messages != messages:
                        optimized_messages = result.messages
                        transforms_applied = result.transforms_applied
                        pipeline_timing = result.timing
                        original_tokens = result.tokens_before
                        optimized_tokens = result.tokens_after

                if result.waste_signals:
                    waste_signals_dict = result.waste_signals.to_dict()
            except Exception as e:
                logger.warning(f"Optimization failed: {e}")
                # Flag compression failure for observability
                _compression_failed = True

        # Guard: if "optimization" inflated tokens, revert to originals
        if optimized_tokens > original_tokens:
            logger.warning(
                f"[{request_id}] Optimization inflated tokens "
                f"({original_tokens} -> {optimized_tokens}), reverting to original messages"
            )
            optimized_messages = original_messages
            optimized_tokens = original_tokens
            transforms_applied = []

        tokens_saved = original_tokens - optimized_tokens
        optimization_latency = (time.time() - start_time) * 1000

        routing_markers = summarize_routing_markers(transforms_applied)
        if routing_markers:
            routed_event = self.pipeline_extensions.emit(
                PipelineStage.INPUT_ROUTED,
                operation="proxy.request",
                request_id=request_id,
                provider="openai",
                model=model,
                messages=optimized_messages,
                metadata={
                    "routing_markers": routing_markers,
                    "transforms_applied": transforms_applied,
                },
            )
            if routed_event.messages is not None:
                optimized_messages = routed_event.messages
                optimized_tokens = tokenizer.count_messages(optimized_messages)
                tokens_saved = original_tokens - optimized_tokens

        compressed_event = self.pipeline_extensions.emit(
            PipelineStage.INPUT_COMPRESSED,
            operation="proxy.request",
            request_id=request_id,
            provider="openai",
            model=model,
            messages=optimized_messages,
            metadata={
                "tokens_before": original_tokens,
                "tokens_after": optimized_tokens,
                "transforms_applied": transforms_applied,
            },
        )
        if compressed_event.messages is not None:
            optimized_messages = compressed_event.messages
            optimized_tokens = tokenizer.count_messages(optimized_messages)
            tokens_saved = original_tokens - optimized_tokens

        # Hook: post_compress
        if self.config.hooks and tokens_saved > 0:
            from headroom.hooks import CompressEvent

            try:
                self.config.hooks.post_compress(
                    CompressEvent(
                        tokens_before=original_tokens,
                        tokens_after=optimized_tokens,
                        tokens_saved=tokens_saved,
                        compression_ratio=tokens_saved / original_tokens
                        if original_tokens > 0
                        else 0,
                        transforms_applied=transforms_applied,
                        model=model,
                        provider="openai",
                    )
                )
            except Exception as e:
                logger.debug(f"[{request_id}] post_compress hook error: {e}")

        # CCR Tool Injection: Inject retrieval tool if compression occurred
        # OR if this session has previously done CCR (PR-B7 sticky-on).
        # See `headroom/proxy/handlers/anthropic.py` and PR-B7 plan
        # `REALIGNMENT/04-phase-B-live-zone.md` for the rationale: once a
        # session has done CCR, the `headroom_retrieve` tool stays
        # registered for every subsequent turn so the prompt cache
        # anchored on the previous turn's tool list never busts.
        tools = body.get("tools")
        _original_tools = tools  # Preserve for diagnostic / future retry
        if (
            self.config.ccr_inject_tool or self.config.ccr_inject_system_instructions
        ) and not _bypass:
            injector = CCRToolInjector(
                provider="openai",
                inject_tool=False,  # routed through sticky helper below
                inject_system_instructions=self.config.ccr_inject_system_instructions,
            )
            injector.scan_for_markers(optimized_messages)
            if self.config.ccr_inject_system_instructions and injector.has_compressed_content:
                optimized_messages = injector.inject_into_system_message(optimized_messages)

            if self.config.ccr_inject_tool:
                from headroom.proxy.helpers import apply_session_sticky_ccr_tool

                tools, ccr_tool_injected = apply_session_sticky_ccr_tool(
                    provider="openai",
                    session_id=openai_session_id,
                    request_id=request_id,
                    existing_tools=tools,
                    has_compressed_content_this_turn=injector.has_compressed_content,
                )
                if ccr_tool_injected:
                    logger.debug(
                        f"[{request_id}] CCR: tool registered (session={openai_session_id}, "
                        f"compressed_this_turn={injector.has_compressed_content}, "
                        f"hashes_seen={len(injector.detected_hashes)})"
                    )

        if is_cache_mode(self.config.mode):
            optimized_messages, restored_count = self._restore_frozen_prefix(
                original_client_messages,
                optimized_messages,
                frozen_message_count=openai_frozen_count,
            )
            if restored_count > 0:
                logger.warning(
                    f"[{request_id}] Restored {restored_count} frozen prefix message(s) "
                    "to preserve cache stability (openai)"
                )

        # Memory: inject context and tools for OpenAI requests.
        #
        # PR-A3 follow-up to A2: memory context now routes exclusively to
        # the live-zone tail (latest user message), never via a system-level
        # prepend. The cache hot zone (system messages) is sacrosanct —
        # invariant I2. See REALIGNMENT/03-phase-A-lockdown.md PR-A3.
        memory_context_injected = False
        memory_tools_injected = False
        if self.memory_handler and memory_user_id:
            try:
                if self.memory_handler.config.inject_context:
                    memory_context = await self.memory_handler.search_and_format_context(
                        memory_user_id, optimized_messages
                    )
                    if memory_context:
                        from headroom.proxy.helpers import (
                            append_text_to_latest_user_chat_message,
                            get_memory_injection_mode,
                            log_memory_injection,
                        )

                        injection_mode = get_memory_injection_mode()
                        if injection_mode == "disabled":
                            log_memory_injection(
                                request_id=request_id,
                                session_id=None,
                                decision="skipped_disabled",
                                bytes_injected=0,
                                query=None,
                            )
                        else:
                            new_messages, bytes_appended = append_text_to_latest_user_chat_message(
                                optimized_messages, memory_context
                            )
                            if bytes_appended > 0:
                                optimized_messages = new_messages
                                memory_context_injected = True
                                log_memory_injection(
                                    request_id=request_id,
                                    session_id=None,
                                    decision="injected_live_zone_tail_chat",
                                    bytes_injected=bytes_appended,
                                    query=None,
                                )
                                logger.info(
                                    f"[{request_id}] Memory: Injected {bytes_appended} chars "
                                    f"into latest user message tail for user {memory_user_id}"
                                )
                            else:
                                log_memory_injection(
                                    request_id=request_id,
                                    session_id=None,
                                    decision="no_eligible_user_message",
                                    bytes_injected=0,
                                    query=None,
                                )

                # Inject memory tools — PR-A7 (P0-6) routes through
                # `apply_session_sticky_memory_tools` so byte-stable across turns.
                from headroom.proxy.helpers import (
                    apply_session_sticky_memory_tools as _apply_sticky_mem_tools,
                )

                memory_tool_defs = (
                    self.memory_handler.compute_memory_tool_definitions("openai")
                    if self.memory_handler.config.inject_tools
                    else []
                )
                tools, mem_tools_injected = _apply_sticky_mem_tools(
                    provider="openai",
                    session_id=openai_session_id,
                    request_id=request_id,
                    existing_tools=tools,
                    memory_tools_to_inject=memory_tool_defs,
                    inject_this_turn=bool(self.memory_handler.config.inject_tools),
                )
                if mem_tools_injected:
                    memory_tools_injected = True
                    logger.info(f"[{request_id}] Memory: Injected memory tools (openai)")
            except Exception as e:
                logger.warning(f"[{request_id}] Memory injection failed: {e}")

        if memory_context_injected or memory_tools_injected:
            remembered_event = self.pipeline_extensions.emit(
                PipelineStage.INPUT_REMEMBERED,
                operation="proxy.request",
                request_id=request_id,
                provider="openai",
                model=model,
                messages=optimized_messages,
                tools=tools,
                metadata={
                    "memory_context_injected": memory_context_injected,
                    "memory_tools_injected": memory_tools_injected,
                },
            )
            if remembered_event.messages is not None:
                optimized_messages = remembered_event.messages
            if remembered_event.tools is not None:
                tools = remembered_event.tools

        body["messages"] = optimized_messages
        if tools is not None:
            body["tools"] = tools

        presend_event = self.pipeline_extensions.emit(
            PipelineStage.PRE_SEND,
            operation="proxy.request",
            request_id=request_id,
            provider="openai",
            model=model,
            messages=optimized_messages,
            tools=tools,
            headers=headers,
            metadata={"path": "/v1/chat/completions", "stream": stream},
        )
        if presend_event.messages is not None:
            optimized_messages = presend_event.messages
            body["messages"] = optimized_messages
        if presend_event.tools is not None:
            tools = presend_event.tools
            body["tools"] = tools
        if presend_event.headers is not None:
            headers = presend_event.headers
        optimized_tokens = tokenizer.count_messages(body["messages"])
        tokens_saved = original_tokens - optimized_tokens

        # Route through LiteLLM/any-llm backend if configured
        if self.anthropic_backend is not None:
            try:
                if stream:
                    self.pipeline_extensions.emit(
                        PipelineStage.POST_SEND,
                        operation="proxy.request",
                        request_id=request_id,
                        provider="openai",
                        model=model,
                        messages=body["messages"],
                        tools=tools,
                        metadata={"path": "/v1/chat/completions", "stream": True},
                    )
                    # Streaming: use stream_openai_message() → SSE events
                    return await self._stream_openai_via_backend(
                        body,
                        headers,
                        model,
                        request_id,
                        start_time,
                        original_tokens,
                        optimized_tokens,
                        tokens_saved,
                        transforms_applied,
                        tags,
                        optimization_latency,
                        pipeline_timing=pipeline_timing,
                    )
                else:
                    # Non-streaming: use send_openai_message() → JSON
                    backend_response = await self.anthropic_backend.send_openai_message(
                        body, headers
                    )
                    self.pipeline_extensions.emit(
                        PipelineStage.POST_SEND,
                        operation="proxy.request",
                        request_id=request_id,
                        provider="openai",
                        model=model,
                        messages=body["messages"],
                        tools=tools,
                        response=backend_response.body,
                        metadata={
                            "path": "/v1/chat/completions",
                            "stream": False,
                            "status_code": backend_response.status_code,
                        },
                    )
                    self.pipeline_extensions.emit(
                        PipelineStage.RESPONSE_RECEIVED,
                        operation="proxy.request",
                        request_id=request_id,
                        provider="openai",
                        model=model,
                        response=backend_response.body,
                        metadata={
                            "path": "/v1/chat/completions",
                            "stream": False,
                            "status_code": backend_response.status_code,
                        },
                    )

                    if backend_response.error:
                        return JSONResponse(
                            status_code=backend_response.status_code,
                            content=backend_response.body,
                        )

                    # Track metrics
                    total_latency = (time.time() - start_time) * 1000
                    usage = backend_response.body.get("usage", {})
                    output_tokens = usage.get("completion_tokens", 0)
                    total_input_tokens = usage.get("prompt_tokens", optimized_tokens)

                    await self.metrics.record_request(
                        provider=self.anthropic_backend.name,
                        model=model,
                        input_tokens=total_input_tokens,
                        output_tokens=output_tokens,
                        tokens_saved=tokens_saved,
                        latency_ms=total_latency,
                        cached=False,
                        overhead_ms=optimization_latency,
                        pipeline_timing=pipeline_timing,
                    )

                    if tokens_saved > 0:
                        logger.info(
                            f"[{request_id}] {model}: {original_tokens:,} → {optimized_tokens:,} "
                            f"(saved {tokens_saved:,} tokens) via {self.anthropic_backend.name}"
                        )

                    return JSONResponse(
                        status_code=backend_response.status_code,
                        content=backend_response.body,
                    )
            except Exception as e:
                logger.error(f"[{request_id}] Backend error: {e}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": {
                            "message": str(e),
                            "type": "api_error",
                            "code": "backend_error",
                        }
                    },
                )

        # Direct OpenAI API (no backend configured)
        url = build_copilot_upstream_url(self.OPENAI_API_URL, "/v1/chat/completions")

        try:
            if stream:
                # Inject stream_options to get usage stats in streaming response
                # This allows accurate token counting instead of byte-based estimation
                if "stream_options" not in body:
                    body["stream_options"] = {"include_usage": True}
                elif isinstance(body.get("stream_options"), dict):
                    body["stream_options"]["include_usage"] = True

                self.pipeline_extensions.emit(
                    PipelineStage.POST_SEND,
                    operation="proxy.request",
                    request_id=request_id,
                    provider="openai",
                    model=model,
                    messages=body["messages"],
                    tools=tools,
                    metadata={"path": "/v1/chat/completions", "stream": True},
                )
                return await self._stream_response(
                    url,
                    headers,
                    body,
                    "openai",
                    model,
                    request_id,
                    original_tokens,
                    optimized_tokens,
                    tokens_saved,
                    transforms_applied,
                    tags,
                    optimization_latency,
                    pipeline_timing=pipeline_timing,
                    prefix_tracker=openai_prefix_tracker,
                )
            else:
                headers = await apply_copilot_api_auth(headers, url=url)
                response = await self._retry_request("POST", url, headers, body)
                self.pipeline_extensions.emit(
                    PipelineStage.POST_SEND,
                    operation="proxy.request",
                    request_id=request_id,
                    provider="openai",
                    model=model,
                    messages=body["messages"],
                    tools=tools,
                    response=response,
                    metadata={
                        "path": "/v1/chat/completions",
                        "stream": False,
                        "status_code": response.status_code,
                    },
                )
                self.pipeline_extensions.emit(
                    PipelineStage.RESPONSE_RECEIVED,
                    operation="proxy.request",
                    request_id=request_id,
                    provider="openai",
                    model=model,
                    response=response,
                    metadata={
                        "path": "/v1/chat/completions",
                        "stream": False,
                        "status_code": response.status_code,
                    },
                )

                # Full diagnostic dump on upstream errors (OpenAI handler)
                if response.status_code >= 400:
                    try:
                        err_body = response.json()
                        err_msg = err_body.get("error", {}).get("message", "")
                        err_type = err_body.get("error", {}).get("type", "")
                    except Exception:
                        err_body = {"raw": response.text[:2000]}
                        err_msg = str(response.text[:500])
                        err_type = "parse_error"

                    logger.warning(
                        f"[{request_id}] UPSTREAM_ERROR "
                        f"status={response.status_code} "
                        f"error_type={err_type} "
                        f"error_msg={err_msg!r} "
                        f"model={model} "
                        f"compressed={'yes' if transforms_applied else 'no'} "
                        f"transforms={transforms_applied} "
                        f"original_tokens={original_tokens} "
                        f"optimized_tokens={optimized_tokens} "
                        f"message_count={len(body.get('messages', []))} "
                        f"stream={stream}"
                    )

                    try:
                        from headroom import paths as _hr_paths

                        debug_dir = _hr_paths.debug_400_dir()
                        debug_dir.mkdir(parents=True, exist_ok=True)
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        debug_file = debug_dir / f"{ts}_{request_id}.json"

                        safe_headers = {}
                        for k, v in headers.items():
                            if k.lower() in ("x-api-key", "authorization"):
                                safe_headers[k] = v[:12] + "..." if v else ""
                            else:
                                safe_headers[k] = v

                        debug_payload = {
                            "request_id": request_id,
                            "timestamp": datetime.now().isoformat(),
                            "status_code": response.status_code,
                            "error_response": err_body,
                            "model": model,
                            "stream": stream,
                            "headers": safe_headers,
                            "compression": {
                                "was_compressed": bool(transforms_applied),
                                "transforms": transforms_applied,
                                "original_tokens": original_tokens,
                                "optimized_tokens": optimized_tokens,
                                "tokens_saved": tokens_saved,
                                "compression_failed": _compression_failed,
                            },
                            "tools_sent": body.get("tools"),
                            "tool_count": len(body.get("tools") or []),
                            "original_tool_count": len(_original_tools or []),
                            "messages_sent": body.get("messages"),
                            "message_count": len(body.get("messages", [])),
                            "original_messages": (
                                original_messages
                                if original_messages is not body.get("messages")
                                else "__same_as_sent__"
                            ),
                            "original_message_count": len(original_messages),
                            "system_prompt": body.get("system"),
                        }

                        with open(debug_file, "w") as f:
                            json.dump(debug_payload, f, indent=2, default=str)

                        logger.warning(f"[{request_id}] Full debug dump: {debug_file}")
                    except Exception as dump_err:
                        logger.error(f"[{request_id}] Failed to write debug dump: {dump_err}")

                total_latency = (time.time() - start_time) * 1000

                total_input_tokens = optimized_tokens  # fallback
                output_tokens = 0
                cache_read_tokens = 0
                resp_json = None
                try:
                    resp_json = response.json()
                    usage = resp_json.get("usage", {})
                    total_input_tokens = usage.get("prompt_tokens", optimized_tokens)
                    output_tokens = usage.get("completion_tokens", 0)
                    # OpenAI returns cached_tokens in prompt_tokens_details
                    # These are charged at 50% of the input price
                    prompt_details = usage.get("prompt_tokens_details") or {}
                    cache_read_tokens = prompt_details.get("cached_tokens", 0)
                except (KeyError, TypeError, AttributeError) as e:
                    logger.debug(
                        f"[{request_id}] Failed to extract cached tokens from OpenAI response: {e}"
                    )

                # Update prefix cache tracker for next turn
                openai_prefix_tracker.update_from_response(
                    cache_read_tokens=cache_read_tokens,
                    cache_write_tokens=0,  # OpenAI doesn't report write tokens
                    messages=optimized_messages,
                )

                # OpenAI has no write penalty — uncached = total - cached
                uncached_input_tokens = max(0, total_input_tokens - cache_read_tokens)

                if self.cost_tracker:
                    self.cost_tracker.record_tokens(
                        model,
                        tokens_saved,
                        optimized_tokens,
                        cache_read_tokens=cache_read_tokens,
                        uncached_tokens=uncached_input_tokens,
                    )

                # Memory: handle memory tool calls in OpenAI Chat Completions response.
                # After executing tools, send a continuation request so the model
                # can produce a final user-facing response (not just tool_calls).
                if (
                    self.memory_handler
                    and memory_user_id
                    and resp_json
                    and response.status_code == 200
                    and self.memory_handler.has_memory_tool_calls(resp_json, "openai")
                ):
                    try:
                        tool_results = await self.memory_handler.handle_memory_tool_calls(
                            resp_json, memory_user_id, "openai"
                        )
                        if tool_results:
                            # Build continuation: original messages + assistant tool_calls + tool results
                            assistant_msg = resp_json.get("choices", [{}])[0].get("message", {})
                            continuation_messages = list(optimized_messages)
                            continuation_messages.append(assistant_msg)
                            continuation_messages.extend(tool_results)

                            continuation_body = {
                                **body,
                                "messages": continuation_messages,
                            }

                            cont_response = await self._retry_request(
                                "POST", url, headers, continuation_body
                            )
                            if cont_response.status_code == 200:
                                resp_json = cont_response.json()
                                response = cont_response

                            logger.info(
                                f"[{request_id}] Memory: Handled {len(tool_results)} "
                                f"tool call(s) with continuation for user {memory_user_id}"
                            )
                    except Exception as e:
                        logger.warning(f"[{request_id}] Memory tool handling failed: {e}")

                # Cache
                if self.cache and response.status_code == 200:
                    await self.cache.set(
                        messages,
                        model,
                        response.content,
                        dict(response.headers),
                        tokens_saved,
                    )

                # Capture Codex rate-limit window data from response headers
                from headroom.subscription.codex_rate_limits import (
                    get_codex_rate_limit_state,
                )

                get_codex_rate_limit_state().update_from_headers(dict(response.headers))

                await self.metrics.record_request(
                    provider="openai",
                    model=model,
                    input_tokens=total_input_tokens,
                    output_tokens=output_tokens,
                    tokens_saved=tokens_saved,
                    latency_ms=total_latency,
                    overhead_ms=optimization_latency,
                    pipeline_timing=pipeline_timing,
                    waste_signals=waste_signals_dict,
                    cache_read_tokens=cache_read_tokens,
                    uncached_input_tokens=uncached_input_tokens,
                )

                if tokens_saved > 0:
                    logger.info(
                        f"[{request_id}] {model}: {original_tokens:,} → {optimized_tokens:,} "
                        f"(saved {tokens_saved:,} tokens)"
                    )

                # Remove compression headers since httpx already decompressed the response
                response_headers = dict(response.headers)
                response_headers.pop("content-encoding", None)
                response_headers.pop("content-length", None)  # Length changed after decompression

                # Inject Headroom compression metrics (for SaaS metering)
                response_headers["x-headroom-tokens-before"] = str(original_tokens)
                response_headers["x-headroom-tokens-after"] = str(optimized_tokens)
                response_headers["x-headroom-tokens-saved"] = str(tokens_saved)
                response_headers["x-headroom-model"] = model
                if transforms_applied:
                    response_headers["x-headroom-transforms"] = ",".join(transforms_applied)
                if cache_read_tokens > 0:
                    response_headers["x-headroom-cached"] = "true"
                if _compression_failed:
                    response_headers["x-headroom-compression-failed"] = "true"

                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=response_headers,
                )
        except Exception as e:
            await self.metrics.record_failed(provider="openai")
            # Log full error details internally for debugging
            logger.error(f"[{request_id}] OpenAI request failed: {type(e).__name__}: {e}")
            # Return sanitized error message to client (don't expose internal details)
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": "An error occurred while processing your request. Please try again.",
                        "type": "server_error",
                        "code": "proxy_error",
                    }
                },
            )

    async def handle_openai_responses(
        self,
        request: Request,
    ) -> Response | StreamingResponse:
        """Handle OpenAI /v1/responses endpoint (new Responses API).

        The Responses API differs from /v1/chat/completions:
        - Input: `input` (string or array) instead of `messages`
        - System: `instructions` instead of system message
        - Output: `output[]` array instead of `choices[].message`
        - State: `previous_response_id` for multi-turn
        - Built-in tools: web_search, file_search, code_interpreter
        """
        from fastapi import HTTPException
        from fastapi.responses import JSONResponse, Response

        from headroom.proxy.helpers import (
            MAX_REQUEST_BODY_SIZE,
            _read_request_json,
        )
        from headroom.tokenizers import get_tokenizer
        from headroom.utils import extract_user_query

        start_time = time.time()
        request_id = await self._next_request_id()

        # Phase F PR-F1: classify auth mode at request entry. The result
        # is stored on `request.state` so downstream handlers (cache
        # gates, header injection, lossy-compressor gates) read it
        # without re-classifying. Pure function, well under 10us.
        auth_mode = classify_auth_mode(request.headers)
        request.state.auth_mode = auth_mode
        logger.debug(f"[{request_id}] auth_mode_classified mode={auth_mode.value}")

        # Check request body size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_REQUEST_BODY_SIZE:
            return JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "message": f"Request body too large. Maximum size is {MAX_REQUEST_BODY_SIZE // (1024 * 1024)}MB",
                        "type": "invalid_request_error",
                        "code": "request_too_large",
                    }
                },
            )

        # Parse request
        try:
            body = await _read_request_json(request)
        except (json.JSONDecodeError, ValueError) as e:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": f"Invalid request body: {e!s}",
                        "type": "invalid_request_error",
                        "code": "invalid_json",
                    }
                },
            )

        model = body.get("model", "unknown")
        stream = body.get("stream", False)

        # PR-C5: Python no longer compresses /v1/responses — Rust handles
        # item-aware compression natively (see crates/headroom-proxy/src/
        # handlers/responses.rs). We synthesise a minimal `messages` list
        # purely for downstream memory injection + telemetry; list-typed
        # `input` is consulted via `body["input"]` directly via the
        # live-zone-tail helpers below.
        input_data = body.get("input", "")
        instructions = body.get("instructions")

        messages: list[dict[str, Any]] = []
        if instructions:
            messages.append({"role": "system", "content": instructions})
        if isinstance(input_data, str):
            messages.append({"role": "user", "content": input_data})

        headers = dict(request.headers.items())
        headers.pop("host", None)
        headers.pop("content-length", None)
        # Strip accept-encoding so httpx negotiates its own encoding.
        # Cloudflare Workers forward "br, zstd" which OpenAI may honor;
        # if httpx lacks brotli support the response body is undecipherable → 502.
        headers.pop("accept-encoding", None)
        tags = self._extract_tags(headers)
        # PR-A5 (P5-49): strip internal x-headroom-* from upstream-bound
        # headers AFTER `_extract_tags` reads them. Memory user-id reads
        # `request.headers` below.
        from headroom.proxy.helpers import _strip_internal_headers, log_outbound_headers

        _pre_strip_count_resp = sum(1 for k in headers if k.lower().startswith("x-headroom-"))
        headers = _strip_internal_headers(headers)
        log_outbound_headers(
            forwarder="openai_responses",
            stripped_count=_pre_strip_count_resp,
            request_id=request_id,
        )

        # PR-A6 (P5-50, preps P0-6): session-sticky `OpenAI-Beta` merge
        # for /v1/responses. Compute a session_id off the same store the
        # chat handler uses so multi-endpoint clients within one
        # conversation share the sticky-token set.
        _responses_session_id = self.session_tracker_store.compute_session_id(
            request, model, messages
        )
        from headroom.proxy.helpers import (
            get_session_beta_tracker as _get_session_beta_tracker_resp,
        )
        from headroom.proxy.helpers import (
            log_beta_header_merge as _log_beta_header_merge_resp,
        )

        _client_resp_beta = headers.get("openai-beta")
        _client_resp_beta_count = (
            len([t for t in (_client_resp_beta or "").split(",") if t.strip()])
            if _client_resp_beta
            else 0
        )
        _sticky_resp_beta = _get_session_beta_tracker_resp().record_and_get_sticky_betas(
            provider="openai",
            session_id=_responses_session_id,
            client_value=_client_resp_beta,
        )
        _sticky_resp_beta_count = (
            len([t for t in _sticky_resp_beta.split(",") if t.strip()]) if _sticky_resp_beta else 0
        )
        if _sticky_resp_beta and _sticky_resp_beta != (_client_resp_beta or ""):
            headers["openai-beta"] = _sticky_resp_beta
        _log_beta_header_merge_resp(
            provider="openai",
            session_id=_responses_session_id,
            client_betas_count=_client_resp_beta_count,
            sticky_betas_count=_sticky_resp_beta_count,
            headroom_added=[],
            request_id=request_id,
        )

        # Memory: Get user ID when memory is enabled. Reads `request.headers`
        # directly because `headers` was stripped of `x-headroom-*` (PR-A5).
        memory_user_id: str | None = None
        if self.memory_handler:
            memory_user_id = request.headers.get(
                "x-headroom-user-id",
                os.environ.get("USER", os.environ.get("USERNAME", "default")),
            )

        # Rate limiting
        if self.rate_limiter:
            rate_key = headers.get("authorization", "default")[:20]
            allowed, wait_seconds = await self.rate_limiter.check_request(rate_key)
            if not allowed:
                await self.metrics.record_rate_limited(provider="openai")
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limited. Retry after {wait_seconds:.1f}s",
                )

        # Token counting on converted messages
        tokenizer = get_tokenizer(model)
        original_tokens = tokenizer.count_messages(messages)

        # PR-C5: Python compression on /v1/responses is retired — the Rust
        # handler at crates/headroom-proxy/src/handlers/responses.rs is the
        # canonical compression path. Defaults below feed downstream
        # telemetry and memory injection without invoking the pipeline.
        optimized_messages = messages
        optimized_tokens = original_tokens
        tokens_saved = 0
        transforms_applied: list[str] = []
        optimization_latency = (time.time() - start_time) * 1000

        # Memory: inject context and tools for Responses API requests
        if self.memory_handler and memory_user_id:
            try:
                # Memory context now routes exclusively to the live-zone tail
                # (latest non-frozen user item). Instructions are part of the
                # cache hot zone and must never be mutated — invariant I2.
                # See REALIGNMENT/03-phase-A-lockdown.md PR-A2.
                if self.memory_handler.config.inject_context:
                    try:
                        memory_context = await asyncio.wait_for(
                            self.memory_handler.search_and_format_context(
                                memory_user_id, optimized_messages
                            ),
                            timeout=RESPONSES_CONTEXT_SEARCH_TIMEOUT_SECONDS,
                        )
                    except asyncio.TimeoutError:
                        memory_context = None
                        logger.info(
                            f"[{request_id}] Memory context lookup exceeded "
                            f"{RESPONSES_CONTEXT_SEARCH_TIMEOUT_SECONDS:.1f}s; continuing without it"
                        )
                    if memory_context:
                        from headroom.proxy.helpers import (
                            append_text_to_latest_user_input_item,
                            get_memory_injection_mode,
                            log_memory_injection,
                        )

                        injection_mode = get_memory_injection_mode()
                        user_query = extract_user_query(optimized_messages) or ""
                        if injection_mode == "disabled":
                            log_memory_injection(
                                request_id=request_id,
                                session_id=None,
                                decision="skipped_disabled",
                                bytes_injected=0,
                                query=user_query,
                            )
                        else:
                            # Route into body["input"] (the canonical Responses API
                            # field) targeting the latest user item's first text
                            # block. body["instructions"] (cache hot zone) is left
                            # untouched.
                            current_input = body.get("input")
                            if isinstance(current_input, str):
                                # String input: append to it. The string IS the
                                # latest user content; appending here is the
                                # equivalent of the live-zone tail.
                                body["input"] = (
                                    current_input + "\n\n" + memory_context
                                    if current_input
                                    else memory_context
                                )
                                log_memory_injection(
                                    request_id=request_id,
                                    session_id=None,
                                    decision="injected_live_zone_tail_string",
                                    bytes_injected=len(memory_context),
                                    query=user_query,
                                )
                            elif isinstance(current_input, list):
                                new_input, bytes_appended = append_text_to_latest_user_input_item(
                                    current_input, memory_context
                                )
                                if bytes_appended > 0:
                                    body["input"] = new_input
                                    log_memory_injection(
                                        request_id=request_id,
                                        session_id=None,
                                        decision="injected_live_zone_tail",
                                        bytes_injected=bytes_appended,
                                        query=user_query,
                                    )
                                else:
                                    log_memory_injection(
                                        request_id=request_id,
                                        session_id=None,
                                        decision="no_eligible_user_item",
                                        bytes_injected=0,
                                        query=user_query,
                                    )
                            else:
                                log_memory_injection(
                                    request_id=request_id,
                                    session_id=None,
                                    decision="no_input_field",
                                    bytes_injected=0,
                                    query=user_query,
                                )

                # Inject memory tools (Responses API format) — PR-A7 (P0-6).
                # Pre-convert the Chat-Completions schema to Responses API
                # format BEFORE handing to the sticky tracker so the
                # canonical bytes pinned in turn 1 already reflect the
                # exact bytes that will hit the wire.
                from headroom.proxy.helpers import (
                    apply_session_sticky_memory_tools as _apply_sticky_mem_tools_resp,
                )

                memory_tool_defs_chat = (
                    self.memory_handler.compute_memory_tool_definitions("openai")
                    if self.memory_handler.config.inject_tools
                    else []
                )
                memory_tool_defs_responses: list[dict[str, Any]] = []
                for t in memory_tool_defs_chat:
                    if t.get("type") == "function" and "function" in t:
                        fn = t["function"]
                        memory_tool_defs_responses.append(
                            {
                                "type": "function",
                                "name": fn.get("name"),
                                "description": fn.get("description", ""),
                                "parameters": fn.get("parameters", {}),
                            }
                        )
                    else:
                        memory_tool_defs_responses.append(t)

                resp_tools = body.get("tools") or []
                resp_tools, mem_tools_injected = _apply_sticky_mem_tools_resp(
                    provider="openai",
                    session_id=_responses_session_id,
                    request_id=request_id,
                    existing_tools=resp_tools,
                    memory_tools_to_inject=memory_tool_defs_responses,
                    inject_this_turn=bool(self.memory_handler.config.inject_tools),
                )
                if mem_tools_injected:
                    body["tools"] = resp_tools
                    logger.info(f"[{request_id}] Memory: Injected memory tools (openai/responses)")
            except Exception as e:
                logger.warning(f"[{request_id}] Memory injection failed (responses): {e}")

        # /v1/responses is OpenAI-specific (Codex) — always routes direct.
        # LiteLLM/AnyLLM backends use /v1/chat/completions or /v1/messages.
        if self.anthropic_backend is not None:
            logger.debug(
                f"[{request_id}] /v1/responses always routes to OpenAI direct "
                f"(backend '{self.anthropic_backend.name}' not used for Responses API)"
            )

        headers, is_chatgpt_auth = _resolve_codex_routing_headers(headers)

        # Route to correct endpoint based on auth mode.
        # ChatGPT session auth (codex login) uses chatgpt.com, not api.openai.com.
        if is_chatgpt_auth:
            url = "https://chatgpt.com/backend-api/codex/responses"
        else:
            url = build_copilot_upstream_url(self.OPENAI_API_URL, "/v1/responses")

        try:
            if stream:
                # Streaming for Responses API uses semantic events
                return await self._stream_response(
                    url,
                    headers,
                    body,
                    "openai",
                    model,
                    request_id,
                    original_tokens,
                    optimized_tokens,
                    tokens_saved,
                    transforms_applied,
                    tags,
                    optimization_latency,
                    memory_user_id=memory_user_id,
                )
            else:
                headers = await apply_copilot_api_auth(headers, url=url)
                response = await self._retry_request("POST", url, headers, body)
                total_latency = (time.time() - start_time) * 1000

                total_input_tokens = original_tokens  # fallback
                output_tokens = 0
                try:
                    resp_json = response.json()
                    usage = resp_json.get("usage", {})
                    total_input_tokens = usage.get("input_tokens", original_tokens)
                    output_tokens = usage.get("output_tokens", 0)
                except (KeyError, TypeError, AttributeError) as e:
                    logger.debug(
                        f"[{request_id}] Failed to extract cached tokens from OpenAI passthrough response: {e}"
                    )

                # Memory: handle memory tool calls in Responses API response
                if (
                    self.memory_handler
                    and memory_user_id
                    and resp_json
                    and response.status_code == 200
                    and self.memory_handler.has_memory_tool_calls(resp_json, "openai")
                ):
                    try:
                        # Extract function_call items from output
                        from headroom.proxy.memory_handler import MEMORY_TOOL_NAMES

                        output_items = resp_json.get("output", [])
                        memory_fc_items = [
                            item
                            for item in output_items
                            if isinstance(item, dict)
                            and item.get("type") == "function_call"
                            and item.get("name") in MEMORY_TOOL_NAMES
                        ]

                        # Execute memory tool calls
                        tool_outputs: list[dict[str, Any]] = []
                        for fc in memory_fc_items:
                            call_id = fc.get("call_id", fc.get("id", ""))
                            name = fc.get("name", "")
                            args_str = fc.get("arguments", "{}")
                            try:
                                args = json.loads(args_str)
                            except json.JSONDecodeError:
                                args = {}

                            await self.memory_handler._ensure_initialized()
                            if self.memory_handler._backend:
                                result = await self.memory_handler._execute_memory_tool(
                                    name, args, memory_user_id, "openai"
                                )
                            else:
                                result = json.dumps({"error": "Memory backend not initialized"})

                            tool_outputs.append(
                                {
                                    "type": "function_call_output",
                                    "call_id": call_id,
                                    "output": result,
                                }
                            )

                        if tool_outputs:
                            # Make continuation request with tool results
                            response_id = resp_json.get("id")
                            continuation_body = {
                                "model": model,
                                "input": tool_outputs,
                            }
                            if response_id:
                                continuation_body["previous_response_id"] = response_id
                            existing_tools = body.get("tools")
                            if existing_tools:
                                continuation_body["tools"] = existing_tools

                            cont_response = await self._retry_request(
                                "POST", url, headers, continuation_body
                            )
                            resp_json = cont_response.json()
                            response = cont_response
                            logger.info(
                                f"[{request_id}] Memory: Handled {len(tool_outputs)} "
                                f"tool call(s) with continuation for user {memory_user_id} (responses)"
                            )
                    except Exception as e:
                        logger.warning(
                            f"[{request_id}] Memory tool handling failed (responses): {e}"
                        )

                if self.cost_tracker:
                    self.cost_tracker.record_tokens(model, tokens_saved, total_input_tokens)

                await self.metrics.record_request(
                    provider="openai",
                    model=model,
                    input_tokens=total_input_tokens,
                    output_tokens=output_tokens,
                    tokens_saved=tokens_saved,
                    latency_ms=total_latency,
                    overhead_ms=optimization_latency,
                )

                logger.info(f"[{request_id}] /v1/responses {model}: {total_input_tokens:,} tokens")

                # Capture Codex rate-limit window data from response headers
                from headroom.subscription.codex_rate_limits import (
                    get_codex_rate_limit_state,
                )

                get_codex_rate_limit_state().update_from_headers(dict(response.headers))

                # Remove compression headers
                response_headers = dict(response.headers)
                response_headers.pop("content-encoding", None)
                response_headers.pop("content-length", None)

                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=response_headers,
                )
        except Exception as e:
            await self.metrics.record_failed(provider="openai")
            logger.error(f"[{request_id}] OpenAI responses request failed: {type(e).__name__}: {e}")
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": "An error occurred while processing your request. Please try again.",
                        "type": "server_error",
                        "code": "proxy_error",
                    }
                },
            )

    async def handle_openai_responses_ws(self, websocket: WebSocket) -> None:
        """WebSocket proxy for /v1/responses (Codex gpt-5.4+).

        Newer Codex versions use WebSocket instead of HTTP POST for the
        Responses API.  This handler:
        1. Accepts the client WebSocket
        2. Receives the first message (``response.create`` request)
        3. Opens an upstream WebSocket to OpenAI
        4. Sends the request upstream as-is (PR-C5: Python compression on
           /v1/responses retired — Rust handles item-aware compression)
        5. Relays all subsequent messages bidirectionally
        """
        try:
            import websockets
        except ImportError:
            await websocket.accept()
            await websocket.close(
                code=1011,
                reason="websockets package not installed. pip install websockets",
            )
            return

        request_id = await self._next_request_id()
        session_id = uuid.uuid4().hex

        # Stage-timer — captures per-stage durations for the structured
        # log emitted on session close. Unit 2 instrumentation.
        stage_timer = StageTimer()
        session_started_at = time.perf_counter()

        # Unit 3: initialize registry variables *before* accept so the
        # outermost ``finally`` can rely on them existing even if
        # registration itself fails for some reason.
        ws_sessions: WebSocketSessionRegistry | None = getattr(self, "ws_sessions", None)
        session_handle: WSSessionHandle | None = None
        termination_cause: TerminationCause = "unknown"

        # Forward client headers to upstream, adding required OpenAI-Beta header
        ws_headers = dict(websocket.headers)

        # Extract subprotocol from client — this is an application-level negotiation
        # that MUST be forwarded end-to-end (unlike sec-websocket-key which is per-connection).
        # Codex and OpenAI negotiate a subprotocol; stripping it causes OpenAI to return 500.
        client_subprotocols: list[str] = []
        raw_protocol = ws_headers.get("sec-websocket-protocol", "")
        if raw_protocol:
            client_subprotocols = [p.strip() for p in raw_protocol.split(",") if p.strip()]

        # Accept client connection with the requested subprotocol
        async with stage_timer.measure("accept"):
            if client_subprotocols:
                await websocket.accept(subprotocol=client_subprotocols[0])
            else:
                await websocket.accept()

        # --- Unit 3: register the session as soon as accept succeeds ---
        client_addr: str | None = None
        client_info = getattr(websocket, "client", None)
        if client_info is not None:
            host = getattr(client_info, "host", None)
            port = getattr(client_info, "port", None)
            if host is not None and port is not None:
                client_addr = f"{host}:{port}"
            elif host is not None:
                client_addr = str(host)
        if ws_sessions is not None:
            session_handle = WSSessionHandle(
                session_id=session_id,
                request_id=request_id,
                client_addr=client_addr,
                upstream_url=None,  # set below once upstream_url is computed
            )
            ws_sessions.register(session_handle)
            metrics = getattr(self, "metrics", None)
            if metrics is not None and hasattr(metrics, "inc_active_ws_sessions"):
                try:
                    metrics.inc_active_ws_sessions()
                except Exception:  # pragma: no cover - defensive
                    pass

        # Forward all client headers except hop-by-hop / per-connection headers.
        # These are WebSocket handshake mechanics that the `websockets` library
        # generates fresh for the upstream connection — forwarding them would conflict.
        # Everything else (auth, org, beta, user-agent, custom headers) is forwarded as-is.
        _skip_headers = frozenset(
            {
                "host",  # must match upstream, not local proxy
                "connection",  # hop-by-hop
                "upgrade",  # hop-by-hop
                "sec-websocket-key",  # per-connection cryptographic nonce
                "sec-websocket-version",  # protocol version (websockets lib sets this)
                "sec-websocket-extensions",  # per-connection negotiation
                "sec-websocket-accept",  # server-side only
                "sec-websocket-protocol",  # handled via subprotocols param below
                "content-length",  # hop-by-hop
                "transfer-encoding",  # hop-by-hop
            }
        )
        # PR-A5 (P5-49): also drop internal x-headroom-* from the upstream
        # WebSocket handshake. Inbound reads on `ws_headers` (memory user-id
        # below) keep working because we filter only when building
        # `upstream_headers`, not when reading from `ws_headers`.
        from headroom.proxy.helpers import (
            _strip_internal_headers as _strip_internal,
        )
        from headroom.proxy.helpers import (
            log_outbound_headers as _log_outbound_headers,
        )

        _ws_pre_strip_filtered: dict[str, str] = {}
        for k, v in ws_headers.items():
            if k.lower() not in _skip_headers:
                _ws_pre_strip_filtered[k] = v
        _ws_pre_strip_count = sum(
            1 for k in _ws_pre_strip_filtered if k.lower().startswith("x-headroom-")
        )
        upstream_headers = _strip_internal(_ws_pre_strip_filtered)
        _log_outbound_headers(
            forwarder="openai_responses_ws",
            stripped_count=_ws_pre_strip_count,
            request_id=request_id,
        )

        upstream_headers, is_chatgpt_auth = _resolve_codex_routing_headers(upstream_headers)
        _lower_headers = {k.lower(): v for k, v in upstream_headers.items()}

        # Build upstream WebSocket URL based on auth mode
        if is_chatgpt_auth:
            # ChatGPT session auth → route to chatgpt.com backend
            upstream_url = "wss://chatgpt.com/backend-api/codex/responses"
            logger.debug(
                f"[{request_id}] WS: ChatGPT session auth detected, routing to chatgpt.com"
            )
        else:
            # API key auth → route to configured OpenAI API URL
            base = self.OPENAI_API_URL
            ws_base = base.replace("https://", "wss://").replace("http://", "ws://")
            upstream_url = build_copilot_upstream_url(ws_base, "/v1/responses")

        # Unit 3: attach the resolved upstream URL to the session handle.
        if session_handle is not None:
            session_handle.upstream_url = upstream_url

        # Ensure Authorization header is present — fall back to OPENAI_API_KEY env var.
        # Safety net for clients that don't forward auth headers via WebSocket upgrade.
        if "authorization" not in _lower_headers:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                upstream_headers["Authorization"] = f"Bearer {api_key}"
                logger.debug(f"[{request_id}] WS: injected Authorization from OPENAI_API_KEY env")
            else:
                logger.warning(
                    f"[{request_id}] WS: no Authorization header from client and "
                    f"OPENAI_API_KEY not set — upstream will likely reject"
                )

        upstream_headers = await apply_copilot_api_auth(upstream_headers, url=upstream_url)

        # Ensure the required beta header is present — OpenAI returns 500 without it.
        # PR-A6 (P5-50): use the deterministic `merge_openai_beta` helper
        # so the auto-injected `responses_websockets=2026-02-06` is
        # appended to the client's value (preserving order, deduping
        # case-insensitively) rather than overwriting it. The
        # SessionBetaTracker also records the merge so a future cross-
        # connection sticky model can replay tokens by session_id.
        from headroom.proxy.helpers import (
            get_session_beta_tracker as _get_session_beta_tracker_ws,
        )
        from headroom.proxy.helpers import (
            log_beta_header_merge as _log_beta_header_merge_ws,
        )
        from headroom.proxy.helpers import merge_openai_beta as _merge_openai_beta_ws

        _ws_required_tokens = ["responses_websockets=2026-02-06"]
        # Read the original (pre-merge) client value from the WS headers
        # to preserve casing and ordering.
        _ws_client_beta_value: str | None = None
        for _k, _v in upstream_headers.items():
            if _k.lower() == "openai-beta":
                _ws_client_beta_value = _v
                break
        # Record session-stickiness BEFORE adding required tokens so the
        # tracker stores the canonical client baseline.
        _ws_sticky_beta = _get_session_beta_tracker_ws().record_and_get_sticky_betas(
            provider="openai",
            session_id=session_id,
            client_value=_ws_client_beta_value,
        )
        _ws_merged_beta = _merge_openai_beta_ws(_ws_sticky_beta, _ws_required_tokens)
        # Replace any existing case-variants of openai-beta with the
        # canonical "OpenAI-Beta" key carrying the merged value.
        _ws_existing_keys = [_k for _k in upstream_headers if _k.lower() == "openai-beta"]
        for _k in _ws_existing_keys:
            del upstream_headers[_k]
        if _ws_merged_beta:
            upstream_headers["OpenAI-Beta"] = _ws_merged_beta
        _ws_client_beta_count = (
            len([t for t in (_ws_client_beta_value or "").split(",") if t.strip()])
            if _ws_client_beta_value
            else 0
        )
        _ws_merged_beta_count = (
            len([t for t in _ws_merged_beta.split(",") if t.strip()]) if _ws_merged_beta else 0
        )
        _log_beta_header_merge_ws(
            provider="openai",
            session_id=session_id,
            client_betas_count=_ws_client_beta_count,
            sticky_betas_count=_ws_merged_beta_count,
            headroom_added=_ws_required_tokens,
            request_id=request_id,
        )

        logger.debug(
            f"[{request_id}] WS upstream headers: "
            f"{[k for k in upstream_headers if k.lower() != 'authorization']}, "
            f"subprotocols={client_subprotocols}"
        )

        try:
            # Receive the first message from client (the response.create request).
            # Bound the wait with WS_FIRST_FRAME_TIMEOUT_SECONDS so a zombie
            # client that opens the WS but never sends a frame cannot hold a
            # session slot indefinitely. The StageTimer measurement still
            # captures the elapsed time up to the timeout so operators can
            # see the slow-client pattern in the stage-timings log.
            try:
                async with stage_timer.measure("first_client_frame"):
                    first_msg_raw = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=WS_FIRST_FRAME_TIMEOUT_SECONDS,
                    )
            except asyncio.TimeoutError:
                logger.info(
                    f"[{request_id}] WS first-frame timeout after "
                    f"{WS_FIRST_FRAME_TIMEOUT_SECONDS:.0f}s; closing session "
                    f"{session_id} (no client data)"
                )
                termination_cause = "client_timeout"
                with contextlib.suppress(Exception):
                    # 1001 (going away): server is cleanly terminating a slow
                    # client, not an internal error.
                    await websocket.close(code=1001, reason="first-frame timeout")
                # Exit the outer try so the session-lifecycle ``finally`` runs
                # deregister / metrics / stage-timings emission as usual.
                return

            # PR-C5: WebSocket /v1/responses no longer compresses in Python.
            # The Rust handler covers item-aware compression; the WS first
            # frame is now passed through unmodified.
            body: dict[str, Any] = {}
            tokens_saved = 0
            try:
                body = json.loads(first_msg_raw)
            except json.JSONDecodeError:
                # Not JSON — pass through as-is
                pass

            # --- Memory: inject context, tools, and instructions ---
            memory_user_id: str | None = None
            if self.memory_handler and body:
                memory_user_id = ws_headers.get(
                    "x-headroom-user-id",
                    os.environ.get("USER", os.environ.get("USERNAME", "default")),
                )
                try:
                    # Unwrap response.create envelope to access the response body
                    ws_response_body = body.get("response", body)

                    # Debug: log what Codex sends so we can see the full tool list
                    existing_tool_names = [
                        t.get("name") or t.get("function", {}).get("name", "?")
                        for t in (ws_response_body.get("tools") or [])
                    ]
                    instr_preview = (ws_response_body.get("instructions") or "")[:200]
                    logger.info(
                        f"[{request_id}] WS Memory: Codex tools={existing_tool_names}, "
                        f"instructions_len={len(ws_response_body.get('instructions') or '')}, "
                        f"instructions_preview={instr_preview!r}"
                    )

                    # Inject memory context into instructions
                    if self.memory_handler.config.inject_context:
                        ws_input = ws_response_body.get("input", "")
                        ws_instructions = ws_response_body.get("instructions")
                        ws_msgs: list[dict[str, Any]] = []
                        if ws_instructions:
                            ws_msgs.append({"role": "system", "content": ws_instructions})
                        if isinstance(ws_input, str) and ws_input:
                            ws_msgs.append({"role": "user", "content": ws_input})
                        # PR-C5: list-typed `input` no longer feeds memory
                        # search via the Python converter — the Rust handler
                        # owns native item-aware processing. Memory context
                        # for list-input WS sessions falls back to the
                        # `instructions` system message only.

                        try:
                            async with stage_timer.measure("memory_context"):
                                memory_context = await asyncio.wait_for(
                                    self.memory_handler.search_and_format_context(
                                        memory_user_id, ws_msgs
                                    ),
                                    timeout=RESPONSES_CONTEXT_SEARCH_TIMEOUT_SECONDS,
                                )
                        except asyncio.TimeoutError:
                            memory_context = None
                            logger.info(
                                f"[{request_id}] WS Memory: Context lookup exceeded "
                                f"{RESPONSES_CONTEXT_SEARCH_TIMEOUT_SECONDS:.1f}s; "
                                f"continuing without it"
                            )
                        if memory_context:
                            existing = ws_response_body.get("instructions") or ""
                            if existing:
                                ws_response_body["instructions"] = f"{existing}\n\n{memory_context}"
                            else:
                                ws_response_body["instructions"] = memory_context
                            logger.info(
                                f"[{request_id}] WS Memory: Injected {len(memory_context)} chars "
                                f"of context into instructions"
                            )

                    # Inject memory tools (Responses API format) — PR-A7 (P0-6).
                    # WS path uses a per-connection UUID; tracker scope is
                    # the WS session (short-lived). Pre-convert to Responses
                    # API format so canonical bytes match the wire format.
                    from headroom.proxy.helpers import (
                        apply_session_sticky_memory_tools as _apply_sticky_mem_tools_ws,
                    )

                    ws_mem_defs_chat = (
                        self.memory_handler.compute_memory_tool_definitions("openai")
                        if self.memory_handler.config.inject_tools
                        else []
                    )
                    ws_mem_defs_responses: list[dict[str, Any]] = []
                    for t in ws_mem_defs_chat:
                        if t.get("type") == "function" and "function" in t:
                            fn = t["function"]
                            ws_mem_defs_responses.append(
                                {
                                    "type": "function",
                                    "name": fn.get("name"),
                                    "description": fn.get("description", ""),
                                    "parameters": fn.get("parameters", {}),
                                }
                            )
                        else:
                            ws_mem_defs_responses.append(t)

                    ws_tools = ws_response_body.get("tools") or []
                    ws_tools, mem_injected = _apply_sticky_mem_tools_ws(
                        provider="openai",
                        session_id=session_id,
                        request_id=request_id,
                        existing_tools=ws_tools,
                        memory_tools_to_inject=ws_mem_defs_responses,
                        inject_this_turn=bool(self.memory_handler.config.inject_tools),
                    )
                    if mem_injected:
                        ws_response_body["tools"] = ws_tools

                        # Add memory instruction so the model uses
                        # memory tools as persistent cross-session knowledge.
                        mem_instruction = (
                            "\n\n## Memory\n"
                            "You have persistent memory via memory_search and "
                            "memory_save tools. Memory stores knowledge across "
                            "sessions — user info, project details, org context, "
                            "decisions, architecture, conventions, anything worth "
                            "remembering.\n\n"
                            "- ALWAYS call memory_search BEFORE searching files "
                            "when the user asks a question that could be answered "
                            "from prior knowledge.\n"
                            "- Call memory_save to store important facts, decisions, "
                            "or context that would be useful in future sessions.\n"
                            "- Memory is your first source of truth for anything "
                            "not visible in the current conversation."
                        )
                        existing_instr = ws_response_body.get("instructions") or ""
                        ws_response_body["instructions"] = existing_instr + mem_instruction
                        logger.info(
                            f"[{request_id}] WS Memory: Injected memory tools + instruction"
                        )

                    # Write back into envelope if it was wrapped
                    if "response" in body and isinstance(body["response"], dict):
                        body["response"] = ws_response_body
                    else:
                        body = ws_response_body

                    first_msg_raw = json.dumps(body)
                except Exception as e:
                    logger.warning(f"[{request_id}] WS Memory injection failed: {e}")

            # --- Connect to upstream OpenAI WebSocket ---
            logger.info(f"[{request_id}] WS /v1/responses connecting to {upstream_url}")

            # Use ssl=True to let the websockets library handle SSL natively.
            # Manual ssl.create_default_context() + certifi doesn't load the
            # Windows system cert store, causing HTTP 500 on wss:// connections.
            use_ssl: bool | None = True if upstream_url.startswith("wss://") else None

            ws_connected = False
            ws_connect_attempts = max(1, getattr(self.config, "retry_max_attempts", 3))
            ws_last_err: Exception | None = None
            _upstream_connect_started = time.perf_counter()
            _upstream_connect_recorded = False
            _upstream_first_event_started: float | None = None

            for ws_attempt in range(ws_connect_attempts):
                try:
                    async with websockets.connect(
                        upstream_url,
                        additional_headers=upstream_headers,
                        subprotocols=(
                            [websockets.Subprotocol(p) for p in client_subprotocols]
                            if client_subprotocols and hasattr(websockets, "Subprotocol")
                            else client_subprotocols or None
                        ),
                        ssl=use_ssl,
                        open_timeout=max(30, self.config.connect_timeout_seconds * 3),
                        close_timeout=10,
                        ping_interval=20,
                        ping_timeout=20,
                    ) as upstream:
                        ws_connected = True
                        if not _upstream_connect_recorded:
                            stage_timer.record(
                                "upstream_connect",
                                (time.perf_counter() - _upstream_connect_started) * 1000.0,
                            )
                            _upstream_connect_recorded = True
                            _upstream_first_event_started = time.perf_counter()
                        await upstream.send(first_msg_raw)

                        # Unit 3: flag the upstream side flips on seeing
                        # ``response.completed`` so the outer cause
                        # classifier can prefer it over the raw
                        # "upstream iterator ended" default.
                        response_completed_seen = False
                        # Captures the first exception surfaced by the
                        # inner relay ``except`` blocks so the outer
                        # classifier can still tell ``upstream_error``
                        # from ``upstream_disconnect`` / ``response_completed``
                        # even though the halves swallow and log.
                        upstream_relay_error: BaseException | None = None
                        client_relay_error: BaseException | None = None

                        async def _client_to_upstream() -> None:
                            nonlocal client_relay_error
                            try:
                                while True:
                                    msg = await websocket.receive_text()
                                    await upstream.send(msg)
                            except asyncio.CancelledError:
                                # Explicit cancel from the outer
                                # orchestrator — re-raise so
                                # ``t.cancelled()`` and ``t.exception()``
                                # behave correctly in the caller.
                                raise
                            except Exception as relay_err:
                                # Surface real errors to the classifier
                                # without re-raising (existing fork
                                # behavior: log and return so the
                                # partner task can be cancelled
                                # deterministically).
                                if "WebSocketDisconnect" not in type(relay_err).__name__:
                                    client_relay_error = relay_err
                                    logger.debug(
                                        f"[{request_id}] WS client→upstream relay ended: {relay_err}"
                                    )
                                with contextlib.suppress(Exception):
                                    await upstream.close()

                        async def _upstream_to_client() -> None:
                            """Relay upstream→client with transparent memory tool handling.

                            Uses a buffer-then-decide approach:
                            1. Buffer events until first output item arrives
                            2. If first output is a memory tool → suppress entire response,
                               execute tools silently, send continuation upstream
                            3. If first output is non-memory → flush buffer, stream normally
                            4. Continuation response events are relayed to Codex seamlessly

                            This prevents orphaned response.created events from confusing Codex.
                            """
                            from headroom.proxy.memory_handler import MEMORY_TOOL_NAMES

                            # Unit 3: surface response.completed observation
                            # to the outer scope so the termination-cause
                            # classifier can prefer ``response_completed``
                            # over ``upstream_disconnect``.
                            nonlocal response_completed_seen
                            nonlocal upstream_relay_error

                            memory_enabled = bool(self.memory_handler and memory_user_id)

                            # Per-response state (reset after each response.completed)
                            event_buffer: list[str] = []
                            decided = False
                            suppress_response = False
                            pending_fcs: list[dict[str, Any]] = []
                            resp_id: str | None = None

                            def _reset() -> None:
                                nonlocal decided, suppress_response, resp_id
                                event_buffer.clear()
                                decided = False
                                suppress_response = False
                                pending_fcs.clear()
                                resp_id = None

                            # The retry-loop variable is safe to close over here:
                            # ``_upstream_to_client`` is defined and awaited within
                            # a single iteration and never escapes.
                            _first_event_started_at = _upstream_first_event_started  # noqa: B023

                            try:
                                async for msg in upstream:
                                    if (
                                        _first_event_started_at is not None
                                        and "upstream_first_event" not in stage_timer
                                    ):
                                        stage_timer.record(
                                            "upstream_first_event",
                                            (time.perf_counter() - _first_event_started_at)
                                            * 1000.0,
                                        )
                                    if isinstance(msg, bytes):
                                        await websocket.send_bytes(msg)
                                        continue
                                    msg_str = msg if isinstance(msg, str) else str(msg)

                                    if not memory_enabled:
                                        await websocket.send_text(msg_str)
                                        continue

                                    # Parse event
                                    try:
                                        event = json.loads(msg_str)
                                    except (json.JSONDecodeError, TypeError):
                                        await websocket.send_text(msg_str)
                                        continue

                                    event_type = event.get("type", "")

                                    # --- Phase 1: Buffer until first output item ---
                                    if not decided:
                                        event_buffer.append(msg_str)

                                        if event_type == "response.output_item.added":
                                            item = event.get("item", {})
                                            if (
                                                item.get("type") == "function_call"
                                                and item.get("name") in MEMORY_TOOL_NAMES
                                            ):
                                                # Memory tool first → suppress entire response
                                                suppress_response = True
                                                decided = True
                                                event_buffer.clear()
                                                logger.info(
                                                    f"[{request_id}] WS Memory: Detected "
                                                    f"{item.get('name')} — suppressing response"
                                                )
                                            else:
                                                # Non-memory first → flush buffer, pass through
                                                decided = True
                                                for buf in event_buffer:
                                                    await websocket.send_text(buf)
                                                event_buffer.clear()

                                        elif event_type == "response.completed":
                                            # No output items at all — flush
                                            decided = True
                                            for buf in event_buffer:
                                                await websocket.send_text(buf)
                                            event_buffer.clear()
                                            _reset()
                                            response_completed_seen = True

                                        continue

                                    # --- Phase 2a: Suppress mode (memory response) ---
                                    if suppress_response:
                                        if event_type == "response.output_item.done":
                                            item = event.get("item", {})
                                            if (
                                                item.get("type") == "function_call"
                                                and item.get("name") in MEMORY_TOOL_NAMES
                                            ):
                                                pending_fcs.append(item)

                                        elif event_type == "response.completed":
                                            response_completed_seen = True
                                            resp = event.get("response", {})
                                            resp_id = resp.get("id")

                                            if pending_fcs:
                                                logger.info(
                                                    f"[{request_id}] WS Memory: Executing "
                                                    f"{len(pending_fcs)} tool(s) transparently"
                                                )

                                                # Execute memory tool calls
                                                tool_outputs: list[dict[str, Any]] = []
                                                for fc in pending_fcs:
                                                    call_id = fc.get("call_id", fc.get("id", ""))
                                                    fc_name = fc.get("name", "")
                                                    args_str = fc.get("arguments", "{}")
                                                    try:
                                                        fc_args = json.loads(args_str)
                                                    except json.JSONDecodeError:
                                                        fc_args = {}

                                                    await self.memory_handler._ensure_initialized()
                                                    if self.memory_handler._backend:
                                                        result = await self.memory_handler._execute_memory_tool(
                                                            fc_name,
                                                            fc_args,
                                                            memory_user_id,
                                                            "openai",
                                                        )
                                                    else:
                                                        result = json.dumps(
                                                            {"error": "backend not ready"}
                                                        )

                                                    tool_outputs.append(
                                                        {
                                                            "type": "function_call_output",
                                                            "call_id": call_id,
                                                            "output": result,
                                                        }
                                                    )
                                                    logger.info(
                                                        f"[{request_id}] WS Memory: Executed "
                                                        f"{fc_name} for user {memory_user_id}"
                                                    )

                                                # Send continuation upstream
                                                cont: dict[str, Any] = {
                                                    "type": "response.create",
                                                    "response": {"input": tool_outputs},
                                                }
                                                if resp_id:
                                                    cont["response"]["previous_response_id"] = (
                                                        resp_id
                                                    )
                                                await upstream.send(json.dumps(cont))
                                                logger.info(
                                                    f"[{request_id}] WS Memory: Sent continuation "
                                                    f"with {len(tool_outputs)} result(s)"
                                                )

                                            _reset()
                                        # All events suppressed in this mode
                                        continue

                                    # --- Phase 2b: Pass-through mode ---
                                    await websocket.send_text(msg_str)

                            except asyncio.CancelledError:
                                raise
                            except Exception as relay_err:
                                if "WebSocketDisconnect" not in type(relay_err).__name__:
                                    # Capture for the outer classifier
                                    # so ``upstream_error`` can be
                                    # distinguished from a clean
                                    # upstream disconnect.
                                    upstream_relay_error = relay_err
                                    logger.debug(
                                        f"[{request_id}] WS upstream→client relay ended: {relay_err}"
                                    )
                            finally:
                                with contextlib.suppress(Exception):
                                    await websocket.close()

                        # --- Unit 3: deterministic relay-task cancellation ---
                        # Spawn each half as a named task so we can:
                        #   (a) attach them to the session registry for
                        #       ``/debug/ws-sessions``,
                        #   (b) cancel the survivor explicitly when the
                        #       first one exits, and
                        #   (c) classify the termination cause for the
                        #       duration histogram.
                        client_task = asyncio.create_task(
                            _client_to_upstream(),
                            name=f"codex-ws-c2u-{session_id}",
                        )
                        upstream_task = asyncio.create_task(
                            _upstream_to_client(),
                            name=f"codex-ws-u2c-{session_id}",
                        )
                        relay_tasks = [client_task, upstream_task]
                        if ws_sessions is not None:
                            ws_sessions.attach_tasks(session_id, relay_tasks)
                            metrics_for_tasks = getattr(self, "metrics", None)
                            if metrics_for_tasks is not None and hasattr(
                                metrics_for_tasks, "inc_active_relay_tasks"
                            ):
                                try:
                                    metrics_for_tasks.inc_active_relay_tasks(len(relay_tasks))
                                except Exception:  # pragma: no cover - defensive
                                    pass

                        try:
                            done, pending = await asyncio.wait(
                                {client_task, upstream_task},
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                            # Cancel the survivor so we don't leak the
                            # partner task. Suppress the CancelledError
                            # we just raised ourselves — any *other*
                            # exception from the cancelled task is
                            # already logged inside its own try/except.
                            for t in pending:
                                t.cancel()
                            if pending:
                                with contextlib.suppress(asyncio.CancelledError):
                                    await asyncio.gather(*pending, return_exceptions=True)

                            # Classify termination cause from whichever
                            # task completed first. ``CancelledError``
                            # can show up on the "done" side if the
                            # handler itself was cancelled from outside
                            # (e.g. server shutdown).
                            for t in done:
                                exc = None
                                # Cancelled tasks raise CancelledError from
                                # .exception(); surface it explicitly so the
                                # downstream ``isinstance(exc, CancelledError)``
                                # branches actually run. For any other
                                # unexpected state (``InvalidStateError`` if
                                # the task somehow isn't done — shouldn't
                                # happen post-gather but defensive), we
                                # suppress and leave ``exc`` as ``None``.
                                if t.cancelled():
                                    exc = asyncio.CancelledError()
                                else:
                                    with contextlib.suppress(asyncio.InvalidStateError):
                                        exc = t.exception()
                                task_name = t.get_name() or ""
                                if t is client_task:
                                    if client_relay_error is not None:
                                        termination_cause = "client_error"
                                    elif exc is None:
                                        termination_cause = "client_disconnect"
                                    elif isinstance(exc, asyncio.CancelledError):
                                        termination_cause = "client_disconnect"
                                    else:
                                        # Distinguish legitimate client
                                        # disconnect exceptions from
                                        # real errors: WebSocketDisconnect
                                        # is a normal client exit.
                                        if "WebSocketDisconnect" in type(exc).__name__:
                                            termination_cause = "client_disconnect"
                                        else:
                                            termination_cause = "client_error"
                                elif t is upstream_task:
                                    if upstream_relay_error is not None:
                                        termination_cause = "upstream_error"
                                        logger.debug(
                                            f"[{request_id}] WS relay {task_name} "
                                            f"raised: {upstream_relay_error!r}"
                                        )
                                    elif exc is None:
                                        termination_cause = (
                                            "response_completed"
                                            if response_completed_seen
                                            else "upstream_disconnect"
                                        )
                                    elif isinstance(exc, asyncio.CancelledError):
                                        termination_cause = "upstream_disconnect"
                                    else:
                                        termination_cause = "upstream_error"
                                        logger.debug(
                                            f"[{request_id}] WS relay {task_name} raised: {exc!r}"
                                        )
                        finally:
                            # In case anything above raised before the
                            # cancel-and-await loop ran.
                            for t in relay_tasks:
                                if not t.done():
                                    t.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await asyncio.gather(*relay_tasks, return_exceptions=True)

                        logger.info(
                            f"[{request_id}] WS /v1/responses completed "
                            f"(tokens_saved={tokens_saved}, cause={termination_cause})"
                        )
                    break
                except Exception as ws_err:
                    if ws_connected:
                        # WS was established but broke mid-stream — re-raise
                        raise

                    ws_last_err = ws_err
                    if ws_attempt >= ws_connect_attempts - 1:
                        break

                    delay_with_jitter = jitter_delay_ms(
                        self.config.retry_base_delay_ms,
                        self.config.retry_max_delay_ms,
                        ws_attempt,
                    )
                    logger.warning(
                        f"[{request_id}] WS upstream connect failed "
                        f"(attempt {ws_attempt + 1}/{ws_connect_attempts}): {ws_err}; "
                        f"retrying in {delay_with_jitter:.0f}ms"
                    )
                    await asyncio.sleep(delay_with_jitter / 1000)

            if not ws_connected:
                # WS upgrade failed (HTTP 500 from OpenAI is common).
                # Fall back to HTTP POST streaming and relay SSE events
                # back over the client WebSocket transparently.
                ws_err = ws_last_err or RuntimeError("unknown websocket connect failure")
                _ws_detail = str(ws_err)
                if hasattr(ws_err, "response"):
                    resp_body = getattr(getattr(ws_err, "response", None), "body", b"")
                    if resp_body:
                        from headroom.proxy.helpers import safe_decode_for_logging

                        _ws_detail += f" | {safe_decode_for_logging(resp_body, max_bytes=300)}"
                logger.warning(
                    f"[{request_id}] WS upstream failed ({_ws_detail}), "
                    f"falling back to HTTP POST streaming"
                )
                await self._ws_http_fallback(
                    websocket, body, first_msg_raw, upstream_headers, request_id
                )

            # Record metrics
            if tokens_saved > 0:
                model_name = body.get("model", "unknown") if isinstance(body, dict) else "unknown"
                await self.metrics.record_request(
                    provider="openai",
                    model=model_name,
                    input_tokens=0,
                    output_tokens=0,
                    tokens_saved=tokens_saved,
                    latency_ms=0,
                )

        except Exception as e:
            if "WebSocketDisconnect" in type(e).__name__:
                # Unit 3: client dropped the socket before or during
                # relay. The registry classifier may already have set
                # ``client_disconnect`` via the relay task exit path;
                # preserve that, otherwise set it here.
                if termination_cause == "unknown":
                    termination_cause = "client_disconnect"
            else:
                # Extract response body from websockets InvalidStatus for better debugging
                error_detail = str(e)
                if hasattr(e, "response"):
                    try:
                        resp = e.response
                        body_bytes = getattr(resp, "body", None) or b""
                        if body_bytes:
                            from headroom.proxy.helpers import safe_decode_for_logging

                            error_detail += (
                                f" | body: {safe_decode_for_logging(body_bytes, max_bytes=500)}"
                            )
                    except Exception:
                        pass
                logger.error(f"[{request_id}] WS proxy error: {error_detail}")
                if termination_cause == "unknown":
                    termination_cause = "client_error"
            with contextlib.suppress(Exception):
                await websocket.close(code=1011, reason=str(e)[:120])
        finally:
            # Unit 2: emit structured per-session stage timings.
            stage_timer.record(
                "total_session",
                (time.perf_counter() - session_started_at) * 1000.0,
            )
            # Unit 3: deregister the session before (or independently
            # of) the stage-timings log so a failure there cannot leak
            # the registry entry. ``deregister`` is idempotent, so a
            # session that never registered is a no-op.
            if ws_sessions is not None and session_handle is not None:
                # Use deregister_and_count so the handle pop and the
                # relay-task count are read atomically inside the
                # registry. Capturing ``len(session_handle.relay_tasks)``
                # separately before ``deregister`` would risk drift if
                # the registry's bookkeeping ever changes.
                _deregistered, released_tasks = ws_sessions.deregister_and_count(
                    session_id, cause=termination_cause
                )
                session_duration_ms = (time.perf_counter() - session_started_at) * 1000.0
                metrics_for_close = getattr(self, "metrics", None)
                if metrics_for_close is not None:
                    with contextlib.suppress(Exception):
                        if hasattr(metrics_for_close, "dec_active_ws_sessions"):
                            metrics_for_close.dec_active_ws_sessions()
                        if released_tasks and hasattr(metrics_for_close, "dec_active_relay_tasks"):
                            metrics_for_close.dec_active_relay_tasks(released_tasks)
                        if hasattr(metrics_for_close, "record_ws_session_duration"):
                            metrics_for_close.record_ws_session_duration(
                                session_duration_ms, termination_cause
                            )
            await emit_stage_timings_log(
                path="openai_responses_ws",
                request_id=request_id,
                session_id=session_id,
                stage_timer=stage_timer,
                expected_stages=(
                    "accept",
                    "first_client_frame",
                    "upstream_connect",
                    "upstream_first_event",
                    "memory_context",
                    "compression",
                    "total_session",
                ),
                metrics=getattr(self, "metrics", None),
            )

    async def _ws_http_fallback(
        self,
        websocket: WebSocket,
        body: dict[str, Any],
        first_msg_raw: str,
        upstream_headers: dict[str, str],
        request_id: str,
    ) -> None:
        """Fall back to HTTP POST streaming when upstream WS fails.

        Converts the WS ``response.create`` message to an HTTP POST to
        ``/v1/responses?stream=true``, reads SSE events, and relays each
        ``data:`` line as a WS text message to the client.  This makes
        Codex work immediately instead of exhausting its WS retry budget.
        """
        # Route to correct endpoint based on auth mode
        _lower = {k.lower() for k in upstream_headers}
        if "chatgpt-account-id" in _lower:
            http_url = "https://chatgpt.com/backend-api/codex/responses"
        else:
            http_url = build_copilot_upstream_url(self.OPENAI_API_URL, "/v1/responses")

        # Build HTTP body from the WS response.create payload.
        # WS messages use {"type": "response.create", "response": {...}} wrapper.
        # The HTTP POST endpoint expects the inner response object directly.
        http_body: dict[str, Any]
        try:
            parsed = json.loads(first_msg_raw) if isinstance(first_msg_raw, str) else body
        except (json.JSONDecodeError, TypeError):
            parsed = body

        # Normalize WebSocket response.create payload into the HTTP request body.
        # Codex may send either:
        # 1. {"type":"response.create","response":{...}}
        # 2. {"type":"response.create", ...response fields...}
        if isinstance(parsed, dict) and isinstance(parsed.get("response"), dict):
            http_body = dict(parsed["response"])
        elif isinstance(parsed, dict):
            http_body = dict(parsed)
            if http_body.get("type") == "response.create":
                http_body.pop("type", None)
        else:
            http_body = body if isinstance(body, dict) else {}

        # Some clients include response-ish metadata that the HTTP endpoint rejects.
        if http_body.get("type") in {"response.create", "response"}:
            http_body.pop("type", None)

        # Ensure streaming is enabled so we get SSE events
        http_body["stream"] = True

        # Build HTTP headers from the upstream headers (already stripped of WS
        # hop-by-hop headers by the caller).
        http_headers = dict(upstream_headers)
        http_headers["content-type"] = "application/json"

        # Byte-faithful re-serialization (PR-A3, fixes P0-2). The WS payload
        # is always synthesized from the WebSocket frame so the body is
        # treated as mutated; we still go through the canonical path so
        # numeric precision and UTF-8 are preserved.
        from headroom.proxy.helpers import (
            log_outbound_request,
            prepare_outbound_body_bytes,
        )

        outbound_bytes, outbound_source = prepare_outbound_body_bytes(
            body=http_body,
            original_body_bytes=None,
            body_mutated=True,
        )
        log_outbound_request(
            forwarder="openai_ws",
            method="POST",
            path=http_url,
            body_bytes_count=len(outbound_bytes),
            body_mutated=True,
            mutation_reasons=["ws_http_fallback_resynthesized"],
            request_id=request_id,
            source=outbound_source,
        )

        logger.debug(f"[{request_id}] WS→HTTP fallback POST to {http_url}")

        try:
            retry_attempts = max(1, getattr(self.config, "retry_max_attempts", 3))
            for http_attempt in range(retry_attempts):
                try:
                    async with self.http_client.stream(
                        "POST",
                        http_url,
                        headers=http_headers,
                        content=outbound_bytes,
                        timeout=120.0,
                    ) as response:
                        if response.status_code != 200:
                            error_body = b""
                            async for chunk in response.aiter_bytes():
                                error_body += chunk
                                if len(error_body) > 2000:
                                    break
                            from headroom.proxy.helpers import safe_decode_for_logging

                            error_text = safe_decode_for_logging(error_body)
                            logger.error(
                                f"[{request_id}] WS→HTTP fallback got {response.status_code}: "
                                f"{error_text[:500]}"
                            )
                            # Send error as WS message so client sees it
                            error_event = {
                                "type": "error",
                                "error": {
                                    "type": "server_error",
                                    "message": f"Upstream returned {response.status_code}",
                                },
                            }
                            await websocket.send_text(json.dumps(error_event))
                            return

                        # Relay SSE events as WS text messages
                        buffer = ""
                        async for chunk in response.aiter_text():
                            buffer += chunk
                            while "\n" in buffer:
                                line, buffer = buffer.split("\n", 1)
                                line = line.strip()
                                if not line:
                                    continue
                                if line.startswith("data: "):
                                    data = line[6:]
                                    if data == "[DONE]":
                                        continue
                                    try:
                                        await websocket.send_text(data)
                                    except Exception:
                                        return
                                elif line.startswith("event: "):
                                    # SSE event type — skip, the data line contains the type
                                    continue

                        # Flush any remaining data in buffer
                        for line in buffer.strip().splitlines():
                            line = line.strip()
                            if line.startswith("data: ") and line[6:] != "[DONE]":
                                with contextlib.suppress(Exception):
                                    await websocket.send_text(line[6:])
                        return
                except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as http_err:
                    if http_attempt >= retry_attempts - 1:
                        raise

                    delay_with_jitter = jitter_delay_ms(
                        self.config.retry_base_delay_ms,
                        self.config.retry_max_delay_ms,
                        http_attempt,
                    )
                    logger.warning(
                        f"[{request_id}] WS→HTTP fallback connect failed "
                        f"(attempt {http_attempt + 1}/{retry_attempts}): {http_err}; "
                        f"retrying in {delay_with_jitter:.0f}ms"
                    )
                    await asyncio.sleep(delay_with_jitter / 1000)

        except Exception as http_err:
            logger.error(f"[{request_id}] WS→HTTP fallback failed: {http_err}")
            error_event = {
                "type": "error",
                "error": {
                    "type": "server_error",
                    "message": f"HTTP fallback failed: {http_err!s}"[:200],
                },
            }
            with contextlib.suppress(Exception):
                await websocket.send_text(json.dumps(error_event))
        finally:
            with contextlib.suppress(Exception):
                await websocket.close()

    async def handle_compress(self, request: Request) -> JSONResponse:
        """Compress messages without calling an LLM.

        POST /v1/compress
        Body: {"messages": [...], "model": "...", "config": {}}
        Returns compressed messages + metrics.
        """
        from fastapi.responses import JSONResponse

        from headroom.proxy.helpers import _read_request_json

        # Check bypass header
        if request.headers.get("x-headroom-bypass", "").lower() == "true":
            try:
                body = await _read_request_json(request)
            except (json.JSONDecodeError, ValueError) as e:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Invalid request body: {e!s}"},
                )
            messages = body.get("messages", [])
            return JSONResponse(
                {
                    "messages": messages,
                    "tokens_before": 0,
                    "tokens_after": 0,
                    "tokens_saved": 0,
                    "compression_ratio": 1.0,
                    "transforms_applied": [],
                    "ccr_hashes": [],
                }
            )

        try:
            body = await _read_request_json(request)
        except Exception:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "type": "invalid_request",
                        "message": "Invalid JSON in request body.",
                    }
                },
            )

        messages = body.get("messages")
        model = body.get("model")

        if messages is None:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "type": "invalid_request",
                        "message": "Missing required field: messages",
                    }
                },
            )

        if model is None:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "type": "invalid_request",
                        "message": "Missing required field: model",
                    }
                },
            )

        if not messages:
            return JSONResponse(
                {
                    "messages": [],
                    "tokens_before": 0,
                    "tokens_after": 0,
                    "tokens_saved": 0,
                    "compression_ratio": 1.0,
                    "transforms_applied": [],
                    "ccr_hashes": [],
                }
            )

        try:
            # Use OpenAI pipeline (messages are in OpenAI format from TS SDK)
            # Allow optional token_budget to override model's context limit
            # (used by OpenClaw compact() and other callers that need tighter budgets)
            token_budget = body.get("token_budget")
            context_limit = (
                token_budget
                if token_budget and isinstance(token_budget, int)
                else self.openai_provider.get_context_limit(model)
            )
            # Extract CompressConfig options from request body
            compress_config = body.get("config", {})
            compress_user_messages = compress_config.get("compress_user_messages", False)
            target_ratio = compress_config.get("target_ratio")
            protect_recent = compress_config.get("protect_recent")
            protect_analysis_context = compress_config.get("protect_analysis_context")

            pipeline_kwargs: dict = {"model_limit": context_limit}
            if compress_user_messages:
                pipeline_kwargs["compress_user_messages"] = True
            if target_ratio is not None:
                pipeline_kwargs["target_ratio"] = float(target_ratio)
            if protect_recent is not None:
                pipeline_kwargs["protect_recent"] = int(protect_recent)
            if protect_analysis_context is not None:
                pipeline_kwargs["protect_analysis_context"] = bool(protect_analysis_context)

            result = self.openai_pipeline.apply(
                messages=messages,
                model=model,
                **pipeline_kwargs,
            )

            return JSONResponse(
                {
                    "messages": result.messages,
                    "tokens_before": result.tokens_before,
                    "tokens_after": result.tokens_after,
                    "tokens_saved": result.tokens_before - result.tokens_after,
                    "compression_ratio": (
                        result.tokens_after / result.tokens_before
                        if result.tokens_before > 0
                        else 1.0
                    ),
                    "transforms_applied": result.transforms_applied,
                    "transforms_summary": result.transforms_summary,
                    "ccr_hashes": result.markers_inserted,
                }
            )
        except Exception as e:
            logger.exception("Compression failed: %s", e)
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "type": "compression_error",
                        "message": str(e),
                    }
                },
            )

    async def handle_passthrough(
        self,
        request: Request,
        base_url: str,
        endpoint_name: str | None = None,
        provider: str | None = None,
    ) -> Response:
        """Pass through request unchanged.

        Args:
            request: The incoming request
            base_url: The upstream API base URL
            endpoint_name: Optional name for stats tracking (e.g., "models", "embeddings")
            provider: Optional provider name for stats (e.g., "openai", "anthropic", "gemini")
        """
        from fastapi.responses import Response

        start_time = time.time()
        path = request.url.path
        url = build_copilot_upstream_url(base_url, path)

        # Preserve query string parameters
        if request.url.query:
            url = f"{url}?{request.url.query}"

        headers = dict(request.headers.items())
        headers.pop("host", None)
        headers.pop("accept-encoding", None)
        # PR-A5 (P5-49): strip internal x-headroom-* before forwarding upstream.
        from headroom.proxy.helpers import _strip_internal_headers, log_outbound_headers

        _pre_strip_count_pt = sum(1 for k in headers if k.lower().startswith("x-headroom-"))
        headers = _strip_internal_headers(headers)
        log_outbound_headers(
            forwarder="openai_passthrough",
            stripped_count=_pre_strip_count_pt,
            request_id=None,
        )

        body = await request.body()

        headers = await apply_copilot_api_auth(headers, url=url)
        response = await self.http_client.request(  # type: ignore[union-attr]
            method=request.method,
            url=url,
            headers=headers,
            content=body,
        )

        # Remove compression headers since httpx already decompressed the response
        response_headers = dict(response.headers)
        response_headers.pop("content-encoding", None)
        response_headers.pop("content-length", None)  # Length changed after decompression

        # Track stats for passthrough requests
        if endpoint_name and provider:
            latency_ms = (time.time() - start_time) * 1000
            await self.metrics.record_request(
                provider=provider,
                model=f"passthrough:{endpoint_name}",
                input_tokens=0,
                output_tokens=0,
                tokens_saved=0,
                latency_ms=latency_ms,
                cached=False,
            )

        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=response_headers,
        )

    async def handle_databricks_invocations(
        self,
        request: Request,
        model: str,
    ) -> Response | StreamingResponse:
        """Handle Databricks native /serving-endpoints/{model}/invocations endpoint.

        This enables using the Databricks CLI directly with Headroom:
            databricks serving-endpoints query <model> --profile HEADROOM --json '{"messages": [...]}'

        The request/response format is identical to OpenAI chat completions,
        so we inject the model from the path and delegate to handle_openai_chat.
        """
        from fastapi.responses import JSONResponse

        from headroom.proxy.helpers import _read_request_json

        request_id = await self._next_request_id()

        try:
            body = await _read_request_json(request)
        except Exception as e:
            logger.error(f"[{request_id}] Failed to parse Databricks request body: {e}")
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": f"Invalid JSON: {e}",
                        "type": "invalid_request_error",
                    }
                },
            )

        # Inject model from path into body (Databricks CLI passes model in URL, not body)
        body["model"] = model

        logger.info(f"[{request_id}] Databricks invocation: model={model}")

        # Create a new request with the modified body
        # We reuse the OpenAI chat handler since the format is identical
        from starlette.requests import Request as StarletteRequest

        # Build new scope with the body already parsed
        scope = dict(request.scope)

        # Create a simple receive function that returns our modified body
        body_bytes = json.dumps(body).encode()

        async def receive():
            return {"type": "http.request", "body": body_bytes}

        modified_request = StarletteRequest(scope, receive)

        # Delegate to the OpenAI chat handler (same format)
        return await self.handle_openai_chat(modified_request)
