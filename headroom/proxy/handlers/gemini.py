"""Gemini handler mixin for HeadroomProxy.

Contains all Google Gemini API handlers including format conversion utilities.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import Request
    from fastapi.responses import JSONResponse, Response, StreamingResponse

logger = logging.getLogger("headroom.proxy")

DEFAULT_CLOUDCODE_API_URL = "https://cloudcode-pa.googleapis.com"
ANTIGRAVITY_DAILY_API_URL = "https://daily-cloudcode-pa.sandbox.googleapis.com"


class GeminiHandlerMixin:
    """Mixin providing Gemini API handler methods for HeadroomProxy."""

    def _is_cloudcode_antigravity_request(
        self, body: dict[str, Any], headers: dict[str, str]
    ) -> bool:
        """Detect Pi/OpenClaw antigravity requests routed via Cloud Code Assist."""
        user_agent = headers.get("user-agent", "").lower()
        body_user_agent = str(body.get("userAgent", "")).lower()
        return (
            body.get("requestType") == "agent"
            or body_user_agent == "antigravity"
            or user_agent.startswith("antigravity/")
        )

    def _resolve_cloudcode_base_url(self, is_antigravity: bool) -> str:
        """Resolve upstream base URL for Pi Cloud Code Assist / Antigravity traffic."""
        if is_antigravity:
            return ANTIGRAVITY_DAILY_API_URL
        return getattr(self, "CLOUDCODE_API_URL", DEFAULT_CLOUDCODE_API_URL).rstrip("/")

    def _has_non_text_parts(self, content: dict) -> bool:
        """Check if a Gemini content entry has non-text parts.

        Non-text parts include:
        - inlineData: Base64-encoded images/media
        - fileData: File references (URI + MIME type)
        - functionCall: Function calls from model
        - functionResponse: Responses to function calls

        Args:
            content: A single Gemini content entry with 'parts' list.

        Returns:
            True if any part contains non-text data.
        """
        parts = content.get("parts", [])
        for part in parts:
            if any(
                key in part
                for key in ("inlineData", "fileData", "functionCall", "functionResponse")
            ):
                return True
        return False

    def _gemini_contents_to_messages(
        self, contents: list[dict], system_instruction: dict | None = None
    ) -> tuple[list[dict], set[int]]:
        """Convert Gemini contents[] format to OpenAI messages[] format for optimization.

        Gemini format:
            contents: [{"role": "user", "parts": [{"text": "..."}]}]
            systemInstruction: {"parts": [{"text": "..."}]}

        OpenAI format:
            messages: [{"role": "user", "content": "..."}]

        Returns:
            Tuple of (messages, preserved_indices) where preserved_indices contains
            the indices of content entries that have non-text parts (images, function
            calls, etc.) and should not be compressed.
        """
        messages = []
        preserved_indices: set[int] = set()

        # Add system instruction as system message
        if system_instruction:
            parts = system_instruction.get("parts", [])
            text_parts = [p.get("text", "") for p in parts if "text" in p]
            if text_parts:
                messages.append({"role": "system", "content": "\n".join(text_parts)})

        # Convert contents to messages
        for idx, content in enumerate(contents):
            # Track content entries with non-text parts
            if self._has_non_text_parts(content):
                preserved_indices.add(idx)

            role = content.get("role", "user")
            # Map Gemini roles to OpenAI roles
            if role == "model":
                role = "assistant"

            parts = content.get("parts", [])
            text_parts = [p.get("text", "") for p in parts if "text" in p]

            if text_parts:
                messages.append({"role": role, "content": "\n".join(text_parts)})

        return messages, preserved_indices

    def _messages_to_gemini_contents(self, messages: list[dict]) -> tuple[list[dict], dict | None]:
        """Convert OpenAI messages[] format back to Gemini contents[] format.

        Returns:
            (contents, system_instruction) tuple
        """
        contents = []
        system_instruction = None

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                # Extract as systemInstruction
                system_instruction = {"parts": [{"text": content}]}
            else:
                # Map OpenAI roles to Gemini roles
                gemini_role = "model" if role == "assistant" else "user"
                contents.append({"role": gemini_role, "parts": [{"text": content}]})

        return contents, system_instruction

    async def handle_gemini_generate_content(
        self,
        request: Request,
        model: str,
    ) -> Response | StreamingResponse:
        """Handle Gemini native /v1beta/models/{model}:generateContent endpoint.

        Gemini's native API differs from OpenAI:
        - Input: `contents[]` with `parts[]` instead of `messages`
        - System: `systemInstruction` instead of system message
        - Auth: `x-goog-api-key` header instead of `Authorization: Bearer`
        - Output: `candidates[].content.parts[].text`
        """
        from fastapi import HTTPException
        from fastapi.responses import JSONResponse, Response

        from headroom.proxy.helpers import MAX_REQUEST_BODY_SIZE, _read_request_json
        from headroom.tokenizers import get_tokenizer
        from headroom.utils import extract_user_query

        start_time = time.time()
        request_id = await self._next_request_id()

        # Check request body size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_REQUEST_BODY_SIZE:
            return JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "message": f"Request body too large. Maximum size is {MAX_REQUEST_BODY_SIZE // (1024 * 1024)}MB",
                        "code": 413,
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
                        "code": 400,
                    }
                },
            )

        contents = body.get("contents", [])

        headers = dict(request.headers.items())
        headers.pop("host", None)
        headers.pop("content-length", None)
        tags = self._extract_tags(headers)
        # PR-A5 (P5-49): strip internal x-headroom-* from upstream-bound
        # headers AFTER `_extract_tags` reads them. Memory user-id reads
        # `request.headers` below.
        from headroom.proxy.helpers import _strip_internal_headers, log_outbound_headers

        _pre_strip_count_gem = sum(1 for k in headers if k.lower().startswith("x-headroom-"))
        headers = _strip_internal_headers(headers)
        log_outbound_headers(
            forwarder="gemini_generate_content",
            stripped_count=_pre_strip_count_gem,
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

        # Rate limiting (use Gemini API key)
        if self.rate_limiter:
            rate_key = headers.get("x-goog-api-key", "default")[:20]
            allowed, wait_seconds = await self.rate_limiter.check_request(rate_key)
            if not allowed:
                await self.metrics.record_rate_limited(provider="gemini")
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limited. Retry after {wait_seconds:.1f}s",
                )

        # Convert Gemini format to messages for optimization
        system_instruction = body.get("systemInstruction")
        messages, preserved_indices = self._gemini_contents_to_messages(
            contents, system_instruction
        )

        # Store original content entries that have non-text parts before compression
        preserved_contents = {idx: contents[idx] for idx in preserved_indices}

        # Early exit if ALL content has non-text parts (nothing to compress)
        if len(preserved_indices) == len(contents):
            # All content has non-text parts, skip compression entirely
            # Just forward the request as-is
            url = f"{self.GEMINI_API_URL}/v1beta/models/{model}:generateContent"
            query_params = dict(request.query_params)
            is_streaming = query_params.get("alt") == "sse"
            if "key" in query_params:
                url += f"?key={query_params['key']}"

            if is_streaming:
                stream_url = (
                    f"{self.GEMINI_API_URL}/v1beta/models/{model}:streamGenerateContent?alt=sse"
                )
                if "key" in query_params:
                    stream_url = f"{self.GEMINI_API_URL}/v1beta/models/{model}:streamGenerateContent?key={query_params['key']}&alt=sse"
                return await self._stream_response(
                    stream_url,
                    headers,
                    body,
                    "gemini",
                    model,
                    request_id,
                    0,
                    0,
                    0,
                    [],
                    tags,
                    0,
                )
            else:
                response = await self._retry_request("POST", url, headers, body)
                response_headers = dict(response.headers)
                response_headers.pop("content-encoding", None)
                response_headers.pop("content-length", None)
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=response_headers,
                )

        # Token counting
        tokenizer = get_tokenizer(model)
        original_tokens = tokenizer.count_messages(messages)

        # Optimization
        transforms_applied: list[str] = []
        waste_signals_dict: dict[str, int] | None = None
        optimized_messages = messages
        optimized_tokens = original_tokens

        _compression_failed = False
        _license_ok = self.usage_reporter.should_compress if self.usage_reporter else True
        if self.config.optimize and messages and _license_ok:
            try:
                # Use OpenAI pipeline (similar message format)
                context_limit = self.openai_provider.get_context_limit(model)
                result = self.openai_pipeline.apply(
                    messages=messages,
                    model=model,
                    model_limit=context_limit,
                    context=extract_user_query(messages),
                )
                if result.messages != messages:
                    optimized_messages = result.messages
                    transforms_applied = result.transforms_applied
                    # Use pipeline's token counts for consistency with pipeline logs
                    original_tokens = result.tokens_before
                    optimized_tokens = result.tokens_after
                if result.waste_signals:
                    waste_signals_dict = result.waste_signals.to_dict()
            except Exception as e:
                _compression_failed = True
                logger.warning(f"[{request_id}] Gemini optimization failed: {e}")

        # Guard: if "optimization" inflated tokens, revert to originals
        if optimized_tokens > original_tokens:
            logger.warning(
                f"[{request_id}] Optimization inflated tokens "
                f"({original_tokens} -> {optimized_tokens}), reverting to original messages"
            )
            optimized_messages = messages
            optimized_tokens = original_tokens
            transforms_applied = []

        tokens_saved = original_tokens - optimized_tokens
        optimization_latency = (time.time() - start_time) * 1000

        # Memory: inject context for Gemini requests.
        #
        # PR-B6: memory context auto-injects to the live-zone tail (the
        # latest user message) — never to the system / systemInstruction
        # field. The cache hot zone is sacrosanct (invariant I2). When
        # the memory handler is in ``MemoryMode.TOOL`` its
        # ``search_and_format_context`` returns ``None`` so nothing flows
        # in here.
        if self.memory_handler and memory_user_id:
            try:
                if self.memory_handler.config.inject_context:
                    memory_context = await self.memory_handler.search_and_format_context(
                        memory_user_id, optimized_messages
                    )
                    if memory_context:
                        new_messages, bytes_appended = (
                            self.memory_handler._append_to_latest_user_tail(
                                optimized_messages,
                                memory_context,
                                provider="openai",
                            )
                        )
                        if bytes_appended > 0:
                            optimized_messages = new_messages
                            logger.info(
                                f"[{request_id}] Memory: Injected {bytes_appended} chars "
                                f"into latest user message tail for user {memory_user_id} (gemini)"
                            )
                        else:
                            logger.debug(
                                f"[{request_id}] Memory: no eligible user message; "
                                "skipped tail injection (gemini)"
                            )
            except Exception as e:
                logger.warning(f"[{request_id}] Memory injection failed (gemini): {e}")

        # Convert back to Gemini format if optimized
        if optimized_messages != messages:
            optimized_contents, optimized_system = self._messages_to_gemini_contents(
                optimized_messages
            )

            # Restore preserved content entries that had non-text parts
            for orig_idx, original_content in preserved_contents.items():
                if orig_idx < len(optimized_contents):
                    optimized_contents[orig_idx] = original_content

            body["contents"] = optimized_contents
            if optimized_system:
                body["systemInstruction"] = optimized_system
            elif "systemInstruction" in body:
                del body["systemInstruction"]

        # Build URL - model is extracted from path
        url = f"{self.GEMINI_API_URL}/v1beta/models/{model}:generateContent"

        # Check if streaming requested via query param
        query_params = dict(request.query_params)
        is_streaming = query_params.get("alt") == "sse"

        # Preserve API key in query params if present
        if "key" in query_params:
            url += f"?key={query_params['key']}"

        try:
            if is_streaming:
                # For streaming, use streamGenerateContent endpoint
                stream_url = (
                    f"{self.GEMINI_API_URL}/v1beta/models/{model}:streamGenerateContent?alt=sse"
                )
                if "key" in query_params:
                    stream_url = f"{self.GEMINI_API_URL}/v1beta/models/{model}:streamGenerateContent?key={query_params['key']}&alt=sse"

                return await self._stream_response(
                    stream_url,
                    headers,
                    body,
                    "gemini",
                    model,
                    request_id,
                    original_tokens,
                    optimized_tokens,
                    tokens_saved,
                    transforms_applied,
                    tags,
                    optimization_latency,
                )
            else:
                response = await self._retry_request("POST", url, headers, body)
                total_latency = (time.time() - start_time) * 1000

                total_input_tokens = optimized_tokens  # fallback
                output_tokens = 0
                cache_read_tokens = 0
                try:
                    resp_json = response.json()
                    usage = resp_json.get("usageMetadata", {})
                    total_input_tokens = usage.get("promptTokenCount", optimized_tokens)
                    output_tokens = usage.get("candidatesTokenCount", 0)
                    # Gemini returns cachedContentTokenCount for context-cached tokens
                    # These are charged at 10-25% of the input price depending on model
                    cache_read_tokens = usage.get("cachedContentTokenCount", 0)
                except (KeyError, TypeError, AttributeError) as e:
                    logger.debug(
                        f"[{request_id}] Failed to extract cached tokens from Gemini response: {e}"
                    )

                uncached_input_tokens = max(0, total_input_tokens - cache_read_tokens)

                if self.cost_tracker:
                    self.cost_tracker.record_tokens(
                        model,
                        tokens_saved,
                        optimized_tokens,
                        cache_read_tokens=cache_read_tokens,
                        uncached_tokens=uncached_input_tokens,
                    )

                await self.metrics.record_request(
                    provider="gemini",
                    model=model,
                    input_tokens=total_input_tokens,
                    output_tokens=output_tokens,
                    tokens_saved=tokens_saved,
                    latency_ms=total_latency,
                    overhead_ms=optimization_latency,
                    waste_signals=waste_signals_dict,
                    cache_read_tokens=cache_read_tokens,
                    uncached_input_tokens=uncached_input_tokens,
                )

                if tokens_saved > 0:
                    logger.info(
                        f"[{request_id}] Gemini {model}: {original_tokens:,} → {optimized_tokens:,} "
                        f"(saved {tokens_saved:,} tokens)"
                    )
                else:
                    logger.info(f"[{request_id}] Gemini {model}: {original_tokens:,} tokens")

                # Remove compression headers
                response_headers = dict(response.headers)
                response_headers.pop("content-encoding", None)
                response_headers.pop("content-length", None)

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
            await self.metrics.record_failed(provider="gemini")
            logger.error(f"[{request_id}] Gemini request failed: {type(e).__name__}: {e}")
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": "An error occurred while processing your request. Please try again.",
                        "code": 502,
                    }
                },
            )

    async def handle_google_cloudcode_stream(
        self,
        request: Request,
    ) -> StreamingResponse | JSONResponse:
        """Handle Pi/OpenClaw Google Cloud Code Assist and Antigravity streaming requests."""
        from fastapi.responses import JSONResponse

        from headroom.proxy.helpers import _read_request_json
        from headroom.tokenizers import get_tokenizer
        from headroom.utils import extract_user_query

        start_time = time.time()
        request_id = await self._next_request_id()

        try:
            body = await _read_request_json(request)
        except (json.JSONDecodeError, ValueError) as e:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": f"Invalid request body: {e!s}",
                        "code": 400,
                    }
                },
            )

        request_payload = body.get("request")
        if not isinstance(request_payload, dict):
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "Invalid Cloud Code Assist request: missing request payload",
                        "code": 400,
                    }
                },
            )

        model = body.get("model", "unknown")
        contents = request_payload.get("contents", [])
        headers = dict(request.headers.items())
        headers.pop("host", None)
        headers.pop("content-length", None)
        headers.pop("accept-encoding", None)
        tags = self._extract_tags(headers)
        is_antigravity = self._is_cloudcode_antigravity_request(body, headers)
        # PR-A5 (P5-49): strip internal x-headroom-* from upstream-bound headers
        # AFTER `_extract_tags` and `is_cloudcode_antigravity` reads.
        from headroom.proxy.helpers import _strip_internal_headers, log_outbound_headers

        _pre_strip_count_cca = sum(1 for k in headers if k.lower().startswith("x-headroom-"))
        headers = _strip_internal_headers(headers)
        log_outbound_headers(
            forwarder="gemini_cloudcode_assist",
            stripped_count=_pre_strip_count_cca,
            request_id=request_id,
        )

        system_instruction = request_payload.get("systemInstruction")
        optimization_system_instruction = None if is_antigravity else system_instruction
        messages, preserved_indices = self._gemini_contents_to_messages(
            contents if isinstance(contents, list) else [], optimization_system_instruction
        )
        preserved_contents = {
            idx: contents[idx]
            for idx in preserved_indices
            if isinstance(contents, list) and idx < len(contents)
        }

        tokenizer = get_tokenizer(model)
        original_tokens = tokenizer.count_messages(messages) if messages else 0
        optimized_messages = messages
        optimized_tokens = original_tokens
        transforms_applied: list[str] = []

        _license_ok = self.usage_reporter.should_compress if self.usage_reporter else True
        if self.config.optimize and messages and _license_ok:
            try:
                context_limit = self.openai_provider.get_context_limit(model)
                result = self.openai_pipeline.apply(
                    messages=messages,
                    model=model,
                    model_limit=context_limit,
                    context=extract_user_query(messages),
                )
                if result.messages != messages:
                    optimized_messages = result.messages
                    transforms_applied = result.transforms_applied
                    original_tokens = result.tokens_before
                    optimized_tokens = result.tokens_after
            except Exception as e:
                logger.warning(f"[{request_id}] Cloud Code Assist optimization failed: {e}")

        if optimized_tokens > original_tokens:
            logger.warning(
                f"[{request_id}] Cloud Code Assist optimization inflated tokens "
                f"({original_tokens} -> {optimized_tokens}), reverting to original messages"
            )
            optimized_messages = messages
            optimized_tokens = original_tokens
            transforms_applied = []

        if optimized_messages != messages:
            optimized_contents, optimized_system = self._messages_to_gemini_contents(
                optimized_messages
            )
            for orig_idx, original_content in preserved_contents.items():
                if orig_idx < len(optimized_contents):
                    optimized_contents[orig_idx] = original_content
            request_payload["contents"] = optimized_contents
            if not is_antigravity:
                if optimized_system:
                    request_payload["systemInstruction"] = optimized_system
                elif "systemInstruction" in request_payload:
                    del request_payload["systemInstruction"]

        tokens_saved = original_tokens - optimized_tokens
        optimization_latency = (time.time() - start_time) * 1000
        base_url = self._resolve_cloudcode_base_url(is_antigravity)
        stream_url = f"{base_url}/v1internal:streamGenerateContent"
        if request.url.query:
            stream_url = f"{stream_url}?{request.url.query}"

        return await self._stream_response(
            stream_url,
            headers,
            body,
            "gemini",
            model,
            request_id,
            original_tokens,
            optimized_tokens,
            tokens_saved,
            transforms_applied,
            tags,
            optimization_latency,
        )

    async def handle_gemini_stream_generate_content(
        self,
        request: Request,
        model: str,
    ) -> StreamingResponse | JSONResponse:
        """Handle Gemini streaming endpoint /v1beta/models/{model}:streamGenerateContent."""
        from fastapi.responses import JSONResponse

        from headroom.proxy.helpers import _read_request_json
        from headroom.tokenizers import get_tokenizer

        start_time = time.time()
        request_id = await self._next_request_id()

        # Parse request
        try:
            body = await _read_request_json(request)
        except (json.JSONDecodeError, ValueError) as e:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": f"Invalid request body: {e!s}",
                        "code": 400,
                    }
                },
            )

        contents = body.get("contents", [])

        headers = dict(request.headers.items())
        headers.pop("host", None)
        headers.pop("content-length", None)
        tags = self._extract_tags(headers)
        # PR-A5 (P5-49): strip internal x-headroom-* before forwarding upstream.
        from headroom.proxy.helpers import _strip_internal_headers, log_outbound_headers

        _pre_strip_count_gem_stream = sum(1 for k in headers if k.lower().startswith("x-headroom-"))
        headers = _strip_internal_headers(headers)
        log_outbound_headers(
            forwarder="gemini_stream_generate_content",
            stripped_count=_pre_strip_count_gem_stream,
            request_id=request_id,
        )

        # Token counting
        tokenizer = get_tokenizer(model)
        original_tokens = 0
        for content in contents:
            parts = content.get("parts", [])
            for part in parts:
                if "text" in part:
                    original_tokens += tokenizer.count_text(part["text"])

        optimization_latency = (time.time() - start_time) * 1000

        # Build URL with SSE param
        query_params = dict(request.query_params)
        url = f"{self.GEMINI_API_URL}/v1beta/models/{model}:streamGenerateContent?alt=sse"
        if "key" in query_params:
            url = f"{self.GEMINI_API_URL}/v1beta/models/{model}:streamGenerateContent?key={query_params['key']}&alt=sse"

        return await self._stream_response(
            url,
            headers,
            body,
            "gemini",
            model,
            request_id,
            original_tokens,
            original_tokens,
            0,  # tokens_saved
            [],  # transforms_applied
            tags,
            optimization_latency,
        )

    async def handle_gemini_count_tokens(
        self,
        request: Request,
        model: str,
    ) -> Response:
        """Handle Gemini /v1beta/models/{model}:countTokens endpoint with compression.

        This endpoint counts tokens AFTER applying compression, so users can see
        how many tokens they'll actually use after optimization.

        The request format is the same as generateContent:
            {"contents": [...], "systemInstruction": {...}}
        """
        from fastapi.responses import JSONResponse, Response

        from headroom.proxy.helpers import _read_request_json
        from headroom.tokenizers import get_tokenizer
        from headroom.utils import extract_user_query

        start_time = time.time()
        request_id = await self._next_request_id()

        # Parse request
        try:
            body = await _read_request_json(request)
        except (json.JSONDecodeError, ValueError) as e:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": f"Invalid request body: {e!s}",
                        "code": 400,
                    }
                },
            )

        contents = body.get("contents", [])

        headers = dict(request.headers.items())
        headers.pop("host", None)
        headers.pop("content-length", None)
        # PR-A5 (P5-49): strip internal x-headroom-* before forwarding upstream.
        from headroom.proxy.helpers import _strip_internal_headers, log_outbound_headers

        _pre_strip_count_gem_count = sum(1 for k in headers if k.lower().startswith("x-headroom-"))
        headers = _strip_internal_headers(headers)
        log_outbound_headers(
            forwarder="gemini_count_tokens",
            stripped_count=_pre_strip_count_gem_count,
            request_id=request_id,
        )

        # Convert Gemini format to messages for optimization
        system_instruction = body.get("systemInstruction")
        messages, preserved_indices = self._gemini_contents_to_messages(
            contents, system_instruction
        )

        # Store original content entries that have non-text parts before compression
        preserved_contents = {idx: contents[idx] for idx in preserved_indices}

        # Early exit if ALL content has non-text parts (nothing to compress)
        if len(preserved_indices) == len(contents):
            # All content has non-text parts, skip compression entirely
            # Just forward the countTokens request as-is
            url = f"{self.GEMINI_API_URL}/v1beta/models/{model}:countTokens"
            query_params = dict(request.query_params)
            if "key" in query_params:
                url += f"?key={query_params['key']}"

            response = await self._retry_request("POST", url, headers, body)
            response_headers = dict(response.headers)
            response_headers.pop("content-encoding", None)
            response_headers.pop("content-length", None)
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=response_headers,
            )

        # Token counting (original)
        tokenizer = get_tokenizer(model)
        original_tokens = tokenizer.count_messages(messages)

        # Apply compression using the same pipeline as generateContent
        transforms_applied: list[str] = []
        optimized_messages = messages

        if self.config.optimize and messages:
            try:
                context_limit = self.openai_provider.get_context_limit(model)
                result = self.openai_pipeline.apply(
                    messages=messages,
                    model=model,
                    model_limit=context_limit,
                    context=extract_user_query(messages),
                )
                if result.messages != messages:
                    optimized_messages = result.messages
                    transforms_applied = result.transforms_applied
            except Exception as e:
                logger.warning(f"[{request_id}] Gemini countTokens optimization failed: {e}")

        # Convert back to Gemini format for the API call
        if optimized_messages != messages:
            optimized_contents, optimized_system = self._messages_to_gemini_contents(
                optimized_messages
            )

            # Restore preserved content entries that had non-text parts
            for orig_idx, original_content in preserved_contents.items():
                if orig_idx < len(optimized_contents):
                    optimized_contents[orig_idx] = original_content

            body["contents"] = optimized_contents
            if optimized_system:
                body["systemInstruction"] = optimized_system
            elif "systemInstruction" in body:
                del body["systemInstruction"]

        # Build URL
        url = f"{self.GEMINI_API_URL}/v1beta/models/{model}:countTokens"

        # Preserve API key in query params if present
        query_params = dict(request.query_params)
        if "key" in query_params:
            url += f"?key={query_params['key']}"

        try:
            response = await self._retry_request("POST", url, headers, body)
            total_latency = (time.time() - start_time) * 1000

            # Parse response to get token count
            compressed_tokens = 0
            try:
                resp_json = response.json()
                compressed_tokens = resp_json.get("totalTokens", 0)
            except (json.JSONDecodeError, ValueError) as e:
                logger.debug(f"[{request_id}] Failed to parse Gemini token count response: {e}")

            # Track stats
            tokens_saved = (
                max(0, original_tokens - compressed_tokens) if compressed_tokens > 0 else 0
            )

            await self.metrics.record_request(
                provider="gemini",
                model=model,
                input_tokens=compressed_tokens,
                output_tokens=0,
                tokens_saved=tokens_saved,
                latency_ms=total_latency,
            )

            if tokens_saved > 0:
                logger.info(
                    f"[{request_id}] Gemini countTokens {model}: {original_tokens:,} → {compressed_tokens:,} "
                    f"(saved {tokens_saved:,} tokens, transforms: {transforms_applied})"
                )
            else:
                logger.info(
                    f"[{request_id}] Gemini countTokens {model}: {compressed_tokens:,} tokens"
                )

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
            await self.metrics.record_failed(provider="gemini")
            logger.error(f"[{request_id}] Gemini countTokens failed: {type(e).__name__}: {e}")
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": "An error occurred while processing your request. Please try again.",
                        "code": 502,
                    }
                },
            )
