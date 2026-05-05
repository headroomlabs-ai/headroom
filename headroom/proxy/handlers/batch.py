"""Batch handler mixin for HeadroomProxy.

Contains all batch API handlers for Google and OpenAI batch operations.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import Request
    from fastapi.responses import Response

logger = logging.getLogger("headroom.proxy")


class BatchHandlerMixin:
    """Mixin providing batch API handler methods for HeadroomProxy."""

    async def handle_google_batch_create(
        self,
        request: Request,
        model: str,
    ) -> Response:
        """Handle Google POST /v1beta/models/{model}:batchGenerateContent endpoint.

        Google batch format:
        {
            "batch": {
                "display_name": "my-batch",
                "input_config": {
                    "requests": {
                        "requests": [
                            {
                                "request": {"contents": [{"parts": [{"text": "..."}]}]},
                                "metadata": {"key": "request-1"}
                            }
                        ]
                    }
                }
            }
        }

        This method applies compression to each request's contents before forwarding.
        """
        from fastapi.responses import JSONResponse, Response

        from headroom.ccr import CCRToolInjector
        from headroom.proxy.helpers import MAX_REQUEST_BODY_SIZE, _read_request_json
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
                        "code": 413,
                        "message": f"Request body too large. Maximum size is {MAX_REQUEST_BODY_SIZE // (1024 * 1024)}MB",
                        "status": "INVALID_ARGUMENT",
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
                        "code": 400,
                        "message": f"Invalid request body: {e!s}",
                        "status": "INVALID_ARGUMENT",
                    }
                },
            )

        # Extract batch config
        batch_config = body.get("batch", {})
        input_config = batch_config.get("input_config", {})
        requests_wrapper = input_config.get("requests", {})
        requests_list = requests_wrapper.get("requests", [])

        if not requests_list:
            # No inline requests - might be using file input, pass through
            logger.debug(f"[{request_id}] Google batch: No inline requests, passing through")
            return await self._google_batch_passthrough(request, model, body)

        # Extract headers
        headers = dict(request.headers.items())
        headers.pop("host", None)
        headers.pop("content-length", None)
        # PR-A5 (P5-49): strip internal x-headroom-* before forwarding upstream.
        from headroom.proxy.helpers import _strip_internal_headers, log_outbound_headers

        _pre_strip_count_gb = sum(1 for k in headers if k.lower().startswith("x-headroom-"))
        headers = _strip_internal_headers(headers)
        log_outbound_headers(
            forwarder="google_batch",
            stripped_count=_pre_strip_count_gb,
            request_id=request_id,
        )

        # Track compression stats
        total_original_tokens = 0
        total_optimized_tokens = 0
        total_tokens_saved = 0
        compressed_requests = []
        pipeline_timing: dict[str, float] = {}

        # Apply compression to each request in the batch
        for idx, batch_req in enumerate(requests_list):
            req_content = batch_req.get("request", {})
            metadata = batch_req.get("metadata", {})
            contents = req_content.get("contents", [])

            if not contents or not self.config.optimize:
                # No contents or optimization disabled - pass through unchanged
                compressed_requests.append(batch_req)
                continue

            # Convert Google format to messages for compression
            system_instruction = req_content.get("systemInstruction")
            messages, preserved_indices = self._gemini_contents_to_messages(
                contents, system_instruction
            )

            # Store original content entries that have non-text parts before compression
            preserved_contents = {idx: contents[idx] for idx in preserved_indices}

            # Early exit if ALL content has non-text parts (nothing to compress)
            if len(preserved_indices) == len(contents):
                # All content has non-text parts, skip compression
                compressed_requests.append(batch_req)
                continue

            # Apply optimization
            original_tokens = 0  # Set before try so error handler can use it
            optimized_tokens = 0
            try:
                # Look up model context limit, fall back to 128K
                context_limit = (
                    self.openai_provider.get_context_limit(model)
                    if hasattr(self, "openai_provider")
                    else 128000
                )

                # Use OpenAI pipeline (similar message format after conversion)
                result = self.openai_pipeline.apply(
                    messages=messages,
                    model=model,
                    model_limit=context_limit,
                    context=extract_user_query(messages),
                )

                optimized_messages = result.messages
                for k, v in result.timing.items():
                    pipeline_timing[k] = pipeline_timing.get(k, 0.0) + v
                # Use pipeline's token counts for consistency with pipeline logs
                original_tokens = result.tokens_before
                optimized_tokens = result.tokens_after
                # Guard: if "optimization" inflated tokens, revert to originals
                if optimized_tokens > original_tokens:
                    logger.warning(
                        f"[{request_id}] Batch item optimization inflated tokens "
                        f"({original_tokens} -> {optimized_tokens}), reverting"
                    )
                    optimized_messages = messages
                    optimized_tokens = original_tokens

                total_original_tokens += original_tokens
                total_optimized_tokens += optimized_tokens
                tokens_saved = original_tokens - optimized_tokens
                total_tokens_saved += tokens_saved

                # CCR Tool Injection: Inject retrieval tool if compression occurred
                tools = req_content.get("tools")
                # Extract existing function declarations if present
                existing_funcs = None
                if tools:
                    for tool in tools:
                        if "functionDeclarations" in tool:
                            existing_funcs = tool["functionDeclarations"]
                            break

                if self.config.ccr_inject_tool and tokens_saved > 0:
                    injector = CCRToolInjector(
                        provider="google",
                        inject_tool=True,
                        inject_system_instructions=self.config.ccr_inject_system_instructions,
                    )
                    optimized_messages, injected_funcs, was_injected = injector.process_request(
                        optimized_messages, existing_funcs
                    )
                    if was_injected:
                        logger.debug(
                            f"[{request_id}] CCR: Injected retrieval tool for Google batch request {idx}"
                        )
                        existing_funcs = injected_funcs

                # Convert back to Google contents format
                optimized_contents, optimized_sys_inst = self._messages_to_gemini_contents(
                    optimized_messages
                )

                # Restore preserved content entries that had non-text parts
                for orig_idx, original_content in preserved_contents.items():
                    if orig_idx < len(optimized_contents):
                        optimized_contents[orig_idx] = original_content

                # Create compressed batch request
                compressed_req_content = {**req_content, "contents": optimized_contents}
                if optimized_sys_inst:
                    compressed_req_content["systemInstruction"] = optimized_sys_inst
                if existing_funcs is not None:
                    compressed_req_content["tools"] = [{"functionDeclarations": existing_funcs}]

                compressed_req = {
                    "request": compressed_req_content,
                    "metadata": metadata,
                }

                compressed_requests.append(compressed_req)

                if tokens_saved > 0:
                    logger.debug(
                        f"[{request_id}] Google batch request {idx}: "
                        f"{original_tokens:,} -> {optimized_tokens:,} tokens "
                        f"(saved {tokens_saved:,})"
                    )

            except Exception as e:
                logger.warning(
                    f"[{request_id}] Optimization failed for Google batch request {idx}: {e}"
                )
                # Pass through unchanged on failure — count original as optimized
                compressed_requests.append(batch_req)
                total_optimized_tokens += original_tokens  # 0 if pipeline never ran

        # Update body with compressed requests
        body["batch"]["input_config"]["requests"]["requests"] = compressed_requests

        optimization_latency = (time.time() - start_time) * 1000

        # Forward request to Google
        url = f"{self.GEMINI_API_URL}/v1beta/models/{model}:batchGenerateContent"

        # Add API key to URL if present in headers
        api_key = headers.pop("x-goog-api-key", None)
        if api_key:
            url = f"{url}?key={api_key}"

        try:
            response = await self._retry_request("POST", url, headers, body)

            # Record metrics
            await self.metrics.record_request(
                provider="google",
                model=f"batch:{model}",
                input_tokens=total_optimized_tokens,
                output_tokens=0,
                tokens_saved=total_tokens_saved,
                latency_ms=optimization_latency,
                overhead_ms=optimization_latency,
                pipeline_timing=pipeline_timing,
            )

            # Log compression stats
            if total_tokens_saved > 0:
                savings_percent = (
                    (total_tokens_saved / total_original_tokens * 100)
                    if total_original_tokens > 0
                    else 0
                )
                logger.info(
                    f"[{request_id}] Google batch compression: "
                    f"{total_original_tokens:,} -> {total_optimized_tokens:,} tokens "
                    f"({savings_percent:.1f}% saved across {len(requests_list)} requests)"
                )

            # Store batch context for CCR result processing
            if response.status_code == 200 and self.config.ccr_inject_tool:
                try:
                    response_data = response.json()
                    batch_name = response_data.get("name")
                    if batch_name:
                        await self._store_google_batch_context(
                            batch_name,
                            requests_list,
                            model,
                            api_key,
                        )
                except Exception as e:
                    logger.warning(f"[{request_id}] Failed to store Google batch context: {e}")

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
            logger.error(f"[{request_id}] Google batch request failed: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": 500,
                        "message": f"Failed to forward batch request: {e!s}",
                        "status": "INTERNAL",
                    }
                },
            )

    async def _google_batch_passthrough(
        self,
        request: Request,
        model: str,
        body: dict | None = None,
    ) -> Response:
        """Pass through Google batch request without modification."""
        from fastapi.responses import Response

        start_time = time.time()

        headers = dict(request.headers.items())
        headers.pop("host", None)
        headers.pop("content-length", None)
        # PR-A5 (P5-49): strip internal x-headroom-* before forwarding upstream.
        from headroom.proxy.helpers import _strip_internal_headers, log_outbound_headers

        _pre_strip_count_gpt = sum(1 for k in headers if k.lower().startswith("x-headroom-"))
        headers = _strip_internal_headers(headers)
        log_outbound_headers(
            forwarder="google_batch_passthrough",
            stripped_count=_pre_strip_count_gpt,
            request_id=None,
        )

        url = f"{self.GEMINI_API_URL}/v1beta/models/{model}:batchGenerateContent"

        # Add API key to URL if present in headers
        api_key = headers.pop("x-goog-api-key", None)
        if api_key:
            url = f"{url}?key={api_key}"

        # Byte-faithful body bytes (PR-A3, fixes P0-2). When ``body`` is
        # None we forward the original bytes verbatim; otherwise the dict
        # has been synthesized by Headroom and is canonically serialized.
        from headroom.proxy.helpers import (
            log_outbound_request,
            prepare_outbound_body_bytes,
            serialize_body_canonical,
        )

        if body is None:
            body_content = await request.body()
            outbound_source = "passthrough"
            body_mutated = False
        else:
            body_content = serialize_body_canonical(body)
            outbound_source = "canonical"
            body_mutated = True
        log_outbound_request(
            forwarder="google_batch_passthrough",
            method="POST",
            path=url,
            body_bytes_count=len(body_content),
            body_mutated=body_mutated,
            mutation_reasons=["google_batch_resynthesized"] if body_mutated else [],
            request_id=None,
            source=outbound_source,
        )
        # ``prepare_outbound_body_bytes`` is consulted only for the legacy
        # operator opt-in path so we honor the env-var override here too.
        from headroom.proxy.helpers import get_python_forwarder_mode

        if get_python_forwarder_mode() == "legacy_json_kwarg" and body is not None:
            outbound_bytes, _ = prepare_outbound_body_bytes(
                body=body,
                original_body_bytes=None,
                body_mutated=True,
            )
            body_content = outbound_bytes

        response = await self.http_client.post(  # type: ignore[union-attr]
            url,
            headers=headers,
            content=body_content,
        )

        # Track metrics
        latency_ms = (time.time() - start_time) * 1000
        await self.metrics.record_request(
            provider="google",
            model=f"passthrough:batch:{model}",
            input_tokens=0,
            output_tokens=0,
            tokens_saved=0,
            latency_ms=latency_ms,
        )

        response_headers = dict(response.headers)
        response_headers.pop("content-encoding", None)
        response_headers.pop("content-length", None)

        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=response_headers,
        )

    async def handle_google_batch_passthrough(
        self,
        request: Request,
        batch_name: str | None = None,
    ) -> Response:
        """Handle Google batch passthrough endpoints.

        Used for:
        - GET /v1beta/batches/{batch_name} - Get batch status
        - POST /v1beta/batches/{batch_name}:cancel - Cancel batch
        - DELETE /v1beta/batches/{batch_name} - Delete batch
        """
        from fastapi.responses import Response

        start_time = time.time()
        path = request.url.path
        url = f"{self.GEMINI_API_URL}{path}"

        # Preserve query string parameters
        if request.url.query:
            url = f"{url}?{request.url.query}"

        headers = dict(request.headers.items())
        headers.pop("host", None)
        # PR-A5 (P5-49): strip internal x-headroom-* before forwarding upstream.
        from headroom.proxy.helpers import _strip_internal_headers, log_outbound_headers

        _pre_strip_count_gp = sum(1 for k in headers if k.lower().startswith("x-headroom-"))
        headers = _strip_internal_headers(headers)
        log_outbound_headers(
            forwarder="gemini_passthrough",
            stripped_count=_pre_strip_count_gp,
            request_id=None,
        )

        # Handle API key
        api_key = headers.pop("x-goog-api-key", None)
        if api_key:
            if "?" in url:
                url = f"{url}&key={api_key}"
            else:
                url = f"{url}?key={api_key}"

        body = await request.body()

        response = await self.http_client.request(  # type: ignore[union-attr]
            method=request.method,
            url=url,
            headers=headers,
            content=body,
        )

        # Track metrics
        latency_ms = (time.time() - start_time) * 1000
        await self.metrics.record_request(
            provider="google",
            model="passthrough:batches",
            input_tokens=0,
            output_tokens=0,
            tokens_saved=0,
            latency_ms=latency_ms,
        )

        response_headers = dict(response.headers)
        response_headers.pop("content-encoding", None)
        response_headers.pop("content-length", None)

        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=response_headers,
        )

    async def _store_google_batch_context(
        self,
        batch_name: str,
        requests_list: list[dict[str, Any]],
        model: str,
        api_key: str | None,
    ) -> None:
        """Store Google batch context for CCR result processing.

        Args:
            batch_name: The batch name from the API response.
            requests_list: The original batch requests.
            model: The model used for the batch.
            api_key: The API key for continuation calls.
        """
        from headroom.ccr import BatchContext, BatchRequestContext, get_batch_context_store

        store = get_batch_context_store()
        context = BatchContext(
            batch_id=batch_name,
            provider="google",
            api_key=api_key,
            api_base_url=self.GEMINI_API_URL,
        )

        for batch_req in requests_list:
            metadata = batch_req.get("metadata", {})
            custom_id = metadata.get("key", "")
            req_content = batch_req.get("request", {})
            contents = req_content.get("contents", [])
            system_instruction = req_content.get("systemInstruction")

            # Convert contents to messages format for CCR handler
            messages, _ = self._gemini_contents_to_messages(contents, system_instruction)

            # Extract system instruction text if present
            sys_text = None
            if system_instruction:
                parts = system_instruction.get("parts", [])
                if parts and isinstance(parts[0], dict):
                    sys_text = parts[0].get("text")

            context.add_request(
                BatchRequestContext(
                    custom_id=custom_id,
                    messages=messages,
                    tools=req_content.get("tools"),
                    model=model,
                    system_instruction=sys_text,
                )
            )

        await store.store(context)
        logger.debug(
            f"Stored Google batch context for {batch_name} with {len(requests_list)} requests"
        )

    async def handle_google_batch_results(
        self,
        request: Request,
        batch_name: str,
    ) -> Response:
        """Handle Google batch results with CCR post-processing.

        Google batch results endpoint returns the batch operation status.
        When status is SUCCEEDED, results are embedded in the response.
        This handler processes CCR tool calls in those results.
        """
        from fastapi.responses import JSONResponse, Response

        from headroom.ccr import BatchResultProcessor, get_batch_context_store

        start_time = time.time()

        # Forward request to get batch status/results
        url = f"{self.GEMINI_API_URL}/v1beta/{batch_name}"

        if request.url.query:
            url = f"{url}?{request.url.query}"

        headers = dict(request.headers.items())
        headers.pop("host", None)
        # PR-A5 (P5-49): strip internal x-headroom-* before forwarding upstream.
        from headroom.proxy.helpers import _strip_internal_headers, log_outbound_headers

        _pre_strip_count_gbr = sum(1 for k in headers if k.lower().startswith("x-headroom-"))
        headers = _strip_internal_headers(headers)
        log_outbound_headers(
            forwarder="google_batch_results",
            stripped_count=_pre_strip_count_gbr,
            request_id=None,
        )

        # Handle API key
        api_key = headers.pop("x-goog-api-key", None)
        if api_key:
            if "?" in url:
                url = f"{url}&key={api_key}"
            else:
                url = f"{url}?key={api_key}"

        response = await self.http_client.get(url, headers=headers)  # type: ignore[union-attr]

        if response.status_code != 200:
            # Error - pass through
            response_headers = dict(response.headers)
            response_headers.pop("content-encoding", None)
            response_headers.pop("content-length", None)
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=response_headers,
            )

        # Parse response
        try:
            response_data = response.json()
        except json.JSONDecodeError:
            # Not JSON - pass through
            response_headers = dict(response.headers)
            response_headers.pop("content-encoding", None)
            response_headers.pop("content-length", None)
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=response_headers,
            )

        # Check if batch has results (state must be SUCCEEDED)
        metadata = response_data.get("metadata", {})
        state = metadata.get("state")

        if state != "SUCCEEDED":
            # Batch not complete - pass through
            response_headers = dict(response.headers)
            response_headers.pop("content-encoding", None)
            response_headers.pop("content-length", None)
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=response_headers,
            )

        # Extract results from response
        # Google embeds results in the batch response
        results = response_data.get("response", {}).get("responses", [])

        if not results:
            # No results to process
            response_headers = dict(response.headers)
            response_headers.pop("content-encoding", None)
            response_headers.pop("content-length", None)
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=response_headers,
            )

        # Check if we have context and CCR processing is enabled
        store = get_batch_context_store()
        batch_context = await store.get(batch_name)

        if batch_context is None or not self.config.ccr_inject_tool:
            # No context or CCR disabled - pass through
            response_headers = dict(response.headers)
            response_headers.pop("content-encoding", None)
            response_headers.pop("content-length", None)
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=response_headers,
            )

        # Process results with CCR handler
        processor = BatchResultProcessor(self.http_client)  # type: ignore[arg-type]
        processed = await processor.process_results(batch_name, results, "google")

        # Update response with processed results
        processed_results = [p.result for p in processed]
        response_data["response"]["responses"] = processed_results

        for p in processed:
            if p.was_processed:
                logger.info(
                    f"CCR: Processed Google batch result {p.custom_id} "
                    f"({p.continuation_rounds} continuation rounds)"
                )

        # Track metrics
        latency_ms = (time.time() - start_time) * 1000
        await self.metrics.record_request(
            provider="google",
            model="batch:ccr-processed",
            input_tokens=0,
            output_tokens=0,
            tokens_saved=0,
            latency_ms=latency_ms,
        )

        return JSONResponse(content=response_data, status_code=200)

    async def handle_batch_create(self, request: Request) -> Response:
        """Handle POST /v1/batches - Create a batch with compression.

        Flow:
        1. Parse request to get input_file_id
        2. Download the JSONL file content from OpenAI
        3. Parse each line and compress the messages
        4. Create a new compressed JSONL file
        5. Upload compressed file to OpenAI
        6. Create batch with the new compressed file_id
        7. Return batch object with compression stats in metadata
        """
        from fastapi.responses import JSONResponse, Response

        from headroom.proxy.helpers import _read_request_json

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
                        "type": "invalid_request_error",
                        "code": "invalid_json",
                    }
                },
            )

        input_file_id = body.get("input_file_id")
        endpoint = body.get("endpoint")
        completion_window = body.get("completion_window", "24h")
        metadata = body.get("metadata", {})

        if not input_file_id:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "input_file_id is required",
                        "type": "invalid_request_error",
                        "code": "missing_parameter",
                    }
                },
            )

        if not endpoint:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "endpoint is required",
                        "type": "invalid_request_error",
                        "code": "missing_parameter",
                    }
                },
            )

        # Only compress chat completions endpoint
        if endpoint != "/v1/chat/completions":
            # Pass through for other endpoints
            return await self._batch_passthrough(request, body)

        headers = dict(request.headers.items())
        headers.pop("host", None)
        headers.pop("content-length", None)
        # PR-A5 (P5-49): strip internal x-headroom-* before forwarding upstream.
        from headroom.proxy.helpers import _strip_internal_headers, log_outbound_headers

        _pre_strip_count_oacc = sum(1 for k in headers if k.lower().startswith("x-headroom-"))
        headers = _strip_internal_headers(headers)
        log_outbound_headers(
            forwarder="openai_batch_chat_completions",
            stripped_count=_pre_strip_count_oacc,
            request_id=request_id,
        )

        try:
            # Step 1: Download the input file from OpenAI
            logger.info(f"[{request_id}] Batch: Downloading input file {input_file_id}")
            file_content = await self._download_openai_file(input_file_id, headers)

            if file_content is None:
                return JSONResponse(
                    status_code=404,
                    content={
                        "error": {
                            "message": f"Failed to download file {input_file_id}",
                            "type": "invalid_request_error",
                            "code": "file_not_found",
                        }
                    },
                )

            # Step 2: Parse and compress each line
            logger.info(f"[{request_id}] Batch: Compressing JSONL content")
            compressed_lines, stats = await self._compress_batch_jsonl(file_content, request_id)

            if stats["total_requests"] == 0:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": {
                            "message": "No valid requests found in input file",
                            "type": "invalid_request_error",
                            "code": "empty_file",
                        }
                    },
                )

            # Step 3: Create compressed JSONL content
            compressed_content = "\n".join(compressed_lines)

            # Step 4: Upload compressed file to OpenAI
            logger.info(f"[{request_id}] Batch: Uploading compressed file")
            new_file_id = await self._upload_openai_file(
                compressed_content, f"compressed_{input_file_id}.jsonl", headers
            )

            if new_file_id is None:
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": {
                            "message": "Failed to upload compressed file",
                            "type": "server_error",
                            "code": "upload_failed",
                        }
                    },
                )

            # Step 5: Create batch with compressed file
            logger.info(f"[{request_id}] Batch: Creating batch with compressed file {new_file_id}")

            # Add compression stats to metadata
            compression_metadata = {
                **metadata,
                "headroom_compressed": "true",
                "headroom_original_file_id": input_file_id,
                "headroom_total_requests": str(stats["total_requests"]),
                "headroom_tokens_saved": str(stats["total_tokens_saved"]),
                "headroom_original_tokens": str(stats["total_original_tokens"]),
                "headroom_compressed_tokens": str(stats["total_compressed_tokens"]),
                "headroom_savings_percent": f"{stats['savings_percent']:.1f}",
            }

            batch_body = {
                "input_file_id": new_file_id,
                "endpoint": endpoint,
                "completion_window": completion_window,
                "metadata": compression_metadata,
            }

            url = f"{self.OPENAI_API_URL}/v1/batches"

            # Byte-faithful re-serialization (PR-A3, fixes P0-2).
            # batch_body is synthesized by Headroom (compressed file_id +
            # metadata), so it is treated as mutated and goes through the
            # canonical serializer.
            from headroom.proxy.helpers import (
                log_outbound_request,
                prepare_outbound_body_bytes,
            )

            outbound_bytes, outbound_source = prepare_outbound_body_bytes(
                body=batch_body,
                original_body_bytes=None,
                body_mutated=True,
            )
            outbound_headers = {**headers, "content-type": "application/json"}
            log_outbound_request(
                forwarder="batch",
                method="POST",
                path=url,
                body_bytes_count=len(outbound_bytes),
                body_mutated=True,
                mutation_reasons=["batch_compressed_file_substitution"],
                request_id=request_id,
                source=outbound_source,
            )
            response = await self.http_client.post(  # type: ignore[union-attr]
                url, content=outbound_bytes, headers=outbound_headers
            )

            total_latency = (time.time() - start_time) * 1000

            # Log compression stats
            logger.info(
                f"[{request_id}] Batch created: {stats['total_requests']} requests, "
                f"{stats['total_original_tokens']:,} -> {stats['total_compressed_tokens']:,} tokens "
                f"(saved {stats['total_tokens_saved']:,} tokens, {stats['savings_percent']:.1f}%) "
                f"in {total_latency:.0f}ms"
            )

            # Record metrics
            await self.metrics.record_request(
                provider="openai",
                model="batch",
                input_tokens=stats["total_compressed_tokens"],
                output_tokens=0,
                tokens_saved=stats["total_tokens_saved"],
                latency_ms=total_latency,
            )

            # Return response with compression info in headers
            response_headers = dict(response.headers)
            response_headers.pop("content-encoding", None)
            response_headers.pop("content-length", None)
            response_headers["x-headroom-tokens-saved"] = str(stats["total_tokens_saved"])
            response_headers["x-headroom-savings-percent"] = f"{stats['savings_percent']:.1f}"

            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=response_headers,
            )

        except Exception as e:
            logger.error(f"[{request_id}] Batch creation failed: {type(e).__name__}: {e}")
            await self.metrics.record_failed(provider="batch")
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": "An error occurred while processing the batch request",
                        "type": "server_error",
                        "code": "batch_processing_error",
                    }
                },
            )

    async def _download_openai_file(self, file_id: str, headers: dict) -> str | None:
        """Download file content from OpenAI."""
        url = f"{self.OPENAI_API_URL}/v1/files/{file_id}/content"
        try:
            response = await self.http_client.get(url, headers=headers)  # type: ignore[union-attr]
            if response.status_code == 200:
                return str(response.text)
            logger.error(f"Failed to download file {file_id}: {response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Error downloading file {file_id}: {e}")
            return None

    async def _upload_openai_file(self, content: str, filename: str, headers: dict) -> str | None:
        """Upload a file to OpenAI for batch processing."""
        url = f"{self.OPENAI_API_URL}/v1/files"

        # Prepare multipart form data
        # We need to use httpx's files parameter for multipart upload
        files = {
            "file": (filename, content.encode("utf-8"), "application/jsonl"),
        }
        data = {
            "purpose": "batch",
        }

        # Remove content-type from headers (httpx will set it for multipart)
        upload_headers = {k: v for k, v in headers.items() if k.lower() != "content-type"}

        try:
            response = await self.http_client.post(  # type: ignore[union-attr]
                url, files=files, data=data, headers=upload_headers
            )
            if response.status_code == 200:
                result = response.json()
                file_id: str | None = result.get("id")
                return file_id
            logger.error(f"Failed to upload file: {response.status_code} - {response.text}")
            return None
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            return None

    async def _compress_batch_jsonl(self, content: str, request_id: str) -> tuple[list[str], dict]:
        """Compress messages in each line of a batch JSONL file.

        Returns:
            Tuple of (compressed_lines, stats_dict)
        """
        from headroom.ccr import CCRToolInjector
        from headroom.tokenizers import get_tokenizer
        from headroom.utils import extract_user_query

        lines = content.strip().split("\n")
        compressed_lines = []
        total_original_tokens = 0
        total_compressed_tokens = 0
        total_requests = 0
        errors = 0

        tokenizer = get_tokenizer("gpt-4")  # Use gpt-4 tokenizer for batch

        for i, line in enumerate(lines):
            if not line.strip():
                continue

            try:
                request_obj = json.loads(line)
                body = request_obj.get("body", {})
                messages = body.get("messages", [])
                model = body.get("model", "gpt-4")

                if not messages:
                    # No messages to compress, pass through
                    compressed_lines.append(line)
                    total_requests += 1
                    continue

                # Compress messages using the OpenAI pipeline
                if self.config.optimize:
                    try:
                        context_limit = self.openai_provider.get_context_limit(model)
                        result = self.openai_pipeline.apply(
                            messages=messages,
                            model=model,
                            model_limit=context_limit,
                            context=extract_user_query(messages),
                        )
                        compressed_messages = result.messages
                        # Use pipeline's token counts for consistency with pipeline logs
                        original_tokens = result.tokens_before
                        compressed_tokens = result.tokens_after
                    except Exception as e:
                        logger.warning(f"[{request_id}] Compression failed for line {i}: {e}")
                        compressed_messages = messages
                        original_tokens = tokenizer.count_messages(messages)
                        compressed_tokens = original_tokens
                else:
                    compressed_messages = messages
                    original_tokens = tokenizer.count_messages(messages)
                    compressed_tokens = original_tokens

                total_original_tokens += original_tokens
                total_compressed_tokens += compressed_tokens
                tokens_saved = original_tokens - compressed_tokens

                # CCR Tool Injection: Inject retrieval tool if compression occurred
                tools = body.get("tools")
                if self.config.ccr_inject_tool and tokens_saved > 0:
                    injector = CCRToolInjector(
                        provider="openai",
                        inject_tool=True,
                        inject_system_instructions=self.config.ccr_inject_system_instructions,
                    )
                    compressed_messages, tools, was_injected = injector.process_request(
                        compressed_messages, tools
                    )
                    if was_injected:
                        logger.debug(
                            f"[{request_id}] CCR: Injected retrieval tool for batch line {i}"
                        )

                # Update body with compressed messages
                body["messages"] = compressed_messages
                if tools is not None:
                    body["tools"] = tools
                request_obj["body"] = body

                compressed_lines.append(json.dumps(request_obj))
                total_requests += 1

            except json.JSONDecodeError as e:
                logger.warning(f"[{request_id}] Invalid JSON on line {i}: {e}")
                errors += 1
                # Keep original line on error
                compressed_lines.append(line)
                total_requests += 1

        total_tokens_saved = total_original_tokens - total_compressed_tokens
        savings_percent = (
            (total_tokens_saved / total_original_tokens * 100) if total_original_tokens > 0 else 0
        )

        stats = {
            "total_requests": total_requests,
            "total_original_tokens": total_original_tokens,
            "total_compressed_tokens": total_compressed_tokens,
            "total_tokens_saved": total_tokens_saved,
            "savings_percent": savings_percent,
            "errors": errors,
        }

        return compressed_lines, stats

    async def _batch_passthrough(self, request: Request, body: dict) -> Response:
        """Pass through batch request to OpenAI without compression.

        Byte-faithful (PR-A3, fixes P0-2). The original request bytes are
        preserved verbatim when no transform mutated the body.
        """
        from fastapi.responses import Response

        from headroom.proxy.helpers import (
            _read_request_body_bytes,
            _strip_internal_headers,
            log_outbound_headers,
            log_outbound_request,
            prepare_outbound_body_bytes,
        )

        headers = dict(request.headers.items())
        headers.pop("host", None)
        headers.pop("content-length", None)
        # PR-A5 (P5-49): strip internal x-headroom-* before forwarding upstream.
        _pre_strip_count_obp = sum(1 for k in headers if k.lower().startswith("x-headroom-"))
        headers = _strip_internal_headers(headers)
        log_outbound_headers(
            forwarder="openai_batch_passthrough",
            stripped_count=_pre_strip_count_obp,
            request_id=None,
        )

        url = f"{self.OPENAI_API_URL}/v1/batches"

        # Best effort: capture the original (decompressed) bytes so the
        # passthrough is truly byte-faithful. If the body was already
        # consumed upstream we fall through to canonical re-serialization.
        try:
            original_body_bytes: bytes | None = await _read_request_body_bytes(request)
        except Exception:
            original_body_bytes = None

        outbound_bytes, outbound_source = prepare_outbound_body_bytes(
            body=body,
            original_body_bytes=original_body_bytes,
            body_mutated=False,
        )
        outbound_headers = {**headers, "content-type": "application/json"}
        log_outbound_request(
            forwarder="batch_passthrough",
            method="POST",
            path=url,
            body_bytes_count=len(outbound_bytes),
            body_mutated=False,
            mutation_reasons=[],
            request_id=None,
            source=outbound_source,
        )
        response = await self.http_client.post(  # type: ignore[union-attr]
            url, content=outbound_bytes, headers=outbound_headers
        )

        response_headers = dict(response.headers)
        response_headers.pop("content-encoding", None)
        response_headers.pop("content-length", None)

        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=response_headers,
        )

    async def handle_batch_list(self, request: Request) -> Response:
        """Handle GET /v1/batches - List batches (passthrough)."""
        return await self.handle_passthrough(request, self.OPENAI_API_URL)

    async def handle_batch_get(self, request: Request, batch_id: str) -> Response:
        """Handle GET /v1/batches/{batch_id} - Get batch (passthrough)."""
        return await self.handle_passthrough(request, self.OPENAI_API_URL)

    async def handle_batch_cancel(self, request: Request, batch_id: str) -> Response:
        """Handle POST /v1/batches/{batch_id}/cancel - Cancel batch (passthrough)."""
        return await self.handle_passthrough(request, self.OPENAI_API_URL)
