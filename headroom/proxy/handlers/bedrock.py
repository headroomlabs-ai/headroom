"""AWS Bedrock ``InvokeModel`` passthrough handler for HeadroomProxy.

Claude Code (and other clients) launched with ``CLAUDE_CODE_USE_BEDROCK=1``
talk to a Bedrock *runtime endpoint* over plain HTTP, POSTing to
``/model/{modelId}/invoke`` and ``/model/{modelId}/invoke-with-response-stream``
instead of the Anthropic ``/v1/messages`` route. Those requests previously fell
through Headroom's catch-all and were forwarded verbatim — no compression.

This mixin intercepts that Bedrock REST shape, compresses the request body with
the **same** ``anthropic_pipeline`` used for ``/v1/messages``, and forwards to a
configurable upstream (``config.bedrock_api_url``). The InvokeModel body for
Anthropic models *is* the Anthropic Messages shape
(``{anthropic_version, system, messages, max_tokens, …}``; the model travels in
the URL), so the existing pipeline applies with no translation.

LIMITATION — SigV4. Rewriting the body invalidates the caller's SigV4 signature
(the signature covers a hash of the body). These routes therefore register
**only** when ``--bedrock-api-url`` / ``BEDROCK_TARGET_API_URL`` is set, and the
target must be a gateway that re-signs or does not verify the inbound signature
(LiteLLM, LocalStack, a corporate Bedrock proxy) — never raw AWS. For
direct-to-AWS compression use ``--backend bedrock`` (which re-signs).

The response is forwarded byte-faithfully: the non-streaming reply is Anthropic
JSON and the streaming reply uses AWS event-stream binary framing — neither is
parsed or mutated, since all compression happens request-side.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING
from urllib.parse import quote

if TYPE_CHECKING:
    from fastapi import Request
    from fastapi.responses import Response, StreamingResponse

logger = logging.getLogger("headroom.proxy")

LOG_TAG = "bedrock_invoke"


class BedrockHandlerMixin:
    """Mixin providing the Bedrock InvokeModel passthrough handler."""

    def _bedrock_upstream_base(self) -> str | None:
        """Resolved Bedrock upstream, or ``None`` when unconfigured.

        Returns the normalized ``config.bedrock_api_url`` (trailing slash
        stripped). ``None`` means the feature is off — the routes are not even
        registered in that case, so a ``None`` here is a defensive guard only.
        """
        base = getattr(self.config, "bedrock_api_url", None)  # type: ignore[attr-defined]
        return base.rstrip("/") if base else None

    async def handle_bedrock_invoke(
        self,
        request: Request,
        model_id: str,
        *,
        stream: bool,
    ) -> Response | StreamingResponse:
        """Compress and forward a Bedrock ``InvokeModel`` request.

        Args:
            request: The inbound FastAPI request.
            model_id: The Bedrock model / inference-profile id captured from the
                URL path (may contain ``.``, ``:`` and ``/``).
            stream: ``True`` for ``invoke-with-response-stream``.
        """
        from fastapi.responses import JSONResponse

        from headroom.proxy.auth_mode import classify_client
        from headroom.proxy.helpers import (
            COMPRESSION_TIMEOUT_SECONDS,
            MAX_MESSAGE_ARRAY_LENGTH,
            _headroom_bypass_enabled,
            _strip_internal_headers,
            extract_tags,
            read_request_json_with_bytes,
        )
        from headroom.proxy.modes import is_cache_mode
        from headroom.utils import extract_user_query

        start_time = time.time()
        request_id = await self._next_request_id()  # type: ignore[attr-defined]

        base = self._bedrock_upstream_base()
        if base is None:
            # Routes only register when configured, so this is unreachable in
            # practice; fail loud rather than silently forwarding nowhere.
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "type": "configuration_error",
                        "message": "Bedrock passthrough requested but --bedrock-api-url is unset.",
                    }
                },
            )

        suffix = "invoke-with-response-stream" if stream else "invoke"
        url = f"{base}/model/{quote(model_id, safe='')}/{suffix}"
        if request.url.query:
            url = f"{url}?{request.url.query}"

        # Outbound headers (case-insensitive drops). Two header sets:
        #   - verbatim: forwards the original bytes, so the inbound
        #     content-length / content-encoding still describe the body.
        #   - rewritten: the body we forward is decompressed JSON (possibly
        #     compressed by the pipeline), so content-length must be recomputed
        #     by httpx and the stale content-encoding dropped. Keeping the
        #     inbound content-length here is the classic "Too little data for
        #     declared Content-Length" footgun once the body shrinks.
        # We never touch the auth headers — the upstream gateway owns
        # (re-)signing.
        in_headers = _strip_internal_headers(dict(request.headers.items()))
        client = classify_client(dict(request.headers.items()))
        tags = extract_tags(dict(request.headers.items()))
        verbatim_drop = {"host", "accept-encoding"}
        rewritten_drop = verbatim_drop | {"content-length", "content-encoding"}
        verbatim_headers = {k: v for k, v in in_headers.items() if k.lower() not in verbatim_drop}
        out_headers = {k: v for k, v in in_headers.items() if k.lower() not in rewritten_drop}

        # Read the body up front so we can fail open to a verbatim forward on any
        # parse error (a malformed body is the gateway's problem, not ours).
        try:
            body, raw = await read_request_json_with_bytes(request)
        except Exception as err:
            logger.warning(
                "[%s] %s could not parse body; forwarding verbatim: %s",
                request_id,
                LOG_TAG,
                err,
            )
            raw_only = await request.body()
            return await self._forward_bedrock(
                url=url,
                headers=verbatim_headers,
                content=raw_only,
                stream=stream,
                request_id=request_id,
            )

        messages = body.get("messages")
        bypass = (
            _headroom_bypass_enabled(request.headers)
            or not getattr(self.config, "optimize", True)  # type: ignore[attr-defined]
            or is_cache_mode(getattr(self.config, "mode", "token"))  # type: ignore[attr-defined]
            or not isinstance(messages, list)
            or not messages
            or len(messages) > MAX_MESSAGE_ARRAY_LENGTH
        )

        outbound = raw
        original_tokens = 0
        optimized_tokens = 0
        tokens_saved = 0
        transforms_applied: tuple[str, ...] = ()
        pipeline_timing: dict[str, float] | None = None

        if not bypass:
            try:
                context_limit = self.anthropic_provider.get_context_limit(model_id)  # type: ignore[attr-defined]
                result = await self._run_compression_in_executor(  # type: ignore[attr-defined]
                    lambda: self.anthropic_pipeline.apply(  # type: ignore[attr-defined]
                        messages=messages,
                        model=model_id,
                        model_limit=context_limit,
                        context=extract_user_query(messages),
                        request_id=request_id,
                    ),
                    timeout=COMPRESSION_TIMEOUT_SECONDS,
                )
                if result.messages != messages:
                    body["messages"] = result.messages
                    outbound = json.dumps(body).encode("utf-8")
                    original_tokens = result.tokens_before
                    optimized_tokens = result.tokens_after
                    tokens_saved = max(0, result.tokens_before - result.tokens_after)
                    transforms_applied = tuple(result.transforms_applied)
                    pipeline_timing = result.timing
                    logger.info(
                        "[%s] %s compressed %d→%d tokens (%d saved) model=%s",
                        request_id,
                        LOG_TAG,
                        result.tokens_before,
                        result.tokens_after,
                        tokens_saved,
                        model_id,
                    )
            except Exception as err:
                # Fail open: never break a request because compression failed.
                logger.warning(
                    "[%s] %s compression failed; forwarding verbatim: %s",
                    request_id,
                    LOG_TAG,
                    err,
                )
                outbound = raw

        out_headers["content-type"] = "application/json"
        response = await self._forward_bedrock(
            url=url,
            headers=out_headers,
            content=outbound,
            stream=stream,
            request_id=request_id,
        )

        # Best-effort metrics. Output tokens are left at 0 (the RequestOutcome
        # contract treats 0 as "not measured") — Bedrock responses are forwarded
        # byte-faithfully and never parsed. The valuable figure, request-side
        # compression, is recorded in full.
        try:
            from headroom.proxy.outcome import RequestOutcome

            await self._record_request_outcome(  # type: ignore[attr-defined]
                RequestOutcome(
                    request_id=request_id,
                    provider="bedrock",
                    model=model_id,
                    original_tokens=original_tokens,
                    optimized_tokens=optimized_tokens,
                    output_tokens=0,
                    tokens_saved=tokens_saved,
                    attempted_input_tokens=original_tokens,
                    total_latency_ms=(time.time() - start_time) * 1000,
                    transforms_applied=transforms_applied,
                    pipeline_timing=pipeline_timing,
                    tags=tags,
                    client=client,
                )
            )
        except Exception:
            logger.debug("[%s] %s outcome recording failed", request_id, LOG_TAG, exc_info=True)

        return response

    async def _forward_bedrock(
        self,
        *,
        url: str,
        headers: dict[str, str],
        content: bytes,
        stream: bool,
        request_id: str,
    ) -> Response | StreamingResponse:
        """Stream a request to the Bedrock upstream, byte-faithfully.

        Uses the canonical httpx-as-reverse-proxy pattern: open the upstream
        with ``stream=True`` so status + headers are available immediately, then
        hand the raw byte iterator to ``StreamingResponse`` and close the
        upstream connection via a background task. Works for both the JSON
        ``invoke`` reply and the event-stream ``invoke-with-response-stream``
        reply — neither is buffered or mutated.
        """
        import httpx
        from fastapi.responses import JSONResponse, StreamingResponse
        from starlette.background import BackgroundTask

        assert self.http_client is not None  # type: ignore[attr-defined]
        upstream_request = self.http_client.build_request(  # type: ignore[attr-defined]
            "POST",
            url,
            headers=headers,
            content=content,
        )
        try:
            upstream = await self.http_client.send(upstream_request, stream=True)  # type: ignore[attr-defined]
        except (httpx.ConnectError, httpx.TimeoutException) as err:
            logger.warning("[%s] %s upstream connect failed: %s", request_id, LOG_TAG, err)
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "type": "connection_error",
                        "message": f"Failed to connect to Bedrock upstream: {err}",
                    }
                },
            )

        # Forward raw (still-encoded) bytes, so strip hop-by-hop headers that
        # would conflict with StreamingResponse's own framing. content-encoding
        # and content-type are preserved.
        resp_headers = {
            k: v
            for k, v in upstream.headers.items()
            if k.lower() not in ("content-length", "transfer-encoding", "connection")
        }
        media_type = upstream.headers.get("content-type")
        return StreamingResponse(
            upstream.aiter_raw(),
            status_code=upstream.status_code,
            headers=resp_headers,
            media_type=media_type,
            background=BackgroundTask(upstream.aclose),
        )
