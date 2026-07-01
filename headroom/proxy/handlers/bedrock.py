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

SigV4. Rewriting the body invalidates the caller's SigV4 signature (the
signature covers a hash of the body). There are two ways to forward safely:

* ``--bedrock-api-url`` / ``BEDROCK_TARGET_API_URL`` — forward to a gateway that
  re-signs or does not verify the inbound signature (LiteLLM, LocalStack, a
  corporate Bedrock proxy). Forwarded verbatim; Headroom does not re-sign.
* ``--bedrock-sign`` / ``HEADROOM_BEDROCK_SIGN`` — re-sign the compressed body
  with SigV4 and forward **direct to the regional AWS endpoint**. This is what
  powers ``headroom wrap claude`` under ``CLAUDE_CODE_USE_BEDROCK=1`` with no
  gateway. See ``headroom/proxy/bedrock_signer.py``.

The routes register when **either** is configured. ``bedrock_api_url`` takes
precedence when both are set (an explicit gateway is forwarded to verbatim).

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

    def _bedrock_signing_enabled(self) -> bool:
        """True when we re-sign direct-to-AWS (no gateway configured)."""
        # An explicit gateway wins: forward to it verbatim, never re-sign.
        if getattr(self.config, "bedrock_api_url", None):  # type: ignore[attr-defined]
            return False
        return bool(getattr(self.config, "bedrock_sign", False))  # type: ignore[attr-defined]

    def _bedrock_signer(self):  # type: ignore[no-untyped-def]
        """Lazily build (and cache) the SigV4 signer for direct-to-AWS mode."""
        signer = getattr(self, "_cached_bedrock_signer", None)
        if signer is None:
            from headroom.proxy.bedrock_signer import BedrockSigner

            signer = BedrockSigner(
                region=getattr(self.config, "bedrock_region", "us-west-2"),  # type: ignore[attr-defined]
                profile=getattr(self.config, "bedrock_profile", None),  # type: ignore[attr-defined]
            )
            self._cached_bedrock_signer = signer
        return signer

    def _bedrock_upstream_base(self) -> str | None:
        """Resolved Bedrock upstream, or ``None`` when unconfigured.

        Precedence: an explicit ``config.bedrock_api_url`` (a re-signing
        gateway) wins; otherwise, when ``config.bedrock_sign`` is set, the
        regional AWS Bedrock runtime endpoint derived from the configured
        region. ``None`` means the feature is off — the routes are not even
        registered in that case, so a ``None`` here is a defensive guard only.
        """
        base = getattr(self.config, "bedrock_api_url", None)  # type: ignore[attr-defined]
        if base:
            return base.rstrip("/")
        if getattr(self.config, "bedrock_sign", False):  # type: ignore[attr-defined]
            return self._bedrock_signer().endpoint_base()
        return None

    def _sign_if_needed(
        self,
        signing: bool,
        *,
        url: str,
        body: bytes,
        inbound_headers: dict[str, str],
    ) -> dict[str, str]:
        """Return outbound headers, SigV4-re-signed when in direct-to-AWS mode.

        In gateway mode (``signing`` is False) the inbound headers pass through
        unchanged — the gateway owns signing. In direct-to-AWS mode the headers
        carry a fresh signature computed over ``body`` (the exact bytes about to
        be forwarded). If signing fails (no credentials), we fail loud rather
        than forward an unsigned request that AWS would 403 anyway.
        """
        if not signing:
            return inbound_headers
        signer = self._bedrock_signer()
        return signer.sign(url=url, body=body, inbound_headers=inbound_headers)

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
                        "message": (
                            "Bedrock passthrough requested but neither "
                            "--bedrock-api-url nor --bedrock-sign is set."
                        ),
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
        # In gateway mode we never touch the auth headers — the upstream gateway
        # owns (re-)signing. In direct-to-AWS signing mode we strip the stale
        # signature and recompute it over the bytes we actually forward (see
        # ``_sign_if_needed``).
        signing = self._bedrock_signing_enabled()
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
            # Even verbatim bytes must be re-signed in direct-to-AWS mode: we
            # changed the Host to the regional endpoint, so the inbound
            # signature no longer matches regardless of the body.
            fwd_headers = self._sign_if_needed(
                signing, url=url, body=raw_only, inbound_headers=verbatim_headers
            )
            return await self._forward_bedrock(
                url=url,
                headers=fwd_headers,
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
        # Sign LAST: the signature must hash exactly the bytes we forward
        # (``outbound``), after compression has finalized them.
        fwd_headers = self._sign_if_needed(
            signing, url=url, body=outbound, inbound_headers=out_headers
        )
        response = await self._forward_bedrock(
            url=url,
            headers=fwd_headers,
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
