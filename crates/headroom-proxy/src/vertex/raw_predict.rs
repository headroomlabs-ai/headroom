//! POST handler for Vertex `:rawPredict` (non-streaming).
//!
//! Path:
//! ```text
//! POST /v1beta1/projects/{project}/locations/{location}/publishers/anthropic/models/{model}:rawPredict
//! ```
//!
//! See [`super`] for the module-level rationale (envelope shape, ADC,
//! routing strategy). This handler:
//!
//! 1. Buffers the request body.
//! 2. Confirms the Vertex envelope shape (`anthropic_version` present,
//!    `model` absent). On envelope mismatch, logs
//!    `event = "vertex_envelope_invalid"` and returns 400.
//! 3. Runs live-zone Anthropic compression — the body is the same
//!    Anthropic Messages shape `/v1/messages` accepts, just with
//!    `anthropic_version` instead of `model`. The dispatcher
//!    preserves `anthropic_version` byte-equal because the
//!    `RawValue`-based surgery only rewrites `messages[*]` entries.
//! 4. Resolves the ADC bearer token (cached, refreshed ahead of
//!    expiry) and attaches `Authorization: Bearer <token>`. On ADC
//!    failure, logs `event = "vertex_adc_fetch_failed"` and returns
//!    502 — never silently forwards unauthenticated.
//! 5. Forwards to the configured Vertex endpoint
//!    (`https://{region}-aiplatform.googleapis.com/<path>`).
//! 6. Streams the response body back unchanged.
//!
//! All decision points emit a structured `tracing::info!` /
//! `tracing::warn!` event so operators can confirm the pipeline is
//! engaged in dashboards.

use axum::body::{to_bytes, Body};
use axum::extract::{ConnectInfo, Path, State};
use axum::http::{HeaderMap, Method, StatusCode, Uri};
use axum::response::Response;
use std::net::SocketAddr;

use crate::compression;
use crate::headers::{build_forward_request_headers, filter_response_headers};
use crate::proxy::AppState;
use crate::vertex::{adc::TokenSourceError, envelope, split_model_action, VertexVerb};

/// Axum handler for the rawPredict route.
///
/// Path parameters:
/// - `project` — GCP project ID.
/// - `location` — Vertex region (e.g. `us-central1`).
/// - `model_action` — combined `<model_id>:<verb>` segment. We split
///   on the last `:` and dispatch on the verb. The streaming sibling
///   `:streamRawPredict` lives in [`super::stream_raw_predict`] and
///   is registered under the same route shape.
pub async fn handle_raw_predict(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    Path((project, location, model_action)): Path<(String, String, String)>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Body,
) -> Response {
    let request_id = headers
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    let (model_id, verb) = match split_model_action(&model_action) {
        Some(parts) => parts,
        None => {
            tracing::warn!(
                event = "vertex_path_parse_failed",
                request_id = %request_id,
                path = %uri.path(),
                segment = %model_action,
                "vertex path final segment missing `:verb` separator"
            );
            return error_response(StatusCode::NOT_FOUND, "vertex path: bad model_action");
        }
    };
    let parsed_verb = match VertexVerb::parse(verb) {
        Some(VertexVerb::RawPredict) => VertexVerb::RawPredict,
        Some(VertexVerb::StreamRawPredict) => {
            // Wrong handler. The router mounts both verbs at this
            // path shape; the dispatcher inside [`super::stream_raw_predict`]
            // wraps the streaming case. If we ever land here it's a
            // routing bug — log and 500 rather than silently forward.
            tracing::error!(
                event = "vertex_verb_routing_bug",
                request_id = %request_id,
                verb = "streamRawPredict",
                "streamRawPredict request reached non-streaming handler"
            );
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "vertex routing bug: stream verb on non-stream handler",
            );
        }
        None => {
            tracing::warn!(
                event = "vertex_unknown_verb",
                request_id = %request_id,
                verb = %verb,
                "vertex path verb not recognized; only rawPredict / streamRawPredict are supported"
            );
            return error_response(StatusCode::NOT_FOUND, "vertex: unknown verb");
        }
    };

    forward_vertex_request(
        state,
        client_addr,
        request_id,
        method,
        uri,
        headers,
        body,
        VertexCallContext {
            project,
            location,
            model_id: model_id.to_string(),
            verb: parsed_verb,
        },
        /* attach_sse_tee */ false,
    )
    .await
}

/// Carrier struct for the bits parsed out of the URL path; passed
/// down so logs and error paths share a consistent set of fields.
#[derive(Debug, Clone)]
pub(crate) struct VertexCallContext {
    pub project: String,
    pub location: String,
    pub model_id: String,
    pub verb: VertexVerb,
}

/// Shared forwarder used by both `:rawPredict` and `:streamRawPredict`
/// handlers. Streaming-specific behaviour lives in the response-side
/// SSE tee in [`crate::proxy::forward_http`] (which we don't reuse
/// here — Vertex's path is not in `is_compressible_path` and we have
/// our own envelope handling). For PR-D4 the streaming handler just
/// passes through with the same auth + envelope + log surface; the
/// upstream SSE bytes flow back to the client unchanged.
///
/// Note: this function takes 9 arguments. Grouping them into a
/// struct (an obvious clippy fix) would obscure that each argument
/// is a distinct axum extractor / handler-supplied value. The
/// argument list mirrors the catch-all `forward_http` in
/// [`crate::proxy`]; consistency wins over the pedantic lint.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn forward_vertex_request(
    state: AppState,
    client_addr: SocketAddr,
    request_id: String,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Body,
    ctx: VertexCallContext,
    attach_sse_tee: bool,
) -> Response {
    let path_for_log = uri.path().to_string();

    // ─── 1. BUFFER BODY ────────────────────────────────────────────────
    let max = state.config.compression_max_body_bytes as usize;
    let buffered = match to_bytes(body, max).await {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!(
                event = "vertex_body_too_large",
                request_id = %request_id,
                path = %path_for_log,
                limit_bytes = max,
                error = %e,
                "vertex request body exceeds compression buffer limit; failing loudly"
            );
            return error_response(
                StatusCode::PAYLOAD_TOO_LARGE,
                "request body exceeds buffer limit",
            );
        }
    };

    // ─── 2. ENVELOPE PARSE ─────────────────────────────────────────────
    match envelope::parse(&buffered) {
        Ok(env) => {
            tracing::info!(
                event = "vertex_envelope_parsed",
                request_id = %request_id,
                path = %path_for_log,
                project = %ctx.project,
                location = %ctx.location,
                model = %ctx.model_id,
                verb = ctx.verb.as_str(),
                anthropic_version = %env.anthropic_version,
                has_messages = env.has_messages,
                "vertex envelope detected"
            );
        }
        Err(e) => {
            tracing::warn!(
                event = "vertex_envelope_invalid",
                request_id = %request_id,
                path = %path_for_log,
                model = %ctx.model_id,
                verb = ctx.verb.as_str(),
                error = %e,
                "vertex envelope did not match expected shape; rejecting with 400"
            );
            return error_response(StatusCode::BAD_REQUEST, "vertex envelope invalid");
        }
    }

    // ─── 3. LIVE-ZONE COMPRESSION (when enabled) ───────────────────────
    //
    // Vertex bodies are Anthropic-shape; we feed the same
    // `compress_anthropic_request` dispatcher that runs on /v1/messages.
    // The dispatcher uses RawValue-based surgery so `anthropic_version`
    // (and any other non-`messages` top-level field) round-trips
    // byte-equal. Compression off → buffered bytes used unchanged.
    let body_to_send = if state.config.compression {
        // PR-E3: Vertex uses GCP ADC bearer-token auth downstream, not
        // Anthropic credentials, so the PAYG/OAuth/subscription
        // classification doesn't apply. Hard-code `AuthMode::OAuth` to
        // skip E3 cache_control auto-placement (and any other PAYG-only
        // mutation). Live-zone compression itself continues to run.
        let outcome = compression::compress_anthropic_request(
            &buffered,
            state.config.compression_mode,
            state.config.cache_control_auto_frozen,
            headroom_core::auth_mode::AuthMode::OAuth,
            &request_id,
        );
        match outcome {
            compression::Outcome::NoCompression => {
                tracing::info!(
                    event = "vertex_compression_skipped",
                    request_id = %request_id,
                    path = %path_for_log,
                    compression_mode = state.config.compression_mode.as_str(),
                    reason = "no_compression",
                    "vertex live-zone dispatcher returned NoCompression"
                );
                buffered
            }
            compression::Outcome::Compressed {
                body,
                tokens_before,
                tokens_after,
                strategies_applied,
                markers_inserted,
            } => {
                tracing::info!(
                    event = "vertex_compression_applied",
                    request_id = %request_id,
                    path = %path_for_log,
                    tokens_before = tokens_before,
                    tokens_after = tokens_after,
                    tokens_freed = tokens_before.saturating_sub(tokens_after),
                    strategies = ?strategies_applied,
                    markers = markers_inserted.len(),
                    "vertex live-zone compression applied"
                );
                body
            }
            compression::Outcome::Passthrough { reason } => {
                tracing::warn!(
                    event = "vertex_compression_passthrough",
                    request_id = %request_id,
                    path = %path_for_log,
                    reason = ?reason,
                    "vertex live-zone dispatcher passthrough on parse/serialize"
                );
                buffered
            }
        }
    } else {
        tracing::info!(
            event = "vertex_compression_skipped",
            request_id = %request_id,
            path = %path_for_log,
            reason = "compression_off",
            "compression master switch off; vertex body forwarded unchanged"
        );
        buffered
    };

    // ─── 4. RESOLVE BEARER TOKEN ───────────────────────────────────────
    let bearer = match state.vertex_token_source.bearer().await {
        Ok(t) => t,
        Err(e) => {
            // Per project rule "no silent fallbacks": never forward
            // unauthenticated. Surface the failure as a structured
            // 502 so operators see the cause clearly in logs.
            tracing::error!(
                event = "vertex_adc_fetch_failed",
                request_id = %request_id,
                path = %path_for_log,
                model = %ctx.model_id,
                verb = ctx.verb.as_str(),
                error = %e,
                "vertex ADC bearer token fetch failed; refusing to forward unauthenticated"
            );
            let status = match e {
                TokenSourceError::ProviderInit(_) => StatusCode::BAD_GATEWAY,
                TokenSourceError::Fetch(_) => StatusCode::BAD_GATEWAY,
            };
            return error_response(status, "vertex ADC token fetch failed");
        }
    };

    // ─── 5. BUILD UPSTREAM URL ─────────────────────────────────────────
    //
    // The Vertex endpoint pattern is
    // `https://{region}-aiplatform.googleapis.com/<path-and-query>`.
    // We honour the same `Config::upstream` override pattern the
    // rest of the proxy uses: when an operator sets `upstream` to the
    // mock server (typical in tests), we forward there and the
    // request still carries the canonical Vertex path.
    //
    // For production, the operator should set `upstream` to the
    // regional Vertex host. We do NOT auto-construct the regional
    // URL from `vertex_region` — that would be a hardcoded provider
    // routing decision. The region setting is exposed for
    // observability only.
    let upstream_url = match crate::proxy::build_upstream_url(&state.config.upstream, &uri) {
        Ok(u) => u,
        Err(e) => {
            tracing::error!(
                event = "vertex_upstream_url_failed",
                request_id = %request_id,
                error = %e,
                "could not construct vertex upstream URL"
            );
            return error_response(StatusCode::BAD_GATEWAY, "vertex upstream URL build failed");
        }
    };

    // ─── 6. BUILD HEADERS ──────────────────────────────────────────────
    let strip_internal = state.config.strip_internal_headers.is_enabled();
    let forwarded_host = headers
        .get(http::header::HOST)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let mut outgoing_headers = build_forward_request_headers(
        &headers,
        client_addr.ip(),
        "http",
        forwarded_host.as_deref(),
        &request_id,
        strip_internal,
    );
    if !state.config.rewrite_host {
        if let Some(h) = headers.get(http::header::HOST) {
            outgoing_headers.insert(http::header::HOST, h.clone());
        }
    }
    // Attach the bearer; if the client already sent an Authorization
    // header we replace it (Vertex rejects the wrong Auth flavour
    // anyway, so keeping the client-provided value would silently
    // break the call).
    match http::HeaderValue::from_str(&format!("Bearer {bearer}")) {
        Ok(v) => {
            outgoing_headers.insert(http::header::AUTHORIZATION, v);
        }
        Err(e) => {
            tracing::error!(
                event = "vertex_authorization_invalid",
                request_id = %request_id,
                error = %e,
                "ADC bearer token contained invalid header bytes; refusing to forward"
            );
            return error_response(StatusCode::BAD_GATEWAY, "vertex auth header build failed");
        }
    }

    // ─── 7. FORWARD ────────────────────────────────────────────────────
    let reqwest_method = match reqwest::Method::from_bytes(method.as_str().as_bytes()) {
        Ok(m) => m,
        Err(e) => {
            tracing::error!(
                event = "vertex_method_invalid",
                request_id = %request_id,
                method = %method,
                error = %e,
                "could not convert axum method to reqwest method"
            );
            return error_response(StatusCode::BAD_REQUEST, "vertex method invalid");
        }
    };
    let upstream_resp = match state
        .client
        .request(reqwest_method, upstream_url.clone())
        .headers(outgoing_headers)
        .body(body_to_send)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(
                event = "vertex_upstream_error",
                request_id = %request_id,
                path = %path_for_log,
                error = %e,
                "vertex upstream call failed"
            );
            return error_response(StatusCode::BAD_GATEWAY, "vertex upstream error");
        }
    };

    // ─── 8. STREAM RESPONSE ────────────────────────────────────────────
    let upstream_status = upstream_resp.status();
    let status = StatusCode::from_u16(upstream_status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let resp_headers = filter_response_headers(upstream_resp.headers());

    // PR-C1 reuse: when `attach_sse_tee` is set AND the upstream
    // response is `text/event-stream`, tee bytes into a bounded mpsc
    // and drive `AnthropicStreamState` in a spawned task — same shape
    // as the `/v1/messages` SSE telemetry tee in
    // `crate::proxy::forward_http`. The byte-passthrough path is
    // unaffected by the tee (best-effort `try_send`, bounded channel).
    let is_sse = upstream_resp
        .headers()
        .get(http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|s| {
            let media = s.split(';').next().unwrap_or("").trim();
            media.eq_ignore_ascii_case("text/event-stream")
        })
        .unwrap_or(false);
    let parser_tx = if attach_sse_tee && is_sse {
        let (tx, rx) = tokio::sync::mpsc::channel::<bytes::Bytes>(VERTEX_SSE_QUEUE_DEPTH);
        let rid = request_id.clone();
        tokio::spawn(run_anthropic_sse_state_machine(rx, rid));
        tracing::info!(
            event = "vertex_sse_tee_engaged",
            request_id = %request_id,
            "vertex stream_raw_predict SSE telemetry tee engaged"
        );
        Some(tx)
    } else {
        None
    };

    use futures_util::StreamExt as _;
    let rid_for_stream = request_id.clone();
    let resp_stream = upstream_resp.bytes_stream().map(move |r| match r {
        Ok(b) => {
            if let Some(tx) = &parser_tx {
                if let Err(e) = tx.try_send(b.clone()) {
                    tracing::debug!(
                        request_id = %rid_for_stream,
                        error = %e,
                        "vertex sse parser queue full or closed; skipping telemetry chunk"
                    );
                }
            }
            Ok(b)
        }
        Err(e) => {
            tracing::warn!(
                request_id = %rid_for_stream,
                error = %e,
                "vertex upstream stream error mid-response"
            );
            Err(e)
        }
    });
    let body = Body::from_stream(resp_stream);

    let mut response = Response::builder().status(status);
    if let Some(h) = response.headers_mut() {
        h.extend(resp_headers);
        if let Ok(v) = http::HeaderValue::from_str(&request_id) {
            h.insert(http::HeaderName::from_static("x-request-id"), v);
        }
    }
    let response = match response.body(body) {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(
                event = "vertex_response_build_failed",
                request_id = %request_id,
                error = %e,
                "could not build vertex response"
            );
            return error_response(StatusCode::INTERNAL_SERVER_ERROR, "response build failed");
        }
    };

    tracing::info!(
        event = "vertex_forwarded",
        request_id = %request_id,
        path = %path_for_log,
        project = %ctx.project,
        location = %ctx.location,
        model = %ctx.model_id,
        verb = ctx.verb.as_str(),
        upstream_status = upstream_status.as_u16(),
        "vertex request forwarded"
    );

    response
}

fn error_response(status: StatusCode, msg: &'static str) -> Response {
    Response::builder()
        .status(status)
        .body(Body::from(msg))
        .expect("static error response")
}

/// Bound on the in-flight queue between the byte-passthrough and the
/// SSE state-machine task. Mirrors the
/// `crate::proxy::SSE_PARSER_QUEUE_DEPTH` rationale (256 events ≈ 5
/// seconds of typical Anthropic streaming under the per-100ms event
/// rate; keeps memory bounded even if the parser stalls).
const VERTEX_SSE_QUEUE_DEPTH: usize = 256;

/// Drive the Anthropic SSE state machine over a stream of byte
/// chunks. Lives in its own spawned task; the byte path is fed via a
/// best-effort tee from [`forward_vertex_request`] and never blocks
/// on this loop.
async fn run_anthropic_sse_state_machine(
    mut rx: tokio::sync::mpsc::Receiver<bytes::Bytes>,
    request_id: String,
) {
    use crate::sse::framing::SseFramer;
    let mut framer = SseFramer::new();
    let mut state = crate::sse::anthropic::AnthropicStreamState::new();
    while let Some(chunk) = rx.recv().await {
        framer.push(&chunk);
        while let Some(ev_result) = framer.next_event() {
            match ev_result {
                Ok(ev) => {
                    if let Err(e) = state.apply(ev) {
                        tracing::warn!(
                            request_id = %request_id,
                            error = %e,
                            "vertex sse anthropic state-machine apply error"
                        );
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        request_id = %request_id,
                        error = %e,
                        "vertex sse framer error"
                    );
                }
            }
        }
    }
    tracing::info!(
        event = "vertex_sse_stream_closed",
        request_id = %request_id,
        provider = "vertex_anthropic",
        input_tokens = state.usage.input_tokens,
        output_tokens = state.usage.output_tokens,
        cache_creation_input_tokens = state.usage.cache_creation_input_tokens,
        cache_read_input_tokens = state.usage.cache_read_input_tokens,
        stop_reason = state.stop_reason.as_deref().unwrap_or(""),
        blocks = state.blocks.len(),
        "vertex sse stream closed"
    );
}
