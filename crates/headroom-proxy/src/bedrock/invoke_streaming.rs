//! POST `/model/{model_id}/invoke-with-response-stream` handler —
//! Phase D PR-D2.
//!
//! # Pipeline
//!
//! 1. Parse the path, body, and `Accept` header.
//! 2. Run the live-zone Anthropic compressor over the (Anthropic-shape)
//!    body — same as PR-D1's non-streaming handler.
//! 3. Sign with SigV4 over the post-compression bytes.
//! 4. Forward to Bedrock via reqwest's streaming response API.
//! 5. Inspect the upstream response's `Content-Type`:
//!    - `application/vnd.amazon.eventstream` (the expected case) →
//!      drive the [`EventStreamParser`] over the byte stream.
//!    - Anything else → forward verbatim (Bedrock returned a JSON
//!      error body or a redirect; we surface it unchanged so the
//!      client sees the real status).
//! 6. Pick output mode by inspecting the inbound `Accept` header:
//!    - `Accept: application/vnd.amazon.eventstream` → passthrough
//!      the upstream bytes BYTE-EQUAL.
//!    - `Accept: text/event-stream` (default) → translate each
//!      `chunk` message's payload into an SSE frame; tee the SSE
//!      frames into [`AnthropicStreamState`] for telemetry.
//!
//! # Cache safety
//!
//! Same as PR-D1's `invoke.rs`: the SigV4 signature covers the bytes
//! we forward (post-compression), and the upstream's response bytes
//! are never modified IN PASSTHROUGH MODE. In SSE-translation mode the
//! response wire format changes (binary → text), but the JSON payload
//! inside each `chunk` is bytewise identical to the upstream — only
//! the framing differs.
//!
//! # Failure modes (all loud)
//!
//! - SigV4 missing creds → `5xx` + `event=bedrock_credentials_missing`.
//! - SigV4 sign failure → `5xx` + `event=bedrock_sigv4_failed`.
//! - EventStream parse failure mid-stream → close the response with
//!   a structured error frame; log
//!   `event=bedrock_eventstream_parse_failed` (or `_crc_mismatch`)
//!   at WARN.
//! - Translator `:message-type == exception` → propagate to client
//!   (the underlying upstream error is the customer's, not ours).

use std::convert::Infallible;
use std::net::SocketAddr;
use std::time::{Instant, SystemTime};

use axum::body::Body;
use axum::extract::{ConnectInfo, Extension, Path, State};
use axum::http::{HeaderMap, Method, StatusCode, Uri};
use axum::response::Response;
use bytes::Bytes;
use futures_util::stream::{self, Stream};
use futures_util::StreamExt as _;
use http::HeaderName;
use std::pin::Pin;
use url::Url;

use crate::bedrock::eventstream::{EventStreamParser, ParseError};
use crate::bedrock::eventstream_to_sse::{
    translate_message, OutputMode, TranslateError, TranslateOutcome,
};
use crate::bedrock::sigv4::{sign_request, SigningInputs};
use crate::compression::{
    compress_anthropic_request, Outcome as AnthropicOutcome, PassthroughReason,
};
use crate::headers::filter_response_headers;
use crate::observability::{
    observe_bedrock_invoke_latency, record_bedrock_eventstream_message, record_bedrock_invoke,
};
use crate::proxy::AppState;
// Phase F PR-F1 + PR-D3: pre-classified by `classify_and_attach_auth_mode`
// middleware on the bedrock router; we read it back via the
// `Extension<AuthMode>` extractor.
use headroom_core::auth_mode::AuthMode;

use crate::bedrock::vendor::is_anthropic_model_id;

/// AWS Bedrock Runtime DNS template.
const BEDROCK_RUNTIME_HOST_TEMPLATE: &str = "bedrock-runtime.{region}.amazonaws.com";

/// Path action for the streaming routes.
const STREAMING_ACTION: &str = "invoke-with-response-stream";
const CONVERSE_STREAM_ACTION: &str = "converse-stream";

/// RAII guard that observes the `bedrock_invoke_latency_seconds`
/// histogram on drop. Mirrors the [`crate::bedrock::invoke`] guard
/// — duplicated to avoid a cross-module type dependency for what
/// is fundamentally a 6-line struct.
struct LatencyGuard {
    model: String,
    region: String,
    start: Instant,
}

impl LatencyGuard {
    fn start(model: &str, region: &str) -> Self {
        Self {
            model: model.to_string(),
            region: region.to_string(),
            start: Instant::now(),
        }
    }
}

impl Drop for LatencyGuard {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed().as_secs_f64();
        observe_bedrock_invoke_latency(&self.model, &self.region, elapsed);
    }
}

/// Axum POST handler for `/model/{model_id}/invoke-with-response-stream`.
#[allow(clippy::too_many_arguments)] // axum extractors demand one argument per role
pub async fn handle_invoke_streaming(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    Extension(auth_mode): Extension<AuthMode>,
    Path(model_id): Path<String>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let _ = client_addr;
    let request_id = headers
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    // PR-D3: latency stopwatch + invoke counter at handler entry.
    // RAII guard observes the histogram regardless of which return
    // path the handler takes. Per-EventStream-message metrics are
    // recorded inside `translate_stream` once the upstream response
    // starts arriving.
    let region = state.config.bedrock_region.clone();
    let _latency_guard = LatencyGuard::start(&model_id, &region);
    // Wall-clock for the savings-store recent-request latency.
    let rec_start = Instant::now();
    record_bedrock_invoke(&model_id, &region, auth_mode);

    tracing::info!(
        event = "bedrock_invoke_streaming_received",
        request_id = %request_id,
        method = %method,
        model_id = %model_id,
        region = %region,
        auth_mode = auth_mode.as_str(),
        body_bytes = body.len(),
        "bedrock invoke-with-response-stream route received request"
    );

    // 1. Live-zone compression for Anthropic-shape bodies (same as D1).
    let is_anthropic = is_anthropic_model_id(&model_id);
    let (outbound_body, rec_tokens_before, rec_tokens_after): (Bytes, u64, u64) = if is_anthropic {
        run_anthropic_compression(&body, &state, auth_mode, &request_id)
    } else {
        tracing::info!(
            event = "bedrock_compression_skipped",
            request_id = %request_id,
            model_id = %model_id,
            reason = "non_anthropic_vendor",
            "bedrock invoke-streaming: skipping compression for non-anthropic vendor"
        );
        (body.clone(), 0, 0)
    };

    // Phase H: build this Bedrock streaming request's savings outcome now (token
    // counts are known) but record it only once the upstream result is known, so
    // `requests.failed` reflects connect errors / non-2xx upstreams. Recording is
    // gated on `should_record()` (compression on AND mode != off) — the shared
    // predicate; narrower than the plain-`compression` interception gate.
    let rec_outcome = state.config.should_record().then(|| {
        crate::observability::stats::RequestOutcome::priced(
            "bedrock",
            model_id.clone(),
            rec_tokens_before,
            rec_tokens_after,
            request_id.clone(),
            &state.price_book,
        )
    });

    // 2. Resolve the Bedrock streaming action from the inbound path and
    // build the upstream URL.
    let action = match extract_streaming_action(uri.path()) {
        Some(a) => a,
        None => {
            tracing::error!(
                event = "bedrock_streaming_action_invalid",
                request_id = %request_id,
                path = %uri.path(),
                "bedrock invoke-streaming: unrecognized streaming action path"
            );
            return super::error_response(
                StatusCode::BAD_REQUEST,
                "bedrock_streaming_action_invalid",
                "Unsupported Bedrock streaming action path",
            );
        }
    };

    let upstream_url = match build_bedrock_streaming_upstream(&state, &model_id, &uri, action) {
        Ok(u) => u,
        Err(msg) => {
            tracing::error!(
                event = "bedrock_endpoint_invalid",
                request_id = %request_id,
                error = %msg,
                "bedrock invoke-streaming: failed to construct upstream URL"
            );
            return super::error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "bedrock_endpoint_invalid",
                &msg,
            );
        }
    };

    // 3. Resolve credentials. No silent fallback.
    let creds = match state.bedrock_credentials.as_ref() {
        Some(c) => c.clone(),
        None => {
            tracing::warn!(
                event = "bedrock_credentials_missing",
                request_id = %request_id,
                model_id = %model_id,
                "bedrock invoke-streaming: refusing to forward without AWS credentials"
            );
            return super::error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "bedrock_credentials_missing",
                "AWS credentials not configured; refusing to forward unsigned",
            );
        }
    };

    // 4. Build the headers we sign + forward.
    let extra_signed: Vec<(String, String)> =
        super::collect_signed_headers(&headers, &upstream_url);
    let extra_signed_refs: Vec<(&str, &str)> = extra_signed
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();

    let sign_inputs = SigningInputs {
        method: method.as_str(),
        url: &upstream_url,
        region: &state.config.bedrock_region,
        credentials: creds.as_ref(),
        body: &outbound_body,
        extra_signed_headers: &extra_signed_refs,
        time: SystemTime::now(),
    };
    let signed = match sign_request(&sign_inputs) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!(
                event = "bedrock_sigv4_failed",
                request_id = %request_id,
                model_id = %model_id,
                error = %e,
                "bedrock invoke-streaming: SigV4 signing failed"
            );
            return super::error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "bedrock_sigv4_failed",
                &e.to_string(),
            );
        }
    };

    // Build outbound HeaderMap (same pattern as D1).
    let mut outbound_headers = HeaderMap::new();
    for (name, value) in extra_signed.iter() {
        if let (Ok(n), Ok(v)) = (
            HeaderName::from_bytes(name.as_bytes()),
            http::HeaderValue::from_str(value),
        ) {
            outbound_headers.insert(n, v);
        }
    }
    for (name, value) in signed.entries.iter() {
        if let (Ok(n), Ok(v)) = (
            HeaderName::from_bytes(name.as_bytes()),
            http::HeaderValue::from_str(value),
        ) {
            outbound_headers.insert(n, v);
        }
    }

    // 5. Forward.
    let reqwest_method = match reqwest::Method::from_bytes(method.as_str().as_bytes()) {
        Ok(m) => m,
        Err(e) => {
            tracing::error!(
                event = "bedrock_invalid_method",
                request_id = %request_id,
                error = %e,
                "bedrock invoke-streaming: invalid HTTP method"
            );
            return super::error_response(
                StatusCode::BAD_REQUEST,
                "bedrock_invalid_method",
                &e.to_string(),
            );
        }
    };

    let upstream_resp = state
        .client
        .request(reqwest_method, upstream_url.clone())
        .headers(outbound_headers)
        .body(outbound_body.clone())
        .send()
        .await;

    let upstream_resp = match upstream_resp {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(
                event = "bedrock_upstream_error",
                request_id = %request_id,
                error = %e,
                "bedrock invoke-streaming: upstream request failed"
            );
            // Connect/timeout failure — record as a failed request.
            if let Some(o) = rec_outcome {
                state.savings.record_finalized(o, true, rec_start);
            }
            let status = if e.is_timeout() {
                StatusCode::GATEWAY_TIMEOUT
            } else {
                StatusCode::BAD_GATEWAY
            };
            return super::error_response(status, "bedrock_upstream_error", &e.to_string());
        }
    };

    let status =
        StatusCode::from_u16(upstream_resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    // For the SSE translation path, defer record_finalized into the stream
    // task so it picks up usage tokens extracted from the final EventStream
    // chunk. The early-record path below is only for non-EventStream bodies
    // (errors, passthrough) where no stream task runs.
    let deferred_record =
        rec_outcome.map(|o| (o, state.savings.clone(), rec_start, !status.is_success()));
    let upstream_content_type = upstream_resp
        .headers()
        .get(http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    // 6. Decide output mode based on the client's `Accept` header.
    let accept_values = OutputMode::default_eventstream_accept_values();
    let output_mode = OutputMode::from_accept(&headers, &accept_values);

    // If upstream is NOT vnd.amazon.eventstream (e.g. it returned an
    // application/json error), forward verbatim regardless of Accept.
    let upstream_is_eventstream = upstream_content_type
        .as_deref()
        .map(is_eventstream_content_type)
        .unwrap_or(false);

    let mut resp_headers = filter_response_headers(upstream_resp.headers());

    tracing::info!(
        event = "bedrock_invoke_streaming_forwarded",
        request_id = %request_id,
        model_id = %model_id,
        upstream_status = status.as_u16(),
        upstream_url = %upstream_url,
        upstream_content_type = upstream_content_type.as_deref().unwrap_or(""),
        upstream_is_eventstream = upstream_is_eventstream,
        client_output_mode = match output_mode {
            OutputMode::EventStream => "eventstream",
            OutputMode::Sse => "sse",
        },
        "bedrock invoke-streaming: response received; selecting output mode"
    );

    if !upstream_is_eventstream {
        // Upstream isn't binary EventStream — pass through verbatim.
        // No stream task will run, so record now.
        if let Some((o, savings, started, failed)) = deferred_record {
            savings.record_finalized(o, failed, started);
        }
        resp_headers.remove(http::header::CONTENT_LENGTH);
        let stream = upstream_resp
            .bytes_stream()
            .map(|r| r.map_err(std::io::Error::other));
        let body_out = Body::from_stream(stream);
        return finish(status, resp_headers, body_out, &request_id);
    }
    // Always drop the upstream content-length: in passthrough mode
    // we may still re-frame; in SSE mode the byte-length changes.
    // hyper assigns transfer-encoding: chunked when content-length
    // is absent.
    resp_headers.remove(http::header::CONTENT_LENGTH);

    // Decide what to emit. In passthrough mode, copy upstream bytes
    // verbatim. In SSE mode, run a parser, translate chunks, and
    // emit SSE frames. Both modes start from the same byte stream.
    let upstream_stream = upstream_resp
        .bytes_stream()
        .map(|r| r.map_err(std::io::Error::other));

    match output_mode {
        OutputMode::EventStream => {
            // Passthrough — bytes flow byte-equal to the client. No SSE
            // translation runs, so record now (no usage extraction possible
            // from raw EventStream without parsing).
            if let Some((o, savings, started, failed)) = deferred_record {
                savings.record_finalized(o, failed, started);
            }
            tracing::info!(
                event = "bedrock_eventstream_passthrough",
                request_id = %request_id,
                "passing upstream eventstream bytes through verbatim"
            );
            if !resp_headers.contains_key(http::header::CONTENT_TYPE) {
                if let Ok(v) = http::HeaderValue::from_str("application/vnd.amazon.eventstream") {
                    resp_headers.insert(http::header::CONTENT_TYPE, v);
                }
            }
            let body_out = Body::from_stream(upstream_stream);
            finish(status, resp_headers, body_out, &request_id)
        }
        OutputMode::Sse => {
            // Translation mode. Defer record_finalized into the stream task
            // so usage tokens from the final EventStream chunk are captured.
            resp_headers.remove(http::header::CONTENT_TYPE);
            if let Ok(v) = http::HeaderValue::from_str("text/event-stream") {
                resp_headers.insert(http::header::CONTENT_TYPE, v);
            }

            let translated = translate_stream(
                upstream_stream,
                state.config.bedrock_validate_eventstream_crc,
                request_id.clone(),
                model_id.clone(),
                region.clone(),
            );
            let translated =
                tee_to_anthropic_state(translated, request_id.clone(), deferred_record);
            let body_out = Body::from_stream(translated);
            finish(status, resp_headers, body_out, &request_id)
        }
    }
}

/// Boxed byte-stream alias used internally so we can keep the
/// `Stream` returned by reqwest pinned and `Unpin`-friendly when it
/// crosses through the `unfold` machinery below.
type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send + 'static>>;

/// Stream adapter: drive an [`EventStreamParser`] over the upstream
/// byte stream, translate each complete message into an SSE frame,
/// and yield the resulting `Bytes`. Errors close the stream with a
/// structured frame so the client sees the failure rather than a
/// truncated response.
fn translate_stream<S>(
    upstream: S,
    validate_crc: bool,
    request_id: String,
    model_id: String,
    region: String,
) -> impl Stream<Item = Result<Bytes, std::io::Error>>
where
    S: Stream<Item = Result<Bytes, std::io::Error>> + Send + 'static,
{
    use crate::bedrock::eventstream::CrcValidation;

    let mut parser = EventStreamParser::new();
    if !validate_crc {
        parser = parser.with_crc_validation(CrcValidation::No);
    }
    let upstream: ByteStream = Box::pin(upstream);
    // Bundle the per-stream identifiers we thread through every
    // unfold step. `unfold` only allows a single state value, so
    // grouping these into a tuple of owned Strings keeps the
    // closure readable.
    let init = (parser, upstream, false, request_id, model_id, region);
    stream::unfold(
        init,
        |(mut parser, mut upstream, mut done, request_id, model_id, region)| {
            Box::pin(async move {
                if done {
                    return None;
                }
                // First, drain any complete messages already in the
                // parser's buffer (bytes from the previous chunk).
                loop {
                    match parser.next_message() {
                        Ok(Some(msg)) => match translate_message(&msg, OutputMode::Sse) {
                            Ok(TranslateOutcome::Emit(frame)) => {
                                // PR-D3: per-message Prometheus metric.
                                // The label `event_type` is bounded by
                                // AWS's documented vocabulary (chunk,
                                // metadata, exception variants); not
                                // customer-controlled.
                                let event_type = msg.event_type().unwrap_or("unknown").to_string();
                                record_bedrock_eventstream_message(&model_id, &region, &event_type);
                                tracing::debug!(
                                    event = "bedrock_eventstream_message",
                                    request_id = %request_id,
                                    event_type = %event_type,
                                    payload_bytes = msg.payload.len(),
                                    "translated bedrock eventstream message"
                                );
                                return Some((
                                    Ok(frame),
                                    (parser, upstream, false, request_id, model_id, region),
                                ));
                            }
                            Ok(TranslateOutcome::Skip { event_type }) => {
                                tracing::warn!(
                                    event = "bedrock_eventstream_unknown_event_type",
                                    request_id = %request_id,
                                    event_type = %event_type,
                                    "skipping unknown bedrock eventstream message"
                                );
                                // Loop and try the next message in the buffer.
                                continue;
                            }
                            Err(TranslateError::UpstreamException { payload_preview }) => {
                                tracing::warn!(
                                    event = "bedrock_eventstream_upstream_exception",
                                    request_id = %request_id,
                                    payload_preview = %payload_preview,
                                    "bedrock eventstream upstream exception"
                                );
                                // Emit the exception as an Anthropic-shape
                                // SSE error frame so the client sees it.
                                let json = serde_json::json!({
                                    "type": "error",
                                    "error": {
                                        "type": "bedrock_upstream_exception",
                                        "message": payload_preview,
                                    }
                                })
                                .to_string();
                                let mut frame = Vec::with_capacity(json.len() + 32);
                                frame.extend_from_slice(b"event: error\ndata: ");
                                frame.extend_from_slice(json.as_bytes());
                                frame.extend_from_slice(b"\n\n");
                                return Some((
                                    Ok(Bytes::from(frame)),
                                    (parser, upstream, true, request_id, model_id, region),
                                ));
                            }
                            Err(TranslateError::MissingEventType) => {
                                tracing::warn!(
                                    event = "bedrock_eventstream_missing_event_type",
                                    request_id = %request_id,
                                    "bedrock eventstream message missing :event-type; emitting error frame"
                                );
                                let frame = error_sse_frame(
                                    "bedrock_eventstream_missing_event_type",
                                    "Bedrock message missing :event-type header",
                                );
                                return Some((
                                    Ok(frame),
                                    (parser, upstream, true, request_id, model_id, region),
                                ));
                            }
                        },
                        Ok(None) => break,
                        Err(parse_err) => {
                            let event_name = match &parse_err {
                                ParseError::PreludeCrcMismatch { .. }
                                | ParseError::MessageCrcMismatch { .. } => {
                                    "bedrock_eventstream_crc_mismatch"
                                }
                                _ => "bedrock_eventstream_parse_failed",
                            };
                            tracing::warn!(
                                event = event_name,
                                request_id = %request_id,
                                error = %parse_err,
                                "bedrock eventstream parse failure; closing translated stream"
                            );
                            let frame = error_sse_frame(event_name, &parse_err.to_string());
                            return Some((
                                Ok(frame),
                                (parser, upstream, true, request_id, model_id, region),
                            ));
                        }
                    }
                }
                // Buffer drained; pull the next chunk from upstream.
                loop {
                    match upstream.next().await {
                        Some(Ok(chunk)) => {
                            parser.push(&chunk);
                            // Loop back through the parser to emit any
                            // newly-complete messages.
                            match parser.next_message() {
                                Ok(Some(msg)) => match translate_message(&msg, OutputMode::Sse) {
                                    Ok(TranslateOutcome::Emit(frame)) => {
                                        // PR-D3: per-message Prometheus
                                        // metric (mirror of the parser-buffer
                                        // drain branch above).
                                        let event_type =
                                            msg.event_type().unwrap_or("unknown").to_string();
                                        record_bedrock_eventstream_message(
                                            &model_id,
                                            &region,
                                            &event_type,
                                        );
                                        tracing::debug!(
                                            event = "bedrock_eventstream_message",
                                            request_id = %request_id,
                                            event_type = %event_type,
                                            payload_bytes = msg.payload.len(),
                                            "translated bedrock eventstream message"
                                        );
                                        return Some((
                                            Ok(frame),
                                            (parser, upstream, false, request_id, model_id, region),
                                        ));
                                    }
                                    Ok(TranslateOutcome::Skip { event_type }) => {
                                        tracing::warn!(
                                            event = "bedrock_eventstream_unknown_event_type",
                                            request_id = %request_id,
                                            event_type = %event_type,
                                            "skipping unknown bedrock eventstream message"
                                        );
                                        // Continue draining the parser /
                                        // pulling more chunks.
                                        continue;
                                    }
                                    Err(TranslateError::UpstreamException { payload_preview }) => {
                                        let json = serde_json::json!({
                                            "type": "error",
                                            "error": {
                                                "type": "bedrock_upstream_exception",
                                                "message": payload_preview,
                                            }
                                        })
                                        .to_string();
                                        let mut frame = Vec::with_capacity(json.len() + 32);
                                        frame.extend_from_slice(b"event: error\ndata: ");
                                        frame.extend_from_slice(json.as_bytes());
                                        frame.extend_from_slice(b"\n\n");
                                        return Some((
                                            Ok(Bytes::from(frame)),
                                            (parser, upstream, true, request_id, model_id, region),
                                        ));
                                    }
                                    Err(TranslateError::MissingEventType) => {
                                        let frame = error_sse_frame(
                                            "bedrock_eventstream_missing_event_type",
                                            "Bedrock message missing :event-type header",
                                        );
                                        return Some((
                                            Ok(frame),
                                            (parser, upstream, true, request_id, model_id, region),
                                        ));
                                    }
                                },
                                Ok(None) => continue,
                                Err(parse_err) => {
                                    let event_name = match &parse_err {
                                        ParseError::PreludeCrcMismatch { .. }
                                        | ParseError::MessageCrcMismatch { .. } => {
                                            "bedrock_eventstream_crc_mismatch"
                                        }
                                        _ => "bedrock_eventstream_parse_failed",
                                    };
                                    tracing::warn!(
                                        event = event_name,
                                        request_id = %request_id,
                                        error = %parse_err,
                                        "bedrock eventstream parse failure"
                                    );
                                    let frame = error_sse_frame(event_name, &parse_err.to_string());
                                    return Some((
                                        Ok(frame),
                                        (parser, upstream, true, request_id, model_id, region),
                                    ));
                                }
                            }
                        }
                        Some(Err(e)) => {
                            tracing::warn!(
                                event = "bedrock_eventstream_upstream_io_error",
                                request_id = %request_id,
                                error = %e,
                                "upstream io error mid-stream"
                            );
                            return Some((
                                Err(e),
                                (parser, upstream, true, request_id, model_id, region),
                            ));
                        }
                        None => {
                            // End of upstream stream. If buffered bytes
                            // remain that did not parse into a message,
                            // log loudly — we are NOT silently dropping
                            // them.
                            if parser.buffered_len() > 0 {
                                tracing::warn!(
                                    event = "bedrock_eventstream_truncated",
                                    request_id = %request_id,
                                    buffered_bytes = parser.buffered_len(),
                                    "upstream stream ended with un-parseable trailing bytes"
                                );
                            }
                            done = true;
                            let _ = done;
                            return None;
                        }
                    }
                }
            })
        },
    )
}

type DeferredRecord = Option<(
    crate::observability::stats::RequestOutcome,
    std::sync::Arc<crate::observability::stats::SavingsStore>,
    std::time::Instant,
    bool,
)>;

/// Tee the translated SSE stream into an `AnthropicStreamState` task
/// so usage telemetry is captured. The byte path is independent of
/// the parser — if the parser falls behind, the channel `try_send`
/// drops chunks rather than blocking.
///
/// `deferred_record` carries the `RequestOutcome` + savings store so
/// the state machine can fill response usage and call `record_finalized`
/// after the stream ends, giving accurate token counts.
fn tee_to_anthropic_state<S>(
    upstream: S,
    request_id: String,
    deferred_record: DeferredRecord,
) -> impl Stream<Item = Result<Bytes, std::io::Error>>
where
    S: Stream<Item = Result<Bytes, std::io::Error>> + Send + 'static,
{
    let (tx, rx) = tokio::sync::mpsc::channel::<Bytes>(SSE_PARSER_QUEUE_DEPTH);
    let parser_request_id = request_id.clone();
    tokio::spawn(async move {
        run_anthropic_state_machine(rx, parser_request_id, deferred_record).await;
    });

    upstream.map(move |r| {
        if let Ok(bytes) = &r {
            // Best-effort tee. Bounded channel; never block byte path.
            if let Err(e) = tx.try_send(bytes.clone()) {
                tracing::debug!(
                    request_id = %request_id,
                    error = %e,
                    "bedrock translated stream parser queue full or closed; dropping telemetry chunk"
                );
            }
        }
        r
    })
}

const SSE_PARSER_QUEUE_DEPTH: usize = 256;

/// Scan one SSE `data` field (raw bytes) for token usage, accumulating
/// into `input`, `output`, `cache_read`.  Handles three wire formats:
///
/// 1. **InvokeModel streaming** — each SSE event wraps the Anthropic JSON
///    in base64: `{"bytes":"BASE64(anthropic_json)","p":"..."}`.
///    Anthropic usage lives at `.message.usage` (message_start) or
///    `.usage` (message_delta) inside the decoded bytes.
///
/// 2. **Converse streaming** — the final `metadata` EventStream event
///    carries `{"usage":{"inputTokens":N,"outputTokens":N,...}}` directly
///    in the SSE data field.
///
/// 3. **Direct Anthropic SSE** (future / test path) — `{"type":
///    "message_delta","usage":{"input_tokens":N,...}}` at the top level.
fn accumulate_stream_usage(
    data: &[u8],
    input: &mut u64,
    output: &mut u64,
    cache_read: &mut u64,
    cache_write: &mut u64,
) {
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(data) else {
        return;
    };

    // --- Format 2: Converse metadata: {"usage":{"inputTokens":N,...}} ---
    if let Some(usage) = v.get("usage") {
        let get = |k: &str| usage.get(k).and_then(|t| t.as_u64()).unwrap_or(0);
        let it = get("inputTokens");
        let ot = get("outputTokens");
        let cr = get("cacheReadInputTokens");
        let cw = get("cacheWriteInputTokens");
        if it > 0 {
            *input = it;
        }
        if ot > 0 {
            *output = ot;
        }
        if cr > 0 {
            *cache_read = cr;
        }
        if cw > 0 {
            *cache_write = cw;
        }
        // Also Anthropic snake_case at top level (format 3 / message_delta)
        let it2 = get("input_tokens");
        let ot2 = get("output_tokens");
        let cr2 = get("cache_read_input_tokens");
        let cw2 = get("cache_creation_input_tokens");
        if it2 > 0 {
            *input = it2;
        }
        if ot2 > 0 {
            *output = ot2;
        }
        if cr2 > 0 {
            *cache_read = cr2;
        }
        if cw2 > 0 {
            *cache_write = cw2;
        }
    }
    // message_start has usage nested under .message.usage (Anthropic format)
    if let Some(usage) = v.get("message").and_then(|m| m.get("usage")) {
        let get = |k: &str| usage.get(k).and_then(|t| t.as_u64()).unwrap_or(0);
        let it = get("input_tokens");
        let cr = get("cache_read_input_tokens");
        let cw = get("cache_creation_input_tokens");
        if it > 0 && *input == 0 {
            *input = it;
        }
        if cr > 0 {
            *cache_read = cr;
        }
        if cw > 0 {
            *cache_write = cw;
        }
    }

    // --- Format 1: InvokeModel wrapped: {"bytes":"BASE64","p":"..."} ---
    if let Some(b64) = v.get("bytes").and_then(|b| b.as_str()) {
        use base64::engine::general_purpose::STANDARD;
        use base64::Engine as _;
        if let Ok(decoded) = STANDARD.decode(b64) {
            // Recurse once — the inner payload is either message_start or
            // message_delta (Anthropic format), handled by format 3 above.
            accumulate_stream_usage(&decoded, input, output, cache_read, cache_write);
        }
    }
}

/// Dedicated task: consume the tee'd translated-SSE byte stream, scan each
/// SSE data field for token usage, then call `record_finalized` once the
/// stream closes.  Replaces the previous `AnthropicStreamState` approach
/// which only handled direct Anthropic SSE, not InvokeModel-wrapped or
/// Converse formats.
async fn run_anthropic_state_machine(
    mut rx: tokio::sync::mpsc::Receiver<Bytes>,
    request_id: String,
    deferred_record: DeferredRecord,
) {
    use crate::sse::framing::SseFramer;

    let mut framer = SseFramer::new();
    let mut input_tokens: u64 = 0;
    let mut output_tokens: u64 = 0;
    let mut cache_read_tokens: u64 = 0;
    let mut cache_write_tokens: u64 = 0;

    while let Some(chunk) = rx.recv().await {
        framer.push(&chunk);
        while let Some(ev_result) = framer.next_event() {
            match ev_result {
                Ok(ev) => {
                    accumulate_stream_usage(
                        &ev.data,
                        &mut input_tokens,
                        &mut output_tokens,
                        &mut cache_read_tokens,
                        &mut cache_write_tokens,
                    );
                }
                Err(e) => {
                    tracing::debug!(
                        request_id = %request_id,
                        error = %e,
                        "bedrock translated stream: sse framer error"
                    );
                }
            }
        }
    }

    tracing::info!(
        request_id = %request_id,
        provider = "bedrock_streaming",
        input_tokens,
        output_tokens,
        cache_read_tokens,
        cache_write_tokens,
        "bedrock translated stream: closed"
    );

    if let Some((mut outcome, savings, started, failed)) = deferred_record {
        if outcome.tokens_before == 0 {
            outcome.tokens_before = input_tokens;
            outcome.tokens_after = input_tokens;
        }
        outcome.output_tokens = output_tokens;
        outcome.cache_read_tokens = cache_read_tokens;
        outcome.cache_write_tokens = cache_write_tokens;
        savings.record_finalized(outcome, failed, started);
    }
}

/// True when the content-type is `application/vnd.amazon.eventstream`
/// (with optional parameters). RFC 7231 §3.1.1.1.
fn is_eventstream_content_type(content_type: &str) -> bool {
    let media_type = content_type.split(';').next().unwrap_or("").trim();
    media_type.eq_ignore_ascii_case("application/vnd.amazon.eventstream")
}

/// Build a structured SSE error frame with `event: error`. The shape
/// matches Anthropic's documented error event so existing clients
/// already know how to surface it.
fn error_sse_frame(event_kind: &str, message: &str) -> Bytes {
    let json = serde_json::json!({
        "type": "error",
        "error": {
            "type": event_kind,
            "message": message,
        }
    })
    .to_string();
    let mut out = Vec::with_capacity(json.len() + 24);
    out.extend_from_slice(b"event: error\ndata: ");
    out.extend_from_slice(json.as_bytes());
    out.extend_from_slice(b"\n\n");
    Bytes::from(out)
}

fn finish(status: StatusCode, headers: HeaderMap, body: Body, request_id: &str) -> Response {
    let mut builder = Response::builder().status(status);
    if let Some(h) = builder.headers_mut() {
        h.extend(headers);
        if let Ok(v) = http::HeaderValue::from_str(request_id) {
            h.insert(HeaderName::from_static("x-request-id"), v);
        }
    }
    builder.body(body).unwrap_or_else(|e| {
        tracing::error!(
            event = "bedrock_response_build_failed",
            request_id = %request_id,
            error = %e,
            "bedrock invoke-streaming: failed to build response"
        );
        Response::builder()
            .status(StatusCode::INTERNAL_SERVER_ERROR)
            .body(Body::from("internal handler error"))
            .expect("static response")
    })
}

/// Same compression-dispatch logic as PR-D1 — duplicated rather than
/// shared because the streaming handler runs in a slightly different
/// flow (no body buffering required at the caller, the handler always
/// owns the bytes).
/// Run the live-zone compressor and return the outbound body plus the
/// pre/post-compression input token counts (`(0, 0)` when nothing compressed),
/// so the caller can attribute Bedrock streaming savings into the stats store.
fn run_anthropic_compression(
    body: &Bytes,
    state: &AppState,
    _auth_mode: AuthMode,
    request_id: &str,
) -> (Bytes, u64, u64) {
    use crate::bedrock::envelope::BedrockEnvelope;

    let parsed_envelope = BedrockEnvelope::parse(body).is_ok();
    if !parsed_envelope {
        tracing::info!(
            event = "bedrock_envelope_parse_skipped",
            request_id = %request_id,
            "bedrock invoke-streaming: envelope parse skipped; attempting generic anthropic compression"
        );
    }

    // PR-E3: Bedrock channel hard-codes OAuth so cache_control
    // auto-placement is skipped (see invoke.rs for rationale).
    let outcome = compress_anthropic_request(
        body,
        state.config.compression_mode,
        state.config.cache_control_auto_frozen,
        headroom_core::auth_mode::AuthMode::OAuth,
        request_id,
    );
    let (tokens_before, tokens_after) = match &outcome {
        AnthropicOutcome::Compressed {
            tokens_before,
            tokens_after,
            ..
        } => (*tokens_before as u64, *tokens_after as u64),
        _ => (0, 0),
    };
    let out_body = match outcome {
        AnthropicOutcome::NoCompression => body.clone(),
        AnthropicOutcome::Passthrough { reason } => {
            tracing::info!(
                event = "bedrock_compression_passthrough",
                request_id = %request_id,
                reason = ?reason,
                "bedrock invoke-streaming: passthrough"
            );
            let _ = (PassthroughReason::ModeOff, PassthroughReason::NoMessages);
            body.clone()
        }
        AnthropicOutcome::Compressed { body: new_body, .. } => {
            if parsed_envelope {
                match BedrockEnvelope::ensure_anthropic_version_first(&new_body) {
                    Ok(b) => b,
                    Err(e) => {
                        tracing::error!(
                            event = "bedrock_envelope_reemit_failed",
                            request_id = %request_id,
                            error = %e,
                            "bedrock invoke-streaming: failed to re-emit envelope"
                        );
                        body.clone()
                    }
                }
            } else {
                new_body
            }
        }
    };
    (out_body, tokens_before, tokens_after)
}

fn build_bedrock_streaming_upstream(
    state: &AppState,
    model_id: &str,
    uri: &Uri,
    action: &str,
) -> Result<Url, String> {
    let base = match state.config.bedrock_endpoint.as_ref() {
        Some(u) => u.clone(),
        None => {
            let host =
                BEDROCK_RUNTIME_HOST_TEMPLATE.replace("{region}", &state.config.bedrock_region);
            Url::parse(&format!("https://{host}/"))
                .map_err(|e| format!("bedrock derived base URL parse error: {e}"))?
        }
    };
    let path = format!(
        "/model/{model_id}/{action}",
        model_id = model_id,
        action = action,
    );
    let mut joined = base;
    joined.set_path(&path);
    if let Some(q) = uri.query() {
        joined.set_query(Some(q));
    }
    Ok(joined)
}

fn extract_streaming_action(path: &str) -> Option<&'static str> {
    if path.ends_with(&format!("/{STREAMING_ACTION}")) {
        Some(STREAMING_ACTION)
    } else if path.ends_with(&format!("/{CONVERSE_STREAM_ACTION}")) {
        Some(CONVERSE_STREAM_ACTION)
    } else {
        None
    }
}

// Keep the unused-import lints happy on rare failure-only branches.
#[allow(dead_code)]
fn _pin_infallible(_: Infallible) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eventstream_content_type_match() {
        assert!(is_eventstream_content_type(
            "application/vnd.amazon.eventstream"
        ));
        assert!(is_eventstream_content_type(
            "application/vnd.amazon.eventstream; charset=utf-8"
        ));
        assert!(!is_eventstream_content_type("application/json"));
        assert!(!is_eventstream_content_type("text/event-stream"));
    }

    #[test]
    fn build_streaming_upstream_uses_region_default() {
        use crate::config::Config;
        let mut config = Config::for_test(Url::parse("http://up:8080").unwrap());
        config.bedrock_region = "eu-west-1".to_string();
        let state = AppState {
            config: std::sync::Arc::new(config),
            client: reqwest::Client::new(),
            bedrock_credentials: None,
            // PR-E6: drift detector is unused by this URL-builder
            // unit test; small capacity to satisfy the struct shape.
            drift_state: crate::cache_stabilization::drift_detector::DriftState::new(8),
            // PR-D4: unit tests for the Bedrock URL builder don't
            // touch the Vertex route, but `AppState` is one struct
            // — supply a dummy token source so the test compiles.
            vertex_token_source: std::sync::Arc::new(crate::vertex::StaticTokenSource::new(
                "test".to_string(),
            )),
            savings: std::sync::Arc::new(crate::observability::stats::SavingsStore::in_memory()),
            price_book: std::sync::Arc::new(crate::observability::pricing::PriceBook::empty()),
        };
        let uri: Uri = "/model/anthropic.claude-3-haiku-20240307-v1:0/invoke-with-response-stream"
            .parse()
            .unwrap();
        let url = build_bedrock_streaming_upstream(
            &state,
            "anthropic.claude-3-haiku-20240307-v1:0",
            &uri,
            STREAMING_ACTION,
        )
        .unwrap();
        assert_eq!(
            url.as_str(),
            "https://bedrock-runtime.eu-west-1.amazonaws.com/model/anthropic.claude-3-haiku-20240307-v1:0/invoke-with-response-stream"
        );
    }

    #[test]
    fn error_sse_frame_shape() {
        let f = error_sse_frame("bedrock_eventstream_crc_mismatch", "boom");
        let s = std::str::from_utf8(&f).unwrap();
        assert!(s.starts_with("event: error\ndata: "));
        assert!(s.ends_with("\n\n"));
        assert!(s.contains("bedrock_eventstream_crc_mismatch"));
    }

    #[test]
    fn build_streaming_upstream_supports_converse_stream_action() {
        use crate::config::Config;
        let mut config = Config::for_test(Url::parse("http://up:8080").unwrap());
        config.bedrock_region = "eu-west-1".to_string();
        let state = AppState {
            config: std::sync::Arc::new(config),
            client: reqwest::Client::new(),
            bedrock_credentials: None,
            drift_state: crate::cache_stabilization::drift_detector::DriftState::new(8),
            vertex_token_source: std::sync::Arc::new(crate::vertex::StaticTokenSource::new(
                "test".to_string(),
            )),
            savings: std::sync::Arc::new(crate::observability::stats::SavingsStore::in_memory()),
            price_book: std::sync::Arc::new(crate::observability::pricing::PriceBook::empty()),
        };
        let uri: Uri = "/model/anthropic.claude-3-haiku-20240307-v1:0/converse-stream"
            .parse()
            .unwrap();
        let url = build_bedrock_streaming_upstream(
            &state,
            "anthropic.claude-3-haiku-20240307-v1:0",
            &uri,
            CONVERSE_STREAM_ACTION,
        )
        .unwrap();
        assert_eq!(
            url.as_str(),
            "https://bedrock-runtime.eu-west-1.amazonaws.com/model/anthropic.claude-3-haiku-20240307-v1:0/converse-stream"
        );
    }

    #[test]
    fn accumulate_stream_usage_converse_camelcase() {
        let (mut i, mut o, mut cr, mut cw) = (0u64, 0u64, 0u64, 0u64);
        // Converse-stream metadata event
        let data = br#"{"usage":{"inputTokens":12,"outputTokens":7,"cacheReadInputTokens":3,"cacheWriteInputTokens":2}}"#;
        accumulate_stream_usage(data, &mut i, &mut o, &mut cr, &mut cw);
        assert_eq!(i, 12);
        assert_eq!(o, 7);
        assert_eq!(cr, 3);
        assert_eq!(cw, 2);
    }

    #[test]
    fn accumulate_stream_usage_anthropic_snake_case() {
        let (mut i, mut o, mut cr, mut cw) = (0u64, 0u64, 0u64, 0u64);
        // Direct Anthropic SSE: message_delta usage
        let data = br#"{"type":"message_delta","usage":{"input_tokens":9,"output_tokens":15,"cache_read_input_tokens":0,"cache_creation_input_tokens":4}}"#;
        accumulate_stream_usage(data, &mut i, &mut o, &mut cr, &mut cw);
        assert_eq!(i, 9);
        assert_eq!(o, 15);
        assert_eq!(cr, 0);
        assert_eq!(cw, 4);
    }

    #[test]
    fn accumulate_stream_usage_anthropic_message_start() {
        let (mut i, mut o, mut cr, mut cw) = (0u64, 0u64, 0u64, 0u64);
        // Anthropic SSE message_start carries input_tokens in .message.usage
        let data = br#"{"type":"message_start","message":{"usage":{"input_tokens":20,"cache_read_input_tokens":5,"cache_creation_input_tokens":3}}}"#;
        accumulate_stream_usage(data, &mut i, &mut o, &mut cr, &mut cw);
        assert_eq!(i, 20);
        assert_eq!(cr, 5);
        assert_eq!(cw, 3);
    }

    #[test]
    fn accumulate_stream_usage_invoke_streaming_base64_wrapped() {
        use base64::Engine as _;
        let (mut i, mut o, mut cr, mut cw) = (0u64, 0u64, 0u64, 0u64);
        // InvokeModel streaming wraps Anthropic-native SSE events in {"bytes":"<base64>"}
        // Simulate message_start (input tokens) then message_delta (output tokens)
        let start = br#"{"type":"message_start","message":{"usage":{"input_tokens":9,"cache_read_input_tokens":0}}}"#;
        let b64 = base64::engine::general_purpose::STANDARD.encode(start);
        let envelope = format!(r#"{{"bytes":"{b64}","p":"abcd"}}"#);
        accumulate_stream_usage(envelope.as_bytes(), &mut i, &mut o, &mut cr, &mut cw);
        assert_eq!(i, 9);

        let delta = br#"{"type":"message_delta","usage":{"output_tokens":15}}"#;
        let b64 = base64::engine::general_purpose::STANDARD.encode(delta);
        let envelope = format!(r#"{{"bytes":"{b64}","p":"abcd"}}"#);
        accumulate_stream_usage(envelope.as_bytes(), &mut i, &mut o, &mut cr, &mut cw);
        assert_eq!(o, 15);
    }

    #[test]
    fn accumulate_stream_usage_noop_on_non_json() {
        let (mut i, mut o, mut cr, mut cw) = (1u64, 2u64, 3u64, 4u64);
        accumulate_stream_usage(b"not json", &mut i, &mut o, &mut cr, &mut cw);
        assert_eq!((i, o, cr, cw), (1, 2, 3, 4));
    }

    #[test]
    fn accumulate_stream_usage_does_not_reset_with_zero_update() {
        let (mut i, mut o, mut cr, mut cw) = (9u64, 0u64, 0u64, 0u64);
        // A delta event with no usage should leave existing counts alone
        let data = br#"{"type":"content_block_delta","delta":{"type":"text_delta","text":"OK"}}"#;
        accumulate_stream_usage(data, &mut i, &mut o, &mut cr, &mut cw);
        assert_eq!(i, 9); // unchanged
    }

    #[test]
    fn extract_streaming_action_supports_both_bedrock_paths() {
        assert_eq!(
            extract_streaming_action(
                "/model/anthropic.claude-3-haiku-20240307-v1:0/invoke-with-response-stream"
            ),
            Some(STREAMING_ACTION)
        );
        assert_eq!(
            extract_streaming_action(
                "/model/anthropic.claude-3-haiku-20240307-v1:0/converse-stream"
            ),
            Some(CONVERSE_STREAM_ACTION)
        );
        assert_eq!(
            extract_streaming_action("/model/anthropic.claude-3-haiku-20240307-v1:0/invoke"),
            None
        );
    }
}
