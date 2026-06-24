//! POST `/v1/responses` handler — Phase C PR-C3 + PR-C4.
//!
//! # Why an explicit handler?
//!
//! The Python proxy currently flattens Responses-shape items into
//! Chat-Completions-shape via
//! `headroom/proxy/responses_converter.py` — a fragile shim that
//! silently breaks every time OpenAI lands a new item type. C3 ports
//! this path to Rust with first-class per-item-type handling.
//!
//! The handler buffers the request body (so the live-zone dispatcher
//! can inspect it) and re-injects it into [`crate::proxy::forward_http`].
//! `forward_http`'s compression gate dispatches on the path
//! classification (`CompressibleEndpoint::OpenAiResponses`) added by
//! C3.
//!
//! # Streaming (PR-C4)
//!
//! When the request carries `Accept: text/event-stream`, the response
//! tee in [`crate::proxy::forward_http`] flips on the
//! [`crate::sse::openai_responses::ResponseState`] state machine
//! (PR-C1) and frames bytes through [`crate::sse::framing::SseFramer`]
//! — never via naive `\n\n` splits. Decoded events update telemetry
//! in a spawned task that can never block the byte path.
//!
//! Per-item-type request-side compression (PR-C3) runs **regardless**
//! of `Accept`: a streaming `/v1/responses` request gets the same
//! request-body compression as a non-streaming one. C4 closes the
//! loop by confirming the full pipeline is active (no more
//! `responses_streaming_passthrough_until_c4` fallback). The
//! pipeline gate is `Config::enable_responses_streaming` (default
//! `true`) — toggle off only as an emergency rollback.
//!
//! Compression of streaming **response** events is NOT performed.
//! Output items are rendered live token-by-token; mid-stream
//! rewriting would corrupt the user-visible UX and is not part of
//! the live-zone-only contract (the live zone is **request**-side).
//!
//! # Per-item-type behaviour
//!
//! See [`crate::responses_items`] for the typed enum. Briefly:
//!
//! - `function_call_output` / `local_shell_call_output` /
//!   `apply_patch_call_output` — output strings are eligible for
//!   live-zone compression when the latest of each kind, above the
//!   2 KiB output-item floor.
//! - `message` (user role) — text content is eligible.
//! - `reasoning.encrypted_content`, `compaction.*`, MCP / computer /
//!   web-search / file-search / code-interpreter / image-generation /
//!   tool-search / custom-tool calls — passthrough byte-equal.
//! - `function_call.arguments` is a STRING the model emitted; never
//!   parsed by the proxy.
//! - `local_shell_call.action.command` is an argv array; never
//!   joined into a string.
//! - `apply_patch_call.operation.diff` is a V4A diff payload; never
//!   re-serialized.
//! - Unknown `type` values log
//!   `event = responses_unknown_item_type` at warn level and pass
//!   through verbatim.

use axum::body::Body;
use axum::extract::{ConnectInfo, State};
use axum::http::{HeaderMap, Method, Request, Uri};
use axum::response::Response;
use bytes::Bytes;
use std::net::SocketAddr;

use crate::observability;
use crate::proxy::{forward_http, AppState};

/// Axum POST handler for `/v1/responses`. Buffers the body, stitches
/// a fresh `Request<Body>` together, and forwards via
/// [`forward_http`]. Compression dispatch + SSE telemetry is handled
/// inside `forward_http`'s shared gate (PR-C1 + PR-C2 + PR-C3).
pub async fn handle_responses(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    // PR-C4: streaming pipeline confirmation. When the client asks
    // for SSE, log a structured breadcrumb so dashboards can confirm
    // the streaming pipeline is engaged (the SSE framer +
    // ResponseState machine in `forward_http`'s tee). The
    // `enable_responses_streaming` switch is honoured here — when
    // disabled, we still forward but emit a distinct event so the
    // operator sees the rollback take effect.
    //
    // Why log INFO (not WARN)? PR-C3 used WARN as a "this path is
    // half-built" signal. PR-C4 wires the streaming state machine
    // through, so the previous WARN is no longer accurate.
    if accepts_sse(&headers) {
        if state.config.enable_responses_streaming {
            tracing::info!(
                event = "responses_streaming_pipeline_active",
                method = %method,
                path = %uri.path(),
                framer = "byte_level_sse",
                state_machine = "openai_responses",
                "responses streaming pipeline engaged: SSE framer + ResponseState telemetry tee"
            );
        } else {
            tracing::warn!(
                event = "responses_streaming_pipeline_disabled",
                method = %method,
                path = %uri.path(),
                "responses streaming pipeline disabled by --enable-responses-streaming=false; \
                 SSE bytes will pass through opaquely (emergency rollback path)"
            );
        }
    }

    // Phase G PR-G3: extract the request-side `service_tier` so we
    // can count tier distribution on the inbound shape too. The
    // response-side tier (from `response.completed`) is captured by
    // the SSE state machine at stream-close; this counter increment
    // pairs them. Body is parsed best-effort; missing/non-JSON
    // bodies do NOT fabricate a tier — per realignment build-
    // constraint "no silent fallbacks", we just skip the emit and
    // log at debug.
    //
    // C1 fix: every raw value is validated against the bounded
    // `service_tier` vocabulary BEFORE being used as a label so a
    // malicious client cannot blow up label cardinality with
    // arbitrary strings.
    if let Some(tier) = extract_request_service_tier(&body) {
        let request_id_for_metric = headers
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("<no-request-id>");
        let bucketed = crate::observability::metric_names::service_tier::validate(&tier);
        observability::record_service_tier(bucketed, request_id_for_metric);
    } else {
        tracing::debug!(
            event = "service_tier_skipped",
            path = %uri.path(),
            reason = "absent_or_unparseable",
            "request body had no parseable service_tier; counter not emitted"
        );
    }

    // Reconstruct the Request<Body> shape forward_http expects.
    let mut builder = Request::builder().method(method).uri(uri);
    if let Some(hs) = builder.headers_mut() {
        *hs = headers;
    }
    let req = match builder.body(Body::from(body)) {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(
                event = "handler_error",
                handler = "responses",
                error = %e,
                "failed to reconstruct request from buffered body"
            );
            return Response::builder()
                .status(http::StatusCode::INTERNAL_SERVER_ERROR)
                .body(Body::from("internal handler error"))
                .expect("static response");
        }
    };

    forward_http(state, client_addr, req)
        .await
        .unwrap_or_else(|e| {
            use axum::response::IntoResponse;
            e.into_response()
        })
}

/// Phase G PR-G3: best-effort parse of `service_tier` from the
/// inbound request body. Returns `None` when the body is not valid
/// JSON, not an object, or lacks the field. The spec defines the
/// field as a string ∈ {auto, default, flex, on_demand, priority,
/// scale}; the returned raw string is normalised against the
/// bounded vocabulary at the call site via
/// [`crate::observability::metric_names::service_tier::validate`]
/// so an arbitrary inbound value cannot drive metric-label
/// cardinality unbounded (C1 fix).
fn extract_request_service_tier(body: &Bytes) -> Option<String> {
    let v: serde_json::Value = serde_json::from_slice(body).ok()?;
    v.get("service_tier")
        .and_then(|x| x.as_str())
        .map(|s| s.to_string())
}

/// Cheap check: is this request asking for an SSE response? Compares
/// `Accept` against `text/event-stream` (case-insensitive on the
/// media-type token, RFC 7231 §3.1.1.1). Multiple media types in
/// `Accept` are split on `,`; any match wins.
fn accepts_sse(headers: &HeaderMap) -> bool {
    let Some(v) = headers.get(http::header::ACCEPT) else {
        return false;
    };
    let Ok(s) = v.to_str() else {
        return false;
    };
    s.split(',').any(|piece| {
        let mt = piece.split(';').next().unwrap_or("").trim();
        mt.eq_ignore_ascii_case("text/event-stream")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use http::HeaderValue;

    #[test]
    fn accepts_sse_explicit() {
        let mut h = HeaderMap::new();
        h.insert(
            http::header::ACCEPT,
            HeaderValue::from_static("text/event-stream"),
        );
        assert!(accepts_sse(&h));
    }

    #[test]
    fn accepts_sse_case_insensitive() {
        let mut h = HeaderMap::new();
        h.insert(
            http::header::ACCEPT,
            HeaderValue::from_static("Text/Event-Stream"),
        );
        assert!(accepts_sse(&h));
    }

    #[test]
    fn accepts_sse_among_others() {
        let mut h = HeaderMap::new();
        h.insert(
            http::header::ACCEPT,
            HeaderValue::from_static("application/json, text/event-stream;q=0.9"),
        );
        assert!(accepts_sse(&h));
    }

    #[test]
    fn accepts_json_only_returns_false() {
        let mut h = HeaderMap::new();
        h.insert(
            http::header::ACCEPT,
            HeaderValue::from_static("application/json"),
        );
        assert!(!accepts_sse(&h));
    }

    #[test]
    fn no_accept_header_returns_false() {
        let h = HeaderMap::new();
        assert!(!accepts_sse(&h));
    }
}
