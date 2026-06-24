//! Native AWS Bedrock InvokeModel route — Phase D PR-D1.
//!
//! # Why a separate module?
//!
//! The Python proxy currently routes Anthropic-on-Bedrock through the
//! `litellm` shim (`headroom/backends/litellm.py`). That shim
//! lossy-converts every request and response between Anthropic and
//! OpenAI shapes, dropping `thinking`, `redacted_thinking`,
//! `document`, `search_result`, `image`, `server_tool_use`, and
//! `mcp_tool_use` blocks (P4-37). It also hardcodes
//! `stop_sequence: null` (§11.1 violation) and re-wraps
//! `function_call.arguments` as a parsed JSON object (§4.4 — P4-43).
//!
//! Phase D rebuilds the Bedrock surface natively in Rust. PR-D1
//! handles the **non-streaming** `POST /model/{model}/invoke` route:
//!
//! 1. Parse the Bedrock envelope (`{"anthropic_version": "...",
//!    ...rest_of_anthropic_body}`).
//! 2. Route Anthropic-shape bodies through the live-zone compression
//!    path (the same one `/v1/messages` uses).
//! 3. Re-emit the envelope with `anthropic_version` preserved as the
//!    first key — Bedrock is strict about schema validation.
//! 4. Sign the **outgoing** body bytes with AWS SigV4 (after
//!    compression) and forward to the configured Bedrock endpoint.
//!
//! # Cache safety
//!
//! The signed bytes are exactly the bytes Bedrock receives. If the
//! compressor mutated the body, the SigV4 signature is computed
//! against the post-compression bytes; the upstream verifier will
//! accept them. There is no "sign before compress" path — that would
//! produce a signature that doesn't match the wire payload.
//!
//! # Module layout
//!
//! - [`envelope`] — `BedrockEnvelope` parse + emit (preserves
//!   `anthropic_version` ordering byte-equal).
//! - [`sigv4`] — AWS SigV4 signing helper. Wraps the `aws-sigv4`
//!   crate with the project's no-fallback / structured-logging
//!   policy.
//! - [`invoke`] — POST handler for `/model/{model}/invoke`.
//!
//! Streaming (`/model/{model}/invoke-with-response-stream`) is
//! handled by [`invoke_streaming`] in PR-D2. Binary EventStream
//! parsing is in [`eventstream`]; the SSE translator is in
//! [`eventstream_to_sse`].

pub mod auth_mode_layer;
pub mod envelope;
pub mod eventstream;
pub mod eventstream_to_sse;
pub mod invoke;
pub mod invoke_streaming;
pub mod sigv4;
pub mod vendor;

pub use auth_mode_layer::classify_and_attach_auth_mode;
pub use envelope::{BedrockEnvelope, EnvelopeError};
pub use eventstream::{
    parse as parse_eventstream, CrcValidation, EventStreamMessage, EventStreamParser, HeaderValue,
    MessageBuilder, ParseError,
};
pub use eventstream_to_sse::{translate_message, OutputMode, TranslateError, TranslateOutcome};
pub use invoke::handle_invoke;
pub use invoke_streaming::handle_invoke_streaming;
pub use sigv4::{sign_request, SigV4Error, SigningInputs};

/// Build the header list to pass to the SigV4 signer, dropping
/// hop-by-hop, proxy-internal, and signer-managed headers, and
/// injecting the correct `host` from the upstream URL.
pub(crate) fn collect_signed_headers(
    headers: &http::HeaderMap,
    upstream_url: &url::Url,
) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::with_capacity(headers.len() + 1);
    for (name, value) in headers.iter() {
        let n = name.as_str().to_ascii_lowercase();
        if matches!(
            n.as_str(),
            "host"
                | "content-length"
                | "connection"
                | "keep-alive"
                | "proxy-authenticate"
                | "proxy-authorization"
                | "te"
                | "trailers"
                | "transfer-encoding"
                | "upgrade"
                | "authorization"
                | "x-amz-date"
                | "x-amz-content-sha256"
        ) {
            continue;
        }
        if n.starts_with("x-headroom-") {
            continue;
        }
        if let Ok(v) = value.to_str() {
            out.push((n, v.to_string()));
        }
    }
    if let Some(host) = upstream_url.host_str() {
        let host_value = match upstream_url.port() {
            Some(p) => format!("{host}:{p}"),
            None => host.to_string(),
        };
        out.push(("host".to_string(), host_value));
    }
    out
}

/// Build a JSON error response in the Anthropic error envelope shape.
pub(crate) fn error_response(
    status: http::StatusCode,
    event: &str,
    msg: &str,
) -> axum::response::Response {
    use axum::body::Body;
    use axum::response::IntoResponse as _;
    let body = serde_json::json!({
        "error": { "type": event, "message": msg }
    })
    .to_string();
    let mut resp = http::Response::builder()
        .status(status)
        .body(Body::from(body))
        .expect("static error response");
    resp.headers_mut().insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("application/json"),
    );
    resp.into_response()
}
