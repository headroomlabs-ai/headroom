//! Error types for the proxy.
//!
//! # Security policy
//!
//! Error messages returned to clients MUST NOT contain internal topology
//! information (upstream URLs, IPs, ports, DNS names). The full error
//! detail is logged at `warn` level for operators; clients receive only
//! a generic status-appropriate message. This prevents reconnaissance
//! of the upstream infrastructure via crafted requests that trigger
//! error responses.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProxyError {
    #[error("upstream request failed: {0}")]
    Upstream(#[from] reqwest::Error),

    #[error("invalid upstream URL: {0}")]
    InvalidUpstream(String),

    #[error("invalid header: {0}")]
    InvalidHeader(String),

    #[error("websocket error: {0}")]
    WebSocket(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// PR-A8 / P5-59: request body exceeded the configured cap. RFC 7231
    /// §6.5.11: 413 Payload Too Large. Previously surfaced as
    /// `InvalidHeader` (400) which mis-classified an oversize body as a
    /// header parse error; clients with retry-on-413 logic broke.
    #[error("request body exceeds configured limit: {0}")]
    PayloadTooLarge(String),

    /// Surfaced when `--compression` is enabled but the proxy can't
    /// build the IntelligentContextManager at startup (e.g. the
    /// embedded tokenizer asset failed to initialize). Bubbles up to
    /// `main` as a fatal startup error rather than a per-request
    /// failure — if compression is configured but the engine won't
    /// build, the operator should know immediately, not at first
    /// LLM request.
    #[error("compression engine startup failed: {0}")]
    CompressionStartup(String),
}

impl IntoResponse for ProxyError {
    fn into_response(self) -> Response {
        // Internal detail for operator logs — never sent to client.
        let internal_detail = self.to_string();

        // Client-facing message: generic, status-appropriate text that
        // does NOT leak upstream URLs, IPs, ports, or DNS names.
        let (status, client_msg) = match &self {
            ProxyError::Upstream(e) if e.is_timeout() => (
                StatusCode::GATEWAY_TIMEOUT,
                "upstream request timed out".to_string(),
            ),
            ProxyError::Upstream(e) if e.is_connect() => (
                StatusCode::BAD_GATEWAY,
                "failed to connect to upstream".to_string(),
            ),
            ProxyError::Upstream(_) => (
                StatusCode::BAD_GATEWAY,
                "upstream request failed".to_string(),
            ),
            ProxyError::InvalidUpstream(_) => (
                StatusCode::BAD_GATEWAY,
                "invalid upstream configuration".to_string(),
            ),
            // InvalidHeader is a client-caused error; safe to echo
            // the detail since it describes the client's own input.
            ProxyError::InvalidHeader(detail) => (
                StatusCode::BAD_REQUEST,
                format!("invalid header: {detail}"),
            ),
            // PayloadTooLarge is client-caused; the configured limit
            // is not sensitive (it's a public contract), so the
            // message from the variant is safe to return.
            ProxyError::PayloadTooLarge(detail) => (
                StatusCode::PAYLOAD_TOO_LARGE,
                detail.clone(),
            ),
            ProxyError::WebSocket(_) => (
                StatusCode::BAD_GATEWAY,
                "websocket upstream error".to_string(),
            ),
            ProxyError::Io(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal server error".to_string(),
            ),
            // CompressionStartup is a startup-time error, not a
            // per-request one — but if it ever surfaces in the
            // handler path, surface as 500 rather than panic.
            ProxyError::CompressionStartup(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal server error".to_string(),
            ),
        };

        // Log the FULL internal detail for operators (never sent to client).
        tracing::warn!(
            error = %internal_detail,
            status = status.as_u16(),
            "proxy error"
        );

        (status, client_msg).into_response()
    }
}
