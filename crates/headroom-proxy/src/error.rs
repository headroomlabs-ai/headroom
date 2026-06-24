//! Error types for the proxy.

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
        let (status, msg) = match &self {
            ProxyError::Upstream(e) if e.is_timeout() => (
                StatusCode::GATEWAY_TIMEOUT,
                format!("upstream timeout: {e}"),
            ),
            ProxyError::Upstream(e) if e.is_connect() => (
                StatusCode::BAD_GATEWAY,
                format!("upstream connect error: {e}"),
            ),
            ProxyError::Upstream(_) => (StatusCode::BAD_GATEWAY, self.to_string()),
            ProxyError::InvalidUpstream(_) => (StatusCode::BAD_GATEWAY, self.to_string()),
            ProxyError::InvalidHeader(_) => (StatusCode::BAD_REQUEST, self.to_string()),
            ProxyError::PayloadTooLarge(_) => (StatusCode::PAYLOAD_TOO_LARGE, self.to_string()),
            ProxyError::WebSocket(_) => (StatusCode::BAD_GATEWAY, self.to_string()),
            ProxyError::Io(_) => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
            // CompressionStartup is a startup-time error, not a
            // per-request one — but if it ever surfaces in the
            // handler path, surface as 500 rather than panic.
            ProxyError::CompressionStartup(_) => {
                (StatusCode::INTERNAL_SERVER_ERROR, self.to_string())
            }
        };
        tracing::warn!(error = %msg, "proxy error");
        (status, msg).into_response()
    }
}
