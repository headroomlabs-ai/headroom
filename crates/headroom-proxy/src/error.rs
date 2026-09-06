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
        // Full detail is for operators only. The `Display` impls above
        // interpolate the source error (e.g. reqwest's chain), which for
        // upstream failures can embed the upstream host/port and other
        // internal wiring. We log that detail but return a generic,
        // stable body to the client so an error response can't be used
        // to fingerprint the upstream. The HTTP status still carries the
        // actionable signal for well-behaved clients.
        let detail = self.to_string();
        let (status, client_msg): (StatusCode, &'static str) = match &self {
            ProxyError::Upstream(e) if e.is_timeout() => {
                (StatusCode::GATEWAY_TIMEOUT, "upstream timeout")
            }
            ProxyError::Upstream(e) if e.is_connect() => {
                (StatusCode::BAD_GATEWAY, "upstream connection failed")
            }
            ProxyError::Upstream(_) => (StatusCode::BAD_GATEWAY, "upstream request failed"),
            ProxyError::InvalidUpstream(_) => (StatusCode::BAD_GATEWAY, "upstream request failed"),
            ProxyError::InvalidHeader(_) => (StatusCode::BAD_REQUEST, "invalid request"),
            // PayloadTooLarge stays descriptive: it leaks no internal
            // detail (the interpolated value is the client's own size vs
            // the configured cap) and clients with retry-on-413 logic
            // rely on recognizing it. See the PayloadTooLarge doc comment.
            ProxyError::PayloadTooLarge(_) => (
                StatusCode::PAYLOAD_TOO_LARGE,
                "request body exceeds configured limit",
            ),
            ProxyError::WebSocket(_) => (StatusCode::BAD_GATEWAY, "upstream request failed"),
            ProxyError::Io(_) => (StatusCode::INTERNAL_SERVER_ERROR, "internal error"),
            // CompressionStartup is a startup-time error, not a
            // per-request one — but if it ever surfaces in the
            // handler path, surface as 500 rather than panic.
            ProxyError::CompressionStartup(_) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "internal error")
            }
        };
        tracing::warn!(status = %status.as_u16(), error = %detail, "proxy error");
        (status, client_msg).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;

    async fn status_and_body(err: ProxyError) -> (StatusCode, String) {
        let resp = err.into_response();
        let status = resp.status();
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        (status, String::from_utf8_lossy(&bytes).to_string())
    }

    // The interpolated detail (which for real upstream errors can embed
    // the upstream host/port) must NEVER reach the client body. Before
    // the hardening these bodies were `self.to_string()` and DID contain
    // the detail, so these assertions fail on old code.
    #[tokio::test]
    async fn invalid_header_detail_is_not_leaked_to_client() {
        let (status, body) = status_and_body(ProxyError::InvalidHeader(
            "upstream-secret-host:9443".into(),
        ))
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body, "invalid request");
        assert!(!body.contains("upstream-secret-host"));
    }

    #[tokio::test]
    async fn invalid_upstream_detail_is_not_leaked_to_client() {
        let (status, body) = status_and_body(ProxyError::InvalidUpstream(
            "http://10.0.0.5:8788/internal".into(),
        ))
        .await;
        assert_eq!(status, StatusCode::BAD_GATEWAY);
        assert_eq!(body, "upstream request failed");
        assert!(!body.contains("10.0.0.5"));
    }

    #[tokio::test]
    async fn payload_too_large_stays_descriptive() {
        // Deliberately kept descriptive: no internal detail, and
        // retry-on-413 clients rely on recognizing it.
        let (status, body) =
            status_and_body(ProxyError::PayloadTooLarge("body exceeds cap".into())).await;
        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(body, "request body exceeds configured limit");
    }
}
