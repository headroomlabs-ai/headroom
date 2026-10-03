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

    /// The request path, or a decoded route parameter, would have to be
    /// rewritten on the way upstream (dot segments, backslashes, or a
    /// segment that would gain or lose a `/` boundary). The proxy forwards
    /// paths verbatim or not at all — see `crate::upstream_path`. RFC 7231
    /// §6.5.1: 400.
    #[error("request path rejected: {0}")]
    InvalidPath(String),

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

impl From<crate::upstream_path::PathError> for ProxyError {
    fn from(e: crate::upstream_path::PathError) -> Self {
        ProxyError::InvalidPath(e.to_string())
    }
}

impl IntoResponse for ProxyError {
    fn into_response(self) -> Response {
        let internal_detail = self.to_string();
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
            ProxyError::InvalidHeader(detail) => {
                (StatusCode::BAD_REQUEST, format!("invalid header: {detail}"))
            }
            ProxyError::PayloadTooLarge(detail) => (StatusCode::PAYLOAD_TOO_LARGE, detail.clone()),
            ProxyError::InvalidPath(_) => (StatusCode::BAD_REQUEST, self.to_string()),
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
        tracing::warn!(error = %internal_detail, status = status.as_u16(), "proxy error");
        (status, client_msg).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use std::time::Duration;

    async fn response_body(error: ProxyError) -> (StatusCode, String) {
        let response = error.into_response();
        let status = response.status();
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        (status, String::from_utf8(body.to_vec()).unwrap())
    }

    #[tokio::test]
    async fn internal_upstream_configuration_is_not_returned_to_client() {
        let secret = "http://10.23.45.67:9443/private";
        let (status, body) = response_body(ProxyError::InvalidUpstream(secret.to_string())).await;

        assert_eq!(status, StatusCode::BAD_GATEWAY);
        assert_eq!(body, "invalid upstream configuration");
        assert!(!body.contains(secret));
    }

    #[tokio::test]
    async fn websocket_and_io_details_are_not_returned_to_client() {
        let (_, websocket_body) =
            response_body(ProxyError::WebSocket("dial tcp 10.0.0.8:443".to_string())).await;
        assert_eq!(websocket_body, "websocket upstream error");

        let (_, io_body) =
            response_body(ProxyError::Io(std::io::Error::other("/srv/internal/token"))).await;
        assert_eq!(io_body, "internal server error");
    }

    #[tokio::test]
    async fn compression_startup_detail_is_not_returned_to_client() {
        let (status, body) = response_body(ProxyError::CompressionStartup(
            "failed to load /srv/models/private-tokenizer.bin".to_string(),
        ))
        .await;

        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(body, "internal server error");
    }

    #[tokio::test]
    async fn reqwest_builder_and_connect_details_are_not_returned_to_client() {
        let builder_error = reqwest::Client::new()
            .get("://invalid-internal-url")
            .send()
            .await
            .unwrap_err();
        let (status, body) = response_body(ProxyError::Upstream(builder_error)).await;
        assert_eq!(status, StatusCode::BAD_GATEWAY);
        assert_eq!(body, "upstream request failed");

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        drop(listener);
        let connect_error = reqwest::Client::new()
            .get(format!("http://{address}/private"))
            .send()
            .await
            .unwrap_err();
        assert!(connect_error.is_connect());
        let (status, body) = response_body(ProxyError::Upstream(connect_error)).await;
        assert_eq!(status, StatusCode::BAD_GATEWAY);
        assert_eq!(body, "failed to connect to upstream");
        assert!(!body.contains(&address.to_string()));
    }

    #[tokio::test]
    async fn reqwest_timeout_detail_is_not_returned_to_client() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (_stream, _) = listener.accept().await.unwrap();
            tokio::time::sleep(Duration::from_secs(2)).await;
        });
        let timeout_error = reqwest::Client::builder()
            .timeout(Duration::from_millis(25))
            .build()
            .unwrap()
            .get(format!("http://{address}/private"))
            .send()
            .await
            .unwrap_err();
        server.abort();

        assert!(timeout_error.is_timeout());
        let (status, body) = response_body(ProxyError::Upstream(timeout_error)).await;
        assert_eq!(status, StatusCode::GATEWAY_TIMEOUT);
        assert_eq!(body, "upstream request timed out");
        assert!(!body.contains(&address.to_string()));
    }

    #[tokio::test]
    async fn client_caused_error_details_remain_available() {
        let (status, body) =
            response_body(ProxyError::InvalidHeader("bad client header".to_string())).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body, "invalid header: bad client header");

        let (status, body) = response_body(ProxyError::PayloadTooLarge(
            "limit is 1024 bytes".to_string(),
        ))
        .await;
        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(body, "limit is 1024 bytes");

        let (status, body) =
            response_body(ProxyError::InvalidPath("dot segment".to_string())).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body, "request path rejected: dot segment");
    }
}
