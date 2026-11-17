//! The proxy forwards the client's path verbatim under the configured
//! prefix, or not at all. `url::Url::set_path` resolves `.`/`..` segments
//! (also spelled `%2e`) and folds `\` into `/`, so without a guard a client
//! could:
//!
//! * walk out of a tenant prefix on the generic forwarder
//!   (`https://gw/tenant-a` + `/../tenant-b/...`),
//! * move a SigV4-signed Bedrock request to a different resource
//!   (`/model/../invoke`, `/model/..%2F..%2Fother/invoke`),
//! * move an ADC-signed Vertex request to another project
//!   (`/v1beta1/projects/../../other/...`).
//!
//! HTTP clients normalise dot segments before sending, so the traversal
//! requests here go over a raw TCP socket. See `headroom_proxy::upstream_path`.

mod common;

use std::net::SocketAddr;

use aws_credential_types::Credentials;
use common::{install_static_token_source, start_proxy, start_proxy_with_state};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use url::Url;
use wiremock::matchers::any;
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Send one HTTP/1.1 request exactly as written and return the status code.
async fn raw_request(
    addr: SocketAddr,
    method: &str,
    target: &str,
    extra_headers: &[(&str, &str)],
    body: &[u8],
) -> u16 {
    let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let mut req = format!("{method} {target} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n");
    for (k, v) in extra_headers {
        req.push_str(&format!("{k}: {v}\r\n"));
    }
    if !body.is_empty() {
        req.push_str("content-type: application/json\r\n");
        req.push_str(&format!("content-length: {}\r\n", body.len()));
    }
    req.push_str("\r\n");
    stream.write_all(req.as_bytes()).await.unwrap();
    stream.write_all(body).await.unwrap();
    let mut out = Vec::new();
    stream.read_to_end(&mut out).await.unwrap();
    let head = String::from_utf8_lossy(&out);
    let status_line = head.lines().next().unwrap_or("");
    status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("no status line in response: {head:?}"))
}

async fn upstream_paths(mock: &MockServer) -> Vec<String> {
    mock.received_requests()
        .await
        .unwrap_or_default()
        .iter()
        .map(|r| r.url.path().to_string())
        .collect()
}

async fn accept_all(mock: &MockServer) {
    Mock::given(any())
        .respond_with(ResponseTemplate::new(200).set_body_string(r#"{"id":"msg_x","content":[]}"#))
        .mount(mock)
        .await;
}

// ─── generic forwarder ───────────────────────────────────────────────────

#[tokio::test]
async fn forwarder_keeps_requests_under_the_base_prefix() {
    let mock = MockServer::start().await;
    accept_all(&mock).await;
    let proxy = start_proxy(&format!("{}/tenant-a", mock.uri())).await;

    // Control: an ordinary path lands under the prefix.
    assert_eq!(
        raw_request(proxy.addr, "GET", "/v1/models", &[], b"").await,
        200
    );
    assert_eq!(
        upstream_paths(&mock).await,
        vec!["/tenant-a/v1/models".to_string()]
    );

    // Every dot-segment spelling and the backslash are refused before any
    // upstream call. `url` would have resolved these to `/tenant-b/...`.
    for target in [
        "/../tenant-b/v1/models",
        "/v1/../../tenant-b/v1/models",
        "/%2e%2e/tenant-b/v1/models",
        "/.%2e/tenant-b/v1/models",
        "/%2E%2E/tenant-b/v1/models",
        "/..\\tenant-b/v1/models",
    ] {
        let status = raw_request(proxy.addr, "GET", target, &[], b"").await;
        assert_eq!(status, 400, "{target} must be rejected");
    }
    assert_eq!(
        upstream_paths(&mock).await.len(),
        1,
        "no traversal request may reach the upstream"
    );
    proxy.shutdown().await;
}

#[tokio::test]
async fn websocket_upgrade_with_traversal_path_is_rejected_before_upgrade() {
    let mock = MockServer::start().await;
    let proxy = start_proxy(&format!("{}/tenant-a", mock.uri())).await;
    let status = raw_request(
        proxy.addr,
        "GET",
        "/../tenant-b/ws",
        &[
            ("Upgrade", "websocket"),
            ("Connection", "Upgrade"),
            ("Sec-WebSocket-Key", "dGhlIHNhbXBsZSBub25jZQ=="),
            ("Sec-WebSocket-Version", "13"),
        ],
        b"",
    )
    .await;
    assert_eq!(status, 400, "a traversal path must not be upgraded (101)");
    proxy.shutdown().await;
}

// ─── Bedrock (SigV4-signed) ──────────────────────────────────────────────

fn test_credentials() -> Credentials {
    Credentials::new(
        "AKIAEXAMPLEAKIDFORTEST",
        "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        None,
        None,
        "test",
    )
}

async fn bedrock_proxy(mock: &MockServer) -> common::ProxyHandle {
    let endpoint: Url = mock.uri().parse().unwrap();
    start_proxy_with_state(
        &mock.uri(),
        |c| c.bedrock_endpoint = Some(endpoint),
        |s| s.with_bedrock_credentials(test_credentials()),
    )
    .await
}

const BODY: &[u8] =
    br#"{"anthropic_version":"bedrock-2023-05-31","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}"#;

#[tokio::test]
async fn bedrock_dot_segment_model_id_is_rejected_unsigned() {
    let mock = MockServer::start().await;
    accept_all(&mock).await;
    let proxy = bedrock_proxy(&mock).await;

    // `/model/../invoke` used to become a signed request for `/invoke`.
    for target in [
        "/model/../invoke",
        "/model/./converse",
        "/model/%2e%2e/invoke",
    ] {
        let status = raw_request(proxy.addr, "POST", target, &[], BODY).await;
        assert_eq!(status, 400, "{target} must be rejected");
    }
    assert!(
        upstream_paths(&mock).await.is_empty(),
        "no signed request may leave for a rejected model id"
    );
    proxy.shutdown().await;
}

#[tokio::test]
async fn bedrock_encoded_slashes_in_model_id_stay_one_segment() {
    let mock = MockServer::start().await;
    accept_all(&mock).await;
    let proxy = bedrock_proxy(&mock).await;

    // The traversal shape: decoded, this model id is `../../guardrail/...`.
    // Before the fix the signed request went to
    // `/guardrail/GID/version/1/apply/invoke`. Now the id is re-encoded and
    // the upstream sees it as one (nonsensical) model segment.
    let resp = reqwest::Client::new()
        .post(format!(
            "{}/model/..%2F..%2Fguardrail%2FGID%2Fversion%2F1%2Fapply/invoke",
            proxy.url()
        ))
        .header("content-type", "application/json")
        .body(BODY)
        .send()
        .await
        .unwrap();
    assert_ne!(resp.status(), 500);
    let paths = upstream_paths(&mock).await;
    assert!(
        !paths.iter().any(|p| p.starts_with("/guardrail/")),
        "traversal reached the upstream: {paths:?}"
    );
    for p in &paths {
        assert!(p.starts_with("/model/") && p.ends_with("/invoke"), "{p}");
        assert!(
            p.contains("%2F"),
            "slashes inside the model id must stay encoded: {p}"
        );
    }

    // The legitimate case with the same shape: an inference-profile ARN
    // carries a `/`, which the AWS SDKs send as `%2F`. It must arrive as one
    // segment, exactly as sent.
    let arn = "arn:aws:bedrock:us-east-1:123456789012:inference-profile%2Fus.anthropic.claude-3-5-haiku-20241022-v1:0";
    let resp = reqwest::Client::new()
        .post(format!("{}/model/{arn}/invoke", proxy.url()))
        .header("content-type", "application/json")
        .body(BODY)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let paths = upstream_paths(&mock).await;
    assert_eq!(
        paths.last().map(String::as_str),
        Some(format!("/model/{arn}/invoke").as_str())
    );
    proxy.shutdown().await;
}

#[tokio::test]
async fn bedrock_streaming_dot_segment_model_id_is_rejected_unsigned() {
    let mock = MockServer::start().await;
    accept_all(&mock).await;
    let proxy = bedrock_proxy(&mock).await;

    for target in [
        "/model/../invoke-with-response-stream",
        "/model/../converse-stream",
    ] {
        let status = raw_request(proxy.addr, "POST", target, &[], BODY).await;
        assert_eq!(status, 400, "{target} must be rejected");
    }
    assert!(upstream_paths(&mock).await.is_empty());
    proxy.shutdown().await;
}

// ─── Vertex (ADC-signed) ─────────────────────────────────────────────────

#[tokio::test]
async fn vertex_dot_segment_project_is_rejected_unsigned() {
    let mock = MockServer::start().await;
    accept_all(&mock).await;
    let proxy = start_proxy_with_state(
        &mock.uri(),
        |_| {},
        |s| install_static_token_source(s, "test-bearer"),
    )
    .await;

    // Would have resolved to `/evil/locations/...` carrying the operator's
    // ADC bearer.
    let target = "/v1beta1/projects/../../evil/locations/us-central1/publishers/anthropic/models/claude-3-5-sonnet@20240620:rawPredict";
    let status = raw_request(proxy.addr, "POST", target, &[], BODY).await;
    assert_eq!(status, 400);
    assert!(upstream_paths(&mock).await.is_empty());
    proxy.shutdown().await;
}
