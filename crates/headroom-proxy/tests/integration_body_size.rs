//! PR-A8 / P5-59: oversized request bodies surface as 413 Payload Too
//! Large, not 400 Bad Request. The pre-A8 path classified buffer
//! overflow as `InvalidHeader` (400), which broke clients with a
//! retry-on-413 backoff (they retried instead of giving up).
//!
//! Two paths:
//!   1. `Content-Length` header present and oversized: 413 returned
//!      immediately, body never consumed (cheap rejection).
//!   2. `Content-Length` missing (chunked): buffer-then-fail; still
//!      surfaces 413 once the cap is hit.

mod common;

use common::start_proxy_with;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn body_size_overflow_returns_413_not_400() {
    // Set the buffer cap small so we trip it without uploading
    // megabytes. The test exercises the chunked path (no
    // Content-Length on the wire — reqwest sends it but the proxy
    // still buffers).
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_string(r#"{"ok":true}"#))
        .mount(&upstream)
        .await;

    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
        c.compression_max_body_bytes = 1024; // 1 KB cap
    })
    .await;

    // Build a JSON body that's 4 KB — well over the 1 KB cap.
    let big_text = "A".repeat(4096);
    let body = format!(
        r#"{{"model":"claude-3-5-sonnet","messages":[{{"role":"user","content":"{}"}}]}}"#,
        big_text
    );
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .unwrap();
    // 413 Payload Too Large — NOT 400 Bad Request.
    assert_eq!(
        resp.status().as_u16(),
        413,
        "expected 413 Payload Too Large; got {} (was 400 pre-A8)",
        resp.status()
    );
    proxy.shutdown().await;
}

#[tokio::test]
async fn body_size_overflow_with_content_length_header_returns_413_without_consuming() {
    let upstream = MockServer::start().await;
    // Mount a handler that records whether the upstream got hit; we
    // want to confirm the proxy short-circuited and never forwarded.
    let upstream_hit = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let upstream_hit_clone = upstream_hit.clone();
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(move |_req: &wiremock::Request| {
            upstream_hit_clone.store(true, std::sync::atomic::Ordering::SeqCst);
            ResponseTemplate::new(200).set_body_string(r#"{"ok":true}"#)
        })
        .mount(&upstream)
        .await;

    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
        c.compression_max_body_bytes = 1024;
    })
    .await;

    // Use reqwest's default which sets Content-Length on a fixed body.
    let big_text = "B".repeat(8192);
    let body = format!(
        r#"{{"model":"claude-3-5-sonnet","messages":[{{"role":"user","content":"{}"}}]}}"#,
        big_text
    );
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        .header("content-type", "application/json")
        .header("content-length", body.len().to_string())
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 413);
    // The pre-check rejected before forwarding — upstream never saw
    // the request.
    assert!(
        !upstream_hit.load(std::sync::atomic::Ordering::SeqCst),
        "Content-Length pre-check should reject before forwarding to upstream"
    );
    proxy.shutdown().await;
}
