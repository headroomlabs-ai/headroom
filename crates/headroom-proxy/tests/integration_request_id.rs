//! PR-A8 / P5-57: capture upstream `request-id` (Anthropic) and
//! `x-request-id` (OpenAI) and forward them in a distinct header so
//! operators can correlate proxy logs without conflating with the
//! proxy's own `x-request-id`.

mod common;

use common::start_proxy;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn upstream_anthropic_request_id_captured() {
    let upstream = MockServer::start().await;
    let upstream_id = "req_anthropic_xyz_123";
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("request-id", upstream_id)
                .set_body_string(r#"{"ok":true}"#),
        )
        .mount(&upstream)
        .await;

    let proxy = start_proxy(&upstream.uri()).await;
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        .header("content-type", "application/json")
        .body(r#"{"model":"claude-3-5-sonnet","messages":[]}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    // Anthropic's `request-id` header is forwarded verbatim AND
    // surfaced under the side-channel header.
    let echoed = resp
        .headers()
        .get("headroom-upstream-request-id")
        .and_then(|v| v.to_str().ok());
    assert_eq!(echoed, Some(upstream_id));
    // The original `request-id` header is also forwarded.
    let raw = resp
        .headers()
        .get("request-id")
        .and_then(|v| v.to_str().ok());
    assert_eq!(raw, Some(upstream_id));
    proxy.shutdown().await;
}

#[tokio::test]
async fn upstream_openai_x_request_id_captured() {
    let upstream = MockServer::start().await;
    let upstream_id = "req_openai_abc_456";
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("x-request-id", upstream_id)
                .set_body_string(r#"{"ok":true}"#),
        )
        .mount(&upstream)
        .await;

    let proxy = start_proxy(&upstream.uri()).await;
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", proxy.url()))
        .header("content-type", "application/json")
        .body(r#"{"model":"gpt-4o","messages":[]}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let echoed = resp
        .headers()
        .get("headroom-upstream-request-id")
        .and_then(|v| v.to_str().ok());
    assert_eq!(echoed, Some(upstream_id));
    proxy.shutdown().await;
}
