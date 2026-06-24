//! Integration tests for PR-E1: tool array deterministic sort.
//!
//! Boots a real Rust proxy in front of a wiremock upstream and
//! exercises the three live-zone walkers via the inbound paths the
//! proxy actually serves. Asserts:
//!
//!   1. **PAYG path** (e.g. `x-api-key` on Anthropic): tools arrive
//!      at the upstream sorted alphabetically, regardless of the
//!      client's input order.
//!   2. **Subscription path** (UA prefix `claude-cli/...`): tools
//!      pass through verbatim — bytes the upstream sees match the
//!      bytes the client sent (asserted via SHA-256 byte-equality).
//!   3. **Customer-marker path** (PAYG, but at least one tool already
//!      carries `cache_control`): tools pass through verbatim — the
//!      sort is gated off so the customer's intentional layout wins.
//!
//! The Phase A cache-safety invariant — the proxy NEVER mutates
//! request bytes when a gate skips — is the contract under test.

mod common;

use common::start_proxy_with;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::sync::{Arc, Mutex};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut s = String::with_capacity(64);
    for b in digest {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Mount a `/v1/messages` handler that captures the upstream-received
/// request body for later inspection.
async fn mount_anthropic_capture(upstream: &MockServer) -> Arc<Mutex<Option<Vec<u8>>>> {
    let captured: Arc<Mutex<Option<Vec<u8>>>> = Arc::new(Mutex::new(None));
    let captured_clone = captured.clone();
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(move |req: &wiremock::Request| {
            *captured_clone.lock().unwrap() = Some(req.body.clone());
            ResponseTemplate::new(200).set_body_string(r#"{"ok":true}"#)
        })
        .mount(upstream)
        .await;
    captured
}

/// PAYG: send tools in reverse alphabetical order, expect upstream to
/// receive them sorted by `name`.
#[tokio::test]
async fn payg_request_with_unsorted_tools_is_sorted_at_upstream() {
    let upstream = MockServer::start().await;
    let captured = mount_anthropic_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
    })
    .await;

    let payload = json!({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {"name": "zebra", "description": "z"},
            {"name": "apple", "description": "a"},
            {"name": "mango", "description": "m"},
        ],
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        // PAYG signal: x-api-key header (Anthropic API-key style).
        .header("x-api-key", "sk-ant-api03-abc")
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let upstream_body = captured
        .lock()
        .unwrap()
        .clone()
        .expect("upstream should have captured");
    let parsed: Value = serde_json::from_slice(&upstream_body).expect("upstream body is JSON");
    let tools = parsed.get("tools").and_then(Value::as_array).unwrap();
    let names: Vec<&str> = tools
        .iter()
        .map(|t| t.get("name").and_then(Value::as_str).unwrap())
        .collect();
    assert_eq!(
        names,
        vec!["apple", "mango", "zebra"],
        "PAYG path must deliver tools to upstream sorted alphabetically by name",
    );

    proxy.shutdown().await;
}

/// Subscription: same body shape but with a `claude-cli/...` UA →
/// proxy must NOT mutate. Upstream-received bytes must be byte-equal
/// to client-sent bytes (SHA-256 match).
#[tokio::test]
async fn subscription_request_passes_tools_through_byte_equal() {
    let upstream = MockServer::start().await;
    let captured = mount_anthropic_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
    })
    .await;

    let payload = json!({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {"name": "zebra", "description": "z"},
            {"name": "apple", "description": "a"},
        ],
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let body_hash = sha256_hex(&body);

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        // Subscription signal: claude-cli UA prefix.
        .header("user-agent", "claude-cli/1.0.0")
        .header("authorization", "Bearer sk-ant-oat-pretend")
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let upstream_body = captured
        .lock()
        .unwrap()
        .clone()
        .expect("upstream should have captured");
    assert_eq!(
        sha256_hex(&upstream_body),
        body_hash,
        "Subscription path must pass body bytes through unchanged"
    );

    proxy.shutdown().await;
}

/// PAYG, but customer placed `cache_control` on a tool → sort is
/// skipped; bytes pass through verbatim.
#[tokio::test]
async fn payg_with_marker_passes_tools_through_byte_equal() {
    let upstream = MockServer::start().await;
    let captured = mount_anthropic_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
    })
    .await;

    let payload = json!({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {"name": "zebra", "description": "z"},
            {"name": "apple", "description": "a", "cache_control": {"type": "ephemeral"}},
        ],
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let body_hash = sha256_hex(&body);

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        .header("x-api-key", "sk-ant-api03-abc")
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let upstream_body = captured
        .lock()
        .unwrap()
        .clone()
        .expect("upstream should have captured");
    assert_eq!(
        sha256_hex(&upstream_body),
        body_hash,
        "PAYG with customer cache_control marker must pass body bytes through unchanged"
    );

    proxy.shutdown().await;
}

/// OAuth: same body shape but with a `Bearer sk-ant-oat-...` token
/// (no claude-cli UA prefix → classified OAuth, not Subscription).
/// Tools pass through verbatim.
#[tokio::test]
async fn oauth_request_passes_tools_through_byte_equal() {
    let upstream = MockServer::start().await;
    let captured = mount_anthropic_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
    })
    .await;

    let payload = json!({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {"name": "zebra", "description": "z"},
            {"name": "apple", "description": "a"},
        ],
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let body_hash = sha256_hex(&body);

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        // OAuth signal: Anthropic OAuth token shape.
        .header("authorization", "Bearer sk-ant-oat-foo")
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let upstream_body = captured
        .lock()
        .unwrap()
        .clone()
        .expect("upstream should have captured");
    assert_eq!(
        sha256_hex(&upstream_body),
        body_hash,
        "OAuth path must pass body bytes through unchanged"
    );

    proxy.shutdown().await;
}
