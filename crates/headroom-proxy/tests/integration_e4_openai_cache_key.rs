//! Integration tests for PR-E4 OpenAI `prompt_cache_key`
//! auto-injection.
//!
//! Boots a real Rust proxy in front of a wiremock upstream and
//! exercises the four matrix points the spec calls out:
//!
//! 1. PAYG `/v1/chat/completions` request without `prompt_cache_key`
//!    → upstream sees an injected 32-hex-char key.
//! 2. PAYG `/v1/responses` request without `prompt_cache_key`
//!    → upstream sees an injected key.
//! 3. PAYG request WITH `prompt_cache_key` already set
//!    → upstream sees the customer-provided value (passthrough).
//! 4. OAuth and Subscription requests
//!    → upstream sees byte-equal bytes (no key injection at all).
//!
//! For (3), passthrough means the customer's value is preserved
//! AND the rest of the body keeps its byte shape.
//! For (4), the body is byte-equal to the inbound bytes (the
//! universal Phase A invariant).

mod common;

use common::start_proxy_with;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::sync::{Arc, Mutex};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Mount a `/v1/chat/completions` capture handler that records the
/// upstream request body for later assertions.
async fn mount_chat_capture(upstream: &MockServer) -> Arc<Mutex<Option<Vec<u8>>>> {
    let captured: Arc<Mutex<Option<Vec<u8>>>> = Arc::new(Mutex::new(None));
    let captured_clone = captured.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |req: &wiremock::Request| {
            *captured_clone.lock().unwrap() = Some(req.body.clone());
            ResponseTemplate::new(200).set_body_string(r#"{"ok":true}"#)
        })
        .mount(upstream)
        .await;
    captured
}

/// Mount a `/v1/responses` capture handler.
async fn mount_responses_capture(upstream: &MockServer) -> Arc<Mutex<Option<Vec<u8>>>> {
    let captured: Arc<Mutex<Option<Vec<u8>>>> = Arc::new(Mutex::new(None));
    let captured_clone = captured.clone();
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(move |req: &wiremock::Request| {
            *captured_clone.lock().unwrap() = Some(req.body.clone());
            ResponseTemplate::new(200).set_body_string(r#"{"ok":true}"#)
        })
        .mount(upstream)
        .await;
    captured
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .fold(String::with_capacity(64), |mut acc, b| {
            use std::fmt::Write as _;
            let _ = write!(acc, "{b:02x}");
            acc
        })
}

#[track_caller]
fn assert_byte_equal(inbound: &[u8], received: &[u8]) {
    assert_eq!(
        inbound.len(),
        received.len(),
        "byte length mismatch: inbound={} received={}",
        inbound.len(),
        received.len(),
    );
    assert_eq!(
        sha256_hex(inbound),
        sha256_hex(received),
        "SHA-256 mismatch — request was mutated",
    );
}

fn captured_body(captured: &Arc<Mutex<Option<Vec<u8>>>>) -> Vec<u8> {
    captured
        .lock()
        .unwrap()
        .clone()
        .expect("upstream should have captured a body")
}

fn parse_json(bytes: &[u8]) -> Value {
    serde_json::from_slice(bytes).expect("upstream body should be valid JSON")
}

// ────────────────────────────────────────────────────────────────────
// PAYG chat completions: key injected
// ────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn payg_chat_completions_injects_prompt_cache_key() {
    let upstream = MockServer::start().await;
    let captured = mount_chat_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
    })
    .await;

    let payload = json!({
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"}
        ]
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", proxy.url()))
        .header("content-type", "application/json")
        // PAYG: standard OpenAI sk- bearer token.
        .header("authorization", "Bearer sk-test-payg-key")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let received = captured_body(&captured);
    let received_json = parse_json(&received);

    let key = received_json
        .get("prompt_cache_key")
        .and_then(Value::as_str)
        .expect("upstream body should have prompt_cache_key on PAYG");
    assert_eq!(key.len(), 32, "key must be 32 hex chars, got {key:?}");
    assert!(
        key.bytes().all(|b| b.is_ascii_hexdigit()),
        "key must be hex: {key}"
    );

    // Other fields preserved structurally (parse-equivalent).
    assert_eq!(
        received_json.get("model").and_then(Value::as_str),
        Some("gpt-4o")
    );
    let inbound_json = parse_json(&body);
    assert_eq!(received_json.get("messages"), inbound_json.get("messages"));

    proxy.shutdown().await;
}

// ────────────────────────────────────────────────────────────────────
// PAYG responses: key injected
// ────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn payg_responses_injects_prompt_cache_key() {
    let upstream = MockServer::start().await;
    let captured = mount_responses_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
    })
    .await;

    let payload = json!({
        "model": "gpt-4o",
        "instructions": "You are a helpful assistant.",
        "input": [{"role": "user", "content": "Hello"}]
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/responses", proxy.url()))
        .header("content-type", "application/json")
        .header("authorization", "Bearer sk-test-payg-key")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let received = captured_body(&captured);
    let received_json = parse_json(&received);

    let key = received_json
        .get("prompt_cache_key")
        .and_then(Value::as_str)
        .expect("upstream body should have prompt_cache_key on PAYG /v1/responses");
    assert_eq!(key.len(), 32, "key must be 32 hex chars, got {key:?}");

    // `instructions` and `input` preserved.
    assert_eq!(
        received_json.get("instructions").and_then(Value::as_str),
        Some("You are a helpful assistant.")
    );
    let inbound_json = parse_json(&body);
    assert_eq!(received_json.get("input"), inbound_json.get("input"));

    proxy.shutdown().await;
}

// ────────────────────────────────────────────────────────────────────
// PAYG with customer-set key: customer wins (passthrough)
// ────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn payg_chat_with_customer_set_key_preserves_value() {
    let upstream = MockServer::start().await;
    let captured = mount_chat_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
    })
    .await;

    let payload = json!({
        "model": "gpt-4o",
        "prompt_cache_key": "user-pinned-tenant-A",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"}
        ]
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", proxy.url()))
        .header("content-type", "application/json")
        .header("authorization", "Bearer sk-test-payg-key")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let received = captured_body(&captured);
    // The dispatcher's live-zone surgery must NOT have rewritten
    // the body (the live-zone is the latest user/tool message,
    // which here is below the byte threshold). And the E4 hook
    // must have skipped (customer key present). Net effect: bytes
    // round-trip byte-equal to inbound.
    assert_byte_equal(&body, &received);

    let received_json = parse_json(&received);
    assert_eq!(
        received_json
            .get("prompt_cache_key")
            .and_then(Value::as_str),
        Some("user-pinned-tenant-A"),
        "customer-set prompt_cache_key must be preserved"
    );

    proxy.shutdown().await;
}

// ────────────────────────────────────────────────────────────────────
// OAuth: no injection, byte-equal passthrough
// ────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn oauth_chat_completions_no_injection_byte_equal() {
    let upstream = MockServer::start().await;
    let captured = mount_chat_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
    })
    .await;

    let payload = json!({
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"}
        ]
    });
    let body = serde_json::to_vec(&payload).unwrap();
    // OAuth: 3-segment JWT shape. Headroom's auth_mode classifier
    // detects this and routes to AuthMode::OAuth, which the E4
    // hook short-circuits.
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", proxy.url()))
        .header("content-type", "application/json")
        .header(
            "authorization",
            "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ0ZXN0In0.signature_bytes",
        )
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let received = captured_body(&captured);
    assert_byte_equal(&body, &received);

    let received_json = parse_json(&received);
    assert!(
        received_json.get("prompt_cache_key").is_none(),
        "OAuth requests must not get prompt_cache_key injected; got {received_json}"
    );

    proxy.shutdown().await;
}

// ────────────────────────────────────────────────────────────────────
// Subscription: no injection, byte-equal passthrough
// ────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn subscription_chat_completions_no_injection_byte_equal() {
    let upstream = MockServer::start().await;
    let captured = mount_chat_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
    })
    .await;

    let payload = json!({
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"}
        ]
    });
    let body = serde_json::to_vec(&payload).unwrap();
    // Subscription clients: identified by user-agent prefix
    // (Codex CLI on `/v1/chat/completions` is the canonical case).
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", proxy.url()))
        .header("content-type", "application/json")
        .header("authorization", "Bearer sk-test-payg-key")
        .header("user-agent", "codex-cli/1.0.0")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let received = captured_body(&captured);
    assert_byte_equal(&body, &received);

    let received_json = parse_json(&received);
    assert!(
        received_json.get("prompt_cache_key").is_none(),
        "Subscription requests must not get prompt_cache_key injected"
    );

    proxy.shutdown().await;
}

// ────────────────────────────────────────────────────────────────────
// OAuth on /v1/responses: no injection, byte-equal passthrough
// ────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn oauth_responses_no_injection_byte_equal() {
    let upstream = MockServer::start().await;
    let captured = mount_responses_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
    })
    .await;

    let payload = json!({
        "model": "gpt-4o",
        "instructions": "Be concise.",
        "input": [{"role": "user", "content": "Hello"}]
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/responses", proxy.url()))
        .header("content-type", "application/json")
        .header(
            "authorization",
            "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ0ZXN0In0.signature_bytes",
        )
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let received = captured_body(&captured);
    assert_byte_equal(&body, &received);

    let received_json = parse_json(&received);
    assert!(
        received_json.get("prompt_cache_key").is_none(),
        "OAuth /v1/responses requests must not get prompt_cache_key injected"
    );

    proxy.shutdown().await;
}
