//! End-to-end integration coverage for the `[1m]` Anthropic
//! model-suffix sanitizer (PR #2027).
//!
//! Review feedback (JerrettDavis):
//!
//! > Please scope the sanitizer to `CompressibleEndpoint::AnthropicMessages`
//! > after classification, and add Rust coverage for both sides: one test
//! > that `/v1/messages` strips `glm-5.2[1m]` / Claude-style suffixes
//! > before forwarding, and one test that an OpenAI chat/responses
//! > request with a literal model ending in `[1m]` is left byte-for-byte
//! > unchanged unless some other explicit OpenAI mutation applies.
//!
//! These tests boot a real Rust proxy in front of a wiremock upstream
//! and observe the bytes the upstream actually receives. The
//! OpenAI-side tests assert SHA-256 byte equality (per the project's
//! cache-safety contract for byte-faithful passthrough) and the
//! Anthropic-side tests assert the upstream receives the suffix-free
//! model ID by parsing the captured body as JSON.
//!
//! Each test uses a model ID that ends in a literal `[1m]`. On the
//! Anthropic path the proxy must strip the suffix before forwarding;
//! on the OpenAI paths the proxy must leave the body byte-equal to
//! what the client sent.
//!
//! Why split into two assertions (JSON-parse + SHA-256) rather than
//! only SHA-256? Because the Anthropic assertion is "the upstream
//! received a body whose `model` field is `glm-5.2`" — i.e. the
//! suffix is actually gone — while the OpenAI assertion is "the
//! upstream received exactly these bytes" (per the cache-safety
//! invariant). Conflating them obscures what each side is pinning.

mod common;

use bytes::Bytes;
use common::start_proxy_with;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::sync::{Arc, Mutex};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Mount a `/v1/messages` capture on the wiremock upstream.
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

/// Mount a `/v1/chat/completions` capture on the wiremock upstream.
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

/// Mount a `/v1/responses` capture on the wiremock upstream.
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
fn assert_byte_equal_sha256(inbound: &[u8], received: &[u8]) {
    let inbound_hash = sha256_hex(inbound);
    let received_hash = sha256_hex(received);
    assert_eq!(
        inbound.len(),
        received.len(),
        "byte length mismatch: inbound={}, upstream-received={}",
        inbound.len(),
        received.len(),
    );
    assert_eq!(
        inbound_hash, received_hash,
        "SHA-256 mismatch: inbound={inbound_hash}, upstream-received={received_hash}",
    );
}

// ─── Anthropic: /v1/messages must strip the suffix ───────────────────

#[tokio::test]
async fn anthropic_messages_strips_1m_suffix_glm() {
    let upstream = MockServer::start().await;
    let captured = mount_anthropic_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
    })
    .await;

    let payload = json!({
        "model": "glm-5.2[1m]",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "hello"}],
    });
    let body = Bytes::from(serde_json::to_vec(&payload).unwrap());
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let got = captured.lock().unwrap().clone().expect("upstream got body");
    let parsed: Value = serde_json::from_slice(&got).expect("upstream body is JSON");
    // The whole point of PR #2027: the upstream Anthropic API must
    // see `glm-5.2`, not `glm-5.2[1m]`. If the suffix slips through,
    // the upstream returns 400.
    assert_eq!(parsed["model"], "glm-5.2");
    // Other fields round-trip.
    assert_eq!(parsed["max_tokens"], 1024);
    assert!(parsed["messages"].is_array());
    proxy.shutdown().await;
}

#[tokio::test]
async fn anthropic_messages_strips_1m_suffix_claude() {
    let upstream = MockServer::start().await;
    let captured = mount_anthropic_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
    })
    .await;

    let payload = json!({
        "model": "claude-3-7-sonnet[1m]",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "hi"}],
    });
    let body = Bytes::from(serde_json::to_vec(&payload).unwrap());
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let got = captured.lock().unwrap().clone().expect("upstream got body");
    let parsed: Value = serde_json::from_slice(&got).expect("upstream body is JSON");
    assert_eq!(parsed["model"], "claude-3-7-sonnet");
    proxy.shutdown().await;
}

#[tokio::test]
async fn anthropic_messages_passthrough_when_no_suffix() {
    // Pin the no-op case end-to-end. The body the upstream receives
    // must be byte-equal to the body the client sent — the
    // cache-safety invariant the OpenAI tests below also assert.
    let upstream = MockServer::start().await;
    let captured = mount_anthropic_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
    })
    .await;

    let payload = json!({
        "model": "claude-3-7-sonnet",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "hi"}],
    });
    let body = Bytes::from(serde_json::to_vec(&payload).unwrap());
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        .header("content-type", "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let got = captured.lock().unwrap().clone().expect("upstream got body");
    assert_byte_equal_sha256(&body, &got);
    proxy.shutdown().await;
}

// ─── OpenAI: /v1/chat/completions must NOT mutate the model ─────────

#[tokio::test]
async fn openai_chat_completions_passthrough_with_1m_model() {
    // Per the review feedback: an OpenAI chat request with a
    // literal model ending in `[1m]` must be left byte-for-byte
    // unchanged unless some other explicit OpenAI mutation applies.
    //
    // Test setup:
    //
    // - `CompressionMode::LiveZone` runs the actual Chat Completions
    //   live-zone dispatcher. For this small body the dispatcher
    //   has nothing to compress, so it returns `NoCompression` and
    //   the body round-trips byte-equal.
    // - The `Authorization: Bearer <jwt>` header classifies the
    //   request as `AuthMode::OAuth` (rule 4: 3 dot-separated
    //   segments), so PR-E4 `prompt_cache_key` injection
    //   short-circuits. Without this header the proxy defaults to
    //   `AuthMode::Payg` and the injector would mutate the body —
    //   the byte-equality assertion would then fail for an
    //   unrelated reason.
    //
    // CRITICAL: if this assertion fails after the fix, the
    // sanitizer is being applied too broadly (the original PR
    // #2027 bug). The OAuth-Authorization is the control variable
    // that isolates the sanitizer's scope from the E4 injector.
    let upstream = MockServer::start().await;
    let captured = mount_chat_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
    })
    .await;

    let payload = json!({
        "model": "gpt-4o[1m]",
        "messages": [{"role": "user", "content": "hello"}],
    });
    let body = Bytes::from(serde_json::to_vec(&payload).unwrap());
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", proxy.url()))
        .header("content-type", "application/json")
        // OAuth bearer token (3 dot-separated segments) → the
        // proxy's auth-mode classifier picks `AuthMode::OAuth`,
        // which gates PR-E4 prompt_cache_key injection to a
        // no-op. This is the same control variable the existing
        // byte-equality tests in `integration_chat_completions.rs`
        // use to isolate dispatcher byte-fidelity from E4.
        .header(
            "authorization",
            "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ0ZXN0In0.signature_bytes",
        )
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let got = captured.lock().unwrap().clone().expect("upstream got body");
    assert_byte_equal_sha256(&body, &got);
    let parsed: Value = serde_json::from_slice(&got).expect("upstream body is JSON");
    assert_eq!(parsed["model"], "gpt-4o[1m]");
    proxy.shutdown().await;
}

// ─── OpenAI: /v1/responses must NOT mutate the model ─────────────────

#[tokio::test]
async fn openai_responses_passthrough_with_1m_model() {
    // Same contract as /v1/chat/completions but for the Responses
    // endpoint. The Responses live-zone dispatcher walks a typed
    // `input` array; for this fixture the input is a single short
    // string, so the dispatcher has nothing to compress and the
    // body round-trips byte-equal. The OAuth Authorization
    // short-circuits the PR-E4 `prompt_cache_key` injector, so
    // the only byte mutation that could fire is the sanitizer
    // itself — which is exactly what we are pinning.
    let upstream = MockServer::start().await;
    let captured = mount_responses_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
    })
    .await;

    let payload = json!({
        "model": "gpt-4o[1m]",
        "input": "hello",
    });
    let body = Bytes::from(serde_json::to_vec(&payload).unwrap());
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

    let got = captured.lock().unwrap().clone().expect("upstream got body");
    assert_byte_equal_sha256(&body, &got);
    let parsed: Value = serde_json::from_slice(&got).expect("upstream body is JSON");
    assert_eq!(parsed["model"], "gpt-4o[1m]");
    proxy.shutdown().await;
}
