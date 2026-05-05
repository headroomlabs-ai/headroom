//! End-to-end integration tests for the compression interceptor.
//!
//! These tests boot a real Rust proxy in front of a wiremock upstream
//! and verify the request body that arrives at the upstream — i.e. we
//! observe the *actual* compression effect on the wire, not the
//! library outcome in isolation.
//!
//! # PR-A1 — Phase A lockdown
//!
//! Per `REALIGNMENT/03-phase-A-lockdown.md`, the `/v1/messages`
//! endpoint is now a byte-faithful passthrough. The cache-safety
//! invariant is asserted via SHA-256 byte equality between the
//! bytes the client sent and the bytes the upstream received. JSON
//! value-equality is not a sound substitute: it misses whitespace,
//! key order, and Unicode escape differences that all bust prompt
//! cache hit rate.
//!
//! Coverage:
//!
//! - `compression_off_passes_body_unchanged` — master switch off.
//! - `compression_on_short_body_passes_through` — small JSON; SHA-256
//!   byte equality (was: `len()` equality; tightened in PR-A1).
//! - `compression_on_long_body_passes_through_in_phase_a` — the
//!   formerly-oversized fixture now passes through unchanged. Old
//!   assertion ("fewer messages arrived") flipped to "same messages
//!   arrived, byte-equal" — documenting that compression is
//!   intentionally off in Phase A.
//! - `compression_on_non_json_skips` — content-type gate.
//! - `compression_on_non_llm_path_skips` — path gate.
//!
//! New PR-A1 tests:
//!
//! - `passthrough_mode_off_byte_equal_sha256` — pure passthrough
//!   over a 4KB mixed-encoding body.
//! - `passthrough_mode_live_zone_currently_passthrough_byte_equal_sha256`
//!   — `live_zone` is reserved for Phase B; in Phase A it warns and
//!   passes through.
//! - `passthrough_preserves_numeric_precision` — `temperature: 1.0`,
//!   `seed: 12345678901234567`, scientific-notation numbers.
//! - `passthrough_preserves_cache_control_markers` — markers in
//!   messages and tools.
//! - `passthrough_preserves_thinking_signature` — assistant
//!   thinking block + signature.
//! - `passthrough_preserves_redacted_thinking_data` — redacted
//!   thinking data field.
//! - `passthrough_recorded_fixture_byte_equal_sha256` — the recorded
//!   production-shaped fixture.

mod common;

use common::start_proxy_with;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::sync::{Arc, Mutex};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Mount a /v1/messages handler that captures the upstream request body
/// into the returned Arc<Mutex<...>> for assertions, and returns 200 OK.
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

/// Compute the lowercase hex SHA-256 of a byte slice. Used to gate
/// "the proxy did not perturb the request body" — the only sound way
/// to assert byte-faithfulness.
fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    digest.iter().fold(String::with_capacity(64), |mut acc, b| {
        use std::fmt::Write as _;
        let _ = write!(acc, "{b:02x}");
        acc
    })
}

/// Assert that the bytes the upstream received are byte-equal to the
/// bytes the client sent. Compares both length and SHA-256 so failure
/// messages distinguish length mismatches (likely Content-Length
/// re-encoded) from same-length-but-different-bytes (likely whitespace
/// or escape mutations).
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

/// Build a payload that's large enough to have forced ICM to trim
/// under the old behaviour. PR-A1: it now passes through unchanged.
fn oversized_anthropic_payload() -> Value {
    let messages: Vec<Value> = (0..30)
        .map(|i| {
            json!({
                "role": if i % 2 == 0 { "user" } else { "assistant" },
                "content": format!("padding token {i} ").repeat(20),
            })
        })
        .collect();
    json!({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 199_500,
        "messages": messages,
    })
}

#[tokio::test]
async fn compression_off_passes_body_unchanged() {
    // Master switch off. Body must arrive byte-equal at upstream.
    let upstream = MockServer::start().await;
    let captured = mount_anthropic_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |_| {
        // compression remains off (Config::for_test default)
    })
    .await;

    let payload = oversized_anthropic_payload();
    let body = serde_json::to_vec(&payload).unwrap();
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

#[tokio::test]
async fn compression_on_short_body_passes_through() {
    // PR-A1 tightening: was `assert_eq!(len, len)`; now SHA-256
    // byte equality. Small body so we exercise the buffered branch
    // even though no compression occurs.
    let upstream = MockServer::start().await;
    let captured = mount_anthropic_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
    })
    .await;

    let payload = json!({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "hello"}],
    });
    let body = serde_json::to_vec(&payload).unwrap();
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

#[tokio::test]
async fn compression_on_long_body_passes_through_in_phase_a() {
    // PR-A1 rename + flip. Was
    // `compression_on_oversized_body_trims_messages` with the
    // assertion "fewer messages arrived". Now: even though the body
    // is oversized, Phase A passthrough means same messages arrive
    // byte-equal — documenting that compression is intentionally
    // off until Phase B.
    let upstream = MockServer::start().await;
    let captured = mount_anthropic_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
    })
    .await;

    let payload = oversized_anthropic_payload();
    let body = serde_json::to_vec(&payload).unwrap();
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

#[tokio::test]
async fn compression_on_non_json_skips() {
    let upstream = MockServer::start().await;
    let captured = mount_anthropic_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
    })
    .await;

    // Path matches /v1/messages but Content-Type isn't JSON. The gate
    // must skip and stream verbatim.
    let body = vec![0xAAu8; 64 * 1024];
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        .header("content-type", "application/octet-stream")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let got = captured.lock().unwrap().clone().expect("upstream got body");
    assert_byte_equal_sha256(&body, &got);
    proxy.shutdown().await;
}

#[tokio::test]
async fn compression_on_non_llm_path_skips() {
    let upstream = MockServer::start().await;
    let captured: Arc<Mutex<Option<Vec<u8>>>> = Arc::new(Mutex::new(None));
    let captured_clone = captured.clone();
    Mock::given(method("POST"))
        .and(path("/some/other/api"))
        .respond_with(move |req: &wiremock::Request| {
            *captured_clone.lock().unwrap() = Some(req.body.clone());
            ResponseTemplate::new(200).set_body_string("ok")
        })
        .mount(&upstream)
        .await;

    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
    })
    .await;

    // Same oversized JSON payload, but at a non-LLM path.
    let payload = oversized_anthropic_payload();
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/some/other/api", proxy.url()))
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

// ─── PR-A1 new tests ──────────────────────────────────────────────────

#[tokio::test]
async fn passthrough_mode_off_byte_equal_sha256() {
    // Pure passthrough; 4KB body with mixed ASCII + non-ASCII
    // (emoji, Japanese) + nested JSON.
    let upstream = MockServer::start().await;
    let captured = mount_anthropic_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::Off;
    })
    .await;

    // Build a body that exercises Unicode escapes and nested JSON.
    let mut content = String::with_capacity(4096);
    content.push_str("ASCII prefix; ");
    while content.len() < 4096 {
        content.push_str("hello 🔥 日本語 — ");
    }
    let payload = json!({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [
                {"type": "text", "text": "nested 💎"},
                {"type": "tool_use", "id": "tu_01", "name": "search", "input": {"q": "🔍"}}
            ]}
        ]
    });
    let body = serde_json::to_vec(&payload).unwrap();
    assert!(body.len() >= 4096, "test body must exercise large path");

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

#[tokio::test]
async fn passthrough_mode_live_zone_currently_passthrough_byte_equal_sha256() {
    // PR-B2: live-zone dispatcher is wired but every per-type
    // compressor is still a no-op skeleton, so the proxy forwards
    // the buffered body byte-equal. This test pins the cache-safety
    // invariant for the live-zone path through the B2 → B3 → B4 →
    // B7 transitions: no-op compressors must never mutate bytes.
    // PR-B3+ replaces this guarantee with the per-type compressor
    // contract (compress only the live zone; bytes outside the
    // live zone byte-equal).
    let upstream = MockServer::start().await;
    let captured = mount_anthropic_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
    })
    .await;

    let payload = oversized_anthropic_payload();
    let body = serde_json::to_vec(&payload).unwrap();
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

#[tokio::test]
async fn passthrough_preserves_numeric_precision() {
    // Numeric precision is the most fragile property under
    // round-trip JSON parsing: f64 can't faithfully hold u64 above
    // 2^53. PR-A1's whole point is that we don't parse, so this
    // must come through bit-for-bit.
    let upstream = MockServer::start().await;
    let captured = mount_anthropic_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
    })
    .await;

    // We can't trust serde_json to emit `1.0` (it emits `1`) or
    // preserve `12345678901234567` exactly through a Value round-
    // trip on default features. Build the body from a literal byte
    // string so we control every digit.
    let body = br#"{
  "model": "claude-3-5-sonnet-20241022",
  "max_tokens": 1024,
  "temperature": 1.0,
  "top_p": 0.95,
  "top_k": 50,
  "seed": 12345678901234567,
  "tiny": 1e-9,
  "huge": 2.5e10,
  "neg": -3.14159265358979,
  "messages": [{"role": "user", "content": "ping"}]
}"#
    .to_vec();

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

#[tokio::test]
async fn passthrough_preserves_cache_control_markers() {
    // cache_control markers are the linchpin of Anthropic prompt
    // caching. If the proxy reorders, drops, or re-emits any of
    // them, the customer's cache hit rate craters. Phase A
    // passthrough must preserve them byte-equal.
    let upstream = MockServer::start().await;
    let captured = mount_anthropic_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
    })
    .await;

    // Built from literal bytes so test-author intent (key order,
    // ttl string casing) is the assertion.
    let body = br#"{"model":"claude-3-5-sonnet-20241022","max_tokens":1024,"system":[{"type":"text","text":"You are helpful.","cache_control":{"type":"ephemeral"}},{"type":"text","text":"Cite sources.","cache_control":{"type":"ephemeral","ttl":"1h"}}],"tools":[{"name":"s","description":"search","input_schema":{"type":"object","properties":{"q":{"type":"string"}},"required":["q"]},"cache_control":{"type":"ephemeral"}}],"messages":[{"role":"user","content":[{"type":"text","text":"hi","cache_control":{"type":"ephemeral"}}]}]}"#
        .to_vec();

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
    // Belt-and-suspenders: confirm the markers are still in the
    // upstream-received bytes verbatim. SHA-256 already proves it,
    // but a substring assertion gives a more readable failure
    // message if a future regression introduces a mutation.
    let got_str = std::str::from_utf8(&got).expect("body is utf-8");
    assert!(got_str.contains(r#""cache_control":{"type":"ephemeral"}"#));
    assert!(got_str.contains(r#""cache_control":{"type":"ephemeral","ttl":"1h"}"#));
    proxy.shutdown().await;
}

#[tokio::test]
async fn passthrough_preserves_thinking_signature() {
    // Thinking blocks with `signature` fields are sacrosanct per
    // the cache-safety invariants (§2.7, §10.1). They must arrive
    // at upstream byte-equal — any whitespace, key-order, or
    // base64 normalization breaks Anthropic's signature check.
    let upstream = MockServer::start().await;
    let captured = mount_anthropic_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
    })
    .await;

    let body = br#"{"model":"claude-3-5-sonnet-20241022","max_tokens":1024,"messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"reasoning here","signature":"ErcBCkgIBhABGAIiQO5fJk0wY2J3aDQ4ckZmZE5Ld2lDV3VYV1JlVlVQQUtpa3lXQVdqREZSc1Y3WkRSWjJsdndPbVlEY1ZNUUUSDDNjMjUwYWY5LWFlMmU="},{"type":"text","text":"answer"}]},{"role":"user","content":"continue"}]}"#
        .to_vec();

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
    let got_str = std::str::from_utf8(&got).expect("body is utf-8");
    assert!(got_str.contains(r#""signature":"ErcBCkgIBhAB"#));
    proxy.shutdown().await;
}

#[tokio::test]
async fn passthrough_preserves_redacted_thinking_data() {
    // `redacted_thinking.data` is opaque to us — Anthropic encodes
    // its own state there. Modifying it would invalidate the next
    // turn's reasoning continuation.
    let upstream = MockServer::start().await;
    let captured = mount_anthropic_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
    })
    .await;

    let body = br#"{"model":"claude-3-5-sonnet-20241022","max_tokens":1024,"messages":[{"role":"assistant","content":[{"type":"redacted_thinking","data":"EsADCkYIBxABGAIiQGtHMHA0QzlpbXJyV2I4QmtuS1JmTjFvUHFwS1NXa1d3Z3FVSlJSc3JKWmhLbDF3WmZmZjJyVTFqUlRYZ0FzSE0="}]},{"role":"user","content":"continue"}]}"#
        .to_vec();

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
    let got_str = std::str::from_utf8(&got).expect("body is utf-8");
    assert!(got_str.contains(r#""redacted_thinking""#));
    assert!(got_str.contains(r#""data":"EsADCkYIBxAB"#));
    proxy.shutdown().await;
}

#[tokio::test]
async fn passthrough_recorded_fixture_byte_equal_sha256() {
    // The "real-shape" fixture: system as block list with
    // cache_control, tools with non-trivial JSON Schema (nested
    // properties + definitions), messages with text + thinking +
    // signature + tool_use + tool_result + image, non-ASCII content,
    // large numbers, cache_control markers in messages and tools.
    //
    // This is the canonical SHA-256 byte-equality test — any future
    // regression in the proxy's body handling fails here first.
    let upstream = MockServer::start().await;
    let captured = mount_anthropic_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
    })
    .await;

    let body = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/anthropic_messages_request_real.json"
    ))
    .expect("fixture present in repo");

    // Sanity: the fixture should parse as JSON. (We never parse it
    // through the proxy — passthrough is byte-faithful — but we
    // want a clear test failure if someone corrupts the file.)
    let _: Value = serde_json::from_slice(&body).expect("fixture parses as json");

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

/// Tracing-capture test for the per-request decision log.
///
/// Lives in its own module rather than at file scope because it
/// installs a *global* tracing subscriber via
/// `tracing::subscriber::set_global_default` — we only do this once
/// per test process and isolate it to a single test to avoid
/// double-registration races with other tests in the same binary.
mod tracing_capture {
    use super::*;
    use std::sync::Mutex as StdMutex;
    use std::sync::OnceLock;
    use tracing_subscriber::fmt::MakeWriter;

    /// In-memory writer that accumulates tracing output. Used by
    /// `make_writer` so each emitted log line gets pushed into the
    /// shared buffer for later assertion.
    #[derive(Clone)]
    struct CaptureWriter {
        inner: Arc<StdMutex<Vec<u8>>>,
    }

    impl CaptureWriter {
        fn new(inner: Arc<StdMutex<Vec<u8>>>) -> Self {
            Self { inner }
        }
    }

    impl std::io::Write for CaptureWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.inner.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> MakeWriter<'a> for CaptureWriter {
        type Writer = Self;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    /// Lazily install the JSON tracing subscriber once per test
    /// process. The buffer is shared across the whole process, but
    /// because we only run one tracing-capture test per binary, we
    /// don't have to worry about cross-test interference.
    fn buffer() -> &'static Arc<StdMutex<Vec<u8>>> {
        static BUFFER: OnceLock<Arc<StdMutex<Vec<u8>>>> = OnceLock::new();
        BUFFER.get_or_init(|| {
            let buf = Arc::new(StdMutex::new(Vec::new()));
            let writer = CaptureWriter::new(buf.clone());
            let subscriber = tracing_subscriber::fmt()
                .json()
                .with_writer(writer)
                .with_max_level(tracing::Level::INFO)
                .finish();
            // try_init returns Err if a global subscriber was already
            // installed. We don't care: as long as *something* is
            // collecting, the test will fail with a clear message.
            let _ = tracing::subscriber::set_global_default(subscriber);
            buf
        })
    }

    #[tokio::test]
    async fn compression_decision_logged() {
        let buf = buffer();
        // Reset for this test; harmless if other tests ran first.
        buf.lock().unwrap().clear();

        let upstream = MockServer::start().await;
        let _captured = mount_anthropic_capture(&upstream).await;
        let proxy = start_proxy_with(&upstream.uri(), |c| {
            c.compression = true;
            c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
            c.log_level = "info".into();
        })
        .await;

        let payload = json!({
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "log me"}],
        });
        let body = serde_json::to_vec(&payload).unwrap();
        let resp = reqwest::Client::new()
            .post(format!("{}/v1/messages", proxy.url()))
            .header("content-type", "application/json")
            .body(body.clone())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);

        // Give the async tracing emitter a beat to flush.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let logs = String::from_utf8(buf.lock().unwrap().clone()).expect("logs are utf-8");

        // PR-B3: live-zone dispatcher logs `decision="no_change"`
        // with `reason="no_block_compressed"` when the live zone
        // had no compressible blocks (or every compressor declined
        // / produced larger output). The `decision="compressed"`
        // path is exercised by
        // `crates/headroom-core/tests/live_zone_dispatch.rs`.
        assert!(
            logs.contains(r#""decision":"no_change""#),
            "decision field missing or wrong; logs: {logs}",
        );
        assert!(
            logs.contains(r#""reason":"no_block_compressed""#),
            "reason field missing or wrong; logs: {logs}",
        );
        assert!(
            logs.contains(r#""compression_mode":"live_zone""#),
            "compression_mode field missing or wrong; logs: {logs}",
        );
        assert!(
            logs.contains(r#""body_bytes":"#),
            "body_bytes field missing; logs: {logs}",
        );
        // The dispatcher exposes the manifest contract (frozen
        // floor + messages_total + live_zone block counts) on
        // every log line so operators can see why a request did
        // or didn't compress without enabling debug logging.
        assert!(
            logs.contains(r#""frozen_message_count":"#),
            "frozen_message_count field missing; logs: {logs}",
        );
        assert!(
            logs.contains(r#""messages_total":"#),
            "messages_total field missing; logs: {logs}",
        );
        assert!(
            logs.contains(r#""live_zone_blocks":"#),
            "live_zone_blocks field missing; logs: {logs}",
        );
        // The "reserved for Phase B" warning that PR-A1 emitted
        // is intentionally gone post-PR-B2. Lock it out so a
        // bad cherry-pick can't reintroduce a stale warning.
        assert!(
            !logs.contains("compression mode 'live_zone' is reserved for Phase B"),
            "obsolete Phase A warning leaked into Phase B logs: {logs}",
        );
        // Sanity: we never log the Authorization header.
        assert!(
            !logs.to_ascii_lowercase().contains("authorization:"),
            "logs unexpectedly contain Authorization header content: {logs}",
        );

        proxy.shutdown().await;
    }
}
