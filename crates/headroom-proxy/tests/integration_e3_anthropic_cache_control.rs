//! Integration tests for PR-E3 Anthropic cache_control auto-placement.
//!
//! Boots a real Rust proxy in front of a wiremock upstream and
//! exercises the three branches of the safety contract:
//!
//! 1. **PAYG body without markers** — proxy auto-places a marker on
//!    the last tool, upstream receives the modified body, every
//!    other byte is preserved.
//! 2. **PAYG body with a customer-placed marker** — proxy passes
//!    through byte-equal (SHA-256 match) and emits an
//!    `e3_skipped` event with `reason = "marker_present"`.
//! 3. **Non-PAYG body** (OAuth bearer here) — proxy passes through
//!    byte-equal and emits an `e3_skipped` event with `reason =
//!    "auth_mode"`.
//!
//! The tracing assertions are scoped to a single test in this binary
//! so we don't fight the global subscriber with other tests in the
//! same crate (mirrors the pattern in `integration_volatile_detector.rs`).

mod common;

use common::start_proxy_with;
use sha2::{Digest, Sha256};
use std::sync::{Arc, Mutex};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Mount a `/v1/messages` handler that captures the upstream-received
/// request body. Returns a shared handle the test can read after the
/// request lands.
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

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut s = String::with_capacity(digest.len() * 2);
    for b in digest {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

#[tokio::test]
async fn payg_body_without_markers_gets_marker_on_last_tool() {
    let upstream = MockServer::start().await;
    let captured = mount_anthropic_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
    })
    .await;

    // Three tools, one short user message, plain-string system. No
    // markers anywhere. Use a PAYG-shaped header (`x-api-key`).
    let payload = serde_json::json!({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 32,
        "system": "You are helpful.",
        "tools": [
            {"name": "alpha", "description": "alpha tool"},
            {"name": "beta", "description": "beta tool"},
            {"name": "gamma", "description": "gamma tool"}
        ],
        "messages": [{"role": "user", "content": "hi"}],
    });
    let body = serde_json::to_vec(&payload).unwrap();

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        .header("content-type", "application/json")
        .header("x-api-key", "sk-ant-api01-fake-key")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let upstream_received = captured
        .lock()
        .unwrap()
        .clone()
        .expect("upstream captured a body");
    let upstream_parsed: serde_json::Value =
        serde_json::from_slice(&upstream_received).expect("upstream body is JSON");

    // Marker landed on the LAST tool only.
    assert_eq!(
        upstream_parsed.pointer("/tools/2/cache_control"),
        Some(&serde_json::json!({"type": "ephemeral"})),
        "expected cache_control on last tool; upstream body: {upstream_parsed}",
    );
    assert!(
        upstream_parsed.pointer("/tools/0/cache_control").is_none(),
        "tools[0] must NOT have cache_control",
    );
    assert!(
        upstream_parsed.pointer("/tools/1/cache_control").is_none(),
        "tools[1] must NOT have cache_control",
    );

    // Every other field is preserved exactly.
    assert_eq!(
        upstream_parsed.get("model"),
        payload.get("model"),
        "model field preserved",
    );
    assert_eq!(
        upstream_parsed.get("system"),
        payload.get("system"),
        "system field preserved",
    );
    assert_eq!(
        upstream_parsed.get("messages"),
        payload.get("messages"),
        "messages field preserved",
    );
    assert_eq!(
        upstream_parsed.pointer("/tools/0/name"),
        payload.pointer("/tools/0/name"),
    );
    assert_eq!(
        upstream_parsed.pointer("/tools/2/name"),
        payload.pointer("/tools/2/name"),
    );

    proxy.shutdown().await;
}

#[tokio::test]
async fn payg_body_with_existing_marker_passes_through_byte_equal() {
    let upstream = MockServer::start().await;
    let captured = mount_anthropic_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
    })
    .await;

    // PAYG-shaped headers, but the body already carries a customer
    // cache_control marker on tools[0]. Customer-placement-wins
    // gate fires → proxy passes through byte-equal.
    let payload = serde_json::json!({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 32,
        "tools": [
            {
                "name": "alpha",
                "description": "alpha tool",
                "cache_control": {"type": "ephemeral"}
            },
            {"name": "beta", "description": "beta tool"}
        ],
        "messages": [{"role": "user", "content": "hi"}],
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let body_sha = sha256_hex(&body);

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        .header("content-type", "application/json")
        .header("x-api-key", "sk-ant-api01-fake-key")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let upstream_received = captured
        .lock()
        .unwrap()
        .clone()
        .expect("upstream captured a body");
    let upstream_sha = sha256_hex(&upstream_received);
    assert_eq!(
        upstream_sha, body_sha,
        "byte-equal passthrough required when customer marker present; \
         body sha {body_sha}, upstream sha {upstream_sha}",
    );

    proxy.shutdown().await;
}

#[tokio::test]
async fn oauth_body_passes_through_byte_equal() {
    let upstream = MockServer::start().await;
    let captured = mount_anthropic_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
    })
    .await;

    // OAuth-shaped header (`Bearer sk-ant-oat-...`). F1 classifies as
    // OAuth → E3 must skip mutation entirely. The body has no
    // markers, so the customer-placement-wins gate is irrelevant —
    // the auth-mode gate is the load-bearing one for this test.
    let payload = serde_json::json!({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 32,
        "tools": [{"name": "alpha", "description": "alpha tool"}],
        "messages": [{"role": "user", "content": "hi"}],
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let body_sha = sha256_hex(&body);

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        .header("content-type", "application/json")
        .header("authorization", "Bearer sk-ant-oat-fake-oauth-token")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let upstream_received = captured
        .lock()
        .unwrap()
        .clone()
        .expect("upstream captured a body");
    let upstream_sha = sha256_hex(&upstream_received);
    assert_eq!(
        upstream_sha, body_sha,
        "byte-equal passthrough required for non-PAYG (OAuth) requests; \
         body sha {body_sha}, upstream sha {upstream_sha}",
    );

    // Sanity: upstream did NOT receive a marker.
    let upstream_parsed: serde_json::Value =
        serde_json::from_slice(&upstream_received).expect("upstream body is JSON");
    assert!(
        upstream_parsed.pointer("/tools/0/cache_control").is_none(),
        "OAuth path must not insert cache_control; got {upstream_parsed}",
    );

    proxy.shutdown().await;
}

#[tokio::test]
async fn subscription_body_passes_through_byte_equal() {
    let upstream = MockServer::start().await;
    let captured = mount_anthropic_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
    })
    .await;

    // Subscription-shaped headers: a Claude Code-like `User-Agent`
    // is enough for F1 to classify as Subscription. E3 must skip
    // mutation.
    let payload = serde_json::json!({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 32,
        "tools": [{"name": "alpha", "description": "alpha tool"}],
        "messages": [{"role": "user", "content": "hi"}],
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let body_sha = sha256_hex(&body);

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        .header("content-type", "application/json")
        .header("user-agent", "claude-code/1.0.0")
        .header("x-api-key", "sk-ant-api01-fake-key")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let upstream_received = captured
        .lock()
        .unwrap()
        .clone()
        .expect("upstream captured a body");
    let upstream_sha = sha256_hex(&upstream_received);
    assert_eq!(
        upstream_sha, body_sha,
        "byte-equal passthrough required for Subscription requests; \
         body sha {body_sha}, upstream sha {upstream_sha}",
    );

    proxy.shutdown().await;
}

/// Tracing-capture test: confirm the `e3_applied` event fires with
/// the expected fields when we auto-place a marker. Lives in its own
/// module to scope the global subscriber installation.
mod tracing_capture {
    use super::*;
    use std::sync::Mutex as StdMutex;
    use std::sync::OnceLock;
    use tracing_subscriber::fmt::MakeWriter;

    #[derive(Clone)]
    struct CaptureWriter {
        inner: Arc<StdMutex<Vec<u8>>>,
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

    fn buffer() -> &'static Arc<StdMutex<Vec<u8>>> {
        static BUFFER: OnceLock<Arc<StdMutex<Vec<u8>>>> = OnceLock::new();
        BUFFER.get_or_init(|| {
            let buf = Arc::new(StdMutex::new(Vec::new()));
            let writer = CaptureWriter { inner: buf.clone() };
            let subscriber = tracing_subscriber::fmt()
                .json()
                .with_writer(writer)
                // INFO is required — `e3_applied` is logged at INFO.
                .with_max_level(tracing::Level::INFO)
                .finish();
            // Best-effort install: tests in other binaries may have
            // already set a default subscriber. We only need *some*
            // subscriber active for our `tracing::info!` to surface.
            let _ = tracing::subscriber::set_global_default(subscriber);
            buf
        })
    }

    #[tokio::test]
    async fn payg_apply_emits_e3_applied_event() {
        let buf = buffer();
        buf.lock().unwrap().clear();

        let upstream = MockServer::start().await;
        let _captured = mount_anthropic_capture(&upstream).await;
        let proxy = start_proxy_with(&upstream.uri(), |c| {
            c.compression = true;
            c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
            c.log_level = "info".into();
        })
        .await;

        let payload = serde_json::json!({
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 32,
            "tools": [{"name": "alpha", "description": "alpha tool"}],
            "messages": [{"role": "user", "content": "hi"}],
        });
        let body = serde_json::to_vec(&payload).unwrap();
        let resp = reqwest::Client::new()
            .post(format!("{}/v1/messages", proxy.url()))
            .header("content-type", "application/json")
            .header("x-api-key", "sk-ant-api01-fake-key")
            .body(body)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);

        // Let the async tracing emitter flush.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let logs = String::from_utf8(buf.lock().unwrap().clone()).expect("logs are utf-8");
        assert!(
            logs.contains("e3_applied"),
            "expected e3_applied event in logs; got: {logs}",
        );
        assert!(
            logs.contains(r#""placed_count":1"#),
            "expected placed_count=1 in logs; got: {logs}",
        );
        assert!(
            logs.contains("tools[0]"),
            "expected location tools[0] in logs; got: {logs}",
        );

        proxy.shutdown().await;
    }
}
