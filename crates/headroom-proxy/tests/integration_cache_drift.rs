//! Integration tests for the PR-E6 cache-bust drift detector.
//!
//! Boots a real Rust proxy in front of a wiremock upstream, sends two
//! requests on the same `Authorization` (= same session), and asserts
//! that:
//!
//! 1. A second request with a *different* system prompt produces a
//!    `cache_drift_observed` warn-level event whose `drift_dims`
//!    field includes `system`.
//! 2. The proxy still forwards bytes byte-equal to upstream — the
//!    detector is read-only.
//! 3. The session key is hashed in the log line; the raw bearer token
//!    (`sk-test-this-is-a-secret`) never appears anywhere in the
//!    captured log buffer.

mod common;

use common::start_proxy_with;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::sync::{Arc, Mutex};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// SHA-256 hex of `bytes`. Used to assert byte-faithful passthrough.
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

/// Mount a /v1/messages handler that captures every body that arrives
/// at upstream into the returned `Vec<Vec<u8>>` for later assertions.
async fn mount_anthropic_capture_all(upstream: &MockServer) -> Arc<Mutex<Vec<Vec<u8>>>> {
    let captured: Arc<Mutex<Vec<Vec<u8>>>> = Arc::new(Mutex::new(Vec::new()));
    let captured_clone = captured.clone();
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(move |req: &wiremock::Request| {
            captured_clone.lock().unwrap().push(req.body.clone());
            ResponseTemplate::new(200).set_body_string(r#"{"ok":true}"#)
        })
        .mount(upstream)
        .await;
    captured
}

fn anthropic_payload(system: &str) -> Value {
    json!({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "system": system,
        "messages": [
            {"role": "user", "content": "hello"},
        ],
    })
}

/// The cache-drift integration test installs a global JSON tracing
/// subscriber. Running it in its own `#[test]` (not `#[tokio::test]`)
/// would deadlock the wiremock client; instead we keep it in a
/// dedicated module that owns the OnceLock'd subscriber and is the
/// only async test in this binary.
mod tracing_capture {
    use super::*;
    use std::sync::Arc;
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
            // INFO level so we also catch `cache_drift_first_request`,
            // not just the warn-level `cache_drift_observed`.
            let subscriber = tracing_subscriber::fmt()
                .json()
                .with_writer(writer)
                .with_max_level(tracing::Level::INFO)
                .finish();
            let _ = tracing::subscriber::set_global_default(subscriber);
            buf
        })
    }

    #[tokio::test]
    async fn cache_drift_observed_when_system_prompt_changes_mid_session() {
        let buf = buffer();
        buf.lock().unwrap().clear();

        let upstream = MockServer::start().await;
        let captured = mount_anthropic_capture_all(&upstream).await;
        let proxy = start_proxy_with(&upstream.uri(), |c| {
            // Drift detection runs inside the buffered branch — the
            // master `compression` switch must be ON, which it is in
            // every realistic deployment.
            c.compression = true;
        })
        .await;

        // Same Authorization header → same session_key. Different
        // system prompts on each turn → drift_dims=system on turn 2.
        let secret = "Bearer sk-test-this-is-a-secret";
        let client = reqwest::Client::new();

        let body1 = serde_json::to_vec(&anthropic_payload("you are an expert assistant")).unwrap();
        let r1 = client
            .post(format!("{}/v1/messages", proxy.url()))
            .header("authorization", secret)
            .header("content-type", "application/json")
            .body(body1.clone())
            .send()
            .await
            .unwrap();
        assert_eq!(r1.status(), 200);

        let body2 = serde_json::to_vec(&anthropic_payload("you are now a poet")).unwrap();
        let r2 = client
            .post(format!("{}/v1/messages", proxy.url()))
            .header("authorization", secret)
            .header("content-type", "application/json")
            .body(body2.clone())
            .send()
            .await
            .unwrap();
        assert_eq!(r2.status(), 200);

        // Byte-faithful passthrough: each upstream-received body must
        // SHA-256 match the corresponding inbound body.
        let received = captured.lock().unwrap().clone();
        assert_eq!(received.len(), 2, "upstream should have seen 2 requests");
        assert_eq!(
            sha256_hex(&body1),
            sha256_hex(&received[0]),
            "request 1 byte-faithful passthrough violated",
        );
        assert_eq!(
            sha256_hex(&body2),
            sha256_hex(&received[1]),
            "request 2 byte-faithful passthrough violated",
        );

        // Logs: a `cache_drift_observed` event must be present and
        // include `system` in `drift_dims`.
        let logs = String::from_utf8(buf.lock().unwrap().clone()).expect("logs are utf-8");
        assert!(
            logs.contains(r#""event":"cache_drift_first_request""#),
            "expected first_request event in logs: {logs}",
        );
        assert!(
            logs.contains(r#""event":"cache_drift_observed""#),
            "expected drift_observed event in logs: {logs}",
        );
        // `drift_dims` should include `system` when only the system
        // prompt mutated. Find any `cache_drift_observed` line and
        // assert its `drift_dims` contains `system`.
        let drift_line = logs
            .lines()
            .find(|line| line.contains(r#""event":"cache_drift_observed""#))
            .expect("drift_observed line missing");
        assert!(
            drift_line.contains(r#""drift_dims":"system""#),
            "expected drift_dims=system in drift line: {drift_line}",
        );

        // Privacy invariant: the raw bearer secret must NEVER appear
        // anywhere in the captured logs.
        assert!(
            !logs.contains("sk-test-this-is-a-secret"),
            "raw bearer secret leaked into logs",
        );
        assert!(
            !logs.contains("Bearer sk-test"),
            "raw 'Bearer ...' leaked into logs",
        );

        proxy.shutdown().await;
    }
}
