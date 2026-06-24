//! Integration tests for PR-E5 volatile-content detector.
//!
//! Boots a real Rust proxy in front of a wiremock upstream. Sends a
//! request whose system prompt embeds an ISO-8601 timestamp, then
//! asserts that:
//!
//!   1. A structured `volatile_content_detected` WARN log was
//!      emitted (captured via a `tracing_subscriber` JSON layer
//!      with an in-memory `MakeWriter`).
//!   2. The bytes that arrived at the upstream are byte-equal to
//!      the bytes the client sent — the detector observes only,
//!      it never mutates.
//!
//! Mirrors the capture pattern from `integration_compression.rs`
//! and `integration_cache_control.rs`: install a JSON subscriber
//! once via `OnceLock`, run only one capture-driven test per
//! binary so we don't fight other tests for the global default.

mod common;

use common::start_proxy_with;
use serde_json::json;
use std::sync::{Arc, Mutex};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Mount a /v1/messages handler that captures the upstream request body.
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
                // WARN gives us volatile_content_detected without
                // flooding the buffer with INFO/DEBUG noise.
                .with_max_level(tracing::Level::WARN)
                .finish();
            // Best-effort install: tests in other binaries may have
            // already set a default subscriber. We only need *some*
            // subscriber active for our `tracing::warn!` to surface
            // to stdout (where this writer captures).
            let _ = tracing::subscriber::set_global_default(subscriber);
            buf
        })
    }

    #[tokio::test]
    async fn volatile_timestamp_in_system_emits_warn_and_passes_through() {
        let buf = buffer();
        buf.lock().unwrap().clear();

        let upstream = MockServer::start().await;
        let captured = mount_anthropic_capture(&upstream).await;
        let proxy = start_proxy_with(&upstream.uri(), |c| {
            c.compression = true;
            c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
            c.log_level = "warn".into();
        })
        .await;

        // System prompt embeds an ISO-8601 timestamp — exactly the
        // pattern that busts prompt cache hits.
        let payload = json!({
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 32,
            "system": "You are a helpful assistant. Today is 2026-05-04T14:30:00Z.",
            "messages": [{"role": "user", "content": "hi"}],
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
        assert!(
            logs.contains("volatile_content_detected"),
            "expected volatile_content_detected event in logs; got: {logs}",
        );
        assert!(
            logs.contains("iso8601_timestamp"),
            "expected kind=iso8601_timestamp in logs; got: {logs}",
        );
        assert!(
            logs.contains(r#""location":"system""#),
            "expected location=system in logs; got: {logs}",
        );

        // Non-mutation invariant: the upstream-received body is
        // byte-equal to the client-sent body.
        let upstream_received = captured
            .lock()
            .unwrap()
            .clone()
            .expect("upstream should have captured a body");
        assert_eq!(
            upstream_received, body,
            "volatile detector must not mutate the request body",
        );

        proxy.shutdown().await;
    }
}
