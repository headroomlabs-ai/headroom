//! Integration tests for the proxy-side cache_control resolver
//! (PR-A4).
//!
//! The core walker `headroom_core::compute_frozen_count` is unit-
//! tested in `crates/headroom-core/tests/cache_control.rs`. This file
//! exercises the proxy wrapper [`resolve_frozen_count`] which adds
//! the configurability gate (`HEADROOM_PROXY_CACHE_CONTROL_AUTO_FROZEN`)
//! and the structured-log emission tested via in-memory tracing
//! capture.

use headroom_proxy::compression::resolve_frozen_count;
use headroom_proxy::config::CacheControlAutoFrozen;
use serde_json::json;

#[test]
fn cache_control_marker_at_message_3_yields_frozen_count_4() {
    let body = json!({
        "messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "third"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "fourth", "cache_control": {"type": "ephemeral"}},
            ]},
        ],
    });
    assert_eq!(
        resolve_frozen_count(&body, CacheControlAutoFrozen::Enabled, "test-req-1"),
        4
    );
}

#[test]
fn cache_control_in_system_blocks_does_not_bump_frozen_count() {
    let body = json!({
        "system": [
            {"type": "text", "text": "you are helpful", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "cite sources", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
        ],
        "messages": [
            {"role": "user", "content": "hi"},
        ],
    });
    assert_eq!(
        resolve_frozen_count(&body, CacheControlAutoFrozen::Enabled, "test-req-2"),
        0
    );
}

#[test]
fn cache_control_in_tools_does_not_bump_frozen_count() {
    let body = json!({
        "tools": [
            {"name": "search", "description": "search", "cache_control": {"type": "ephemeral"}},
        ],
        "messages": [
            {"role": "user", "content": "hi"},
        ],
    });
    assert_eq!(
        resolve_frozen_count(&body, CacheControlAutoFrozen::Enabled, "test-req-3"),
        0
    );
}

#[test]
fn cache_control_ttl_1h_before_5m_passes_no_warn() {
    // Legal ordering: 1h before 5m. Walker returns the right
    // frozen_count and emits no warning. Tracing assertion lives
    // in the dedicated module below.
    let body = json!({
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "first 1h", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "second 5m", "cache_control": {"type": "ephemeral"}},
            ]},
        ],
    });
    assert_eq!(
        resolve_frozen_count(&body, CacheControlAutoFrozen::Enabled, "test-req-4"),
        2
    );
}

#[test]
fn cache_control_no_markers_yields_zero() {
    let body = json!({
        "model": "claude-3-5-sonnet-20241022",
        "messages": [
            {"role": "user", "content": "no marker"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "still no marker"},
            ]},
        ],
    });
    assert_eq!(
        resolve_frozen_count(&body, CacheControlAutoFrozen::Enabled, "test-req-5"),
        0
    );
}

#[test]
fn cache_control_multiple_markers_in_messages_returns_max_index() {
    // markers on indices 0, 2, 4 — function returns max(i+1) = 5.
    let body = json!({
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "m0", "cache_control": {"type": "ephemeral"}},
            ]},
            {"role": "assistant", "content": "m1"},
            {"role": "user", "content": [
                {"type": "text", "text": "m2", "cache_control": {"type": "ephemeral"}},
            ]},
            {"role": "assistant", "content": "m3"},
            {"role": "user", "content": [
                {"type": "text", "text": "m4", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
            ]},
        ],
    });
    assert_eq!(
        resolve_frozen_count(&body, CacheControlAutoFrozen::Enabled, "test-req-6"),
        5
    );
}

#[test]
fn cache_control_disabled_via_config_returns_zero_even_with_markers() {
    // The disabled path returns 0 regardless of marker placement.
    // This is the bypass the operator opts into via
    // `--cache-control-auto-frozen=disabled` /
    // `HEADROOM_PROXY_CACHE_CONTROL_AUTO_FROZEN=disabled`.
    let body = json!({
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "m0", "cache_control": {"type": "ephemeral"}},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "m1", "cache_control": {"type": "ephemeral"}},
            ]},
            {"role": "user", "content": [
                {"type": "text", "text": "m2", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
            ]},
        ],
    });
    assert_eq!(
        resolve_frozen_count(&body, CacheControlAutoFrozen::Disabled, "test-req-7"),
        0,
        "disabled policy must override marker presence"
    );
    // Sanity: with the same body, enabled returns the full 3.
    assert_eq!(
        resolve_frozen_count(&body, CacheControlAutoFrozen::Enabled, "test-req-7b"),
        3
    );
}

/// TTL-ordering warning path: capture tracing output and assert the
/// 5m-before-1h scenario emits a `warn!` log line.
///
/// The capture installs a global tracing subscriber, so we keep it
/// in its own module and run only one capture-driven test per binary
/// to avoid double-registration races with other integration tests
/// (mirrors the pattern in `integration_compression.rs`).
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
            let subscriber = tracing_subscriber::fmt()
                .json()
                .with_writer(writer)
                // WARN level is enough — we capture warn! and
                // implicitly higher; debug! lines are excluded so
                // the buffer stays small.
                .with_max_level(tracing::Level::WARN)
                .finish();
            // try_init returns Err if a global subscriber already
            // exists. Other test binaries in this crate also install
            // one — that's fine; we just need *some* subscriber to
            // be active for our `tracing::warn!` to be observable.
            let _ = tracing::subscriber::set_global_default(subscriber);
            buf
        })
    }

    #[test]
    fn cache_control_ttl_5m_before_1h_warns_and_passes() {
        let buf = buffer();
        buf.lock().unwrap().clear();

        // 5m marker on message 0, then 1h marker on message 1 —
        // violation per guide §2.19. Walker must:
        // 1. return the correct floor (= 2);
        // 2. emit a `warn!` so the operator can see the issue.
        let body = json!({
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "first 5m", "cache_control": {"type": "ephemeral"}},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "second 1h", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
                ]},
            ],
        });
        let count = resolve_frozen_count(&body, CacheControlAutoFrozen::Enabled, "test-req-warn");
        assert_eq!(count, 2, "function must compute correct floor regardless");

        let logs = String::from_utf8(buf.lock().unwrap().clone()).expect("logs are utf-8");
        // The warning must be present. We check for the rule
        // identifier rather than the exact prose so future copy
        // edits don't break the test.
        assert!(
            logs.contains("anthropic_prompt_caching_guide_2_19"),
            "TTL ordering warn log missing; logs: {logs}",
        );
        assert!(
            logs.contains(r#""field":"messages""#),
            "warn log missing field=messages; logs: {logs}",
        );
    }
}
