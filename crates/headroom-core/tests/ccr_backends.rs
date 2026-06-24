//! Integration tests for the persistent CCR backends (PR-B7).
//!
//! Covers SQLite round-trip + TTL purge + restart-survival, the cross-
//! backend byte-equal-key invariant, and (cfg-gated) the Redis backend.

use std::time::Duration;

use headroom_core::ccr::backends::{
    from_config, CcrBackendConfig, InMemoryCcrStore, SqliteCcrStore,
};
use headroom_core::ccr::{compute_key, CcrStore};

#[test]
fn sqlite_round_trip() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("ccr.sqlite");
    let store = SqliteCcrStore::open(&path, 300).expect("open sqlite store");
    let payload = r#"[{"id":1},{"id":2},{"id":3}]"#;
    let hash = compute_key(payload.as_bytes());
    store.put(&hash, payload);
    let fetched = store.get(&hash);
    assert_eq!(fetched.as_deref(), Some(payload));
    assert_eq!(store.len(), 1);
    // Missing key returns None.
    assert_eq!(store.get("missing-hash-key"), None);
}

#[test]
fn sqlite_ttl_purge() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("ccr.sqlite");
    // 0-second TTL forces every entry to be expired the moment we read it.
    let store = SqliteCcrStore::open(&path, 0).expect("open sqlite store");
    let hash = compute_key(b"to be purged");
    store.put(&hash, "to be purged");
    // Sleep long enough for `created_at + ttl_seconds <= now()` (1s clock
    // resolution on unix-seconds).
    std::thread::sleep(Duration::from_millis(1_100));
    assert_eq!(store.get(&hash), None, "expired entry must be purged");
    assert_eq!(store.len(), 0, "expired entry must be physically deleted");
}

#[test]
fn sqlite_persists_across_proxy_restart() {
    // Acceptance criterion #4 from the plan: write via SqliteCcrStore,
    // drop the store, reconstruct from the same DB path, retrieve same
    // hash → original bytes recover.
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("ccr.sqlite");
    let payload = "long-lived original payload";
    let hash = compute_key(payload.as_bytes());

    {
        let store = SqliteCcrStore::open(&path, 300).expect("open sqlite store (turn 1)");
        store.put(&hash, payload);
        // `store` drops here, simulating worker shutdown.
    }

    // Reconstruct from the same path — simulates `--workers 1` restart.
    let store = SqliteCcrStore::open(&path, 300).expect("re-open sqlite store (turn 2)");
    let fetched = store.get(&hash);
    assert_eq!(
        fetched.as_deref(),
        Some(payload),
        "re-opened sqlite store must recover the original bytes"
    );
}

#[test]
fn from_config_sqlite_roundtrip() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("ccr.sqlite");
    let cfg = CcrBackendConfig::Sqlite {
        path: path.clone(),
        ttl_seconds: 300,
    };
    let store = from_config(&cfg).expect("from_config(sqlite)");
    let hash = compute_key(b"hello");
    store.put(&hash, "hello");
    assert_eq!(store.get(&hash).as_deref(), Some("hello"));
}

#[test]
fn from_config_in_memory_roundtrip() {
    let cfg = CcrBackendConfig::in_memory_default();
    let store = from_config(&cfg).expect("from_config(in_memory)");
    let hash = compute_key(b"bye");
    store.put(&hash, "bye");
    assert_eq!(store.get(&hash).as_deref(), Some("bye"));
}

#[cfg(not(feature = "redis"))]
#[test]
fn from_config_redis_unsupported_when_feature_off() {
    use headroom_core::ccr::backends::CcrBackendInitError;

    let cfg = CcrBackendConfig::Redis {
        url: "redis://127.0.0.1:6379".to_string(),
        ttl_seconds: 300,
        key_prefix: None,
    };
    match from_config(&cfg) {
        Err(CcrBackendInitError::UnsupportedBackend { backend, feature }) => {
            assert_eq!(backend, "redis");
            assert_eq!(feature, "redis");
        }
        Err(other) => panic!("expected UnsupportedBackend, got {other:?}"),
        Ok(_) => panic!("redis must error when feature is off"),
    }
}

#[test]
fn backend_swap_byte_equal_keys() {
    // Stage data through one backend, swap to another with the same
    // payload, and assert the keys are byte-equal. This is the
    // load-bearing invariant: operators may migrate between backends
    // (e.g. SQLite → Redis when scaling out) and the in-flight CCR
    // markers must keep working — the marker bytes are the hash, and
    // the hash function is fixed in `ccr::compute_key`.
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("ccr.sqlite");

    let sqlite = SqliteCcrStore::open(&path, 300).expect("open sqlite store");
    let in_memory = InMemoryCcrStore::new();

    let payloads = [
        "alpha",
        r#"[{"id":1}]"#,
        "the quick brown fox jumps over the lazy dog",
        "<<<<>>>>", // marker-adjacent characters — sanity check on the BLAKE3 trim
    ];

    for payload in &payloads {
        let key_a = compute_key(payload.as_bytes());
        let key_b = compute_key(payload.as_bytes());
        // Step 1: same payload yields byte-equal keys.
        assert_eq!(key_a, key_b, "compute_key must be deterministic");

        // Step 2: store in sqlite, mirror to in-memory under the same
        // key — both backends recover byte-equal values.
        sqlite.put(&key_a, payload);
        in_memory.put(&key_b, payload);

        let v_sqlite = sqlite.get(&key_a);
        let v_mem = in_memory.get(&key_b);
        assert_eq!(v_sqlite.as_deref(), Some(*payload));
        assert_eq!(v_mem.as_deref(), Some(*payload));
        assert_eq!(
            v_sqlite, v_mem,
            "sqlite and in-memory must return byte-equal payloads"
        );
    }
}

// ─── Redis-feature-gated tests ─────────────────────────────────────────

#[cfg(feature = "redis")]
mod redis_tests {
    use super::*;
    use headroom_core::ccr::backends::RedisCcrStore;

    /// Reads `HEADROOM_TEST_REDIS_URL` from the environment — when the
    /// feature is on but no URL is configured we silently no-op. CI
    /// runs the redis test in a docker-compose'd matrix.
    fn redis_url() -> Option<String> {
        std::env::var("HEADROOM_TEST_REDIS_URL").ok()
    }

    #[test]
    fn redis_round_trip() {
        let Some(url) = redis_url() else {
            eprintln!("skipping redis_round_trip: HEADROOM_TEST_REDIS_URL not set");
            return;
        };
        let store = RedisCcrStore::open(&url, 300).expect("open redis store");
        let payload = "redis payload";
        let hash = compute_key(payload.as_bytes());
        store.put(&hash, payload);
        assert_eq!(store.get(&hash).as_deref(), Some(payload));
    }

    #[test]
    fn redis_round_trip_via_from_config() {
        let Some(url) = redis_url() else {
            eprintln!("skipping redis_round_trip_via_from_config: HEADROOM_TEST_REDIS_URL not set");
            return;
        };
        let cfg = CcrBackendConfig::Redis {
            url,
            ttl_seconds: 300,
            key_prefix: Some("ccr_test".to_string()),
        };
        let store = from_config(&cfg).expect("from_config(redis)");
        let payload = "via factory";
        let hash = compute_key(payload.as_bytes());
        store.put(&hash, payload);
        assert_eq!(store.get(&hash).as_deref(), Some(payload));
    }
}
