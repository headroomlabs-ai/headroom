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

// ─── Sliding (idle-window) TTL semantics — #2604 ───────────────────────
//
// The Python `CompressionStore` treats `HEADROOM_CCR_TTL_SECONDS` as an
// idle window that restarts on every successful retrieval, bounded by an
// absolute max lifetime (8x the idle TTL). These tests pin the same
// semantics onto the Rust backends so an entry a session keeps touching
// does not expire mid-burst.

#[test]
fn in_memory_get_refreshes_idle_ttl() {
    let store = InMemoryCcrStore::with_capacity_and_ttl(10, Duration::from_millis(120));
    let hash = compute_key(b"hot entry");
    store.put(&hash, "hot entry");
    // Touch the entry every 60ms for ~4 idle windows' worth of wall
    // clock. Wall-clock expiry would kill it at 120ms; a sliding idle
    // window keeps it alive because every hit restarts the clock.
    for _ in 0..8 {
        std::thread::sleep(Duration::from_millis(60));
        assert_eq!(
            store.get(&hash).as_deref(),
            Some("hot entry"),
            "an entry accessed within its idle window must stay alive"
        );
    }
    // Now go idle past the window: the entry must expire.
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(
        store.get(&hash),
        None,
        "an entry idle past its window must expire"
    );
}

#[test]
fn in_memory_max_lifetime_caps_sliding_window() {
    // Idle TTL 40ms → max lifetime 320ms (8x). Constant access must not
    // keep the entry alive forever.
    let store = InMemoryCcrStore::with_capacity_and_ttl(10, Duration::from_millis(40));
    let hash = compute_key(b"immortal?");
    store.put(&hash, "immortal?");
    let deadline = std::time::Instant::now() + Duration::from_millis(600);
    let mut expired = false;
    while std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
        if store.get(&hash).is_none() {
            expired = true;
            break;
        }
    }
    assert!(
        expired,
        "constant access must not extend an entry past its max lifetime"
    );
}

#[test]
fn sqlite_get_refreshes_idle_ttl() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("ccr.sqlite");
    // 3-second idle window (unix-second resolution needs whole seconds).
    let store = SqliteCcrStore::open(&path, 3).expect("open sqlite store");
    let hash = compute_key(b"sliding sqlite");
    store.put(&hash, "sliding sqlite");
    // t+2s: hit inside the window — restarts the idle clock.
    std::thread::sleep(Duration::from_millis(2_000));
    assert_eq!(
        store.get(&hash).as_deref(),
        Some("sliding sqlite"),
        "first access within the idle window must hit"
    );
    // t+4s: wall-clock expiry would have purged at t+3s; the refresh at
    // t+2s must keep it alive until t+5s.
    std::thread::sleep(Duration::from_millis(2_000));
    assert_eq!(
        store.get(&hash).as_deref(),
        Some("sliding sqlite"),
        "an entry accessed within its idle window must stay alive past the wall-clock TTL"
    );
    // Go idle past the window.
    std::thread::sleep(Duration::from_millis(4_100));
    assert_eq!(
        store.get(&hash),
        None,
        "an entry idle past its window must be purged"
    );
}

#[test]
fn sqlite_max_lifetime_caps_sliding_window() {
    // Timing note: the backend stores unix-SECONDS (`as_secs()` truncates)
    // and purges on `last_accessed + ttl <= now`, so apparent elapsed time
    // is `floor(t0 + s) - floor(t0)` — it rounds UP by nearly a second
    // depending on where t0 lands within its second. Every margin here is
    // therefore kept a full second clear of the boundary in both
    // directions; a sub-second margin makes this test phase-dependent
    // (the previous 1.5s-against-a-2s-window "still alive" assertion
    // failed ~70% of runs whenever `frac(t0) >= 0.5`).
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("ccr.sqlite");
    // Idle 2s with a 3s ceiling: constant access must not outlive t+3s.
    let store =
        SqliteCcrStore::open_with_ttls(&path, 2, 3).expect("open sqlite store with ceiling");
    let hash = compute_key(b"capped sqlite");
    store.put(&hash, "capped sqlite");
    // 0.5s: apparent elapsed is 0s or 1s — always under the 2s window.
    std::thread::sleep(Duration::from_millis(500));
    assert_eq!(
        store.get(&hash).as_deref(),
        Some("capped sqlite"),
        "entry inside idle window and ceiling must hit"
    );
    // Keep touching, but cross the 3s ceiling. The touches must stay INSIDE
    // the idle window or the entry dies of idleness and the assertion below
    // passes without ever exercising the ceiling — the thing under test.
    // 0.7s gaps read as at most 1s apparent, comfortably under the 2s idle
    // window, while the 4 of them carry total age past 3s.
    for _ in 0..4 {
        std::thread::sleep(Duration::from_millis(700));
        let _ = store.get(&hash);
    }
    assert_eq!(
        store.get(&hash),
        None,
        "constant access must not extend an entry past its max lifetime"
    );
}

#[test]
fn sqlite_migrates_legacy_schema_without_last_accessed() {
    // A DB created by a pre-sliding-TTL build has no `last_accessed`
    // column. Opening it must migrate in place and keep the rows
    // retrievable (backfilling last_accessed from created_at).
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("ccr.sqlite");
    let payload = "legacy row";
    let hash = compute_key(payload.as_bytes());
    {
        let conn = rusqlite::Connection::open(&path).expect("open raw connection");
        conn.execute(
            "CREATE TABLE ccr_entries (
                 hash         TEXT PRIMARY KEY,
                 original     BLOB NOT NULL,
                 created_at   INTEGER NOT NULL,
                 ttl_seconds  INTEGER NOT NULL
             )",
            [],
        )
        .expect("create legacy schema");
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        conn.execute(
            "INSERT INTO ccr_entries (hash, original, created_at, ttl_seconds)
             VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![hash, payload.as_bytes(), now, 300_i64],
        )
        .expect("insert legacy row");
    }
    let store = SqliteCcrStore::open(&path, 300).expect("open must migrate legacy schema");
    assert_eq!(
        store.get(&hash).as_deref(),
        Some(payload),
        "legacy rows must survive the schema migration"
    );
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

    #[test]
    fn redis_get_refreshes_idle_ttl() {
        let Some(url) = redis_url() else {
            eprintln!("skipping redis_get_refreshes_idle_ttl: HEADROOM_TEST_REDIS_URL not set");
            return;
        };
        // 2-second idle window (Redis EXPIRE has 1s resolution).
        let store = RedisCcrStore::open_with_prefix(&url, "ccr_test_sliding".to_string(), 2)
            .expect("open redis store");
        let payload = "sliding redis";
        let hash = compute_key(payload.as_bytes());
        store.put(&hash, payload);
        // Touch at t+1.5s (inside window) — restarts the idle clock.
        std::thread::sleep(Duration::from_millis(1_500));
        assert_eq!(store.get(&hash).as_deref(), Some(payload));
        // t+3s: wall-clock expiry would have fired at t+2s.
        std::thread::sleep(Duration::from_millis(1_500));
        assert_eq!(
            store.get(&hash).as_deref(),
            Some(payload),
            "an entry accessed within its idle window must stay alive past the wall-clock TTL"
        );
        // Go idle past the window.
        std::thread::sleep(Duration::from_millis(3_100));
        assert_eq!(store.get(&hash), None);
    }
}
