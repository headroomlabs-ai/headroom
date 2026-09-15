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
    // window. Five gaps carry total age to at least 4s, which is strictly
    // beyond the 3s ceiling even after unix-second truncation. Four gaps
    // only reach 3.3s and can land exactly on the now-valid 3s boundary.
    for _ in 0..5 {
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

    /// A key prefix that is genuinely unique per run, so a leftover key
    /// from an earlier run — or a concurrent run against a shared
    /// server — cannot influence a result. The pid alone is not enough:
    /// pids are recycled.
    fn unique_prefix(tag: &str) -> String {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        format!("ccr_test_{tag}_{}_{nanos}", std::process::id())
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

    #[test]
    fn redis_max_lifetime_caps_sliding_window() {
        let Some(url) = redis_url() else {
            eprintln!(
                "skipping redis_max_lifetime_caps_sliding_window: HEADROOM_TEST_REDIS_URL not set"
            );
            return;
        };
        // Idle 2s with a 5s ceiling, both set explicitly rather than
        // riding the 8x default — deriving the ceiling would force an
        // idle TTL of 1s (the whole-second floor for EXPIRE/TTL) and so
        // an 8s ceiling, roughly doubling this test's wall clock.
        let store = RedisCcrStore::open_with_ttls(&url, unique_prefix("ceiling"), 2, 5)
            .expect("open redis store with ceiling");
        let payload = "capped redis";
        let hash = compute_key(payload.as_bytes());
        store.put(&hash, payload);

        // Assert a hit before the ceiling first. Without this the test
        // passes if the very first `get` returns None — i.e. it cannot
        // tell "purged by the ceiling" from "purged on contact", which
        // is what a ceiling-arithmetic bug at small TTLs looks like.
        // Mirrors the positive assertion in the SQLite equivalent.
        std::thread::sleep(Duration::from_millis(700));
        assert_eq!(
            store.get(&hash).as_deref(),
            Some(payload),
            "entry inside both the idle window and the ceiling must hit"
        );

        // Touch every 700ms: inside the 2s idle window, so the entry
        // never dies of idleness and the purge below can only come from
        // the ceiling — the thing under test.
        let start = std::time::Instant::now();
        let deadline = start + Duration::from_millis(12_000);
        let mut expired_after = None;
        while std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(700));
            if store.get(&hash).is_none() {
                expired_after = Some(start.elapsed());
                break;
            }
        }
        let expired_after =
            expired_after.expect("constant access must not extend an entry past its max lifetime");
        // The 5s ceiling runs from the `put` ~0.7s before `start`, and
        // TTL/EXPIRE truncate to whole seconds, so the earliest
        // legitimate purge is ~3.3s into this loop. Anything much faster
        // means the entry died of something other than the ceiling.
        assert!(
            expired_after >= Duration::from_millis(1_500),
            "entry was purged after only {expired_after:?} — too early to be the 5s ceiling"
        );
    }

    #[test]
    fn redis_ceiling_is_independent_of_the_reading_stores_ttl() {
        let Some(url) = redis_url() else {
            eprintln!(
                "skipping redis_ceiling_is_independent_of_the_reading_stores_ttl: \
                 HEADROOM_TEST_REDIS_URL not set"
            );
            return;
        };
        // Regression test: the ceiling used to be encoded as the born
        // marker's TTL plus a grace period equal to the *writing* store's
        // idle TTL, which `get` then subtracted using the *reading*
        // store's idle TTL. Any store configured with a larger idle TTL
        // than the writer therefore computed a negative remainder,
        // saturated to zero, and purged a live entry on first read. That
        // is the shared-keyspace case this backend exists for, and it is
        // also what raising HEADROOM_CCR_TTL_SECONDS would have done to
        // every in-flight entry.
        let prefix = unique_prefix("reconfig");
        let payload = "survives a ttl reconfiguration";
        let hash = compute_key(payload.as_bytes());

        // Writer: short idle window, generous ceiling.
        let writer =
            RedisCcrStore::open_with_ttls(&url, prefix.clone(), 2, 60).expect("open writing store");
        writer.put(&hash, payload);

        // Reader: same keyspace, much larger idle window — a second
        // worker configured differently, or the same worker after an
        // operator raised the TTL. The birth instant is absolute, so the
        // reader must derive a ceiling that has barely elapsed and hit.
        let reader =
            RedisCcrStore::open_with_prefix(&url, prefix, 300).expect("open reading store");
        assert_eq!(
            reader.get(&hash).as_deref(),
            Some(payload),
            "a store with a larger idle TTL must not purge an entry nowhere near its ceiling"
        );
    }

    #[test]
    fn redis_legacy_born_placeholder_is_backfilled_not_purged() {
        use redis::Commands as _;

        let Some(url) = redis_url() else {
            eprintln!(
                "skipping redis_legacy_born_placeholder_is_backfilled_not_purged: \
                 HEADROOM_TEST_REDIS_URL not set"
            );
            return;
        };
        // Builds before the birth-instant change wrote a `1` placeholder
        // as the born marker's value. Read naively as a unix timestamp
        // that is 1970, making every such entry look infinitely old and
        // purging it on the first `get` after an upgrade. It must be
        // recognised as a legacy marker and backfilled instead.
        let prefix = unique_prefix("legacy_born");
        let payload = "legacy born marker";
        let hash = compute_key(payload.as_bytes());
        let store = RedisCcrStore::open_with_ttls(&url, prefix.clone(), 300, 2_400)
            .expect("open redis store");
        store.put(&hash, payload);

        let client = redis::Client::open(url.as_str()).expect("raw client");
        let mut conn = client.get_connection().expect("raw connection");
        let born_key = format!("{prefix}:{hash}:born");
        let _: () = conn
            .set_ex(&born_key, 1_u8, 2_400)
            .expect("overwrite born marker with the legacy placeholder");

        assert_eq!(
            store.get(&hash).as_deref(),
            Some(payload),
            "a legacy `1` placeholder must backfill the ceiling, not purge the entry"
        );
        let born: String = conn.get(&born_key).expect("read back born marker");
        assert!(
            born.parse::<u64>().expect("born marker must be numeric") >= 1_600_000_000,
            "backfill must replace the placeholder with a unix-second birth instant, got {born}"
        );
    }

    #[test]
    fn redis_rejects_a_zero_idle_ttl() {
        // No server needed — the guard runs before the client is opened,
        // so this covers contributors with no Redis available too.
        //
        // A zero idle TTL is accepted by `SqliteCcrStore` but makes every
        // Redis `SETEX` fail server-side, so without this guard the store
        // opens fine and then silently stores nothing.
        //
        // Matched rather than `expect_err`'d because `RedisCcrStore` holds a
        // `redis::Client` and so cannot derive `Debug`.
        let Err(err) = RedisCcrStore::open("redis://127.0.0.1:6379", 0) else {
            panic!("a zero idle TTL must be rejected at open");
        };
        assert!(
            err.to_string().contains("default_ttl_seconds"),
            "the error must name the offending setting, got: {err}"
        );
    }
}
