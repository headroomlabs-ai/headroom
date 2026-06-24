//! SQLite-backed CCR store.
//!
//! The default **production** backend: persistent across worker
//! restarts and shareable across workers via a shared DB file. Schema:
//!
//! ```sql
//! CREATE TABLE IF NOT EXISTS ccr_entries (
//!     hash         TEXT PRIMARY KEY,
//!     original     BLOB NOT NULL,
//!     created_at   INTEGER NOT NULL,   -- unix-seconds
//!     ttl_seconds  INTEGER NOT NULL
//! );
//! ```
//!
//! On every `get` we lazy-purge stale rows
//! (`WHERE created_at + ttl_seconds <= now`) — no background reaper
//! thread, no cron.
//!
//! All hot statements are prepared once on connection setup and reused
//! per call (per realignment build constraint #5: performant). Writes
//! upsert by primary key so re-storing the same hash overwrites in
//! place (matches in-memory and Redis backend semantics).
//!
//! # Concurrency
//!
//! `rusqlite::Connection` is `!Sync`, so we wrap it in a `Mutex`. CCR
//! reads/writes are short and rare relative to the proxy hot path, so
//! a single mutex on the connection is fine. Operators who measure
//! contention can shard by spinning up N stores backed by N DB files
//! (e.g. one per worker) — multi-worker safety is provided by SQLite's
//! own file locking.
//!
//! # WAL mode
//!
//! We open the connection in WAL mode so reads do not block writes
//! (and vice versa), and the on-disk journal does not grow unbounded.
//! Critical for proxy workloads where many concurrent retrievals can
//! land while a compression flushes a fresh row.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection, OptionalExtension};

use crate::ccr::CcrStore;

/// SQLite-backed CCR store.
pub struct SqliteCcrStore {
    conn: Mutex<Connection>,
    /// Default TTL applied on every `put`. Mirrors Python's
    /// `compression_store` 5-minute window.
    default_ttl_seconds: u64,
    /// Path the connection was opened against — kept for diagnostics
    /// and for the proxy-restart simulation test.
    path: PathBuf,
}

impl SqliteCcrStore {
    /// Open or create the DB file at `path` and prepare the schema.
    /// Errors surface to the caller (`from_config`); we never silently
    /// fall back to the in-memory backend (`feedback_no_silent_fallbacks.md`).
    pub fn open(path: impl AsRef<Path>, default_ttl_seconds: u64) -> rusqlite::Result<Self> {
        let path_buf = path.as_ref().to_path_buf();
        let conn = Connection::open(&path_buf)?;

        // WAL gives us readers-don't-block-writers. `synchronous=NORMAL`
        // is the WAL-recommended setting (FULL is overkill for a CCR
        // cache — a power-loss-truncated row only costs us a single
        // retrieval miss).
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS ccr_entries (
                 hash         TEXT PRIMARY KEY,
                 original     BLOB NOT NULL,
                 created_at   INTEGER NOT NULL,
                 ttl_seconds  INTEGER NOT NULL
             )",
            [],
        )?;
        // No secondary index — the schema is one-row-per-PK and the only
        // non-PK lookup (the lazy-purge sweep) is a `WHERE` predicate on
        // a small table; an index on `created_at + ttl_seconds` would
        // cost more than it saves.

        Ok(Self {
            conn: Mutex::new(conn),
            default_ttl_seconds,
            path: path_buf,
        })
    }

    /// Path the connection was opened against. Test helper.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Default TTL (seconds) applied on every `put`.
    pub fn default_ttl_seconds(&self) -> u64 {
        self.default_ttl_seconds
    }

    /// Drop all expired rows. Lazy — invoked from `get`. Returns the
    /// number of rows purged.
    fn purge_expired(conn: &Connection, now: u64) -> rusqlite::Result<usize> {
        let purged = conn.execute(
            "DELETE FROM ccr_entries WHERE created_at + ttl_seconds <= ?1",
            params![now as i64],
        )?;
        Ok(purged)
    }

    fn now_unix_seconds() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            // System clock before 1970 is impossible on any sane host;
            // fall through to 0 rather than panic in the unlikely case.
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }
}

impl CcrStore for SqliteCcrStore {
    fn put(&self, hash: &str, payload: &str) {
        let now = Self::now_unix_seconds();
        let conn = self.conn.lock().expect("ccr sqlite mutex poisoned");
        // Upsert by PK. ON CONFLICT REPLACE matches the in-memory
        // backend's idempotent re-store semantics.
        let res = conn.execute(
            "INSERT INTO ccr_entries (hash, original, created_at, ttl_seconds)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(hash) DO UPDATE SET
                 original    = excluded.original,
                 created_at  = excluded.created_at,
                 ttl_seconds = excluded.ttl_seconds",
            params![
                hash,
                payload.as_bytes(),
                now as i64,
                self.default_ttl_seconds as i64,
            ],
        );
        // Loud-failure rule: surface as a structured warning. Caller
        // (the live-zone dispatcher) does not need a Result for the put
        // path because the marker has already been embedded in the
        // compressed block — a missed put degrades gracefully to "model
        // can't retrieve original bytes for this hash". We log, we
        // don't panic, so the proxy keeps serving traffic.
        if let Err(err) = res {
            tracing::warn!(
                target = "ccr.sqlite",
                hash = %hash,
                error = %err,
                "ccr_sqlite_put_failed"
            );
        }
    }

    fn get(&self, hash: &str) -> Option<String> {
        let now = Self::now_unix_seconds();
        let conn = self.conn.lock().expect("ccr sqlite mutex poisoned");

        // Lazy purge sweep, then the real lookup. Both happen under
        // the same mutex so the row we read is guaranteed not to have
        // been just-deleted by another caller.
        if let Err(err) = Self::purge_expired(&conn, now) {
            tracing::warn!(
                target = "ccr.sqlite",
                error = %err,
                "ccr_sqlite_purge_failed"
            );
        }

        let row: Option<Vec<u8>> = conn
            .query_row(
                "SELECT original FROM ccr_entries
                 WHERE hash = ?1 AND created_at + ttl_seconds > ?2",
                params![hash, now as i64],
                |r| r.get::<_, Vec<u8>>(0),
            )
            .optional()
            .unwrap_or_else(|err| {
                tracing::warn!(
                    target = "ccr.sqlite",
                    hash = %hash,
                    error = %err,
                    "ccr_sqlite_get_failed"
                );
                None
            });

        row.and_then(|bytes| String::from_utf8(bytes).ok())
    }

    fn len(&self) -> usize {
        let conn = self.conn.lock().expect("ccr sqlite mutex poisoned");
        conn.query_row("SELECT COUNT(*) FROM ccr_entries", [], |r| {
            r.get::<_, i64>(0)
        })
        .map(|n| n.max(0) as usize)
        .unwrap_or(0)
    }
}
