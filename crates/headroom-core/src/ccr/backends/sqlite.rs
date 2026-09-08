//! SQLite-backed CCR store.
//!
//! The default **production** backend: persistent across worker
//! restarts and shareable across workers via a shared DB file. Schema:
//!
//! ```sql
//! CREATE TABLE IF NOT EXISTS ccr_entries (
//!     hash          TEXT PRIMARY KEY,
//!     original      BLOB NOT NULL,
//!     created_at    INTEGER NOT NULL,   -- unix-seconds
//!     ttl_seconds   INTEGER NOT NULL,   -- idle window, restarted on get
//!     last_accessed INTEGER NOT NULL    -- unix-seconds
//! );
//! ```
//!
//! The TTL is an **idle window** (#2604): every successful `get`
//! restarts the row's clock via `last_accessed`, bounded by an absolute
//! max lifetime measured from `created_at`. On every `get` we
//! lazy-purge stale rows (`WHERE last_accessed + ttl_seconds < now OR
//! created_at + max_lifetime < now`) — no background reaper thread,
//! no cron. DBs created by pre-sliding builds are migrated in place
//! (the `last_accessed` column is added, backfilled from `created_at`).
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

use crate::ccr::{max_lifetime_for, CcrStore};

/// SQLite-backed CCR store.
pub struct SqliteCcrStore {
    conn: Mutex<Connection>,
    /// Default idle TTL applied on every `put`. Mirrors Python's
    /// `compression_store` idle window.
    default_ttl_seconds: u64,
    /// Absolute max lifetime (seconds since `created_at`) that caps the
    /// sliding idle window. Defaults to 8x the idle TTL.
    max_lifetime_seconds: u64,
    /// Path the connection was opened against — kept for diagnostics
    /// and for the proxy-restart simulation test.
    path: PathBuf,
}

impl SqliteCcrStore {
    /// Open or create the DB file at `path` and prepare the schema.
    /// `default_ttl_seconds` is the idle window; the absolute max
    /// lifetime defaults to 8x that (see
    /// [`crate::ccr::DEFAULT_MAX_LIFETIME_MULTIPLIER`]).
    /// Errors surface to the caller (`from_config`); we never silently
    /// fall back to the in-memory backend (`feedback_no_silent_fallbacks.md`).
    pub fn open(path: impl AsRef<Path>, default_ttl_seconds: u64) -> rusqlite::Result<Self> {
        let max_lifetime =
            max_lifetime_for(std::time::Duration::from_secs(default_ttl_seconds)).as_secs();
        Self::open_with_ttls(path, default_ttl_seconds, max_lifetime)
    }

    /// Full-control constructor: idle window and absolute max lifetime
    /// specified independently.
    pub fn open_with_ttls(
        path: impl AsRef<Path>,
        default_ttl_seconds: u64,
        max_lifetime_seconds: u64,
    ) -> rusqlite::Result<Self> {
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
                 hash          TEXT PRIMARY KEY,
                 original      BLOB NOT NULL,
                 created_at    INTEGER NOT NULL,
                 ttl_seconds   INTEGER NOT NULL,
                 last_accessed INTEGER NOT NULL
             )",
            [],
        )?;
        Self::migrate_legacy_schema(&conn)?;
        // No secondary index — the schema is one-row-per-PK and the only
        // non-PK lookup (the lazy-purge sweep) is a `WHERE` predicate on
        // a small table; an index on the expiry expressions would cost
        // more than it saves.

        Ok(Self {
            conn: Mutex::new(conn),
            default_ttl_seconds,
            max_lifetime_seconds,
            path: path_buf,
        })
    }

    /// DBs created before the sliding-TTL change lack `last_accessed`.
    /// Add it in place and backfill from `created_at` so legacy rows
    /// keep their original expiry baseline rather than being purged or
    /// artificially refreshed.
    fn migrate_legacy_schema(conn: &Connection) -> rusqlite::Result<()> {
        let has_last_accessed = conn
            .prepare("SELECT 1 FROM pragma_table_info('ccr_entries') WHERE name = 'last_accessed'")?
            .exists([])?;
        if !has_last_accessed {
            conn.execute(
                "ALTER TABLE ccr_entries ADD COLUMN last_accessed INTEGER NOT NULL DEFAULT 0",
                [],
            )?;
            conn.execute(
                "UPDATE ccr_entries SET last_accessed = created_at WHERE last_accessed = 0",
                [],
            )?;
        }
        Ok(())
    }

    /// Path the connection was opened against. Test helper.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Default TTL (seconds) applied on every `put`.
    pub fn default_ttl_seconds(&self) -> u64 {
        self.default_ttl_seconds
    }

    /// Drop all expired rows: idle past their window, or past the
    /// absolute max lifetime. Lazy — invoked from `get`. Returns the
    /// number of rows purged.
    fn purge_expired(&self, conn: &Connection, now: u64) -> rusqlite::Result<usize> {
        // Timestamps have whole-second resolution. Use a strict boundary so
        // truncation can extend a cache entry by less than one second but can
        // never expire it before the configured idle or lifetime window.
        let purged = conn.execute(
            "DELETE FROM ccr_entries
             WHERE last_accessed + ttl_seconds < ?1
                OR created_at + ?2 < ?1",
            params![now as i64, self.max_lifetime_seconds as i64],
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

    fn get_at(&self, hash: &str, now: u64) -> Option<String> {
        let conn = self.conn.lock().expect("ccr sqlite mutex poisoned");

        // Lazy purge sweep, then the real lookup. Both happen under
        // the same mutex so the row we read is guaranteed not to have
        // been just-deleted by another caller.
        if let Err(err) = self.purge_expired(&conn, now) {
            tracing::warn!(
                target = "ccr.sqlite",
                error = %err,
                "ccr_sqlite_purge_failed"
            );
        }

        let row: Option<Vec<u8>> = conn
            .query_row(
                "SELECT original FROM ccr_entries
                 WHERE hash = ?1
                   AND last_accessed + ttl_seconds >= ?2
                   AND created_at + ?3 >= ?2",
                params![hash, now as i64, self.max_lifetime_seconds as i64],
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

        let row = row?;
        // Sliding idle window (#2604): a successful hit restarts the
        // row's idle clock. Still under the same mutex as the lookup.
        if let Err(err) = conn.execute(
            "UPDATE ccr_entries SET last_accessed = ?2 WHERE hash = ?1",
            params![hash, now as i64],
        ) {
            tracing::warn!(
                target = "ccr.sqlite",
                hash = %hash,
                error = %err,
                "ccr_sqlite_touch_failed"
            );
        }

        String::from_utf8(row).ok()
    }
}

impl CcrStore for SqliteCcrStore {
    fn put(&self, hash: &str, payload: &str) {
        let now = Self::now_unix_seconds();
        let conn = self.conn.lock().expect("ccr sqlite mutex poisoned");
        // Upsert by PK. ON CONFLICT REPLACE matches the in-memory
        // backend's idempotent re-store semantics.
        let res = conn.execute(
            "INSERT INTO ccr_entries (hash, original, created_at, ttl_seconds, last_accessed)
             VALUES (?1, ?2, ?3, ?4, ?3)
             ON CONFLICT(hash) DO UPDATE SET
                 original      = excluded.original,
                 created_at    = excluded.created_at,
                 ttl_seconds   = excluded.ttl_seconds,
                 last_accessed = excluded.last_accessed",
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
        self.get_at(hash, Self::now_unix_seconds())
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

#[cfg(test)]
mod tests {
    use super::*;

    fn store_with_row(
        idle_ttl: u64,
        max_lifetime: u64,
        created_at: u64,
        last_accessed: u64,
    ) -> (tempfile::TempDir, SqliteCcrStore, String) {
        let dir = tempfile::tempdir().expect("tempdir");
        let store =
            SqliteCcrStore::open_with_ttls(dir.path().join("ccr.sqlite"), idle_ttl, max_lifetime)
                .expect("open sqlite store");
        let hash = "boundary-entry".to_string();
        {
            let conn = store.conn.lock().expect("ccr sqlite mutex poisoned");
            conn.execute(
                "INSERT INTO ccr_entries
                    (hash, original, created_at, ttl_seconds, last_accessed)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    &hash,
                    b"payload".as_slice(),
                    created_at as i64,
                    idle_ttl as i64,
                    last_accessed as i64,
                ],
            )
            .expect("insert boundary row");
        }
        (dir, store, hash)
    }

    #[test]
    fn exact_idle_ttl_boundary_is_still_valid() {
        let (_dir, store, hash) = store_with_row(5, 20, 100, 100);

        assert_eq!(store.get_at(&hash, 105).as_deref(), Some("payload"));
        assert_eq!(store.get_at(&hash, 111), None);
        assert_eq!(store.len(), 0, "expired row must be purged");
    }

    #[test]
    fn exact_max_lifetime_boundary_is_still_valid() {
        let (_dir, store, hash) = store_with_row(5, 10, 100, 108);

        assert_eq!(store.get_at(&hash, 110).as_deref(), Some("payload"));
        assert_eq!(store.get_at(&hash, 111), None);
        assert_eq!(store.len(), 0, "expired row must be purged");
    }
}
