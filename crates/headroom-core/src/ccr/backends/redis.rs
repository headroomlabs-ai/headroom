//! Redis-backed CCR store.
//!
//! Opt-in **multi-worker** backend: every worker hits the same Redis
//! instance, so no sticky-session is required at the load balancer.
//! Compiled only when the `redis` feature is enabled — production
//! deployments wanting Redis pull this in via the workspace feature
//! flag, deployments running single-worker or persistent-disk-only
//! avoid the Redis client cost.
//!
//! # Storage model
//!
//! Each entry maps to a Redis key `ccr:{hash}` containing the original
//! payload bytes, with a `SETEX` TTL applied on every write. The TTL is
//! an **idle window** (#2604): every successful `get` re-arms the key's
//! expiry, bounded by an absolute max lifetime tracked in a companion
//! `ccr:{hash}:born` key. That marker's *value* is the entry's birth
//! instant in unix seconds, and the ceiling is derived by comparing it
//! to now — the same absolute-birth-instant model the SQLite backend
//! uses in its `created_at` column and the in-memory backend uses in
//! its `inserted: Instant`. The marker's own TTL is only a cleanup
//! bound, so it carries no part of the ceiling arithmetic and a store
//! reading it needs no knowledge of the TTL the writer was configured
//! with. Redis handles purging via key expiry — no application-side
//! sweep needed (matching the SQLite backend's lazy-purge but at the
//! Redis level).
//!
//! # Concurrency
//!
//! `redis::Client` is `Send + Sync`; we hold one per store instance.
//! `get_connection` returns a fresh blocking connection per call; this
//! is the recommended pattern for short-lived puts/gets and avoids the
//! `MultiplexedConnection`'s tokio-runtime requirement (CCR is called
//! both from sync and tokio contexts in the proxy crate).

#![cfg(feature = "redis")]

use std::time::{SystemTime, UNIX_EPOCH};

use redis::Commands;

use crate::ccr::{max_lifetime_for, CcrStore};

/// Key prefix applied to every CCR entry. Configurable per-deployment
/// so multiple proxies sharing one Redis don't collide.
const DEFAULT_KEY_PREFIX: &str = "ccr";

/// Smallest value accepted as a birth instant in a born marker
/// (2020-09-13). Builds before this one wrote a `1` placeholder there
/// rather than a timestamp, so anything below the threshold is treated
/// as a legacy marker and backfilled instead of being read as a birth
/// instant somewhere in 1970 — which would purge every in-flight entry
/// on upgrade.
const MIN_PLAUSIBLE_BORN_AT: u64 = 1_600_000_000;

/// Redis-backed CCR store. Cfg-gated behind `feature = "redis"`.
pub struct RedisCcrStore {
    client: redis::Client,
    key_prefix: String,
    default_ttl_seconds: u64,
    /// Absolute max lifetime (seconds since `put`) that caps the
    /// sliding idle window. Defaults to 8x the idle TTL.
    max_lifetime_seconds: u64,
}

impl RedisCcrStore {
    /// Open a Redis connection at `url` (e.g. `redis://127.0.0.1:6379`).
    /// Errors surface to the caller (`from_config`).
    pub fn open(url: &str, default_ttl_seconds: u64) -> redis::RedisResult<Self> {
        Self::open_with_prefix(url, DEFAULT_KEY_PREFIX.to_string(), default_ttl_seconds)
    }

    /// Open with an explicit key prefix; the max lifetime is derived as
    /// `8x` the idle TTL (see [`max_lifetime_for`]).
    pub fn open_with_prefix(
        url: &str,
        key_prefix: String,
        default_ttl_seconds: u64,
    ) -> redis::RedisResult<Self> {
        let max_lifetime_seconds =
            max_lifetime_for(std::time::Duration::from_secs(default_ttl_seconds)).as_secs();
        Self::open_with_ttls(url, key_prefix, default_ttl_seconds, max_lifetime_seconds)
    }

    /// Open with both the idle window and the absolute ceiling set
    /// explicitly. Mirrors `SqliteCcrStore::open_with_ttls` and
    /// `InMemoryCcrStore::with_capacity_and_ttls` so all three backends
    /// expose the same knobs — without it the ceiling can only ever be
    /// `8x` the idle TTL, which forces any test of the ceiling to wait
    /// out eight idle windows.
    pub fn open_with_ttls(
        url: &str,
        key_prefix: String,
        default_ttl_seconds: u64,
        max_lifetime_seconds: u64,
    ) -> redis::RedisResult<Self> {
        // Reject a zero idle TTL loudly instead of accepting a store
        // whose every write then fails (`feedback_no_silent_fallbacks.md`).
        // `SETEX` requires a positive expire, so at 0 the payload write
        // is rejected by the server and `put` becomes a no-op that logs
        // a warning and stores nothing — a silent data-loss config.
        if default_ttl_seconds == 0 {
            return Err(redis::RedisError::from((
                redis::ErrorKind::InvalidClientConfig,
                "CCR redis backend requires default_ttl_seconds >= 1",
                format!("got {default_ttl_seconds}; SETEX rejects a zero expire"),
            )));
        }
        let client = redis::Client::open(url)?;
        // Smoke-test the connection at startup so init failures are
        // loud (`feedback_no_silent_fallbacks.md`). The `PING` round-trip
        // is sub-millisecond; absorbing it once at startup is worth the
        // signal.
        let mut conn = client.get_connection()?;
        let _: String = redis::cmd("PING").query(&mut conn)?;
        Ok(Self {
            client,
            key_prefix,
            default_ttl_seconds,
            max_lifetime_seconds,
        })
    }

    fn key_for(&self, hash: &str) -> String {
        format!("{}:{}", self.key_prefix, hash)
    }

    /// Companion key whose *value* is the entry's birth instant in unix
    /// seconds. Every idle-window re-arm is capped by what remains of
    /// `max_lifetime_seconds` measured from that instant.
    fn born_key_for(&self, hash: &str) -> String {
        format!("{}:{}:born", self.key_prefix, hash)
    }

    /// TTL to write on the born marker.
    ///
    /// This is a cleanup bound only — the ceiling lives in the marker's
    /// value, not in its expiry — but the marker must still outlive the
    /// payload key it governs. If it expires first, `get` cannot tell
    /// "past the ceiling" from "legacy entry with no marker", takes the
    /// backfill path, and resets the very ceiling it was enforcing;
    /// under constant access the entry is then pinned indefinitely.
    ///
    /// A payload key's TTL is never more than `default_ttl_seconds` and
    /// is never re-armed beyond the ceiling, so holding the marker one
    /// idle window past the ceiling is sufficient.
    fn born_key_ttl_seconds(&self) -> u64 {
        self.max_lifetime_seconds
            .saturating_add(self.default_ttl_seconds)
    }

    /// Default TTL (seconds) applied on every `put`.
    pub fn default_ttl_seconds(&self) -> u64 {
        self.default_ttl_seconds
    }

    /// Absolute max lifetime (seconds) capping the sliding idle window.
    pub fn max_lifetime_seconds(&self) -> u64 {
        self.max_lifetime_seconds
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

impl CcrStore for RedisCcrStore {
    fn put(&self, hash: &str, payload: &str) {
        let key = self.key_for(hash);
        let mut conn = match self.client.get_connection() {
            Ok(c) => c,
            Err(err) => {
                tracing::warn!(
                    target = "ccr.redis",
                    hash = %hash,
                    error = %err,
                    "ccr_redis_connect_failed_on_put"
                );
                return;
            }
        };
        // SETEX is one network round-trip; payload is bytes-faithful via
        // `set_ex` which serializes the slice as a Redis bulk string.
        let res: redis::RedisResult<()> =
            conn.set_ex(&key, payload.as_bytes(), self.default_ttl_seconds);
        if let Err(err) = res {
            tracing::warn!(
                target = "ccr.redis",
                hash = %hash,
                error = %err,
                "ccr_redis_put_failed"
            );
            return;
        }
        // Companion max-lifetime marker holding this entry's birth
        // instant. `get` derives the remaining ceiling from it, so
        // constant access cannot pin an entry past
        // `max_lifetime_seconds`. Storing the instant rather than
        // encoding the ceiling in the marker's TTL is what keeps the
        // ceiling absolute: a store with a different `default_ttl_seconds`
        // reads the same birth instant and computes the same ceiling.
        let born: redis::RedisResult<()> = conn.set_ex(
            self.born_key_for(hash),
            Self::now_unix_seconds(),
            self.born_key_ttl_seconds(),
        );
        if let Err(err) = born {
            tracing::warn!(
                target = "ccr.redis",
                hash = %hash,
                error = %err,
                "ccr_redis_put_born_failed"
            );
        }
    }

    fn get(&self, hash: &str) -> Option<String> {
        let key = self.key_for(hash);
        let mut conn = match self.client.get_connection() {
            Ok(c) => c,
            Err(err) => {
                tracing::warn!(
                    target = "ccr.redis",
                    hash = %hash,
                    error = %err,
                    "ccr_redis_connect_failed_on_get"
                );
                return None;
            }
        };
        let bytes: redis::RedisResult<Option<Vec<u8>>> = conn.get(&key);
        let payload = match bytes {
            Ok(Some(bytes)) => String::from_utf8(bytes).ok()?,
            Ok(None) => return None,
            Err(err) => {
                tracing::warn!(
                    target = "ccr.redis",
                    hash = %hash,
                    error = %err,
                    "ccr_redis_get_failed"
                );
                return None;
            }
        };

        // Sliding idle window (#2604): re-arm the key's expiry on every
        // hit, capped by what remains of the ceiling measured from the
        // birth instant in the companion born key.
        let born_key = self.born_key_for(hash);
        let born_raw: redis::RedisResult<Option<String>> = conn.get(&born_key);
        let born_at = match born_raw {
            // A failed read is not evidence about the ceiling, and it is
            // not a legacy entry either. Serve the payload and leave the
            // key's existing TTL alone: backfilling here would reset the
            // ceiling on every transient error, which is exactly what
            // lets constant access pin an entry indefinitely.
            Err(err) => {
                tracing::warn!(
                    target = "ccr.redis",
                    hash = %hash,
                    error = %err,
                    "ccr_redis_born_read_failed"
                );
                return Some(payload);
            }
            Ok(raw) => raw
                .and_then(|raw| raw.parse::<u64>().ok())
                .filter(|born_at| *born_at >= MIN_PLAUSIBLE_BORN_AT),
        };
        let remaining = match born_at {
            Some(born_at) => {
                // Absolute ceiling: elapsed since birth, not something
                // recovered from a TTL, so it survives a re-arm and does
                // not depend on this store's own TTL configuration.
                let age = Self::now_unix_seconds().saturating_sub(born_at);
                self.max_lifetime_seconds.saturating_sub(age)
            }
            None => {
                // No marker (entry written by a pre-sliding build) or a
                // marker holding the old `1` placeholder rather than a
                // birth instant: backfill the ceiling from now rather
                // than dropping data.
                let backfill: redis::RedisResult<()> = conn.set_ex(
                    &born_key,
                    Self::now_unix_seconds(),
                    self.born_key_ttl_seconds(),
                );
                if let Err(err) = backfill {
                    tracing::warn!(
                        target = "ccr.redis",
                        hash = %hash,
                        error = %err,
                        "ccr_redis_born_backfill_failed"
                    );
                }
                self.max_lifetime_seconds
            }
        };
        let new_ttl = self.default_ttl_seconds.min(remaining);
        if new_ttl == 0 {
            // Past the max lifetime: purge rather than serve a pinned
            // entry that should have died.
            let _: redis::RedisResult<()> = conn.del(&key);
            return None;
        }
        let rearm: redis::RedisResult<()> = conn.expire(&key, new_ttl as i64);
        if let Err(err) = rearm {
            tracing::warn!(
                target = "ccr.redis",
                hash = %hash,
                error = %err,
                "ccr_redis_ttl_rearm_failed"
            );
        }
        Some(payload)
    }

    fn len(&self) -> usize {
        // Redis has no efficient global count; we'd need to KEYS-scan
        // the prefix which is O(N) and not safe in production. The
        // CcrStore::len() contract is documented as "informational; used
        // by tests + telemetry" — return 0 here. Tests for the Redis
        // backend assert get/put behavior, not len().
        0
    }
}
