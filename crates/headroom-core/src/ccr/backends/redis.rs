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
//! payload bytes, with a `SETEX` TTL applied on every write. Read path
//! is a single `GET`. Redis handles purging via key expiry — no
//! application-side sweep needed (matching the SQLite backend's
//! lazy-purge but at the Redis level).
//!
//! # Concurrency
//!
//! `redis::Client` is `Send + Sync`; we hold one per store instance.
//! `get_connection` returns a fresh blocking connection per call; this
//! is the recommended pattern for short-lived puts/gets and avoids the
//! `MultiplexedConnection`'s tokio-runtime requirement (CCR is called
//! both from sync and tokio contexts in the proxy crate).

#![cfg(feature = "redis")]

use redis::Commands;

use crate::ccr::CcrStore;

/// Key prefix applied to every CCR entry. Configurable per-deployment
/// so multiple proxies sharing one Redis don't collide.
const DEFAULT_KEY_PREFIX: &str = "ccr";

/// Redis-backed CCR store. Cfg-gated behind `feature = "redis"`.
pub struct RedisCcrStore {
    client: redis::Client,
    key_prefix: String,
    default_ttl_seconds: u64,
}

impl RedisCcrStore {
    /// Open a Redis connection at `url` (e.g. `redis://127.0.0.1:6379`).
    /// Errors surface to the caller (`from_config`).
    pub fn open(url: &str, default_ttl_seconds: u64) -> redis::RedisResult<Self> {
        Self::open_with_prefix(url, DEFAULT_KEY_PREFIX.to_string(), default_ttl_seconds)
    }

    pub fn open_with_prefix(
        url: &str,
        key_prefix: String,
        default_ttl_seconds: u64,
    ) -> redis::RedisResult<Self> {
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
        })
    }

    fn key_for(&self, hash: &str) -> String {
        format!("{}:{}", self.key_prefix, hash)
    }

    /// Default TTL (seconds) applied on every `put`.
    pub fn default_ttl_seconds(&self) -> u64 {
        self.default_ttl_seconds
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
        match bytes {
            Ok(Some(bytes)) => String::from_utf8(bytes).ok(),
            Ok(None) => None,
            Err(err) => {
                tracing::warn!(
                    target = "ccr.redis",
                    hash = %hash,
                    error = %err,
                    "ccr_redis_get_failed"
                );
                None
            }
        }
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
