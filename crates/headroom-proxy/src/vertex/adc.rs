//! GCP Application Default Credentials → bearer token resolution.
//!
//! Vertex AI's `:rawPredict` / `:streamRawPredict` endpoints expect
//! `Authorization: Bearer <jwt>` where the JWT is a short-lived
//! Google-signed access token. The token is obtained from the ADC
//! provider chain (gcloud user creds, GCE/GKE metadata server,
//! service-account JSON via `GOOGLE_APPLICATION_CREDENTIALS`,
//! workload-identity federation, etc.) and refreshed before expiry.
//!
//! # Why an abstraction (not direct `gcp_auth::provider()`)?
//!
//! 1. **Tests must NOT hit real GCP.** Per project rule "no silent
//!    fallbacks", we cannot dummy out the real provider in tests
//!    silently — we need a mock implementation that is explicit and
//!    distinct from production. The `TokenSource` trait gives us
//!    exactly two impls: production (`GcpAdcTokenSource`) and tests
//!    (`StaticTokenSource`).
//! 2. **Caching is policy.** `gcp_auth` exposes per-call token fetches.
//!    The cache + refresh-ahead-of-expiry policy lives at this layer
//!    so it's testable and tunable independent of the provider.
//!
//! # Refresh policy
//!
//! Tokens are refreshed when their remaining lifetime drops below
//! [`REFRESH_AHEAD_SECS`] (60s). That avoids the cliff-failure where
//! a request fired right at expiry races the upstream's clock.
//! `gcp_auth` itself caches internally, but we still wrap it so:
//!
//! - Tests can substitute a `StaticTokenSource`.
//! - We control the refresh-ahead window (gcp_auth's internal default
//!   is implementation-defined and could change).
//! - We emit structured `event = "vertex_adc_token_refreshed"` logs
//!   per refresh so operators can confirm the cache is live.
//!
//! # Failure mode
//!
//! When ADC fetch fails (no creds configured, metadata server
//! unreachable, IAM permission denied, etc.) the handler converts the
//! returned [`TokenSourceError`] to a structured 5xx response. We do
//! NOT silently forward without a token — per project rule
//! "no silent fallbacks", an unauthenticated forward to Vertex would
//! return a 401 from upstream that's harder to debug than our own
//! 502 with `event = "vertex_adc_fetch_failed"`.

use std::sync::Arc;
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use thiserror::Error;
use tokio::sync::Mutex;

/// Refresh tokens this many seconds before their expiry to avoid the
/// cliff-failure race. 60s comfortably covers a slow upstream and
/// clock skew.
pub const REFRESH_AHEAD_SECS: u64 = 60;

/// Default OAuth2 scope for Vertex / cloud-platform calls. The
/// `cloud-platform` scope is the broadest and is what gcloud emits
/// for ADC by default.
pub const DEFAULT_VERTEX_SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform";

/// Errors fetching / refreshing an ADC bearer token. Surfaced
/// verbatim by the handler as structured log events plus an HTTP 5xx
/// response.
#[derive(Debug, Error)]
pub enum TokenSourceError {
    /// `gcp_auth` failed to resolve any provider in the ADC chain.
    /// Common cause: developer never ran `gcloud auth application-default
    /// login` and no service-account JSON is in scope.
    #[error("gcp ADC provider initialization failed: {0}")]
    ProviderInit(String),
    /// The provider was resolved but `.token(scopes)` failed.
    #[error("gcp ADC token fetch failed: {0}")]
    Fetch(String),
}

/// A source of bearer tokens for Vertex calls. Production uses
/// [`GcpAdcTokenSource`]; tests use [`StaticTokenSource`]. There is
/// NO blanket impl — every concrete impl must be explicit (no silent
/// fallback to a "default token").
#[async_trait]
pub trait TokenSource: Send + Sync + std::fmt::Debug {
    /// Return a non-empty bearer token suitable for the
    /// `Authorization: Bearer <token>` header. The token is cached
    /// internally; concurrent calls de-dup to a single fetch.
    async fn bearer(&self) -> Result<String, TokenSourceError>;
}

/// Token + expiry pair held in the cache.
#[derive(Debug, Clone)]
struct CachedToken {
    token: String,
    expires_at: SystemTime,
}

impl CachedToken {
    /// `true` when the token has more than `REFRESH_AHEAD_SECS` of
    /// life left.
    fn fresh(&self) -> bool {
        match self.expires_at.duration_since(SystemTime::now()) {
            Ok(remaining) => remaining > Duration::from_secs(REFRESH_AHEAD_SECS),
            // `expires_at` already past now → not fresh.
            Err(_) => false,
        }
    }
}

/// Production token source backed by `gcp_auth`'s default ADC chain.
///
/// `gcp_auth::provider()` resolves lazily; the first `.bearer()` call
/// triggers the resolution and the result is memoized. Subsequent
/// calls re-use the provider and the cached token until its expiry
/// approaches.
pub struct GcpAdcTokenSource {
    /// Configured OAuth scope. Defaults to `cloud-platform`.
    scope: String,
    /// Lazily-initialized provider. We wrap in a `Mutex<Option<...>>`
    /// (rather than `OnceCell`) because the provider initialization
    /// is fallible and we want the next call after a transient
    /// failure to retry — not lock the cell to a permanent error.
    provider: Mutex<Option<Arc<dyn ::gcp_auth::TokenProvider>>>,
    /// Cached token + expiry. `Mutex` is fine here: the critical
    /// section (compare expiry, optionally refresh) is sub-microsecond
    /// in the cache-hit case and the refresh-miss path is rate-limited
    /// by the upstream metadata server anyway.
    cached: Mutex<Option<CachedToken>>,
}

impl std::fmt::Debug for GcpAdcTokenSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GcpAdcTokenSource")
            .field("scope", &self.scope)
            .field(
                "provider_initialized",
                &self
                    .provider
                    .try_lock()
                    .map(|g| g.is_some())
                    .unwrap_or(false),
            )
            .finish()
    }
}

impl GcpAdcTokenSource {
    /// Construct with the default `cloud-platform` scope. The
    /// provider is NOT resolved here — we defer until the first
    /// `.bearer()` call. That keeps proxy startup cheap when the
    /// operator hasn't actually wired a Vertex route yet.
    pub fn new() -> Self {
        Self::with_scope(DEFAULT_VERTEX_SCOPE)
    }

    /// Construct with an explicit scope. Used by tests / by operators
    /// who want a narrower scope than `cloud-platform`.
    pub fn with_scope(scope: impl Into<String>) -> Self {
        Self {
            scope: scope.into(),
            provider: Mutex::new(None),
            cached: Mutex::new(None),
        }
    }

    /// Resolve the provider lazily. On success, stores it for re-use
    /// and returns a clone of the `Arc`. On failure, returns the
    /// error verbatim — the next call retries (no permanent lock).
    async fn ensure_provider(
        &self,
    ) -> Result<Arc<dyn ::gcp_auth::TokenProvider>, TokenSourceError> {
        let mut guard = self.provider.lock().await;
        if let Some(p) = guard.as_ref() {
            return Ok(p.clone());
        }
        let provider = ::gcp_auth::provider()
            .await
            .map_err(|e| TokenSourceError::ProviderInit(e.to_string()))?;
        *guard = Some(provider.clone());
        Ok(provider)
    }
}

impl Default for GcpAdcTokenSource {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TokenSource for GcpAdcTokenSource {
    async fn bearer(&self) -> Result<String, TokenSourceError> {
        // Fast path: cached + fresh.
        {
            let guard = self.cached.lock().await;
            if let Some(c) = guard.as_ref() {
                if c.fresh() {
                    return Ok(c.token.clone());
                }
            }
        }

        // Slow path: refresh. We re-take the lock around the fetch
        // so concurrent callers serialize on a single network round
        // trip (rather than thundering-herd the metadata server).
        let mut guard = self.cached.lock().await;
        // Double-check inside the lock — another waiter may have
        // refreshed while we were queued.
        if let Some(c) = guard.as_ref() {
            if c.fresh() {
                return Ok(c.token.clone());
            }
        }

        let provider = self.ensure_provider().await?;
        let scopes = [self.scope.as_str()];
        let token = provider
            .token(&scopes)
            .await
            .map_err(|e| TokenSourceError::Fetch(e.to_string()))?;
        let token_str = token.as_str().to_string();
        // `gcp_auth::Token::expires_at()` returns a `chrono::DateTime<Utc>`.
        // Convert to `SystemTime` via the UNIX timestamp; `chrono` clamps
        // to its valid range so this never panics. Negative timestamps
        // (pre-epoch) are not legal for token expiries; if encountered we
        // treat the token as already expired so the next call refreshes.
        let expires_at = {
            let dt = token.expires_at();
            let unix_ts = dt.timestamp();
            if unix_ts < 0 {
                tracing::warn!(
                    event = "vertex_adc_token_negative_expiry",
                    expires_at_unix = unix_ts,
                    "gcp_auth token expires_at is pre-epoch; treating as already-expired"
                );
                SystemTime::UNIX_EPOCH
            } else {
                SystemTime::UNIX_EPOCH + Duration::from_secs(unix_ts as u64)
            }
        };
        let lifetime_remaining = expires_at
            .duration_since(SystemTime::now())
            .map(|d| d.as_secs())
            .unwrap_or(0);
        tracing::info!(
            event = "vertex_adc_token_refreshed",
            scope = %self.scope,
            lifetime_remaining_secs = lifetime_remaining,
            "vertex ADC bearer token refreshed"
        );
        *guard = Some(CachedToken {
            token: token_str.clone(),
            expires_at,
        });
        Ok(token_str)
    }
}

/// Static-token mock for tests. Production callers MUST NOT
/// instantiate this — there is no fallback to a default value, and
/// the test harness wires it explicitly.
#[derive(Debug, Clone)]
pub struct StaticTokenSource {
    token: String,
}

impl StaticTokenSource {
    pub fn new(token: impl Into<String>) -> Self {
        Self {
            token: token.into(),
        }
    }
}

#[async_trait]
impl TokenSource for StaticTokenSource {
    async fn bearer(&self) -> Result<String, TokenSourceError> {
        Ok(self.token.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn static_token_source_returns_token() {
        let src = StaticTokenSource::new("test-bearer-abc123");
        let t = src.bearer().await.expect("bearer");
        assert_eq!(t, "test-bearer-abc123");
        // Multiple calls return the same value.
        let t2 = src.bearer().await.expect("bearer 2");
        assert_eq!(t2, "test-bearer-abc123");
    }

    #[test]
    fn cached_token_freshness_window() {
        let now = SystemTime::now();
        let fresh = CachedToken {
            token: "x".into(),
            expires_at: now + Duration::from_secs(REFRESH_AHEAD_SECS + 30),
        };
        assert!(fresh.fresh());

        let stale = CachedToken {
            token: "x".into(),
            expires_at: now + Duration::from_secs(REFRESH_AHEAD_SECS - 1),
        };
        assert!(!stale.fresh());

        let already_expired = CachedToken {
            token: "x".into(),
            expires_at: now - Duration::from_secs(1),
        };
        assert!(!already_expired.fresh());
    }
}
