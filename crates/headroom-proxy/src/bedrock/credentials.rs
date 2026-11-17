//! AWS credentials for Bedrock SigV4 signing, refreshed before expiry.
//!
//! The `aws-config` default chain can return temporary credentials
//! (IMDS, ECS task role, SSO, assumed role). Holding one resolved
//! `Credentials` value for the life of the process means every request
//! after expiry is signed with a dead key. This source keeps the
//! provider and the last credentials it returned, and asks the
//! provider again once the cached value is within
//! [`REFRESH_AHEAD_SECS`] of its expiry (the same policy as
//! `vertex::adc`). Credentials without an expiry (long-lived access
//! keys, or the static test credentials) are cached for good.
//!
//! The provider returned by `SdkConfig::credentials_provider()` does
//! not cache on its own (the SDK caches in the client's identity
//! cache, which this proxy does not use), so the cache lives here.

use std::time::{Duration, SystemTime};

use aws_credential_types::provider::error::CredentialsError;
use aws_credential_types::provider::{ProvideCredentials, SharedCredentialsProvider};
use aws_credential_types::Credentials;
use tokio::sync::Mutex;

/// Refresh credentials this many seconds before their expiry.
pub const REFRESH_AHEAD_SECS: u64 = 60;

/// A credentials provider plus the last credentials it returned.
#[derive(Debug)]
pub struct BedrockCredentials {
    provider: SharedCredentialsProvider,
    cached: Mutex<Option<Credentials>>,
}

impl BedrockCredentials {
    pub fn new(provider: impl ProvideCredentials + 'static) -> Self {
        Self {
            provider: SharedCredentialsProvider::new(provider),
            cached: Mutex::new(None),
        }
    }

    /// Return credentials that are not within the refresh-ahead window
    /// of their expiry, asking the provider again when needed. The lock
    /// is held across the fetch so concurrent requests share one refresh.
    pub async fn get(&self) -> Result<Credentials, CredentialsError> {
        let mut guard = self.cached.lock().await;
        if let Some(c) = guard.as_ref() {
            if fresh(c) {
                return Ok(c.clone());
            }
        }
        let creds = self.provider.provide_credentials().await?;
        tracing::info!(
            event = "bedrock_credentials_refreshed",
            expires_in_secs = creds
                .expiry()
                .and_then(|e| e.duration_since(SystemTime::now()).ok())
                .map(|d| d.as_secs()),
        );
        *guard = Some(creds.clone());
        Ok(creds)
    }
}

fn fresh(c: &Credentials) -> bool {
    match c.expiry() {
        None => true,
        Some(expiry) => match expiry.duration_since(SystemTime::now()) {
            Ok(remaining) => remaining > Duration::from_secs(REFRESH_AHEAD_SECS),
            Err(_) => false,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aws_credential_types::provider::future;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// Returns `AKID_<n>` on the n-th call, expiring `lifetime` from now.
    #[derive(Debug)]
    struct CountingProvider {
        calls: Arc<AtomicUsize>,
        lifetime: Duration,
    }

    impl ProvideCredentials for CountingProvider {
        fn provide_credentials<'a>(&'a self) -> future::ProvideCredentials<'a>
        where
            Self: 'a,
        {
            let n = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            future::ProvideCredentials::ready(Ok(Credentials::new(
                format!("AKID_{n}"),
                "secret",
                Some("token".into()),
                Some(SystemTime::now() + self.lifetime),
                "counting",
            )))
        }
    }

    fn source(lifetime: Duration) -> (BedrockCredentials, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = CountingProvider {
            calls: calls.clone(),
            lifetime,
        };
        (BedrockCredentials::new(provider), calls)
    }

    #[tokio::test]
    async fn expiring_credentials_are_replaced_on_next_get() {
        // 1s of life is inside the 60s refresh-ahead window.
        let (src, calls) = source(Duration::from_secs(1));
        assert_eq!(src.get().await.unwrap().access_key_id(), "AKID_1");
        assert_eq!(src.get().await.unwrap().access_key_id(), "AKID_2");
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn long_lived_credentials_are_cached() {
        let (src, calls) = source(Duration::from_secs(3600));
        for _ in 0..3 {
            assert_eq!(src.get().await.unwrap().access_key_id(), "AKID_1");
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    /// Fails until `ready` is set, then returns long-lived credentials.
    #[derive(Debug)]
    struct LateProvider {
        ready: Arc<std::sync::atomic::AtomicBool>,
        calls: Arc<AtomicUsize>,
    }

    impl ProvideCredentials for LateProvider {
        fn provide_credentials<'a>(&'a self) -> future::ProvideCredentials<'a>
        where
            Self: 'a,
        {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.ready.load(Ordering::SeqCst) {
                future::ProvideCredentials::ready(Ok(Credentials::new(
                    "AKID_LATE",
                    "secret",
                    None,
                    None,
                    "late",
                )))
            } else {
                future::ProvideCredentials::ready(Err(CredentialsError::not_loaded("not yet")))
            }
        }
    }

    #[tokio::test]
    async fn failed_fetch_is_not_cached() {
        // A startup resolution that fails must not stick: main.rs keeps
        // the source either way and relies on the next get() asking again.
        let ready = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let calls = Arc::new(AtomicUsize::new(0));
        let src = BedrockCredentials::new(LateProvider {
            ready: ready.clone(),
            calls: calls.clone(),
        });
        assert!(src.get().await.is_err());
        ready.store(true, Ordering::SeqCst);
        assert_eq!(src.get().await.unwrap().access_key_id(), "AKID_LATE");
        assert_eq!(src.get().await.unwrap().access_key_id(), "AKID_LATE");
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn static_credentials_without_expiry_are_returned() {
        let src = BedrockCredentials::new(Credentials::new("AKID_STATIC", "s", None, None, "t"));
        assert_eq!(src.get().await.unwrap().access_key_id(), "AKID_STATIC");
        assert_eq!(src.get().await.unwrap().access_key_id(), "AKID_STATIC");
    }
}
