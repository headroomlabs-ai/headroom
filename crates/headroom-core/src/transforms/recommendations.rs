//! Startup-time loader for `recommendations.toml` (PR-B5).
//!
//! # Why this module exists
//!
//! Pre-PR-B5, the live-zone dispatcher could call back into Python's
//! TOIN per request to get a [`CompressionHint`]. That coupling made
//! per-request output non-deterministic — same input could compress
//! differently across runs depending on TOIN's mutable state — which
//! broke prompt caching (P2-27, P5-56). PR-B5 retired the request-time
//! hint API; recommendations now flow through this loader at startup:
//!
//! 1. The Python `headroom.cli.toin_publish` CLI walks the on-disk TOIN
//!    store and emits `recommendations.toml`.
//! 2. The deploy pipeline ships that TOML alongside the Rust binary.
//! 3. At startup, [`RecommendationStore::load_default`] reads the file
//!    once and exposes the recommendations via a process-wide
//!    [`OnceLock`].
//! 4. [`get`] / [`RecommendationStore::lookup`] return the row matching
//!    `(auth_mode, model_family, structure_hash)`, or `None` when no
//!    advice was published. The dispatcher (PR-B3's
//!    `dispatch_compressor`) does **not** consume this surface yet —
//!    PR-F3 is responsible for wiring it.
//!
//! # File schema
//!
//! ```toml
//! [[recommendation]]
//! auth_mode = "payg"
//! model_family = "claude-3-5"
//! structure_hash = "deadbeef..."
//! strategy_hint = "smart_crusher"
//! confidence = 0.87
//! observations = 142
//! ```
//!
//! # Failure modes (loud, never silent)
//!
//! Per project memory `feedback_no_silent_fallbacks.md`: a missing or
//! malformed file degrades to "no advice, use static defaults" — but
//! the load attempt always logs a structured `tracing::warn!` event.
//! Production deployments grep for `event=recommendations_load_failed`
//! to catch a broken publish pipeline.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::Deserialize;

/// Environment variable that overrides the default `recommendations.toml`
/// path. The Rust proxy reads it once at startup; runtime changes do
/// not propagate.
pub const RECOMMENDATIONS_PATH_ENV_VAR: &str = "HEADROOM_RECOMMENDATIONS_PATH";

/// Default file the proxy looks at when the env var is unset.
const DEFAULT_RECOMMENDATIONS_PATH: &str = "./recommendations.toml";

// AuthMode is the canonical enum from `super::live_zone` (PR-B3).
// Re-exported here so recommendations callers can import it via
// `transforms::recommendations::AuthMode` without crossing module
// boundaries; the underlying enum is shared with the live-zone
// dispatcher to avoid drift between the dispatcher's auth slice and
// the published recommendations'. PR-B5 originally introduced its
// own copy; merged into the live-zone enum during integration so
// there's only one source of truth.
pub use super::live_zone::AuthMode;

/// A single published recommendation row.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct Recommendation {
    pub auth_mode: String,
    pub model_family: String,
    pub structure_hash: String,
    pub strategy_hint: String,
    pub confidence: f64,
    pub observations: u64,
}

/// Top-level TOML envelope: `[[recommendation]]` array.
#[derive(Debug, Default, Deserialize)]
struct RecommendationFile {
    #[serde(default)]
    recommendation: Vec<Recommendation>,
}

/// In-memory recommendation index, keyed by
/// `(auth_mode, model_family, structure_hash)`.
///
/// The keys are owned `String`s rather than `&str` — the values come
/// from `toml::from_str`, which gives us `String`s anyway, and trying
/// to borrow into the original buffer would require self-referential
/// storage. Recommendation files are small (≪ 1 MB even at large
/// fleets), so the allocation cost is irrelevant compared to the
/// parsing cost we already paid.
#[derive(Debug, Default, Clone)]
pub struct RecommendationStore {
    by_key: HashMap<(String, String, String), Recommendation>,
}

impl RecommendationStore {
    /// Build an empty store. Used for tests and as the fallback when
    /// no `recommendations.toml` is present.
    pub fn empty() -> Self {
        Self {
            by_key: HashMap::new(),
        }
    }

    /// Number of indexed rows.
    pub fn len(&self) -> usize {
        self.by_key.len()
    }

    /// Whether this store has zero recommendations.
    pub fn is_empty(&self) -> bool {
        self.by_key.is_empty()
    }

    /// Look up a recommendation by tenant slice + structure hash.
    /// Returns `None` when no advice was published for that key.
    pub fn lookup(
        &self,
        auth_mode: AuthMode,
        model_family: &str,
        structure_hash: &str,
    ) -> Option<&Recommendation> {
        // HashMap::get on a tuple key requires `Borrow` on tuples,
        // which Rust doesn't provide for mixed `&str`/`String` tuples.
        // Allocate a short-lived owned key — recommendation lookups
        // happen once per request at most, so this isn't hot.
        let key = (
            auth_mode.as_str().to_string(),
            model_family.to_string(),
            structure_hash.to_string(),
        );
        self.by_key.get(&key)
    }

    /// Parse a TOML string into a [`RecommendationStore`].
    pub fn from_toml_str(s: &str) -> Result<Self, RecommendationsError> {
        let parsed: RecommendationFile = toml::from_str(s).map_err(RecommendationsError::Parse)?;
        let mut by_key = HashMap::with_capacity(parsed.recommendation.len());
        for row in parsed.recommendation {
            let key = (
                row.auth_mode.clone(),
                row.model_family.clone(),
                row.structure_hash.clone(),
            );
            by_key.insert(key, row);
        }
        Ok(Self { by_key })
    }

    /// Read a TOML file from disk and parse it.
    ///
    /// Missing files yield [`RecommendationsError::Missing`] —
    /// callers usually downgrade that to "use defaults" without
    /// panicking. Malformed files surface [`RecommendationsError::Parse`].
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, RecommendationsError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                RecommendationsError::Missing(path.to_path_buf())
            } else {
                RecommendationsError::Io {
                    path: path.to_path_buf(),
                    source: e,
                }
            }
        })?;
        Self::from_toml_str(&text)
    }

    /// Best-effort load: returns an empty store and logs structured
    /// warnings when the file is missing or malformed. This is the
    /// path the Rust proxy uses at startup — it's not fatal for the
    /// publish pipeline to be down.
    pub fn load_or_empty(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref();
        match Self::from_file(path) {
            Ok(store) => {
                tracing::info!(
                    event = "recommendations_loaded",
                    path = %path.display(),
                    rows = store.len(),
                    "TOIN recommendations loaded",
                );
                store
            }
            Err(RecommendationsError::Missing(_)) => {
                tracing::info!(
                    event = "recommendations_missing",
                    path = %path.display(),
                    "no recommendations.toml present; using static defaults",
                );
                Self::empty()
            }
            Err(err) => {
                tracing::warn!(
                    event = "recommendations_load_failed",
                    path = %path.display(),
                    error = %err,
                    "TOIN recommendations failed to load — falling back to empty store",
                );
                Self::empty()
            }
        }
    }
}

/// Process-wide store populated at first call to [`load_default`].
static GLOBAL: OnceLock<RecommendationStore> = OnceLock::new();

/// Compute the path the loader will read.
///
/// Honors `HEADROOM_RECOMMENDATIONS_PATH` for prod overrides; falls
/// back to [`DEFAULT_RECOMMENDATIONS_PATH`].
pub fn default_path() -> PathBuf {
    std::env::var(RECOMMENDATIONS_PATH_ENV_VAR)
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_RECOMMENDATIONS_PATH))
}

/// Initialize and return the global [`RecommendationStore`].
///
/// On first call, reads the file at [`default_path`]; subsequent calls
/// return the cached store. Idempotent and thread-safe via [`OnceLock`].
pub fn load_default() -> &'static RecommendationStore {
    GLOBAL.get_or_init(|| RecommendationStore::load_or_empty(default_path()))
}

/// Module-level convenience: look up a recommendation in the global
/// store. PR-F3 will wire this into `dispatch_compressor`. PR-B5 only
/// exposes the API surface.
pub fn get(
    auth_mode: AuthMode,
    model: &str,
    structure_hash: &str,
) -> Option<&'static Recommendation> {
    load_default().lookup(auth_mode, model, structure_hash)
}

/// Errors surfaced by the loader. Marked non-exhaustive so we can add
/// future variants without breaking callers.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RecommendationsError {
    /// File doesn't exist on disk.
    #[error("recommendations file not found: {0}")]
    Missing(PathBuf),
    /// Filesystem error other than NotFound.
    #[error("recommendations IO error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// TOML parse failure (typed wrapper for ergonomics).
    #[error("recommendations TOML parse error: {0}")]
    Parse(#[from] toml::de::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_toml() -> &'static str {
        r#"
[[recommendation]]
auth_mode = "payg"
model_family = "claude-3-5"
structure_hash = "deadbeef"
strategy_hint = "smart_crusher"
confidence = 0.87
observations = 142

[[recommendation]]
auth_mode = "oauth"
model_family = "gpt-4o"
structure_hash = "cafebabe"
strategy_hint = "log_compressor"
confidence = 0.42
observations = 60
"#
    }

    #[test]
    fn from_toml_str_indexes_by_tuple_key() {
        let store = RecommendationStore::from_toml_str(sample_toml()).expect("parses");
        assert_eq!(store.len(), 2);

        let r = store
            .lookup(AuthMode::Payg, "claude-3-5", "deadbeef")
            .expect("hit");
        assert_eq!(r.strategy_hint, "smart_crusher");
        assert!((r.confidence - 0.87).abs() < 1e-9);
        assert_eq!(r.observations, 142);
    }

    #[test]
    fn lookup_returns_none_for_missing_slice() {
        let store = RecommendationStore::from_toml_str(sample_toml()).expect("parses");
        assert!(store
            .lookup(AuthMode::Unknown, "gpt-4o", "cafebabe")
            .is_none());
    }

    #[test]
    fn empty_store_lookup_is_none() {
        let store = RecommendationStore::empty();
        assert!(store.is_empty());
        assert!(store.lookup(AuthMode::Payg, "claude-3-5", "any").is_none());
    }

    #[test]
    fn malformed_toml_yields_parse_error() {
        let bad = "this is not valid toml [[\n\n";
        let err = RecommendationStore::from_toml_str(bad).unwrap_err();
        assert!(matches!(err, RecommendationsError::Parse(_)));
    }

    #[test]
    fn auth_mode_strings_match_python_publish_cli() {
        assert_eq!(AuthMode::Payg.as_str(), "payg");
        assert_eq!(AuthMode::OAuth.as_str(), "oauth");
        assert_eq!(AuthMode::Subscription.as_str(), "subscription");
        assert_eq!(AuthMode::Unknown.as_str(), "unknown");
    }
}
