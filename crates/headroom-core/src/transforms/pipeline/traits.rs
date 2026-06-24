//! Compression pipeline traits — `Reformat` and `Offload`.
//!
//! # Why two traits, both lossless w.r.t. information
//!
//! With CCR (Compress-Cache-Retrieve), no transform in this pipeline
//! destroys information. Bytes drop from the wire, but the original
//! payload is stashed in a [`CcrStore`] keyed by a hash; the LLM can
//! retrieve any dropped piece via a tool call. So calling transforms
//! "lossy" misnames the architecture — they all preserve information,
//! they just differ in *how* they shrink the wire output.
//!
//! Two distinct mechanisms, two traits:
//!
//! * [`ReformatTransform`] — pack denser without dropping anything.
//!   Output bytes, when read, are semantically equivalent to the
//!   input. Examples: `JsonMinifier` (whitespace), log RLE
//!   deduplication, code-comment stripping, schema extraction.
//!   **No CCR needed** — the LLM doesn't retrieve anything; the wire
//!   output carries everything already.
//!
//! * [`OffloadTransform`] — drop bytes from the wire, stash the
//!   original via a [`CcrStore`], emit a retrieval marker. Examples:
//!   line-importance filtering, diff hunk sampling, search match
//!   thinning. **CCR is required** — the trait method takes
//!   `&dyn CcrStore`, and `OffloadOutput::cache_key` is `String`
//!   (not `Option<String>`) so the contract is type-system-enforced.
//!
//! # Per-domain bloat estimation
//!
//! Different content shapes have different "bloat" signals — a
//! generic byte-redundancy heuristic (zlib, etc.) misses domain
//! semantics. So [`OffloadTransform`] carries an [`estimate_bloat`]
//! method that runs a CHEAP structural read and returns a 0.0–1.0
//! score representing how much THIS transform would benefit the
//! input. The orchestrator gates `apply` on `estimate_bloat`
//! clearing a configurable threshold, and runs estimates in parallel
//! with the reformat phase via `rayon::join`.
//!
//! [`estimate_bloat`]: OffloadTransform::estimate_bloat

use crate::ccr::CcrStore;
use crate::transforms::ContentType;

/// Errors a transform can return.
///
/// All three variants signal "skip this transform, continue the
/// pipeline" — the orchestrator never panics on a transform error.
/// `Internal` surfaces to logs at WARN; the others at TRACE.
#[derive(Debug, thiserror::Error)]
pub enum TransformError {
    /// The transform couldn't parse the input. Caller skips it.
    #[error("invalid input for {transform}: {message}")]
    InvalidInput {
        transform: &'static str,
        message: String,
    },
    /// Ran cleanly, found nothing to do (empty input, content
    /// already minimal). Caller skips silently.
    #[error("{transform} skipped: {message}")]
    Skipped {
        transform: &'static str,
        message: String,
    },
    /// Internal failure (serializer, store write error, logic bug).
    /// Caller surfaces to WARN logs but continues.
    #[error("{transform} internal error: {message}")]
    Internal {
        transform: &'static str,
        message: String,
    },
}

impl TransformError {
    pub fn invalid_input(transform: &'static str, message: impl Into<String>) -> Self {
        Self::InvalidInput {
            transform,
            message: message.into(),
        }
    }
    pub fn skipped(transform: &'static str, message: impl Into<String>) -> Self {
        Self::Skipped {
            transform,
            message: message.into(),
        }
    }
    pub fn internal(transform: &'static str, message: impl Into<String>) -> Self {
        Self::Internal {
            transform,
            message: message.into(),
        }
    }
}

/// Result of a [`ReformatTransform`] — output bytes are semantically
/// equivalent to the input. No CCR handle, no tool call required.
#[derive(Debug, Clone)]
pub struct ReformatOutput {
    pub output: String,
    pub bytes_saved: usize,
}

impl ReformatOutput {
    pub fn from_lengths(input_len: usize, output: String) -> Self {
        Self {
            bytes_saved: input_len.saturating_sub(output.len()),
            output,
        }
    }
}

/// Result of an [`OffloadTransform`] — output bytes are a SUBSET of
/// the input, the original is in the supplied store, and `cache_key`
/// is the lookup handle. Required, not optional.
#[derive(Debug, Clone)]
pub struct OffloadOutput {
    pub output: String,
    pub bytes_saved: usize,
    /// Cache key under which the original payload is stored.
    /// Required by trait contract.
    pub cache_key: String,
}

impl OffloadOutput {
    pub fn from_lengths(input_len: usize, output: String, cache_key: String) -> Self {
        Self {
            bytes_saved: input_len.saturating_sub(output.len()),
            output,
            cache_key,
        }
    }
}

/// Per-call context the orchestrator passes to each transform.
#[derive(Debug, Default, Clone)]
pub struct CompressionContext {
    /// User question for relevance scoring inside offload transforms.
    pub query: String,
    /// Token budget the orchestrator is targeting (None = no budget
    /// signal; transforms apply their default aggressiveness).
    pub token_budget: Option<usize>,
}

impl CompressionContext {
    pub fn with_query(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            token_budget: None,
        }
    }
    pub fn with_budget(token_budget: usize) -> Self {
        Self {
            query: String::new(),
            token_budget: Some(token_budget),
        }
    }
}

/// A transform that packs the input denser without dropping
/// information. The orchestrator runs reformats first because they
/// don't need any CCR backing — surviving bytes round-trip
/// semantically.
pub trait ReformatTransform: Send + Sync {
    /// Stable telemetry name (lowercase snake_case, no spaces).
    /// Used as the strategy key in the per-strategy stats nest from
    /// Phase 3e.0.
    fn name(&self) -> &'static str;

    /// Content types this transform accepts. Borrowed from `&self`
    /// so impls can return either a `&'static` literal or a
    /// runtime-configured slice.
    fn applies_to(&self) -> &[ContentType];

    /// Run the transform.
    fn apply(&self, content: &str) -> Result<ReformatOutput, TransformError>;
}

/// A transform that drops bytes from the wire and stashes the
/// original via CCR. The trait carries a cheap, domain-specific
/// [`estimate_bloat`] method so the orchestrator can decide whether
/// running the full `apply` is worthwhile — the estimate is the
/// gating signal.
///
/// Trait contract:
/// 1. `estimate_bloat` returns a 0.0–1.0 score. It MUST be cheap
///    (structural-only, no full compression pass) and MUST be safe
///    to call on any input including the empty string.
/// 2. If the orchestrator calls `apply`, it has already gated on
///    `estimate_bloat ≥ threshold`. `apply` MUST emit a `cache_key`
///    on success — return `Err(Skipped)` if the implementation
///    decides post-hoc that the offload wasn't worth it.
/// 3. `apply` MUST stash the payload in `store` before returning,
///    and the returned `cache_key` MUST resolve in that store.
///
/// [`estimate_bloat`]: Self::estimate_bloat
pub trait OffloadTransform: Send + Sync {
    /// Stable telemetry name (lowercase snake_case).
    fn name(&self) -> &'static str;

    /// Content types this transform accepts.
    fn applies_to(&self) -> &[ContentType];

    /// Cheap structural estimate of bloat in `content`, scoped to
    /// THIS transform's domain. 0.0 = nothing here would benefit
    /// from offload; 1.0 = entire input is offloadable. Runs in
    /// parallel with the reformat phase via `rayon::join` — keep
    /// it under O(n) on input length, no allocations beyond what's
    /// needed for the structural read.
    ///
    /// MUST be safe on empty input (returns 0.0 by convention).
    fn estimate_bloat(&self, content: &str) -> f32;

    /// Run the offload. The orchestrator only calls this when
    /// `estimate_bloat(content) ≥ threshold`. On success, the
    /// returned `cache_key` MUST resolve to the original payload
    /// in `store`.
    fn apply(
        &self,
        content: &str,
        ctx: &CompressionContext,
        store: &dyn CcrStore,
    ) -> Result<OffloadOutput, TransformError>;

    /// Calibrated 0.0–1.0 quality score for telemetry. Future PR4
    /// may use it to select between competing offload transforms.
    fn confidence(&self) -> f32;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ccr::InMemoryCcrStore;

    pub struct TestReformat;
    impl ReformatTransform for TestReformat {
        fn name(&self) -> &'static str {
            "test_reformat"
        }
        fn applies_to(&self) -> &[ContentType] {
            &[ContentType::PlainText]
        }
        fn apply(&self, content: &str) -> Result<ReformatOutput, TransformError> {
            Ok(ReformatOutput::from_lengths(
                content.len(),
                content.to_string(),
            ))
        }
    }

    pub struct TestOffload {
        bloat: f32,
    }
    impl OffloadTransform for TestOffload {
        fn name(&self) -> &'static str {
            "test_offload"
        }
        fn applies_to(&self) -> &[ContentType] {
            &[ContentType::PlainText]
        }
        fn estimate_bloat(&self, _content: &str) -> f32 {
            self.bloat
        }
        fn apply(
            &self,
            content: &str,
            _ctx: &CompressionContext,
            store: &dyn CcrStore,
        ) -> Result<OffloadOutput, TransformError> {
            let key = format!("test_key_{:024x}", content.len());
            store.put(&key, content);
            Ok(OffloadOutput::from_lengths(
                content.len(),
                content.to_string(),
                key,
            ))
        }
        fn confidence(&self) -> f32 {
            0.5
        }
    }

    #[test]
    fn reformat_output_clamps_negative_savings_to_zero() {
        let r = ReformatOutput::from_lengths(10, "this is much longer than 10 bytes".into());
        assert_eq!(r.bytes_saved, 0);
    }

    #[test]
    fn offload_output_clamps_negative_savings_to_zero() {
        let r = OffloadOutput::from_lengths(10, "this is much longer".into(), "k".into());
        assert_eq!(r.bytes_saved, 0);
    }

    #[test]
    fn transform_error_messages_round_trip() {
        let e = TransformError::invalid_input("json_minifier", "bad token at line 3");
        let msg = e.to_string();
        assert!(msg.contains("json_minifier"));
        assert!(msg.contains("bad token at line 3"));
    }

    #[test]
    fn compression_context_constructors() {
        let q = CompressionContext::with_query("find errors");
        assert_eq!(q.query, "find errors");
        assert_eq!(q.token_budget, None);

        let b = CompressionContext::with_budget(2048);
        assert!(b.query.is_empty());
        assert_eq!(b.token_budget, Some(2048));
    }

    #[test]
    fn reformat_trait_smoke() {
        let t = TestReformat;
        let r = t.apply("hello").expect("reformat passes through");
        assert_eq!(r.output, "hello");
        assert_eq!(r.bytes_saved, 0);
    }

    #[test]
    fn offload_trait_writes_to_store_and_returns_required_cache_key() {
        let store = InMemoryCcrStore::new();
        let t = TestOffload { bloat: 0.9 };
        let r = t
            .apply("hello", &CompressionContext::default(), &store)
            .expect("offload writes");
        // Trait contract: cache_key is required and resolves in the store.
        assert!(!r.cache_key.is_empty());
        assert_eq!(store.get(&r.cache_key).as_deref(), Some("hello"));
    }

    #[test]
    fn offload_estimate_bloat_is_safe_on_empty_input() {
        let t = TestOffload { bloat: 0.0 };
        // Should not panic.
        let _score = t.estimate_bloat("");
    }
}
