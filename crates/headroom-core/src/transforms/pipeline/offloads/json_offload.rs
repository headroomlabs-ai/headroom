//! `JsonOffload` — wraps [`SmartCrusher`] as an [`OffloadTransform`].
//!
//! # What this offload does
//!
//! JSON arrays of dicts (rows) are the worst tool-output shape for
//! token bloat: every row repeats its field names, and many tools
//! return long arrays with near-duplicate or low-information rows
//! (audit logs, search index dumps, paginated API results). SmartCrusher
//! is the existing Rust port that handles this — schema dedup, row
//! sampling, anchor-aware row selection, lossless tabular compaction —
//! and emits CCR markers for any rows it drops.
//!
//! `JsonOffload` plugs SmartCrusher into the pipeline's
//! [`OffloadTransform`] contract:
//!
//! 1. `estimate_bloat` is a CHEAP byte scan that spots the
//!    array-of-objects shape and counts row separators (`}` followed
//!    by `,` or `]`). No JSON parse — that's reserved for `apply`.
//! 2. `apply` delegates to `SmartCrusher::crush(content, ctx.query)`
//!    and adds a wrapper-level CCR marker. The orchestrator-supplied
//!    store gets the original payload under the wrapper hash.
//!
//! # Why a wrapper-level CCR hash on top of SmartCrusher's internal one
//!
//! SmartCrusher mints PER-SUB-ARRAY hashes (`<<ccr:abc 42_rows_offloaded>>`)
//! that resolve in its **internal** store. The pipeline orchestrator
//! has its **own** store (passed to `apply` via the trait). To honor
//! the trait contract — `cache_key` MUST resolve in the orchestrator's
//! store — `JsonOffload` hashes the WHOLE input and stashes it via the
//! orchestrator store under that hash, returning that hash as the
//! `cache_key`.
//!
//! For retrieval the LLM uses the wrapper's outer hash (recovers the
//! whole original JSON). SmartCrusher's per-array markers in the
//! compressed body are informational hints — not directly resolvable
//! through the pipeline-level store, but the LLM doesn't need them
//! when the wrapper hash gives back the full payload.
//!
//! # No regex
//!
//! Per project convention. Estimator is byte-prefix scan +
//! `str::matches` over fixed substrings. SmartCrusher itself uses
//! `serde_json` for parsing, never regex.
//!
//! [`SmartCrusher`]: crate::transforms::smart_crusher::SmartCrusher
//! [`OffloadTransform`]: crate::transforms::pipeline::traits::OffloadTransform

use md5::{Digest, Md5};

use crate::ccr::CcrStore;
use crate::transforms::pipeline::config::JsonOffloadConfig;
use crate::transforms::pipeline::traits::{
    CompressionContext, OffloadOutput, OffloadTransform, TransformError,
};
use crate::transforms::smart_crusher::{SmartCrusher, SmartCrusherConfig};
use crate::transforms::ContentType;

const NAME: &str = "json_offload";
/// SmartCrusher has 50+ parity fixtures and shadow-validated against
/// Python — high confidence.
const CONFIDENCE: f32 = 0.85;

pub struct JsonOffload {
    crusher: SmartCrusher,
    config: JsonOffloadConfig,
}

impl JsonOffload {
    /// Default constructor — builds SmartCrusher with the OSS default
    /// composition (scorer + constraints + compaction stage + internal
    /// CCR store). The internal store handles SmartCrusher's per-array
    /// markers; the wrapper still emits its own outer marker that
    /// resolves in the orchestrator-supplied store.
    pub fn new(config: JsonOffloadConfig) -> Self {
        Self {
            crusher: SmartCrusher::new(SmartCrusherConfig::default()),
            config,
        }
    }

    /// Custom constructor — used by tests that want a stubbed crusher
    /// or a custom SmartCrusher config.
    pub fn with_crusher(crusher: SmartCrusher, config: JsonOffloadConfig) -> Self {
        Self { crusher, config }
    }
}

impl OffloadTransform for JsonOffload {
    fn name(&self) -> &'static str {
        NAME
    }

    fn applies_to(&self) -> &[ContentType] {
        &[ContentType::JsonArray]
    }

    fn estimate_bloat(&self, content: &str) -> f32 {
        if content.is_empty() {
            return 0.0;
        }
        let trimmed = content.trim_start();
        if !trimmed.starts_with('[') {
            return 0.0;
        }
        // Cheap structural scan: count `},{` or `}, {` occurrences as
        // row boundaries. Doesn't parse JSON — that's reserved for
        // `apply`. Conservative: counts only adjacent-row patterns,
        // not the trailing `}` before `]` (so a 5-row array reports 4
        // separators, but for our threshold logic that rounds correctly).
        let separators = count_row_separators(content);
        if separators < self.config.min_array_rows.saturating_sub(1) {
            return 0.0;
        }
        let saturation = self.config.saturation_rows.saturating_sub(1).max(1);
        (separators as f32 / saturation as f32).clamp(0.0, 1.0)
    }

    fn apply(
        &self,
        content: &str,
        ctx: &CompressionContext,
        store: &dyn CcrStore,
    ) -> Result<OffloadOutput, TransformError> {
        let result = self.crusher.crush(content, &ctx.query, 0.0);
        if !result.was_modified {
            return Err(TransformError::skipped(
                NAME,
                "smart crusher returned passthrough",
            ));
        }
        if result.compressed.len() >= content.len() {
            return Err(TransformError::skipped(NAME, "no savings after crush"));
        }

        // Wrapper-level CCR: hash the WHOLE original input, stash it
        // through the orchestrator-supplied store, append a marker so
        // the LLM sees a resolvable handle. SmartCrusher's per-array
        // markers in `result.compressed` stay informational — the LLM
        // retrieves the full original via this outer hash.
        let key = md5_hex_24(content);
        store.put(&key, content);
        let mut output = result.compressed;
        output.push_str("\n[json_offload CCR: hash=");
        output.push_str(&key);
        output.push(']');

        Ok(OffloadOutput::from_lengths(content.len(), output, key))
    }

    fn confidence(&self) -> f32 {
        CONFIDENCE
    }
}

/// Count plausible row-boundary patterns in `content`. Used by the
/// bloat estimator as a cheap proxy for JSON array length. Patterns
/// counted (no overlap):
///
/// * `},{`  — compact JSON, no whitespace.
/// * `}, {` — pretty-printed JSON with single-space.
/// * `},\n` — pretty-printed JSON with newline-after-comma.
///
/// We don't attempt to be perfect — the estimator just needs an
/// order-of-magnitude signal. False positives on `},` strings inside
/// quoted values are tolerated; their rate is low and the eventual
/// `apply` does the rigorous JSON parse anyway.
fn count_row_separators(content: &str) -> usize {
    content.matches("},{").count()
        + content.matches("}, {").count()
        + content.matches("},\n").count()
}

/// MD5 of `content`, hex-encoded, truncated to 24 chars. Matches the
/// CCR convention used by other offload wrappers.
fn md5_hex_24(content: &str) -> String {
    let mut h = Md5::new();
    h.update(content.as_bytes());
    let digest = h.finalize();
    let mut hex = String::with_capacity(32);
    for b in digest.iter() {
        hex.push_str(&format!("{:02x}", b));
    }
    hex.truncate(24);
    hex
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ccr::InMemoryCcrStore;
    use crate::transforms::pipeline::config::PipelineConfig;

    fn cfg() -> JsonOffloadConfig {
        PipelineConfig::default().offload.json
    }

    fn offload() -> JsonOffload {
        JsonOffload::new(cfg())
    }

    /// Build a JSON array of N similar dicts with id + name + value.
    /// Compact JSON (no extra whitespace) so byte counts are predictable.
    fn build_tabular_array(n: usize) -> String {
        let mut s = String::from("[");
        for i in 0..n {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!(
                "{{\"id\":{},\"name\":\"item-{}\",\"value\":{}}}",
                i,
                i,
                i * 100
            ));
        }
        s.push(']');
        s
    }

    #[test]
    fn name_and_applies_to() {
        let o = offload();
        assert_eq!(o.name(), "json_offload");
        assert_eq!(o.applies_to(), &[ContentType::JsonArray]);
    }

    #[test]
    fn estimate_bloat_empty_input_zero() {
        assert_eq!(offload().estimate_bloat(""), 0.0);
    }

    #[test]
    fn estimate_bloat_non_array_input_zero() {
        // Object, not array.
        assert_eq!(offload().estimate_bloat(r#"{"a":1,"b":2}"#), 0.0);
        // Plain text.
        assert_eq!(offload().estimate_bloat("just words here"), 0.0);
        // Number.
        assert_eq!(offload().estimate_bloat("42"), 0.0);
    }

    #[test]
    fn estimate_bloat_below_min_rows_zero() {
        let arr = build_tabular_array(3); // 2 separators, default min_array_rows=5 → need ≥4
        assert_eq!(offload().estimate_bloat(&arr), 0.0);
    }

    #[test]
    fn estimate_bloat_at_saturation_is_one() {
        // 100 rows → 99 separators. saturation_rows=50 → 49 saturates.
        let arr = build_tabular_array(100);
        let score = offload().estimate_bloat(&arr);
        assert!(score >= 0.99, "expected ~1.0 at saturation, got {score}");
    }

    #[test]
    fn estimate_bloat_scales_linearly_in_middle_range() {
        // 25 rows → 24 separators. saturation=49 → score ≈ 0.49.
        let arr = build_tabular_array(25);
        let score = offload().estimate_bloat(&arr);
        assert!(
            (0.4..=0.6).contains(&score),
            "expected mid-range score, got {score}"
        );
    }

    #[test]
    fn estimate_bloat_handles_whitespace_after_array_open() {
        // Pretty-printed JSON: `[ {`, then `}, {`, etc.
        let mut s = String::from("[\n");
        for i in 0..30 {
            if i > 0 {
                s.push_str(",\n");
            }
            s.push_str(&format!(
                "  {{\"id\":{i},\"name\":\"item-{i}\",\"value\":{}}}",
                i * 100
            ));
        }
        s.push_str("\n]");
        let score = offload().estimate_bloat(&s);
        assert!(
            score > 0.4,
            "pretty-printed array should still score, got {score}"
        );
    }

    #[test]
    fn estimate_bloat_handles_huge_input_safely() {
        let arr = build_tabular_array(50_000);
        // Just must not panic; saturation handles this fine.
        let score = offload().estimate_bloat(&arr);
        assert!((0.99..=1.0).contains(&score));
    }

    #[test]
    fn apply_compresses_large_tabular_array_and_stores_original() {
        // Big enough that SmartCrusher will engage.
        let arr = build_tabular_array(500);
        let store = InMemoryCcrStore::new();
        let r = offload()
            .apply(&arr, &CompressionContext::default(), &store)
            .expect("smart crusher should compress");
        assert!(r.bytes_saved > 0);
        assert!(!r.cache_key.is_empty());
        // Wrapper marker present.
        assert!(r.output.contains("[json_offload CCR: hash="));
        // Original recoverable through the orchestrator's store.
        assert_eq!(store.get(&r.cache_key).as_deref(), Some(arr.as_str()));
    }

    #[test]
    fn apply_skipped_when_smart_crusher_passes_through() {
        // 2-row array: below SmartCrusher's `min_items_to_analyze` so
        // it returns passthrough — wrapper must surface that as Skipped.
        let arr = build_tabular_array(2);
        let store = InMemoryCcrStore::new();
        let err = offload()
            .apply(&arr, &CompressionContext::default(), &store)
            .expect_err("must skip");
        match err {
            TransformError::Skipped { transform, .. } => assert_eq!(transform, "json_offload"),
            _ => panic!("expected Skipped, got {err:?}"),
        }
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn apply_skipped_for_non_json_input() {
        let store = InMemoryCcrStore::new();
        let err = offload()
            .apply("not json at all", &CompressionContext::default(), &store)
            .expect_err("must skip non-json");
        match err {
            TransformError::Skipped { .. } => {}
            _ => panic!("expected Skipped, got {err:?}"),
        }
    }

    #[test]
    fn apply_propagates_query_anchors_into_smart_crusher() {
        // Construct a tabular array where one row has an anchor that
        // matches the query. SmartCrusher should be biased to keep
        // that row. We don't assert exactly what survives — just that
        // it ran with the query in scope.
        let mut s = String::from("[");
        for i in 0..50 {
            if i > 0 {
                s.push(',');
            }
            let name = if i == 17 {
                "needle".to_string()
            } else {
                format!("hay-{i}")
            };
            s.push_str(&format!(
                "{{\"id\":{i},\"name\":\"{name}\",\"score\":{}}}",
                i % 7
            ));
        }
        s.push(']');
        let store = InMemoryCcrStore::new();
        let ctx = CompressionContext::with_query("needle");
        let r = offload()
            .apply(&s, &ctx, &store)
            .expect("crusher should run");
        // Output should reference "needle" — SmartCrusher's anchor logic
        // should keep that row.
        assert!(
            r.output.contains("needle"),
            "anchor row should survive crush"
        );
    }

    #[test]
    fn cache_key_is_stable_across_calls_for_same_input() {
        let arr = build_tabular_array(100);
        let store_a = InMemoryCcrStore::new();
        let store_b = InMemoryCcrStore::new();
        let r_a = offload()
            .apply(&arr, &CompressionContext::default(), &store_a)
            .expect("ok");
        let r_b = offload()
            .apply(&arr, &CompressionContext::default(), &store_b)
            .expect("ok");
        assert_eq!(
            r_a.cache_key, r_b.cache_key,
            "cache_key should be a deterministic hash of input"
        );
    }

    #[test]
    fn count_row_separators_handles_compact_and_pretty() {
        assert_eq!(count_row_separators(""), 0);
        assert_eq!(count_row_separators("[]"), 0);
        assert_eq!(count_row_separators(r#"[{"a":1}]"#), 0);
        assert_eq!(count_row_separators(r#"[{"a":1},{"a":2}]"#), 1);
        assert_eq!(count_row_separators(r#"[{"a":1}, {"a":2}, {"a":3}]"#), 2);
    }
}
