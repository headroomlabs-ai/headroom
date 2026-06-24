//! `DiffOffload` — wraps [`DiffCompressor`] as an [`OffloadTransform`].
//!
//! # Bloat heuristic — context-to-change ratio
//!
//! A unified diff is bloaty when most of its lines are context: a
//! 200-line diff that changes only 3 lines is 197 lines of noise.
//! The estimator walks the input, counts:
//!
//! * `change` lines — start with `+` or `-` (excluding the per-file
//!   `+++`/`---` headers).
//! * `context` lines — start with ` ` (a single space) inside any
//!   hunk.
//!
//! Score = `(context / (context + change) − normal_context_ratio) /
//!          (1 − normal_context_ratio)`, clamped to [0.0, 1.0]. Below
//! the configured `normal_context_ratio` the diff is dense enough that
//! offload isn't worth the retrieval cost (score 0); at 100% context
//! it scores 1.0.
//!
//! No regex — pure byte-prefix checks. Cost: O(n) over the input
//! (single pass, no allocations).
//!
//! # CCR persistence
//!
//! `DiffCompressor::compress_with_store` writes the original payload to
//! the orchestrator-supplied store under the same `cache_key` it embeds
//! in the wire marker. The trait contract is satisfied by the
//! compressor itself; no double-store hack needed at the wrapper level.
//! (Earlier revisions of this offload did the post-hoc store from here
//! because the compressor lacked a store parameter — that's been
//! upstreamed in the audit-cleanup PR.)
//!
//! [`DiffCompressor`]: crate::transforms::diff_compressor::DiffCompressor
//! [`OffloadTransform`]: crate::transforms::pipeline::traits::OffloadTransform

use crate::ccr::CcrStore;
use crate::transforms::diff_compressor::{DiffCompressor, DiffCompressorConfig};
use crate::transforms::pipeline::config::DiffBloatConfig;
use crate::transforms::pipeline::traits::{
    CompressionContext, OffloadOutput, OffloadTransform, TransformError,
};
use crate::transforms::ContentType;

const NAME: &str = "diff_offload";
/// Confidence is high — DiffCompressor has 20+ parity fixtures.
const CONFIDENCE: f32 = 0.85;

pub struct DiffOffload {
    compressor: DiffCompressor,
    bloat: DiffBloatConfig,
}

impl DiffOffload {
    pub fn new(bloat: DiffBloatConfig) -> Self {
        Self::with_compressor(DiffCompressor::new(DiffCompressorConfig::default()), bloat)
    }

    pub fn with_compressor(compressor: DiffCompressor, bloat: DiffBloatConfig) -> Self {
        Self { compressor, bloat }
    }
}

impl OffloadTransform for DiffOffload {
    fn name(&self) -> &'static str {
        NAME
    }

    fn applies_to(&self) -> &[ContentType] {
        &[ContentType::GitDiff]
    }

    fn estimate_bloat(&self, content: &str) -> f32 {
        if content.is_empty() {
            return 0.0;
        }
        let mut total_lines = 0usize;
        let mut change = 0usize;
        let mut context = 0usize;
        let mut in_hunk = false;

        for line in content.lines() {
            total_lines += 1;
            // Hunk header signals "the next ` `/`+`/`-` lines are part
            // of a real hunk." Bare prefix checks — no regex.
            if line.starts_with("@@") {
                in_hunk = true;
                continue;
            }
            // File-level headers can carry `+++`/`---` prefixes; those
            // aren't change lines. Reset hunk state on a new file
            // header (begins with `diff --git`).
            if line.starts_with("diff --git") {
                in_hunk = false;
                continue;
            }
            if line.starts_with("+++") || line.starts_with("---") {
                continue;
            }
            if !in_hunk {
                continue;
            }
            match line.as_bytes().first() {
                Some(b'+') | Some(b'-') => change += 1,
                Some(b' ') => context += 1,
                _ => {}
            }
        }

        if total_lines < self.bloat.min_lines {
            return 0.0;
        }
        let denom = (context + change) as f64;
        if denom == 0.0 {
            return 0.0;
        }
        let ratio = context as f64 / denom;
        let normal = self.bloat.normal_context_ratio;
        if ratio <= normal {
            return 0.0;
        }
        // Map (normal, 1.0] → (0, 1] linearly.
        let span = 1.0 - normal;
        if span <= 0.0 {
            // Degenerate config: treat any context as full bloat.
            return 1.0;
        }
        ((ratio - normal) / span).clamp(0.0, 1.0) as f32
    }

    fn apply(
        &self,
        content: &str,
        ctx: &CompressionContext,
        store: &dyn CcrStore,
    ) -> Result<OffloadOutput, TransformError> {
        // `compress_with_store` (added in the audit-cleanup PR) writes
        // the original to `store` under the same `cache_key` it embeds
        // in the marker. Earlier versions of this offload had to
        // double-store post-hoc because the compressor lacked a store
        // parameter — that hack is gone.
        let (result, _) = self
            .compressor
            .compress_with_store(content, &ctx.query, Some(store));

        let Some(key) = result.cache_key else {
            return Err(TransformError::skipped(
                NAME,
                "diff compressor did not emit a cache_key",
            ));
        };

        Ok(OffloadOutput::from_lengths(
            content.len(),
            result.compressed,
            key,
        ))
    }

    fn confidence(&self) -> f32 {
        CONFIDENCE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ccr::InMemoryCcrStore;
    use crate::transforms::pipeline::config::PipelineConfig;

    fn default_bloat() -> DiffBloatConfig {
        PipelineConfig::default().bloat.diff
    }

    fn offload() -> DiffOffload {
        DiffOffload::new(default_bloat())
    }

    fn build_diff(num_files: usize, context_per_file: usize, changes_per_file: usize) -> String {
        let mut s = String::new();
        for f in 0..num_files {
            s.push_str(&format!(
                "diff --git a/file{f}.txt b/file{f}.txt\n--- a/file{f}.txt\n+++ b/file{f}.txt\n@@ -1,{} +1,{} @@\n",
                context_per_file + changes_per_file,
                context_per_file + changes_per_file
            ));
            for _ in 0..context_per_file {
                s.push_str(" context line\n");
            }
            for c in 0..changes_per_file {
                s.push_str(&format!("-removed line {c}\n"));
                s.push_str(&format!("+added line {c}\n"));
            }
        }
        s
    }

    #[test]
    fn name_and_applies_to() {
        let o = offload();
        assert_eq!(o.name(), "diff_offload");
        assert_eq!(o.applies_to(), &[ContentType::GitDiff]);
    }

    #[test]
    fn estimate_bloat_empty_input_is_zero() {
        assert_eq!(offload().estimate_bloat(""), 0.0);
    }

    #[test]
    fn estimate_bloat_below_min_lines_is_zero() {
        let small = build_diff(1, 5, 1); // ~13 lines
        assert_eq!(offload().estimate_bloat(&small), 0.0);
    }

    #[test]
    fn estimate_bloat_dense_diff_scores_zero() {
        // Lots of changes, little context — below `normal_context_ratio=0.6`.
        let diff = build_diff(2, 5, 60); // 120 changes, 10 context
        let score = offload().estimate_bloat(&diff);
        assert_eq!(score, 0.0, "dense diff should score zero, got {score}");
    }

    #[test]
    fn estimate_bloat_context_heavy_diff_scores_high() {
        // Mostly context, few changes — far above 0.6.
        // 200 context lines, 5 changes each side = 200 / (200+10) ≈ 0.95.
        let diff = build_diff(1, 200, 5);
        let score = offload().estimate_bloat(&diff);
        assert!(
            score > 0.7,
            "context-heavy diff should score high, got {score}"
        );
    }

    #[test]
    fn estimate_bloat_at_threshold_scores_zero() {
        // Exactly normal_context_ratio worth of context — maps to 0.
        // Construct: 60 context, 20 changes-per-side (total 40 changes)
        // → context / (context+change) = 60/100 = 0.6 = threshold.
        let diff = build_diff(1, 60, 20);
        // Make sure it clears min_lines.
        let score = offload().estimate_bloat(&diff);
        assert_eq!(score, 0.0, "at threshold should be zero, got {score}");
    }

    #[test]
    fn estimate_bloat_safe_on_huge_inputs() {
        let diff = build_diff(50, 100, 50); // many files, big hunks
        let _ = offload().estimate_bloat(&diff);
    }

    #[test]
    fn apply_emits_key_and_persists_original() {
        let diff = build_diff(1, 200, 5);
        let store = InMemoryCcrStore::new();
        let r = offload()
            .apply(&diff, &CompressionContext::default(), &store)
            .expect("should compress");
        assert!(!r.cache_key.is_empty());
        // Bug-fix verification: DiffCompressor itself wouldn't have
        // stored, but our wrapper must.
        assert_eq!(store.get(&r.cache_key).as_deref(), Some(diff.as_str()));
    }

    #[test]
    fn apply_skipped_when_compressor_declines_ccr() {
        // Below DiffCompressor's `min_lines_for_ccr=50`.
        let diff = build_diff(1, 5, 2);
        let store = InMemoryCcrStore::new();
        let err = offload()
            .apply(&diff, &CompressionContext::default(), &store)
            .expect_err("must skip");
        match err {
            TransformError::Skipped { transform, .. } => assert_eq!(transform, "diff_offload"),
            _ => panic!("expected Skipped, got {err:?}"),
        }
        assert_eq!(store.len(), 0);
    }
}
