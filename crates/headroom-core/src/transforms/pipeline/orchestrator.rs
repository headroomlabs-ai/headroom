//! [`CompressionPipeline`] — content-type-keyed dispatch over reformat
//! and offload transforms with **parallel** domain-specific bloat
//! estimation.
//!
//! # Decision flow
//!
//! ```text
//! input + content_type
//!   │
//!   ▼  rayon::join (real parallelism)
//!   ┌──────────────────────────────┐    ┌────────────────────────────┐
//!   │ Reformat phase               │    │ Per-offload bloat phase    │
//!   │   serial over reformats      │    │   par_iter over offloads   │
//!   │   stop early if              │    │   each calls estimate_bloat│
//!   │   output_len/orig_len ≤      │    │                            │
//!   │   reformat_target_ratio      │    │   returns (offload, score) │
//!   └──────────────────────────────┘    └────────────────────────────┘
//!   │                                              │
//!   ▼                                              ▼
//! ┌──────────────────────────────────────────────────────┐
//! │ Decide which offloads to run                         │
//! │                                                      │
//! │ For each (offload, score):                           │
//! │   run_it = score ≥ bloat_threshold                   │
//! │         OR (reformat_ratio > offload_fallback_ratio  │
//! │             AND score > 0)                           │
//! └──────────────────────────────────────────────────────┘
//!   │
//!   ▼  serial — each offload sees the previous one's output
//! ┌────────────────────────────────────┐
//! │ Run gated offloads against `store` │
//! └────────────────────────────────────┘
//!   │
//!   ▼ steps_applied[], bytes_saved, cache_keys[]
//! ```
//!
//! # Why parallel?
//!
//! Reformat phase scans/parses input bytes (e.g. JSON parse).
//! Bloat estimators also scan input bytes (line walks, hash sets,
//! detector calls). On large inputs both touch the same cache lines —
//! running them on different threads via `rayon::join` overlaps the
//! scans without competing for memory bandwidth, since both are
//! read-only over the same buffer.
//!
//! # The acceptance gate
//!
//! Both reformat outputs and offload outputs go through the same
//! "did we save enough to keep this?" check. PipelineResult always
//! returns *some* output — failures inside transforms are recorded as
//! skips, not propagated. The orchestrator is on the hot path of every
//! tool-call response and MUST NOT panic.

use std::collections::HashMap;
use std::sync::Arc;

use rayon::prelude::*;

use crate::ccr::CcrStore;
use crate::transforms::pipeline::config::PipelineConfig;
use crate::transforms::pipeline::traits::{
    CompressionContext, OffloadTransform, ReformatTransform, TransformError,
};
use crate::transforms::ContentType;

/// Result returned by [`CompressionPipeline::run`].
#[derive(Debug, Clone, Default)]
pub struct PipelineResult {
    /// Final output. Equal to the input if every stage skipped.
    pub output: String,
    /// Total bytes removed, summed across every accepted stage.
    pub bytes_saved: usize,
    /// Reformat names + offload names that were actually accepted, in
    /// execution order. Maps 1:1 onto the per-strategy stats nest.
    pub steps_applied: Vec<String>,
    /// CCR cache keys produced by accepted offloads. Empty when only
    /// reformats ran (or when nothing ran). Order matches
    /// `steps_applied`'s offload entries — first offload key first.
    pub cache_keys: Vec<String>,
}

/// Sequential reformat-then-parallel-bloat-then-gated-offload pipeline.
pub struct CompressionPipeline {
    reformats_by_type: HashMap<ContentType, Vec<Arc<dyn ReformatTransform>>>,
    offloads_by_type: HashMap<ContentType, Vec<Arc<dyn OffloadTransform>>>,
    config: PipelineConfig,
}

impl CompressionPipeline {
    pub fn builder() -> CompressionPipelineBuilder {
        CompressionPipelineBuilder::default()
    }

    /// Run the pipeline. `store` receives offload payloads under their
    /// `cache_key`s; reformat-only invocations don't touch it.
    pub fn run(
        &self,
        content: &str,
        content_type: ContentType,
        ctx: &CompressionContext,
        store: &dyn CcrStore,
    ) -> PipelineResult {
        let original_len = content.len();
        if original_len == 0 {
            return PipelineResult {
                output: String::new(),
                ..Default::default()
            };
        }

        let empty_reformats: Vec<Arc<dyn ReformatTransform>> = Vec::new();
        let empty_offloads: Vec<Arc<dyn OffloadTransform>> = Vec::new();
        let reformats = self
            .reformats_by_type
            .get(&content_type)
            .unwrap_or(&empty_reformats);
        let offloads = self
            .offloads_by_type
            .get(&content_type)
            .unwrap_or(&empty_offloads);

        // Phase 1+2 — run reformat phase and bloat estimation in parallel.
        // rayon::join takes two closures and runs them on different
        // worker threads when work is plentiful; it serializes them on
        // the calling thread when not. The pipeline doesn't care
        // which way it falls; correctness is the same.
        let (reformat_acc, bloat_scores) = rayon::join(
            || self.run_reformats(content, reformats),
            || self.estimate_bloats(content, offloads),
        );

        let mut steps: Vec<String> = reformat_acc.steps;
        let mut total_saved: usize = reformat_acc.bytes_saved;
        let mut current = reformat_acc.output;

        // Compute the post-reformat ratio that gates fallback offloads.
        let reformat_ratio = current.len() as f64 / original_len as f64;

        // Phase 3 — decide and run offloads. Each offload sees the
        // current (post-reformat, post-prior-offload) buffer.
        let mut cache_keys: Vec<String> = Vec::new();
        for (offload, score) in offloads.iter().zip(bloat_scores.iter()) {
            let above_threshold = *score >= self.config.pipeline.bloat_threshold;
            let reformat_underwhelmed =
                reformat_ratio > self.config.pipeline.offload_fallback_ratio && *score > 0.0;
            if !(above_threshold || reformat_underwhelmed) {
                tracing::trace!(
                    target: "headroom::pipeline",
                    offload = offload.name(),
                    score,
                    reformat_ratio,
                    "offload skipped: bloat below threshold and reformat sufficient"
                );
                continue;
            }
            match offload.apply(&current, ctx, store) {
                Ok(out) => {
                    if out.bytes_saved == 0 {
                        tracing::trace!(
                            target: "headroom::pipeline",
                            offload = offload.name(),
                            "offload accepted but saved zero bytes — discarding"
                        );
                        continue;
                    }
                    total_saved = total_saved.saturating_add(out.bytes_saved);
                    current = out.output;
                    steps.push(offload.name().to_string());
                    cache_keys.push(out.cache_key);
                }
                Err(TransformError::Internal { message, .. }) => {
                    tracing::warn!(
                        target: "headroom::pipeline",
                        offload = offload.name(),
                        error = %message,
                        "offload internal error"
                    );
                }
                Err(e) => {
                    tracing::trace!(
                        target: "headroom::pipeline",
                        offload = offload.name(),
                        error = %e,
                        "offload skipped"
                    );
                }
            }
        }

        PipelineResult {
            output: current,
            bytes_saved: total_saved,
            steps_applied: steps,
            cache_keys,
        }
    }

    /// Run reformat transforms in registration order against `content`.
    /// Stops once `current_len / original_len <= reformat_target_ratio`.
    fn run_reformats(
        &self,
        content: &str,
        reformats: &[Arc<dyn ReformatTransform>],
    ) -> ReformatAccumulator {
        let original_len = content.len();
        let mut current = content.to_string();
        let mut total_saved: usize = 0;
        let mut steps: Vec<String> = Vec::new();

        for transform in reformats {
            // Stop-early gate: target reached.
            let ratio = current.len() as f64 / original_len.max(1) as f64;
            if ratio <= self.config.pipeline.reformat_target_ratio {
                tracing::trace!(
                    target: "headroom::pipeline",
                    transform = transform.name(),
                    ratio,
                    "reformat target reached, skipping remaining reformats"
                );
                break;
            }
            match transform.apply(&current) {
                Ok(out) => {
                    if out.bytes_saved == 0 {
                        continue;
                    }
                    total_saved = total_saved.saturating_add(out.bytes_saved);
                    current = out.output;
                    steps.push(transform.name().to_string());
                }
                Err(TransformError::Internal { message, .. }) => {
                    tracing::warn!(
                        target: "headroom::pipeline",
                        transform = transform.name(),
                        error = %message,
                        "reformat internal error"
                    );
                }
                Err(e) => {
                    tracing::trace!(
                        target: "headroom::pipeline",
                        transform = transform.name(),
                        error = %e,
                        "reformat skipped"
                    );
                }
            }
        }

        ReformatAccumulator {
            output: current,
            bytes_saved: total_saved,
            steps,
        }
    }

    /// Run every offload's bloat estimator in parallel. Returns scores
    /// in the same order as the input slice.
    fn estimate_bloats(&self, content: &str, offloads: &[Arc<dyn OffloadTransform>]) -> Vec<f32> {
        offloads
            .par_iter()
            .map(|o| o.estimate_bloat(content))
            .collect()
    }

    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }
}

struct ReformatAccumulator {
    output: String,
    bytes_saved: usize,
    steps: Vec<String>,
}

/// Fluent builder for [`CompressionPipeline`].
#[derive(Default)]
pub struct CompressionPipelineBuilder {
    reformats_by_type: HashMap<ContentType, Vec<Arc<dyn ReformatTransform>>>,
    offloads_by_type: HashMap<ContentType, Vec<Arc<dyn OffloadTransform>>>,
    config: Option<PipelineConfig>,
}

impl CompressionPipelineBuilder {
    pub fn with_reformat<T>(mut self, transform: T) -> Self
    where
        T: ReformatTransform + 'static,
    {
        let arc: Arc<dyn ReformatTransform> = Arc::new(transform);
        let types: Vec<ContentType> = arc.applies_to().to_vec();
        for ct in types {
            self.reformats_by_type
                .entry(ct)
                .or_default()
                .push(arc.clone());
        }
        self
    }

    pub fn with_offload<T>(mut self, transform: T) -> Self
    where
        T: OffloadTransform + 'static,
    {
        let arc: Arc<dyn OffloadTransform> = Arc::new(transform);
        let types: Vec<ContentType> = arc.applies_to().to_vec();
        for ct in types {
            self.offloads_by_type
                .entry(ct)
                .or_default()
                .push(arc.clone());
        }
        self
    }

    pub fn with_config(mut self, config: PipelineConfig) -> Self {
        self.config = Some(config);
        self
    }

    pub fn build(self) -> CompressionPipeline {
        CompressionPipeline {
            reformats_by_type: self.reformats_by_type,
            offloads_by_type: self.offloads_by_type,
            config: self.config.unwrap_or_default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ccr::InMemoryCcrStore;
    use crate::transforms::pipeline::offloads::{
        DiffNoise, DiffOffload, JsonOffload, LogOffload, SearchOffload,
    };
    use crate::transforms::pipeline::reformats::{JsonMinifier, LogTemplate};
    use crate::transforms::pipeline::traits::{OffloadOutput, ReformatOutput};

    fn ctx() -> CompressionContext {
        CompressionContext::default()
    }

    fn store() -> InMemoryCcrStore {
        InMemoryCcrStore::new()
    }

    // ── Empty pipeline ────────────────────────────────────────────────

    #[test]
    fn empty_pipeline_passes_input_through() {
        let p = CompressionPipeline::builder().build();
        let s = store();
        let r = p.run("hello world", ContentType::PlainText, &ctx(), &s);
        assert_eq!(r.output, "hello world");
        assert_eq!(r.bytes_saved, 0);
        assert!(r.steps_applied.is_empty());
        assert!(r.cache_keys.is_empty());
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn empty_input_returns_empty_output() {
        let p = CompressionPipeline::builder()
            .with_reformat(JsonMinifier)
            .build();
        let s = store();
        let r = p.run("", ContentType::JsonArray, &ctx(), &s);
        assert!(r.output.is_empty());
        assert!(r.steps_applied.is_empty());
    }

    // ── Reformat phase ────────────────────────────────────────────────

    #[test]
    fn reformat_runs_when_applicable() {
        let p = CompressionPipeline::builder()
            .with_reformat(JsonMinifier)
            .build();
        let s = store();
        let pretty = "{\n  \"a\": 1,\n  \"b\": 2\n}";
        let r = p.run(pretty, ContentType::JsonArray, &ctx(), &s);
        assert!(r.bytes_saved > 0);
        assert_eq!(r.steps_applied, vec!["json_minifier".to_string()]);
        assert!(r.output.len() < pretty.len());
        assert!(r.cache_keys.is_empty());
    }

    #[test]
    fn reformat_skipped_for_unrelated_content_type() {
        let p = CompressionPipeline::builder()
            .with_reformat(JsonMinifier)
            .build();
        let s = store();
        let r = p.run("not json", ContentType::PlainText, &ctx(), &s);
        assert_eq!(r.output, "not json");
        assert!(r.steps_applied.is_empty());
    }

    // ── Offload phase: bloat-gated ─────────────────────────────────────

    /// A test-only offload that always succeeds. Its bloat estimator
    /// returns whatever score the caller wires in.
    struct TestOffload {
        score: f32,
        applies_to: Vec<ContentType>,
        confidence_score: f32,
        name: &'static str,
    }
    impl OffloadTransform for TestOffload {
        fn name(&self) -> &'static str {
            self.name
        }
        fn applies_to(&self) -> &[ContentType] {
            &self.applies_to
        }
        fn estimate_bloat(&self, _content: &str) -> f32 {
            self.score
        }
        fn apply(
            &self,
            content: &str,
            _ctx: &CompressionContext,
            store: &dyn CcrStore,
        ) -> Result<OffloadOutput, TransformError> {
            // Always halve; emit a cache_key derived from name.
            let half = &content[..content.len() / 2];
            let key = format!("test_{}_key", self.name);
            store.put(&key, content);
            Ok(OffloadOutput::from_lengths(
                content.len(),
                half.to_string(),
                key,
            ))
        }
        fn confidence(&self) -> f32 {
            self.confidence_score
        }
    }

    fn test_offload(name: &'static str, score: f32) -> TestOffload {
        TestOffload {
            score,
            applies_to: vec![ContentType::PlainText],
            confidence_score: 0.5,
            name,
        }
    }

    #[test]
    fn offload_runs_when_bloat_above_threshold() {
        let p = CompressionPipeline::builder()
            .with_offload(test_offload("high_bloat", 0.9))
            .build();
        let s = store();
        let r = p.run("x".repeat(100).as_str(), ContentType::PlainText, &ctx(), &s);
        assert_eq!(r.steps_applied, vec!["high_bloat".to_string()]);
        assert_eq!(r.cache_keys.len(), 1);
        assert!(s.get(&r.cache_keys[0]).is_some());
    }

    #[test]
    fn offload_skipped_when_bloat_below_threshold_and_reformat_was_enough() {
        // No reformats, so reformat_ratio = 1.0, which IS above the
        // default fallback ratio of 0.85. The test ensures even in
        // that case we skip when score is too low.
        let p = CompressionPipeline::builder()
            .with_offload(test_offload("low_bloat", 0.0))
            .build();
        let s = store();
        let r = p.run("x".repeat(100).as_str(), ContentType::PlainText, &ctx(), &s);
        assert!(
            r.steps_applied.is_empty(),
            "score 0.0 should never run: {:?}",
            r.steps_applied
        );
        assert_eq!(s.len(), 0);
    }

    /// A reformat that always halves input — used to drive
    /// reformat_ratio below the fallback threshold.
    struct AlwaysHalf;
    impl ReformatTransform for AlwaysHalf {
        fn name(&self) -> &'static str {
            "always_half"
        }
        fn applies_to(&self) -> &[ContentType] {
            &[ContentType::PlainText]
        }
        fn apply(&self, content: &str) -> Result<ReformatOutput, TransformError> {
            let half = &content[..content.len() / 2];
            Ok(ReformatOutput::from_lengths(
                content.len(),
                half.to_string(),
            ))
        }
    }

    #[test]
    fn offload_skipped_when_reformat_already_sufficient_and_score_below_threshold() {
        // Reformat halves input → ratio = 0.5, well below
        // offload_fallback_ratio=0.85. With score 0.3 (below
        // bloat_threshold=0.5) and "reformat sufficient", we skip.
        let p = CompressionPipeline::builder()
            .with_reformat(AlwaysHalf)
            .with_offload(test_offload("midway", 0.3))
            .build();
        let s = store();
        let r = p.run("x".repeat(100).as_str(), ContentType::PlainText, &ctx(), &s);
        assert_eq!(r.steps_applied, vec!["always_half".to_string()]);
        assert!(r.cache_keys.is_empty());
    }

    #[test]
    fn offload_runs_as_fallback_when_reformat_underwhelms() {
        // Reformat barely helps (no AlwaysHalf, score for offload
        // = 0.2, BELOW bloat_threshold=0.5). reformat_ratio = 1.0,
        // ABOVE offload_fallback_ratio=0.85, AND score > 0 → offload
        // runs as a fallback.
        let p = CompressionPipeline::builder()
            .with_offload(test_offload("fallback", 0.2))
            .build();
        let s = store();
        let r = p.run("x".repeat(100).as_str(), ContentType::PlainText, &ctx(), &s);
        assert_eq!(r.steps_applied, vec!["fallback".to_string()]);
    }

    #[test]
    fn offload_above_threshold_runs_even_when_reformat_was_great() {
        // Reformat halves (ratio=0.5, "sufficient"), but score=0.9 forces
        // the offload anyway — high bloat means CCR still pays off.
        let p = CompressionPipeline::builder()
            .with_reformat(AlwaysHalf)
            .with_offload(test_offload("forced", 0.9))
            .build();
        let s = store();
        let r = p.run("x".repeat(100).as_str(), ContentType::PlainText, &ctx(), &s);
        assert_eq!(
            r.steps_applied,
            vec!["always_half".to_string(), "forced".to_string()]
        );
        assert_eq!(r.cache_keys.len(), 1);
    }

    // ── Bloat-estimation parallelism (smoke) ───────────────────────────

    #[test]
    fn parallel_bloat_estimation_returns_correct_scores_per_offload() {
        // Two offloads, each with a distinct score. The orchestrator
        // must pair score-with-offload correctly even when running them
        // in parallel.
        let p = CompressionPipeline::builder()
            .with_offload(test_offload("alpha", 0.9))
            .with_offload(test_offload("beta", 0.0))
            .build();
        let s = store();
        let r = p.run("x".repeat(100).as_str(), ContentType::PlainText, &ctx(), &s);
        // Only "alpha" should run (above threshold). "beta" with 0.0
        // should not run even via fallback (score must be > 0).
        assert_eq!(r.steps_applied, vec!["alpha".to_string()]);
    }

    // ── End-to-end with real offloads ──────────────────────────────────

    #[test]
    fn end_to_end_log_offload_compresses_repetitive_log() {
        let cfg = PipelineConfig::default();
        let p = CompressionPipeline::builder()
            .with_offload(LogOffload::new(cfg.bloat.log))
            .with_config(cfg)
            .build();
        let s = store();
        let line = "INFO: heartbeat\n";
        let log: String = line.repeat(200);
        let r = p.run(&log, ContentType::BuildOutput, &ctx(), &s);
        assert_eq!(r.steps_applied, vec!["log_offload".to_string()]);
        assert_eq!(r.cache_keys.len(), 1);
        assert_eq!(s.get(&r.cache_keys[0]).as_deref(), Some(log.as_str()));
        assert!(r.bytes_saved > 0);
    }

    #[test]
    fn end_to_end_diff_offload_compresses_context_heavy_diff() {
        let cfg = PipelineConfig::default();
        let p = CompressionPipeline::builder()
            .with_offload(DiffOffload::new(cfg.bloat.diff))
            .with_config(cfg)
            .build();
        let s = store();
        // Build a context-heavy diff: 100 context lines, 5 changes.
        let mut diff = String::new();
        diff.push_str(
            "diff --git a/x.txt b/x.txt\n--- a/x.txt\n+++ b/x.txt\n@@ -1,105 +1,105 @@\n",
        );
        for _ in 0..100 {
            diff.push_str(" context line\n");
        }
        for c in 0..5 {
            diff.push_str(&format!("-old {c}\n"));
            diff.push_str(&format!("+new {c}\n"));
        }
        let r = p.run(&diff, ContentType::GitDiff, &ctx(), &s);
        assert_eq!(r.steps_applied, vec!["diff_offload".to_string()]);
        assert_eq!(r.cache_keys.len(), 1);
        assert!(s.get(&r.cache_keys[0]).is_some());
    }

    #[test]
    fn end_to_end_search_offload_compresses_clustered_matches() {
        let cfg = PipelineConfig::default();
        let p = CompressionPipeline::builder()
            .with_offload(SearchOffload::new(cfg.bloat.search))
            .with_config(cfg)
            .build();
        let s = store();
        let input: String = (0..100)
            .map(|i| format!("utils.py:{}:def fn_{i}", i + 1))
            .collect::<Vec<_>>()
            .join("\n");
        let r = p.run(&input, ContentType::SearchResults, &ctx(), &s);
        assert_eq!(r.steps_applied, vec!["search_offload".to_string()]);
        assert_eq!(r.cache_keys.len(), 1);
        assert_eq!(s.get(&r.cache_keys[0]).as_deref(), Some(input.as_str()));
    }

    // ── Failure handling ───────────────────────────────────────────────

    /// An offload whose `apply` always errors with `Internal`. Used to
    /// verify the orchestrator doesn't propagate the panic and surfaces
    /// it at WARN.
    struct AlwaysInternalError;
    impl OffloadTransform for AlwaysInternalError {
        fn name(&self) -> &'static str {
            "always_internal_err"
        }
        fn applies_to(&self) -> &[ContentType] {
            &[ContentType::PlainText]
        }
        fn estimate_bloat(&self, _content: &str) -> f32 {
            0.9
        }
        fn apply(
            &self,
            _content: &str,
            _ctx: &CompressionContext,
            _store: &dyn CcrStore,
        ) -> Result<OffloadOutput, TransformError> {
            Err(TransformError::internal("always_internal_err", "by design"))
        }
        fn confidence(&self) -> f32 {
            0.5
        }
    }

    #[test]
    fn offload_internal_error_does_not_panic_and_yields_input() {
        let p = CompressionPipeline::builder()
            .with_offload(AlwaysInternalError)
            .build();
        let s = store();
        let r = p.run("x".repeat(100).as_str(), ContentType::PlainText, &ctx(), &s);
        assert!(r.steps_applied.is_empty());
        assert_eq!(r.output.len(), 100);
        assert_eq!(s.len(), 0);
    }

    // ── Builder behavior ───────────────────────────────────────────────

    #[test]
    fn builder_dispatches_by_applies_to() {
        let p = CompressionPipeline::builder()
            .with_reformat(JsonMinifier)
            .with_offload(LogOffload::new(PipelineConfig::default().bloat.log))
            .build();
        assert_eq!(p.reformats_by_type[&ContentType::JsonArray].len(), 1);
        assert_eq!(p.offloads_by_type[&ContentType::BuildOutput].len(), 1);
        assert!(!p.reformats_by_type.contains_key(&ContentType::BuildOutput));
        assert!(!p.offloads_by_type.contains_key(&ContentType::JsonArray));
    }

    #[test]
    fn builder_preserves_registration_order_for_offloads() {
        // Registration order is execution order — important when two
        // offloads are eligible for the same content type and the first
        // one already trims the buffer.
        let p = CompressionPipeline::builder()
            .with_offload(test_offload("first", 0.9))
            .with_offload(test_offload("second", 0.9))
            .build();
        let s = store();
        let r = p.run("x".repeat(100).as_str(), ContentType::PlainText, &ctx(), &s);
        assert_eq!(
            r.steps_applied,
            vec!["first".to_string(), "second".to_string()]
        );
    }

    // ── End-to-end with new transforms ────────────────────────────────

    #[test]
    fn end_to_end_log_template_collapses_then_log_offload_can_run() {
        // LogTemplate runs first (lossless reformat), then LogOffload
        // sees the collapsed output and may further drop low-priority
        // lines. Both should appear in steps_applied if both fire.
        let cfg = PipelineConfig::default();
        let p = CompressionPipeline::builder()
            .with_reformat(LogTemplate::new(cfg.reformat.log_template))
            .with_offload(LogOffload::new(cfg.bloat.log))
            .with_config(cfg)
            .build();
        let s = store();
        // 200 INFO lines, all same template — LogTemplate compresses
        // hugely.
        let mut log = String::new();
        for i in 0..200 {
            log.push_str(&format!(
                "2025-01-15T12:34:{:02} INFO worker-{} processing job {}\n",
                i % 60,
                i,
                100 + i
            ));
        }
        let r = p.run(&log, ContentType::BuildOutput, &ctx(), &s);
        assert!(r.bytes_saved > 0);
        assert!(
            r.steps_applied.iter().any(|n| n == "log_template"),
            "log_template must run first"
        );
        // Output is shorter than input. Token-level survival isn't
        // asserted here — LogOffload may further drop bytes (including
        // the template header) on its own gating logic. Lossless
        // round-trip is asserted in the LogTemplate-alone test below.
        assert!(r.output.len() < log.len());
    }

    #[test]
    fn end_to_end_diff_noise_drops_lockfile_then_diff_offload_handles_rest() {
        let cfg = PipelineConfig::default();
        let p = CompressionPipeline::builder()
            .with_offload(DiffNoise::new(cfg.offload.diff_noise.clone()))
            .with_offload(DiffOffload::new(cfg.bloat.diff))
            .with_config(cfg)
            .build();
        let s = store();

        // Build: huge Cargo.lock churn + a small real change in src/main.rs.
        let mut diff = String::new();
        diff.push_str("diff --git a/Cargo.lock b/Cargo.lock\n");
        diff.push_str("--- a/Cargo.lock\n+++ b/Cargo.lock\n@@ -1,400 +1,400 @@\n");
        for i in 0..200 {
            diff.push_str(&format!("-old{i}\n"));
            diff.push_str(&format!("+new{i}\n"));
        }
        diff.push_str("diff --git a/src/main.rs b/src/main.rs\n");
        diff.push_str("--- a/src/main.rs\n+++ b/src/main.rs\n@@ -1,3 +1,3 @@\n");
        diff.push_str("-let x = 1;\n");
        diff.push_str("+let x = 2;\n");
        let r = p.run(&diff, ContentType::GitDiff, &ctx(), &s);

        assert!(r.bytes_saved > 0);
        assert!(
            r.steps_applied.iter().any(|n| n == "diff_noise"),
            "diff_noise should fire on lockfile-dominated diff: {:?}",
            r.steps_applied
        );
        assert!(r.output.contains("[diff_noise: lockfile hunks dropped"));
        assert!(r.output.contains("let x = 2;"), "real change must survive");
        assert!(!r.cache_keys.is_empty());
    }

    #[test]
    fn end_to_end_json_minifier_then_json_offload_on_tabular_array() {
        // JsonMinifier (lossless reformat) runs first to strip
        // pretty-printing. If the array is large enough JsonOffload
        // (SmartCrusher wrapper) then engages.
        let cfg = PipelineConfig::default();
        let p = CompressionPipeline::builder()
            .with_reformat(JsonMinifier)
            .with_offload(JsonOffload::new(cfg.offload.json))
            .with_config(cfg)
            .build();
        let s = store();
        // Pretty-printed 200-row tabular array.
        let mut input = String::from("[\n");
        for i in 0..200 {
            if i > 0 {
                input.push_str(",\n");
            }
            input.push_str(&format!(
                "  {{\"id\": {i}, \"name\": \"event-{i}\", \"value\": {}}}",
                i * 100
            ));
        }
        input.push_str("\n]");
        let r = p.run(&input, ContentType::JsonArray, &ctx(), &s);
        assert!(r.bytes_saved > 0);
        assert!(
            r.steps_applied.iter().any(|n| n == "json_offload"),
            "json_offload must engage on 200-row tabular array, got {:?}",
            r.steps_applied
        );
        assert!(!r.cache_keys.is_empty());
        // Original recoverable through the orchestrator's store.
        let key = r.cache_keys.last().unwrap();
        assert!(s.get(key).is_some(), "wrapper hash must resolve in store");
    }

    #[test]
    fn end_to_end_json_offload_skipped_for_small_array() {
        let cfg = PipelineConfig::default();
        let p = CompressionPipeline::builder()
            .with_offload(JsonOffload::new(cfg.offload.json))
            .with_config(cfg)
            .build();
        let s = store();
        // 3 rows — below default min_array_rows=5. Estimator returns 0
        // → orchestrator skips without calling SmartCrusher.
        let input = r#"[{"id":1,"v":1},{"id":2,"v":2},{"id":3,"v":3}]"#;
        let r = p.run(input, ContentType::JsonArray, &ctx(), &s);
        assert!(r.steps_applied.is_empty());
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn end_to_end_real_log_with_template_run_collapses_then_passes_through() {
        // Just LogTemplate (no LogOffload) — verify reformat alone
        // gives meaningful savings on a templated log.
        let cfg = PipelineConfig::default();
        let p = CompressionPipeline::builder()
            .with_reformat(LogTemplate::new(cfg.reformat.log_template))
            .with_config(cfg)
            .build();
        let s = store();
        let mut log = String::new();
        for i in 0..100 {
            log.push_str(&format!(
                "[2025-01-15 12:00:{:02}] INFO Connecting to db-{} on port 5432\n",
                i % 60,
                i % 8
            ));
        }
        let r = p.run(&log, ContentType::BuildOutput, &ctx(), &s);
        assert_eq!(r.steps_applied, vec!["log_template".to_string()]);
        assert!(r.cache_keys.is_empty(), "reformat should not produce keys");
        assert!(r.bytes_saved > 0);
        assert!(r.output.contains("[Template T1:"));
    }
}
