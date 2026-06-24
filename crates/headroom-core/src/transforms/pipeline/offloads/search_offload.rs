//! `SearchOffload` — wraps [`SearchCompressor`] as an [`OffloadTransform`].
//!
//! # Status: not registered in the default pipeline (2026-04-30)
//!
//! Modern coding agents (Claude Code, Codex, etc.) drive `rg` / `grep`
//! with sensible scoping, so broad noisy search output is rare in
//! practice. When it does happen, the LLM benefits from seeing the
//! match clustering directly — compressing it adds maintenance burden
//! for marginal token savings. The type stays accessible at
//! `crate::transforms::pipeline::offloads::search_offload::SearchOffload`
//! for callers who want to opt in, but the orchestrator's default wiring
//! omits it. Re-evaluate once usage telemetry shows real demand.
//!
//! # Bloat heuristic — match clustering
//!
//! Search output (grep / ripgrep) is bloaty when matches cluster
//! heavily into a few files: 50 hits in `utils.py` and one hit
//! elsewhere is mostly redundant — the LLM only needs a representative
//! sample plus a count. The estimator computes:
//!
//! 1. `total` — count of lines that look like a `file:line:` match.
//! 2. `unique_files` — distinct file prefixes among those matches.
//! 3. `avg = total / unique_files`.
//! 4. `score = (avg − 1) / cluster_threshold`, clamped to [0.0, 1.0].
//!
//! `cluster_threshold = 10.0` means "10 matches per file on average is
//! 100% bloat" (the offload should fire). Below `min_matches`, score 0
//! regardless — too small to bother with retrieval round trip.
//!
//! No regex — pure byte scan: walk lines, find the first colon (or
//! Windows drive-letter colon) that's followed by digits and another
//! colon, treat what's before as the file. Cost: O(n) over input.
//!
//! [`SearchCompressor`]: crate::transforms::search_compressor::SearchCompressor
//! [`OffloadTransform`]: crate::transforms::pipeline::traits::OffloadTransform

use std::collections::HashSet;

use crate::ccr::CcrStore;
use crate::transforms::pipeline::config::SearchBloatConfig;
use crate::transforms::pipeline::traits::{
    CompressionContext, OffloadOutput, OffloadTransform, TransformError,
};
use crate::transforms::search_compressor::{SearchCompressor, SearchCompressorConfig};
use crate::transforms::ContentType;

const NAME: &str = "search_offload";
/// Confidence is high — SearchCompressor has parity fixtures.
const CONFIDENCE: f32 = 0.85;

pub struct SearchOffload {
    compressor: SearchCompressor,
    bloat: SearchBloatConfig,
    /// Bias for the underlying compressor's adaptive sizer.
    bias: f64,
}

impl SearchOffload {
    pub fn new(bloat: SearchBloatConfig) -> Self {
        Self::with_compressor(
            SearchCompressor::new(SearchCompressorConfig::default()),
            bloat,
        )
    }

    pub fn with_compressor(compressor: SearchCompressor, bloat: SearchBloatConfig) -> Self {
        Self {
            compressor,
            bloat,
            bias: 0.0,
        }
    }

    pub fn with_bias(mut self, bias: f64) -> Self {
        self.bias = bias;
        self
    }
}

impl OffloadTransform for SearchOffload {
    fn name(&self) -> &'static str {
        NAME
    }

    fn applies_to(&self) -> &[ContentType] {
        &[ContentType::SearchResults]
    }

    fn estimate_bloat(&self, content: &str) -> f32 {
        if content.is_empty() {
            return 0.0;
        }
        let mut total = 0usize;
        let mut files: HashSet<&str> = HashSet::new();

        for line in content.lines() {
            if let Some(file) = extract_file_prefix(line) {
                total += 1;
                files.insert(file);
            }
        }

        if total < self.bloat.min_matches || files.is_empty() {
            return 0.0;
        }
        let avg = total as f32 / files.len() as f32;
        if avg <= 1.0 {
            return 0.0;
        }
        let score = (avg - 1.0) / self.bloat.cluster_threshold;
        score.clamp(0.0, 1.0)
    }

    fn apply(
        &self,
        content: &str,
        ctx: &CompressionContext,
        store: &dyn CcrStore,
    ) -> Result<OffloadOutput, TransformError> {
        let (result, stats) =
            self.compressor
                .compress_with_store(content, &ctx.query, self.bias, Some(store));

        let Some(key) = result.cache_key else {
            let reason = stats.ccr_skip_reason.unwrap_or("no cache_key emitted");
            return Err(TransformError::skipped(NAME, reason));
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

/// Extract the file prefix from a single grep-style match line. Returns
/// `None` if the line doesn't look like a match. Handles:
///
/// * `path:42:content` — standard `grep -n`
/// * `path-42-content` — ripgrep context lines (a separate non-match
///   indicator; we treat them as matches for clustering purposes since
///   they belong to the same file)
/// * `C:\path:42:content` — Windows-style drive-prefixed paths (skip
///   the drive colon when scanning for the line-number marker)
///
/// No regex — manual byte scan. Returns the file prefix as a borrowed
/// `&str` over the input.
fn extract_file_prefix(line: &str) -> Option<&str> {
    let bytes = line.as_bytes();
    if bytes.is_empty() {
        return None;
    }
    // Skip a Windows-style drive prefix (`C:` or `c:`).
    let scan_start =
        if bytes.len() >= 2 && bytes[1] == b':' && (bytes[0] as char).is_ascii_alphabetic() {
            2
        } else {
            0
        };

    let mut i = scan_start;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b':' || b == b'-' {
            // Saw a separator. Need the next chars to be digits, then
            // another matching separator.
            let sep = b;
            let mut j = i + 1;
            let digit_start = j;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            if j > digit_start && j < bytes.len() && bytes[j] == sep {
                // Found `<file><sep><digits><sep>`. File prefix is [0..i].
                return Some(&line[..i]);
            }
        }
        i += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ccr::InMemoryCcrStore;
    use crate::transforms::pipeline::config::PipelineConfig;

    fn default_bloat() -> SearchBloatConfig {
        PipelineConfig::default().bloat.search
    }

    fn offload() -> SearchOffload {
        SearchOffload::new(default_bloat())
    }

    #[test]
    fn name_and_applies_to() {
        let o = offload();
        assert_eq!(o.name(), "search_offload");
        assert_eq!(o.applies_to(), &[ContentType::SearchResults]);
    }

    #[test]
    fn extract_file_prefix_handles_grep() {
        assert_eq!(
            extract_file_prefix("src/utils.py:42:def foo():"),
            Some("src/utils.py")
        );
    }

    #[test]
    fn extract_file_prefix_handles_ripgrep_context() {
        assert_eq!(
            extract_file_prefix("src/main.py-43-some context"),
            Some("src/main.py")
        );
    }

    #[test]
    fn extract_file_prefix_handles_dashed_filenames() {
        assert_eq!(
            extract_file_prefix("pre-commit-config.yaml:7:line"),
            Some("pre-commit-config.yaml")
        );
    }

    #[test]
    fn extract_file_prefix_handles_windows_paths() {
        assert_eq!(
            extract_file_prefix(r"C:\Users\foo\bar.py:42:line"),
            Some(r"C:\Users\foo\bar.py")
        );
    }

    #[test]
    fn extract_file_prefix_rejects_non_matches() {
        assert_eq!(extract_file_prefix(""), None);
        assert_eq!(extract_file_prefix("just some text"), None);
        assert_eq!(extract_file_prefix("file:notdigits:content"), None);
    }

    #[test]
    fn estimate_bloat_empty_is_zero() {
        assert_eq!(offload().estimate_bloat(""), 0.0);
    }

    #[test]
    fn estimate_bloat_below_min_matches_is_zero() {
        let s = "a.py:1:x\nb.py:2:y\nc.py:3:z";
        assert_eq!(offload().estimate_bloat(s), 0.0);
    }

    #[test]
    fn estimate_bloat_clustered_matches_score_high() {
        // 100 matches in a single file — avg 100/1 = 100. Above
        // cluster_threshold=10 → score saturates at 1.0.
        let s: String = (0..100)
            .map(|i| format!("utils.py:{}:line", i + 1))
            .collect::<Vec<_>>()
            .join("\n");
        let score = offload().estimate_bloat(&s);
        assert!(score > 0.9, "expected high score, got {score}");
    }

    #[test]
    fn estimate_bloat_distributed_matches_score_low() {
        // 20 matches across 20 files — avg 1.0. Score should be 0.
        let s: String = (0..20)
            .map(|i| format!("file{i}.py:1:line"))
            .collect::<Vec<_>>()
            .join("\n");
        let score = offload().estimate_bloat(&s);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn estimate_bloat_moderate_clustering() {
        // 30 matches across 5 files — avg 6.0. Score = (6-1)/10 = 0.5.
        let mut s = String::new();
        for f in 0..5 {
            for line in 0..6 {
                s.push_str(&format!("file{f}.py:{}:line\n", line + 1));
            }
        }
        let score = offload().estimate_bloat(&s);
        assert!((0.4..=0.6).contains(&score), "expected ~0.5, got {score}");
    }

    #[test]
    fn estimate_bloat_safe_on_huge_inputs() {
        let s: String = (0..50_000)
            .map(|i| format!("file{}.py:1:line", i % 100))
            .collect::<Vec<_>>()
            .join("\n");
        let _ = offload().estimate_bloat(&s);
    }

    #[test]
    fn apply_emits_cache_key_and_stores_original_for_clustered_input() {
        let s: String = (0..100)
            .map(|i| format!("utils.py:{}:def fn_{i}", i + 1))
            .collect::<Vec<_>>()
            .join("\n");
        let store = InMemoryCcrStore::new();
        let r = offload()
            .apply(&s, &CompressionContext::default(), &store)
            .expect("offload should produce a key on clustered input");
        assert!(!r.cache_key.is_empty());
        assert_eq!(store.get(&r.cache_key).as_deref(), Some(s.as_str()));
    }

    #[test]
    fn apply_skipped_when_compressor_declines_ccr() {
        // Single match — far below SearchCompressor's threshold.
        let s = "only.py:1:trivial";
        let store = InMemoryCcrStore::new();
        let err = offload()
            .apply(s, &CompressionContext::default(), &store)
            .expect_err("must skip");
        match err {
            TransformError::Skipped { transform, .. } => assert_eq!(transform, "search_offload"),
            _ => panic!("expected Skipped, got {err:?}"),
        }
    }
}
