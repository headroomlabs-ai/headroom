//! Hybrid BM25 + Embedding relevance scorer with adaptive alpha tuning.
//!
//! Direct port of `headroom/relevance/hybrid.py`. Combines keyword
//! matching (BM25) with semantic similarity (embeddings) by weighted
//! linear fusion:
//!
//!   combined = alpha * BM25 + (1 - alpha) * Embedding
//!
//! When `adaptive=true` (the default), `alpha` is tuned per-query
//! based on regex pattern detection in the context: queries with
//! UUIDs, numeric IDs, hostnames, or emails get a higher BM25 weight
//! (exact match matters more than semantic match for those cases).
//!
//! # Graceful BM25 fallback
//!
//! When the embedding scorer reports `is_available() == false` (which
//! is currently always — the Rust ONNX backend is stubbed), the hybrid
//! scorer falls back to **BM25 with a small score boost**:
//!
//! - Items with any matched term get `score >= 0.3`.
//! - Items with two or more matched terms get `+0.2`, capped at 1.0.
//!
//! This compensates for BM25's known weakness on single-term matches
//! (BM25 of "Alice" against `{"name": "Alice"}` is ~0.07 raw, well
//! below typical relevance thresholds). The boost ensures the
//! fallback path keeps roughly the right items even without semantic
//! understanding. Pinned to Python's `score`/`score_batch` boost
//! rules.
//!
//! When the real ONNX `EmbeddingScorer` lands, this module needs zero
//! changes — the fallback path automatically goes dormant the moment
//! `embedding.is_available()` flips to `true`.

use std::sync::LazyLock;

use regex::Regex;

use super::base::{RelevanceScore, RelevanceScorer};
use super::bm25::BM25Scorer;
use super::embedding::EmbeddingScorer;

// Regex patterns that indicate exact-match is important.
// Translated literally from Python `hybrid.py:53-60`. The `[A-Z|a-z]`
// in the email pattern is a Python typo — `|` inside `[...]` becomes
// a literal pipe character, not alternation. We mirror that quirk
// faithfully for parity.

static UUID_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
        .expect("UUID regex compiles")
});

static NUMERIC_ID_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b\d{4,}\b").expect("numeric ID regex compiles"));

static HOSTNAME_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b[a-zA-Z0-9][-a-zA-Z0-9]*\.[a-zA-Z0-9][-a-zA-Z0-9]*(?:\.[a-zA-Z]{2,})?\b")
        .expect("hostname regex compiles")
});

static EMAIL_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    // Python: r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    // The `|` inside `[A-Z|a-z]` is a literal pipe in Python — keep
    // for byte-for-byte parity even though it's a Python source bug.
    Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
        .expect("email regex compiles")
});

pub struct HybridScorer {
    pub base_alpha: f64,
    pub adaptive: bool,
    pub bm25: BM25Scorer,
    pub embedding: EmbeddingScorer,
    /// Cached at construction time so we don't repeatedly call
    /// `embedding.is_available()`. Mirrors Python's
    /// `self._embedding_available` field.
    embedding_available: bool,
}

impl Default for HybridScorer {
    fn default() -> Self {
        let embedding = EmbeddingScorer::default();
        let embedding_available = embedding.is_available();
        HybridScorer {
            base_alpha: 0.5,
            adaptive: true,
            bm25: BM25Scorer::default(),
            embedding,
            embedding_available,
        }
    }
}

impl HybridScorer {
    pub fn new(alpha: f64, adaptive: bool) -> Self {
        let embedding = EmbeddingScorer::default();
        let embedding_available = embedding.is_available();
        HybridScorer {
            base_alpha: alpha,
            adaptive,
            bm25: BM25Scorer::default(),
            embedding,
            embedding_available,
        }
    }

    pub fn with_scorers(
        alpha: f64,
        adaptive: bool,
        bm25: BM25Scorer,
        embedding: EmbeddingScorer,
    ) -> Self {
        let embedding_available = embedding.is_available();
        HybridScorer {
            base_alpha: alpha,
            adaptive,
            bm25,
            embedding,
            embedding_available,
        }
    }

    pub fn has_embedding_support(&self) -> bool {
        self.embedding_available
    }

    /// Compute the per-query alpha. Mirrors Python `_compute_alpha`
    /// (`hybrid.py:115-151`). Returns `[0.3, 0.9]`-clamped.
    fn compute_alpha(&self, context: &str) -> f64 {
        if !self.adaptive {
            return self.base_alpha;
        }
        let context_lower = context.to_lowercase();

        let uuid_count = UUID_PATTERN.find_iter(context).count();
        let id_count = NUMERIC_ID_PATTERN.find_iter(context).count();
        let hostname_count = HOSTNAME_PATTERN.find_iter(&context_lower).count();
        let email_count = EMAIL_PATTERN.find_iter(&context_lower).count();

        let mut alpha = self.base_alpha;
        if uuid_count > 0 {
            alpha = alpha.max(0.85);
        } else if id_count >= 2 {
            alpha = alpha.max(0.75);
        } else if id_count == 1 {
            alpha = alpha.max(0.65);
        } else if hostname_count > 0 || email_count > 0 {
            alpha = alpha.max(0.6);
        }

        alpha.clamp(0.3, 0.9)
    }

    /// BM25-only fallback boost. Mirrors Python `score`/`score_batch`
    /// boost rules (`hybrid.py:171-181` and `225-238`).
    fn boost_bm25_only(&self, bm25_result: &RelevanceScore) -> RelevanceScore {
        let mut boosted = bm25_result.score;
        if !bm25_result.matched_terms.is_empty() {
            boosted = boosted.max(0.3);
            if bm25_result.matched_terms.len() >= 2 {
                boosted = (boosted + 0.2).min(1.0);
            }
        }
        RelevanceScore::new(
            boosted,
            format!("Hybrid (BM25 only, boosted): {}", bm25_result.reason),
            bm25_result.matched_terms.clone(),
        )
    }
}

impl RelevanceScorer for HybridScorer {
    fn score(&self, item: &str, context: &str) -> RelevanceScore {
        let bm25_result = self.bm25.score(item, context);

        if !self.embedding_available {
            return self.boost_bm25_only(&bm25_result);
        }

        let emb_result = self.embedding.score(item, context);
        let alpha = self.compute_alpha(context);
        let combined = alpha * bm25_result.score + (1.0 - alpha) * emb_result.score;

        RelevanceScore::new(
            combined,
            format!(
                "Hybrid (\u{3b1}={:.2}): BM25={:.2}, Semantic={:.2}",
                alpha, bm25_result.score, emb_result.score
            ),
            bm25_result.matched_terms,
        )
    }

    fn score_batch(&self, items: &[&str], context: &str) -> Vec<RelevanceScore> {
        if items.is_empty() {
            return Vec::new();
        }

        let bm25_results = self.bm25.score_batch(items, context);

        if !self.embedding_available {
            return bm25_results
                .iter()
                .map(|r| self.boost_bm25_only(r))
                .collect();
        }

        let emb_results = self.embedding.score_batch(items, context);
        let alpha = self.compute_alpha(context);

        bm25_results
            .into_iter()
            .zip(emb_results)
            .map(|(bm25_r, emb_r)| {
                let combined = alpha * bm25_r.score + (1.0 - alpha) * emb_r.score;
                RelevanceScore::new(
                    combined,
                    format!(
                        "Hybrid (\u{3b1}={:.2}): BM25={:.2}, Emb={:.2}",
                        alpha, bm25_r.score, emb_r.score
                    ),
                    bm25_r.matched_terms,
                )
            })
            .collect()
    }

    fn is_available(&self) -> bool {
        // Hybrid is always available — falls back to BM25 if needed.
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scorer() -> HybridScorer {
        HybridScorer::default()
    }

    #[test]
    fn hybrid_always_available() {
        assert!(scorer().is_available());
    }

    #[test]
    fn hybrid_reports_no_embedding_support_when_stubbed() {
        // Until the real ONNX scorer lands, hybrid runs in fallback
        // mode by default.
        assert!(!scorer().has_embedding_support());
    }

    #[test]
    fn fallback_score_boosts_single_match_to_at_least_0_3() {
        // BM25 alone often gives ~0.07 for a single-term match.
        // Fallback ensures the item clears 0.3.
        let s = scorer();
        let r = s.score(r#"{"name": "alice"}"#, "alice");
        assert!(
            r.score >= 0.3,
            "single-match item should clear 0.3 in fallback: got {}",
            r.score
        );
        assert!(r.reason.starts_with("Hybrid (BM25 only, boosted)"));
    }

    #[test]
    fn fallback_score_no_match_stays_zero() {
        let r = scorer().score(r#"{"id": 1}"#, "completely unrelated query");
        assert_eq!(r.score, 0.0);
    }

    #[test]
    fn fallback_score_two_or_more_matches_gets_extra_boost() {
        let s = scorer();
        // Three matched terms: should get the +0.2 second-tier boost.
        let r = s.score(
            r#"{"name": "alice", "role": "admin", "team": "engineering"}"#,
            "alice admin engineering",
        );
        assert!(
            r.score >= 0.5,
            "multi-match item should clear 0.5 in fallback: got {}",
            r.score
        );
    }

    #[test]
    fn fallback_score_batch_consistent_with_single() {
        let s = scorer();
        let items = [r#"{"name": "alice"}"#, r#"{"name": "bob"}"#];
        let single = s.score(items[0], "alice");
        let batch = s.score_batch(&items, "alice");
        // Boost rules apply to both paths; first item should match.
        assert!((batch[0].score - single.score).abs() < 0.5);
    }

    #[test]
    fn fallback_score_batch_empty_returns_empty() {
        let s = scorer();
        let result = s.score_batch(&[], "anything");
        assert!(result.is_empty());
    }

    // ---------- adaptive alpha ----------

    #[test]
    fn alpha_uuid_query_pushes_to_high_bm25_weight() {
        let s = scorer();
        let alpha = s.compute_alpha("find 550e8400-e29b-41d4-a716-446655440000");
        assert!(
            alpha >= 0.85,
            "UUID query should pin alpha >= 0.85: got {}",
            alpha
        );
    }

    #[test]
    fn alpha_multiple_numeric_ids_boosts_alpha() {
        let s = scorer();
        let alpha = s.compute_alpha("look up records 12345 and 67890");
        assert!(alpha >= 0.75, "got {}", alpha);
    }

    #[test]
    fn alpha_single_numeric_id_modest_boost() {
        let s = scorer();
        let alpha = s.compute_alpha("show record 12345");
        assert!(alpha >= 0.65, "got {}", alpha);
        assert!(alpha < 0.75);
    }

    #[test]
    fn alpha_hostname_modest_boost() {
        let s = scorer();
        let alpha = s.compute_alpha("status of api.example.com");
        assert!(alpha >= 0.6, "got {}", alpha);
    }

    #[test]
    fn alpha_natural_language_query_is_base() {
        let s = scorer();
        let alpha = s.compute_alpha("show me failed requests");
        assert_eq!(alpha, 0.5);
    }

    #[test]
    fn alpha_clamped_within_range() {
        let s = scorer();
        // Even with multiple boost-triggering patterns, alpha stays in [0.3, 0.9].
        let alpha = s.compute_alpha(
            "find 550e8400-e29b-41d4-a716-446655440000 and 12345 at api.example.com",
        );
        assert!((0.3..=0.9).contains(&alpha));
    }

    #[test]
    fn alpha_non_adaptive_returns_base() {
        let s = HybridScorer::new(0.7, false);
        let alpha = s.compute_alpha("find 550e8400-e29b-41d4-a716-446655440000");
        assert_eq!(alpha, 0.7);
    }
}
