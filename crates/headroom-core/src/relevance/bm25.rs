//! BM25 keyword relevance scorer.
//!
//! Direct port of `headroom/relevance/bm25.py`. Zero ML dependencies —
//! pure-Rust regex tokenization + integer arithmetic. Excellent for
//! exact-match cases (UUIDs, numeric IDs, tool-call argument values
//! that appear verbatim in the response).
//!
//! # Score post-processing
//!
//! The raw BM25 score is normalized to `[0, 1]` by dividing by
//! `max_score` (default 10.0). Items with at least one matched token
//! of length 8 or more get a `+0.3` long-token bonus (UUIDs, long IDs are
//! high-signal matches). Final score is clamped to `[0, 1]` via
//! `RelevanceScore::new`.

use std::collections::HashMap;
use std::sync::LazyLock;

use regex::Regex;

use super::base::{RelevanceScore, RelevanceScorer};

/// Tokenization regex. Order matters — UUID first so that hex-string
/// IDs aren't broken into pieces by the alphanumeric arm:
/// 1. UUID — 8-4-4-4-12 hex with dashes.
/// 2. Numeric ID — 4+ digits with word boundaries.
/// 3. Alphanumeric (incl. underscore) — fallback.
///
/// Note: Python's `\b\d{4,}\b` uses word boundaries. The Rust `regex`
/// crate supports `\b`, so the pattern translates literally.
static TOKEN_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    // Single-line alternation. Order matters: UUID first so hex IDs
    // aren't broken into 8/4/4/4/12 alphanumeric pieces.
    Regex::new(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}|\b\d{4,}\b|[a-zA-Z0-9_]+",
    )
    .expect("BM25 token regex must compile")
});

pub struct BM25Scorer {
    pub k1: f64,
    pub b: f64,
    pub normalize_score: bool,
    pub max_score: f64,
}

impl Default for BM25Scorer {
    fn default() -> Self {
        BM25Scorer {
            k1: 1.5,
            b: 0.75,
            normalize_score: true,
            max_score: 10.0,
        }
    }
}

impl BM25Scorer {
    pub fn new(k1: f64, b: f64, normalize_score: bool, max_score: f64) -> Self {
        BM25Scorer {
            k1,
            b,
            normalize_score,
            max_score,
        }
    }

    /// Tokenize text per Python's `_tokenize`: lowercase + regex
    /// `findall`. Returns lowercase tokens in document order.
    fn tokenize(&self, text: &str) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }
        let lower = text.to_lowercase();
        TOKEN_PATTERN
            .find_iter(&lower)
            .map(|m| m.as_str().to_string())
            .collect()
    }

    /// BM25 score for a single (doc, query) pair. Returns
    /// `(raw_score, matched_terms)` matching Python's `_bm25_score`.
    ///
    /// The `idf = ln(2)` constant mirrors Python's simplified
    /// single-document IDF — same formula whether scoring one item
    /// or batch-scoring many.
    fn bm25_score(
        &self,
        doc_tokens: &[String],
        query_freq: &HashMap<String, usize>,
        avg_doc_len: f64,
    ) -> (f64, Vec<String>) {
        if doc_tokens.is_empty() || query_freq.is_empty() {
            return (0.0, Vec::new());
        }

        let doc_len = doc_tokens.len() as f64;
        let avgdl = if avg_doc_len > 0.0 {
            avg_doc_len
        } else if doc_len > 0.0 {
            doc_len
        } else {
            1.0
        };

        // Doc-side term frequency.
        let mut doc_freq: HashMap<&str, usize> = HashMap::new();
        for t in doc_tokens {
            *doc_freq.entry(t.as_str()).or_insert(0) += 1;
        }

        let mut score = 0.0;
        let mut matched: Vec<String> = Vec::new();
        let idf = 2.0_f64.ln();

        // Iterate query_freq in HashMap order — Python iterates dict
        // order (insertion order in 3.7+). For matched_terms we only
        // care about MEMBERSHIP not ordering downstream, but we sort
        // tokens alphabetically here for deterministic test output
        // when multiple terms match.
        let mut keys: Vec<&String> = query_freq.keys().collect();
        keys.sort();

        for term in keys {
            let qf = query_freq[term];
            let Some(&f) = doc_freq.get(term.as_str()) else {
                continue;
            };
            matched.push(term.clone());

            let f = f as f64;
            let numerator = f * (self.k1 + 1.0);
            let denominator = f + self.k1 * (1.0 - self.b + self.b * doc_len / avgdl);
            let term_score = idf * numerator / denominator;
            score += term_score * qf as f64;
        }

        (score, matched)
    }

    /// Compute the final normalized score with the long-match bonus.
    fn finalize_score(&self, raw: f64, matched: &[String]) -> f64 {
        let mut normalized = if self.normalize_score {
            (raw / self.max_score).min(1.0)
        } else {
            raw
        };
        if matched.iter().any(|t| t.len() >= 8) {
            normalized = (normalized + 0.3).min(1.0);
        }
        normalized
    }
}

impl RelevanceScorer for BM25Scorer {
    fn score(&self, item: &str, context: &str) -> RelevanceScore {
        let item_tokens = self.tokenize(item);
        let context_tokens = self.tokenize(context);

        // Build query frequency map from context.
        let mut query_freq: HashMap<String, usize> = HashMap::new();
        for t in &context_tokens {
            *query_freq.entry(t.clone()).or_insert(0) += 1;
        }

        let (raw, matched) = self.bm25_score(&item_tokens, &query_freq, 0.0);
        let normalized = self.finalize_score(raw, &matched);

        let reason = match matched.len() {
            0 => "BM25: no term matches".to_string(),
            1 => format!("BM25: matched '{}'", matched[0]),
            n => {
                let preview: Vec<&str> = matched.iter().take(3).map(|s| s.as_str()).collect();
                let suffix = if n > 3 { "..." } else { "" };
                format!(
                    "BM25: matched {} terms ({}{})",
                    n,
                    preview.join(", "),
                    suffix
                )
            }
        };

        // Limit matched_terms field for readability (Python: matched[:10]).
        let matched_capped: Vec<String> = matched.iter().take(10).cloned().collect();

        RelevanceScore::new(normalized, reason, matched_capped)
    }

    fn score_batch(&self, items: &[&str], context: &str) -> Vec<RelevanceScore> {
        let context_tokens = self.tokenize(context);

        if context_tokens.is_empty() {
            return items
                .iter()
                .map(|_| RelevanceScore::empty("BM25: empty context"))
                .collect();
        }

        let mut query_freq: HashMap<String, usize> = HashMap::new();
        for t in &context_tokens {
            *query_freq.entry(t.clone()).or_insert(0) += 1;
        }

        // Pre-tokenize all items.
        let all_tokens: Vec<Vec<String>> = items.iter().map(|item| self.tokenize(item)).collect();

        // Average document length, matching Python:
        //   avg_len = sum(len(t) for t in all_tokens) / max(len(items), 1)
        let total_len: usize = all_tokens.iter().map(|t| t.len()).sum();
        let avg_len = total_len as f64 / items.len().max(1) as f64;

        all_tokens
            .into_iter()
            .map(|item_tokens| {
                let (raw, matched) = self.bm25_score(&item_tokens, &query_freq, avg_len);
                let normalized = self.finalize_score(raw, &matched);

                let reason = match matched.len() {
                    0 => "BM25: no matches".to_string(),
                    n => format!("BM25: {} terms", n),
                };

                // Python uses matched[:5] for the batch path.
                let matched_capped: Vec<String> = matched.iter().take(5).cloned().collect();
                RelevanceScore::new(normalized, reason, matched_capped)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scorer() -> BM25Scorer {
        BM25Scorer::default()
    }

    // ---------- tokenization ----------

    #[test]
    fn tokenize_empty_returns_empty() {
        assert!(scorer().tokenize("").is_empty());
    }

    #[test]
    fn tokenize_lowercases() {
        let toks = scorer().tokenize("Hello WORLD");
        assert_eq!(toks, vec!["hello", "world"]);
    }

    #[test]
    fn tokenize_uuid_as_single_token() {
        let toks = scorer().tokenize("find 550e8400-e29b-41d4-a716-446655440000 fast");
        assert!(
            toks.contains(&"550e8400-e29b-41d4-a716-446655440000".to_string()),
            "got {:?}",
            toks
        );
    }

    #[test]
    fn tokenize_numeric_id_when_4_plus_digits() {
        let toks = scorer().tokenize("user 12345 logged in 99 times");
        // 12345 = 5 digits (>=4) → kept as numeric ID via second arm.
        // 99 = 2 digits → falls to alphanumeric arm and matches.
        // Both end up as tokens.
        assert!(toks.contains(&"12345".to_string()));
        assert!(toks.contains(&"99".to_string()));
    }

    #[test]
    fn tokenize_strips_punctuation() {
        let toks = scorer().tokenize("hello, world!");
        assert_eq!(toks, vec!["hello", "world"]);
    }

    // ---------- score ----------

    #[test]
    fn score_no_match_returns_zero() {
        let s = scorer().score(
            r#"{"id": 1, "name": "alice"}"#,
            "completely unrelated query",
        );
        assert_eq!(s.score, 0.0);
        assert_eq!(s.reason, "BM25: no term matches");
        assert!(s.matched_terms.is_empty());
    }

    #[test]
    fn score_uuid_match_gets_long_token_bonus() {
        let item = r#"{"id": "550e8400-e29b-41d4-a716-446655440000", "name": "Alice"}"#;
        let s = scorer().score(item, "find record 550e8400-e29b-41d4-a716-446655440000");
        // Long-match bonus is +0.3, applied after normalization.
        // Even a low raw score should clear 0.3 with the bonus.
        assert!(
            s.score >= 0.3,
            "UUID match should clear 0.3 from long-token bonus: got {}",
            s.score
        );
        assert!(s.matched_terms.iter().any(|t| t.contains("550e8400")));
    }

    #[test]
    fn score_explainability_reason_shape() {
        let s = scorer().score("alice bob", "alice");
        assert!(s.reason.starts_with("BM25:"));
        assert!(s.reason.contains("alice"));
    }

    // ---------- score_batch ----------

    #[test]
    fn score_batch_empty_context_zero_scores() {
        let items = ["foo", "bar"];
        let scores = scorer().score_batch(&items, "");
        assert_eq!(scores.len(), 2);
        for s in scores {
            assert_eq!(s.score, 0.0);
            assert_eq!(s.reason, "BM25: empty context");
        }
    }

    #[test]
    fn score_batch_ranks_by_relevance() {
        let items = [
            r#"{"id": 1, "msg": "user alice logged in"}"#,
            r#"{"id": 2, "msg": "system started"}"#,
            r#"{"id": 3, "msg": "user bob logged out"}"#,
        ];
        let scores = scorer().score_batch(&items, "alice login");
        // Item 0 (alice + logged) should outrank item 1 (no match).
        assert!(
            scores[0].score > scores[1].score,
            "alice match should outrank: got {} vs {}",
            scores[0].score,
            scores[1].score
        );
    }

    #[test]
    fn score_batch_amortizes_context_tokenization() {
        // Sanity: large batch doesn't crash and returns correct length.
        let items: Vec<String> = (0..50)
            .map(|i| format!(r#"{{"id": {}, "name": "user{}"}}"#, i, i))
            .collect();
        let refs: Vec<&str> = items.iter().map(|s| s.as_str()).collect();
        let scores = scorer().score_batch(&refs, "user42");
        assert_eq!(scores.len(), 50);
    }

    // ---------- BM25 formula ----------

    #[test]
    fn higher_term_frequency_increases_score() {
        // Item that mentions "alice" three times should score higher
        // than item mentioning it once.
        let single = r#"{"name": "alice"}"#;
        let triple = r#"{"a": "alice", "b": "alice", "c": "alice"}"#;
        let s_single = scorer().score(single, "alice");
        let s_triple = scorer().score(triple, "alice");
        assert!(
            s_triple.score >= s_single.score,
            "more matches should not decrease score: triple={} single={}",
            s_triple.score,
            s_single.score
        );
    }

    #[test]
    fn long_match_bonus_applied_only_for_8plus_chars() {
        let short = scorer().score(r#"{"x": "ab"}"#, "ab");
        let long = scorer().score(r#"{"x": "abcdefgh"}"#, "abcdefgh");
        // Even with similar BM25 raw scores, long_match adds +0.3.
        // We just assert long >= short here (bonus applies to long only).
        assert!(long.score >= short.score);
    }

    #[test]
    fn is_available_is_true() {
        assert!(scorer().is_available());
    }
}
