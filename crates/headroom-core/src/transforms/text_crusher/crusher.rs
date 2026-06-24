//! TextCrusher: fast deterministic extractive prose compressor (Phase 2, #1171).
//!
//! Splits prose into sentence segments, scores each by recency + query
//! relevance + structural salience, suppresses near-duplicates via a global
//! word-shingle index, and keeps the top segments (in original order) up to a
//! target ratio. Output is extractive: the kept sentences are verbatim words
//! (each segment trimmed, re-joined with `\n`) -- no invented words, no rewrite.
//!
//! The relevance term REUSES the shared [`BM25Scorer`](crate::relevance) rather
//! than reimplementing BM25 -- only the prose-specific splitting + selection
//! lives here.

use std::cmp::Ordering;
use std::collections::HashSet;

use super::config::TextCrusherConfig;
use crate::relevance::{BM25Scorer, RelevanceScorer};

const KEYWORDS: [&str; 10] = [
    "error",
    "exception",
    "failed",
    "failure",
    "fail",
    "warning",
    "traceback",
    "assert",
    "todo",
    "fixme",
];

#[derive(Debug, Clone)]
pub struct TextCrusherResult {
    pub compressed: String,
    pub original_tokens: usize,
    pub compressed_tokens: usize,
    pub compression_ratio: f64,
    pub kept_segments: usize,
    pub total_segments: usize,
}

pub struct TextCrusher {
    config: TextCrusherConfig,
    scorer: BM25Scorer,
}

impl Default for TextCrusher {
    fn default() -> Self {
        TextCrusher::new(TextCrusherConfig::default())
    }
}

impl TextCrusher {
    pub fn new(config: TextCrusherConfig) -> Self {
        TextCrusher {
            config,
            scorer: BM25Scorer::default(),
        }
    }

    fn passthrough(content: &str, n_segments: usize) -> TextCrusherResult {
        let toks = content.split_whitespace().count();
        TextCrusherResult {
            compressed: content.to_string(),
            original_tokens: toks,
            compressed_tokens: toks,
            compression_ratio: 1.0,
            kept_segments: n_segments,
            total_segments: n_segments,
        }
    }

    pub fn compress(
        &self,
        content: &str,
        context: &str,
        target_ratio: Option<f64>,
    ) -> TextCrusherResult {
        let cfg = &self.config;
        let ratio = target_ratio.unwrap_or(cfg.target_ratio).clamp(0.05, 1.0);

        let segments = split_segments(content);
        if segments.len() < cfg.min_segments_for_crush {
            return Self::passthrough(content, segments.len());
        }

        let n = segments.len();
        let total_chars: usize = segments.iter().map(|s| s.len()).sum();
        // .max(1) so a tiny input never truncates the budget to 0 (which would
        // admit nothing and silently fall back to a 100% passthrough).
        let target_chars = ((total_chars as f64 * ratio) as usize).max(1);

        // Relevance via the shared BM25 scorer (already [0, 1]).
        let seg_refs: Vec<&str> = segments.iter().map(|s| s.as_str()).collect();
        let relevance = self.scorer.score_batch(&seg_refs, context);

        let seg_tokens: Vec<Vec<String>> = segments.iter().map(|s| tokens(s)).collect();

        let mut scores = vec![0.0f64; n];
        for i in 0..n {
            let recency = (i as f64 + 1.0) / n as f64;
            let rel = relevance.get(i).map(|r| r.score).unwrap_or(0.0);
            let words: Vec<&str> = segments[i].split_whitespace().collect();
            let salient = words.iter().filter(|w| is_salient(w)).count();
            let salience = salient as f64 / (words.len() as f64 + 1.0);
            let mut score =
                cfg.w_recency * recency + cfg.w_relevance * rel + cfg.w_salience * salience;
            if segments[i].len() < cfg.min_segment_chars {
                score *= 0.25;
            }
            scores[i] = score;
        }

        // Highest score first; stable tiebreak by index for determinism.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(Ordering::Equal)
                .then(a.cmp(&b))
        });

        let mut kept = vec![false; n];
        let mut seen: HashSet<String> = HashSet::new();
        let mut kept_chars = 0usize;
        let mut kept_count = 0usize;
        for &i in &order {
            if kept_chars >= target_chars {
                break;
            }
            let sh = shingles(&seg_tokens[i], 3);
            if !sh.is_empty() {
                let covered =
                    sh.iter().filter(|s| seen.contains(*s)).count() as f64 / sh.len() as f64;
                if covered >= cfg.near_dup_threshold {
                    continue; // near-duplicate: most shingles already kept
                }
            }
            kept[i] = true;
            kept_count += 1;
            for s in sh {
                seen.insert(s);
            }
            kept_chars += segments[i].len();
        }

        if kept_count == 0 {
            return Self::passthrough(content, n);
        }

        let compressed = (0..n)
            .filter(|&i| kept[i])
            .map(|i| segments[i].as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let orig_tok = content.split_whitespace().count();
        let comp_tok = compressed.split_whitespace().count();
        TextCrusherResult {
            compression_ratio: if orig_tok > 0 {
                comp_tok as f64 / orig_tok as f64
            } else {
                1.0
            },
            compressed,
            original_tokens: orig_tok,
            compressed_tokens: comp_tok,
            kept_segments: kept_count,
            total_segments: n,
        }
    }
}

/// Split into sentence/line segments: on newlines, and after `.`/`!`/`?`
/// followed by whitespace. Byte-faithful (kept segments are joined verbatim).
fn split_segments(text: &str) -> Vec<String> {
    let mut segs = Vec::new();
    for line in text.split('\n') {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut cur = String::new();
        let mut prev_term = false;
        for c in trimmed.chars() {
            if prev_term && c.is_whitespace() {
                let s = cur.trim();
                if !s.is_empty() {
                    segs.push(s.to_string());
                }
                cur.clear();
                prev_term = false;
                continue;
            }
            cur.push(c);
            prev_term = matches!(c, '.' | '!' | '?');
        }
        let s = cur.trim();
        if !s.is_empty() {
            segs.push(s.to_string());
        }
    }
    segs
}

fn tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in text.chars() {
        if c.is_alphanumeric() || c == '_' {
            for lc in c.to_lowercase() {
                cur.push(lc);
            }
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn shingles(words: &[String], k: usize) -> HashSet<String> {
    let mut set = HashSet::new();
    if words.is_empty() {
        return set;
    }
    if words.len() < k {
        // Short segment: emit every sub-window (1..=len) so identical/overlapping
        // short segments still near-dup-match each other. (They can't match a
        // longer segment's k-grams, but short segments are score-penalized and
        // rarely survive selection anyway.)
        for size in 1..=words.len() {
            for w in words.windows(size) {
                set.insert(w.join("\u{1}"));
            }
        }
        return set;
    }
    for w in words.windows(k) {
        set.insert(w.join("\u{1}"));
    }
    set
}

/// A word carries specific, hard-to-reconstruct information if it has a digit,
/// is an error/status keyword, is ALLCAPS (2+ letters), or is a dotted
/// identifier (`foo.bar`).
fn is_salient(word: &str) -> bool {
    if word.chars().any(|c| c.is_ascii_digit()) {
        return true;
    }
    let lower = word
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_lowercase();
    if KEYWORDS.contains(&lower.as_str()) {
        return true;
    }
    let alpha: Vec<char> = word.chars().filter(|c| c.is_alphabetic()).collect();
    if alpha.len() >= 2 && alpha.iter().all(|c| c.is_uppercase()) {
        return true;
    }
    if let Some(dot) = word.find('.') {
        let a = &word[..dot];
        let b = &word[dot + 1..];
        if !a.is_empty()
            && !b.is_empty()
            && a.chars()
                .next()
                .is_some_and(|c| c.is_alphabetic() || c == '_')
            && b.chars()
                .next()
                .is_some_and(|c| c.is_alphabetic() || c == '_')
        {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(n: usize) -> String {
        (0..n)
            .map(|i| format!("Sentence number {i} describes a distinct topic {i} in some detail."))
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[test]
    fn extractive_and_compresses() {
        let content = doc(40);
        let r = TextCrusher::default().compress(&content, "", Some(0.3));
        assert!(r.compressed_tokens < r.original_tokens);
        // extractive: every output word appears in the input
        let orig: HashSet<&str> = content.split_whitespace().collect();
        assert!(r.compressed.split_whitespace().all(|w| orig.contains(w)));
    }

    #[test]
    fn deterministic() {
        let content = doc(40);
        let tc = TextCrusher::default();
        assert_eq!(
            tc.compress(&content, "", Some(0.4)).compressed,
            tc.compress(&content, "", Some(0.4)).compressed
        );
    }

    #[test]
    fn passthrough_when_small() {
        let r = TextCrusher::default().compress("one. two. three.", "", None);
        assert_eq!(r.compression_ratio, 1.0);
    }
}
