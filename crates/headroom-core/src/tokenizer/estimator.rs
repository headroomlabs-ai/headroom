//! Character-density estimator. Used as a fallback for any tokenizer family
//! we haven't wired in yet (Anthropic Claude, Google Gemini, Cohere, …).
//!
//! Mirrors `headroom.tokenizers.estimator.EstimatingTokenCounter`. The formula
//! is `ceil(chars / chars_per_token)`. `chars` is *Unicode scalar count*, not
//! byte length, to match Python's `len(text)` semantics on str.

use super::{Backend, Tokenizer};

#[derive(Debug, Clone, Copy)]
pub struct EstimatingCounter {
    chars_per_token: f64,
}

impl Default for EstimatingCounter {
    fn default() -> Self {
        Self {
            chars_per_token: 4.0,
        }
    }
}

impl EstimatingCounter {
    /// `chars_per_token` must be `> 0.0`. Common calibrations:
    /// - 3.5 — Claude-family (Python uses this in `_create_anthropic`)
    /// - 4.0 — Gemini, Cohere, generic fallback
    pub fn new(chars_per_token: f64) -> Self {
        assert!(
            chars_per_token > 0.0,
            "chars_per_token must be positive, got {chars_per_token}"
        );
        Self { chars_per_token }
    }

    pub fn chars_per_token(&self) -> f64 {
        self.chars_per_token
    }
}

impl Tokenizer for EstimatingCounter {
    fn count_text(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }
        // Match Python `EstimatingTokenCounter.count_text`:
        //     max(1, int(len(text) / chars_per_token + 0.5))
        // Python `int()` truncates toward zero; for non-negative inputs that's
        // identical to `as usize` saturating-cast semantics in Rust >= 1.45.
        // Adding 0.5 then truncating yields round-half-up. We previously used
        // ceil, which over-counted in the middle of the range (e.g. "aaaaa"
        // at 4.0 cpt returned 2 here vs 1 in Python).
        let chars = text.chars().count() as f64;
        let raw = (chars / self.chars_per_token + 0.5) as usize;
        raw.max(1)
    }

    fn backend(&self) -> Backend {
        Backend::Estimation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_string_is_zero() {
        let est = EstimatingCounter::default();
        assert_eq!(est.count_text(""), 0);
    }

    /// Expected values cross-checked against Python:
    ///     max(1, int(len / chars_per_token + 0.5))
    #[test]
    fn default_is_four_chars_per_token() {
        let est = EstimatingCounter::default();
        // 4 / 4.0 = 1.0 + 0.5 -> int(1.5) -> 1
        assert_eq!(est.count_text("aaaa"), 1);
        // 5 / 4.0 = 1.25 + 0.5 -> int(1.75) -> 1   (Python rounds half-up)
        assert_eq!(est.count_text("aaaaa"), 1);
        // 6 / 4.0 = 1.5 + 0.5 -> int(2.0) -> 2
        assert_eq!(est.count_text("aaaaaa"), 2);
        // 40 / 4.0 = 10.0 + 0.5 -> int(10.5) -> 10
        assert_eq!(est.count_text(&"a".repeat(40)), 10);
    }

    #[test]
    fn claude_density_matches_python() {
        let est = EstimatingCounter::new(3.5);
        // 35 / 3.5 = 10.0  -> int(10.5) -> 10
        assert_eq!(est.count_text(&"a".repeat(35)), 10);
        // 36 / 3.5 ≈ 10.286 -> int(10.786) -> 10  (NOT 11 — that was ceil)
        assert_eq!(est.count_text(&"a".repeat(36)), 10);
        // 38 / 3.5 ≈ 10.857 -> int(11.357) -> 11
        assert_eq!(est.count_text(&"a".repeat(38)), 11);
    }

    #[test]
    fn unicode_uses_char_count_not_bytes() {
        let est = EstimatingCounter::default();
        // "héllo" is 5 chars, 6 bytes; 5/4.0 = 1.25 -> int(1.75) -> 1
        assert_eq!(est.count_text("héllo"), 1);
        // 4 emojis = 4 chars; 4/4.0 = 1.0 -> int(1.5) -> 1
        assert_eq!(est.count_text("🦀🦀🦀🦀"), 1);
    }

    #[test]
    fn min_is_one_for_non_empty_input() {
        let est = EstimatingCounter::default();
        // 1/4.0 = 0.25 + 0.5 -> int(0.75) -> 0; max(1, 0) -> 1
        assert_eq!(est.count_text("a"), 1);
        assert_eq!(est.count_text("ab"), 1);
        // 3/4.0 = 0.75 + 0.5 -> int(1.25) -> 1
        assert_eq!(est.count_text("abc"), 1);
        // 6/4.0 = 1.5 + 0.5 -> int(2.0) -> 2
        assert_eq!(est.count_text("aaaaaa"), 2);
    }

    #[test]
    fn deterministic() {
        let est = EstimatingCounter::default();
        let s = "the quick brown fox jumps over the lazy dog";
        let a = est.count_text(s);
        let b = est.count_text(s);
        assert_eq!(a, b);
    }

    #[test]
    fn very_long_input_does_not_overflow() {
        let est = EstimatingCounter::default();
        let s = "a".repeat(1_000_000);
        assert_eq!(est.count_text(&s), 250_000);
    }

    #[test]
    #[should_panic(expected = "chars_per_token must be positive")]
    fn rejects_zero_density() {
        let _ = EstimatingCounter::new(0.0);
    }

    #[test]
    #[should_panic(expected = "chars_per_token must be positive")]
    fn rejects_negative_density() {
        let _ = EstimatingCounter::new(-1.0);
    }

    #[test]
    fn backend_is_estimation() {
        let est = EstimatingCounter::default();
        assert_eq!(est.backend(), Backend::Estimation);
    }
}
