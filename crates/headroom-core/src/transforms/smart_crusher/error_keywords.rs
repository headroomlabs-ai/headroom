//! Canonical error keyword set for item preservation.
//!
//! Direct port of `ERROR_KEYWORDS` from `headroom/transforms/error_detection.py:18-33`.
//! These are the **FALLBACK** preservation signal when TOIN field
//! semantics aren't available yet (per the Python module-level
//! comment). Intentionally broad — better to over-preserve than to
//! drop a real error item.
//!
//! Used by `detect_error_items_for_preservation`. The list is small
//! enough to keep as a `&[&str]`; if we ever cross ~50 keywords, switch
//! to a `phf::Set` or pre-built FST for sub-linear lookup.

/// 12 error/failure keywords. Order doesn't matter for correctness, but
/// matches Python's set-literal order so reading both side-by-side is
/// easier. Lowercase by construction; callers must lowercase the
/// haystack before substring-matching.
pub const ERROR_KEYWORDS: &[&str] = &[
    "error",
    "exception",
    "failed",
    "failure",
    "critical",
    "fatal",
    "crash",
    "panic",
    "abort",
    "timeout",
    "denied",
    "rejected",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_python_count() {
        // Python `len(ERROR_KEYWORDS) == 12`. If this drifts, the
        // Python set was edited and the Rust list needs a matching
        // update.
        assert_eq!(ERROR_KEYWORDS.len(), 12);
    }

    #[test]
    fn all_lowercase_invariant() {
        for &kw in ERROR_KEYWORDS {
            assert_eq!(
                kw,
                kw.to_lowercase(),
                "ERROR_KEYWORDS must all be lowercase"
            );
        }
    }

    #[test]
    fn pinned_membership() {
        // Pin the exact set so accidental edits surface in CI rather
        // than silently changing item-preservation behavior.
        let expected = [
            "error",
            "exception",
            "failed",
            "failure",
            "critical",
            "fatal",
            "crash",
            "panic",
            "abort",
            "timeout",
            "denied",
            "rejected",
        ];
        let actual: std::collections::BTreeSet<&str> = ERROR_KEYWORDS.iter().copied().collect();
        let expected: std::collections::BTreeSet<&str> = expected.iter().copied().collect();
        assert_eq!(actual, expected);
    }
}
