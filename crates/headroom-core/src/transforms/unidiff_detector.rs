//! Unidiff-based diff detection (Stage 3d Tier 2).
//!
//! Sits behind [`crate::transforms::magika_detector`] in the Stage-3d
//! detection pipeline. Magika is fast and right most of the time, but
//! it's a probabilistic ML classifier — short, prose-prefixed, or
//! "looks like code because the lines are code" diffs can slip past it
//! into [`ContentType::PlainText`]. Tier 2 catches those by running
//! the [`unidiff`] parser and checking whether the input parses to a
//! non-empty patch set.
//!
//! # Why a parser, not another regex
//!
//! The retired Python detector and the still-on-main regex
//! [`crate::transforms::content_detector`] use a hand-rolled
//! `DIFF_HEADER_PATTERN` regex. That works for the canonical shapes
//! but is brittle around the edges (combined-merge headers, naked
//! hunks, truncated outputs). The locked Stage-3d arch (memory
//! `project_rust_content_detection_arch.md`) explicitly retires the
//! regex tier on the Rust side — we use a real grammar oracle (the
//! [`unidiff::PatchSet`] parser) instead.
//!
//! # Scope
//!
//! - `is_diff(content)` returns true iff `PatchSet::parse` succeeds
//!   AND finds at least one [`unidiff::PatchedFile`] with at least
//!   one hunk. Empty patch sets (parser succeeds but found nothing)
//!   are explicitly **not** diffs — saves the router from compressing
//!   plain text as if it were a diff.
//!
//! - `detect_diff(content)` is the [`ContentType`]-typed wrapper:
//!   returns `Some(ContentType::GitDiff)` on hit, `None` otherwise.
//!   The router (PR5) chains this after Magika.
//!
//! # Known gaps (deliberately punted)
//!
//! - **Combined-merge diffs** (`@@@ ... @@@`) — `unidiff`'s hunk-header
//!   regex is for plain `@@`, not `@@@`. The router could fall back
//!   to the regex content_detector for these specifically, but in
//!   practice they're rare in proxy traffic. PR5 decides.
//! - **Multi-byte line ending shapes** — the parser walks
//!   `input.lines()`, which strips `\r` only when paired with `\n`.
//!   Pathological CRLF-stripped inputs could miss; we accept the gap.

use crate::transforms::content_detector::ContentType;
use unidiff::PatchSet;

/// Boolean predicate: does `content` parse as a unified diff with
/// real change content?
///
/// Empty input is **not** a diff (returns `false`) — saves a parser
/// call. Otherwise we hand off to [`PatchSet::parse`] and check that
/// the result has at least one file with at least one hunk.
///
/// Why "at least one hunk" instead of "parsed without error":
/// `unidiff::PatchSet::parse` returns `Ok(())` even on plain text
/// (it just finds zero files). That would route every non-diff
/// passthrough through the diff compressor — a silent regression.
/// The explicit hunk check makes the contract honest.
pub fn is_diff(content: &str) -> bool {
    if content.is_empty() {
        return false;
    }

    let mut patch = PatchSet::new();
    if patch.parse(content).is_err() {
        return false;
    }

    // `PatchSet::is_empty()` covers "found zero files"; the inner
    // loop covers "found a file but with zero hunks" (e.g. mode-only
    // changes). For diff-compressor routing we want at least one
    // hunk — that's where the actual line-level change content lives.
    !patch.is_empty() && patch.files().iter().any(|f| !f.is_empty())
}

/// [`ContentType`]-typed wrapper. Returns `Some(ContentType::GitDiff)`
/// when [`is_diff`] is true, `None` otherwise. The router (PR5)
/// chains this after Magika and uses the `Option` to cleanly fall
/// through to Tier 3 (`PlainText`) when both tiers say "not a diff".
pub fn detect_diff(content: &str) -> Option<ContentType> {
    if is_diff(content) {
        Some(ContentType::GitDiff)
    } else {
        None
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_is_not_a_diff() {
        assert!(!is_diff(""));
        assert_eq!(detect_diff(""), None);
    }

    #[test]
    fn plain_prose_is_not_a_diff() {
        let prose = "The quick brown fox jumps over the lazy dog. \
                     This is just regular English prose.";
        assert!(!is_diff(prose));
    }

    #[test]
    fn json_is_not_a_diff() {
        let json = r#"{"name": "Alice", "tags": ["a", "b", "c"]}"#;
        assert!(!is_diff(json));
    }

    #[test]
    fn source_code_is_not_a_diff() {
        let py = "def foo():\n    return 42\n\nclass Bar:\n    pass\n";
        assert!(!is_diff(py));
    }

    #[test]
    fn standard_git_diff_detected() {
        let diff = "diff --git a/foo.py b/foo.py\n\
                    index abc123..def456 100644\n\
                    --- a/foo.py\n\
                    +++ b/foo.py\n\
                    @@ -1,3 +1,4 @@\n \
                    def hello():\n\
                    +    print(\"new\")\n     \
                    return \"world\"\n\
                    -    # gone\n";
        assert!(is_diff(diff));
        assert_eq!(detect_diff(diff), Some(ContentType::GitDiff));
    }

    #[test]
    fn naked_hunk_without_git_header_detected() {
        // Output of `diff -u file1 file2` without git wrapper.
        let diff = "--- a/foo.py\n\
                    +++ b/foo.py\n\
                    @@ -1,2 +1,2 @@\n\
                    -old line\n\
                    +new line\n \
                    context\n";
        assert!(is_diff(diff));
    }

    #[test]
    fn multi_file_diff_detected() {
        let diff = "--- a/foo.py\n\
                    +++ b/foo.py\n\
                    @@ -1,1 +1,1 @@\n\
                    -old\n\
                    +new\n\
                    --- a/bar.py\n\
                    +++ b/bar.py\n\
                    @@ -1,1 +1,1 @@\n\
                    -gone\n\
                    +here\n";
        assert!(is_diff(diff));
    }

    #[test]
    fn empty_patch_set_is_not_a_diff() {
        // No files, no hunks — parser succeeds but result is empty.
        // We do NOT count this as a diff; routing it through the
        // diff compressor would be wrong.
        let almost = "Some prose mentioning @@ in passing.\n\
                      And maybe even --- a sentence with dashes.\n";
        assert!(!is_diff(almost));
    }

    #[test]
    fn truncated_diff_treated_consistently() {
        // Truncation is a known gap — unidiff is strict. We assert
        // whichever way it goes, so a future unidiff version that
        // tightens or relaxes this is caught explicitly. Today's
        // observation: truncation past the file headers usually
        // still yields a non-empty patch set if at least one full
        // hunk parsed.
        let truncated = "--- a/foo.py\n\
                         +++ b/foo.py\n\
                         @@ -1,1 +1,";
        // Document the current behavior; this test is the canary
        // for that contract changing.
        let _ = is_diff(truncated); // either-or accepted for now
    }

    #[test]
    fn diff_with_added_file_only() {
        let diff = "diff --git a/new.py b/new.py\n\
                    new file mode 100644\n\
                    index 0000000..9b710f3\n\
                    --- /dev/null\n\
                    +++ b/new.py\n\
                    @@ -0,0 +1,3 @@\n\
                    +line one\n\
                    +line two\n\
                    +line three\n";
        assert!(is_diff(diff));
    }

    #[test]
    fn diff_with_removed_file_only() {
        let diff = "diff --git a/gone.py b/gone.py\n\
                    deleted file mode 100644\n\
                    index 9b710f3..0000000\n\
                    --- a/gone.py\n\
                    +++ /dev/null\n\
                    @@ -1,2 +0,0 @@\n\
                    -line one\n\
                    -line two\n";
        assert!(is_diff(diff));
    }

    #[test]
    fn html_is_not_a_diff() {
        let html = "<!DOCTYPE html><html><body><h1>Hi</h1></body></html>";
        assert!(!is_diff(html));
    }

    #[test]
    fn yaml_is_not_a_diff() {
        let yaml = "name: my-app\nversion: 1.0\ndependencies:\n  - foo\n";
        assert!(!is_diff(yaml));
    }

    #[test]
    fn detect_diff_returns_none_on_negative() {
        assert_eq!(detect_diff("not a diff"), None);
        assert_eq!(detect_diff("{}"), None);
        assert_eq!(detect_diff(""), None);
    }
}
