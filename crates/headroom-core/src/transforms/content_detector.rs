//! Content type detection for multi-format compression.
//!
//! Direct port of `headroom/transforms/content_detector.py`. This module
//! detects the type of tool output content so the upstream
//! `ContentRouter` can dispatch it to the right compressor:
//!
//! - **JsonArray**: Structured JSON data → `SmartCrusher`
//! - **SourceCode**: Python, JavaScript, Go, Rust, etc. → `CodeAwareCompressor`
//! - **SearchResults**: grep / ripgrep output (`file:line:content`)
//! - **BuildOutput**: Compiler / test / lint logs
//! - **GitDiff**: Unified diff format → `DiffCompressor`
//! - **Html**: Web pages (needs extraction, not compression)
//! - **PlainText**: Generic fallback
//!
//! Detection is **regex-based** — no ML, no model loading, no I/O.
//! Magika integration lives one level up in `ContentRouter`, not here.
//!
//! # Parity with Python
//!
//! Regex patterns, dispatch order, confidence formulas, and line-count
//! caps are byte-equal with the Python source. Recorded fixtures in
//! `tests/parity/fixtures/content_detector/` lock the output across
//! the bridge.

use std::sync::LazyLock;

use regex::Regex;
use serde_json::{json, Map, Value};

/// Content types recognized by the detector. String tags match Python's
/// `ContentType` enum values 1:1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContentType {
    JsonArray,
    SourceCode,
    SearchResults,
    BuildOutput,
    GitDiff,
    Html,
    PlainText,
}

impl ContentType {
    /// Stable string tag — matches Python's `ContentType.<NAME>.value`.
    pub fn as_str(&self) -> &'static str {
        match self {
            ContentType::JsonArray => "json_array",
            ContentType::SourceCode => "source_code",
            ContentType::SearchResults => "search",
            ContentType::BuildOutput => "build",
            ContentType::GitDiff => "diff",
            ContentType::Html => "html",
            ContentType::PlainText => "text",
        }
    }
}

/// Result of `detect_content_type`. `metadata` is per-type free-form key/
/// value data — same shape as Python's `dict[str, Any]`. We use
/// `serde_json::Map` so PyO3 can convert it to a Python dict on the
/// boundary without losing type fidelity.
#[derive(Debug, Clone)]
pub struct DetectionResult {
    pub content_type: ContentType,
    pub confidence: f64,
    pub metadata: Map<String, Value>,
}

impl DetectionResult {
    fn new(content_type: ContentType, confidence: f64, metadata: Map<String, Value>) -> Self {
        Self {
            content_type,
            confidence,
            metadata,
        }
    }

    fn plain_text(confidence: f64) -> Self {
        Self::new(ContentType::PlainText, confidence, Map::new())
    }
}

// ─── Regex patterns (compiled once, shared) ───────────────────────────

/// `file:line:` (grep -n style) — first column on a non-blank line.
static SEARCH_RESULT_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[^\s:]+:\d+:").unwrap());

/// Diff-header detection. Recognizes:
/// - `git diff` (`diff --git`, `--- a/`)
/// - merge-commit headers (`diff --combined`, `diff --cc`)
/// - regular hunk headers (`@@ -A,B +C,D @@`)
/// - combined-diff hunk headers (`@@@ ... @@@`)
///
/// Mirrors Python's bug-fix from 2026-04-25 that widened the grammar
/// to handle merge-commit diffs from `git log -p`.
static DIFF_HEADER_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"^(diff --git|diff --combined |diff --cc |--- a/|@@\s+-\d+,\d+\s+\+\d+,\d+\s+@@|@@@+\s+-\d+(?:,\d+)?\s+(?:-\d+(?:,\d+)?\s+)+\+\d+(?:,\d+)?\s+@@@+)",
    )
    .unwrap()
});

/// Lines starting with `+` or `-` followed by a non-`+`/`-` char (i.e.
/// real change lines, not header lines like `+++ b/file`).
static DIFF_CHANGE_PATTERN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[+-][^+-]").unwrap());

// ─── Code patterns by language ─────────────────────────────────────────

struct CodePatterns {
    name: &'static str,
    patterns: Vec<Regex>,
}

static CODE_PATTERNS: LazyLock<Vec<CodePatterns>> = LazyLock::new(|| {
    vec![
        CodePatterns {
            name: "python",
            patterns: vec![
                Regex::new(r"^\s*(def|class|import|from|async def)\s+\w+").unwrap(),
                Regex::new(r"^\s*@\w+").unwrap(),
                Regex::new(r#"^\s*""""#).unwrap(),
                Regex::new(r"^\s*if __name__\s*==").unwrap(),
            ],
        },
        CodePatterns {
            name: "javascript",
            patterns: vec![
                Regex::new(r"^\s*(function|const|let|var|class|import|export)\s+").unwrap(),
                Regex::new(r"^\s*(async\s+function|=>\s*\{)").unwrap(),
                Regex::new(r"^\s*module\.exports").unwrap(),
            ],
        },
        CodePatterns {
            name: "typescript",
            patterns: vec![
                Regex::new(r"^\s*(interface|type|enum|namespace)\s+\w+").unwrap(),
                // Python uses `pattern.match(line)` which is start-anchored,
                // so this pattern only ever fires on lines literally starting
                // with `:`. We anchor with `^` to keep parity (the `regex`
                // crate's `is_match` is unanchored by default).
                Regex::new(r"^:\s*(string|number|boolean|any|void)\b").unwrap(),
            ],
        },
        CodePatterns {
            name: "go",
            patterns: vec![
                Regex::new(r"^\s*(func|type|package|import)\s+").unwrap(),
                Regex::new(r"^\s*func\s+\([^)]+\)\s+\w+").unwrap(),
            ],
        },
        CodePatterns {
            name: "rust",
            patterns: vec![
                Regex::new(r"^\s*(fn|struct|enum|impl|mod|use|pub)\s+").unwrap(),
                Regex::new(r"^\s*#\[").unwrap(),
            ],
        },
        CodePatterns {
            name: "java",
            patterns: vec![
                Regex::new(r"^\s*(public|private|protected)\s+(class|interface|enum)").unwrap(),
                Regex::new(r"^\s*@\w+").unwrap(),
                Regex::new(r"^\s*package\s+[\w.]+;").unwrap(),
            ],
        },
    ]
});

// ─── Log / build output patterns ───────────────────────────────────────
//
// Order matters: indices 0–1 (`ERROR` and `WARN` family) are treated as
// "error" matches by `try_detect_log`, contributing extra to confidence.
// Same ordering as Python.

static LOG_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    vec![
        Regex::new(r"(?i)\b(ERROR|FAIL|FAILED|FATAL|CRITICAL)\b").unwrap(),
        Regex::new(r"(?i)\b(WARN|WARNING)\b").unwrap(),
        Regex::new(r"(?i)\b(INFO|DEBUG|TRACE)\b").unwrap(),
        Regex::new(r"^\s*\d{4}-\d{2}-\d{2}").unwrap(),
        Regex::new(r"^\s*\[\d{2}:\d{2}:\d{2}\]").unwrap(),
        Regex::new(r"^={3,}|^-{3,}").unwrap(),
        Regex::new(r"^\s*PASSED|^\s*FAILED|^\s*SKIPPED").unwrap(),
        Regex::new(r"^npm ERR!|^yarn error|^cargo error").unwrap(),
        Regex::new(r"Traceback \(most recent call last\)").unwrap(),
        Regex::new(r"^\s*at\s+[\w.$]+\(").unwrap(),
    ]
});

// ─── HTML patterns ─────────────────────────────────────────────────────

static HTML_DOCTYPE_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)^\s*<!doctype\s+html").unwrap());
static HTML_TAG_PATTERN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)<html[\s>]").unwrap());
static HTML_HEAD_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)<head[\s>]").unwrap());
static HTML_BODY_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)<body[\s>]").unwrap());
static HTML_STRUCTURAL_TAGS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)<(div|span|script|style|link|meta|nav|header|footer|aside|article|section|main)[\s>]",
    )
    .unwrap()
});

// ─── Public entry point ────────────────────────────────────────────────

/// Detect the type of `content` for routing. Mirrors Python's
/// `detect_content_type`.
///
/// Dispatch order (matches Python verbatim):
/// 1. Empty / whitespace-only → `PlainText` confidence 0.0
/// 2. JSON array (highest priority for `SmartCrusher`)
/// 3. Git diff (≥ 0.7 confidence required)
/// 4. HTML (≥ 0.7 confidence required)
/// 5. Search results (≥ 0.6 confidence required)
/// 6. Build / log output (≥ 0.5 confidence required)
/// 7. Source code (≥ 0.5 confidence required)
/// 8. Fallback to `PlainText` confidence 0.5
pub fn detect_content_type(content: &str) -> DetectionResult {
    if content.is_empty() || content.trim().is_empty() {
        return DetectionResult::plain_text(0.0);
    }

    if let Some(r) = try_detect_json(content) {
        return r;
    }
    if let Some(r) = try_detect_diff(content) {
        if r.confidence >= 0.7 {
            return r;
        }
    }
    if let Some(r) = try_detect_html(content) {
        if r.confidence >= 0.7 {
            return r;
        }
    }
    if let Some(r) = try_detect_search(content) {
        if r.confidence >= 0.6 {
            return r;
        }
    }
    if let Some(r) = try_detect_log(content) {
        if r.confidence >= 0.5 {
            return r;
        }
    }
    if let Some(r) = try_detect_code(content) {
        if r.confidence >= 0.5 {
            return r;
        }
    }
    DetectionResult::plain_text(0.5)
}

/// Quick check: is `content` a JSON array of dictionaries (the format
/// `SmartCrusher` natively handles)? Convenience wrapper around
/// `detect_content_type`.
pub fn is_json_array_of_dicts(content: &str) -> bool {
    let result = detect_content_type(content);
    if result.content_type != ContentType::JsonArray {
        return false;
    }
    result
        .metadata
        .get("is_dict_array")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

// ─── Per-type detection helpers ────────────────────────────────────────

fn try_detect_json(content: &str) -> Option<DetectionResult> {
    let trimmed = content.trim();
    if !trimmed.starts_with('[') {
        return None;
    }
    let parsed: Value = serde_json::from_str(trimmed).ok()?;
    let arr = parsed.as_array()?;
    let item_count = arr.len();
    let is_dict_array = !arr.is_empty() && arr.iter().all(|v| v.is_object());
    let confidence = if is_dict_array { 1.0 } else { 0.8 };
    Some(DetectionResult::new(
        ContentType::JsonArray,
        confidence,
        json!({
            "item_count": item_count,
            "is_dict_array": is_dict_array,
        })
        .as_object()
        .cloned()
        .unwrap(),
    ))
}

fn try_detect_diff(content: &str) -> Option<DetectionResult> {
    // Window: 500 lines (extended from 50 in Python's 2026-04-25 fix).
    let mut header_matches: u32 = 0;
    let mut change_matches: u32 = 0;
    for line in content.split('\n').take(500) {
        if DIFF_HEADER_PATTERN.is_match(line) {
            header_matches += 1;
        }
        if DIFF_CHANGE_PATTERN.is_match(line) {
            change_matches += 1;
        }
    }
    if header_matches == 0 {
        return None;
    }
    // Same formula as Python: 0.5 + 0.2 * headers + 0.05 * changes, capped at 1.0
    let confidence =
        (0.5 + (header_matches as f64) * 0.2 + (change_matches as f64) * 0.05).min(1.0);
    Some(DetectionResult::new(
        ContentType::GitDiff,
        confidence,
        json!({
            "header_matches": header_matches,
            "change_lines": change_matches,
        })
        .as_object()
        .cloned()
        .unwrap(),
    ))
}

fn try_detect_html(content: &str) -> Option<DetectionResult> {
    // Sample first 3000 chars (byte-indexed; matches Python's str slice
    // for ASCII inputs which is the common HTML case).
    let sample: &str = if content.len() > 3000 {
        // Find the last char-boundary <= 3000 so we don't slice mid-codepoint.
        let mut cutoff = 3000;
        while !content.is_char_boundary(cutoff) {
            cutoff -= 1;
        }
        &content[..cutoff]
    } else {
        content
    };

    let has_doctype = HTML_DOCTYPE_PATTERN.is_match(sample);
    let has_html_tag = HTML_TAG_PATTERN.is_match(sample);
    let has_head = HTML_HEAD_PATTERN.is_match(sample);
    let has_body = HTML_BODY_PATTERN.is_match(sample);
    let structural_matches = HTML_STRUCTURAL_TAGS.find_iter(sample).count() as u32;

    if !has_doctype && !has_html_tag && structural_matches < 3 {
        return None;
    }

    let mut confidence = 0.0_f64;
    if has_doctype {
        confidence += 0.5;
    }
    if has_html_tag {
        confidence += 0.3;
    }
    if has_head {
        confidence += 0.1;
    }
    if has_body {
        confidence += 0.1;
    }
    confidence += (structural_matches as f64 * 0.03).min(0.3);
    confidence = confidence.min(1.0);

    if confidence < 0.5 {
        return None;
    }
    Some(DetectionResult::new(
        ContentType::Html,
        confidence,
        json!({
            "has_doctype": has_doctype,
            "has_html_tag": has_html_tag,
            "structural_tags": structural_matches,
        })
        .as_object()
        .cloned()
        .unwrap(),
    ))
}

fn try_detect_search(content: &str) -> Option<DetectionResult> {
    let lines: Vec<&str> = content.split('\n').take(100).collect();
    if lines.is_empty() {
        return None;
    }
    let mut matching_lines: u32 = 0;
    for line in &lines {
        if !line.trim().is_empty() && SEARCH_RESULT_PATTERN.is_match(line) {
            matching_lines += 1;
        }
    }
    if matching_lines == 0 {
        return None;
    }
    let non_empty_lines = lines.iter().filter(|l| !l.trim().is_empty()).count() as u32;
    if non_empty_lines == 0 {
        return None;
    }
    let ratio = matching_lines as f64 / non_empty_lines as f64;
    if ratio < 0.3 {
        return None;
    }
    let confidence = (0.4 + ratio * 0.6).min(1.0);
    Some(DetectionResult::new(
        ContentType::SearchResults,
        confidence,
        json!({
            "matching_lines": matching_lines,
            "total_lines": non_empty_lines,
        })
        .as_object()
        .cloned()
        .unwrap(),
    ))
}

fn try_detect_log(content: &str) -> Option<DetectionResult> {
    let lines: Vec<&str> = content.split('\n').take(200).collect();
    if lines.is_empty() {
        return None;
    }
    let mut pattern_matches: u32 = 0;
    let mut error_matches: u32 = 0;
    for line in &lines {
        for (i, pattern) in LOG_PATTERNS.iter().enumerate() {
            if pattern.is_match(line) {
                pattern_matches += 1;
                if i < 2 {
                    error_matches += 1;
                }
                break; // one pattern per line is enough
            }
        }
    }
    if pattern_matches == 0 {
        return None;
    }
    let non_empty_lines = lines.iter().filter(|l| !l.trim().is_empty()).count() as u32;
    if non_empty_lines == 0 {
        return None;
    }
    let ratio = pattern_matches as f64 / non_empty_lines as f64;
    if ratio < 0.1 {
        return None;
    }
    let confidence = (0.3 + ratio * 0.5 + (error_matches as f64) * 0.05).min(1.0);
    Some(DetectionResult::new(
        ContentType::BuildOutput,
        confidence,
        json!({
            "pattern_matches": pattern_matches,
            "error_matches": error_matches,
            "total_lines": non_empty_lines,
        })
        .as_object()
        .cloned()
        .unwrap(),
    ))
}

fn try_detect_code(content: &str) -> Option<DetectionResult> {
    let lines: Vec<&str> = content.split('\n').take(100).collect();
    if lines.is_empty() {
        return None;
    }
    // Track scores in **first-match insertion order** to mirror Python's
    // dict semantics. Python:
    //
    //   language_scores: dict[str, int] = {}
    //   ...
    //   best_lang = max(language_scores, key=lambda k: language_scores[k])
    //
    // - Languages are inserted into the dict the first time they match a
    //   line, so the dict's iteration order is the order languages first
    //   showed up — NOT registration order.
    // - `max(...)` returns the FIRST element with the maximum value when
    //   multiple keys tie, per the language spec.
    //
    // We replicate both with a Vec and a manual `find(score == max)` for
    // the first-on-tie tie-break (Rust's `max_by` returns LAST on ties).
    let mut language_scores: Vec<(&'static str, u32)> = Vec::new();

    for line in &lines {
        for cp in CODE_PATTERNS.iter() {
            for pattern in &cp.patterns {
                if pattern.is_match(line) {
                    if let Some(entry) = language_scores.iter_mut().find(|(n, _)| *n == cp.name) {
                        entry.1 += 1;
                    } else {
                        language_scores.push((cp.name, 1));
                    }
                    break;
                }
            }
        }
    }

    if language_scores.is_empty() {
        return None;
    }
    let max_score = language_scores.iter().map(|x| x.1).max().unwrap_or(0);
    let (best_lang, best_score) = *language_scores
        .iter()
        .find(|x| x.1 == max_score)
        .expect("language_scores non-empty");
    if best_score < 3 {
        return None;
    }
    let non_empty_lines = lines.iter().filter(|l| !l.trim().is_empty()).count() as u32;
    let ratio = best_score as f64 / non_empty_lines.max(1) as f64;
    let confidence = (0.4 + ratio * 0.4 + (best_score as f64) * 0.02).min(1.0);
    Some(DetectionResult::new(
        ContentType::SourceCode,
        confidence,
        json!({
            "language": best_lang,
            "pattern_matches": best_score,
        })
        .as_object()
        .cloned()
        .unwrap(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_returns_plain_text_zero_confidence() {
        let r = detect_content_type("");
        assert_eq!(r.content_type, ContentType::PlainText);
        assert_eq!(r.confidence, 0.0);
    }

    #[test]
    fn whitespace_only_returns_plain_text_zero_confidence() {
        let r = detect_content_type("   \n\t  ");
        assert_eq!(r.content_type, ContentType::PlainText);
        assert_eq!(r.confidence, 0.0);
    }

    #[test]
    fn json_array_of_dicts_high_confidence() {
        let r = detect_content_type(r#"[{"id": 1}, {"id": 2}]"#);
        assert_eq!(r.content_type, ContentType::JsonArray);
        assert_eq!(r.confidence, 1.0);
        assert_eq!(
            r.metadata.get("is_dict_array").unwrap().as_bool(),
            Some(true)
        );
        assert_eq!(r.metadata.get("item_count").unwrap().as_u64(), Some(2));
    }

    #[test]
    fn json_array_of_scalars_lower_confidence() {
        let r = detect_content_type(r#"[1, 2, 3]"#);
        assert_eq!(r.content_type, ContentType::JsonArray);
        assert_eq!(r.confidence, 0.8);
        assert_eq!(
            r.metadata.get("is_dict_array").unwrap().as_bool(),
            Some(false)
        );
    }

    #[test]
    fn empty_json_array_not_dict_array() {
        let r = detect_content_type("[]");
        assert_eq!(r.content_type, ContentType::JsonArray);
        assert_eq!(r.confidence, 0.8);
        assert_eq!(
            r.metadata.get("is_dict_array").unwrap().as_bool(),
            Some(false)
        );
    }

    #[test]
    fn json_object_falls_through_to_text() {
        // Detector only handles arrays.
        let r = detect_content_type(r#"{"id": 1}"#);
        assert_eq!(r.content_type, ContentType::PlainText);
    }

    #[test]
    fn search_results_detected() {
        let content =
            "src/main.py:42:def process():\nsrc/util.py:13:    return None\nlib/x.py:7:class X:";
        let r = detect_content_type(content);
        assert_eq!(r.content_type, ContentType::SearchResults);
        assert!(r.confidence >= 0.6);
    }

    #[test]
    fn git_diff_detected() {
        let content = "\
diff --git a/foo.py b/foo.py
--- a/foo.py
+++ b/foo.py
@@ -1,3 +1,4 @@
 def hello():
-    print('hi')
+    print('hello')
+    print('world')
";
        let r = detect_content_type(content);
        assert_eq!(r.content_type, ContentType::GitDiff);
        assert!(r.confidence >= 0.7);
    }

    #[test]
    fn html_doctype_detected() {
        let content = "\
<!DOCTYPE html>
<html>
<head><title>X</title></head>
<body><div>hi</div></body>
</html>";
        let r = detect_content_type(content);
        assert_eq!(r.content_type, ContentType::Html);
        assert!(r.confidence >= 0.7);
    }

    #[test]
    fn build_output_detected() {
        let content = "\
[INFO] Starting build
[INFO] Compiling 42 sources
[ERROR] Compilation failed
[WARN] Deprecated API
FAILED test_one
PASSED test_two
";
        let r = detect_content_type(content);
        assert_eq!(r.content_type, ContentType::BuildOutput);
        assert!(r.confidence >= 0.5);
    }

    #[test]
    fn python_code_detected() {
        let content = "\
import os
from typing import Any

def process(data):
    return data

class Service:
    def __init__(self):
        pass

    @property
    def x(self):
        return 1

if __name__ == '__main__':
    process({})
";
        let r = detect_content_type(content);
        assert_eq!(r.content_type, ContentType::SourceCode);
        assert_eq!(r.metadata.get("language").unwrap().as_str(), Some("python"));
    }

    #[test]
    fn rust_code_detected() {
        let content = "\
use std::sync::Arc;

#[derive(Debug)]
pub struct Foo {
    bar: u32,
}

pub fn baz() -> u32 {
    42
}

impl Foo {
    pub fn new() -> Self {
        Self { bar: 0 }
    }
}
";
        let r = detect_content_type(content);
        assert_eq!(r.content_type, ContentType::SourceCode);
        assert_eq!(r.metadata.get("language").unwrap().as_str(), Some("rust"));
    }

    #[test]
    fn go_code_detected() {
        let content = "\
package main

import \"fmt\"

func main() {
    fmt.Println(\"hello\")
}

type Service struct{}

func (s *Service) Do() {}

func helper() {}
";
        let r = detect_content_type(content);
        assert_eq!(r.content_type, ContentType::SourceCode);
        assert_eq!(r.metadata.get("language").unwrap().as_str(), Some("go"));
    }

    #[test]
    fn fallback_to_plain_text() {
        let content = "Just some random text without any special structure.";
        let r = detect_content_type(content);
        assert_eq!(r.content_type, ContentType::PlainText);
        assert_eq!(r.confidence, 0.5);
    }

    #[test]
    fn is_json_array_of_dicts_true_path() {
        assert!(is_json_array_of_dicts(r#"[{"a": 1}, {"a": 2}]"#));
    }

    #[test]
    fn is_json_array_of_dicts_scalars_returns_false() {
        assert!(!is_json_array_of_dicts(r#"[1, 2, 3]"#));
    }

    #[test]
    fn is_json_array_of_dicts_object_returns_false() {
        assert!(!is_json_array_of_dicts(r#"{"a": 1}"#));
    }

    #[test]
    fn is_json_array_of_dicts_empty_returns_false() {
        // Empty array is JsonArray but not is_dict_array.
        assert!(!is_json_array_of_dicts("[]"));
    }

    #[test]
    fn diff_low_confidence_does_not_short_circuit() {
        // Single header with no change lines yields 0.7 — borderline.
        // Should still register as diff (>= 0.7 threshold).
        let content = "diff --git a/x b/x\n";
        let r = detect_content_type(content);
        assert_eq!(r.content_type, ContentType::GitDiff);
    }

    #[test]
    fn html_below_threshold_falls_through() {
        // Just one structural tag — not enough.
        let r = detect_content_type("<div>hello</div>");
        assert_ne!(r.content_type, ContentType::Html);
    }

    #[test]
    fn content_type_string_tags_match_python() {
        assert_eq!(ContentType::JsonArray.as_str(), "json_array");
        assert_eq!(ContentType::SourceCode.as_str(), "source_code");
        assert_eq!(ContentType::SearchResults.as_str(), "search");
        assert_eq!(ContentType::BuildOutput.as_str(), "build");
        assert_eq!(ContentType::GitDiff.as_str(), "diff");
        assert_eq!(ContentType::Html.as_str(), "html");
        assert_eq!(ContentType::PlainText.as_str(), "text");
    }
}
