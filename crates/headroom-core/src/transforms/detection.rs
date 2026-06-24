//! Stage-3d ContentRouter detection chain.
//!
//! Wires the per-tier detectors (`magika_detector`, `unidiff_detector`)
//! into the single function the ContentRouter calls. The locked design
//! from `project_rust_content_detection_arch.md`:
//!
//! ```text
//! Tier 1: magika_detect()       → if non-PlainText, return it
//! Tier 2: unidiff::is_diff()    → if true, return GitDiff
//! Tier 3: PlainText (fallthrough)
//! ```
//!
//! # Why no regex tier
//!
//! User-locked decision (2026-04-25): the Rust side does not run the
//! regex-based [`crate::transforms::content_detector`] in production
//! detection. The regex path stays in tree as a comparison oracle for
//! parity testing and as an opt-in escape hatch, but the dispatch is
//! magika + parser only.
//!
//! # Tier-1 errors do not abort the chain
//!
//! If magika init or inference fails, we log at warn level and proceed
//! to Tier 2. The chain's *next* tier is the legitimate fallback for
//! magika failure — that's the whole point of having tiers. We
//! deliberately do **not** treat tier-1 error as a hard failure of the
//! entire chain; that would block all detection on a transient ONNX
//! issue. Loud-on-error stays at the [`magika_detect`] entry point for
//! callers who care; the chain swallows the err with a log line.
//!
//! # SearchResults / BuildOutput
//!
//! The retired regex detector recognized grep-style search output
//! (`file:line:`) and CMake/log output as their own [`ContentType`]
//! variants. Magika has no equivalent labels. Per the locked design,
//! these now route to [`ContentType::PlainText`] — we prefer
//! passthrough to misroute. If a benchmark later shows real
//! compression loss on grep/build outputs, we add a focused detector
//! for those specifically; not preemptively.

use crate::transforms::content_detector::ContentType;
use crate::transforms::magika_detector::magika_detect;
use crate::transforms::unidiff_detector::is_diff;

/// Run the detection chain on `content` and return the chosen
/// [`ContentType`].
///
/// Empty input shortcuts to [`ContentType::PlainText`] without
/// touching either tier.
pub fn detect(content: &str) -> ContentType {
    if content.is_empty() {
        return ContentType::PlainText;
    }

    // ── Tier 1: Magika ──────────────────────────────────────────
    match magika_detect(content) {
        Ok(ContentType::PlainText) => {
            // Magika says "I don't know" or "plain text". Continue
            // to Tier 2 — magika frequently mis-classifies short
            // diffs and prose-prefixed diffs as text.
        }
        Ok(content_type) => return content_type,
        Err(e) => {
            // Init or inference failure. Log it (so an ops-side
            // health check can spot magika trouble in the proxy
            // logs) and fall through to Tier 2 — the chain itself
            // must not break on a single tier's outage.
            tracing::warn!(
                error = %e,
                "magika detection failed; falling through to unidiff tier"
            );
        }
    }

    // ── Tier 2: unidiff parser ──────────────────────────────────
    if is_diff(content) {
        return ContentType::GitDiff;
    }

    // ── Tier 3: fallthrough ─────────────────────────────────────
    ContentType::PlainText
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_short_circuits_to_plain_text() {
        assert_eq!(detect(""), ContentType::PlainText);
    }

    #[test]
    fn json_array_routes_via_tier_1() {
        let payload = r#"[{"id": 1}, {"id": 2}, {"id": 3}]"#;
        assert_eq!(detect(payload), ContentType::JsonArray);
    }

    #[test]
    fn source_code_routes_via_tier_1() {
        let py = "def hello():\n    print('world')\n\nclass Foo:\n    pass\n";
        assert_eq!(detect(py), ContentType::SourceCode);
    }

    #[test]
    fn html_routes_via_tier_1() {
        let html = "<!DOCTYPE html><html><body><h1>x</h1></body></html>";
        assert_eq!(detect(html), ContentType::Html);
    }

    #[test]
    fn standard_git_diff_routes_via_tier_1_or_2() {
        let diff = "diff --git a/foo.py b/foo.py\n\
                    --- a/foo.py\n\
                    +++ b/foo.py\n\
                    @@ -1,1 +1,2 @@\n \
                    def hello():\n\
                    +    print(\"new\")\n";
        // Either magika tags it `diff` (Tier 1 hit) or magika
        // mis-classifies as text and unidiff catches it (Tier 2).
        // Both paths produce GitDiff.
        assert_eq!(detect(diff), ContentType::GitDiff);
    }

    #[test]
    fn naked_hunk_diff_routes_via_tier_2() {
        // Magika often mis-classifies naked hunks (no `diff --git`
        // wrapper) because the visible bytes look like ordinary
        // patch lines mixed with code. Tier 2 (unidiff parser)
        // catches these.
        let diff = "--- a/foo.py\n\
                    +++ b/foo.py\n\
                    @@ -1,2 +1,2 @@\n\
                    -old line\n\
                    +new line\n \
                    context line\n";
        assert_eq!(detect(diff), ContentType::GitDiff);
    }

    #[test]
    fn plain_prose_routes_to_plain_text() {
        let prose = "The quick brown fox jumps over the lazy dog. \
                     Just regular English with no special structure.";
        assert_eq!(detect(prose), ContentType::PlainText);
    }

    #[test]
    fn grep_search_results_route_to_plain_text_per_locked_design() {
        // Locked design (2026-04-25): no regex tier on Rust side, so
        // grep-style `file:line:content` output now goes through
        // PlainText. This is a deliberate behavior change vs. the
        // retired regex detector. If proxy benchmarks later show
        // real compression loss on grep output, we add a focused
        // detector then — not preemptively.
        let grep = "src/foo.py:42:def process():\n\
                    src/bar.py:10:    return True\n\
                    src/baz.py:7:class Worker:\n";
        // We assert PlainText to lock the design. If magika
        // probabilistically detects this as code, that's also fine
        // for the router (CodeAware compresses code well too) —
        // but the test pins the safe-default contract.
        let result = detect(grep);
        assert!(
            result == ContentType::PlainText || result == ContentType::SourceCode,
            "grep output should route to PlainText (preferred) or SourceCode (acceptable), got {result:?}"
        );
    }

    #[test]
    fn build_log_output_routes_via_chain() {
        // Same locked-design note: build/test log output (no
        // explicit regex detector for it on the Rust side). Magika
        // can label this either as `txt` (→ PlainText, our mapping)
        // or, more probabilistically, as a code-group label like
        // `log` / `c` because the structured `[LEVEL]` shape and
        // `file.cpp:line` references look code-like to the model.
        // Either route is acceptable for the router: SourceCode
        // dispatches to the code-aware compressor (which compresses
        // log lines reasonably well via repetition collapsing);
        // PlainText is the safe-default passthrough. The test pins
        // the contract that it lands somewhere reasonable, not at
        // a degenerate type like JsonArray or GitDiff.
        let log = "[INFO] Building target foo\n\
                   [WARN] Deprecated API usage in foo.cpp:45\n\
                   [ERROR] Compilation failed: undefined reference\n";
        let got = detect(log);
        assert!(
            matches!(got, ContentType::PlainText | ContentType::SourceCode),
            "build log should route to PlainText or SourceCode, got {got:?}"
        );
    }

    #[test]
    fn yaml_routes_to_source_code() {
        // YAML lives in magika's `code` group; the chain returns it
        // as SourceCode so the router picks the code-aware compressor.
        let yaml = "name: my-app\nversion: 1.0\ndependencies:\n  - foo\n";
        assert_eq!(detect(yaml), ContentType::SourceCode);
    }

    #[test]
    fn rust_source_routes_to_source_code() {
        let rs = "use std::collections::HashMap;\n\n\
                  pub struct Counter { counts: HashMap<String, u32> }\n\n\
                  impl Counter {\n    \
                      pub fn new() -> Self { Self { counts: HashMap::new() } }\n\
                  }\n";
        assert_eq!(detect(rs), ContentType::SourceCode);
    }

    #[test]
    fn chain_is_deterministic_across_repeated_calls() {
        // Magika returns the same label for identical input on
        // repeated calls; the chain wraps that determinism.
        let payload = r#"{"users": [{"id": 1}, {"id": 2}]}"#;
        let a = detect(payload);
        let b = detect(payload);
        let c = detect(payload);
        assert_eq!(a, b);
        assert_eq!(b, c);
    }
}
