//! Tier-3 pattern-based [`super::LineImportanceDetector`] backed by
//! `aho-corasick`.
//!
//! Replaces the Python `error_detection.py` regex registry. A single
//! deterministic-finite-automaton scan finds every keyword on a line in
//! `O(n + m)` — much faster than `len(patterns)` independent regex
//! searches, and it's harder to misuse (one source of truth for the
//! keyword set, no drift between sets and compiled patterns).
//!
//! # Bug fixes vs Python (2026-04-29)
//!
//! Python's `error_detection.py` had two bugs the parity fixtures
//! lock against:
//!
//! 1. `ERROR_KEYWORDS` listed `{abort, timeout, denied, rejected}` but
//!    `ERROR_PATTERN` regex omitted all four. Lines saying
//!    "Connection timeout" therefore never flagged as errors despite
//!    the keyword being canonical. **Fixed here**: the four keywords
//!    are part of the error set the automaton consumes.
//! 2. `SECURITY_KEYWORDS` included `token`, which false-positives on
//!    every reference to LLM tokens (`input_tokens`,
//!    `tokens_saved`, …). In an LLM-token-saturated codebase the
//!    security signal was uselessly noisy. **Fixed here**: `token` is
//!    dropped from the security set.
//!
//! Parity fixtures (in `tests/`) carry explicit `// fixed_in_3e1`
//! markers on each diverging line so the audit trail is clear.

use std::collections::BTreeMap;

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};

use super::line_importance::{
    ImportanceCategory, ImportanceContext, ImportanceSignal, LineImportanceDetector,
};

/// Confidence used by the keyword tier. Below the
/// [`super::tiered::ESCALATE_THRESHOLD`] used by [`super::tiered::Tiered`]
/// so a future ML tier can override on borderline cases — but high
/// enough that an unambiguous keyword match isn't second-guessed.
const KEYWORD_CONFIDENCE: f32 = 0.7;

/// Priority returned for a confirmed match. Compressors use this as the
/// score they sort by; tweak per category if a future caller wants
/// errors to outrank importance markers in routing decisions.
const ERROR_PRIORITY: f32 = 0.95;
const WARNING_PRIORITY: f32 = 0.75;
const SECURITY_PRIORITY: f32 = 0.85;
const IMPORTANCE_PRIORITY: f32 = 0.6;
const MARKDOWN_PRIORITY: f32 = 0.45;

/// Static keyword data for each importance category.
///
/// Exported so the Python shim can reflect on it for legacy regex
/// re-export. A `BTreeMap` keeps iteration order deterministic without
/// extra allocations.
#[derive(Debug, Clone)]
pub struct KeywordRegistry {
    pub error: Vec<&'static str>,
    pub warning: Vec<&'static str>,
    pub importance: Vec<&'static str>,
    pub security: Vec<&'static str>,
    /// Per-context line prefixes that count as importance signals (e.g.
    /// markdown headers `# `, blockquotes `> `). Matched as
    /// *prefix-only*, not whole-line keywords.
    pub markdown_prefixes: Vec<&'static str>,
    /// Substring indicators used by [`KeywordDetector::contains_error_indicator`]
    /// for fast triage (no word-boundary requirement). Distinct from
    /// `error` because the triage callsite (e.g. message-signature
    /// classification) cares about Python tracebacks specifically.
    pub error_indicators: Vec<&'static str>,
}

impl KeywordRegistry {
    /// The default Headroom keyword set — superset of Python's pre-3e.1
    /// `error_detection.py` minus the dropped `token` security keyword
    /// and plus the four error keywords the Python regex was missing.
    pub fn default_set() -> Self {
        Self {
            error: vec![
                "error",
                "exception",
                "fail",
                "failed",
                "failure",
                "fatal",
                "critical",
                "crash",
                "panic",
                "abort",
                "timeout",
                "denied",
                "rejected",
            ],
            warning: vec!["warn", "warning"],
            importance: vec![
                "important",
                "note",
                "todo",
                "fixme",
                "hack",
                "xxx",
                "bug",
                "fix",
            ],
            security: vec!["security", "auth", "password", "secret"],
            markdown_prefixes: vec!["# ", "## ", "### ", "#### ", "**", "> "],
            error_indicators: vec![
                "error",
                "fail",
                "exception",
                "traceback",
                "fatal",
                "panic",
                "crash",
            ],
        }
    }

    /// Snapshot for Python-side reflection. `BTreeMap` so iteration is
    /// deterministic across PyO3 calls.
    pub fn as_map(&self) -> BTreeMap<&'static str, Vec<&'static str>> {
        let mut m = BTreeMap::new();
        m.insert("error", self.error.clone());
        m.insert("warning", self.warning.clone());
        m.insert("importance", self.importance.clone());
        m.insert("security", self.security.clone());
        m.insert("markdown_prefixes", self.markdown_prefixes.clone());
        m.insert("error_indicators", self.error_indicators.clone());
        m
    }
}

/// One automaton + the parallel category lookup table. The automaton is
/// built case-insensitively; word-boundary checks happen as a post-filter
/// on the byte offsets it returns.
struct CategoryAutomaton {
    automaton: AhoCorasick,
    categories: Vec<ImportanceCategory>,
}

impl CategoryAutomaton {
    fn build(entries: &[(ImportanceCategory, &[&'static str])]) -> Self {
        let mut patterns = Vec::new();
        let mut categories = Vec::new();
        for (cat, words) in entries {
            for w in *words {
                patterns.push(*w);
                categories.push(*cat);
            }
        }
        let automaton = AhoCorasickBuilder::new()
            .ascii_case_insensitive(true)
            .match_kind(MatchKind::LeftmostLongest)
            .build(&patterns)
            .expect("keyword automaton must build (static input)");
        Self {
            automaton,
            categories,
        }
    }

    /// Highest-priority category whose keyword appears as a *whole word*
    /// in `line`, or `None` if nothing matched.
    fn first_word_match(&self, line: &str) -> Option<ImportanceCategory> {
        let bytes = line.as_bytes();
        for m in self.automaton.find_iter(line) {
            if is_word_boundary(bytes, m.start(), m.end()) {
                return Some(self.categories[m.pattern().as_usize()]);
            }
        }
        None
    }
}

/// Pattern-based [`LineImportanceDetector`] backed by aho-corasick.
///
/// Construct with [`KeywordDetector::new`] for the default Headroom
/// keyword set, or [`KeywordDetector::with_registry`] for a custom one.
pub struct KeywordDetector {
    registry: KeywordRegistry,
    /// Categories that fire across all contexts (error/importance).
    universal: CategoryAutomaton,
    /// Warning fires in Search/Log/Text contexts but is omitted in
    /// Diff (matches Python's `PRIORITY_PATTERNS_DIFF` shape).
    warning: CategoryAutomaton,
    /// Security fires in Diff context only.
    security: CategoryAutomaton,
    /// Substring-only indicators for fast triage; deliberately separate
    /// from `universal` because (a) it matches without word boundaries
    /// and (b) the indicator set diverges from the line-scoring set
    /// (carries `traceback`, omits the four extras like `timeout`).
    indicators: AhoCorasick,
}

impl KeywordDetector {
    pub fn new() -> Self {
        Self::with_registry(KeywordRegistry::default_set())
    }

    pub fn with_registry(registry: KeywordRegistry) -> Self {
        let universal = CategoryAutomaton::build(&[
            (ImportanceCategory::Error, &registry.error),
            (ImportanceCategory::Importance, &registry.importance),
        ]);
        let warning = CategoryAutomaton::build(&[(ImportanceCategory::Warning, &registry.warning)]);
        let security =
            CategoryAutomaton::build(&[(ImportanceCategory::Security, &registry.security)]);
        let indicators = AhoCorasickBuilder::new()
            .ascii_case_insensitive(true)
            .match_kind(MatchKind::LeftmostLongest)
            .build(&registry.error_indicators)
            .expect("indicator automaton must build (static input)");
        Self {
            registry,
            universal,
            warning,
            security,
            indicators,
        }
    }

    /// Fast keyword-presence check used by callers that only want
    /// "does this contain anything error-shaped?" (the legacy
    /// `content_has_error_indicators` callsite).
    ///
    /// Substring match — no word-boundary requirement — to preserve
    /// the lax semantics Python had. Distinct keyword set from
    /// [`Self::score`] (carries `traceback`, omits the four 3e1 extras
    /// like `timeout`) because the triage callsite cares about
    /// Python-style exception output more than connection states.
    pub fn contains_error_indicator(&self, text: &str) -> bool {
        self.indicators.is_match(text)
    }

    pub fn registry(&self) -> &KeywordRegistry {
        &self.registry
    }

    fn match_in_context(
        &self,
        line: &str,
        ctx: ImportanceContext,
    ) -> Option<(ImportanceCategory, f32)> {
        if let Some(cat) = self.universal.first_word_match(line) {
            let priority = priority_for(cat);
            return Some((cat, priority));
        }
        match ctx {
            ImportanceContext::Diff => {
                if let Some(cat) = self.security.first_word_match(line) {
                    return Some((cat, priority_for(cat)));
                }
            }
            ImportanceContext::Text | ImportanceContext::Search | ImportanceContext::Log => {
                if let Some(cat) = self.warning.first_word_match(line) {
                    return Some((cat, priority_for(cat)));
                }
            }
        }
        // Markdown structural prefixes only count in Text context.
        if matches!(ctx, ImportanceContext::Text) {
            if let Some(prefix) = self
                .registry
                .markdown_prefixes
                .iter()
                .find(|p| line.starts_with(*p))
            {
                let _ = prefix;
                return Some((ImportanceCategory::Markdown, MARKDOWN_PRIORITY));
            }
        }
        None
    }
}

impl Default for KeywordDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl LineImportanceDetector for KeywordDetector {
    fn score(&self, line: &str, ctx: ImportanceContext) -> ImportanceSignal {
        match self.match_in_context(line, ctx) {
            Some((category, priority)) => {
                ImportanceSignal::matched(category, priority, KEYWORD_CONFIDENCE)
            }
            None => ImportanceSignal::neutral(),
        }
    }
}

const fn priority_for(category: ImportanceCategory) -> f32 {
    match category {
        ImportanceCategory::Error => ERROR_PRIORITY,
        ImportanceCategory::Warning => WARNING_PRIORITY,
        ImportanceCategory::Security => SECURITY_PRIORITY,
        ImportanceCategory::Importance => IMPORTANCE_PRIORITY,
        ImportanceCategory::Markdown => MARKDOWN_PRIORITY,
    }
}

/// True when `[start..end)` in `bytes` is bounded by a non-word-character
/// (or string boundary) on each side. ASCII word characters: `[A-Za-z0-9_]`.
fn is_word_boundary(bytes: &[u8], start: usize, end: usize) -> bool {
    let left_ok = start == 0 || !is_word_byte(bytes[start - 1]);
    let right_ok = end == bytes.len() || !is_word_byte(bytes[end]);
    left_ok && right_ok
}

#[inline]
fn is_word_byte(b: u8) -> bool {
    matches!(b, b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detect(line: &str, ctx: ImportanceContext) -> ImportanceSignal {
        KeywordDetector::new().score(line, ctx)
    }

    #[test]
    fn fires_on_uppercase_error_in_search() {
        let s = detect("ERROR: connection refused", ImportanceContext::Search);
        assert_eq!(s.category, Some(ImportanceCategory::Error));
        assert!(s.priority > 0.9);
    }

    #[test]
    fn timeout_now_classified_as_error_in_diff() {
        // fixed_in_3e1: Python's ERROR_PATTERN regex omitted "timeout",
        // so this line was misclassified as neutral despite being
        // canonical in ERROR_KEYWORDS.
        let s = detect(
            "FATAL: timeout connecting upstream",
            ImportanceContext::Diff,
        );
        assert_eq!(s.category, Some(ImportanceCategory::Error));
    }

    #[test]
    fn rejected_now_classified_as_error() {
        // fixed_in_3e1: parity gap with Python.
        let s = detect("auth request rejected", ImportanceContext::Diff);
        assert_eq!(s.category, Some(ImportanceCategory::Error));
    }

    #[test]
    fn token_no_longer_flags_security_in_llm_proxy_context() {
        // fixed_in_3e1: dropping "token" from the security set means an
        // LLM-metric line stops false-positively routing as a security
        // signal.
        let s = detect(
            "input_tokens=512 output_tokens=256",
            ImportanceContext::Diff,
        );
        assert!(!s.is_match());
    }

    #[test]
    fn auth_still_flags_security_in_diff() {
        let s = detect("missing auth header", ImportanceContext::Diff);
        assert_eq!(s.category, Some(ImportanceCategory::Security));
    }

    #[test]
    fn warning_fires_in_search_but_not_diff() {
        let in_search = detect("warning: deprecated API", ImportanceContext::Search);
        assert_eq!(in_search.category, Some(ImportanceCategory::Warning));

        // Python's PRIORITY_PATTERNS_DIFF excluded WARNING_PATTERN; we
        // preserve that.
        let in_diff = detect(
            "warning: deprecated API alone with no errors",
            ImportanceContext::Diff,
        );
        assert_ne!(in_diff.category, Some(ImportanceCategory::Warning));
    }

    #[test]
    fn markdown_header_fires_only_in_text() {
        let in_text = detect("# Important section", ImportanceContext::Text);
        // "important" is itself an importance keyword, so this line
        // fires as Importance (universal) before we reach the markdown
        // prefix check. Drop the keyword to isolate the prefix path.
        let _ = in_text;
        let prefix_only = detect("# Section", ImportanceContext::Text);
        assert_eq!(prefix_only.category, Some(ImportanceCategory::Markdown));
        let same_line_in_diff = detect("# Section", ImportanceContext::Diff);
        assert!(!same_line_in_diff.is_match());
    }

    #[test]
    fn word_boundary_excludes_substring_matches() {
        // Without word boundaries, "preferred" would match "fail" via
        // the substring "fer" -> not a real risk, but
        // "tokenize" must NOT be misread as the error keyword "token"
        // (we dropped that one anyway), and "panicker" must not match
        // "panic" inside a normal English word.
        let s = detect("the panicker showed up late", ImportanceContext::Search);
        assert!(!s.is_match());
    }

    #[test]
    fn neutral_line_returns_zero_confidence() {
        let s = detect("the quick brown fox", ImportanceContext::Text);
        assert!(!s.is_match());
        assert_eq!(s.confidence, 0.0);
    }

    #[test]
    fn contains_error_indicator_is_lax_substring_match() {
        // Preserves Python `content_has_error_indicators` semantics:
        // "errored" -> matches "error". This is intentional for fast
        // triage; the strict version is `score()`.
        let det = KeywordDetector::new();
        assert!(det.contains_error_indicator("the request errored out"));
        assert!(det.contains_error_indicator("traceback follows"));
        assert!(!det.contains_error_indicator("everything is fine"));
    }

    #[test]
    fn registry_snapshot_has_token_dropped() {
        let reg = KeywordRegistry::default_set();
        assert!(!reg.security.contains(&"token"));
        assert!(reg.security.contains(&"auth"));
        assert!(reg.error.contains(&"timeout"));
        assert!(reg.error.contains(&"abort"));
    }
}
