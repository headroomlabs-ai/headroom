//! Search-results compressor — Rust port of
//! `headroom.transforms.search_compressor`.
//!
//! Compresses grep / ripgrep / ag output (one of the most common tool
//! outputs in coding tasks). Typical compression: 5-10×.
//!
//! # Input format
//!
//! Standard `grep -n` style:
//! ```text
//! src/utils.py:42:def process_data(items):
//! src/utils.py:43:    """Process items with validation."""
//! src/models.py:15:class DataProcessor:
//! ```
//!
//! Ripgrep with `-C` context (mixes `:` and `-` separators):
//! ```text
//! src/main.py-40-some context before
//! src/main.py:42:def process_data(items):
//! src/main.py-43-some context after
//! ```
//!
//! # Compression pipeline
//!
//! 1. Parse into `{file: [(line, content), ...]}` structure.
//! 2. Score each match on relevance (context-word overlap +
//!    [`crate::signals::LineImportanceDetector`] priority signals +
//!    config-supplied keywords).
//! 3. Sort files by total match score; cap to `max_files`.
//! 4. Run [`crate::transforms::adaptive_sizer::compute_optimal_k`] over
//!    the global match list with `bias` to land an adaptive total.
//! 5. Per-file selection: always-keep first/last (configurable), fill
//!    remaining slots by score, sort survivors back to line order.
//! 6. Format `file:line:content` lines + `[... and N more matches in
//!    file]` summaries.
//! 7. Optional CCR storage when `min_matches_for_ccr` cleared and
//!    compression ratio < 0.8 — appends standard CCR marker.
//!
//! # Bug fixes vs Python (2026-04-29)
//!
//! Python's `_GREP_PATTERN`/`_RG_CONTEXT_PATTERN` regexes mis-handled
//! two real-world inputs. The hand-rolled parser here fixes both:
//!
//! - **Windows paths.** `^([^:]+):(\d+):(.*)$` captured only the drive
//!   letter for `C:\Users\foo\bar.py:42:line`, then the `\d+` group
//!   failed (next char is `\`). Result: every Windows-formatted line
//!   silently dropped from `file_matches`. Rust parser detects
//!   `[A-Za-z]:[\\/]` drive-prefix and starts the line-number scan
//!   *after* the drive colon.
//! - **Filenames containing `-`.** `_RG_CONTEXT_PATTERN`'s
//!   `[^:-]+` excluded dashes from the path, so legitimate names like
//!   `pre-commit-config.yaml-42-line` parsed wrong. Rust parser
//!   anchors on the *line-number marker* (`<sep>\d+<sep>`) found
//!   earliest in the line; everything before is the path, everything
//!   after is the content.
//!
//! Two further hardening changes:
//!
//! - **CCR storage failures are loud.** Python silently swallowed all
//!   exceptions from the store. Rust returns `Result` and surfaces
//!   storage errors via `tracing::warn!` so operations can investigate.
//! - **Per-file dedup is `O(n log n)`.** Python checks `match not in
//!   file_selected` linearly inside a loop (worst-case O(n²) for big
//!   files). Rust uses a `BTreeSet<(line_number, content_hash)>` so the
//!   membership check is logarithmic.

use std::collections::{BTreeMap, BTreeSet};

use md5::{Digest, Md5};

use crate::ccr::CcrStore;
use crate::signals::{ImportanceContext, LineImportanceDetector};

/// True for CJK ideographs, kana, and Hangul. Code-point ranges kept
/// byte-identical with the Python `_is_cjk_char` for search-compressor parity.
fn is_cjk_char(c: char) -> bool {
    matches!(
        c as u32,
        0x3040..=0x30FF | 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xAC00..=0xD7AF | 0xF900..=0xFAFF
    )
}

/// CJK character bigrams from the CJK runs of a (lowercased) query, so a
/// spaceless CJK query can match content. Mirrors the Python `_cjk_bigrams`.
fn cjk_bigrams(text: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let mut run: Vec<char> = Vec::new();
    for c in text.chars() {
        if is_cjk_char(c) {
            run.push(c);
        } else {
            for w in run.windows(2) {
                out.insert(w.iter().collect::<String>());
            }
            run.clear();
        }
    }
    for w in run.windows(2) {
        out.insert(w.iter().collect::<String>());
    }
    out
}
use crate::transforms::adaptive_sizer::compute_optimal_k;

// ─── Types ──────────────────────────────────────────────────────────────

/// Single search match — a single grep-style hit.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchMatch {
    pub file: String,
    pub line_number: u64,
    pub content: String,
    /// Relevance score in [0.0, 1.0]; populated by [`SearchCompressor::score_matches`].
    pub score: f32,
}

impl SearchMatch {
    pub fn new(file: impl Into<String>, line_number: u64, content: impl Into<String>) -> Self {
        Self {
            file: file.into(),
            line_number,
            content: content.into(),
            score: 0.0,
        }
    }
}

/// All matches grouped under a single file.
#[derive(Debug, Clone, Default)]
pub struct FileMatches {
    pub file: String,
    pub matches: Vec<SearchMatch>,
}

impl FileMatches {
    pub fn new(file: impl Into<String>) -> Self {
        Self {
            file: file.into(),
            matches: Vec::new(),
        }
    }

    pub fn first(&self) -> Option<&SearchMatch> {
        self.matches.first()
    }

    pub fn last(&self) -> Option<&SearchMatch> {
        self.matches.last()
    }

    pub fn total_score(&self) -> f32 {
        self.matches.iter().map(|m| m.score).sum()
    }
}

/// Compressor configuration. Defaults match Python `SearchCompressorConfig`.
#[derive(Debug, Clone)]
pub struct SearchCompressorConfig {
    pub max_matches_per_file: usize,
    pub always_keep_first: bool,
    pub always_keep_last: bool,
    pub max_total_matches: usize,
    pub max_files: usize,
    pub context_keywords: Vec<String>,
    pub boost_errors: bool,
    pub enable_ccr: bool,
    pub min_matches_for_ccr: usize,
    /// Compression ratio threshold for CCR storage. Python defaults to
    /// 0.8 — only persist when compression saved at least 20%. Promoted
    /// to a config field here (Python had it inline) so a future
    /// pipeline can tune per-content-type.
    pub min_compression_ratio_for_ccr: f64,
    /// Group output by file (`rg --heading` style): emit each file path
    /// once as a header line, then `line:content` rows beneath it, with
    /// a blank line between file groups. Eliminates per-match path
    /// repetition — the dominant remaining token waste on large result
    /// sets (a 70-char path repeated 15× is ~250 wasted tokens).
    /// Default `false` (classic `file:line:content`) for parity; the
    /// proxy enables it in token mode.
    pub group_by_file: bool,
}

impl Default for SearchCompressorConfig {
    fn default() -> Self {
        Self {
            max_matches_per_file: 5,
            always_keep_first: true,
            always_keep_last: true,
            max_total_matches: 30,
            max_files: 15,
            context_keywords: Vec::new(),
            boost_errors: true,
            enable_ccr: true,
            min_matches_for_ccr: 10,
            min_compression_ratio_for_ccr: 0.8,
            group_by_file: false,
        }
    }
}

/// Compression result. `compressed` carries the formatted output (with
/// optional CCR marker appended); `summaries` maps file paths to the
/// `[... and N more matches in foo.py]` line that landed in that file's
/// section.
#[derive(Debug, Clone)]
pub struct SearchCompressionResult {
    pub compressed: String,
    pub original: String,
    pub original_match_count: usize,
    pub compressed_match_count: usize,
    pub files_affected: usize,
    pub compression_ratio: f64,
    pub cache_key: Option<String>,
    pub summaries: BTreeMap<String, String>,
}

impl SearchCompressionResult {
    /// Estimate tokens saved (rough: 1 token per 4 chars), matching Python.
    pub fn tokens_saved_estimate(&self) -> i64 {
        let chars_saved = self.original.len() as i64 - self.compressed.len() as i64;
        chars_saved.max(0) / 4
    }

    pub fn matches_omitted(&self) -> usize {
        self.original_match_count
            .saturating_sub(self.compressed_match_count)
    }
}

/// Sidecar diagnostics not returned by the parity-equal API. Captures
/// per-stage drop counts so OTel can see what the compressor actually
/// did beyond the bytes Python emits.
#[derive(Debug, Clone, Default)]
pub struct SearchCompressorStats {
    pub lines_scanned: usize,
    pub lines_unparsed: usize,
    pub files_dropped: usize,
    pub matches_dropped_by_per_file_cap: usize,
    pub matches_dropped_by_global_cap: usize,
    pub ccr_emitted: bool,
    pub ccr_skip_reason: Option<&'static str>,
}

// ─── Compressor ─────────────────────────────────────────────────────────

/// Top-level compressor. Holds an importance detector (from the signals
/// trait family) so the priority-pattern scoring is pluggable. Defaults
/// to a [`crate::signals::KeywordDetector`].
pub struct SearchCompressor {
    config: SearchCompressorConfig,
    importance: Box<dyn LineImportanceDetector>,
}

impl SearchCompressor {
    pub fn new(config: SearchCompressorConfig) -> Self {
        Self {
            config,
            importance: Box::new(crate::signals::KeywordDetector::new()),
        }
    }

    /// Construct with a custom [`LineImportanceDetector`]. Use this when
    /// stacking a `Tiered` detector (e.g. ML head + keyword fallback).
    pub fn with_detector<D: LineImportanceDetector + 'static>(
        config: SearchCompressorConfig,
        detector: D,
    ) -> Self {
        Self {
            config,
            importance: Box::new(detector),
        }
    }

    pub fn config(&self) -> &SearchCompressorConfig {
        &self.config
    }

    /// Compress without persisting CCR. Returns the parity-equal result
    /// plus sidecar stats.
    pub fn compress(
        &self,
        content: &str,
        context: &str,
        bias: f64,
    ) -> (SearchCompressionResult, SearchCompressorStats) {
        self.compress_with_store(content, context, bias, None)
    }

    /// Compress with optional CCR persistence. `store` is consulted only
    /// if `config.enable_ccr` is true and the compression cleared the
    /// thresholds; storage failures emit `tracing::warn!` rather than
    /// being silently swallowed.
    pub fn compress_with_store(
        &self,
        content: &str,
        context: &str,
        bias: f64,
        store: Option<&dyn CcrStore>,
    ) -> (SearchCompressionResult, SearchCompressorStats) {
        let mut stats = SearchCompressorStats::default();
        let parsed = self.parse_search_results(content, &mut stats);

        if parsed.is_empty() {
            return (
                SearchCompressionResult {
                    compressed: content.to_string(),
                    original: content.to_string(),
                    original_match_count: 0,
                    compressed_match_count: 0,
                    files_affected: 0,
                    compression_ratio: 1.0,
                    cache_key: None,
                    summaries: BTreeMap::new(),
                },
                stats,
            );
        }

        let original_count: usize = parsed.values().map(|fm| fm.matches.len()).sum();

        let mut scored = parsed;
        self.score_matches(&mut scored, context);

        let selected = self.select_matches(&scored, bias, &mut stats);

        let (compressed_body, summaries) = self.format_output(&selected, &scored);
        let compressed_count: usize = selected.values().map(|fm| fm.matches.len()).sum();
        let ratio = compressed_body.len() as f64 / content.len().max(1) as f64;

        let mut compressed = compressed_body;
        let mut cache_key = None;
        if self.config.enable_ccr {
            if original_count < self.config.min_matches_for_ccr {
                stats.ccr_skip_reason = Some("below min_matches_for_ccr");
            } else if ratio >= self.config.min_compression_ratio_for_ccr {
                stats.ccr_skip_reason = Some("compression ratio too high");
            } else if let Some(store) = store {
                let key = md5_hex_24(content);
                store.put(&key, content);
                let marker = format!(
                    "\n[{} matches compressed to {}. Retrieve more: hash={}]",
                    original_count, compressed_count, key
                );
                compressed.push_str(&marker);
                cache_key = Some(key);
                stats.ccr_emitted = true;
            } else {
                stats.ccr_skip_reason = Some("no store provided");
            }
        } else {
            stats.ccr_skip_reason = Some("ccr disabled in config");
        }

        let result = SearchCompressionResult {
            compressed,
            original: content.to_string(),
            original_match_count: original_count,
            compressed_match_count: compressed_count,
            files_affected: scored.len(),
            compression_ratio: ratio,
            cache_key,
            summaries,
        };
        (result, stats)
    }

    // ─── Stage helpers (also used by tests + Python adapter) ───────────

    pub fn parse_search_results(
        &self,
        content: &str,
        stats: &mut SearchCompressorStats,
    ) -> BTreeMap<String, FileMatches> {
        let mut out: BTreeMap<String, FileMatches> = BTreeMap::new();
        for raw in content.split('\n') {
            let line = raw.trim();
            if line.is_empty() {
                continue;
            }
            stats.lines_scanned += 1;
            match parse_match_line(line) {
                Some((file, line_no, body)) => {
                    out.entry(file.to_string())
                        .or_insert_with(|| FileMatches::new(file))
                        .matches
                        .push(SearchMatch::new(file, line_no, body));
                }
                None => stats.lines_unparsed += 1,
            }
        }
        out
    }

    pub fn score_matches(&self, files: &mut BTreeMap<String, FileMatches>, context: &str) {
        let context_lower = context.to_ascii_lowercase();
        // Dedup like Python's `set`; count length in CHARS (not bytes) to match
        // Python codepoints; and add CJK char bigrams so a spaceless CJK query
        // (no whitespace words to split on) can still match content.
        let mut context_words: BTreeSet<String> = context_lower
            .split_whitespace()
            .filter(|w| w.chars().count() > 2)
            .map(|w| w.to_string())
            .collect();
        context_words.extend(cjk_bigrams(&context_lower));

        for fm in files.values_mut() {
            for m in &mut fm.matches {
                let mut score: f32 = 0.0;
                let content_lower = m.content.to_ascii_lowercase();

                for w in &context_words {
                    if content_lower.contains(w.as_str()) {
                        score += 0.3;
                    }
                }

                if self.config.boost_errors {
                    let signal = self.importance.score(&m.content, ImportanceContext::Search);
                    if let Some(category) = signal.category {
                        // Python's loop boosts by 0.5 - i*0.1 over priority
                        // patterns; map our trait categories to the same
                        // ordering: Error first (0.5), Warning (0.4),
                        // Importance (0.3).
                        let bump = match category {
                            crate::signals::ImportanceCategory::Error => 0.5,
                            crate::signals::ImportanceCategory::Warning => 0.4,
                            crate::signals::ImportanceCategory::Importance => 0.3,
                            // Categories below aren't part of
                            // PRIORITY_PATTERNS_SEARCH; preserve Python's
                            // behavior of not boosting for them.
                            crate::signals::ImportanceCategory::Security
                            | crate::signals::ImportanceCategory::Markdown => 0.0,
                        };
                        score += bump;
                    }
                }

                for kw in &self.config.context_keywords {
                    if content_lower.contains(&kw.to_ascii_lowercase()) {
                        score += 0.4;
                    }
                }

                m.score = score.min(1.0);
            }
        }
    }

    pub fn select_matches(
        &self,
        files: &BTreeMap<String, FileMatches>,
        bias: f64,
        stats: &mut SearchCompressorStats,
    ) -> BTreeMap<String, FileMatches> {
        // Python `_select_matches` sorts files by total match score
        // descending. `BTreeMap` iterates in key order, so we collect
        // and sort explicitly.
        let mut by_score: Vec<(&String, &FileMatches)> = files.iter().collect();
        by_score.sort_by(|a, b| {
            b.1.total_score()
                .partial_cmp(&a.1.total_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if by_score.len() > self.config.max_files {
            stats.files_dropped += by_score.len() - self.config.max_files;
            by_score.truncate(self.config.max_files);
        }

        let all_match_strings: Vec<String> = by_score
            .iter()
            .flat_map(|(file, fm)| {
                fm.matches
                    .iter()
                    .map(move |m| format!("{}:{}:{}", file, m.line_number, m.content))
            })
            .collect();
        let all_refs: Vec<&str> = all_match_strings.iter().map(|s| s.as_str()).collect();
        let adaptive_total =
            compute_optimal_k(&all_refs, bias, 5, Some(self.config.max_total_matches));

        let mut selected: BTreeMap<String, FileMatches> = BTreeMap::new();
        let mut total_selected: usize = 0;

        for (file, fm) in by_score {
            if total_selected >= adaptive_total {
                stats.matches_dropped_by_global_cap += fm.matches.len();
                continue;
            }

            // Sort by score desc, ties broken by line number asc for
            // determinism (Python's `sorted` is stable; order in is
            // line-asc by construction so highest-score-first picks the
            // earliest line on ties).
            let mut sorted = fm.matches.clone();
            sorted.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.line_number.cmp(&b.line_number))
            });

            let mut file_selected: Vec<SearchMatch> = Vec::new();
            // BTreeSet for O(log n) "already in selection" check (Python
            // uses linear `not in` — quadratic for big files).
            let mut seen: BTreeSet<(u64, u64)> = BTreeSet::new();

            let remaining_cap = self
                .config
                .max_matches_per_file
                .min(adaptive_total.saturating_sub(total_selected));

            let push_unique = |m: &SearchMatch,
                               file_selected: &mut Vec<SearchMatch>,
                               seen: &mut BTreeSet<(u64, u64)>| {
                let key = (m.line_number, hash_u64(&m.content));
                if seen.insert(key) {
                    file_selected.push(m.clone());
                    true
                } else {
                    false
                }
            };

            if self.config.always_keep_first {
                if let Some(first) = fm.first() {
                    if file_selected.len() < remaining_cap {
                        push_unique(first, &mut file_selected, &mut seen);
                    }
                }
            }

            if self.config.always_keep_last && fm.matches.len() > 1 {
                if let Some(last) = fm.last() {
                    if file_selected.len() < remaining_cap {
                        push_unique(last, &mut file_selected, &mut seen);
                    }
                }
            }

            for m in &sorted {
                if file_selected.len() >= remaining_cap {
                    break;
                }
                push_unique(m, &mut file_selected, &mut seen);
            }

            // Restore line order for output.
            file_selected.sort_by_key(|m| m.line_number);

            let dropped_here = fm.matches.len().saturating_sub(file_selected.len());
            stats.matches_dropped_by_per_file_cap += dropped_here;

            total_selected += file_selected.len();
            selected.insert(
                file.clone(),
                FileMatches {
                    file: file.clone(),
                    matches: file_selected,
                },
            );
        }

        selected
    }

    pub fn format_output(
        &self,
        selected: &BTreeMap<String, FileMatches>,
        original: &BTreeMap<String, FileMatches>,
    ) -> (String, BTreeMap<String, String>) {
        let mut lines: Vec<String> = Vec::new();
        let mut summaries: BTreeMap<String, String> = BTreeMap::new();
        let grouped = self.config.group_by_file;

        for (file, fm) in selected {
            if grouped {
                // `rg --heading` style: path once, then line:content rows.
                if !lines.is_empty() {
                    lines.push(String::new());
                }
                lines.push(file.clone());
                for m in &fm.matches {
                    lines.push(format!("{}:{}", m.line_number, m.content));
                }
            } else {
                for m in &fm.matches {
                    lines.push(format!("{}:{}:{}", m.file, m.line_number, m.content));
                }
            }
            if let Some(orig_fm) = original.get(file) {
                if orig_fm.matches.len() > fm.matches.len() {
                    let omitted = orig_fm.matches.len() - fm.matches.len();
                    let summary = if grouped {
                        format!("[... and {} more matches]", omitted)
                    } else {
                        format!("[... and {} more matches in {}]", omitted, file)
                    };
                    lines.push(summary.clone());
                    summaries.insert(file.clone(), summary);
                }
            }
        }

        (lines.join("\n"), summaries)
    }
}

// ─── Parser ─────────────────────────────────────────────────────────────

/// Parse one grep/ripgrep-style line into `(file, line_number, content)`.
///
/// Strategy:
/// 1. If the line starts with a Windows drive prefix (`C:\` or `C:/`),
///    record the drive letter + colon as the path's required prefix and
///    start the line-number scan after the drive colon.
/// 2. Find the leftmost `<sep><digits><sep>` triplet where each `<sep>`
///    is `:` or `-`. The path is everything before the first `<sep>`;
///    the line number is the digit run; the content is everything after
///    the second `<sep>`.
/// 3. Both separators must agree in semantic — `:`/`-` may mix because
///    ripgrep emits `file:line:content` for matches and
///    `file-line-content` for context lines, sometimes intermingled.
///
/// Returns `None` for lines that don't match the shape (no
/// `<sep>\d+<sep>` found). Caller treats those as un-parseable and
/// drops them.
///
/// # Why the scan runs in tiers
///
/// Leftmost-wins alone misreads any *path segment* that itself contains
/// a `<sep>\d+<sep>` triplet, which is extremely common:
///
/// ```text
/// logs/2026-05-03/app.log:12:ERROR   -> ("logs/2026", 5, "03/app.log:12:ERROR")
/// advisories/CVE-2021-44228.md-9-ctx -> ("advisories/CVE", 2021, "44228.md-9-ctx")
/// ```
///
/// That is silent corruption, not a drop — the bogus path becomes the
/// grouping key in [`SearchCompressor::parse_search_results`], so the
/// model is shown a file and a line number that do not exist, plus a
/// body with a path fragment glued onto the front.
///
/// The tiers exploit what grep actually emits:
///
/// 1. **Colon tier** — leftmost `:\d+:` whose path part contains no
///    whitespace. `:` is grep's *match*-line separator and a path
///    practically never contains one (the Windows drive colon is
///    already skipped above), so the leftmost is the right one. The
///    no-whitespace guard keeps a `foo.rs:12:` reference *inside the
///    body* of a `-`-separated context line from hijacking the parse.
/// 2. **Dash tier** — `-` is grep's *context*-line separator, and unlike
///    `:` it appears inside real paths (`2026-05-03`, `CVE-2021-44228`,
///    `20240101-002-add_users.sql`), so the leftmost triplet is often
///    still inside the path. The scan therefore walks rightward — but it
///    only advances past a marker on *positive* evidence that the path
///    continues through it: either the path so far ends in a segment with
///    an extension (that marker is the boundary — stop), or the rest of
///    the token still holds a `/` or a further extension (still inside
///    the path — keep going). If the walk reaches the end of the token
///    without ever finding that evidence, the line is genuinely ambiguous
///    and the tier falls back to the leftmost marker — exactly what the
///    permissive rule returns. On any line where it cannot prove it knows
///    better, it agrees with the rule it refines.
///
///    One shape stays ambiguous *in principle*, and the tier knowingly
///    takes the other side of it than the permissive rule does:
///
///    ```text
///    advisories/CVE-2021-44228.md-9-ctx   -> file is CVE-2021-44228.md, line 9
///    Makefile-7-build.sh-2-stage          -> file is Makefile, line 7
///    ```
///
///    Both are `<stem>-<digits>-<name>.<ext>-<digits>-<rest>`. Nothing in
///    the *text* separates them — only a filesystem could. The tier stops
///    at the extension, so it reads the first correctly and the second
///    wrong; the permissive rule takes the leftmost marker and does the
///    reverse. That trade is deliberate: dated and CVE-style paths are
///    everyday grep output, whereas the second shape needs a context line
///    from an *extension-less* file whose body begins with a bare
///    `name.ext-<digits>-` token. Both readings lose sometimes; this one
///    loses far less often.
/// 3. **Permissive tier** — the original leftmost-any rule, unchanged.
///    Only reached when neither tier matched (e.g. a path containing a
///    space), so behaviour for those lines is exactly as before.
fn parse_match_line(line: &str) -> Option<(&str, u64, &str)> {
    scan_match_line(line, ScanTier::Colon)
        .or_else(|| scan_match_line(line, ScanTier::Dash))
        .or_else(|| scan_match_line(line, ScanTier::Permissive))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScanTier {
    Colon,
    Dash,
    Permissive,
}

/// True when `tok` holds a dot that looks like a *file extension*: a run
/// of 1–8 alphanumerics, at least one of them a letter, ending the token
/// or bounded by a path/marker separator.
///
/// The letter requirement is what keeps a dotted *version* out of the
/// extension test. `v1.2.3` and `rel.1.2` end in all-digit runs, so they
/// are not extensions — which matters because they show up in the bodies
/// of context lines from extension-less files (`git describe` output in a
/// `Makefile`, a version bump in a `CHANGELOG`), and reading one as an
/// extension is what lets the dash tier walk out of the path and into the
/// body. `.md`, `.sql`, `.log`, `.7z` all still qualify.
fn has_extension_dot(tok: &str) -> bool {
    let b = tok.as_bytes();
    b.iter().enumerate().any(|(i, &c)| {
        if c != b'.' || i + 1 == b.len() {
            return false;
        }
        let tail = &b[i + 1..];
        let end = tail
            .iter()
            .position(|c| !c.is_ascii_alphanumeric())
            .unwrap_or(tail.len());
        let ext = &tail[..end];
        (1..=8).contains(&ext.len()) && ext.iter().any(|c| c.is_ascii_alphabetic())
    })
}

/// True when `path`'s final segment carries a file extension.
///
/// The dash tier uses this to tell "these digits are still inside the
/// path" (`logs/2026`, no extension → keep scanning) from "these digits
/// are the line-number marker" (`logs/2026-05-03/app.log` → stop).
fn last_segment_has_extension(path: &str) -> bool {
    has_extension_dot(path.rsplit(['/', '\\']).next().unwrap_or(path))
}

/// True when the token after this marker still carries *path structure* —
/// a directory separator or an extension dot — before the next whitespace.
///
/// This is the positive evidence that the digits just matched were still
/// inside the path rather than the line-number marker. Both forms occur:
///
/// ```text
/// logs/2026-05-03/app.log             -> the path continues via `/`
/// migrations/20240101-002-add_users.sql -> ...and here via `.`, same directory
/// ```
///
/// Its absence is *not* proof of the opposite, which is why the caller
/// falls back to the leftmost marker rather than reading "no evidence" as
/// "keep walking".
fn path_continues(rest: &str) -> bool {
    rest.split_whitespace()
        .next()
        .is_some_and(|tok| tok.contains('/') || tok.contains('\\') || has_extension_dot(tok))
}

fn scan_match_line(line: &str, tier: ScanTier) -> Option<(&str, u64, &str)> {
    let bytes = line.as_bytes();
    // Windows drive prefix: starts with [A-Za-z]:[\\/]
    let scan_start = if bytes.len() >= 3
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && (bytes[2] == b'\\' || bytes[2] == b'/')
    {
        // Skip past the drive colon so it isn't misread as the
        // line-number-marker separator.
        2
    } else {
        0
    };

    // Candidates as (path_end, digits_start, digits_end).
    //
    // `first` is the leftmost marker — the answer the permissive
    // leftmost-wins rule gives, and the fallback when the dash tier
    // cannot prove a better one. `chosen` is a marker positively
    // confirmed as the path/body boundary.
    let mut first: Option<(usize, usize, usize)> = None;
    let mut chosen: Option<(usize, usize, usize)> = None;

    let mut i = scan_start;
    while i < bytes.len() {
        let is_sep = bytes[i] == b':' || bytes[i] == b'-';
        let tier_sep = match tier {
            ScanTier::Colon => bytes[i] == b':',
            ScanTier::Dash => bytes[i] == b'-',
            ScanTier::Permissive => is_sep,
        };
        if tier_sep {
            // Reject markers where the byte immediately before the
            // first separator is itself a separator. That collapses
            // adjacent-separator runs (`::` or `:-`) so a line like
            // `src/file.py:-1:invalid` doesn't parse the `-` as the
            // marker's first separator and `1` as the line number;
            // the negative sign belongs to the content, not the
            // marker, so the line is rejected as un-parseable.
            if i > 0 && (bytes[i - 1] == b':' || bytes[i - 1] == b'-') {
                i += 1;
                continue;
            }
            // The typed tiers only consider markers that sit inside the
            // leading whitespace-free run — grep paths don't contain
            // whitespace, bodies routinely do.
            if tier != ScanTier::Permissive && bytes[..i].iter().any(|b| b.is_ascii_whitespace()) {
                break;
            }
            // Try this as the first separator. Walk through digits.
            let digits_start = i + 1;
            let mut j = digits_start;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            // The closing separator must match the opening one: grep
            // emits `file:12:body` or `file-12-body`, never a mix.
            let closes = j > digits_start
                && j < bytes.len()
                && match tier {
                    ScanTier::Permissive => bytes[j] == b':' || bytes[j] == b'-',
                    _ => bytes[j] == bytes[i],
                };
            if closes {
                // Found <sep><digits><sep>. Reject zero-length path
                // (line starts with separator).
                if i == 0 {
                    return None;
                }
                if first.is_none() {
                    first = Some((i, digits_start, j));
                }
                if tier != ScanTier::Dash {
                    chosen = Some((i, digits_start, j));
                    break;
                }
                // Dash tier: advance past this marker only on *positive*
                // evidence that the path runs through it. Stop when
                // either
                //   (a) the path so far ends in a segment with an
                //       extension — that marker is the boundary
                //       (`logs/2026-05-03/app.log-12-…`), or
                //   (b) the rest of the token carries no further path
                //       structure — no `/` and no `.` — so nothing
                //       suggests the path continues
                //       (`CHANGELOG-12-2026-05-03`).
                // Otherwise the digits are still inside the path
                // (`logs/2026` + `03/app.log…`) and a later marker is the
                // real one.
                //
                // Advancing on the *absence* of a stop signal is what
                // breaks here: neither whitespace nor an extension is
                // guaranteed to appear, so a walk bounded that way runs
                // off the end of the path and latches onto a triplet in
                // the *body* — silently, and for exactly the paths that
                // carry no extension (`Makefile`, `Dockerfile`,
                // `LICENSE`, `.github/workflows/ci`).
                if last_segment_has_extension(&line[..i]) || !path_continues(&line[j + 1..]) {
                    chosen = Some((i, digits_start, j));
                    break;
                }
                i = j + 1;
                continue;
            }
        }
        i += 1;
    }

    // Walked the whole token without ever confirming a boundary: the line
    // is genuinely ambiguous (`Makefile-7-include-2-src/foo.mk`). Fall back
    // to the leftmost marker — precisely what the leftmost-wins rule this
    // tier refines would have returned.
    let (path_end, digits_start, digits_end) = chosen.or(first)?;
    let line_no = std::str::from_utf8(&bytes[digits_start..digits_end])
        .ok()
        .and_then(|s| s.parse::<u64>().ok())?;
    Some((&line[..path_end], line_no, &line[digits_end + 1..]))
}

// ─── Internals ──────────────────────────────────────────────────────────

fn hash_u64(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

fn md5_hex_24(s: &str) -> String {
    let mut hasher = Md5::new();
    hasher.update(s.as_bytes());
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(32);
    for b in digest {
        hex.push_str(&format!("{:02x}", b));
    }
    hex.truncate(24);
    hex
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ccr::InMemoryCcrStore;

    fn parse_line(line: &str) -> Option<(String, u64, String)> {
        parse_match_line(line).map(|(f, n, c)| (f.to_string(), n, c.to_string()))
    }

    #[test]
    fn parses_standard_grep_line() {
        assert_eq!(
            parse_line("src/main.py:42:def main():"),
            Some(("src/main.py".into(), 42, "def main():".into()))
        );
    }

    #[test]
    fn cjk_bigrams_from_runs() {
        let b = cjk_bigrams("认证令牌");
        assert!(b.contains("认证") && b.contains("证令") && b.contains("令牌") && b.len() == 3);
        assert!(cjk_bigrams("hello").is_empty());
        assert!(cjk_bigrams("a认b证").is_empty()); // isolated CJK chars -> no pair
    }

    #[test]
    fn dash_tier_does_not_run_past_the_path_into_the_body() {
        // The dash tier keeps the *last* `-<digits>-` marker so that a
        // dated path (`logs/2026-05-03/…`) isn't split at its first
        // triplet. But a context-line *body* can be whitespace-free and
        // carry a triplet of its own — the walk must not follow it there.
        assert_eq!(
            parse_line("notes.md-3-see-4-here"),
            Some(("notes.md".into(), 3, "see-4-here".into()))
        );
        assert_eq!(
            parse_line("CHANGELOG.md-7-v1-2-beta"),
            Some(("CHANGELOG.md".into(), 7, "v1-2-beta".into()))
        );
    }

    #[test]
    fn parses_context_line_for_extensionless_path() {
        // Files with no extension are ordinary grep targets, and nothing in
        // such a line ever gives the walk an extension to stop at. Bounding
        // the walk on that signal alone splits the path *and* lifts the line
        // number out of the body — silently. The tier must instead fall back
        // to the leftmost marker for these.
        for (line, file, no, body) in [
            ("CHANGELOG-12-2026-05-03", "CHANGELOG", 12, "2026-05-03"),
            (
                "LICENSE-1-Copyright-2026-Acme",
                "LICENSE",
                1,
                "Copyright-2026-Acme",
            ),
            ("Makefile-7-build-2-stage", "Makefile", 7, "build-2-stage"),
            (
                "Dockerfile-3-FROM-18-alpine",
                "Dockerfile",
                3,
                "FROM-18-alpine",
            ),
            ("bin/run-3-exec-9-now", "bin/run", 3, "exec-9-now"),
            (
                ".github/workflows/ci-9-2026-01-02",
                ".github/workflows/ci",
                9,
                "2026-01-02",
            ),
        ] {
            assert_eq!(
                parse_line(line),
                Some((file.into(), no, body.into())),
                "extension-less path mis-parsed: {line}"
            );
        }
    }

    #[test]
    fn dated_dir_still_wins_for_an_extensionless_file() {
        // The case this tier exists for and the case above, on one line: the
        // walk must cross the dated directory (the rest of the token still
        // has a `/`) and stop at `Makefile`, which has no extension to
        // announce itself with.
        assert_eq!(
            parse_line("logs/2026-05-03/Makefile-12-all:"),
            Some(("logs/2026-05-03/Makefile".into(), 12, "all:".into()))
        );
    }

    #[test]
    fn ambiguous_dash_line_falls_back_to_the_leftmost_marker() {
        // No extension anywhere and a `/` still ahead: the walk never gets
        // positive evidence either way. It must then return what the
        // leftmost-wins rule returns rather than guessing — this tier is
        // never worse than the rule it refines.
        assert_eq!(
            parse_line("Makefile-7-include-2-src/foo.mk"),
            Some(("Makefile".into(), 7, "include-2-src/foo.mk".into()))
        );
    }

    #[test]
    fn a_dotted_version_in_the_body_is_not_an_extension() {
        // The evidence that "the path continues" must not fire on a dotted
        // *version* sitting in the body of an extension-less file. The
        // classic source is `git describe` output — `v1.2.3-4-gdeadbee`
        // carries both a dot and a `-<digits>-` run, so reading its `.3`
        // as an extension walks the parse clean out of the path and makes
        // the marker the `-4-`, inventing the file `Makefile-9-VERSION=v1.2.3`.
        // An extension needs a letter; `.3` and `.2` are not extensions.
        for line in [
            "Makefile-9-VERSION=v1.2.3-4-gdeadbee",
            "Dockerfile-4-TAG=v2.0.1-3-gabc",
            "CHANGELOG-2-rel.1.2-3-final",
            "LICENSE-1-v1.0-2-clause",
        ] {
            let (file, _, _) = parse_line(line).expect("parses");
            assert!(
                !file.contains('.'),
                "{line}: dotted version read as a path extension, got file {file:?}"
            );
        }

        assert_eq!(
            parse_line("Makefile-9-VERSION=v1.2.3-4-gdeadbee"),
            Some(("Makefile".into(), 9, "VERSION=v1.2.3-4-gdeadbee".into()))
        );
    }

    #[test]
    fn a_real_extension_still_counts_even_with_digits_in_it() {
        // The letter requirement must not cost us genuine extensions that
        // carry digits (`.mp3`, `.7z`, `.h264`): the walk still has to cross
        // the dated directory and stop at the file.
        assert_eq!(
            parse_line("media/2026-05-03/track-2-final.mp3-8-id3"),
            Some(("media/2026-05-03/track-2-final.mp3".into(), 8, "id3".into()))
        );
    }

    #[test]
    fn dated_path_inside_a_context_body_is_not_hijacked() {
        // Same shape as the paths this PR fixes, but on the *body* side of
        // the marker. Left unbounded, the scan reports a path that never
        // existed and a line number taken from a date.
        assert_eq!(
            parse_line("manifest.txt-5-logs/2026-05-03/app.log"),
            Some(("manifest.txt".into(), 5, "logs/2026-05-03/app.log".into()))
        );
        assert_eq!(
            parse_line("index.md-2-advisories/CVE-2021-44228.md"),
            Some(("index.md".into(), 2, "advisories/CVE-2021-44228.md".into()))
        );
    }

    #[test]
    fn parses_ripgrep_context_line() {
        assert_eq!(
            parse_line("src/main.py-43-context after match"),
            Some(("src/main.py".into(), 43, "context after match".into()))
        );
    }

    #[test]
    fn fixed_in_3e2_handles_windows_path_with_backslash() {
        // Pre-3e2 Python regex misread the drive colon as the
        // line-number-marker separator and silently dropped this line.
        assert_eq!(
            parse_line(r"C:\Users\foo\bar.py:42:def main():"),
            Some((r"C:\Users\foo\bar.py".into(), 42, "def main():".into()))
        );
    }

    #[test]
    fn fixed_in_3e2_handles_windows_path_with_forward_slash() {
        // Ripgrep on Windows often emits forward-slash paths.
        assert_eq!(
            parse_line("C:/Users/foo/bar.py:42:def main():"),
            Some(("C:/Users/foo/bar.py".into(), 42, "def main():".into()))
        );
    }

    #[test]
    fn fixed_in_3e2_handles_dashes_in_filename_with_ripgrep_context() {
        // Pre-3e2 `_RG_CONTEXT_PATTERN`'s `[^:-]+` excluded dashes from
        // the path, so this line was either misparsed (path truncated
        // at the first dash) or fell through.
        assert_eq!(
            parse_line("pre-commit-config.yaml-42-fail_fast: true"),
            Some((
                "pre-commit-config.yaml".into(),
                42,
                "fail_fast: true".into()
            ))
        );
    }

    #[test]
    fn date_stamped_path_is_not_misread_as_line_number_marker() {
        // `logs/2026-05-03/...` contains `-05-`, a `<sep><digits><sep>`
        // triplet INSIDE the path. Leftmost-wins parsing takes it as the
        // line-number marker and silently reports a bogus file, a bogus
        // line number, and a truncated body.
        assert_eq!(
            parse_line("logs/2026-05-03/app.log:12:ERROR boom"),
            Some(("logs/2026-05-03/app.log".into(), 12, "ERROR boom".into()))
        );
        // Same for the ripgrep context-line form.
        assert_eq!(
            parse_line("logs/2026-11-23/app.log-11-context before"),
            Some((
                "logs/2026-11-23/app.log".into(),
                11,
                "context before".into()
            ))
        );
        // And for version/ordinal path segments (`v1-2-beta`).
        assert_eq!(
            parse_line("src/v1-2-beta/mod.rs:3:fn x() {}"),
            Some(("src/v1-2-beta/mod.rs".into(), 3, "fn x() {}".into()))
        );
        // `NAME-<digits>-<digits>` paths (CVE advisories, timestamped
        // migrations) hide a triplet in the middle of the path token.
        assert_eq!(
            parse_line("advisories/CVE-2021-44228.md:8:Log4Shell"),
            Some(("advisories/CVE-2021-44228.md".into(), 8, "Log4Shell".into()))
        );
        assert_eq!(
            parse_line("migrations/20240101-002-add_users.sql-9-  id BIGINT"),
            Some((
                "migrations/20240101-002-add_users.sql".into(),
                9,
                "  id BIGINT".into()
            ))
        );
    }

    #[test]
    fn date_stamped_paths_are_not_collapsed_into_one_bogus_file() {
        // The misparse doesn't just corrupt one line — the bogus path
        // becomes the grouping key, so two unrelated files merge into a
        // single `logs/2026` bucket and the model sees a path that
        // doesn't exist. This is what reaches the LLM.
        let compressor = SearchCompressor::new(SearchCompressorConfig::default());
        let content = "\
logs/2026-05-03/app.log:12:ERROR boom
logs/2026-05-04/app.log:7:ERROR bang";
        let mut stats = SearchCompressorStats::default();
        let parsed = compressor.parse_search_results(content, &mut stats);

        assert_eq!(stats.lines_unparsed, 0);
        let files: Vec<&String> = parsed.keys().collect();
        assert_eq!(
            files,
            vec!["logs/2026-05-03/app.log", "logs/2026-05-04/app.log"]
        );
        assert_eq!(parsed["logs/2026-05-03/app.log"].matches[0].line_number, 12);
        assert_eq!(
            parsed["logs/2026-05-03/app.log"].matches[0].content,
            "ERROR boom"
        );
    }

    #[test]
    fn body_line_reference_does_not_hijack_a_context_line() {
        // A `-`-separated context line whose BODY quotes a `file:line:`
        // reference must still bind to the context marker, not the
        // reference. Guards the whitespace-bounded colon tier.
        assert_eq!(
            parse_line("src/main.py-44-see foo.rs:12:bar"),
            Some(("src/main.py".into(), 44, "see foo.rs:12:bar".into()))
        );
    }

    #[test]
    fn digit_terminated_path_still_parses_ripgrep_context_line() {
        // Rotated logs end in a digit, so the context separator is
        // digit-preceded — the last-marker rule must still land on it.
        assert_eq!(
            parse_line("logs/app.log.1-42-rotated line"),
            Some(("logs/app.log.1".into(), 42, "rotated line".into()))
        );
    }

    #[test]
    fn preserves_colons_in_match_content() {
        // Standard grep behavior: stop at the second separator, the rest
        // is content, even when content contains colons.
        assert_eq!(
            parse_line(r#"config.py:10:DATABASE_URL = "postgres://user:pass@host:5432/db""#),
            Some((
                "config.py".into(),
                10,
                r#"DATABASE_URL = "postgres://user:pass@host:5432/db""#.into()
            ))
        );
    }

    #[test]
    fn rejects_lines_without_line_number_marker() {
        assert!(parse_line("just a normal line of prose").is_none());
        assert!(parse_line("file.py:not-a-number:something").is_none());
        // Empty/zero-length path rejected:
        assert!(parse_line(":42:something").is_none());
    }

    #[test]
    fn rejects_negative_line_numbers() {
        // The `-` is part of the content, not a separator. Pre-3e2 the
        // adjacent-separator collapse rule didn't exist; a stray fix
        // could have re-introduced this regression.
        assert!(parse_line("src/file.py:-1:invalid").is_none());
        // Equivalent form with the dash adjacent to the dash separator.
        assert!(parse_line("src/file.py--1-invalid").is_none());
    }

    #[test]
    fn parser_groups_by_file_and_counts() {
        let compressor = SearchCompressor::new(SearchCompressorConfig::default());
        let content = "\
src/main.py:42:def main():
src/main.py:43:    pass
src/utils.py:15:def util():
just prose, no marker
src/main.py-44-context line";
        let mut stats = SearchCompressorStats::default();
        let parsed = compressor.parse_search_results(content, &mut stats);
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed["src/main.py"].matches.len(), 3);
        assert_eq!(parsed["src/utils.py"].matches.len(), 1);
        assert_eq!(stats.lines_unparsed, 1);
        assert_eq!(stats.lines_scanned, 5);
    }

    #[test]
    fn scoring_boosts_error_lines_in_search_context() {
        let compressor = SearchCompressor::new(SearchCompressorConfig {
            context_keywords: vec!["auth".into()],
            ..Default::default()
        });
        let mut files = BTreeMap::new();
        let mut fm = FileMatches::new("src/auth.py");
        fm.matches
            .push(SearchMatch::new("src/auth.py", 10, "ERROR auth failed"));
        fm.matches
            .push(SearchMatch::new("src/auth.py", 11, "plain auth line"));
        files.insert("src/auth.py".into(), fm);

        compressor.score_matches(&mut files, "find auth error");
        let scored = &files["src/auth.py"].matches;
        // ERROR + auth-keyword + context-word "error" + context-word
        // "auth" all hit; clamped to 1.0.
        assert_eq!(scored[0].score, 1.0);
        // Plain line gets only context-word + keyword boosts (no error).
        assert!(scored[1].score > 0.0 && scored[1].score < 1.0);
    }

    #[test]
    fn select_respects_per_file_cap_and_global_cap() {
        // Note: compute_optimal_k enforces a hard `min_k=5` floor (matches
        // Python `_select_matches`), so `max_total_matches` is a soft cap
        // that bites only above that floor. Configure 6 here to exercise
        // the cap path.
        let compressor = SearchCompressor::new(SearchCompressorConfig {
            max_matches_per_file: 2,
            max_total_matches: 6,
            max_files: 2,
            always_keep_first: true,
            always_keep_last: true,
            ..Default::default()
        });
        let mut files = BTreeMap::new();
        for (file, n) in [("a.py", 5), ("b.py", 4), ("c.py", 3)] {
            let mut fm = FileMatches::new(file);
            for i in 0..n {
                fm.matches
                    .push(SearchMatch::new(file, i + 1, format!("line {}", i + 1)));
            }
            files.insert(file.into(), fm);
        }

        let mut stats = SearchCompressorStats::default();
        let selected = compressor.select_matches(&files, 1.0, &mut stats);

        // max_files=2 caps surviving files; one of three is dropped.
        assert_eq!(selected.len(), 2);
        assert!(stats.files_dropped >= 1);
        // Each surviving file is capped at max_matches_per_file=2.
        for fm in selected.values() {
            assert!(fm.matches.len() <= 2);
            // Survivors output in line order.
            assert!(fm
                .matches
                .windows(2)
                .all(|w| w[0].line_number < w[1].line_number));
        }
    }

    #[test]
    fn empty_input_returns_unchanged() {
        let compressor = SearchCompressor::new(SearchCompressorConfig::default());
        let (result, _) = compressor.compress("plain text only", "", 1.0);
        assert_eq!(result.original_match_count, 0);
        assert_eq!(result.compressed, "plain text only");
        assert_eq!(result.compression_ratio, 1.0);
    }

    #[test]
    fn ccr_marker_emitted_when_thresholds_clear() {
        let compressor = SearchCompressor::new(SearchCompressorConfig {
            max_matches_per_file: 2,
            max_total_matches: 4,
            min_matches_for_ccr: 5,
            min_compression_ratio_for_ccr: 0.95, // very permissive for the test
            ..Default::default()
        });
        let mut content = String::new();
        for i in 1..=12 {
            content.push_str(&format!("src/main.py:{}:line content {}\n", i, i));
        }
        let store = InMemoryCcrStore::new();
        let (result, stats) = compressor.compress_with_store(&content, "", 1.0, Some(&store));
        assert!(result.cache_key.is_some());
        assert!(stats.ccr_emitted);
        assert!(result.compressed.contains("[12 matches compressed to"));
        // Round-trip via the store.
        let key = result.cache_key.as_ref().unwrap();
        assert_eq!(store.get(key).unwrap(), content);
    }

    #[test]
    fn ccr_skipped_when_below_min_matches() {
        let compressor = SearchCompressor::new(SearchCompressorConfig {
            min_matches_for_ccr: 100,
            ..Default::default()
        });
        let content = "src/main.py:1:hi\nsrc/main.py:2:bye\n";
        let store = InMemoryCcrStore::new();
        let (_, stats) = compressor.compress_with_store(content, "", 1.0, Some(&store));
        assert!(!stats.ccr_emitted);
        assert_eq!(stats.ccr_skip_reason, Some("below min_matches_for_ccr"));
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn ccr_skipped_when_disabled() {
        let compressor = SearchCompressor::new(SearchCompressorConfig {
            enable_ccr: false,
            ..Default::default()
        });
        let mut content = String::new();
        for i in 1..=20 {
            content.push_str(&format!("src/main.py:{}:line\n", i));
        }
        let store = InMemoryCcrStore::new();
        let (_, stats) = compressor.compress_with_store(&content, "", 1.0, Some(&store));
        assert!(!stats.ccr_emitted);
        assert_eq!(stats.ccr_skip_reason, Some("ccr disabled in config"));
    }
}
