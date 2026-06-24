//! Unified-diff compressor — Rust port of `headroom.transforms.diff_compressor`.
//!
//! Compresses verbose `git diff` output by:
//! 1. Parsing the unified-diff format into files + hunks.
//! 2. Capping the file count (`max_files`) — when fired, sorts by total
//!    changes and keeps the heaviest files.
//! 3. Capping per-file hunk count (`max_hunks_per_file`) — keeps first +
//!    last + top-scored middle hunks (relevance-aware via priority patterns
//!    + user query-context word overlap).
//! 4. Trimming context lines around each `+`/`-` to `max_context_lines`
//!    on either side.
//! 5. Hashing the original with MD5 truncated to 24 hex chars for a CCR
//!    cache_key (only emitted if compression saved >20% of lines).
//!
//! # Parity contract
//! Output bytes (`compressed` field + all numeric counts) must be
//! byte-identical to the Python implementation. The 20 fixtures in
//! `tests/parity/fixtures/diff_compressor/` are the spec.
//!
//! # Information preservation hardening (no parity impact)
//! - Below `min_lines_for_ccr`, we return the input unchanged (matches
//!   Python). Important for short diffs that don't benefit from compression
//!   and would lose context-trim slack.
//! - On parse failure (no `diff --git` headers found), we return the input
//!   unchanged (matches Python). Malformed input is preserved verbatim.
//! - `\ No newline at end of file` markers and any other non-`+`/`-`/space
//!   "other" lines inside a hunk are appended to the hunk's lines. Whether
//!   they survive the context trim is determined by their distance from
//!   the nearest `+`/`-` line (matches Python `_reduce_context`).
//!
//! # Observability
//! [`DiffCompressorStats`] carries the granular metrics Python doesn't
//! emit (per-file hunk drop counts, dropped file names, context lines
//! trimmed, parse warnings, processing duration). The `compress_with_stats`
//! method returns it alongside the parity-equal result; `compress` is the
//! parity-only API that just emits a `tracing::info_span`.

use std::collections::BTreeMap;
use std::sync::OnceLock;
use std::time::Instant;

use md5::{Digest, Md5};
use regex::Regex;

use crate::ccr::CcrStore;

// ─── Score-weight constants ────────────────────────────────────────────────
//
// These knobs tune the relevance scorer (used only when `max_hunks_per_file`
// fires and we have to rank middle hunks). Promoted from inline magic numbers
// so a future tuning PR can move them with full visibility, and so reviewers
// can see exactly what bias the scorer encodes. Defaults match the Python
// implementation byte-for-byte.

/// Per-line-change weight in the change-density base term. The base score is
/// `min(CHANGE_DENSITY_CAP, change_count * CHANGE_DENSITY_WEIGHT)`.
pub const SCORE_CHANGE_DENSITY_WEIGHT: f64 = 0.03;
/// Cap on the change-density base term; beyond this, additional changes
/// don't keep raising the score.
pub const SCORE_CHANGE_DENSITY_CAP: f64 = 0.3;
/// Boost added per matching word from the user-query context that appears in
/// the hunk content (case-insensitive substring match).
pub const SCORE_CONTEXT_WORD_WEIGHT: f64 = 0.2;
/// Minimum word length (exclusive of) for context-word matching. Words of
/// length ≤ this are skipped (matches Python's `len(word) > 2`). Filters out
/// stop-words like "is", "to", "a".
pub const SCORE_CONTEXT_MIN_WORD_LEN: usize = 2;
/// Boost added when ANY priority pattern matches (only one boost per hunk —
/// matches Python's `break` after first match).
pub const SCORE_PRIORITY_PATTERN_BOOST: f64 = 0.3;
/// Cap on the total hunk score after all boosts.
pub const SCORE_TOTAL_CAP: f64 = 1.0;

// ─── Public API ─────────────────────────────────────────────────────────────

/// Configuration. All defaults match Python `DiffCompressorConfig`.
#[derive(Debug, Clone)]
pub struct DiffCompressorConfig {
    /// How many context lines (` ` prefix) to keep on either side of each
    /// `+`/`-` change line. Python default: 2.
    pub max_context_lines: usize,
    /// Cap on the number of hunks kept per file. When exceeded, keeps first
    /// + last + top-scored middle. Python default: 10.
    pub max_hunks_per_file: usize,
    /// Cap on the number of files kept across the whole diff. When exceeded,
    /// sorts files by total changes (desc) and keeps top N. Files beyond
    /// the cap are silently dropped from the output (their names appear in
    /// [`DiffCompressorStats::files_dropped`] for observability). Python
    /// default: 20.
    pub max_files: usize,
    /// Reserved — Python config exposes this but the algorithm always keeps
    /// `+` lines. Kept for fixture-config schema compatibility.
    pub always_keep_additions: bool,
    /// Reserved — same as `always_keep_additions` but for `-` lines.
    pub always_keep_deletions: bool,
    /// If true, attach an MD5-based CCR retrieval marker to the compressed
    /// output when compression met the savings threshold. Python default: true.
    pub enable_ccr: bool,
    /// **Misnomer alert.** This actually gates the entire compression path,
    /// not just the CCR marker: when `original_line_count <
    /// min_lines_for_ccr`, the input is returned unchanged, no parsing, no
    /// summary, no CCR. The name comes from the Python implementation; we
    /// keep it to maintain fixture-config compatibility. Treat it as
    /// "minimum diff size before we bother compressing." Python default: 50.
    pub min_lines_for_ccr: usize,
    /// CCR retrieval marker is emitted only when
    /// `compressed_line_count < original_line_count *
    /// min_compression_ratio_for_ccr`. Lower values demand more aggressive
    /// compression before we bother emitting the marker. **Rust-only knob**
    /// — Python hardcodes 0.8. Default: 0.8 (matches Python).
    pub min_compression_ratio_for_ccr: f64,
}

impl Default for DiffCompressorConfig {
    fn default() -> Self {
        Self {
            max_context_lines: 2,
            max_hunks_per_file: 10,
            max_files: 20,
            always_keep_additions: true,
            always_keep_deletions: true,
            enable_ccr: true,
            min_lines_for_ccr: 50,
            min_compression_ratio_for_ccr: 0.8,
        }
    }
}

/// Parity-equal result. Field set matches Python `DiffCompressionResult`'s
/// non-`@property` fields — `compression_ratio` and `tokens_saved_estimate`
/// are computed properties on the Python side and not in fixture outputs.
#[derive(Debug, Clone)]
pub struct DiffCompressionResult {
    pub compressed: String,
    pub original_line_count: usize,
    pub compressed_line_count: usize,
    pub files_affected: usize,
    pub additions: usize,
    pub deletions: usize,
    pub hunks_kept: usize,
    pub hunks_removed: usize,
    pub cache_key: Option<String>,
}

/// Sidecar stats. Not part of the parity output; surfaced to callers and
/// `tracing` spans for prod observability. None of these fields exist in
/// Python's `DiffCompressionResult`.
#[derive(Debug, Clone, Default)]
pub struct DiffCompressorStats {
    pub input_lines: usize,
    pub output_lines: usize,
    /// `output_lines / input_lines`. 1.0 means no compression (or input was
    /// returned unchanged); lower is more aggressive compression.
    pub compression_ratio: f64,

    pub files_total: usize,
    pub files_kept: usize,
    /// Names (`old_file -> new_file` from the diff header) of files dropped
    /// when `max_files` fired. Empty unless that cap engaged.
    pub files_dropped: Vec<String>,

    pub hunks_total: usize,
    pub hunks_kept: usize,
    pub hunks_dropped: usize,
    /// Per-file hunk drop counts. Stable iteration order via `BTreeMap`.
    pub hunks_dropped_per_file: BTreeMap<String, usize>,

    pub context_lines_input: usize,
    pub context_lines_kept: usize,
    pub context_lines_trimmed: usize,

    /// Lines in the largest hunk we kept. Useful for spotting cases where
    /// a single oversized hunk dominates the output.
    pub largest_hunk_kept_lines: usize,
    /// Lines in the largest hunk we dropped (per-file cap). 0 if none dropped.
    pub largest_hunk_dropped_lines: usize,

    /// Files whose original `new file mode` / `deleted file mode` line was
    /// normalized to `100644` on output. Each entry is `(file_label,
    /// original_mode_line)` — e.g. `("a/foo.sh -> b/foo.sh", "new file
    /// mode 100755")`. Empty when no normalization occurred (mode was
    /// already 100644, or no mode line was present).
    ///
    /// Why this matters: parity with Python forces us to hardcode `100644`
    /// in the emit path regardless of what the input said. An input with
    /// executable bit `100755` becomes a non-executable `100644` on output —
    /// silent information loss. Surfacing this lets prod monitoring catch
    /// real cases where it bites.
    pub file_mode_normalizations: Vec<(String, String)>,

    /// Binary file marker lines whose original detail (e.g. `Binary files
    /// a/x.png and b/x.png differ`) was simplified to `Binary files differ`
    /// on output. Each entry is the full original line. Empty when no
    /// simplification occurred (input was already `Binary files differ`,
    /// or the file wasn't binary).
    ///
    /// Same parity-loss pattern as `file_mode_normalizations`: Python's
    /// emitter hardcodes `Binary files differ`, dropping the filename
    /// detail. This stat surfaces what was lost.
    pub binary_files_simplified: Vec<String>,

    /// Non-fatal parser hiccups: unrecognized line patterns, malformed
    /// hunk headers, etc. Surfacing them rather than dropping silently.
    pub parse_warnings: Vec<String>,

    pub processing_duration_us: u64,

    /// True if the CCR cache_key was attached to the output (compression
    /// saved >20% of lines AND `enable_ccr` was true).
    pub cache_key_emitted: bool,
    /// When `cache_key_emitted == false`, why. e.g. `"below threshold"`,
    /// `"ccr disabled"`, `"input below min_lines_for_ccr"`.
    pub ccr_skipped_reason: Option<String>,
}

/// Compressor. Cheap to clone; holds only the config.
#[derive(Debug, Clone)]
pub struct DiffCompressor {
    config: DiffCompressorConfig,
}

impl Default for DiffCompressor {
    fn default() -> Self {
        Self::new(DiffCompressorConfig::default())
    }
}

impl DiffCompressor {
    pub fn new(config: DiffCompressorConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &DiffCompressorConfig {
        &self.config
    }

    /// Compress `content`. `context` is an optional user-query string used
    /// for relevance scoring when `max_hunks_per_file` fires; pass `""` if
    /// not applicable. Parity-only API: emits a `tracing::info_span` but
    /// discards the granular sidecar stats. Use [`compress_with_stats`]
    /// when you want them.
    ///
    /// [`compress_with_stats`]: Self::compress_with_stats
    pub fn compress(&self, content: &str, context: &str) -> DiffCompressionResult {
        self.compress_with_stats(content, context).0
    }

    /// Same as [`compress`] but also returns rich observability stats.
    /// Equivalent to `compress_with_store(content, context, None).0`.
    ///
    /// [`compress`]: Self::compress
    pub fn compress_with_stats(
        &self,
        content: &str,
        context: &str,
    ) -> (DiffCompressionResult, DiffCompressorStats) {
        self.compress_with_store(content, context, None)
    }

    /// Compress with optional CCR persistence.
    ///
    /// Mirrors [`crate::transforms::log_compressor::LogCompressor::compress_with_store`]
    /// and the search-compressor sibling: when `store` is `Some`, the
    /// original `content` is written to it under the same `cache_key`
    /// the wire-output marker carries. When `store` is `None`, the
    /// `cache_key` is still emitted but the caller is responsible for
    /// persisting (e.g. the Python shim's `_persist_to_python_ccr` path).
    ///
    /// **Why this exists:** prior to this method, `DiffCompressor` minted
    /// a `cache_key` and embedded it in the output marker but never
    /// stored — leaving Python ContentRouter with a dangling marker that
    /// would 404 on retrieval. The `DiffOffload` orchestrator wrapper
    /// papered over this for the new pipeline path; this method
    /// upstreams the fix so any caller can wire in storage cleanly.
    pub fn compress_with_store(
        &self,
        content: &str,
        context: &str,
        store: Option<&dyn CcrStore>,
    ) -> (DiffCompressionResult, DiffCompressorStats) {
        let start = Instant::now();
        let mut stats = DiffCompressorStats::default();

        // Python: `lines = content.split("\n")`. Rust `split` matches `str.split`
        // semantics: a trailing newline produces an empty final element, exactly
        // like Python. Critical for byte-equal `original_line_count`.
        let lines: Vec<&str> = content.split('\n').collect();
        let original_line_count = lines.len();
        stats.input_lines = original_line_count;

        // Short-circuit 1: input below CCR threshold → pass through unchanged.
        // This is the information-preservation path: a 5-line diff isn't worth
        // compressing and the original carries all the signal.
        if original_line_count < self.config.min_lines_for_ccr {
            stats.output_lines = original_line_count;
            stats.compression_ratio = 1.0;
            stats.ccr_skipped_reason = Some("input below min_lines_for_ccr".into());
            stats.processing_duration_us = start.elapsed().as_micros() as u64;
            return (
                pass_through_result(content, original_line_count),
                emit_span_and_return(stats),
            );
        }

        // Parse the unified diff into files + hunks (and any pre-diff
        // content — commit headers, email headers — which gets re-emitted
        // verbatim by `format_output`).
        let parsed = parse_diff(&lines);
        let pre_diff_lines = parsed.pre_diff_lines;
        let mut diff_files = parsed.files;
        stats.parse_warnings = parsed.parse_warnings;
        stats.files_total = diff_files.len();
        stats.hunks_total = diff_files.iter().map(|f| f.hunks.len()).sum();
        stats.context_lines_input = diff_files
            .iter()
            .flat_map(|f| f.hunks.iter())
            .map(|h| h.context_lines)
            .sum();

        // Short-circuit 2: parser found no diff sections → pass through.
        // Same info-preservation rationale: malformed or non-diff input is
        // returned verbatim rather than emitted as an empty compressed result.
        if diff_files.is_empty() {
            stats.output_lines = original_line_count;
            stats.compression_ratio = 1.0;
            stats.ccr_skipped_reason = Some("no diff sections parsed".into());
            stats.processing_duration_us = start.elapsed().as_micros() as u64;
            return (
                pass_through_result(content, original_line_count),
                emit_span_and_return(stats),
            );
        }

        // Score hunks by relevance to the user query (used only if
        // `max_hunks_per_file` fires).
        score_hunks(&mut diff_files, context);

        // File cap: if too many, sort by total changes (most first) and
        // keep the top `max_files`. The dropped files' names are kept in
        // stats for observability — Python silently discards them.
        if diff_files.len() > self.config.max_files {
            diff_files.sort_by(|a, b| {
                let a_changes = a.total_additions() + a.total_deletions();
                let b_changes = b.total_additions() + b.total_deletions();
                b_changes.cmp(&a_changes)
            });
            let dropped: Vec<DiffFile> = diff_files.split_off(self.config.max_files);
            stats.files_dropped = dropped
                .iter()
                .map(|f| format!("{} -> {}", f.old_file, f.new_file))
                .collect();
        }
        stats.files_kept = diff_files.len();

        // Capture lossy-emit signals on the files that survived the file cap.
        // These cases are parity-bound (Python's emit hardcodes `100644` and
        // `Binary files differ` regardless of input), so the only honest move
        // is to surface the loss via observability rather than fix it.
        for file in diff_files.iter() {
            let label = format!("{} -> {}", file.old_file, file.new_file);
            // File-mode normalization: any original mode line not literally
            // `new file mode 100644` / `deleted file mode 100644` is lost on
            // emit. Includes `100755` (executable), `100600` (private),
            // `120000` (symlink), `160000` (gitlink/submodule), etc.
            if let Some(orig) = &file.original_new_file_mode_line {
                if orig != "new file mode 100644" {
                    stats
                        .file_mode_normalizations
                        .push((label.clone(), orig.clone()));
                }
            }
            if let Some(orig) = &file.original_deleted_file_mode_line {
                if orig != "deleted file mode 100644" {
                    stats
                        .file_mode_normalizations
                        .push((label.clone(), orig.clone()));
                }
            }
            // Binary detail: any line richer than the bare `Binary files
            // differ` (which is virtually all of them — git emits filenames)
            // gets simplified on emit.
            if let Some(orig) = &file.original_binary_line {
                if orig != "Binary files differ" {
                    stats.binary_files_simplified.push(orig.clone());
                }
            }
        }

        // Compress each file's hunks: cap count, then trim context.
        let mut compressed_files: Vec<DiffFile> = Vec::with_capacity(diff_files.len());
        let mut total_additions = 0usize;
        let mut total_deletions = 0usize;
        let mut hunks_kept_total = 0usize;
        let mut hunks_removed_total = 0usize;
        let mut largest_kept = 0usize;
        let mut largest_dropped = 0usize;
        let mut context_kept_total = 0usize;

        for file in diff_files {
            total_additions += file.total_additions();
            total_deletions += file.total_deletions();

            let original_hunk_count = file.hunks.len();
            let file_label = format!("{} -> {}", file.old_file, file.new_file);

            let (selected, dropped) = select_hunks(file.hunks, self.config.max_hunks_per_file);
            let dropped_count = dropped.len();
            if dropped_count > 0 {
                stats
                    .hunks_dropped_per_file
                    .insert(file_label, dropped_count);
                let max_dropped = dropped.iter().map(|h| h.lines.len()).max().unwrap_or(0);
                if max_dropped > largest_dropped {
                    largest_dropped = max_dropped;
                }
            }

            // Trim context inside each kept hunk.
            let mut compressed_hunks: Vec<DiffHunk> = Vec::with_capacity(selected.len());
            for hunk in selected {
                let trimmed = reduce_context(&hunk, self.config.max_context_lines);
                if trimmed.lines.len() > largest_kept {
                    largest_kept = trimmed.lines.len();
                }
                context_kept_total += trimmed.context_lines;
                compressed_hunks.push(trimmed);
            }

            hunks_kept_total += compressed_hunks.len();
            hunks_removed_total += original_hunk_count - compressed_hunks.len();

            compressed_files.push(DiffFile {
                hunks: compressed_hunks,
                ..file
            });
        }

        stats.hunks_kept = hunks_kept_total;
        stats.hunks_dropped = hunks_removed_total;
        stats.context_lines_kept = context_kept_total;
        stats.context_lines_trimmed = stats.context_lines_input.saturating_sub(context_kept_total);
        stats.largest_hunk_kept_lines = largest_kept;
        stats.largest_hunk_dropped_lines = largest_dropped;

        let files_affected = compressed_files.len();

        // Format compressed output. The footer summary line goes inside
        // `compressed`; the CCR retrieval marker (if present) is appended
        // after — both must match Python's emitter byte-for-byte.
        let mut compressed_output = format_output(
            &pre_diff_lines,
            &compressed_files,
            files_affected,
            total_additions,
            total_deletions,
            hunks_removed_total,
        );
        let compressed_line_count = count_split_lines(&compressed_output);

        // CCR layer: hash original with MD5[:24], append retrieval marker
        // *only* if compression met `min_compression_ratio_for_ccr`. Python
        // hardcodes 0.8 (>20% savings); we expose it as a config knob with
        // the same default.
        //
        // CRITICAL: `compressed_line_count` is captured BEFORE the CCR marker
        // is appended, both for the marker's own text ("compressed to N")
        // and for the result field. The output string ends up with one more
        // line than `compressed_line_count` reports, by design — Python
        // does the same. Mismatching this by recounting after the append
        // breaks parity by 1.
        let savings_threshold = self.config.min_compression_ratio_for_ccr;
        let mut cache_key: Option<String> = None;
        if self.config.enable_ccr
            && (compressed_line_count as f64) < (original_line_count as f64) * savings_threshold
        {
            let key = md5_hex_24(content);
            compressed_output.push('\n');
            compressed_output.push_str(&format!(
                "[{} lines compressed to {}. Retrieve full diff: hash={}]",
                original_line_count, compressed_line_count, key
            ));
            // Persist the original under the same key. When `store` is
            // `Some`, the marker we just emitted resolves through it on
            // the LLM's retrieval tool call. When `None`, the caller
            // (typically the Python shim) is responsible — see the
            // method-level docs.
            if let Some(s) = store {
                s.put(&key, content);
            }
            cache_key = Some(key);
            stats.cache_key_emitted = true;
        } else if !self.config.enable_ccr {
            stats.ccr_skipped_reason = Some("ccr disabled".into());
        } else {
            stats.ccr_skipped_reason = Some(format!(
                "compression ratio {:.3} above threshold {:.3}",
                if original_line_count == 0 {
                    1.0
                } else {
                    compressed_line_count as f64 / original_line_count as f64
                },
                savings_threshold
            ));
        }

        stats.output_lines = compressed_line_count;
        stats.compression_ratio = if original_line_count == 0 {
            1.0
        } else {
            compressed_line_count as f64 / original_line_count as f64
        };
        stats.processing_duration_us = start.elapsed().as_micros() as u64;

        let result = DiffCompressionResult {
            compressed: compressed_output,
            original_line_count,
            compressed_line_count,
            files_affected,
            additions: total_additions,
            deletions: total_deletions,
            hunks_kept: hunks_kept_total,
            hunks_removed: hunks_removed_total,
            cache_key,
        };

        (result, emit_span_and_return(stats))
    }
}

// ─── Internal types ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct DiffHunk {
    header: String,
    lines: Vec<String>,
    additions: usize,
    deletions: usize,
    context_lines: usize,
    /// Relevance score; only meaningful when `max_hunks_per_file` fires.
    score: f64,
}

#[derive(Debug, Clone)]
struct DiffFile {
    header: String,
    old_file: String,
    new_file: String,
    hunks: Vec<DiffHunk>,
    is_binary: bool,
    is_new_file: bool,
    is_deleted_file: bool,
    is_renamed: bool,
    /// Bug-fix: rename / similarity / dissimilarity / copy marker lines
    /// captured verbatim from the parser (e.g. `rename from old.py`,
    /// `rename to new.py`, `similarity index 95%`). Re-emitted after the
    /// `diff --git` header so the LLM can see that a file was renamed
    /// rather than modified-in-place. Previously dropped in both Python
    /// and Rust — fixed in both as part of the same change.
    rename_lines: Vec<String>,
    /// Full original `new file mode <NNNNNN>` line if present. Captured so
    /// we can detect when emit-time normalization to `100644` lost the
    /// executable bit (or any other mode signal).
    original_new_file_mode_line: Option<String>,
    /// Full original `deleted file mode <NNNNNN>` line if present.
    original_deleted_file_mode_line: Option<String>,
    /// Full original `Binary files X and Y differ` line if present.
    /// Captured so we can detect when emit simplifies to `Binary files differ`.
    original_binary_line: Option<String>,
}

impl DiffFile {
    fn total_additions(&self) -> usize {
        self.hunks.iter().map(|h| h.additions).sum()
    }
    fn total_deletions(&self) -> usize {
        self.hunks.iter().map(|h| h.deletions).sum()
    }
}

// ─── Parser ────────────────────────────────────────────────────────────────

/// Matches any hunk header — regular `@@ -A,B +C,D @@` AND combined-diff
/// `@@@ -A,B -C,D +E,F @@@` (3-way merge) and `@@@@ ... @@@@` (4-way merge,
/// extremely rare).
///
/// Bug-fix: the previous regex only matched `@@`, so combined-diff hunks
/// from merge commits had ALL their content silently dropped — `current_hunk`
/// was never set, so subsequent +/- lines fell through to the no-op branch.
/// Fixed in tandem with the Python source.
///
/// Implementation note: Rust's `regex` crate (RE2-based) doesn't support
/// backreferences, so the matched-pair count of `@`s on each side is
/// hand-rolled as alternation. Octopus merges with >3 parents (5+ `@`s)
/// would slip through this regex back to the content-line branch — they're
/// vanishingly rare in practice (n-parent merges with n>3 essentially
/// never appear in real repos). When they do, a `parse_warnings` entry is
/// emitted so prod monitoring can flag the case rather than silently
/// dropping it.
fn hunk_header_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(concat!(
            r"^(?:",
            // Regular @@ ... @@
            r"@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@",
            r"|",
            // 3-way merge @@@ ... @@@
            r"@@@ -\d+(?:,\d+)? -\d+(?:,\d+)? \+\d+(?:,\d+)? @@@",
            r"|",
            // 4-way merge @@@@ ... @@@@
            r"@@@@ -\d+(?:,\d+)? -\d+(?:,\d+)? -\d+(?:,\d+)? \+\d+(?:,\d+)? @@@@",
            r")(.*)$"
        ))
        .expect("static regex compiles")
    })
}

/// Extracts the new-file starting line number (`+N`) from any hunk header,
/// regardless of whether it's regular or combined-diff. Used for in-order
/// resort after middle-hunk selection.
fn hunk_new_range_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\+(\d+)").expect("static regex compiles"))
}

fn diff_git_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^diff --git a/(.+) b/(.+)$").expect("static regex compiles"))
}

/// Bug-fix (2026-04-25): merge-commit headers `diff --combined <path>` and
/// `diff --cc <path>`. Single-path file diffs paired with combined-diff
/// hunk syntax (`@@@`+). Previously not recognized — merge diffs from
/// `git log -p` were treated as a single non-diff blob because the header
/// didn't match `diff --git`, so they fell into pre-diff content.
fn diff_combined_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^diff --combined (.+)$").expect("static regex compiles"))
}

fn diff_cc_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^diff --cc (.+)$").expect("static regex compiles"))
}

/// Returns true if `line` is any kind of `diff --…` file-section header
/// (regular `--git`, or the combined-diff `--combined` / `--cc` variants).
fn is_diff_header(line: &str) -> bool {
    diff_git_regex().is_match(line)
        || diff_combined_regex().is_match(line)
        || diff_cc_regex().is_match(line)
}

fn old_file_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^--- (a/(.+)|/dev/null)$").expect("static regex compiles"))
}

fn new_file_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\+\+\+ (b/(.+)|/dev/null)$").expect("static regex compiles"))
}

fn binary_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^Binary files .+ differ$").expect("static regex compiles"))
}

/// Parser output: pre-diff content (commit headers, email headers from
/// `git format-patch`, anything before the first `diff --git`), plus the
/// parsed file structures. Bug-fix: pre-diff content used to be dropped
/// silently; now preserved verbatim and re-emitted by `format_output`.
struct ParsedDiff {
    pre_diff_lines: Vec<String>,
    files: Vec<DiffFile>,
    parse_warnings: Vec<String>,
}

fn parse_diff(lines: &[&str]) -> ParsedDiff {
    let mut files: Vec<DiffFile> = Vec::new();
    let mut current_file: Option<DiffFile> = None;
    let mut current_hunk: Option<DiffHunk> = None;
    let mut pre_diff_lines: Vec<String> = Vec::new();
    let warnings: Vec<String> = Vec::new();

    for &line in lines {
        // New file section. Includes regular `diff --git` AND merge-commit
        // `diff --combined <path>` / `diff --cc <path>` (bug-fix
        // 2026-04-25). Without these, merge diffs from `git log -p` got
        // treated as one giant pre-diff blob and never reached the
        // hunk-parsing path.
        if is_diff_header(line) {
            if let Some(h) = current_hunk.take() {
                if let Some(f) = current_file.as_mut() {
                    f.hunks.push(h);
                }
            }
            if let Some(f) = current_file.take() {
                files.push(f);
            }
            current_file = Some(DiffFile {
                header: line.to_string(),
                old_file: String::new(),
                new_file: String::new(),
                hunks: Vec::new(),
                is_binary: false,
                is_new_file: false,
                is_deleted_file: false,
                is_renamed: false,
                rename_lines: Vec::new(),
                original_new_file_mode_line: None,
                original_deleted_file_mode_line: None,
                original_binary_line: None,
            });
            continue;
        }

        // Bug-fix: lines before the first `diff --git` are pre-diff content
        // (commit metadata, email headers, etc.) — capture verbatim rather
        // than drop silently. They get re-emitted at the head of the
        // compressed output.
        if current_file.is_none() {
            pre_diff_lines.push(line.to_string());
            continue;
        }

        // File-level mode/binary/rename markers. Capture the full original
        // line in addition to the boolean — Python only sets the flag and
        // discards the actual mode/detail, but we want the original around
        // so we can surface emit-time normalizations as observability.
        if let Some(f) = current_file.as_mut() {
            if line.starts_with("new file mode") {
                f.is_new_file = true;
                f.original_new_file_mode_line = Some(line.to_string());
            } else if line.starts_with("deleted file mode") {
                f.is_deleted_file = true;
                f.original_deleted_file_mode_line = Some(line.to_string());
            } else if line.starts_with("rename ")
                || line.starts_with("similarity ")
                || line.starts_with("copy ")
                || line.starts_with("dissimilarity ")
            {
                // Bug-fix: capture the rename / similarity / dissimilarity /
                // copy marker lines verbatim. Previously only `is_renamed`
                // was set and the lines were dropped, so emit looked like
                // a plain modification.
                f.is_renamed = true;
                f.rename_lines.push(line.to_string());
            } else if binary_regex().is_match(line) {
                f.is_binary = true;
                f.original_binary_line = Some(line.to_string());
            }
        }

        // `--- a/file` or `--- /dev/null`.
        if old_file_regex().is_match(line) {
            if let Some(f) = current_file.as_mut() {
                f.old_file = line.to_string();
            }
            continue;
        }

        // `+++ b/file` or `+++ /dev/null`.
        if new_file_regex().is_match(line) {
            if let Some(f) = current_file.as_mut() {
                f.new_file = line.to_string();
            }
            continue;
        }

        // Hunk header.
        if hunk_header_regex().is_match(line) {
            if let Some(h) = current_hunk.take() {
                if let Some(f) = current_file.as_mut() {
                    f.hunks.push(h);
                }
            }
            current_hunk = Some(DiffHunk {
                header: line.to_string(),
                lines: Vec::new(),
                additions: 0,
                deletions: 0,
                context_lines: 0,
                score: 0.0,
            });
            continue;
        }

        // Hunk content.
        if let Some(h) = current_hunk.as_mut() {
            if line.starts_with('+') && !line.starts_with("+++") {
                h.additions += 1;
                h.lines.push(line.to_string());
            } else if line.starts_with('-') && !line.starts_with("---") {
                h.deletions += 1;
                h.lines.push(line.to_string());
            } else if line.starts_with(' ') || line.is_empty() {
                h.context_lines += 1;
                h.lines.push(line.to_string());
            } else {
                // "Other" line: `\ No newline at end of file`, trailing
                // junk like comments, etc. Append verbatim — the context
                // trim later decides whether it survives based on
                // proximity to a `+`/`-` line.
                h.lines.push(line.to_string());
            }
        }
    }

    if let Some(h) = current_hunk.take() {
        if let Some(f) = current_file.as_mut() {
            f.hunks.push(h);
        }
    }
    if let Some(f) = current_file.take() {
        files.push(f);
    }

    ParsedDiff {
        pre_diff_lines,
        files,
        parse_warnings: warnings,
    }
}

// ─── Scoring ───────────────────────────────────────────────────────────────

/// Priority patterns matching `headroom.transforms.error_detection.PRIORITY_PATTERNS_DIFF`:
/// ERROR + IMPORTANCE + SECURITY. Used in scoring so error-relevant hunks
/// survive `max_hunks_per_file` capping.
fn priority_patterns() -> &'static [Regex] {
    static RES: OnceLock<Vec<Regex>> = OnceLock::new();
    RES.get_or_init(|| {
        vec![
            Regex::new(r"(?i)\b(error|exception|fail(?:ed|ure)?|fatal|critical|crash|panic)\b")
                .unwrap(),
            Regex::new(r"(?i)\b(important|note|todo|fixme|hack|xxx|bug|fix)\b").unwrap(),
            Regex::new(r"(?i)\b(security|auth|password|secret|token)\b").unwrap(),
        ]
    })
}

fn score_hunks(files: &mut [DiffFile], context: &str) {
    let context_lower = context.to_lowercase();
    let context_words: Vec<&str> = context_lower.split_whitespace().collect();

    for file in files.iter_mut() {
        for hunk in file.hunks.iter_mut() {
            let mut score: f64 = 0.0;
            // Base score from change density (capped).
            score += (hunk.additions as f64 + hunk.deletions as f64) * SCORE_CHANGE_DENSITY_WEIGHT;
            if score > SCORE_CHANGE_DENSITY_CAP {
                score = SCORE_CHANGE_DENSITY_CAP;
            }

            let hunk_content_lower = hunk.lines.join("\n").to_lowercase();

            for word in &context_words {
                if word.len() > SCORE_CONTEXT_MIN_WORD_LEN && hunk_content_lower.contains(word) {
                    score += SCORE_CONTEXT_WORD_WEIGHT;
                }
            }

            for pat in priority_patterns() {
                if pat.is_match(&hunk_content_lower) {
                    score += SCORE_PRIORITY_PATTERN_BOOST;
                    break;
                }
            }

            if score > SCORE_TOTAL_CAP {
                score = SCORE_TOTAL_CAP;
            }
            hunk.score = score;
        }
    }
}

// ─── Hunk selection (max_hunks_per_file cap) ───────────────────────────────

/// Mirror of Python's `_compress_hunks` (the file-internal hunk cap, not
/// the file count cap). Returns `(selected_in_original_order, dropped)`.
fn select_hunks(hunks: Vec<DiffHunk>, max_per_file: usize) -> (Vec<DiffHunk>, Vec<DiffHunk>) {
    if hunks.len() <= max_per_file {
        return (hunks, Vec::new());
    }
    if hunks.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // Python keeps first + last + top-scored middle, then resorts by hunk
    // header start-line to restore appearance order.
    let n = hunks.len();
    let mut indexed: Vec<(usize, DiffHunk)> = hunks.into_iter().enumerate().collect();

    let first = indexed.remove(0);
    let last = if !indexed.is_empty() {
        Some(indexed.pop().unwrap())
    } else {
        None
    };
    let middle: Vec<(usize, DiffHunk)> = indexed;

    let remaining_slots = if last.is_some() {
        max_per_file.saturating_sub(2)
    } else {
        max_per_file.saturating_sub(1)
    };

    // Sort middle by score desc; pick top.
    let mut middle_sorted = middle;
    middle_sorted.sort_by(|a, b| {
        b.1.score
            .partial_cmp(&a.1.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let (kept_middle, dropped_middle): (Vec<_>, Vec<_>) = middle_sorted
        .into_iter()
        .enumerate()
        .partition(|(rank, _)| *rank < remaining_slots);
    let kept_middle: Vec<(usize, DiffHunk)> = kept_middle.into_iter().map(|(_, x)| x).collect();
    let dropped_middle: Vec<DiffHunk> = dropped_middle.into_iter().map(|(_, (_, h))| h).collect();

    // Reassemble in original order using the captured indices.
    let mut selected: Vec<(usize, DiffHunk)> = Vec::with_capacity(max_per_file);
    selected.push(first);
    selected.extend(kept_middle);
    if let Some(l) = last {
        selected.push(l);
    }
    // Python sorts by `_extract_line_number(header)` (the @@ start line);
    // we match that exactly because two hunks could in principle share an
    // index (they don't in practice but match Python's tiebreak).
    selected.sort_by(|a, b| {
        let la = extract_line_number(&a.1.header);
        let lb = extract_line_number(&b.1.header);
        la.cmp(&lb)
    });

    let _ = n;
    (
        selected.into_iter().map(|(_, h)| h).collect(),
        dropped_middle,
    )
}

fn extract_line_number(header: &str) -> usize {
    // Use the dedicated `+N` regex — works for both `@@` and `@@@` headers.
    // The previous implementation captured group(1) of the hunk-header
    // regex, which was the line number for `@@` only; under the new combined
    // diff regex, group(1) is the `@`-prefix.
    if let Some(caps) = hunk_new_range_regex().captures(header) {
        if let Some(m) = caps.get(1) {
            if let Ok(n) = m.as_str().parse::<usize>() {
                return n;
            }
        }
    }
    0
}

// ─── Context trimming ──────────────────────────────────────────────────────

fn reduce_context(hunk: &DiffHunk, max_context: usize) -> DiffHunk {
    // Indices of `+`/`-` lines.
    let change_positions: Vec<usize> = hunk
        .lines
        .iter()
        .enumerate()
        .filter_map(|(i, l)| {
            if l.starts_with('+') || l.starts_with('-') {
                Some(i)
            } else {
                None
            }
        })
        .collect();

    if change_positions.is_empty() {
        // No-changes hunk → keep up to `max_context` leading lines, like Python.
        let take = max_context.min(hunk.lines.len());
        let lines: Vec<String> = hunk.lines.iter().take(take).cloned().collect();
        return DiffHunk {
            header: hunk.header.clone(),
            lines,
            additions: 0,
            deletions: 0,
            context_lines: take,
            score: hunk.score,
        };
    }

    // Indices to keep: each change + `max_context` lines either side.
    let mut keep = std::collections::BTreeSet::new();
    for &pos in &change_positions {
        keep.insert(pos);
        let lo = pos.saturating_sub(max_context);
        for i in lo..pos {
            keep.insert(i);
        }
        let hi = (pos + max_context + 1).min(hunk.lines.len());
        for i in (pos + 1)..hi {
            keep.insert(i);
        }
    }

    // Bug-fix: ALWAYS keep `\ No newline at end of file` markers (and any
    // other backslash-prefixed metadata) regardless of distance from a
    // change. These are structural patch markers, not context — losing
    // them breaks round-trippable patches and changes the semantic
    // meaning of the trailing line in the file. Mirrors the same fix in
    // the Python source.
    for (i, line) in hunk.lines.iter().enumerate() {
        if line.starts_with('\\') {
            keep.insert(i);
        }
    }

    let mut new_lines: Vec<String> = Vec::with_capacity(keep.len());
    let mut additions = 0usize;
    let mut deletions = 0usize;
    let mut context_lines = 0usize;
    for &i in &keep {
        let line = &hunk.lines[i];
        new_lines.push(line.clone());
        if line.starts_with('+') {
            additions += 1;
        } else if line.starts_with('-') {
            deletions += 1;
        } else {
            context_lines += 1;
        }
    }

    DiffHunk {
        header: hunk.header.clone(),
        lines: new_lines,
        additions,
        deletions,
        context_lines,
        score: hunk.score,
    }
}

// ─── Output formatter ──────────────────────────────────────────────────────

fn format_output(
    pre_diff_lines: &[String],
    files: &[DiffFile],
    files_affected: usize,
    total_additions: usize,
    total_deletions: usize,
    hunks_removed: usize,
) -> String {
    let mut out_lines: Vec<String> = Vec::new();

    // Bug-fix: emit pre-diff content (commit headers, email headers)
    // verbatim before the file sections. Previously dropped silently.
    for l in pre_diff_lines {
        out_lines.push(l.clone());
    }

    for f in files {
        out_lines.push(f.header.clone());

        // Bug-fix: emit rename / similarity / dissimilarity / copy marker
        // lines immediately after `diff --git`, matching git's canonical
        // output ordering. Previously these were captured into
        // `is_renamed=true` and dropped — output looked like a plain
        // modification of the old path.
        for l in &f.rename_lines {
            out_lines.push(l.clone());
        }

        if f.is_new_file {
            out_lines.push("new file mode 100644".into());
        } else if f.is_deleted_file {
            out_lines.push("deleted file mode 100644".into());
        }

        if f.is_binary {
            out_lines.push("Binary files differ".into());
            continue;
        }

        if !f.old_file.is_empty() {
            out_lines.push(f.old_file.clone());
        }
        if !f.new_file.is_empty() {
            out_lines.push(f.new_file.clone());
        }

        for h in &f.hunks {
            out_lines.push(h.header.clone());
            for l in &h.lines {
                out_lines.push(l.clone());
            }
        }
    }

    // Summary footer — only when we touched at least one file.
    if hunks_removed > 0 || files_affected > 0 {
        let mut parts = Vec::with_capacity(3);
        parts.push(format!("{} files changed", files_affected));
        parts.push(format!("+{} -{} lines", total_additions, total_deletions));
        if hunks_removed > 0 {
            parts.push(format!("{} hunks omitted", hunks_removed));
        }
        out_lines.push(format!("[{}]", parts.join(", ")));
    }

    out_lines.join("\n")
}

// ─── Helpers ───────────────────────────────────────────────────────────────

fn pass_through_result(content: &str, line_count: usize) -> DiffCompressionResult {
    DiffCompressionResult {
        compressed: content.to_string(),
        original_line_count: line_count,
        compressed_line_count: line_count,
        files_affected: 0,
        additions: 0,
        deletions: 0,
        hunks_kept: 0,
        hunks_removed: 0,
        cache_key: None,
    }
}

/// `s.split('\n').count()` — matches Python's `len(content.split("\n"))` so
/// the line count is byte-for-byte identical regardless of trailing newlines.
fn count_split_lines(s: &str) -> usize {
    s.split('\n').count()
}

/// MD5 of `s`'s UTF-8 bytes, hex-encoded, truncated to 24 chars. Matches
/// `hashlib.md5(s.encode()).hexdigest()[:24]` from
/// `headroom.cache.compression_store.CompressionStore.store`.
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

/// Emit a `tracing::info` event for the OTel pipeline and return the stats
/// unchanged (so callers can see them too).
fn emit_span_and_return(stats: DiffCompressorStats) -> DiffCompressorStats {
    tracing::info!(
        target: "diff_compressor",
        input_lines = stats.input_lines,
        output_lines = stats.output_lines,
        compression_ratio = stats.compression_ratio,
        files_total = stats.files_total,
        files_kept = stats.files_kept,
        files_dropped = stats.files_dropped.len(),
        hunks_total = stats.hunks_total,
        hunks_kept = stats.hunks_kept,
        hunks_dropped = stats.hunks_dropped,
        context_lines_trimmed = stats.context_lines_trimmed,
        largest_hunk_kept_lines = stats.largest_hunk_kept_lines,
        largest_hunk_dropped_lines = stats.largest_hunk_dropped_lines,
        parse_warnings = stats.parse_warnings.len(),
        processing_duration_us = stats.processing_duration_us,
        cache_key_emitted = stats.cache_key_emitted,
        file_mode_normalizations = stats.file_mode_normalizations.len(),
        binary_files_simplified = stats.binary_files_simplified.len(),
        "diff_compressor finished"
    );
    stats
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_input_passes_through() {
        let c = DiffCompressor::default();
        let input = "diff --git a/x b/x\n@@ -1 +1 @@\n-a\n+b";
        let r = c.compress(input, "");
        // 4 lines, below min_lines_for_ccr (50) → pass-through.
        assert_eq!(r.compressed, input);
        assert_eq!(r.original_line_count, 4);
        assert_eq!(r.compressed_line_count, 4);
        assert_eq!(r.files_affected, 0);
        assert!(r.cache_key.is_none());
    }

    #[test]
    fn non_diff_input_passes_through() {
        let c = DiffCompressor::default();
        let input = "this is not a diff\n".repeat(60); // > min_lines_for_ccr
        let r = c.compress(&input, "");
        assert_eq!(r.compressed, input);
        assert_eq!(r.files_affected, 0);
    }

    #[test]
    fn md5_24_matches_python() {
        // Verified against Python: hashlib.md5(b"hello").hexdigest()[:24]
        assert_eq!(md5_hex_24("hello"), "5d41402abc4b2a76b9719d91");
        assert_eq!(md5_hex_24(""), "d41d8cd98f00b204e9800998");
    }

    #[test]
    fn count_split_lines_matches_python_split_n() {
        // `"".split("\n")` == [""] in Python → 1
        assert_eq!(count_split_lines(""), 1);
        assert_eq!(count_split_lines("a"), 1);
        assert_eq!(count_split_lines("a\n"), 2);
        assert_eq!(count_split_lines("a\nb"), 2);
        assert_eq!(count_split_lines("\n"), 2);
    }

    #[test]
    fn stats_are_emitted_with_compress_with_stats() {
        let c = DiffCompressor::default();
        let input = "noise\n".repeat(60);
        let (_r, stats) = c.compress_with_stats(&input, "");
        assert_eq!(stats.input_lines, 61); // 60 newlines split into 61 elements
        assert_eq!(stats.output_lines, 61);
        assert_eq!(stats.compression_ratio, 1.0);
        assert!(stats.parse_warnings.is_empty());
        assert!(stats.ccr_skipped_reason.is_some());
    }

    /// Build a synthetic 8-file diff matching the parity-fixture shape so we
    /// can sanity-check the algorithm before running parity.
    fn build_synthetic_diff(n_files: usize) -> String {
        let mut s = String::new();
        for i in 0..n_files {
            s.push_str(&format!(
                "diff --git a/file_{i}.py b/file_{i}.py\n--- a/file_{i}.py\n+++ b/file_{i}.py\n@@ -1,10 +1,12 @@\n",
            ));
            for k in 0..5 {
                s.push_str(&format!(" context_{k}_{i}\n"));
            }
            for k in 0..3 {
                s.push_str(&format!("-removed_{k}_{i}\n"));
            }
            for k in 0..5 {
                s.push_str(&format!("+added_{k}_{i}\n"));
            }
            for k in 0..5 {
                s.push_str(&format!(" tail_{k}_{i}\n"));
            }
        }
        s.push_str("# variant 1");
        s
    }

    #[test]
    fn synthetic_eight_file_diff_matches_known_shape() {
        let c = DiffCompressor::default();
        let input = build_synthetic_diff(8);
        let r = c.compress(&input, "");
        assert_eq!(r.original_line_count, 177);
        assert_eq!(r.files_affected, 8);
        assert_eq!(r.additions, 40);
        assert_eq!(r.deletions, 24);
        assert_eq!(r.hunks_kept, 8);
        assert_eq!(r.hunks_removed, 0);
        // Compressed line count should be 129 (matches the parity fixture).
        assert_eq!(r.compressed_line_count, 129);
        assert!(r.cache_key.is_some());
    }

    // ─── Lossy-path tests ───────────────────────────────────────────────────

    /// Build a single-file diff with N hunks. Each hunk has 2 context lines,
    /// 1 deletion, 1 addition, 2 context lines. Hunk headers use distinct
    /// start lines so the in-order resort after middle-hunk selection works.
    fn build_n_hunk_diff(n: usize) -> String {
        let mut s = String::from("diff --git a/big.py b/big.py\n--- a/big.py\n+++ b/big.py\n");
        for i in 0..n {
            // 100 lines apart per hunk so they're independent.
            let start = i * 100 + 1;
            s.push_str(&format!("@@ -{0},6 +{0},6 @@\n", start));
            s.push_str(&format!(" ctx_a_{i}\n"));
            s.push_str(&format!(" ctx_b_{i}\n"));
            s.push_str(&format!("-old_{i}\n"));
            s.push_str(&format!("+new_{i}\n"));
            s.push_str(&format!(" ctx_c_{i}\n"));
            s.push_str(&format!(" ctx_d_{i}\n"));
        }
        s
    }

    #[test]
    fn max_hunks_per_file_cap_drops_excess_and_records_stats() {
        // 15 hunks, cap = 10 → 5 dropped. First + last + 8 top-scored middle kept.
        let cfg = DiffCompressorConfig {
            max_hunks_per_file: 10,
            ..Default::default()
        };
        let input = build_n_hunk_diff(15);
        let (result, stats) = DiffCompressor::new(cfg).compress_with_stats(&input, "");

        assert_eq!(result.hunks_kept, 10, "kept 10 hunks");
        assert_eq!(result.hunks_removed, 5, "dropped 5");
        assert_eq!(stats.hunks_total, 15);
        assert_eq!(stats.hunks_dropped, 5);
        // Per-file accounting must match overall.
        let per_file_total: usize = stats.hunks_dropped_per_file.values().sum();
        assert_eq!(per_file_total, 5);
        // The dropped hunks have 6 lines each (after parsing); largest_dropped
        // should reflect that.
        assert!(stats.largest_hunk_dropped_lines >= 6);
    }

    #[test]
    fn max_files_cap_drops_files_and_records_names_in_stats() {
        // 25 files, cap = 20 → 5 dropped. files_dropped should carry the names.
        let cfg = DiffCompressorConfig {
            max_files: 20,
            ..Default::default()
        };
        let input = build_synthetic_diff(25);
        let (_result, stats) = DiffCompressor::new(cfg).compress_with_stats(&input, "");

        assert_eq!(stats.files_total, 25);
        assert_eq!(stats.files_kept, 20);
        assert_eq!(
            stats.files_dropped.len(),
            5,
            "expected 5 dropped file labels"
        );
        // Each label should be the `old_file -> new_file` form.
        for label in &stats.files_dropped {
            assert!(
                label.contains("-> "),
                "label `{label}` should contain ` -> `"
            );
        }
    }

    #[test]
    fn file_mode_normalization_is_recorded_for_executable_bit() {
        // Construct a long-enough diff that introduces an executable file.
        // Mode 100755 != 100644, so emit will silently normalize and stats
        // must capture the original.
        let mut input = String::from(
            "diff --git a/script.sh b/script.sh\n\
             new file mode 100755\n\
             --- /dev/null\n\
             +++ b/script.sh\n\
             @@ -0,0 +1,3 @@\n\
             +#!/bin/sh\n\
             +echo hi\n\
             +exit 0\n",
        );
        // Pad to clear `min_lines_for_ccr` so compression runs.
        for _ in 0..50 {
            input.push_str("# pad\n");
        }
        let (_r, stats) = DiffCompressor::default().compress_with_stats(&input, "");
        assert_eq!(stats.file_mode_normalizations.len(), 1, "{stats:?}");
        let (label, original) = &stats.file_mode_normalizations[0];
        assert!(label.contains("script.sh"));
        assert_eq!(original, "new file mode 100755");
    }

    #[test]
    fn binary_files_simplification_is_recorded() {
        let mut input = String::from(
            "diff --git a/img.png b/img.png\n\
             Binary files a/img.png and b/img.png differ\n",
        );
        // Pad to clear min_lines_for_ccr.
        for _ in 0..60 {
            input.push_str("# pad\n");
        }
        let (_r, stats) = DiffCompressor::default().compress_with_stats(&input, "");
        assert_eq!(stats.binary_files_simplified.len(), 1, "{stats:?}");
        assert_eq!(
            stats.binary_files_simplified[0],
            "Binary files a/img.png and b/img.png differ"
        );
    }

    #[test]
    fn min_compression_ratio_for_ccr_is_configurable() {
        // With default 0.8, the 8-file synthetic compresses 177→129 (ratio
        // 0.729) which beats the threshold → CCR marker emitted.
        let r = DiffCompressor::default().compress(&build_synthetic_diff(8), "");
        assert!(r.cache_key.is_some(), "default 0.8 should emit CCR");

        // With 0.5, the same compression (0.729 ratio) does NOT beat
        // 0.5 → no CCR marker, no cache_key.
        let cfg = DiffCompressorConfig {
            min_compression_ratio_for_ccr: 0.5,
            ..Default::default()
        };
        let (r2, stats) =
            DiffCompressor::new(cfg).compress_with_stats(&build_synthetic_diff(8), "");
        assert!(
            r2.cache_key.is_none(),
            "0.5 threshold should suppress CCR for 0.729-ratio compression"
        );
        assert!(!stats.cache_key_emitted);
        assert!(stats.ccr_skipped_reason.is_some());
    }

    #[test]
    fn compress_with_store_persists_original_under_cache_key() {
        // Regression: pre-fix, `DiffCompressor` minted a `cache_key` and
        // embedded `[... hash=abc123]` in the output marker but never
        // wrote the original anywhere — leaving Python ContentRouter
        // with a dangling marker that 404'd on retrieval. Now any caller
        // can pass a store and the marker resolves.
        use crate::ccr::InMemoryCcrStore;
        let store = InMemoryCcrStore::new();
        let input = build_synthetic_diff(8);
        let (r, stats) = DiffCompressor::default().compress_with_store(&input, "", Some(&store));
        let key = r.cache_key.expect("default 0.8 should emit CCR");
        assert!(stats.cache_key_emitted);
        // Marker text must reference the same key.
        assert!(r.compressed.contains(&format!("hash={key}")));
        // Original must round-trip through the store.
        assert_eq!(store.get(&key).as_deref(), Some(input.as_str()));
    }

    #[test]
    fn compress_with_store_none_matches_compress_with_stats_behavior() {
        // Passing `None` must be byte-identical to the legacy API:
        // emits cache_key, leaves persistence to the caller. This pins
        // the parity-preserving-default contract.
        let input = build_synthetic_diff(8);
        let (legacy_result, _) = DiffCompressor::default().compress_with_stats(&input, "");
        let (new_result, _) = DiffCompressor::default().compress_with_store(&input, "", None);
        assert_eq!(new_result.compressed, legacy_result.compressed);
        assert_eq!(new_result.cache_key, legacy_result.cache_key);
    }

    #[test]
    fn compress_with_store_no_op_when_ccr_skipped() {
        // When compression doesn't clear the savings threshold, no
        // cache_key is minted AND no store write happens.
        use crate::ccr::InMemoryCcrStore;
        let cfg = DiffCompressorConfig {
            min_compression_ratio_for_ccr: 0.1, // very strict
            ..Default::default()
        };
        let store = InMemoryCcrStore::new();
        let (r, _) = DiffCompressor::new(cfg).compress_with_store(
            &build_synthetic_diff(8),
            "",
            Some(&store),
        );
        assert!(r.cache_key.is_none());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn score_constants_match_inline_values() {
        // Pin the constants so a future tuning PR has to update both.
        // (If you're updating these, also update the docs in the parity
        // contract — the scorer only fires when max_hunks_per_file caps,
        // so the impact is limited but observable.)
        assert_eq!(SCORE_CHANGE_DENSITY_WEIGHT, 0.03);
        assert_eq!(SCORE_CHANGE_DENSITY_CAP, 0.3);
        assert_eq!(SCORE_CONTEXT_WORD_WEIGHT, 0.2);
        assert_eq!(SCORE_CONTEXT_MIN_WORD_LEN, 2);
        assert_eq!(SCORE_PRIORITY_PATTERN_BOOST, 0.3);
        assert_eq!(SCORE_TOTAL_CAP, 1.0);
    }

    // ─── Bug-fix tests (rename/combined-diff/no-newline/pre-diff) ──────────

    /// Bug-fix test: rename markers (`rename from`, `rename to`,
    /// `similarity index`) must survive into the compressed output. Before
    /// the fix the parser captured them as `is_renamed=true` and the
    /// emitter dropped them entirely — output looked like a plain
    /// modification of the old path.
    #[test]
    fn bugfix_rename_markers_are_preserved_in_output() {
        let input = "diff --git a/old.py b/new.py\n\
                     similarity index 92%\n\
                     rename from old.py\n\
                     rename to new.py\n\
                     --- a/old.py\n\
                     +++ b/new.py\n\
                     @@ -1,3 +1,3 @@\n\
                      ctx_a\n\
                     -old_line\n\
                     +new_line\n\
                      ctx_b\n";
        // Below default min_lines_for_ccr=50 — drop the threshold so the
        // parser+emitter actually run.
        let cfg = DiffCompressorConfig {
            min_lines_for_ccr: 5,
            ..Default::default()
        };
        let r = DiffCompressor::new(cfg).compress(input, "");
        assert!(
            r.compressed.contains("similarity index 92%"),
            "missing 'similarity index' marker:\n{}",
            r.compressed
        );
        assert!(
            r.compressed.contains("rename from old.py"),
            "missing 'rename from':\n{}",
            r.compressed
        );
        assert!(
            r.compressed.contains("rename to new.py"),
            "missing 'rename to':\n{}",
            r.compressed
        );
    }

    /// Bug-fix test: combined-diff (`@@@`) hunk content must NOT be
    /// silently dropped. Before the fix the hunk-header regex only
    /// matched `@@`, so 3-way merge hunks had `current_hunk` never set
    /// and all their content fell through to the no-op branch.
    #[test]
    fn bugfix_combined_diff_3way_content_is_parsed_and_emitted() {
        let input = "diff --git a/merge.py b/merge.py\n\
                     --- a/merge.py\n\
                     +++ b/merge.py\n\
                     @@@ -1,3 -1,3 +1,4 @@@\n\
                       unchanged_a\n\
                      -old_branch_1\n\
                     - old_branch_2\n\
                     ++new_in_merge\n\
                      +new_added\n\
                       unchanged_b\n";
        let cfg = DiffCompressorConfig {
            min_lines_for_ccr: 5,
            ..Default::default()
        };
        let (r, stats) = DiffCompressor::new(cfg).compress_with_stats(input, "");
        assert!(
            r.compressed.contains("@@@ -1,3 -1,3 +1,4 @@@"),
            "@@@ header not preserved:\n{}",
            r.compressed
        );
        assert!(
            r.compressed.contains("++new_in_merge"),
            "combined-diff +/+ content not preserved:\n{}",
            r.compressed
        );
        assert!(
            stats.files_total > 0,
            "parser found no files; combined-diff still broken"
        );
    }

    /// Bug-fix test: `\ No newline at end of file` markers must survive
    /// the context trim regardless of distance from a `+`/`-` line. Before
    /// the fix, `_reduce_context` only kept lines within max_context_lines
    /// of a change; trailing `\` markers got cut whenever they were too
    /// far away.
    #[test]
    fn bugfix_no_newline_marker_preserved_despite_distance() {
        // Place the `+/-` change far from the trailing `\` marker so the
        // context trim would, if buggy, drop the marker.
        let input = "diff --git a/last.txt b/last.txt\n\
                     --- a/last.txt\n\
                     +++ b/last.txt\n\
                     @@ -1,8 +1,8 @@\n\
                     -old_first\n\
                     +new_first\n\
                      ctx_a\n\
                      ctx_b\n\
                      ctx_c\n\
                      ctx_d\n\
                      ctx_e\n\
                      ctx_f\n\
                     \\ No newline at end of file\n";
        let cfg = DiffCompressorConfig {
            min_lines_for_ccr: 5,
            ..Default::default()
        };
        let r = DiffCompressor::new(cfg).compress(input, "");
        assert!(
            r.compressed.contains("\\ No newline at end of file"),
            "no-newline marker dropped by context trim:\n{}",
            r.compressed
        );
    }

    /// Routing-gap test: `diff --combined <path>` (merge-commit header)
    /// must start a new file section. Before the fix, only `diff --git`
    /// was recognized — merge diffs from `git log -p` got treated as one
    /// big pre-diff blob and passed through unchanged.
    #[test]
    fn gap_diff_combined_header_starts_a_file() {
        let input = "diff --combined merge.py\n\
                     index abc..def..ghi 100644\n\
                     --- a/merge.py\n\
                     +++ b/merge.py\n\
                     @@@ -1,3 -1,3 +1,4 @@@\n\
                       ctx_a\n\
                     - removed_p1\n\
                      -removed_p2\n\
                     ++added_in_merge\n\
                       ctx_b\n";
        let cfg = DiffCompressorConfig {
            min_lines_for_ccr: 5,
            ..Default::default()
        };
        let r = DiffCompressor::new(cfg).compress(input, "");
        assert_eq!(r.files_affected, 1);
        assert!(r.compressed.contains("diff --combined merge.py"));
        assert!(r.compressed.contains("@@@ -1,3 -1,3 +1,4 @@@"));
        assert!(r.compressed.contains("++added_in_merge"));
    }

    /// Routing-gap test: `diff --cc <path>` (alternate merge-commit form).
    /// Same reasoning as `diff_combined`.
    #[test]
    fn gap_diff_cc_header_starts_a_file() {
        let input = "diff --cc cc_target.py\n\
                     index abc..def..ghi\n\
                     --- a/cc_target.py\n\
                     +++ b/cc_target.py\n\
                     @@@ -1,3 -1,3 +1,4 @@@\n\
                       ctx\n\
                     - p1_removed\n\
                      -p2_removed\n\
                     ++merge_added\n\
                       more_ctx\n";
        let cfg = DiffCompressorConfig {
            min_lines_for_ccr: 5,
            ..Default::default()
        };
        let r = DiffCompressor::new(cfg).compress(input, "");
        assert_eq!(r.files_affected, 1);
        assert!(r.compressed.contains("diff --cc cc_target.py"));
        assert!(r.compressed.contains("++merge_added"));
    }

    /// Bug-fix test: pre-diff content (commit headers, email-style
    /// metadata) is preserved verbatim before the diff sections. Before
    /// the fix, anything before the first `diff --git` was silently
    /// dropped — `git log -p` output lost commit messages, Author, Date,
    /// etc.
    #[test]
    fn bugfix_pre_diff_content_is_preserved() {
        let input = "commit abc1234567890\n\
                     Author: Tester <t@example.com>\n\
                     Date:   Mon Apr 25 12:00:00 2026\n\
                     \n    Refactor: rename and modify\n\n\
                     diff --git a/x.py b/x.py\n\
                     --- a/x.py\n\
                     +++ b/x.py\n\
                     @@ -1 +1 @@\n\
                     -a\n\
                     +b\n";
        let cfg = DiffCompressorConfig {
            min_lines_for_ccr: 5,
            ..Default::default()
        };
        let r = DiffCompressor::new(cfg).compress(input, "");
        assert!(
            r.compressed.starts_with("commit abc1234567890"),
            "pre-diff commit header dropped:\n{}",
            r.compressed
        );
        assert!(
            r.compressed.contains("Author: Tester"),
            "pre-diff Author header dropped:\n{}",
            r.compressed
        );
        assert!(
            r.compressed.contains("Refactor: rename and modify"),
            "pre-diff commit message dropped:\n{}",
            r.compressed
        );
        // And the diff itself is still there.
        assert!(r.compressed.contains("diff --git a/x.py b/x.py"));
        assert!(r.compressed.contains("-a"));
        assert!(r.compressed.contains("+b"));
    }
}
