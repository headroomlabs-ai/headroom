//! `DiffNoise` — drop hunks the LLM doesn't need (lockfiles + whitespace-only).
//!
//! # What this offload removes
//!
//! Real-world `git diff` output is dominated by two categories of bytes
//! the LLM rarely needs:
//!
//! 1. **Lockfile churn.** A `npm install foo` commit re-shuffles
//!    thousands of lines in `package-lock.json` while changing one line
//!    in `package.json`. The LLM only needs to know "we bumped a
//!    dependency" — the manifest line carries that, the lockfile is
//!    noise. Same for `Cargo.lock`, `yarn.lock`, `poetry.lock`,
//!    `go.sum`, etc.
//! 2. **Whitespace-only changes.** Reformat / lint-fix commits churn
//!    huge amounts of bytes without changing semantics. Detected by
//!    pairing `-` lines against `+` lines: if every pair, when
//!    whitespace-collapsed, is equal, the hunk is whitespace-only.
//!
//! Both categories are **dropped from the wire and stashed via CCR** so
//! the LLM can retrieve the original on demand. Strict accuracy: bytes
//! removed are recoverable through the cache key.
//!
//! # Why this is an Offload (not a Reformat)
//!
//! The dropped bytes are *gone* from the wire output. Even though they
//! carry near-zero semantic value for typical LLM tasks, calling that
//! "lossless" would misrepresent the contract. CCR is exactly the
//! mechanism for "drop bytes, keep retrievable" — that's an Offload.
//!
//! # Bloat heuristic
//!
//! Score = fraction of input bytes that fall inside a droppable
//! section. Computed cheaply over a single pass:
//!
//! - For each file header, if filename matches a configured lockfile
//!   suffix, count its hunk bytes as droppable.
//! - For each non-lockfile hunk, if `drop_whitespace_only_hunks` is on
//!   AND the hunk is whitespace-only, count its bytes as droppable.
//!
//! The orchestrator gates `apply` on this score clearing the
//! configurable threshold.
//!
//! # No regex
//!
//! Lockfile matching is suffix comparison via `str::ends_with` against
//! the post-`b/` path of the `diff --git` header. Hunk parsing is
//! prefix matching on `@@`, `diff --git`, `+++`, `---`. Whitespace
//! comparison strips ASCII whitespace by hand.

use crate::ccr::CcrStore;
use crate::transforms::pipeline::config::DiffNoiseConfig;
use crate::transforms::pipeline::traits::{
    CompressionContext, OffloadOutput, OffloadTransform, TransformError,
};
use crate::transforms::ContentType;

use md5::{Digest, Md5};

const NAME: &str = "diff_noise";
const CONFIDENCE: f32 = 0.9;

pub struct DiffNoise {
    config: DiffNoiseConfig,
}

impl DiffNoise {
    pub fn new(config: DiffNoiseConfig) -> Self {
        Self { config }
    }
}

impl OffloadTransform for DiffNoise {
    fn name(&self) -> &'static str {
        NAME
    }

    fn applies_to(&self) -> &[ContentType] {
        &[ContentType::GitDiff]
    }

    fn estimate_bloat(&self, content: &str) -> f32 {
        if content.is_empty() {
            return 0.0;
        }
        let total_lines = content.lines().count();
        if total_lines < self.config.min_lines {
            return 0.0;
        }
        let segments = parse_segments(content);
        if segments.is_empty() {
            return 0.0;
        }

        let mut droppable_bytes = 0usize;
        let mut total_bytes = 0usize;
        for seg in &segments {
            let body_bytes: usize = seg
                .body_lines
                .iter()
                .map(|l| l.len() + 1 /* the '\n' */)
                .sum();
            total_bytes += body_bytes;
            let droppable = self.is_lockfile(&seg.new_path)
                || (self.config.drop_whitespace_only_hunks && seg.body_is_whitespace_only());
            if droppable {
                droppable_bytes += body_bytes;
            }
        }
        if total_bytes == 0 {
            return 0.0;
        }
        (droppable_bytes as f32 / total_bytes as f32).clamp(0.0, 1.0)
    }

    fn apply(
        &self,
        content: &str,
        _ctx: &CompressionContext,
        store: &dyn CcrStore,
    ) -> Result<OffloadOutput, TransformError> {
        let segments = parse_segments(content);
        if segments.is_empty() {
            return Err(TransformError::skipped(NAME, "no diff sections"));
        }

        let mut output = String::with_capacity(content.len());
        let mut dropped_any = false;
        for seg in &segments {
            // Always emit the segment's pre-body lines verbatim
            // (`diff --git`, `index`, `+++`, `---`, mode lines, etc.)
            // so the LLM sees that the file changed.
            for h in &seg.header_lines {
                output.push_str(h);
                output.push('\n');
            }
            let drop_lockfile = self.is_lockfile(&seg.new_path);
            let drop_whitespace =
                self.config.drop_whitespace_only_hunks && seg.body_is_whitespace_only();
            if drop_lockfile || drop_whitespace {
                let reason = if drop_lockfile {
                    "lockfile"
                } else {
                    "whitespace-only"
                };
                output.push_str("[diff_noise: ");
                output.push_str(reason);
                output.push_str(" hunks dropped (");
                let body_lines = seg.body_lines.len();
                output.push_str(&body_lines.to_string());
                output.push_str(" lines)]\n");
                dropped_any = true;
            } else {
                for b in &seg.body_lines {
                    output.push_str(b);
                    output.push('\n');
                }
            }
        }

        // Pre-diff content (anything before the first `diff --git`)
        // gets prepended verbatim. parse_segments doesn't claim those
        // lines so we splice them in by walking the input again.
        let pre_diff = leading_pre_diff_lines(content);
        if !pre_diff.is_empty() {
            let mut prefixed = String::with_capacity(pre_diff.len() + output.len());
            prefixed.push_str(&pre_diff);
            prefixed.push_str(&output);
            output = prefixed;
        }

        if !dropped_any || output.len() >= content.len() {
            return Err(TransformError::skipped(NAME, "no droppable hunks"));
        }

        // CCR: hash original, stash, append marker.
        let key = md5_hex_24(content);
        store.put(&key, content);
        output.push_str(&format!("\n[diff_noise CCR: hash={key}]"));

        Ok(OffloadOutput::from_lengths(content.len(), output, key))
    }

    fn confidence(&self) -> f32 {
        CONFIDENCE
    }
}

impl DiffNoise {
    fn is_lockfile(&self, path: &str) -> bool {
        if path.is_empty() {
            return false;
        }
        for suffix in &self.config.lockfile_suffixes {
            // Match if the path ENDS WITH this suffix at a path-segment
            // boundary (so `Cargo.lock` matches `crates/foo/Cargo.lock`
            // but not `MyCargo.lock` — defensive against accidentally
            // dropping a user-named file).
            if path.ends_with(suffix.as_str()) {
                let prefix_len = path.len() - suffix.len();
                if prefix_len == 0 {
                    return true;
                }
                let prev_byte = path.as_bytes()[prefix_len - 1];
                if prev_byte == b'/' || prev_byte == b'\\' {
                    return true;
                }
            }
        }
        false
    }
}

/// One file's segment of a diff: the pre-body header lines (`diff
/// --git`, `index ...`, `+++/---`) and the body (every line until the
/// next `diff --git` or EOF), plus the parsed-out new file path.
struct Segment<'a> {
    new_path: String,
    header_lines: Vec<&'a str>,
    body_lines: Vec<&'a str>,
}

impl<'a> Segment<'a> {
    /// True if every `+` and `-` line in the body, when ASCII-whitespace-
    /// stripped and paired up in order, leaves equal token sequences.
    /// Approach: collect the pluses and the minuses, strip-and-compare.
    /// Order-aware so that `swapping two lines` doesn't accidentally
    /// pass.
    fn body_is_whitespace_only(&self) -> bool {
        let mut adds: Vec<String> = Vec::new();
        let mut subs: Vec<String> = Vec::new();
        let mut saw_change = false;
        for line in &self.body_lines {
            let bytes = line.as_bytes();
            match bytes.first() {
                Some(b'+') if !line.starts_with("+++") => {
                    saw_change = true;
                    adds.push(strip_ws(&line[1..]));
                }
                Some(b'-') if !line.starts_with("---") => {
                    saw_change = true;
                    subs.push(strip_ws(&line[1..]));
                }
                _ => {}
            }
        }
        if !saw_change {
            return false;
        }
        adds == subs
    }
}

fn strip_ws(s: &str) -> String {
    s.chars().filter(|c| !c.is_ascii_whitespace()).collect()
}

/// Walk the input and emit one [`Segment`] per `diff --git` header.
/// Lines before the first header are NOT included (caller picks them
/// up via [`leading_pre_diff_lines`]).
fn parse_segments(content: &str) -> Vec<Segment<'_>> {
    let mut segments: Vec<Segment<'_>> = Vec::new();
    let mut current: Option<Segment<'_>> = None;
    let mut in_body = false;

    for line in content.lines() {
        if line.starts_with("diff --git") {
            if let Some(s) = current.take() {
                segments.push(s);
            }
            current = Some(Segment {
                new_path: parse_new_path(line),
                header_lines: vec![line],
                body_lines: Vec::new(),
            });
            in_body = false;
            continue;
        }
        let Some(seg) = current.as_mut() else {
            continue; // pre-diff prelude — handled separately
        };
        if !in_body {
            // Treat everything until the first `@@` as header.
            if line.starts_with("@@") {
                in_body = true;
                seg.body_lines.push(line);
                continue;
            }
            seg.header_lines.push(line);
        } else {
            seg.body_lines.push(line);
        }
    }
    if let Some(s) = current.take() {
        segments.push(s);
    }
    segments
}

/// Lines from the start of `content` up to (but not including) the
/// first `diff --git`. Returned as one string with original newlines
/// re-attached, ready to be prepended to the rebuilt diff output.
fn leading_pre_diff_lines(content: &str) -> String {
    let mut out = String::new();
    for line in content.lines() {
        if line.starts_with("diff --git") {
            break;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// Parse the new-file path from a `diff --git a/X b/Y` header. Returns
/// the `Y` segment (after the last ` b/`); empty string if not found.
fn parse_new_path(header: &str) -> String {
    if let Some(idx) = header.rfind(" b/") {
        return header[idx + 3..].to_string();
    }
    String::new()
}

/// MD5 of `content`, hex-encoded, truncated to 24 chars (matches the
/// CCR convention used by other compressors).
fn md5_hex_24(content: &str) -> String {
    let mut h = Md5::new();
    h.update(content.as_bytes());
    let digest = h.finalize();
    let mut hex = String::with_capacity(32);
    for b in digest.iter() {
        hex.push_str(&format!("{:02x}", b));
    }
    hex.truncate(24);
    hex
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ccr::InMemoryCcrStore;
    use crate::transforms::pipeline::config::PipelineConfig;

    fn cfg() -> DiffNoiseConfig {
        PipelineConfig::default().offload.diff_noise
    }

    fn offload() -> DiffNoise {
        DiffNoise::new(cfg())
    }

    fn build_diff(files: &[(&str, &[&str])]) -> String {
        let mut s = String::new();
        for (path, body) in files {
            s.push_str(&format!("diff --git a/{path} b/{path}\n"));
            s.push_str(&format!("--- a/{path}\n+++ b/{path}\n"));
            s.push_str("@@ -1 +1 @@\n");
            for b in *body {
                s.push_str(b);
                s.push('\n');
            }
        }
        s
    }

    #[test]
    fn name_and_applies_to() {
        let o = offload();
        assert_eq!(o.name(), "diff_noise");
        assert_eq!(o.applies_to(), &[ContentType::GitDiff]);
    }

    #[test]
    fn estimate_bloat_below_min_lines_zero() {
        // Only ~6 lines.
        let diff = build_diff(&[("Cargo.lock", &["-old", "+new"])]);
        assert_eq!(offload().estimate_bloat(&diff), 0.0);
    }

    #[test]
    fn estimate_bloat_lockfile_dominates_scores_high() {
        // Tiny manifest change + huge lockfile churn.
        let big_lock_body: Vec<String> = (0..200)
            .flat_map(|i| vec![format!("-old{i}"), format!("+new{i}")])
            .collect();
        let body_refs: Vec<&str> = big_lock_body.iter().map(|s| s.as_str()).collect();
        let diff = build_diff(&[
            ("Cargo.lock", &body_refs),
            ("Cargo.toml", &["-foo = \"1\"", "+foo = \"2\""]),
        ]);
        let score = offload().estimate_bloat(&diff);
        assert!(
            score > 0.9,
            "lockfile-dominated diff should score high, got {score}"
        );
    }

    #[test]
    fn estimate_bloat_no_noise_zero() {
        let body: Vec<String> = (0..40)
            .flat_map(|i| vec![format!("-old line {i}"), format!("+new line {i}")])
            .collect();
        let body_refs: Vec<&str> = body.iter().map(|s| s.as_str()).collect();
        let diff = build_diff(&[("src/main.rs", &body_refs)]);
        let score = offload().estimate_bloat(&diff);
        assert_eq!(score, 0.0, "real code diff should not be flagged");
    }

    #[test]
    fn estimate_bloat_whitespace_only_hunk_scores_high() {
        // Hunk where every change is just trailing whitespace removal.
        let body: Vec<String> = (0..40)
            .flat_map(|i| vec![format!("-line {i}   "), format!("+line {i}")])
            .collect();
        let body_refs: Vec<&str> = body.iter().map(|s| s.as_str()).collect();
        let diff = build_diff(&[("src/main.rs", &body_refs)]);
        let score = offload().estimate_bloat(&diff);
        assert!(
            score > 0.5,
            "whitespace-only hunk should score high, got {score}"
        );
    }

    #[test]
    fn apply_drops_lockfile_and_stores_original() {
        let big_lock_body: Vec<String> = (0..200)
            .flat_map(|i| vec![format!("-old{i}"), format!("+new{i}")])
            .collect();
        let body_refs: Vec<&str> = big_lock_body.iter().map(|s| s.as_str()).collect();
        let diff = build_diff(&[
            ("Cargo.lock", &body_refs),
            ("Cargo.toml", &["-foo = \"1\"", "+foo = \"2\""]),
        ]);
        let store = InMemoryCcrStore::new();
        let r = offload()
            .apply(&diff, &CompressionContext::default(), &store)
            .expect("must compress");
        assert!(r.bytes_saved > 0);
        assert!(r.output.contains("[diff_noise: lockfile hunks dropped"));
        // Real code should survive untouched.
        assert!(r.output.contains("foo = \"2\""));
        // Original recoverable.
        assert_eq!(store.get(&r.cache_key).as_deref(), Some(diff.as_str()));
    }

    #[test]
    fn apply_drops_whitespace_only_hunk() {
        let body: Vec<String> = (0..40)
            .flat_map(|i| vec![format!("-line {i}   "), format!("+line {i}")])
            .collect();
        let body_refs: Vec<&str> = body.iter().map(|s| s.as_str()).collect();
        let diff = build_diff(&[("src/main.rs", &body_refs)]);
        let store = InMemoryCcrStore::new();
        let r = offload()
            .apply(&diff, &CompressionContext::default(), &store)
            .expect("must compress");
        assert!(r
            .output
            .contains("[diff_noise: whitespace-only hunks dropped"));
        assert!(r.bytes_saved > 0);
    }

    #[test]
    fn apply_skipped_when_no_droppable_hunks() {
        let body: Vec<String> = (0..40)
            .flat_map(|i| vec![format!("-old line {i}"), format!("+new line {i}")])
            .collect();
        let body_refs: Vec<&str> = body.iter().map(|s| s.as_str()).collect();
        let diff = build_diff(&[("src/main.rs", &body_refs)]);
        let store = InMemoryCcrStore::new();
        let err = offload()
            .apply(&diff, &CompressionContext::default(), &store)
            .expect_err("nothing droppable");
        match err {
            TransformError::Skipped { .. } => {}
            _ => panic!("expected Skipped, got {err:?}"),
        }
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn lockfile_path_match_is_path_segment_aware() {
        // `MyCargo.lock` — should NOT match because there's no `/`
        // before `Cargo.lock`.
        let o = offload();
        assert!(o.is_lockfile("Cargo.lock"));
        assert!(o.is_lockfile("crates/foo/Cargo.lock"));
        assert!(!o.is_lockfile("MyCargo.lock"));
        assert!(!o.is_lockfile("FakeCargo.lockfile"));
    }

    #[test]
    fn handles_diff_with_leading_commit_message() {
        // git format-patch style — commit message header before the diff.
        let mut diff = String::new();
        diff.push_str("From abc123 Mon Sep 17 2025\n");
        diff.push_str("Subject: bump deps\n\n");
        diff.push_str("commit body line\n\n");
        let lock: Vec<String> = (0..40)
            .flat_map(|i| vec![format!("-old{i}"), format!("+new{i}")])
            .collect();
        let lock_refs: Vec<&str> = lock.iter().map(|s| s.as_str()).collect();
        diff.push_str(&build_diff(&[("yarn.lock", &lock_refs)]));
        let store = InMemoryCcrStore::new();
        let r = offload()
            .apply(&diff, &CompressionContext::default(), &store)
            .expect("compresses");
        // Pre-diff prelude must survive.
        assert!(r.output.contains("Subject: bump deps"));
        assert!(r.output.contains("[diff_noise: lockfile hunks dropped"));
    }

    #[test]
    fn empty_input_is_safe() {
        assert_eq!(offload().estimate_bloat(""), 0.0);
        let store = InMemoryCcrStore::new();
        let err = offload()
            .apply("", &CompressionContext::default(), &store)
            .expect_err("must skip");
        match err {
            TransformError::Skipped { .. } => {}
            _ => panic!("expected Skipped"),
        }
    }
}
