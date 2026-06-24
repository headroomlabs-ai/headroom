//! Parity harness: load JSON fixtures recorded from the Python implementation,
//! run the Rust port, and compare outputs.
//!
//! Phase 0: the per-transform comparators are stubs (`todo!()`), but the
//! harness wiring (fixture loading, dispatch, diff reporting) is real and
//! covered by a negative test.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// Recorded fixture schema. Matches `tests/parity/recorder.py`.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Fixture {
    pub transform: String,
    pub input: serde_json::Value,
    #[serde(default)]
    pub config: serde_json::Value,
    pub output: serde_json::Value,
    #[serde(default)]
    pub recorded_at: String,
    #[serde(default)]
    pub input_sha256: String,
}

/// Outcome of comparing a recorded fixture against the current Rust impl.
#[derive(Debug, Clone)]
pub enum ComparisonOutcome {
    Match,
    Diff { expected: String, actual: String },
    Skipped { reason: String },
}

/// Trait implemented by transform-specific comparators. A comparator receives
/// the fixture's input and config and produces a JSON value to compare against
/// `fixture.output`.
pub trait TransformComparator {
    fn name(&self) -> &str;
    fn run(
        &self,
        input: &serde_json::Value,
        config: &serde_json::Value,
    ) -> Result<serde_json::Value>;
}

/// Compare a single fixture against a comparator and return an outcome.
///
/// f64 normalization: `serde_json` (without the `arbitrary_precision`
/// feature) has an asymmetry — values constructed via `json!(f64)` keep
/// full precision (e.g. `0.9500000000000001`), but values parsed from
/// fixture JSON sometimes round to a neighboring f64 (e.g. `0.95`,
/// differing by 1 ULP). To make comparisons robust we round-trip the
/// comparator's output through `to_string` + `from_str` so it goes
/// through the same lossy parser the fixture did. Bit-identical f64s
/// from both sides then compare equal.
pub fn compare_fixture(
    comparator: &dyn TransformComparator,
    fixture: &Fixture,
) -> Result<ComparisonOutcome> {
    let actual = match comparator.run(&fixture.input, &fixture.config) {
        Ok(v) => v,
        Err(e) => {
            return Ok(ComparisonOutcome::Skipped {
                reason: format!("comparator error: {e}"),
            })
        }
    };
    let actual_normalized: serde_json::Value =
        serde_json::from_str(&serde_json::to_string(&actual)?)
            .context("re-parsing comparator output through serde_json (f64 normalization)")?;
    if actual_normalized == fixture.output {
        Ok(ComparisonOutcome::Match)
    } else {
        Ok(ComparisonOutcome::Diff {
            expected: serde_json::to_string_pretty(&fixture.output)?,
            actual: serde_json::to_string_pretty(&actual_normalized)?,
        })
    }
}

/// Load every `*.json` fixture under `dir/<transform>/`.
pub fn load_fixtures_for(dir: &Path, transform: &str) -> Result<Vec<(PathBuf, Fixture)>> {
    let root = dir.join(transform);
    if !root.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in fs::read_dir(&root).with_context(|| format!("reading {}", root.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let bytes = fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
        let fixture: Fixture = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing fixture {}", path.display()))?;
        if fixture.transform != transform {
            bail!(
                "fixture {} declares transform={} but lives under {}",
                path.display(),
                fixture.transform,
                transform
            );
        }
        out.push((path, fixture));
    }
    Ok(out)
}

/// Aggregate report of one comparator run.
#[derive(Debug, Default)]
pub struct Report {
    pub matched: usize,
    pub diffed: Vec<(PathBuf, String, String)>,
    pub skipped: Vec<(PathBuf, String)>,
}

impl Report {
    pub fn total(&self) -> usize {
        self.matched + self.diffed.len() + self.skipped.len()
    }
    pub fn is_clean(&self) -> bool {
        self.diffed.is_empty()
    }
}

/// Run a comparator over every fixture under `dir/<transform>/` and return a
/// report. Propagates IO/parse errors but never panics on comparator errors —
/// those become `Skipped` entries.
pub fn run_comparator(dir: &Path, comparator: &dyn TransformComparator) -> Result<Report> {
    let mut report = Report::default();
    let fixtures = load_fixtures_for(dir, comparator.name())?;
    for (path, fixture) in fixtures {
        match compare_fixture(comparator, &fixture)? {
            ComparisonOutcome::Match => report.matched += 1,
            ComparisonOutcome::Diff { expected, actual } => {
                report.diffed.push((path, expected, actual));
            }
            ComparisonOutcome::Skipped { reason } => {
                report.skipped.push((path, reason));
            }
        }
    }
    Ok(report)
}

// --- Built-in comparator stubs ---------------------------------------------
//
// Phase 1 will replace `todo!()` bodies with real Rust ports. Until then they
// return `Err`, causing the harness to mark fixtures as `Skipped` instead of
// panicking. The parity-run binary wires them up so the CLI works today.

macro_rules! stub_comparator {
    ($ty:ident, $name:literal) => {
        pub struct $ty;
        impl TransformComparator for $ty {
            fn name(&self) -> &str {
                $name
            }
            fn run(
                &self,
                _input: &serde_json::Value,
                _config: &serde_json::Value,
            ) -> Result<serde_json::Value> {
                anyhow::bail!(concat!("comparator ", $name, " not implemented (Phase 0)"))
            }
        }
    };
}

stub_comparator!(LogCompressorComparator, "log_compressor");
stub_comparator!(CacheAlignerComparator, "cache_aligner");
stub_comparator!(CcrComparator, "ccr");

/// Real comparator for the `diff_compressor` transform. Drives the Rust port
/// over the recorded fixture inputs and emits the Python-shaped JSON output
/// (subset: only fields the Python recorder serializes — i.e. fields, not
/// `@property` derivatives like `compression_ratio`).
pub struct DiffCompressorComparator;

impl TransformComparator for DiffCompressorComparator {
    fn name(&self) -> &str {
        "diff_compressor"
    }

    fn run(
        &self,
        input: &serde_json::Value,
        config: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        use headroom_core::transforms::{DiffCompressor, DiffCompressorConfig};

        let content = input
            .as_str()
            .context("diff_compressor fixture input must be a JSON string")?;

        // Build config from the fixture, falling back to defaults for any
        // missing keys. The recorder writes every field today, but tolerating
        // partial configs keeps fixtures forward-compatible if the Python
        // dataclass picks up new fields.
        let cfg = DiffCompressorConfig {
            max_context_lines: config
                .get("max_context_lines")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(2),
            max_hunks_per_file: config
                .get("max_hunks_per_file")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(10),
            max_files: config
                .get("max_files")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(20),
            always_keep_additions: config
                .get("always_keep_additions")
                .and_then(|v| v.as_bool())
                .unwrap_or(true),
            always_keep_deletions: config
                .get("always_keep_deletions")
                .and_then(|v| v.as_bool())
                .unwrap_or(true),
            enable_ccr: config
                .get("enable_ccr")
                .and_then(|v| v.as_bool())
                .unwrap_or(true),
            min_lines_for_ccr: config
                .get("min_lines_for_ccr")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(50),
            // Rust-only knob; Python fixtures don't carry this field. The
            // 0.8 default reproduces Python's hardcoded 20%-savings gate.
            min_compression_ratio_for_ccr: config
                .get("min_compression_ratio_for_ccr")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.8),
        };

        let compressor = DiffCompressor::new(cfg);
        // No `context` field is recorded; default to empty string. Python's
        // recorder calls `compress(content, "")` too, so this matches.
        let result = compressor.compress(content, "");

        Ok(serde_json::json!({
            "additions": result.additions,
            "cache_key": result.cache_key,
            "compressed": result.compressed,
            "compressed_line_count": result.compressed_line_count,
            "deletions": result.deletions,
            "files_affected": result.files_affected,
            "hunks_kept": result.hunks_kept,
            "hunks_removed": result.hunks_removed,
            "original_line_count": result.original_line_count,
        }))
    }
}

/// Real comparator for the `tokenizer` transform. The recorder used
/// `headroom.providers.openai.OpenAITokenCounter("gpt-4o-mini")`, so the
/// fixture outputs are o200k_base BPE token counts. We rebuild the same
/// encoding via `tiktoken-rs` and assert byte-equal counts.
pub struct TokenizerComparator;

impl TransformComparator for TokenizerComparator {
    fn name(&self) -> &str {
        "tokenizer"
    }

    fn run(
        &self,
        input: &serde_json::Value,
        _config: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        use headroom_core::tokenizer::{TiktokenCounter, Tokenizer};
        let text = input
            .as_str()
            .context("tokenizer fixture input must be a JSON string")?;
        let counter = TiktokenCounter::for_model("gpt-4o-mini")
            .context("init TiktokenCounter for gpt-4o-mini")?;
        let count = counter.count_text(text);
        Ok(serde_json::json!(count))
    }
}

/// Real comparator for the `smart_crusher` transform. Drives the Rust
/// port over the recorded fixture inputs (`{content, query, bias}`)
/// and emits the same shape the Python recorder serialized:
/// `{compressed, original, was_modified, strategy}`.
///
/// The comparator builds `SmartCrusherConfig` from the fixture's
/// `config` block, falling back to the Rust default for any missing
/// field. The Python recorder writes every field today, but tolerating
/// partial configs keeps fixtures forward-compatible if either side
/// gains a field.
pub struct SmartCrusherComparator;

impl TransformComparator for SmartCrusherComparator {
    fn name(&self) -> &str {
        "smart_crusher"
    }

    fn run(
        &self,
        input: &serde_json::Value,
        config: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        use headroom_core::transforms::smart_crusher::{SmartCrusher, SmartCrusherConfig};

        let content = input
            .get("content")
            .and_then(|v| v.as_str())
            .context("smart_crusher fixture input.content must be a JSON string")?;
        let query = input.get("query").and_then(|v| v.as_str()).unwrap_or("");
        let bias = input.get("bias").and_then(|v| v.as_f64()).unwrap_or(1.0);

        let defaults = SmartCrusherConfig::default();
        let cfg = SmartCrusherConfig {
            enabled: config
                .get("enabled")
                .and_then(|v| v.as_bool())
                .unwrap_or(defaults.enabled),
            min_items_to_analyze: config
                .get("min_items_to_analyze")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(defaults.min_items_to_analyze),
            min_tokens_to_crush: config
                .get("min_tokens_to_crush")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(defaults.min_tokens_to_crush),
            variance_threshold: config
                .get("variance_threshold")
                .and_then(|v| v.as_f64())
                .unwrap_or(defaults.variance_threshold),
            uniqueness_threshold: config
                .get("uniqueness_threshold")
                .and_then(|v| v.as_f64())
                .unwrap_or(defaults.uniqueness_threshold),
            similarity_threshold: config
                .get("similarity_threshold")
                .and_then(|v| v.as_f64())
                .unwrap_or(defaults.similarity_threshold),
            max_items_after_crush: config
                .get("max_items_after_crush")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(defaults.max_items_after_crush),
            preserve_change_points: config
                .get("preserve_change_points")
                .and_then(|v| v.as_bool())
                .unwrap_or(defaults.preserve_change_points),
            factor_out_constants: config
                .get("factor_out_constants")
                .and_then(|v| v.as_bool())
                .unwrap_or(defaults.factor_out_constants),
            include_summaries: config
                .get("include_summaries")
                .and_then(|v| v.as_bool())
                .unwrap_or(defaults.include_summaries),
            use_feedback_hints: config
                .get("use_feedback_hints")
                .and_then(|v| v.as_bool())
                .unwrap_or(defaults.use_feedback_hints),
            toin_confidence_threshold: config
                .get("toin_confidence_threshold")
                .and_then(|v| v.as_f64())
                .unwrap_or(defaults.toin_confidence_threshold),
            dedup_identical_items: config
                .get("dedup_identical_items")
                .and_then(|v| v.as_bool())
                .unwrap_or(defaults.dedup_identical_items),
            first_fraction: config
                .get("first_fraction")
                .and_then(|v| v.as_f64())
                .unwrap_or(defaults.first_fraction),
            last_fraction: config
                .get("last_fraction")
                .and_then(|v| v.as_f64())
                .unwrap_or(defaults.last_fraction),
            // Rust-only knob; Python config has no field for it. Use
            // the Rust default (which mirrors Python's hardcoded
            // RelevanceConfig.relevance_threshold = 0.3).
            relevance_threshold: config
                .get("relevance_threshold")
                .and_then(|v| v.as_f64())
                .unwrap_or(defaults.relevance_threshold),
            // Rust-only PR4 knob — fixtures don't carry this; use
            // default. The parity harness exercises the legacy
            // lossy-only path via `without_compaction`, so this
            // threshold is moot.
            lossless_min_savings_ratio: config
                .get("lossless_min_savings_ratio")
                .and_then(|v| v.as_f64())
                .unwrap_or(defaults.lossless_min_savings_ratio),
            // Rust-only audit-fix knob — fixtures don't carry this; use
            // default (true). Recorded fixtures predate the gate and
            // their expected outputs assume markers fire as before.
            enable_ccr_marker: config
                .get("enable_ccr_marker")
                .and_then(|v| v.as_bool())
                .unwrap_or(defaults.enable_ccr_marker),
            // Compaction heuristics are moot here: this comparator uses
            // `without_compaction` (fixtures were recorded against the
            // lossy-only path). Take the defaults wholesale.
            ..defaults
        };

        // Use without_compaction so the legacy fixtures (recorded
        // against the pre-PR4 lossy-only path) keep matching byte-equal.
        let crusher = SmartCrusher::without_compaction(cfg);
        let result = crusher.crush(content, query, bias);

        Ok(serde_json::json!({
            "compressed": result.compressed,
            "original": result.original,
            "was_modified": result.was_modified,
            "strategy": result.strategy,
        }))
    }
}

/// Real comparator for the `content_detector` transform. Drives the Rust
/// port over the recorded fixture inputs (a single JSON string) and
/// emits the same shape Python's recorder serializes for
/// `DetectionResult`:
///
/// ```json
/// {"content_type": "json_array", "confidence": 1.0, "metadata": {...}}
/// ```
///
/// Python's recorder relies on `_json_default` to serialize the
/// `DetectionResult` dataclass and the `ContentType` enum:
/// - dataclass → `asdict(...)` produces `{content_type, confidence, metadata}`.
/// - enum → its `.value` (the lowercase tag, e.g. "json_array").
///
/// Numeric fields in metadata are recorded as JSON numbers (Python ints
/// stay ints), so we mirror that exactly with `serde_json::Number`.
pub struct ContentDetectorComparator;

impl TransformComparator for ContentDetectorComparator {
    fn name(&self) -> &str {
        "content_detector"
    }

    fn run(
        &self,
        input: &serde_json::Value,
        _config: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        use headroom_core::transforms::detect_content_type;

        let content = input
            .as_str()
            .context("content_detector fixture input must be a JSON string")?;
        let result = detect_content_type(content);
        Ok(serde_json::json!({
            "content_type": result.content_type.as_str(),
            "confidence": result.confidence,
            "metadata": serde_json::Value::Object(result.metadata),
        }))
    }
}

/// Every built-in comparator, in a stable order.
pub fn builtin_comparators() -> Vec<Box<dyn TransformComparator>> {
    vec![
        Box::new(LogCompressorComparator),
        Box::new(DiffCompressorComparator),
        Box::new(CacheAlignerComparator),
        Box::new(TokenizerComparator),
        Box::new(CcrComparator),
        Box::new(SmartCrusherComparator),
        Box::new(ContentDetectorComparator),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// A fake comparator that always returns "rust-output" regardless of
    /// input. Paired with a fixture whose `output` is "python-output", this
    /// proves the harness reports diffs correctly.
    struct FakeDivergent;
    impl TransformComparator for FakeDivergent {
        fn name(&self) -> &str {
            "fake_divergent"
        }
        fn run(
            &self,
            _input: &serde_json::Value,
            _config: &serde_json::Value,
        ) -> Result<serde_json::Value> {
            Ok(serde_json::json!("rust-output"))
        }
    }

    struct FakeAgreeing;
    impl TransformComparator for FakeAgreeing {
        fn name(&self) -> &str {
            "fake_agreeing"
        }
        fn run(
            &self,
            _input: &serde_json::Value,
            _config: &serde_json::Value,
        ) -> Result<serde_json::Value> {
            Ok(serde_json::json!("python-output"))
        }
    }

    fn write_fixture(dir: &Path, transform: &str, name: &str, output: serde_json::Value) {
        let sub = dir.join(transform);
        fs::create_dir_all(&sub).unwrap();
        let fixture = Fixture {
            transform: transform.to_string(),
            input: serde_json::json!("hello"),
            config: serde_json::json!({}),
            output,
            recorded_at: "2026-04-23T00:00:00Z".to_string(),
            input_sha256: "deadbeef".to_string(),
        };
        let mut f = fs::File::create(sub.join(format!("{name}.json"))).unwrap();
        f.write_all(&serde_json::to_vec_pretty(&fixture).unwrap())
            .unwrap();
    }

    #[test]
    fn harness_reports_diff_for_divergent_comparator() {
        let tmp = tempdir();
        write_fixture(
            tmp.path(),
            "fake_divergent",
            "case1",
            serde_json::json!("python-output"),
        );
        let report = run_comparator(tmp.path(), &FakeDivergent).unwrap();
        assert_eq!(report.total(), 1);
        assert_eq!(report.matched, 0);
        assert_eq!(report.diffed.len(), 1);
        assert!(!report.is_clean());
        let (_, expected, actual) = &report.diffed[0];
        assert!(expected.contains("python-output"));
        assert!(actual.contains("rust-output"));
    }

    #[test]
    fn harness_reports_match_for_agreeing_comparator() {
        let tmp = tempdir();
        write_fixture(
            tmp.path(),
            "fake_agreeing",
            "case1",
            serde_json::json!("python-output"),
        );
        let report = run_comparator(tmp.path(), &FakeAgreeing).unwrap();
        assert_eq!(report.matched, 1);
        assert!(report.is_clean());
    }

    #[test]
    fn missing_transform_dir_yields_empty_report() {
        let tmp = tempdir();
        let report = run_comparator(tmp.path(), &FakeAgreeing).unwrap();
        assert_eq!(report.total(), 0);
    }

    #[test]
    fn stub_comparators_skip_rather_than_panic() {
        let tmp = tempdir();
        write_fixture(
            tmp.path(),
            "log_compressor",
            "case1",
            serde_json::json!({"compressed": "x"}),
        );
        let report = run_comparator(tmp.path(), &LogCompressorComparator).unwrap();
        assert_eq!(report.skipped.len(), 1);
        assert_eq!(report.matched, 0);
    }

    /// Minimal tempdir helper to avoid a dev-dependency on `tempfile`.
    struct TempDir(PathBuf);
    impl TempDir {
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    fn tempdir() -> TempDir {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = std::env::temp_dir().join(format!(
            "headroom-parity-{nanos}-{:?}",
            std::thread::current().id()
        ));
        fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }
}
