//! Pipeline configuration — TOML-backed defaults plus runtime overrides.
//!
//! # Why a config file
//!
//! The orchestrator and every offload bloat estimator carry tunable
//! thresholds (savings ratios, sample sizes, weighting between bloat
//! signals). Hardcoding them in Rust means a tuning change ships as a
//! binary release. Putting them in TOML — embedded into the binary at
//! build time, override-loadable at startup — gives ops a knob without
//! losing the "stock binary works" property.
//!
//! Defaults live in `crates/headroom-core/config/pipeline.toml` and are
//! pulled in via `include_str!`. `PipelineConfig::default()` deserializes
//! that string; `PipelineConfig::from_toml_str` accepts an override TOML.
//! All thresholds are intentionally conservative — Claude Code, Codex,
//! and similar tool-driven agents sit on the hot path of every
//! compressed response, and a wrongly-fired CCR offload costs both
//! latency (retrieval round trip) and accuracy (LLM may not retrieve
//! when it should).
//!
//! # Schema
//!
//! ```toml
//! [pipeline]
//! reformat_target_ratio = 0.5
//! bloat_threshold = 0.5
//! offload_fallback_ratio = 0.85
//!
//! [bloat.log]
//! min_lines = 50
//! sample_size = 100
//! high_priority_threshold = 0.4
//! uniqueness_weight = 0.5
//! priority_dilution_weight = 0.5
//!
//! [bloat.diff]
//! min_lines = 50
//! normal_context_ratio = 0.6
//!
//! [bloat.search]
//! min_matches = 10
//! cluster_threshold = 10.0
//! ```

use serde::Deserialize;

/// Embedded default configuration. Built into the binary so a stock
/// install needs no external file.
const DEFAULT_TOML: &str = include_str!("../../../config/pipeline.toml");

/// Top-level config, deserialized from TOML.
#[derive(Debug, Clone, Deserialize)]
pub struct PipelineConfig {
    pub pipeline: OrchestratorConfig,
    pub bloat: BloatConfigs,
    pub reformat: ReformatConfigs,
    pub offload: OffloadConfigs,
}

impl PipelineConfig {
    /// Load the embedded defaults. Panics only if the embedded TOML is
    /// malformed — caught at build time by `from_default_str_does_not_panic`.
    pub fn from_default_str() -> Self {
        toml::from_str(DEFAULT_TOML)
            .expect("embedded config/pipeline.toml must parse — checked by tests")
    }

    /// Parse a caller-supplied TOML string. Used by production deployments
    /// that ship their own override file.
    ///
    /// Named `from_toml_str` rather than `from_str` to avoid colliding
    /// with the [`std::str::FromStr`] trait method (and the corresponding
    /// clippy lint), since this fallible parse doesn't need the `FromStr`
    /// surface at any callsite today.
    pub fn from_toml_str(s: &str) -> Result<Self, ConfigError> {
        toml::from_str(s).map_err(ConfigError::from)
    }

    /// Read and parse a TOML file. Convenience wrapper.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, ConfigError> {
        let text = std::fs::read_to_string(path).map_err(ConfigError::Io)?;
        Self::from_toml_str(&text)
    }
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self::from_default_str()
    }
}

/// Orchestrator-level knobs.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct OrchestratorConfig {
    /// After reformat, if `output_len / input_len <= this`, we treat
    /// the reformat as sufficient and skip offloads UNLESS bloat
    /// estimation demands them.
    pub reformat_target_ratio: f64,
    /// Bloat score above which the orchestrator runs offload regardless
    /// of reformat outcome.
    pub bloat_threshold: f32,
    /// After reformat, if `output_len / input_len > this`, we run
    /// offloads even when bloat is below threshold (the "reformat
    /// barely helped" fallback path).
    pub offload_fallback_ratio: f64,
}

/// Per-domain bloat-estimator knobs.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct BloatConfigs {
    pub log: LogBloatConfig,
    pub diff: DiffBloatConfig,
    pub search: SearchBloatConfig,
}

/// Log-domain bloat estimator config.
///
/// The estimator combines two structural signals weighted to sum to ≤ 1.0:
/// - **Repetition** (`uniqueness_weight`): `1 − unique_lines / sample_size`.
///   Catches "log full of identical INFO heartbeats."
/// - **Priority dilution** (`priority_dilution_weight`):
///   `(low_priority_lines / sample_size)`, where low-priority is
///   `priority ≤ high_priority_threshold`. Catches "log full of unique
///   noise burying a few errors."
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct LogBloatConfig {
    pub min_lines: usize,
    pub sample_size: usize,
    pub high_priority_threshold: f32,
    pub uniqueness_weight: f32,
    pub priority_dilution_weight: f32,
}

/// Diff-domain bloat estimator config.
///
/// Bloat is high when context lines dominate change lines: a diff of
/// 200 lines that only changes 3 of them is mostly noise the LLM
/// doesn't need on the wire.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct DiffBloatConfig {
    pub min_lines: usize,
    /// Below this fraction of `context / (context + change)` the diff
    /// is considered dense; above, it's mostly context (high bloat).
    pub normal_context_ratio: f64,
}

/// Search-domain bloat estimator config.
///
/// Bloat is high when matches cluster heavily into a few files
/// (`avg_matches_per_file` is large) — the LLM rarely needs all
/// 50 hits in `utils.py` and one summary keeps the signal.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct SearchBloatConfig {
    pub min_matches: usize,
    pub cluster_threshold: f32,
}

/// Reformat-transform configs (one struct per reformat).
#[derive(Debug, Clone, Deserialize)]
pub struct ReformatConfigs {
    pub log_template: LogTemplateConfig,
}

/// Knobs for the [`crate::transforms::pipeline::reformats::LogTemplate`]
/// miner. See module docs for the algorithm; the relevant tunables are:
///
/// - `min_lines` — short logs aren't worth the bucket walk.
/// - `min_run` — minimum consecutive same-template lines that justify
///   collapsing into a template block.
/// - `similarity_threshold` — fraction of positions that must match
///   for two same-length lines to be considered the same template.
/// - `min_constant_tokens` — a template with too few anchor tokens
///   carries no readable signal; emit verbatim instead.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct LogTemplateConfig {
    pub min_lines: usize,
    pub min_run: usize,
    pub similarity_threshold: f32,
    pub min_constant_tokens: usize,
}

/// Offload-transform configs (one struct per offload that needs them).
#[derive(Debug, Clone, Deserialize)]
pub struct OffloadConfigs {
    pub json: JsonOffloadConfig,
    pub diff_noise: DiffNoiseConfig,
}

/// Knobs for the [`crate::transforms::pipeline::offloads::JsonOffload`]
/// (SmartCrusher wrapper). The estimator scans byte-prefix-cheaply for
/// JSON array-of-objects shape; SmartCrusher itself does the heavy work
/// when the orchestrator decides to fire.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct JsonOffloadConfig {
    pub min_array_rows: usize,
    pub saturation_rows: usize,
}

/// Knobs for the [`crate::transforms::pipeline::offloads::DiffNoise`]
/// offload. Lockfile suffixes are matched against the new-file path
/// at the end of each `diff --git` header.
#[derive(Debug, Clone, Deserialize)]
pub struct DiffNoiseConfig {
    pub min_lines: usize,
    pub lockfile_suffixes: Vec<String>,
    pub drop_whitespace_only_hunks: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("invalid pipeline config TOML: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("could not read pipeline config file: {0}")]
    Io(std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_default_str_does_not_panic() {
        // Embedded TOML must always deserialize cleanly.
        let _ = PipelineConfig::from_default_str();
    }

    #[test]
    fn defaults_match_documented_thresholds() {
        let cfg = PipelineConfig::default();
        assert_eq!(cfg.pipeline.reformat_target_ratio, 0.5);
        assert_eq!(cfg.pipeline.bloat_threshold, 0.5);
        assert_eq!(cfg.pipeline.offload_fallback_ratio, 0.85);
        assert_eq!(cfg.bloat.log.min_lines, 50);
        assert_eq!(cfg.bloat.log.sample_size, 100);
        assert_eq!(cfg.bloat.diff.min_lines, 50);
        assert_eq!(cfg.bloat.search.min_matches, 10);
    }

    #[test]
    fn bloat_log_weights_sum_to_at_most_one() {
        let cfg = PipelineConfig::default();
        let total = cfg.bloat.log.uniqueness_weight + cfg.bloat.log.priority_dilution_weight;
        assert!(
            total <= 1.0001,
            "log bloat weights must sum to ≤ 1.0, got {total}"
        );
    }

    #[test]
    fn from_toml_str_overrides_defaults() {
        let toml = r#"
            [pipeline]
            reformat_target_ratio = 0.3
            bloat_threshold = 0.7
            offload_fallback_ratio = 0.9

            [bloat.log]
            min_lines = 25
            sample_size = 50
            high_priority_threshold = 0.6
            uniqueness_weight = 0.4
            priority_dilution_weight = 0.6

            [bloat.diff]
            min_lines = 30
            normal_context_ratio = 0.7

            [bloat.search]
            min_matches = 5
            cluster_threshold = 20.0

            [reformat.log_template]
            min_lines = 10
            min_run = 5
            similarity_threshold = 0.8
            min_constant_tokens = 3

            [offload.json]
            min_array_rows = 3
            saturation_rows = 25

            [offload.diff_noise]
            min_lines = 20
            lockfile_suffixes = ["custom.lock"]
            drop_whitespace_only_hunks = false
        "#;
        let cfg = PipelineConfig::from_toml_str(toml).expect("override parses");
        assert_eq!(cfg.pipeline.reformat_target_ratio, 0.3);
        assert_eq!(cfg.bloat.log.min_lines, 25);
        assert_eq!(cfg.bloat.search.cluster_threshold, 20.0);
        assert_eq!(cfg.reformat.log_template.min_run, 5);
        assert_eq!(
            cfg.offload.diff_noise.lockfile_suffixes,
            vec!["custom.lock"]
        );
    }

    #[test]
    fn defaults_carry_reformat_and_offload_sections() {
        let cfg = PipelineConfig::default();
        assert_eq!(cfg.reformat.log_template.min_lines, 20);
        assert_eq!(cfg.reformat.log_template.min_run, 3);
        assert_eq!(cfg.offload.json.min_array_rows, 5);
        assert_eq!(cfg.offload.json.saturation_rows, 50);
        assert!(!cfg.offload.diff_noise.lockfile_suffixes.is_empty());
        assert!(cfg
            .offload
            .diff_noise
            .lockfile_suffixes
            .iter()
            .any(|s| s == "Cargo.lock"));
    }

    #[test]
    fn malformed_toml_returns_error() {
        let r = PipelineConfig::from_toml_str("this is not toml = [unterminated");
        assert!(r.is_err(), "malformed TOML should fail loudly");
    }

    #[test]
    fn missing_section_returns_error() {
        // Only `[pipeline]` — missing `[bloat.*]` sections.
        let toml = r#"
            [pipeline]
            reformat_target_ratio = 0.5
            bloat_threshold = 0.5
            offload_fallback_ratio = 0.85
        "#;
        assert!(PipelineConfig::from_toml_str(toml).is_err());
    }
}
