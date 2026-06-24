//! Core data types for SmartCrusher.
//!
//! Direct port of the dataclasses in `smart_crusher.py:318-924`. These
//! mirror the Python shapes 1:1 so the PyO3 bridge in stage 3c.1b can
//! reconstruct Python dataclasses from the Rust output without a manual
//! field-by-field translator.

use serde_json::Value;
use std::collections::BTreeMap;

/// Compression strategies based on data patterns.
///
/// Mirrors `CompressionStrategy` enum at `smart_crusher.py:318-326`. The
/// string variants must match Python's `Enum.value` exactly — they appear
/// in strategy debug strings (e.g. `"top_n(100->10)"`) and the parity
/// fixtures lock those bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompressionStrategy {
    /// No compression needed.
    None,
    /// Explicitly skip — not safe to crush.
    Skip,
    /// Time-series: keep change points, summarize stable runs.
    TimeSeries,
    /// Cluster-sample: dedupe similar items.
    ClusterSample,
    /// Top-N: keep highest-scored items.
    TopN,
    /// Smart-sample: statistical sampling with anchor-preservation.
    SmartSample,
}

impl CompressionStrategy {
    /// Lowercase string matching Python's `Enum.value`. Pinned by the
    /// parity fixtures — must not drift.
    pub fn as_str(self) -> &'static str {
        match self {
            CompressionStrategy::None => "none",
            CompressionStrategy::Skip => "skip",
            CompressionStrategy::TimeSeries => "time_series",
            CompressionStrategy::ClusterSample => "cluster",
            CompressionStrategy::TopN => "top_n",
            CompressionStrategy::SmartSample => "smart_sample",
        }
    }
}

/// Statistics for a single field across array items.
///
/// Mirrors the `FieldStats` dataclass at `smart_crusher.py:864-885`.
/// Field naming and Optional<T> shape match Python exactly so the PyO3
/// bridge can `from_dict`-reconstruct the Python dataclass.
#[derive(Debug, Clone)]
pub struct FieldStats {
    pub name: String,
    /// One of: `"numeric"`, `"string"`, `"boolean"`, `"object"`, `"array"`,
    /// `"null"`. String literals match Python's `field_type` values.
    pub field_type: String,
    pub count: usize,
    pub unique_count: usize,
    pub unique_ratio: f64,
    pub is_constant: bool,
    pub constant_value: Option<Value>,

    // Numeric-specific
    pub min_val: Option<f64>,
    pub max_val: Option<f64>,
    pub mean_val: Option<f64>,
    pub variance: Option<f64>,
    pub change_points: Vec<usize>,

    // String-specific
    pub avg_length: Option<f64>,
    /// Top values by frequency, descending. Bounded list so this stays
    /// cheap to build and serialize. Same shape as Python's `list[tuple[str, int]]`.
    pub top_values: Vec<(String, usize)>,
}

/// Analysis of whether an array is safe to crush.
///
/// Mirrors `CrushabilityAnalysis` at `smart_crusher.py:833-860`. The key
/// invariant: **if we don't have a reliable signal to determine which
/// items are important, we don't crush at all**. Signals include score
/// fields, error keywords, numeric anomalies, and low uniqueness.
#[derive(Debug, Clone)]
pub struct CrushabilityAnalysis {
    pub crushable: bool,
    pub confidence: f64,
    pub reason: String,
    pub signals_present: Vec<String>,
    pub signals_absent: Vec<String>,

    // Detailed metrics (mirroring Python field-by-field)
    pub has_id_field: bool,
    pub id_uniqueness: f64,
    pub avg_string_uniqueness: f64,
    pub has_score_field: bool,
    pub error_item_count: usize,
    pub anomaly_count: usize,
}

impl CrushabilityAnalysis {
    /// Helper to build a "not crushable" verdict — used in several early
    /// exits in `analyze_crushability`. Mirrors the Python pattern where
    /// `crushable=False` paths don't bother filling in detail metrics.
    pub fn skip(reason: impl Into<String>, confidence: f64) -> Self {
        CrushabilityAnalysis {
            crushable: false,
            confidence,
            reason: reason.into(),
            signals_present: Vec::new(),
            signals_absent: Vec::new(),
            has_id_field: false,
            id_uniqueness: 0.0,
            avg_string_uniqueness: 0.0,
            has_score_field: false,
            error_item_count: 0,
            anomaly_count: 0,
        }
    }
}

/// Complete analysis of an array.
///
/// Mirrors `ArrayAnalysis` at `smart_crusher.py:887-897`. `field_stats`
/// and `constant_fields` use `BTreeMap` for sorted-by-key iteration.
///
/// # Sort vs insertion order — known parity nuance
///
/// Python's `dict` preserves insertion order, and `_analyze_field` is
/// called once per key as it appears in `items[0].keys()` (i.e., JSON
/// parse order). With `serde_json/preserve_order` enabled at the
/// workspace level, `serde_json::Map` is an `IndexMap` and parse order
/// matches Python.
///
/// `BTreeMap` here gives sorted-key iteration — which differs from
/// Python's parse-order `dict`. This matters only if downstream code
/// observes the iteration order of `field_stats` (e.g., when emitting
/// debug output, picking a "first" field, or computing strategy
/// strings that include field names).
///
/// During the analyzer port (Stage 3c.1 commit 2), we'll either:
///   1. Switch this to `IndexMap` if any code path observes order, OR
///   2. Document that Python's order-sensitive paths get rewritten to
///      iterate sorted, then mirror that in Rust.
///
/// Tracked in the design doc at
/// `~/Desktop/SmartCrusher-Architecture-Improvements.md`.
#[derive(Debug, Clone)]
pub struct ArrayAnalysis {
    pub item_count: usize,
    pub field_stats: BTreeMap<String, FieldStats>,
    /// One of: `"time_series"`, `"logs"`, `"search_results"`, `"generic"`.
    pub detected_pattern: String,
    pub recommended_strategy: CompressionStrategy,
    pub constant_fields: BTreeMap<String, Value>,
    pub estimated_reduction: f64,
    pub crushability: Option<CrushabilityAnalysis>,
}

/// Plan for how to compress an array.
///
/// Mirrors `CompressionPlan` at `smart_crusher.py:900-910`. `keep_indices`
/// is the list of original-array indices that survive compression;
/// `summary_ranges` carries `(start, end, summary_dict)` for runs we
/// summarized rather than dropped (currently unused in the Python impl
/// but plumbed through for parity with the dataclass).
#[derive(Debug, Clone)]
pub struct CompressionPlan {
    pub strategy: CompressionStrategy,
    pub keep_indices: Vec<usize>,
    pub constant_fields: BTreeMap<String, Value>,
    /// `(start, end, summary)` triples for summarized runs. Python uses
    /// `list[tuple[int, int, dict]]`; we use `Value` for the summary so
    /// any JSON shape is representable.
    pub summary_ranges: Vec<(usize, usize, Value)>,
    pub cluster_field: Option<String>,
    pub sort_field: Option<String>,
    pub keep_count: usize,
}

impl Default for CompressionPlan {
    fn default() -> Self {
        // Mirrors Python's @dataclass defaults at line 900-910.
        CompressionPlan {
            strategy: CompressionStrategy::None,
            keep_indices: Vec::new(),
            constant_fields: BTreeMap::new(),
            summary_ranges: Vec::new(),
            cluster_field: None,
            sort_field: None,
            keep_count: 10,
        }
    }
}

/// Result from `SmartCrusher.crush()` — used by ContentRouter when
/// routing JSON arrays. Mirrors `CrushResult` at `smart_crusher.py:913-923`.
#[derive(Debug, Clone)]
pub struct CrushResult {
    pub compressed: String,
    pub original: String,
    pub was_modified: bool,
    pub strategy: String,
}

impl CrushResult {
    /// Pass-through result: same as input, no modification, strategy
    /// `"passthrough"`. Used when content can't be compressed (not JSON,
    /// too small, no crushable arrays, etc.).
    pub fn passthrough(content: impl Into<String>) -> Self {
        let s = content.into();
        CrushResult {
            compressed: s.clone(),
            original: s,
            was_modified: false,
            strategy: "passthrough".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compression_strategy_strings_match_python() {
        // Strategy debug strings appear in the parity fixtures; these must
        // not drift. If a value here changes, every fixture breaks.
        assert_eq!(CompressionStrategy::None.as_str(), "none");
        assert_eq!(CompressionStrategy::Skip.as_str(), "skip");
        assert_eq!(CompressionStrategy::TimeSeries.as_str(), "time_series");
        assert_eq!(CompressionStrategy::ClusterSample.as_str(), "cluster");
        assert_eq!(CompressionStrategy::TopN.as_str(), "top_n");
        assert_eq!(CompressionStrategy::SmartSample.as_str(), "smart_sample");
    }

    #[test]
    fn crushability_skip_helper() {
        let r = CrushabilityAnalysis::skip("too small", 1.0);
        assert!(!r.crushable);
        assert_eq!(r.confidence, 1.0);
        assert_eq!(r.reason, "too small");
    }

    #[test]
    fn compression_plan_default_keep_count_matches_python() {
        // Python's @dataclass default is `keep_count: int = 10`.
        let p = CompressionPlan::default();
        assert_eq!(p.keep_count, 10);
        assert_eq!(p.strategy, CompressionStrategy::None);
        assert!(p.keep_indices.is_empty());
    }

    #[test]
    fn crush_result_passthrough() {
        let r = CrushResult::passthrough("hello");
        assert_eq!(r.compressed, "hello");
        assert_eq!(r.original, "hello");
        assert!(!r.was_modified);
        assert_eq!(r.strategy, "passthrough");
    }
}
