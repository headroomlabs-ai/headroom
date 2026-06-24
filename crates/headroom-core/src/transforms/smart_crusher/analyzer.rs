//! `SmartAnalyzer` — statistical brain that decides whether and how to crush
//! a JSON array.
//!
//! Direct port of the Python `SmartAnalyzer` class at `smart_crusher.py:960-1489`.
//! All eight methods are mirrored faithfully here:
//!
//! - `analyze_array` — top-level entry: builds field stats, detects pattern,
//!   runs crushability, picks strategy.
//! - `analyze_field` — per-field statistics (counts, uniqueness, type-specific).
//! - `detect_change_points` — sliding-window mean shift detector for numeric
//!   fields.
//! - `detect_pattern` — classifies the array as `time_series`, `logs`,
//!   `search_results`, or `generic`.
//! - `detect_temporal_field` — structural date/timestamp detection (no
//!   field-name heuristics).
//! - `analyze_crushability` — the main "is it SAFE to crush?" decision.
//! - `select_strategy` — picks `CompressionStrategy` from pattern + crushability.
//! - `estimate_reduction` — coarse compression-ratio estimate for telemetry.
//!
//! # Field iteration order — known parity nuance
//!
//! Python's `_analyze_field` is called once per key in `set(item.keys())`
//! union, then results are stored in a dict. **Set iteration order is
//! non-deterministic in CPython** (depends on hash and insertion). Several
//! downstream paths short-circuit on first match (`_select_strategy` for
//! "message"-like field, `_detect_pattern` for first score field), so a
//! field-order divergence between Python and Rust would silently change
//! parity output.
//!
//! Rust uses `BTreeMap<String, FieldStats>` here, which iterates in
//! ASCII-sorted key order. To match, the Python implementation needs a
//! `sorted(all_keys)` call before building `field_stats`. Tracked as bug
//! #5 in the architecture doc; the Python fix lands in Stage 3c.1
//! commit 7 alongside fixture regeneration.

use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

use super::config::SmartCrusherConfig;
use super::field_detect::{detect_id_field_statistically, detect_score_field_statistically};
use super::stats_math::{mean, sample_stdev, sample_variance};
use super::types::{ArrayAnalysis, CompressionStrategy, CrushabilityAnalysis, FieldStats};

/// Statistical analyzer for compression decisions.
///
/// Stateless aside from `config`. Construct once per request and call
/// `analyze_array` per array — same API as Python.
pub struct SmartAnalyzer {
    pub config: SmartCrusherConfig,
}

impl SmartAnalyzer {
    pub fn new(config: SmartCrusherConfig) -> Self {
        SmartAnalyzer { config }
    }

    /// Top-level analysis. Mirrors `analyze_array` at
    /// `smart_crusher.py:966-1014`.
    pub fn analyze_array(&self, items: &[Value]) -> ArrayAnalysis {
        // Empty / non-dict-first guard: Python returns NONE strategy with
        // empty stats. We mirror exactly.
        let first_is_dict = items.first().map(|v| v.is_object()).unwrap_or(false);
        if !first_is_dict {
            return ArrayAnalysis {
                item_count: items.len(),
                field_stats: BTreeMap::new(),
                detected_pattern: "generic".to_string(),
                recommended_strategy: CompressionStrategy::None,
                constant_fields: BTreeMap::new(),
                estimated_reduction: 0.0,
                crushability: None,
            };
        }

        // Union of all keys across dict items. BTreeSet → sorted iteration,
        // matching the BTreeMap we'll build below. Python also unions keys
        // but iterates a set; sorted order is the deterministic choice for
        // both languages.
        let mut all_keys: BTreeSet<String> = BTreeSet::new();
        for item in items {
            if let Some(obj) = item.as_object() {
                for k in obj.keys() {
                    all_keys.insert(k.clone());
                }
            }
        }

        let mut field_stats: BTreeMap<String, FieldStats> = BTreeMap::new();
        for key in &all_keys {
            field_stats.insert(key.clone(), self.analyze_field(key, items));
        }

        let pattern = self.detect_pattern(&field_stats, items);

        // Constant fields: name → value snapshot. Iteration is BTreeMap
        // sorted, so result map is also key-sorted.
        let constant_fields: BTreeMap<String, Value> = field_stats
            .iter()
            .filter_map(|(k, v)| {
                if v.is_constant {
                    v.constant_value.clone().map(|val| (k.clone(), val))
                } else {
                    None
                }
            })
            .collect();

        let crushability = self.analyze_crushability(items, &field_stats);

        let strategy =
            self.select_strategy(&field_stats, &pattern, items.len(), Some(&crushability));

        let reduction = if strategy == CompressionStrategy::Skip {
            0.0
        } else {
            self.estimate_reduction(&field_stats, strategy, items.len())
        };

        ArrayAnalysis {
            item_count: items.len(),
            field_stats,
            detected_pattern: pattern,
            recommended_strategy: strategy,
            constant_fields,
            estimated_reduction: reduction,
            crushability: Some(crushability),
        }
    }

    /// Per-field statistics. Mirrors `_analyze_field` at
    /// `smart_crusher.py:1016-1093`.
    pub fn analyze_field(&self, key: &str, items: &[Value]) -> FieldStats {
        // Collect raw values across dict items. `item.get(key)` in Python
        // returns None for missing keys; serde_json returns Value::Null
        // for explicit nulls but no entry for missing. Mirror both as
        // Value::Null in our local `values` vec — Python's downstream
        // `non_null_values` filter unifies both forms anyway.
        let values: Vec<Value> = items
            .iter()
            .filter_map(|i| i.as_object())
            .map(|obj| obj.get(key).cloned().unwrap_or(Value::Null))
            .collect();
        let non_null: Vec<&Value> = values.iter().filter(|v| !v.is_null()).collect();

        if non_null.is_empty() {
            return FieldStats {
                name: key.to_string(),
                field_type: "null".to_string(),
                count: values.len(),
                unique_count: 0,
                unique_ratio: 0.0,
                is_constant: true,
                constant_value: None,
                min_val: None,
                max_val: None,
                mean_val: None,
                variance: None,
                change_points: Vec::new(),
                avg_length: None,
                top_values: Vec::new(),
            };
        }

        let first = non_null[0];
        // Python `isinstance(first, bool)` precedes `int|float` — bool is
        // a subclass of int in Python. We model JSON's bool/number split
        // directly: serde_json::Value::Bool vs Value::Number.
        let field_type = match first {
            Value::Bool(_) => "boolean",
            Value::Number(_) => "numeric",
            Value::String(_) => "string",
            Value::Object(_) => "object",
            Value::Array(_) => "array",
            _ => "unknown",
        }
        .to_string();

        // Uniqueness: stringify ALL values (including nulls), dedupe, count.
        // Python: `str(v)` for any v (None → "None"). Match exactly to keep
        // unique-count parity with fixtures. python_repr handles None as
        // "None", bool as "True"/"False", etc.
        let str_values: Vec<String> = values.iter().map(python_repr).collect();
        let unique_set: BTreeSet<&String> = str_values.iter().collect();
        let unique_count = unique_set.len();
        let unique_ratio = if values.is_empty() {
            0.0
        } else {
            unique_count as f64 / values.len() as f64
        };

        let is_constant = unique_count == 1;
        let constant_value = if is_constant {
            Some(non_null[0].clone())
        } else {
            None
        };

        let mut stats = FieldStats {
            name: key.to_string(),
            field_type: field_type.clone(),
            count: values.len(),
            unique_count,
            unique_ratio,
            is_constant,
            constant_value,
            min_val: None,
            max_val: None,
            mean_val: None,
            variance: None,
            change_points: Vec::new(),
            avg_length: None,
            top_values: Vec::new(),
        };

        match field_type.as_str() {
            "numeric" => {
                // Filter to finite f64 only — Python rejects NaN/Inf via
                // `math.isfinite`. We mirror exactly so the same set of
                // values feeds mean/variance/change-points.
                let nums: Vec<f64> = non_null
                    .iter()
                    .filter_map(|v| v.as_f64().filter(|f| f.is_finite()))
                    .collect();
                if !nums.is_empty() {
                    let min_val = nums.iter().cloned().reduce(f64::min);
                    let max_val = nums.iter().cloned().reduce(f64::max);
                    let mean_val = mean(&nums);
                    // `variance = 0` when n < 2 (Python: `if len(nums) > 1`).
                    let variance = if nums.len() > 1 {
                        sample_variance(&nums)
                    } else {
                        Some(0.0)
                    };
                    // Python wraps the numeric-stats block in
                    // `try/except (OverflowError, ValueError)` and resets
                    // ALL fields to None on failure. Mirror that
                    // all-or-nothing reset: if any computed stat is non-
                    // finite (or computation returned None), drop the
                    // entire numeric stats block and leave change_points
                    // empty.
                    let all_finite = mean_val.map(f64::is_finite).unwrap_or(false)
                        && variance.map(f64::is_finite).unwrap_or(false)
                        && min_val.map(f64::is_finite).unwrap_or(false)
                        && max_val.map(f64::is_finite).unwrap_or(false);
                    if all_finite {
                        stats.min_val = min_val;
                        stats.max_val = max_val;
                        stats.mean_val = mean_val;
                        stats.variance = variance;
                        stats.change_points = self.detect_change_points(&nums, 5);
                    } else {
                        // Python parity: the except block sets `variance = 0`
                        // (int literal) but min/max/mean to None. Downstream
                        // truthiness checks (`if stats.variance:` and
                        // `(variance or 0) > 0`) treat 0 the same as None,
                        // but the FieldStats serialization shape matters
                        // for parity fixtures. Pin variance to Some(0.0).
                        stats.min_val = None;
                        stats.max_val = None;
                        stats.mean_val = None;
                        stats.variance = Some(0.0);
                        stats.change_points = Vec::new();
                    }
                }
            }
            "string" => {
                let strs: Vec<&str> = non_null.iter().filter_map(|v| v.as_str()).collect();
                if !strs.is_empty() {
                    let lens: Vec<f64> = strs.iter().map(|s| s.chars().count() as f64).collect();
                    stats.avg_length = mean(&lens);
                    stats.top_values = top_n_by_count(&strs, 5);
                }
            }
            _ => {}
        }

        stats
    }

    /// Sliding-window change-point detector. Mirrors `_detect_change_points`
    /// at `smart_crusher.py:1095-1125`.
    pub fn detect_change_points(&self, values: &[f64], window: usize) -> Vec<usize> {
        if values.len() < window * 2 {
            return Vec::new();
        }

        let overall_std = match sample_stdev(values) {
            Some(s) if s > 0.0 => s,
            _ => return Vec::new(),
        };

        let threshold = self.config.variance_threshold * overall_std;

        // Python: `for i in range(window, len(values) - window)`.
        let mut change_points: Vec<usize> = Vec::new();
        for i in window..values.len().saturating_sub(window) {
            let before = mean(&values[i - window..i]).unwrap_or(0.0);
            let after = mean(&values[i..i + window]).unwrap_or(0.0);
            if (after - before).abs() > threshold {
                change_points.push(i);
            }
        }

        if change_points.is_empty() {
            return Vec::new();
        }

        // Greedy dedup: keep first, then any cp where `cp - last > window`.
        let mut deduped: Vec<usize> = vec![change_points[0]];
        for &cp in &change_points[1..] {
            let last = *deduped.last().unwrap();
            if cp - last > window {
                deduped.push(cp);
            }
        }
        deduped
    }

    /// Pattern classifier. Mirrors `_detect_pattern` at
    /// `smart_crusher.py:1127-1171`. Returns one of `time_series`, `logs`,
    /// `search_results`, `generic`.
    pub fn detect_pattern(
        &self,
        field_stats: &BTreeMap<String, FieldStats>,
        items: &[Value],
    ) -> String {
        let has_timestamp = self.detect_temporal_field(field_stats, items);

        let has_numeric_with_variance = field_stats
            .values()
            .filter(|v| v.field_type == "numeric")
            .any(|v| v.variance.unwrap_or(0.0) > 0.0);

        if has_timestamp && has_numeric_with_variance {
            return "time_series".to_string();
        }

        // logs pattern: high-cardinality string (message) + low-cardinality
        // categorical (level/status).
        let mut has_message_like = false;
        let mut has_level_like = false;
        for stats in field_stats.values() {
            if stats.field_type != "string" {
                continue;
            }
            let avg_len = stats.avg_length.unwrap_or(0.0);
            if stats.unique_ratio > 0.5 && avg_len > 20.0 {
                has_message_like = true;
            } else if stats.unique_ratio < 0.1 && (2..=10).contains(&stats.unique_count) {
                has_level_like = true;
            }
        }
        if has_message_like && has_level_like {
            return "logs".to_string();
        }

        // search_results: any field with score-like signal at confidence >=0.5.
        for stats in field_stats.values() {
            let (is_score, confidence) = detect_score_field_statistically(stats, items);
            if is_score && confidence >= 0.5 {
                return "search_results".to_string();
            }
        }

        "generic".to_string()
    }

    /// Temporal-field detector. Mirrors `_detect_temporal_field` at
    /// `smart_crusher.py:1173-1209`.
    pub fn detect_temporal_field(
        &self,
        field_stats: &BTreeMap<String, FieldStats>,
        items: &[Value],
    ) -> bool {
        for (name, stats) in field_stats {
            match stats.field_type.as_str() {
                "string" => {
                    // First 10 values, str-typed only. Python: `items[:10]`.
                    let sample: Vec<&str> = items
                        .iter()
                        .take(10)
                        .filter_map(|i| i.as_object())
                        .filter_map(|o| o.get(name))
                        .filter_map(|v| v.as_str())
                        .collect();
                    if sample.is_empty() {
                        continue;
                    }
                    let iso_count = sample
                        .iter()
                        .filter(|s| is_iso_datetime(s) || is_iso_date(s))
                        .count();
                    if (iso_count as f64 / sample.len() as f64) > 0.5 {
                        return true;
                    }
                }
                "numeric" => {
                    if let (Some(mn), Some(_)) = (stats.min_val, stats.max_val) {
                        // Unix epoch range checks. Python uses `if min_val and max_val`
                        // which is falsy for 0; we mirror by checking `mn != 0` to
                        // match Python's behavior (very unlikely range edge but pinned).
                        let unix_seconds = (1_000_000_000.0..=2_000_000_000.0).contains(&mn);
                        let unix_millis = (1_000_000_000_000.0..=2_000_000_000_000.0).contains(&mn);
                        if unix_seconds || unix_millis {
                            return true;
                        }
                    }
                }
                _ => {}
            }
        }
        false
    }

    /// Crushability decision — the main "is it SAFE?" check. Mirrors
    /// `analyze_crushability` at `smart_crusher.py:1211-1430`.
    ///
    /// Returns a `CrushabilityAnalysis` with the verdict, confidence, and
    /// the signals that drove the decision. Callers consult `crushable`
    /// before invoking any actual compression.
    pub fn analyze_crushability(
        &self,
        items: &[Value],
        field_stats: &BTreeMap<String, FieldStats>,
    ) -> CrushabilityAnalysis {
        use super::outliers::{detect_error_items_for_preservation, detect_structural_outliers};

        let mut signals_present: Vec<String> = Vec::new();
        let mut signals_absent: Vec<String> = Vec::new();

        // 1. ID field detection — keep best (highest confidence) match.
        let mut id_field_name: Option<String> = None;
        let mut id_uniqueness: f64 = 0.0;
        let mut id_confidence: f64 = 0.0;
        for (name, stats) in field_stats {
            let values: Vec<Value> = items
                .iter()
                .filter_map(|i| i.as_object())
                .map(|o| o.get(name).cloned().unwrap_or(Value::Null))
                .collect();
            let (is_id, confidence) = detect_id_field_statistically(stats, &values);
            if is_id && confidence > id_confidence {
                id_field_name = Some(name.clone());
                id_uniqueness = stats.unique_ratio;
                id_confidence = confidence;
            }
        }
        let has_id_field = id_field_name.is_some() && id_confidence >= 0.7;

        // 2. Score field detection — short-circuit on first match.
        let mut has_score_field = false;
        for (name, stats) in field_stats {
            let (is_score, confidence) = detect_score_field_statistically(stats, items);
            if is_score {
                has_score_field = true;
                signals_present.push(format!("score_field:{}(conf={:.2})", name, confidence));
                break;
            }
        }
        if !has_score_field {
            signals_absent.push("score_field".to_string());
        }

        // 3. Structural outliers.
        let outlier_indices = detect_structural_outliers(items);
        let structural_outlier_count = outlier_indices.len();
        if structural_outlier_count > 0 {
            signals_present.push(format!("structural_outliers:{}", structural_outlier_count));
        } else {
            signals_absent.push("structural_outliers".to_string());
        }

        // 3b. Error-keyword fallback when no structural signal.
        let error_keyword_indices = detect_error_items_for_preservation(items, None);
        let keyword_error_count = error_keyword_indices.len();
        if keyword_error_count > 0 && structural_outlier_count == 0 {
            signals_present.push(format!("error_keywords:{}", keyword_error_count));
        }

        let error_count = structural_outlier_count.max(keyword_error_count);

        // 4. Numeric anomalies (>variance_threshold σ from mean).
        let mut anomaly_indices: BTreeSet<usize> = BTreeSet::new();
        for stats in field_stats.values() {
            if stats.field_type != "numeric" {
                continue;
            }
            let (Some(mean_val), Some(var)) = (stats.mean_val, stats.variance) else {
                continue;
            };
            if var <= 0.0 {
                continue;
            }
            let std = var.sqrt();
            if std <= 0.0 {
                continue;
            }
            let threshold = self.config.variance_threshold * std;
            for (i, item) in items.iter().enumerate() {
                let Some(obj) = item.as_object() else {
                    continue;
                };
                let Some(v) = obj.get(&stats.name) else {
                    continue;
                };
                if let Some(num) = v.as_f64() {
                    if !num.is_nan() && (num - mean_val).abs() > threshold {
                        anomaly_indices.insert(i);
                    }
                }
            }
        }
        let anomaly_count = anomaly_indices.len();
        if anomaly_count > 0 {
            signals_present.push(format!("anomalies:{}", anomaly_count));
        } else {
            signals_absent.push("anomalies".to_string());
        }

        // 5. Average string uniqueness, EXCLUDING the detected ID field.
        let id_name_ref = id_field_name.as_deref();
        let string_ratios: Vec<f64> = field_stats
            .values()
            .filter(|s| s.field_type == "string" && Some(s.name.as_str()) != id_name_ref)
            .map(|s| s.unique_ratio)
            .collect();
        let avg_string_uniqueness = if string_ratios.is_empty() {
            0.0
        } else {
            mean(&string_ratios).unwrap_or(0.0)
        };

        let non_id_numeric_ratios: Vec<f64> = field_stats
            .values()
            .filter(|s| s.field_type == "numeric" && Some(s.name.as_str()) != id_name_ref)
            .map(|s| s.unique_ratio)
            .collect();
        let avg_non_id_numeric_uniqueness = if non_id_numeric_ratios.is_empty() {
            0.0
        } else {
            mean(&non_id_numeric_ratios).unwrap_or(0.0)
        };

        let max_uniqueness = avg_string_uniqueness.max(id_uniqueness).max(0.0);
        let non_id_content_uniqueness = avg_string_uniqueness.max(avg_non_id_numeric_uniqueness);

        // 6. Change points.
        let has_change_points = field_stats
            .values()
            .filter(|s| s.field_type == "numeric")
            .any(|s| !s.change_points.is_empty());
        if has_change_points {
            signals_present.push("change_points".to_string());
        }

        let has_any_signal = !signals_present.is_empty();

        // Decision tree — order matters; mirrors Python case-by-case.
        let make = |crushable: bool,
                    confidence: f64,
                    reason: &str,
                    signals_present: Vec<String>,
                    signals_absent: Vec<String>|
         -> CrushabilityAnalysis {
            CrushabilityAnalysis {
                crushable,
                confidence,
                reason: reason.to_string(),
                signals_present,
                signals_absent,
                has_id_field,
                id_uniqueness,
                avg_string_uniqueness,
                has_score_field,
                error_item_count: error_count,
                anomaly_count,
            }
        };

        // Case 0: repetitive content with unique IDs.
        if non_id_content_uniqueness < 0.1 && has_id_field {
            let mut sp = signals_present.clone();
            sp.push("repetitive_content".to_string());
            return make(
                true,
                0.85,
                "repetitive_content_with_ids",
                sp,
                signals_absent,
            );
        }

        // Case 1: low uniqueness.
        if max_uniqueness < 0.3 {
            return make(
                true,
                0.9,
                "low_uniqueness_safe_to_sample",
                signals_present,
                signals_absent,
            );
        }

        // Case 2: high uniqueness + ID field + NO signal = DON'T CRUSH.
        if has_id_field && max_uniqueness > 0.8 && !has_any_signal {
            return make(
                false,
                0.85,
                "unique_entities_no_signal",
                signals_present,
                signals_absent,
            );
        }

        // Case 3: high uniqueness + has signal = crush.
        if max_uniqueness > 0.8 && has_any_signal {
            return make(
                true,
                0.7,
                "unique_entities_with_signal",
                signals_present,
                signals_absent,
            );
        }

        // Case 4: medium uniqueness + no signal = don't crush.
        if !has_any_signal {
            return make(
                false,
                0.6,
                "medium_uniqueness_no_signal",
                signals_present,
                signals_absent,
            );
        }

        // Case 5: medium uniqueness + has signal = crush with caution.
        make(
            true,
            0.5,
            "medium_uniqueness_with_signal",
            signals_present,
            signals_absent,
        )
    }

    /// Strategy selector. Mirrors `_select_strategy` at
    /// `smart_crusher.py:1432-1466`.
    pub fn select_strategy(
        &self,
        field_stats: &BTreeMap<String, FieldStats>,
        pattern: &str,
        item_count: usize,
        crushability: Option<&CrushabilityAnalysis>,
    ) -> CompressionStrategy {
        if item_count < self.config.min_items_to_analyze {
            return CompressionStrategy::None;
        }

        if let Some(c) = crushability {
            if !c.crushable {
                return CompressionStrategy::Skip;
            }
        }

        if pattern == "time_series" {
            let has_change_points = field_stats
                .values()
                .filter(|f| f.field_type == "numeric")
                .any(|f| !f.change_points.is_empty());
            if has_change_points {
                return CompressionStrategy::TimeSeries;
            }
        }

        if pattern == "logs" {
            // Python: `next((v for k, v in field_stats.items() if "message" in k.lower()), None)`
            // We mirror — first BTreeMap iteration order match wins. With
            // sorted iteration, this is deterministic.
            let message_field = field_stats
                .iter()
                .find(|(k, _)| k.to_lowercase().contains("message"))
                .map(|(_, v)| v);
            if let Some(mf) = message_field {
                if mf.unique_ratio < 0.5 {
                    return CompressionStrategy::ClusterSample;
                }
            }
        }

        if pattern == "search_results" {
            return CompressionStrategy::TopN;
        }

        CompressionStrategy::SmartSample
    }

    /// Reduction estimator. Mirrors `_estimate_reduction` at
    /// `smart_crusher.py:1468-1489`. Returns ∈ [0, 0.95].
    pub fn estimate_reduction(
        &self,
        field_stats: &BTreeMap<String, FieldStats>,
        strategy: CompressionStrategy,
        _item_count: usize,
    ) -> f64 {
        if strategy == CompressionStrategy::None {
            return 0.0;
        }

        // Python divides by `len(field_stats)` unconditionally. With an
        // empty stats map this raises ZeroDivisionError. We mirror by
        // returning 0.0 — analyze_array's empty-input guard prevents this
        // path from ever being reached in practice.
        if field_stats.is_empty() {
            return 0.0;
        }

        let constant_count = field_stats.values().filter(|v| v.is_constant).count();
        let constant_ratio = constant_count as f64 / field_stats.len() as f64;

        let base = match strategy {
            CompressionStrategy::TimeSeries => 0.7,
            CompressionStrategy::ClusterSample => 0.8,
            CompressionStrategy::TopN => 0.6,
            CompressionStrategy::SmartSample => 0.5,
            _ => 0.3,
        };

        (base + constant_ratio * 0.2).min(0.95)
    }
}

// ---------- helpers ----------

/// Python-equivalent `str(v)` for `serde_json::Value`. Used by
/// `_analyze_field`'s uniqueness count where Python does `[str(v) for v in
/// values]`. Python conventions:
/// - `None` → `"None"`
/// - `True`/`False` → `"True"`/`"False"`
/// - numbers → str-form of the number (no quotes, JSON-style is fine
///   because Python's `str(3.14)` matches `serde_json`'s for finite vals)
/// - strings → unquoted body
/// - dict/list → repr-form (matches Python's `str(dict)` since dicts use
///   `repr` for str). We approximate via JSON for parity-locked counts;
///   the only case this can drift is if a field carries nested dicts
///   with mixed types in its values, which is rare for crushable arrays.
///   Tracked as a Stage-3c.2 follow-up.
fn python_repr(v: &Value) -> String {
    match v {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => s.clone(),
        // Nested values aren't typically the unique-count drivers, so we
        // stringify with JSON. Used only for cardinality, not surfaced.
        _ => v.to_string(),
    }
}

/// Counter.most_common(n) equivalent. Returns up to `n` (value, count)
/// pairs sorted by count descending; ties broken by FIRST OCCURRENCE
/// order (mirrors Python's `Counter.most_common` via dict insertion
/// order + `heapq.nlargest`).
fn top_n_by_count(strs: &[&str], n: usize) -> Vec<(String, usize)> {
    use std::collections::HashMap;

    let mut order: Vec<&str> = Vec::new();
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for &s in strs {
        if !counts.contains_key(s) {
            order.push(s);
        }
        *counts.entry(s).or_insert(0) += 1;
    }

    // Stable sort by count desc preserves first-occurrence tie order.
    let mut pairs: Vec<(&&str, usize)> = order.iter().map(|k| (k, counts[k])).collect();
    pairs.sort_by_key(|b| std::cmp::Reverse(b.1));

    pairs
        .into_iter()
        .take(n)
        .map(|(k, c)| ((*k).to_string(), c))
        .collect()
}

// ISO 8601 patterns — pinned to Python's compiled regexes at
// `smart_crusher.py:96-97`:
//   `^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}`
//   `^\d{4}-\d{2}-\d{2}$`
// Implemented as direct char-position checks rather than full regex to
// avoid pulling in a regex compilation for every call site. Same
// behavior on the prefixes Python checks.
fn is_iso_datetime(s: &str) -> bool {
    let b = s.as_bytes();
    if b.len() < 19 {
        return false;
    }
    is_digit(b[0])
        && is_digit(b[1])
        && is_digit(b[2])
        && is_digit(b[3])
        && b[4] == b'-'
        && is_digit(b[5])
        && is_digit(b[6])
        && b[7] == b'-'
        && is_digit(b[8])
        && is_digit(b[9])
        && (b[10] == b'T' || b[10] == b' ')
        && is_digit(b[11])
        && is_digit(b[12])
        && b[13] == b':'
        && is_digit(b[14])
        && is_digit(b[15])
        && b[16] == b':'
        && is_digit(b[17])
        && is_digit(b[18])
}

fn is_iso_date(s: &str) -> bool {
    let b = s.as_bytes();
    if b.len() != 10 {
        return false;
    }
    is_digit(b[0])
        && is_digit(b[1])
        && is_digit(b[2])
        && is_digit(b[3])
        && b[4] == b'-'
        && is_digit(b[5])
        && is_digit(b[6])
        && b[7] == b'-'
        && is_digit(b[8])
        && is_digit(b[9])
}

#[inline]
fn is_digit(b: u8) -> bool {
    b.is_ascii_digit()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn analyzer() -> SmartAnalyzer {
        SmartAnalyzer::new(SmartCrusherConfig::default())
    }

    // ---------- analyze_array ----------

    #[test]
    fn empty_array_returns_none_strategy() {
        let a = analyzer().analyze_array(&[]);
        assert_eq!(a.item_count, 0);
        assert!(a.field_stats.is_empty());
        assert_eq!(a.detected_pattern, "generic");
        assert_eq!(a.recommended_strategy, CompressionStrategy::None);
        assert_eq!(a.estimated_reduction, 0.0);
        assert!(a.crushability.is_none());
    }

    #[test]
    fn non_dict_first_returns_none_strategy() {
        let items = vec![json!("hello"), json!("world")];
        let a = analyzer().analyze_array(&items);
        assert_eq!(a.item_count, 2);
        assert_eq!(a.recommended_strategy, CompressionStrategy::None);
    }

    #[test]
    fn small_array_below_threshold_returns_none() {
        // 4 items < min_items_to_analyze=5
        let items: Vec<Value> = (0..4).map(|i| json!({"id": i, "v": i})).collect();
        let a = analyzer().analyze_array(&items);
        assert_eq!(a.recommended_strategy, CompressionStrategy::None);
    }

    // ---------- analyze_field ----------

    #[test]
    fn analyze_field_all_null_yields_null_type_constant() {
        let items: Vec<Value> = (0..5).map(|_| json!({"x": null})).collect();
        let s = analyzer().analyze_field("x", &items);
        assert_eq!(s.field_type, "null");
        assert!(s.is_constant);
        assert_eq!(s.unique_count, 0);
        assert_eq!(s.count, 5);
    }

    #[test]
    fn analyze_field_numeric_basic_stats() {
        let items: Vec<Value> = (1..=10).map(|i| json!({"n": i})).collect();
        let s = analyzer().analyze_field("n", &items);
        assert_eq!(s.field_type, "numeric");
        assert_eq!(s.min_val, Some(1.0));
        assert_eq!(s.max_val, Some(10.0));
        assert_eq!(s.mean_val, Some(5.5));
        // Python: statistics.variance(1..=10) = 9.166666...
        let v = s.variance.expect("variance present");
        assert!((v - 9.166666666666666).abs() < 1e-9);
    }

    #[test]
    fn analyze_field_numeric_overflow_resets_all_stats_to_none() {
        // Python parity: when stats computation overflows, the
        // `try/except (OverflowError, ValueError)` block resets ALL
        // numeric fields to None. We mirror by checking finiteness across
        // the bundle and dropping the whole numeric stats group on
        // failure.
        let huge = 1e200;
        // Two extreme opposite values: variance overflows.
        let items = vec![json!({"n": huge}), json!({"n": -huge})];
        let s = analyzer().analyze_field("n", &items);
        assert_eq!(s.field_type, "numeric");
        // Per Python: min/max/mean reset to None; variance = 0 (int);
        // change_points empty.
        assert_eq!(s.min_val, None);
        assert_eq!(s.max_val, None);
        assert_eq!(s.mean_val, None);
        assert_eq!(s.variance, Some(0.0));
        assert!(s.change_points.is_empty());
        // Non-numeric stats (count, unique, is_constant) should still hold.
        assert_eq!(s.count, 2);
        assert_eq!(s.unique_count, 2);
    }

    #[test]
    fn analyze_field_numeric_filters_nan_and_inf() {
        // Tricky: serde_json doesn't allow NaN/Inf in JSON, so we build a
        // Number directly. Use `json!` with regular ints/floats only —
        // we just verify the finite-only path doesn't crash on a single
        // value (variance=0 then).
        let items: Vec<Value> = vec![json!({"n": 42.0}), json!({"n": 42.0})];
        let s = analyzer().analyze_field("n", &items);
        assert_eq!(s.variance, Some(0.0));
    }

    #[test]
    fn analyze_field_string_avg_length_and_top_values() {
        let items = vec![
            json!({"s": "ok"}),
            json!({"s": "ok"}),
            json!({"s": "warn"}),
            json!({"s": "fail"}),
            json!({"s": "ok"}),
        ];
        let s = analyzer().analyze_field("s", &items);
        assert_eq!(s.field_type, "string");
        // mean(2,2,4,4,2) = 2.8
        assert_eq!(s.avg_length, Some(2.8));
        // most_common: ok=3, warn=1, fail=1 (tie order: first-occurrence)
        assert_eq!(s.top_values[0], ("ok".to_string(), 3));
        assert_eq!(s.top_values[1].1, 1);
        assert_eq!(s.top_values[2].1, 1);
    }

    #[test]
    fn analyze_field_constant_detected() {
        let items: Vec<Value> = (0..10).map(|_| json!({"flag": true})).collect();
        let s = analyzer().analyze_field("flag", &items);
        assert!(s.is_constant);
        assert_eq!(s.constant_value, Some(json!(true)));
    }

    // ---------- detect_change_points ----------

    #[test]
    fn change_points_too_few_values_empty() {
        let cps = analyzer().detect_change_points(&[1.0, 2.0, 3.0], 5);
        assert!(cps.is_empty());
    }

    #[test]
    fn change_points_constant_values_empty() {
        // stdev=0 → early return.
        let cps = analyzer().detect_change_points(&[5.0; 20], 5);
        assert!(cps.is_empty());
    }

    #[test]
    fn change_points_step_function_detected() {
        // Three-segment: 30×0, 30×100, 30×0. Two transitions at i=30 and i=60.
        // For a pure two-segment step, diff = |b-a| ≈ 2σ exactly, so the
        // strict `> threshold` check would miss. Three segments let stdev
        // shrink relative to the step diff, so the boundary jumps clear it.
        let mut v: Vec<f64> = Vec::with_capacity(90);
        v.extend(vec![0.0; 30]);
        v.extend(vec![100.0; 30]);
        v.extend(vec![0.0; 30]);
        let cps = analyzer().detect_change_points(&v, 5);
        assert!(
            cps.contains(&30) || cps.contains(&60),
            "expected change point at i=30 or i=60, got {:?}",
            cps
        );
    }

    // ---------- detect_pattern ----------

    #[test]
    fn pattern_logs_message_and_level() {
        // 30 items, 2 distinct levels → unique_ratio = 2/30 ≈ 0.067 < 0.1 ✓.
        // Long unique messages → unique_ratio = 1.0 > 0.5 and avg_length > 20.
        let items: Vec<Value> = (0..30)
            .map(|i| {
                json!({
                    "msg": format!("Some long unique log message body text #{}", i),
                    "level": if i % 2 == 0 { "INFO" } else { "ERROR" },
                })
            })
            .collect();
        let mut field_stats: BTreeMap<String, FieldStats> = BTreeMap::new();
        let a = analyzer();
        for k in ["msg", "level"] {
            field_stats.insert(k.to_string(), a.analyze_field(k, &items));
        }
        let p = a.detect_pattern(&field_stats, &items);
        assert_eq!(p, "logs");
    }

    #[test]
    fn pattern_generic_when_nothing_matches() {
        let items: Vec<Value> = (0..10).map(|i| json!({"a": i, "b": i * 2})).collect();
        let mut fs: BTreeMap<String, FieldStats> = BTreeMap::new();
        let a = analyzer();
        for k in ["a", "b"] {
            fs.insert(k.to_string(), a.analyze_field(k, &items));
        }
        let p = a.detect_pattern(&fs, &items);
        // No timestamps, no logs shape, no obvious score → generic.
        assert_eq!(p, "generic");
    }

    // ---------- detect_temporal_field ----------

    #[test]
    fn temporal_iso_date() {
        let items: Vec<Value> = (1..=10)
            .map(|i| json!({"d": format!("2025-01-{:02}", i)}))
            .collect();
        let a = analyzer();
        let mut fs: BTreeMap<String, FieldStats> = BTreeMap::new();
        fs.insert("d".to_string(), a.analyze_field("d", &items));
        assert!(a.detect_temporal_field(&fs, &items));
    }

    #[test]
    fn temporal_iso_datetime() {
        let items: Vec<Value> = (1..=10)
            .map(|i| json!({"t": format!("2025-01-{:02}T12:00:00Z", i)}))
            .collect();
        let a = analyzer();
        let mut fs: BTreeMap<String, FieldStats> = BTreeMap::new();
        fs.insert("t".to_string(), a.analyze_field("t", &items));
        assert!(a.detect_temporal_field(&fs, &items));
    }

    #[test]
    fn temporal_unix_seconds_range() {
        // Timestamps in the 2024-2025 range.
        let items: Vec<Value> = (0..10)
            .map(|i| json!({"ts": 1_700_000_000_i64 + i * 86400}))
            .collect();
        let a = analyzer();
        let mut fs: BTreeMap<String, FieldStats> = BTreeMap::new();
        fs.insert("ts".to_string(), a.analyze_field("ts", &items));
        assert!(a.detect_temporal_field(&fs, &items));
    }

    #[test]
    fn temporal_normal_numbers_not_detected() {
        let items: Vec<Value> = (1..=10).map(|i| json!({"n": i})).collect();
        let a = analyzer();
        let mut fs: BTreeMap<String, FieldStats> = BTreeMap::new();
        fs.insert("n".to_string(), a.analyze_field("n", &items));
        assert!(!a.detect_temporal_field(&fs, &items));
    }

    // ---------- analyze_crushability ----------

    #[test]
    fn crushability_low_uniqueness_safe_to_sample() {
        // 30 items, all 'status':'ok' — high redundancy.
        let items: Vec<Value> = (0..30).map(|_| json!({"status": "ok"})).collect();
        let a = analyzer();
        let mut fs: BTreeMap<String, FieldStats> = BTreeMap::new();
        fs.insert("status".to_string(), a.analyze_field("status", &items));
        let c = a.analyze_crushability(&items, &fs);
        assert!(c.crushable);
        // Only "status" string field with unique_ratio=1/30=0.033 → max
        // uniqueness ≈ 0.033 < 0.3 → low_uniqueness path.
        assert_eq!(c.reason, "low_uniqueness_safe_to_sample");
    }

    #[test]
    fn crushability_unique_entities_no_signal_skips() {
        // Sequential IDs, distinct names, no errors, no change points.
        // Max uniqueness > 0.8, has_id_field=true, no signals → skip.
        let items: Vec<Value> = (0..20)
            .map(|i| json!({"id": i, "name": format!("user_{}", i)}))
            .collect();
        let a = analyzer();
        let mut fs: BTreeMap<String, FieldStats> = BTreeMap::new();
        for k in ["id", "name"] {
            fs.insert(k.to_string(), a.analyze_field(k, &items));
        }
        let c = a.analyze_crushability(&items, &fs);
        assert!(!c.crushable);
        assert_eq!(c.reason, "unique_entities_no_signal");
    }

    #[test]
    fn crushability_repetitive_content_with_ids_crushes() {
        // Unique ID + constant content field → repetitive_content path.
        let items: Vec<Value> = (0..20).map(|i| json!({"id": i, "status": "ok"})).collect();
        let a = analyzer();
        let mut fs: BTreeMap<String, FieldStats> = BTreeMap::new();
        for k in ["id", "status"] {
            fs.insert(k.to_string(), a.analyze_field(k, &items));
        }
        let c = a.analyze_crushability(&items, &fs);
        assert!(c.crushable);
        assert_eq!(c.reason, "repetitive_content_with_ids");
    }

    // ---------- select_strategy ----------

    #[test]
    fn select_strategy_below_min_returns_none() {
        let fs = BTreeMap::new();
        let s = analyzer().select_strategy(&fs, "generic", 3, None);
        assert_eq!(s, CompressionStrategy::None);
    }

    #[test]
    fn select_strategy_skip_when_not_crushable() {
        let fs = BTreeMap::new();
        let crush = CrushabilityAnalysis::skip("nope", 0.9);
        let s = analyzer().select_strategy(&fs, "generic", 100, Some(&crush));
        assert_eq!(s, CompressionStrategy::Skip);
    }

    #[test]
    fn select_strategy_search_results_returns_top_n() {
        let fs = BTreeMap::new();
        let s = analyzer().select_strategy(&fs, "search_results", 100, None);
        assert_eq!(s, CompressionStrategy::TopN);
    }

    #[test]
    fn select_strategy_generic_returns_smart_sample() {
        let fs = BTreeMap::new();
        let s = analyzer().select_strategy(&fs, "generic", 100, None);
        assert_eq!(s, CompressionStrategy::SmartSample);
    }

    // ---------- estimate_reduction ----------

    #[test]
    fn estimate_reduction_none_returns_zero() {
        let fs = BTreeMap::new();
        let r = analyzer().estimate_reduction(&fs, CompressionStrategy::None, 100);
        assert_eq!(r, 0.0);
    }

    #[test]
    fn estimate_reduction_caps_at_0_95() {
        // All-constant field stats → constant_ratio=1.0 → base+0.2 = 1.0,
        // capped at 0.95.
        let mut fs: BTreeMap<String, FieldStats> = BTreeMap::new();
        for k in ["a", "b"] {
            fs.insert(
                k.to_string(),
                FieldStats {
                    name: k.to_string(),
                    field_type: "string".to_string(),
                    count: 10,
                    unique_count: 1,
                    unique_ratio: 0.1,
                    is_constant: true,
                    constant_value: Some(json!("v")),
                    min_val: None,
                    max_val: None,
                    mean_val: None,
                    variance: None,
                    change_points: Vec::new(),
                    avg_length: None,
                    top_values: Vec::new(),
                },
            );
        }
        let r = analyzer().estimate_reduction(&fs, CompressionStrategy::ClusterSample, 10);
        assert_eq!(r, 0.95);
    }

    #[test]
    fn estimate_reduction_smart_sample_no_constants() {
        let mut fs: BTreeMap<String, FieldStats> = BTreeMap::new();
        fs.insert(
            "id".to_string(),
            FieldStats {
                name: "id".to_string(),
                field_type: "numeric".to_string(),
                count: 100,
                unique_count: 100,
                unique_ratio: 1.0,
                is_constant: false,
                constant_value: None,
                min_val: Some(0.0),
                max_val: Some(99.0),
                mean_val: Some(49.5),
                variance: Some(841.66),
                change_points: Vec::new(),
                avg_length: None,
                top_values: Vec::new(),
            },
        );
        let r = analyzer().estimate_reduction(&fs, CompressionStrategy::SmartSample, 100);
        // base 0.5 + constant_ratio 0 * 0.2 = 0.5
        assert_eq!(r, 0.5);
    }

    // ---------- helpers ----------

    #[test]
    fn iso_datetime_pattern_matches() {
        assert!(is_iso_datetime("2025-01-15T12:00:00"));
        assert!(is_iso_datetime("2025-01-15 12:00:00"));
        assert!(is_iso_datetime("2025-01-15T12:00:00.123Z"));
        assert!(!is_iso_datetime("2025-01-15"));
        assert!(!is_iso_datetime("not a date"));
    }

    #[test]
    fn iso_date_pattern_matches() {
        assert!(is_iso_date("2025-01-15"));
        assert!(!is_iso_date("2025-01-15T12:00:00"));
        assert!(!is_iso_date("2025/01/15"));
    }

    #[test]
    fn python_repr_basics() {
        assert_eq!(python_repr(&Value::Null), "None");
        assert_eq!(python_repr(&json!(true)), "True");
        assert_eq!(python_repr(&json!(false)), "False");
        assert_eq!(python_repr(&json!(42)), "42");
        assert_eq!(python_repr(&json!("hello")), "hello");
    }

    #[test]
    fn top_n_first_occurrence_tie_break() {
        // a appears first, b second, both count 2.
        let strs = vec!["a", "b", "a", "b", "c"];
        let top = top_n_by_count(&strs, 5);
        assert_eq!(top[0].0, "a");
        assert_eq!(top[1].0, "b");
        assert_eq!(top[2].0, "c");
    }
}
