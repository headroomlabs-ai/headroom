//! Statistical detectors for ID-like and score-like fields.
//!
//! Direct ports of `_detect_id_field_statistically` and
//! `_detect_score_field_statistically` from `smart_crusher.py:484-603`.
//!
//! These run *after* per-field statistics are computed and consume a
//! `FieldStats` plus the raw values. They're called by the analyzer's
//! crushability logic to decide whether a field carries a meaningful
//! ranking signal (score) or is just a unique identifier (ID) that
//! shouldn't drive compression decisions.

use serde_json::Value;

use super::statistics::{calculate_string_entropy, detect_sequential_pattern, is_uuid_format};
use super::types::FieldStats;

/// Detect whether a field is an "ID field" — high-uniqueness column
/// that doesn't carry semantic information.
///
/// Direct port of `_detect_id_field_statistically` (Python
/// `smart_crusher.py:484-530`). Returns `(is_id, confidence)` where
/// confidence ∈ [0.0, 1.0]. The caller uses this to decide whether the
/// field is a strong enough signal to drive crushability analysis.
///
/// # Detection rules (mirroring Python step-by-step)
///
/// 1. Hard gate: `unique_ratio < 0.9` → not an ID field.
/// 2. String fields:
///    - >80% of first-20 sample values look like UUIDs → confidence 0.95.
///    - Average entropy >0.7 AND `unique_ratio > 0.95` → confidence 0.8.
/// 3. Numeric fields:
///    - Sequential pattern (via `detect_sequential_pattern`) AND
///      `unique_ratio > 0.95` → confidence 0.9.
///    - Has a value range AND `unique_ratio > 0.95` → confidence 0.85.
/// 4. Catch-all: very high uniqueness (`> 0.98`) → confidence 0.7.
pub fn detect_id_field_statistically(stats: &FieldStats, values: &[Value]) -> (bool, f64) {
    // Hard gate matching Python line 494.
    if stats.unique_ratio < 0.9 {
        return (false, 0.0);
    }

    // String-field branches.
    if stats.field_type == "string" {
        // First 20 string-typed values for sampling. Python: `values[:20]`
        // then filters by `isinstance(v, str)` — order-preserving slice
        // before filter, so we mirror that.
        let sample_values: Vec<&str> = values.iter().take(20).filter_map(|v| v.as_str()).collect();

        if !sample_values.is_empty() {
            let uuid_count = sample_values.iter().filter(|s| is_uuid_format(s)).count();
            // Python: `uuid_count / len(sample_values) > 0.8`.
            if (uuid_count as f64 / sample_values.len() as f64) > 0.8 {
                return (true, 0.95);
            }

            // Python: average entropy across the sample.
            let avg_entropy = sample_values
                .iter()
                .map(|s| calculate_string_entropy(s))
                .sum::<f64>()
                / sample_values.len() as f64;
            if avg_entropy > 0.7 && stats.unique_ratio > 0.95 {
                return (true, 0.8);
            }
        }
    }

    // Numeric-field branches.
    if stats.field_type == "numeric" {
        // Python: passes `values` (full list, may include strings) through
        // `_detect_sequential_pattern` with default `check_order=True`.
        if detect_sequential_pattern(values, true) && stats.unique_ratio > 0.95 {
            return (true, 0.9);
        }

        // High-uniqueness numeric with non-trivial range — likely an ID
        // even without sequential structure (e.g., random ints in a wide
        // band).
        if let (Some(min_v), Some(max_v)) = (stats.min_val, stats.max_val) {
            let value_range = max_v - min_v;
            if value_range > 0.0 && stats.unique_ratio > 0.95 {
                return (true, 0.85);
            }
        }
    }

    // Catch-all: very high uniqueness alone is a signal.
    if stats.unique_ratio > 0.98 {
        return (true, 0.7);
    }

    (false, 0.0)
}

/// Detect whether a field is a "score field" — bounded-range numeric
/// where higher values mean "more relevant".
///
/// Direct port of `_detect_score_field_statistically` (Python
/// `smart_crusher.py:533-603`). Returns `(is_score, confidence)`.
///
/// # Detection rules (mirroring Python)
///
/// 1. Field must be numeric AND have both `min_val` and `max_val`.
/// 2. Range must match a "common score range":
///    - `[0, 1]` (most common ML score range) → +0.4
///    - `[0, 10]` → +0.3
///    - `[0, 100]` → +0.25
///    - `[-1, 1]` (signed similarity) → +0.35
/// 3. Must NOT be a sequential pattern (IDs are sequential; scores aren't).
/// 4. If first-50 values appear sorted descending (>70% of pairs) → +0.3.
/// 5. If >30% of first-20 are non-integer floats → +0.1.
/// 6. Returns `(confidence >= 0.4, min(confidence, 0.95))`.
///
/// `items` is the list of original-array dict items so we can pull the
/// field's values in array order for the descending-sort check.
pub fn detect_score_field_statistically(stats: &FieldStats, items: &[Value]) -> (bool, f64) {
    if stats.field_type != "numeric" {
        return (false, 0.0);
    }

    let (min_val, max_val) = match (stats.min_val, stats.max_val) {
        (Some(min_v), Some(max_v)) => (min_v, max_v),
        _ => return (false, 0.0),
    };

    let mut confidence: f64 = 0.0;

    // Range check (Python lines 555-568). The conditions are arranged
    // exactly as in Python — `if/elif` chain, first match wins.
    let is_bounded = if (0.0..=1.0).contains(&min_val) && (0.0..=1.0).contains(&max_val) {
        confidence += 0.4;
        true
    } else if (0.0..=10.0).contains(&min_val) && (0.0..=10.0).contains(&max_val) {
        confidence += 0.3;
        true
    } else if (0.0..=100.0).contains(&min_val) && (0.0..=100.0).contains(&max_val) {
        confidence += 0.25;
        true
    } else if min_val >= -1.0 && max_val <= 1.0 {
        // Python: `elif -1 <= min_val and max_val <= 1`. Note Python's
        // chained `<=` only on max_val side; min_val is checked
        // separately. Pinned exactly.
        confidence += 0.35;
        true
    } else {
        false
    };

    if !is_bounded {
        return (false, 0.0);
    }

    // Pull this field's values from the FIRST 50 items, dict-style.
    // Python: `[item.get(stats.name) for item in items[:50] if stats.name in item]`.
    let sample_values: Vec<&Value> = items
        .iter()
        .take(50)
        .filter_map(|item| item.as_object().and_then(|m| m.get(&stats.name)))
        .collect();

    // Sequential check — IDs are sequential, scores aren't. We need
    // owned `Value`s for `detect_sequential_pattern`; clone the sample.
    let sample_owned: Vec<Value> = sample_values.iter().map(|v| (*v).clone()).collect();
    if detect_sequential_pattern(&sample_owned, true) {
        return (false, 0.0);
    }

    // Descending-sort check on the full items list (Python line 580-593).
    // Filter to finite-numeric values, preserving array order.
    let values_in_order: Vec<f64> = items
        .iter()
        .filter_map(|item| item.as_object().and_then(|m| m.get(&stats.name)))
        .filter_map(|v| v.as_f64())
        .filter(|f| f.is_finite())
        .collect();

    if values_in_order.len() >= 5 {
        let num_pairs = values_in_order.len() - 1;
        let descending_count = values_in_order.windows(2).filter(|w| w[0] >= w[1]).count();
        if num_pairs > 0 && (descending_count as f64 / num_pairs as f64) > 0.7 {
            confidence += 0.3;
        }
    }

    // Float-fraction check on first 20 ordered values (Python line 597-601).
    // "Not equal to its int-truncation" ≈ has a fractional part.
    let first_20: &[f64] = if values_in_order.len() > 20 {
        &values_in_order[..20]
    } else {
        &values_in_order[..]
    };
    let float_count = first_20
        .iter()
        .filter(|&&v| v.is_finite() && v != v.trunc())
        .count();
    if !first_20.is_empty() && (float_count as f64) > (first_20.len() as f64 * 0.3) {
        confidence += 0.1;
    }

    let is_score = confidence >= 0.4;
    let bounded_confidence = confidence.min(0.95);
    (is_score, bounded_confidence)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn stats(name: &str, field_type: &str, unique_ratio: f64) -> FieldStats {
        FieldStats {
            name: name.to_string(),
            field_type: field_type.to_string(),
            count: 100,
            unique_count: (100.0 * unique_ratio) as usize,
            unique_ratio,
            is_constant: false,
            constant_value: None,
            min_val: None,
            max_val: None,
            mean_val: None,
            variance: None,
            change_points: Vec::new(),
            avg_length: None,
            top_values: Vec::new(),
        }
    }

    fn stats_with_range(name: &str, min_v: f64, max_v: f64) -> FieldStats {
        let mut s = stats(name, "numeric", 1.0);
        s.min_val = Some(min_v);
        s.max_val = Some(max_v);
        s
    }

    // ---------- detect_id_field_statistically ----------

    #[test]
    fn id_field_low_uniqueness_rejected() {
        let s = stats("status", "string", 0.5);
        let values: Vec<Value> = vec![json!("ok"), json!("error"), json!("ok"), json!("ok")];
        assert_eq!(detect_id_field_statistically(&s, &values), (false, 0.0));
    }

    #[test]
    fn id_field_uuid_strings_high_confidence() {
        let s = stats("uid", "string", 1.0);
        let values: Vec<Value> = (0..20)
            .map(|i| json!(format!("550e8400-e29b-41d4-a716-{:012x}", i)))
            .collect();
        let (is_id, conf) = detect_id_field_statistically(&s, &values);
        assert!(is_id);
        assert_eq!(conf, 0.95);
    }

    #[test]
    fn id_field_high_entropy_strings() {
        let mut s = stats("uid", "string", 1.0);
        s.unique_ratio = 0.96;
        // 20 random-looking hex-ish strings — high entropy.
        let values: Vec<Value> = (0..20)
            .map(|i| json!(format!("a3f7b2c{:06x}d8e1f4a7", i)))
            .collect();
        let (is_id, conf) = detect_id_field_statistically(&s, &values);
        assert!(is_id);
        assert!((conf - 0.8).abs() < 1e-9);
    }

    #[test]
    fn id_field_sequential_numeric() {
        let mut s = stats("id", "numeric", 1.0);
        s.unique_ratio = 0.96;
        s.min_val = Some(1.0);
        s.max_val = Some(100.0);
        let values: Vec<Value> = (1..=100).map(|i| json!(i)).collect();
        let (is_id, conf) = detect_id_field_statistically(&s, &values);
        assert!(is_id);
        assert!((conf - 0.9).abs() < 1e-9);
    }

    #[test]
    fn id_field_high_uniqueness_alone_triggers_catchall() {
        // Numeric field with unique_ratio = 0.99 but no other signal.
        // Falls through to the 0.98 catch-all.
        let mut s = stats("misc", "numeric", 0.99);
        s.min_val = Some(0.0);
        s.max_val = Some(0.0); // zero range → numeric range branch fails
        let values: Vec<Value> = (0..100).map(|_| json!(0)).collect();
        let (is_id, conf) = detect_id_field_statistically(&s, &values);
        assert!(is_id);
        assert!((conf - 0.7).abs() < 1e-9);
    }

    // ---------- detect_score_field_statistically ----------

    #[test]
    fn score_field_unit_range_with_descending_sort() {
        // Range [0, 1] (+0.4) + descending sort (+0.3) = 0.7, which
        // exceeds the 0.4 threshold and is below the 0.95 cap.
        let s = stats_with_range("score", 0.0, 1.0);
        let items: Vec<Value> = (0..10)
            .rev()
            .map(|i| json!({"score": (i as f64) / 10.0}))
            .collect();
        let (is_score, conf) = detect_score_field_statistically(&s, &items);
        assert!(is_score);
        assert!(conf >= 0.7);
        assert!(conf <= 0.95);
    }

    #[test]
    fn score_field_sequential_rejected() {
        // Bounded but sequential — must NOT be a score.
        let s = stats_with_range("score", 1.0, 10.0);
        let items: Vec<Value> = (1..=10).map(|i| json!({"score": i})).collect();
        let (is_score, _) = detect_score_field_statistically(&s, &items);
        assert!(!is_score);
    }

    #[test]
    fn score_field_unbounded_range_rejected() {
        // Range [0, 1000] doesn't match any of the bounded-range cases.
        let s = stats_with_range("metric", 0.0, 1000.0);
        let items: Vec<Value> = (0..10).map(|i| json!({"metric": i * 100})).collect();
        let (is_score, _) = detect_score_field_statistically(&s, &items);
        assert!(!is_score);
    }

    #[test]
    fn score_field_signed_similarity_range() {
        // Range [-1, 1] is the cosine-similarity bucket (+0.35).
        let s = stats_with_range("similarity", -0.9, 0.95);
        let items: Vec<Value> = (0..10)
            .rev()
            .map(|i| json!({"similarity": (i as f64) / 10.0 - 0.5}))
            .collect();
        let (is_score, _) = detect_score_field_statistically(&s, &items);
        assert!(is_score);
    }

    #[test]
    fn score_field_below_threshold_rejected() {
        // Range [0, 100] alone (+0.25) is below the 0.4 threshold.
        // No descending sort, no float fraction → stays below 0.4.
        let s = stats_with_range("metric", 0.0, 100.0);
        // Random unsorted ints — no descending hint.
        let items: Vec<Value> = vec![
            json!({"metric": 50}),
            json!({"metric": 10}),
            json!({"metric": 80}),
            json!({"metric": 20}),
            json!({"metric": 90}),
        ];
        let (is_score, _) = detect_score_field_statistically(&s, &items);
        assert!(!is_score);
    }

    #[test]
    fn score_field_non_numeric_rejected() {
        let s = stats("name", "string", 0.5);
        let items: Vec<Value> = vec![
            json!({"name": "alice"}),
            json!({"name": "bob"}),
            json!({"name": "alice"}),
        ];
        let (is_score, _) = detect_score_field_statistically(&s, &items);
        assert!(!is_score);
    }

    #[test]
    fn score_field_missing_min_max_rejected() {
        let s = stats("score", "numeric", 1.0); // no min/max set
        let items: Vec<Value> = vec![json!({"score": 0.5})];
        let (is_score, _) = detect_score_field_statistically(&s, &items);
        assert!(!is_score);
    }

    #[test]
    fn score_field_confidence_capped_at_95() {
        // Range [0, 1] (+0.4) + descending sort (+0.3) + float fraction
        // (+0.1) = 0.8. Below cap. To exceed 0.95 we'd need >0.55 from
        // bonuses, which isn't possible with current rules. Pin the cap
        // anyway by constructing the highest-scoring case and confirming
        // it doesn't go above 0.95.
        let s = stats_with_range("score", 0.0, 1.0);
        let items: Vec<Value> = (0..50)
            .rev()
            .map(|i| json!({"score": (i as f64) / 50.0}))
            .collect();
        let (_, conf) = detect_score_field_statistically(&s, &items);
        assert!(conf <= 0.95);
    }
}
