//! Outlier detectors used to mark items as "must preserve" during compression.
//!
//! Direct port of `_detect_structural_outliers`, `_detect_rare_status_values`,
//! and `_detect_error_items_for_preservation` from
//! `smart_crusher.py:606-748`.
//!
//! `_detect_items_by_learned_semantics` is deferred to a later commit
//! because it depends on the TOIN `FieldSemantics` type, which isn't
//! ported yet.
//!
//! # Bug #3 fix — `detect_rare_status_values`
//!
//! Python's original guard at `smart_crusher.py:674`
//! `if not (2 <= len(unique_values) <= 10): continue`
//! caps cardinality at 10, so error-code domains with 50+ codes are
//! skipped entirely — even when one or two codes appear at <1% rates
//! and clearly deserve outlier flagging.
//!
//! The fix replaces the cap-and-dominance approach with a Pareto check:
//!
//! 1. Cardinality cap raised to **50** (above which the field is
//!    almost certainly an ID/free-form column, not a status enum).
//! 2. Sort value frequencies descending. Find the smallest K such
//!    that the top-K values cover ≥80% of items.
//! 3. If `K ≤ 5`, the remaining values are "rare" and items
//!    containing them are outliers.
//!
//! This unifies both cases the original algorithm partially handled:
//!
//! - **Low cardinality + dominant**: 95×"ok" + 5 errors → top-1 covers
//!   95% → 4 rare values flagged. Same as before.
//! - **Higher cardinality + bimodal**: 60×"info" + 25×"warn" + 15
//!   distinct rare errors → top-2 covers 85% → 15 rare values flagged.
//!   New, correct, and was missed entirely by the old code.
//! - **Uniform distribution**: 50 distinct values, 2 each → top-K
//!   never reaches 80% with K ≤ 5 → skip. Correctly identifies as
//!   non-categorical.
//!
//! The same fix lands in `headroom/transforms/smart_crusher.py` later
//! in this PR so the parity fixtures continue to byte-match.

use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashSet};

use super::error_keywords::ERROR_KEYWORDS;

/// Detect items that are structural outliers (error-like or
/// uncommonly-shaped).
///
/// Direct port of `_detect_structural_outliers` (Python
/// `smart_crusher.py:606-650`). Returns deduplicated, ascending-sorted
/// indices.
///
/// # Detection
///
/// 1. **Rare-field outliers**: items containing a field that appears
///    in <20% of the array.
/// 2. **Rare-status outliers**: forwarded to `detect_rare_status_values`,
///    which finds items with statistically rare categorical values
///    (with bug #3 fixed).
pub fn detect_structural_outliers(items: &[Value]) -> Vec<usize> {
    if items.len() < 5 {
        return Vec::new();
    }

    // Field counts across the whole array.
    let mut field_counts: BTreeMap<&str, usize> = BTreeMap::new();
    for item in items {
        if let Some(obj) = item.as_object() {
            for key in obj.keys() {
                *field_counts.entry(key.as_str()).or_insert(0) += 1;
            }
        }
    }

    let n = items.len();
    let common_fields: HashSet<String> = field_counts
        .iter()
        .filter(|(_, &c)| c as f64 >= n as f64 * 0.8)
        .map(|(k, _)| (*k).to_string())
        .collect();
    let rare_fields: HashSet<&str> = field_counts
        .iter()
        .filter(|(_, &c)| (c as f64) < n as f64 * 0.2)
        .map(|(k, _)| *k)
        .collect();

    // Use a BTreeSet for stable order — Python uses `set()` then `list(set(...))`
    // which is non-deterministic order; we pin to ascending for parity tests.
    let mut outlier_set: BTreeSet<usize> = BTreeSet::new();

    // 1. Rare-field outliers.
    for (i, item) in items.iter().enumerate() {
        let Some(obj) = item.as_object() else {
            continue;
        };
        let has_rare = obj.keys().any(|k| rare_fields.contains(k.as_str()));
        if has_rare {
            outlier_set.insert(i);
        }
    }

    // 2. Rare-status outliers.
    for idx in detect_rare_status_values(items, &common_fields) {
        outlier_set.insert(idx);
    }

    outlier_set.into_iter().collect()
}

/// Detect items with rare values in status-like categorical fields.
///
/// **Bug #3 fix** — see module-level doc. Algorithm:
///
/// 1. Cardinality 2..=50 (was 2..=10 in Python).
/// 2. Pareto check: top-K values covering ≥80% with `K ≤ 5`.
/// 3. Items NOT in top-K → outliers.
///
/// Returns indices in the order they were discovered (mirrors
/// Python's append-order behavior; downstream `detect_structural_outliers`
/// dedupes via BTreeSet).
pub fn detect_rare_status_values(items: &[Value], common_fields: &HashSet<String>) -> Vec<usize> {
    let mut outlier_indices: Vec<usize> = Vec::new();

    // Iterate fields in sorted order for determinism. Python iterates
    // a `set`, which has non-deterministic order — but the eventual
    // output is deduped via the caller's set, so order here only
    // affects which fields drive detection if multiple status-like
    // fields exist. Sorting gives us a stable, fixture-friendly order.
    let mut sorted_fields: Vec<&String> = common_fields.iter().collect();
    sorted_fields.sort();

    for field_name in sorted_fields {
        // Collect this field's values across all items, mirroring Python:
        //   `[item.get(field_name) for item in items if isinstance(item, dict) and field_name in item]`
        let values: Vec<&Value> = items
            .iter()
            .filter_map(|item| item.as_object())
            .filter_map(|m| m.get(field_name))
            .collect();

        // Stringify non-null values and dedupe to get cardinality. Python:
        //   `unique_values = {str(v) for v in values if v is not None}`
        // We use `python_repr_value`-equivalent stringification: simple
        // scalars use their natural form; nested values use serde_json
        // serialization. This stringification is only used for set-
        // dedup and frequency counting, not surfaced to callers, so the
        // python_repr-vs-json distinction we made for anchors doesn't
        // matter here — the SAME stringification is used for both the
        // "is this rare" computation and the per-item lookup, so the
        // surface is internally consistent.
        let stringify = |v: &Value| -> String {
            match v {
                Value::Null => unreachable!("null filtered above"),
                Value::Bool(b) => b.to_string(),
                Value::Number(n) => n.to_string(),
                Value::String(s) => s.clone(),
                _ => v.to_string(),
            }
        };

        let unique_values: BTreeSet<String> = values
            .iter()
            .filter(|v| !matches!(v, Value::Null))
            .map(|v| stringify(v))
            .collect();

        // Cardinality cap (BUG #3 FIX: was 2..=10, now 2..=50).
        if !(2..=50).contains(&unique_values.len()) {
            continue;
        }

        // Frequency count. Python: `value_counts: dict[str, int]`,
        // `key = str(v) if v is not None else "__none__"`.
        let mut value_counts: BTreeMap<String, usize> = BTreeMap::new();
        for v in &values {
            let key = if matches!(v, Value::Null) {
                "__none__".to_string()
            } else {
                stringify(v)
            };
            *value_counts.entry(key).or_insert(0) += 1;
        }
        if value_counts.is_empty() {
            continue;
        }

        let total = values.len();

        // Pareto check (BUG #3 FIX): find smallest K such that top-K
        // values cover ≥80% of items.
        let mut sorted_counts: Vec<(&String, &usize)> = value_counts.iter().collect();
        // Sort by count descending; tiebreak by key ascending so the
        // result is deterministic when multiple values have the same
        // frequency.
        sorted_counts.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));

        let threshold = (total as f64 * 0.8).ceil() as usize;
        let mut cumulative: usize = 0;
        let mut top_k_values: HashSet<String> = HashSet::new();
        for (value, count) in &sorted_counts {
            cumulative += **count;
            top_k_values.insert((*value).clone());
            if cumulative >= threshold {
                break;
            }
        }

        // Only flag rare values if the top-K is small (≤5). Above this
        // the distribution is too uniform to label any value "rare".
        if top_k_values.len() > 5 {
            continue;
        }

        // Items with values NOT in top_k_values are outliers.
        for (i, item) in items.iter().enumerate() {
            let Some(obj) = item.as_object() else {
                continue;
            };
            let Some(field_value) = obj.get(field_name) else {
                continue;
            };
            let item_value = if matches!(field_value, Value::Null) {
                "__none__".to_string()
            } else {
                stringify(field_value)
            };
            if !top_k_values.contains(&item_value) {
                outlier_indices.push(i);
            }
        }
    }

    outlier_indices
}

/// Detect items containing error keywords for PRESERVATION.
///
/// Direct port of `_detect_error_items_for_preservation` (Python
/// `smart_crusher.py:711-748`). Used by the orchestrator's
/// `_prioritize_indices` to ensure error items are NEVER dropped.
///
/// # Args
///
/// - `items`: array items to scan.
/// - `item_strings`: pre-computed JSON serializations to avoid
///   redundant `to_string` work. Pass `None` to serialize on the fly.
///   When provided, must be the same length as `items` (Python's
///   bounds-check via `i < len(item_strings)` is mirrored).
pub fn detect_error_items_for_preservation(
    items: &[Value],
    item_strings: Option<&[String]>,
) -> Vec<usize> {
    let mut error_indices: Vec<usize> = Vec::new();

    for (i, item) in items.iter().enumerate() {
        // Python: `if not isinstance(item, dict): continue`. Mirror.
        if !item.is_object() {
            continue;
        }

        // Reuse cached serialization or serialize fresh. Python:
        //   `if item_strings is not None and i < len(item_strings):
        //        item_str = item_strings[i].lower()
        //    else:
        //        item_str = json.dumps(item).lower()`
        let serialized: String = match item_strings {
            Some(arr) if i < arr.len() => arr[i].to_lowercase(),
            _ => match serde_json::to_string(item) {
                Ok(s) => s.to_lowercase(),
                Err(_) => continue,
            },
        };

        // Python: `for keyword in _ERROR_KEYWORDS_FOR_PRESERVATION:
        //              if keyword in item_str: ...; break`
        if ERROR_KEYWORDS.iter().any(|kw| serialized.contains(kw)) {
            error_indices.push(i);
        }
    }

    error_indices
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ---------- detect_structural_outliers ----------

    #[test]
    fn outliers_too_few_items_returns_empty() {
        let items: Vec<Value> = (0..3).map(|i| json!({"a": i})).collect();
        assert_eq!(detect_structural_outliers(&items), Vec::<usize>::new());
    }

    #[test]
    fn outliers_rare_field_flags_item() {
        // 9 items with `{"a"}`, 1 item with extra `{"a", "x"}` — `x`
        // appears in 10% of items, below the 20% rare-field threshold.
        let mut items: Vec<Value> = (0..9).map(|i| json!({"a": i})).collect();
        items.push(json!({"a": 9, "x": "rare"}));
        let outliers = detect_structural_outliers(&items);
        assert!(outliers.contains(&9));
    }

    #[test]
    fn outliers_no_dict_items_silently_skipped() {
        // Mixed array: only dict items count. A non-dict in the middle
        // doesn't crash anything.
        let items: Vec<Value> = vec![
            json!({"status": "ok"}),
            json!("string-not-dict"),
            json!({"status": "ok"}),
            json!({"status": "ok"}),
            json!({"status": "ok"}),
        ];
        // Should not panic.
        let _ = detect_structural_outliers(&items);
    }

    // ---------- detect_rare_status_values (BUG #3 FIX coverage) ----------

    #[test]
    fn rare_status_low_cardinality_dominant_value() {
        // Pre-bug-fix path still works: 95×ok + 5 errors.
        let mut items: Vec<Value> = (0..95).map(|_| json!({"status": "ok"})).collect();
        items.push(json!({"status": "error"}));
        items.push(json!({"status": "timeout"}));
        items.push(json!({"status": "error"}));
        items.push(json!({"status": "timeout"}));
        items.push(json!({"status": "fail"}));
        let common: HashSet<String> = ["status".to_string()].into_iter().collect();
        let outliers = detect_rare_status_values(&items, &common);
        // All 5 non-"ok" items flagged.
        assert_eq!(outliers.len(), 5);
    }

    #[test]
    fn rare_status_bug3_fix_high_cardinality_bimodal() {
        // BUG #3 case: cardinality 17 (1 dominant + 1 second + 15
        // singletons). Old code: 17 > 10 → skip. New code: top-2 covers
        // 85%, K=2 ≤ 5, remaining 15 values flagged.
        let mut items: Vec<Value> = Vec::new();
        for _ in 0..60 {
            items.push(json!({"code": "INFO"}));
        }
        for _ in 0..25 {
            items.push(json!({"code": "WARN"}));
        }
        for i in 0..15 {
            items.push(json!({"code": format!("ERR_{}", i)}));
        }
        let common: HashSet<String> = ["code".to_string()].into_iter().collect();
        let outliers = detect_rare_status_values(&items, &common);
        // 15 rare-error items flagged. Pre-fix this would be 0.
        assert_eq!(outliers.len(), 15);
    }

    #[test]
    fn rare_status_uniform_distribution_no_outliers() {
        // 50 items, 50 distinct values, 1 each. Top-K never reaches
        // 80% with K ≤ 5 → no outliers (correctly identified as
        // non-categorical).
        let items: Vec<Value> = (0..50)
            .map(|i| json!({"code": format!("CAT_{}", i)}))
            .collect();
        let common: HashSet<String> = ["code".to_string()].into_iter().collect();
        let outliers = detect_rare_status_values(&items, &common);
        assert!(
            outliers.is_empty(),
            "uniform distribution must not produce rare-status outliers"
        );
    }

    #[test]
    fn rare_status_cardinality_above_50_skipped() {
        // 60 distinct values → cardinality cap rejects.
        let items: Vec<Value> = (0..60)
            .map(|i| json!({"code": format!("V_{}", i)}))
            .collect();
        let common: HashSet<String> = ["code".to_string()].into_iter().collect();
        let outliers = detect_rare_status_values(&items, &common);
        assert!(outliers.is_empty());
    }

    #[test]
    fn rare_status_cardinality_one_skipped() {
        // 100 items, all same value → cardinality 1, fails 2..=50 gate.
        let items: Vec<Value> = (0..100).map(|_| json!({"status": "ok"})).collect();
        let common: HashSet<String> = ["status".to_string()].into_iter().collect();
        let outliers = detect_rare_status_values(&items, &common);
        assert!(outliers.is_empty());
    }

    #[test]
    fn rare_status_nulls_filtered_from_cardinality() {
        // Pinned Python parity: `unique_values = {str(v) for v in values
        // if v is not None}` — nulls are excluded from the cardinality
        // computation. With 95×"ok" + 5×null, cardinality = 1 (just "ok"),
        // which fails the 2..=50 gate and the field is skipped entirely.
        // Pre-fix Python had the same behavior; the null-aware
        // `__none__` mapping only kicks in inside `value_counts` which
        // is unreachable when the cardinality gate fails first.
        //
        // If we ever want null to count as a distinct categorical value
        // (so missing-status items get flagged), that's a behavior
        // change beyond bug #3 and lives in Stage 3c.2.
        let mut items: Vec<Value> = (0..95).map(|_| json!({"s": "ok"})).collect();
        for _ in 0..5 {
            items.push(json!({"s": null}));
        }
        let common: HashSet<String> = ["s".to_string()].into_iter().collect();
        let outliers = detect_rare_status_values(&items, &common);
        assert!(
            outliers.is_empty(),
            "cardinality 1 (after null filter) must skip the field"
        );
    }

    #[test]
    fn rare_status_nulls_count_in_value_counts_when_cardinality_passes() {
        // Once cardinality >= 2 (with nulls excluded from the set), the
        // value_counts loop maps null → "__none__" and treats it as a
        // distinct value for frequency counting. Mirrors Python's
        // `key = str(v) if v is not None else "__none__"`.
        //
        // Setup: 90×"ok" + 5×"warn" + 5×null → unique_values = {"ok", "warn"},
        // cardinality 2, gate passes. value_counts: ok=90, warn=5,
        // __none__=5. top-1 = "ok" (90/100 = 90%) covers ≥80%, K=1 ≤ 5.
        // Items with "warn" or null are flagged.
        let mut items: Vec<Value> = (0..90).map(|_| json!({"s": "ok"})).collect();
        for _ in 0..5 {
            items.push(json!({"s": "warn"}));
        }
        for _ in 0..5 {
            items.push(json!({"s": null}));
        }
        let common: HashSet<String> = ["s".to_string()].into_iter().collect();
        let outliers = detect_rare_status_values(&items, &common);
        assert_eq!(outliers.len(), 10, "5 warn + 5 null = 10 outliers");
    }

    // ---------- detect_error_items_for_preservation ----------

    #[test]
    fn error_keywords_preservation_basic() {
        let items: Vec<Value> = vec![
            json!({"status": "ok"}),
            json!({"status": "error", "msg": "boom"}),
            json!({"status": "ok"}),
            json!({"msg": "request failed"}),
            json!({"status": "ok"}),
        ];
        let errs = detect_error_items_for_preservation(&items, None);
        assert_eq!(errs, vec![1, 3]);
    }

    #[test]
    fn error_keywords_case_insensitive() {
        let items: Vec<Value> = vec![
            json!({"msg": "FATAL: out of memory"}),
            json!({"msg": "panic at line 42"}),
        ];
        let errs = detect_error_items_for_preservation(&items, None);
        assert_eq!(errs, vec![0, 1]);
    }

    #[test]
    fn error_keywords_no_match() {
        let items: Vec<Value> = vec![json!({"name": "alice"}), json!({"count": 5})];
        let errs = detect_error_items_for_preservation(&items, None);
        assert!(errs.is_empty());
    }

    #[test]
    fn error_keywords_uses_cached_strings_when_provided() {
        // If `item_strings` is passed, we use those rather than
        // re-serializing. Test that a custom cached string can drive
        // a hit even when the actual item wouldn't.
        let items: Vec<Value> = vec![json!({"a": 1}), json!({"b": 2})];
        let cached = vec!["error".to_string(), "ok".to_string()];
        let errs = detect_error_items_for_preservation(&items, Some(&cached));
        assert_eq!(errs, vec![0]);
    }

    #[test]
    fn error_keywords_falls_back_when_cache_too_short() {
        // i >= len(item_strings) → fall back to fresh serialization.
        let items: Vec<Value> = vec![json!({"a": 1}), json!({"msg": "error"})];
        let cached = vec!["ok".to_string()]; // only 1 entry
        let errs = detect_error_items_for_preservation(&items, Some(&cached));
        assert_eq!(errs, vec![1]); // index 1 falls back to serialization, hits "error"
    }

    #[test]
    fn error_keywords_skips_non_dict_items() {
        let items: Vec<Value> = vec![
            json!({"msg": "error"}),
            json!("error string"), // not a dict — Python skips
            json!({"msg": "error"}),
        ];
        let errs = detect_error_items_for_preservation(&items, None);
        assert_eq!(errs, vec![0, 2]);
    }
}
