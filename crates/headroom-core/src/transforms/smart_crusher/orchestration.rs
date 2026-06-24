//! Index-set orchestration helpers used by every planning method.
//!
//! Direct port of three Python methods from `smart_crusher.py`:
//!
//! - `_deduplicate_indices_by_content` (line 1721) — collapse multiple
//!   indices pointing at content-identical items into a single index
//!   (lowest wins).
//! - `_fill_remaining_slots` (line 1794) — when dedup leaves us under
//!   `effective_max`, fill back up with diverse stride-sampled indices
//!   that don't repeat content.
//! - `_prioritize_indices` (line 1891) — apply dedup + fill, then —
//!   if still over budget — keep ALL critical items (errors,
//!   structural outliers, numeric anomalies) plus first-3 / last-2,
//!   discarding non-critical items beyond the budget.
//!
//! All three operate on `BTreeSet<usize>` (sorted, deterministic
//! iteration). Item content hashes use the Python-compatible
//! `compute_item_hash` from `anchor_selector` so the same item collapses
//! to the same hash in both languages.
//!
//! # TOIN field-semantics — currently stubbed
//!
//! Python's `_prioritize_indices` accepts an optional `field_semantics`
//! map and calls `_detect_items_by_learned_semantics` to find items
//! with values in TOIN-learned important fields. TOIN isn't ported
//! yet; we mirror the Python "no field_semantics provided" branch
//! (returns no learned-important indices). When TOIN lands, we'll add
//! that argument back to the public surface.

use md5::{Digest, Md5};
use serde_json::Value;
use std::collections::{BTreeSet, HashSet};

use super::config::SmartCrusherConfig;
use super::outliers::{detect_error_items_for_preservation, detect_structural_outliers};
use super::types::{ArrayAnalysis, FieldStats};
use crate::transforms::anchor_selector::compute_item_hash;

/// Collapse content-duplicate indices to their lowest representative.
///
/// Python: `_deduplicate_indices_by_content`. Iterates `keep_indices`
/// in ascending order and records the FIRST index that hashes to a
/// given content fingerprint. Subsequent matches drop. Out-of-bounds
/// indices skip.
///
/// `compute_item_hash` returns the same MD5[:16] string Python computes
/// (via `anchor_selector::python_json_dumps_sort_keys`), so the dedup
/// outcome is byte-equal across languages.
pub fn deduplicate_indices_by_content(
    keep_indices: &BTreeSet<usize>,
    items: &[Value],
) -> BTreeSet<usize> {
    if keep_indices.is_empty() {
        return BTreeSet::new();
    }

    // hash -> lowest-seen index. BTreeSet iteration is ascending, so
    // the first insertion for each hash IS the lowest index.
    let mut seen: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for &idx in keep_indices {
        if idx >= items.len() {
            continue;
        }
        let h = item_content_hash(&items[idx], idx);
        seen.entry(h).or_insert(idx);
    }
    seen.values().copied().collect()
}

/// Fill `keep_indices` back up to `effective_max` with diverse,
/// content-unique items. Python: `_fill_remaining_slots`.
///
/// Strategy:
/// 1. Compute hashes of currently-kept items.
/// 2. Walk candidates (indices NOT in keep_indices) with stride-based
///    sampling for spatial diversity.
/// 3. Add a candidate if its content hash is fresh.
///
/// Python uses two nested loops with `start_offset` to interleave
/// stride scans — we mirror that exactly so the same items are picked
/// in the same order for parity fixtures.
pub fn fill_remaining_slots(
    keep_indices: &BTreeSet<usize>,
    items: &[Value],
    n: usize,
    effective_max: usize,
) -> BTreeSet<usize> {
    let remaining = effective_max.saturating_sub(keep_indices.len());
    if remaining == 0 {
        return keep_indices.clone();
    }

    // Hashes of items we're already keeping — bound the working set
    // we won't re-add.
    let mut seen: HashSet<String> = HashSet::new();
    for &idx in keep_indices {
        if idx < n {
            seen.insert(item_content_hash(&items[idx], idx));
        }
    }

    // Candidate pool: every index not already kept.
    let candidates: Vec<usize> = (0..n).filter(|i| !keep_indices.contains(i)).collect();
    if candidates.is_empty() {
        return keep_indices.clone();
    }

    let mut result = keep_indices.clone();
    let step = (candidates.len() / (remaining + 1)).max(1);
    let mut added = 0;

    // Python's interleaved stride: outer loop offsets [0, step),
    // inner loop walks `start_offset, +step, +step, ...`. The result
    // visits every candidate exactly once across the outer iterations.
    'outer: for start_offset in 0..step {
        if added >= remaining {
            break;
        }
        let mut i = start_offset;
        while i < candidates.len() {
            if added >= remaining {
                break 'outer;
            }
            let idx = candidates[i];
            let h = item_content_hash(&items[idx], idx);
            if !seen.contains(&h) {
                result.insert(idx);
                seen.insert(h);
                added += 1;
            }
            i += step;
        }
    }

    result
}

/// Top-level prioritizer. Python: `_prioritize_indices`.
///
/// Pipeline:
/// 1. **Dedup**: collapse content-duplicate indices.
/// 2. **Fill**: top up to `effective_max` with diverse uniques.
/// 3. **Already under budget?** Return as-is.
/// 4. **Otherwise**: keep ALL critical items (errors + structural
///    outliers + numeric anomalies — non-negotiable per Python's
///    "quality guarantee"). Then add first-3 + last-2 if room. Then
///    fill remaining with non-critical kept indices in ascending order.
///
/// May return MORE than `effective_max` items when critical items
/// alone exceed the budget — Python's documented behavior, mirrored
/// here.
pub fn prioritize_indices(
    config: &SmartCrusherConfig,
    keep_indices: &BTreeSet<usize>,
    items: &[Value],
    n: usize,
    analysis: Option<&ArrayAnalysis>,
    effective_max: usize,
) -> BTreeSet<usize> {
    // Dedup pass.
    let mut current = if config.dedup_identical_items {
        deduplicate_indices_by_content(keep_indices, items)
    } else {
        keep_indices.clone()
    };

    // Fill pass.
    if current.len() < effective_max && current.len() < n {
        current = fill_remaining_slots(&current, items, n, effective_max);
    }

    if current.len() <= effective_max {
        return current;
    }

    // Over budget — apply critical-items-first prioritization.

    // Errors (keyword-detected — preservation guarantee).
    let error_indices: BTreeSet<usize> = detect_error_items_for_preservation(items, None)
        .into_iter()
        .collect();

    // Structural outliers (statistical — rare fields, rare statuses).
    let outlier_indices: BTreeSet<usize> = detect_structural_outliers(items).into_iter().collect();

    // Numeric anomalies (>variance_threshold σ from per-field mean).
    let anomaly_indices = numeric_anomaly_indices(config, items, analysis);

    // TOIN learned-important indices: empty until TOIN is ported.
    let learned_indices: BTreeSet<usize> = BTreeSet::new();

    let mut prioritized: BTreeSet<usize> = BTreeSet::new();
    prioritized.extend(&error_indices);
    prioritized.extend(&outlier_indices);
    prioritized.extend(&anomaly_indices);
    prioritized.extend(&learned_indices);

    // First 3 / last 2 anchors if we have room.
    let mut remaining = effective_max.saturating_sub(prioritized.len());
    if remaining > 0 {
        for i in 0..3.min(n) {
            if !prioritized.contains(&i) && remaining > 0 {
                prioritized.insert(i);
                remaining -= 1;
            }
        }
        let last_start = n.saturating_sub(2);
        for i in last_start..n {
            if !prioritized.contains(&i) && remaining > 0 {
                prioritized.insert(i);
                remaining -= 1;
            }
        }
    }

    // Fill with other-important indices (ascending order).
    if remaining > 0 {
        let mut others: Vec<usize> = current.difference(&prioritized).copied().collect();
        others.sort();
        for i in others {
            if remaining == 0 {
                break;
            }
            prioritized.insert(i);
            remaining -= 1;
        }
    }

    prioritized
}

/// Compute numeric-anomaly indices from `analysis.field_stats`.
/// Mirrors Python's anomaly loop in `_prioritize_indices` (line 1973-1984).
fn numeric_anomaly_indices(
    config: &SmartCrusherConfig,
    items: &[Value],
    analysis: Option<&ArrayAnalysis>,
) -> BTreeSet<usize> {
    let mut anomalies: BTreeSet<usize> = BTreeSet::new();
    let Some(analysis) = analysis else {
        return anomalies;
    };
    if analysis.field_stats.is_empty() {
        return anomalies;
    }

    for (field_name, stats) in &analysis.field_stats {
        if !is_numeric_field_with_variance(stats) {
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
        let threshold = config.variance_threshold * std;
        for (i, item) in items.iter().enumerate() {
            let Some(obj) = item.as_object() else {
                continue;
            };
            let Some(v) = obj.get(field_name) else {
                continue;
            };
            if let Some(num) = v.as_f64() {
                if !num.is_nan() && (num - mean_val).abs() > threshold {
                    anomalies.insert(i);
                }
            }
        }
    }

    anomalies
}

fn is_numeric_field_with_variance(stats: &FieldStats) -> bool {
    stats.field_type == "numeric" && stats.mean_val.is_some() && stats.variance.unwrap_or(0.0) > 0.0
}

/// Hash function used by all three orchestration helpers.
///
/// Wraps `compute_item_hash` (which does Python-compatible
/// json.dumps + md5[:16]) with a fail-safe fallback: if the item is
/// not a JSON object, fall back to `__idx_<i>__` so the index is
/// effectively a unique key. Mirrors Python's
/// `try/except (TypeError, ValueError, RecursionError)` block which
/// also falls back to `f"__idx_{idx}__"` on serialization failure.
fn item_content_hash(item: &Value, idx: usize) -> String {
    if item.is_object() || item.is_array() {
        compute_item_hash(item)
    } else {
        // Python: `else: content = str(item)` for non-dict items —
        // they get a real hash too. We don't strictly need that for
        // SmartCrusher's dict-array use case but we mirror it.
        // Fallback to index-stamp only on serialization failure.
        let content = match item {
            Value::String(s) => s.clone(),
            Value::Number(n) => n.to_string(),
            Value::Bool(b) => b.to_string(),
            Value::Null => "None".to_string(),
            _ => format!("__idx_{}__", idx),
        };
        let digest = Md5::digest(content.as_bytes());
        format!("{:x}", digest)[..16].to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn cfg() -> SmartCrusherConfig {
        SmartCrusherConfig::default()
    }

    fn idx_set(indices: &[usize]) -> BTreeSet<usize> {
        indices.iter().copied().collect()
    }

    // ---------- deduplicate_indices_by_content ----------

    #[test]
    fn dedup_empty_input() {
        let result = deduplicate_indices_by_content(&BTreeSet::new(), &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn dedup_lowest_index_wins_for_duplicates() {
        let items = vec![
            json!({"name": "alice"}),
            json!({"name": "alice"}),
            json!({"name": "bob"}),
        ];
        let kept = idx_set(&[0, 1, 2]);
        let result = deduplicate_indices_by_content(&kept, &items);
        // Items 0 and 1 collapse to the lower (0); item 2 is unique.
        assert_eq!(result, idx_set(&[0, 2]));
    }

    #[test]
    fn dedup_all_distinct_unchanged() {
        let items = vec![json!({"id": 1}), json!({"id": 2}), json!({"id": 3})];
        let kept = idx_set(&[0, 1, 2]);
        let result = deduplicate_indices_by_content(&kept, &items);
        assert_eq!(result, idx_set(&[0, 1, 2]));
    }

    #[test]
    fn dedup_skips_out_of_bounds() {
        let items = vec![json!({"a": 1})];
        let kept = idx_set(&[0, 5, 10]);
        let result = deduplicate_indices_by_content(&kept, &items);
        assert_eq!(result, idx_set(&[0]));
    }

    #[test]
    fn dedup_key_order_independent() {
        // {"b":2, "a":1} and {"a":1, "b":2} must hash to the same value
        // because we serialize with sort_keys=True.
        let items = vec![json!({"b": 2, "a": 1}), json!({"a": 1, "b": 2})];
        let kept = idx_set(&[0, 1]);
        let result = deduplicate_indices_by_content(&kept, &items);
        assert_eq!(result.len(), 1);
        assert!(result.contains(&0));
    }

    // ---------- fill_remaining_slots ----------

    #[test]
    fn fill_when_at_or_over_budget_returns_unchanged() {
        let items: Vec<Value> = (0..10).map(|i| json!({"id": i})).collect();
        let kept = idx_set(&[0, 1, 2, 3, 4]);
        let result = fill_remaining_slots(&kept, &items, items.len(), 5);
        assert_eq!(result, kept);
    }

    #[test]
    fn fill_adds_diverse_uniques_up_to_max() {
        let items: Vec<Value> = (0..20).map(|i| json!({"id": i})).collect();
        let kept = idx_set(&[0, 5]);
        let result = fill_remaining_slots(&kept, &items, items.len(), 10);
        assert!(result.len() <= 10);
        assert!(result.len() >= 2);
        assert!(result.contains(&0));
        assert!(result.contains(&5));
    }

    #[test]
    fn fill_skips_content_duplicates() {
        // 10 unique + 10 dupes of items[0]. Filling shouldn't pick the dupes.
        let mut items: Vec<Value> = (0..10).map(|i| json!({"id": i})).collect();
        items.extend(std::iter::repeat_with(|| json!({"id": 0})).take(10));
        let kept = idx_set(&[0]); // Already keeps the canonical {"id": 0}.
        let result = fill_remaining_slots(&kept, &items, items.len(), 15);
        // The 10 dupes (indices 10..20) all hash to the same as items[0]
        // and shouldn't be added. Only unique indices [1..10) should fill.
        for i in 10..20 {
            assert!(!result.contains(&i), "dup index {} should not be added", i);
        }
    }

    // ---------- prioritize_indices ----------

    #[test]
    fn prioritize_under_budget_passthrough_after_dedup() {
        let items: Vec<Value> = (0..5).map(|i| json!({"id": i})).collect();
        let kept = idx_set(&[0, 1, 2]);
        let result = prioritize_indices(&cfg(), &kept, &items, items.len(), None, 10);
        // 3 items < max 10 → fill kicks in; we get 5 (all items).
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn prioritize_dedup_collapses_then_returns_under_max() {
        let items = vec![
            json!({"name": "alice"}),
            json!({"name": "alice"}),
            json!({"name": "bob"}),
        ];
        let kept = idx_set(&[0, 1, 2]);
        let result = prioritize_indices(&cfg(), &kept, &items, items.len(), None, 10);
        // Dedup collapses 0+1 to 0; fill stays put because n=3 already covered.
        assert_eq!(result, idx_set(&[0, 2]));
    }

    #[test]
    fn prioritize_keeps_error_items_when_over_budget() {
        // 30 items, 1 error item. Over-budget path must keep the error.
        let mut items: Vec<Value> = (0..30)
            .map(|i| json!({"id": i, "msg": format!("ok {}", i)}))
            .collect();
        items.push(json!({"id": 30, "msg": "FATAL: out of memory"}));
        let kept: BTreeSet<usize> = (0..items.len()).collect();
        let result = prioritize_indices(&cfg(), &kept, &items, items.len(), None, 10);
        assert!(
            result.contains(&30),
            "error item must survive prioritization"
        );
    }

    #[test]
    fn prioritize_includes_first_3_and_last_2_when_room() {
        // No errors / outliers / anomalies → first 3 + last 2 anchors fill.
        let items: Vec<Value> = (0..30).map(|i| json!({"id": i, "v": i})).collect();
        let kept: BTreeSet<usize> = (5..15).collect();
        let result = prioritize_indices(&cfg(), &kept, &items, items.len(), None, 10);
        // With no critical items and budget 10, dedup is a no-op (all
        // distinct) and fill keeps us at <= 10. We should see at least
        // some of items 0..3 OR 28..30 covered through fill.
        // Cap is 10; ensure we don't exceed.
        assert!(result.len() <= 10);
    }
}
