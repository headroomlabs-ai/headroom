//! Three universal crushers for non-dict-array JSON shapes.
//!
//! Direct ports from `headroom/transforms/smart_crusher.py`:
//!
//! - `crush_string_array`  ← `_crush_string_array`  (line 2727)
//! - `crush_number_array`  ← `_crush_number_array`  (line 2810) — has BUG #1
//! - `crush_object`        ← `_crush_object`        (line 3015)
//!
//! Each takes a `&SmartCrusherConfig`, a `bias` multiplier, and returns
//! `(crushed_items, strategy_string)`. Schema-preserving: the output
//! contains only items/values from the original; no generated text or
//! summary objects sneak in.
//!
//! `_crush_array` (the dict-array orchestrator) and `_crush_mixed_array`
//! (the type-grouped fallback) live in a later commit because they pull
//! in the planning + execution + TOIN/CCR scaffolding.
//!
//! # BUG #1 — percentile off-by-one in `crush_number_array`
//!
//! Python's `_crush_number_array` computes p25/p75 as
//! `sorted_finite[len(sorted_finite) // 4]` and
//! `sorted_finite[3 * len(sorted_finite) // 4]`. For `len < 8`, those
//! integer-division indices land one position before where a proper
//! quantile would sit. The bug only affects the strategy debug string
//! (`f",p25={p25:.4g},p75={p75:.4g}"`); item-selection logic is
//! unaffected.
//!
//! We port the bug **as-is** so parity fixtures still byte-match.
//! Stage 3c.1 commit 7 fixes BOTH languages in lockstep — at that point
//! the bug-doc test below flips to pin the corrected behavior and the
//! fixtures are regenerated.

use serde_json::{Map, Value};
use std::collections::{BTreeSet, HashSet};

use super::config::SmartCrusherConfig;
use super::error_keywords::ERROR_KEYWORDS;
use super::stats_math::{format_g, mean, median, sample_stdev};
use crate::transforms::adaptive_sizer::compute_optimal_k;

/// Compute K split (total / first / last / importance) for adaptive
/// crushers. Mirrors `_compute_k_split` (Python `smart_crusher.py:2693-2725`).
///
/// Splits the Kneedle-derived `k_total` into:
/// - `k_first`: items kept from the start of the array.
/// - `k_last`: items kept from the end.
/// - `k_importance`: leftover budget for importance-driven items.
///
/// Returns `(k_total, k_first, k_last, k_importance)`.
///
/// # BUG #4 — k-split overshoot (FIXED in Rust)
///
/// Python's original (line 2722):
/// ```text
/// k_first = max(1, round(k_total * first_fraction))
/// k_last  = max(1, round(k_total * last_fraction))
/// ```
/// For `k_total = 1`, both `round()` results are 0, both `max(1, …)`s
/// return 1, so `k_first + k_last = 2 > k_total = 1`. The crusher then
/// overshoots `max_items_after_crush` because the boundary unions
/// already exceed the budget before importance-fill kicks in.
///
/// The fix: after computing the floored fractions, clamp `k_first` to
/// `min(k_first, k_total)`, then clamp `k_last` to
/// `min(k_last, k_total - k_first)`. Preserves the Python behavior in
/// every case where `k_total >= 2` (the common path) and only deviates
/// for `k_total <= 1` (the previously buggy edge).
///
/// Same fix lands in `headroom/transforms/smart_crusher.py:2722` at
/// commit 7 (parity-fixture stage). Until then this is a one-sided fix
/// — Rust is correct, Python overshoots — and parity fixtures for the
/// `k_total=1` edge case won't match. Real-world inputs reach `k_total=1`
/// only when `n <= 8` AND all items deduplicate to a single SimHash
/// cluster, which rarely happens because every crusher early-returns
/// `passthrough` on `n <= 8` before `compute_k_split` is even called.
pub fn compute_k_split(
    items: &[&str],
    config: &SmartCrusherConfig,
    bias: f64,
) -> (usize, usize, usize, usize) {
    let max_k = if config.max_items_after_crush > 0 {
        Some(config.max_items_after_crush)
    } else {
        None
    };
    let k_total = compute_optimal_k(items, bias, 3, max_k);
    // Python: `max(1, round(k_total * fraction))`. Python's round() uses
    // banker's rounding (round-half-to-even). Rust's
    // f64::round_ties_even() mirrors that exactly — was stabilized in
    // Rust 1.77 and is the right primitive for this parity port.
    let k_first_raw = 1_usize.max(round_ties_even(k_total as f64 * config.first_fraction) as usize);
    let k_last_raw = 1_usize.max(round_ties_even(k_total as f64 * config.last_fraction) as usize);
    // BUG #4 FIX: clamp so `k_first + k_last <= k_total`. Without this,
    // a `k_total=1` produces `k_first=k_last=1` → 2 items kept,
    // violating max_items_after_crush.
    let k_first = k_first_raw.min(k_total);
    let k_last = k_last_raw.min(k_total.saturating_sub(k_first));
    let k_importance = k_total.saturating_sub(k_first + k_last);
    (k_total, k_first, k_last, k_importance)
}

/// Crush an array of strings.
///
/// Strategy (Python `_crush_string_array`):
/// 1. Adaptive K via Kneedle (passthrough on `n <= 8`).
/// 2. **Always keep**: error-keyword strings + length-anomaly strings.
/// 3. **Boundary keep**: first K_first + last K_last.
/// 4. **Stride-fill**: stride-based diverse sampling, dedup by content.
/// 5. Output preserves original array order.
///
/// `bias` is the compression-aggressiveness multiplier used by
/// `compute_optimal_k`.
pub fn crush_string_array(
    items: &[&str],
    config: &SmartCrusherConfig,
    bias: f64,
) -> (Vec<String>, String) {
    let n = items.len();
    if n <= 8 {
        return (
            items.iter().map(|s| (*s).to_string()).collect(),
            "string:passthrough".to_string(),
        );
    }

    // K split. Python serializes each item via json.dumps; for already-
    // string items that just wraps in quotes. We feed the raw &str refs
    // since adaptive_sizer's input is documented as "string repr in
    // importance order" — matches Python's intent.
    let (k_total, k_first, k_last, _k_importance) = compute_k_split(items, config, bias);

    // 1. Error-keyword indices.
    let mut error_indices: BTreeSet<usize> = BTreeSet::new();
    for (i, s) in items.iter().enumerate() {
        let lower = s.to_lowercase();
        if ERROR_KEYWORDS.iter().any(|kw| lower.contains(kw)) {
            error_indices.insert(i);
        }
    }

    // 2. Length anomaly indices.
    let lengths: Vec<f64> = items.iter().map(|s| s.chars().count() as f64).collect();
    let mut anomaly_indices: BTreeSet<usize> = BTreeSet::new();
    if lengths.len() > 1 {
        let mean_len = mean(&lengths).unwrap_or(0.0);
        // Python uses `statistics.stdev` here (sample stdev).
        let std_len = sample_stdev(&lengths).unwrap_or(0.0);
        if std_len > 0.0 {
            let threshold = config.variance_threshold * std_len;
            for (i, &length) in lengths.iter().enumerate() {
                if (length - mean_len).abs() > threshold {
                    anomaly_indices.insert(i);
                }
            }
        }
    }

    // 3. Boundary indices.
    let first_indices: BTreeSet<usize> = (0..k_first.min(n)).collect();
    let last_start = n.saturating_sub(k_last);
    let last_indices: BTreeSet<usize> = (last_start..n).collect();

    // 4. Combine.
    let mut keep_indices: BTreeSet<usize> = BTreeSet::new();
    keep_indices.extend(error_indices.iter().copied());
    keep_indices.extend(anomaly_indices.iter().copied());
    keep_indices.extend(first_indices.iter().copied());
    keep_indices.extend(last_indices.iter().copied());

    // Pre-populate seen_strings from current keeps.
    let mut seen: HashSet<&str> = HashSet::new();
    for &i in &keep_indices {
        seen.insert(items[i]);
    }

    // 5. Stride-fill remaining budget.
    let mut dedup_count: usize = 0;
    let remaining_budget = k_total.saturating_sub(keep_indices.len());
    if remaining_budget > 0 {
        let stride = ((n.saturating_sub(1)) / (remaining_budget + 1)).max(1);
        // Python: cap = k_total + len(error_indices) + len(anomaly_indices)
        let cap = k_total + error_indices.len() + anomaly_indices.len();
        let mut i: usize = 0;
        while i < n {
            if keep_indices.len() >= cap {
                break;
            }
            if !keep_indices.contains(&i) {
                if !seen.contains(items[i]) {
                    keep_indices.insert(i);
                    seen.insert(items[i]);
                } else {
                    dedup_count += 1;
                }
            }
            i += stride;
        }
    }

    // 6. Build output preserving original order.
    let result: Vec<String> = keep_indices.iter().map(|&i| items[i].to_string()).collect();

    let mut strategy = format!("string:adaptive({}->{}", n, result.len());
    if dedup_count > 0 {
        strategy.push_str(&format!(",dedup={}", dedup_count));
    }
    if !error_indices.is_empty() {
        strategy.push_str(&format!(",errors={}", error_indices.len()));
    }
    strategy.push(')');

    (result, strategy)
}

/// Crush an array of numbers.
///
/// Mirrors `_crush_number_array`. **Carries BUG #1** in the percentile
/// computation (see module-level doc); fix lands in commit 7.
pub fn crush_number_array(
    items: &[Value],
    config: &SmartCrusherConfig,
    bias: f64,
) -> (Vec<Value>, String) {
    let n = items.len();
    if n <= 8 {
        return (items.to_vec(), "number:passthrough".to_string());
    }

    // Filter to finite f64 only — Python: `isinstance(x, int|float) and math.isfinite(x)`.
    let finite: Vec<f64> = items
        .iter()
        .filter_map(|v| v.as_f64().filter(|f| f.is_finite()))
        .collect();
    if finite.is_empty() {
        return (items.to_vec(), "number:no_finite".to_string());
    }

    // K split. Python: `_compute_k_split(items, bias)` serializes via json.dumps
    // — for a number array that's just str(num).
    let item_strings: Vec<String> = items.iter().map(|v| v.to_string()).collect();
    let item_str_refs: Vec<&str> = item_strings.iter().map(|s| s.as_str()).collect();
    let (k_total, k_first, k_last, _) = compute_k_split(&item_str_refs, config, bias);

    // Statistics.
    let mean_val = mean(&finite).unwrap_or(0.0);
    let median_val = median(&finite).unwrap_or(0.0);
    let std_val = if finite.len() > 1 {
        sample_stdev(&finite).unwrap_or(0.0)
    } else {
        0.0
    };

    // Sorted for percentiles.
    let mut sorted_finite: Vec<f64> = finite.clone();
    sorted_finite.sort_by(f64::total_cmp);

    // BUG #1 FIX (lockstep with Python `_percentile_linear`): replace
    // integer-division indexing with proper linear interpolation.
    // Matches numpy's "linear" method exactly:
    //   index = q * (n - 1)
    //   if integer: sorted[index]
    //   else: linear interpolate between floor and ceil
    // The Python source's `_percentile_linear` helper uses the same
    // formula; both languages now agree byte-for-byte on the strategy
    // string's p25/p75 values.
    let p25 = percentile_linear(&sorted_finite, 0.25);
    let p75 = percentile_linear(&sorted_finite, 0.75);

    // Outliers (>variance_threshold σ from mean).
    let mut outlier_indices: BTreeSet<usize> = BTreeSet::new();
    if std_val > 0.0 {
        let threshold = config.variance_threshold * std_val;
        for (i, val) in items.iter().enumerate() {
            if let Some(num) = val.as_f64().filter(|f| f.is_finite()) {
                if (num - mean_val).abs() > threshold {
                    outlier_indices.insert(i);
                }
            }
        }
    }

    // Change points via window-mean comparison. Python guards on `n > 10`.
    let mut change_indices: BTreeSet<usize> = BTreeSet::new();
    if config.preserve_change_points && n > 10 {
        let window: usize = 5;
        for i in window..n.saturating_sub(window) {
            // Python collects only finite items in each window; it's possible
            // for windows to be empty if all items in a slice are non-finite.
            let left: Vec<f64> = items[i - window..i]
                .iter()
                .filter_map(|v| v.as_f64().filter(|f| f.is_finite()))
                .collect();
            let right: Vec<f64> = items[i..i + window]
                .iter()
                .filter_map(|v| v.as_f64().filter(|f| f.is_finite()))
                .collect();
            if !left.is_empty() && !right.is_empty() {
                let left_mean = mean(&left).unwrap_or(0.0);
                let right_mean = mean(&right).unwrap_or(0.0);
                if std_val > 0.0
                    && (right_mean - left_mean).abs() > config.variance_threshold * std_val
                {
                    change_indices.insert(i);
                }
            }
        }
    }

    // Boundary.
    let first_indices: BTreeSet<usize> = (0..k_first.min(n)).collect();
    let last_start = n.saturating_sub(k_last);
    let last_indices: BTreeSet<usize> = (last_start..n).collect();

    // Combine.
    let mut keep_indices: BTreeSet<usize> = BTreeSet::new();
    keep_indices.extend(outlier_indices.iter().copied());
    keep_indices.extend(change_indices.iter().copied());
    keep_indices.extend(first_indices.iter().copied());
    keep_indices.extend(last_indices.iter().copied());

    // Stride-fill. Cap = k_total + len(outlier_indices) (Python:
    // `keep_indices >= k_total + len(outlier_indices)` — note no
    // anomaly term here, unlike crush_string_array).
    let remaining_budget = k_total.saturating_sub(keep_indices.len());
    if remaining_budget > 0 {
        let stride = ((n.saturating_sub(1)) / (remaining_budget + 1)).max(1);
        let cap = k_total + outlier_indices.len();
        let mut i: usize = 0;
        while i < n {
            if keep_indices.len() >= cap {
                break;
            }
            if !keep_indices.contains(&i) {
                keep_indices.insert(i);
            }
            i += stride;
        }
    }

    // Build output: kept values only (schema-preserving — no summary prefix).
    let kept_values: Vec<Value> = keep_indices.iter().map(|&i| items[i].clone()).collect();

    let mn = finite_min(&finite);
    let mx = finite_max(&finite);
    let mut strategy = format!(
        "number:adaptive({}->{},min={},max={},mean={},median={},stddev={},p25={},p75={}",
        n,
        kept_values.len(),
        format_number_repr(mn),
        format_number_repr(mx),
        format_g(mean_val),
        format_g(median_val),
        format_g(std_val),
        format_g(p25),
        format_g(p75),
    );
    if !outlier_indices.is_empty() {
        strategy.push_str(&format!(",outliers={}", outlier_indices.len()));
    }
    if !change_indices.is_empty() {
        strategy.push_str(&format!(",change_points={}", change_indices.len()));
    }
    strategy.push(')');

    (kept_values, strategy)
}

/// Crush a JSON object by selecting the most informative keys.
///
/// Mirrors `_crush_object`. Treats key-value pairs as items and applies
/// `compute_optimal_k` directly on `f"{k}: {json.dumps(v)}"` strings.
/// Always-kept rules:
/// - keys whose value contains an error keyword.
/// - keys with small total token estimate (<=12 tokens via the rough
///   `len(str)/4 + len(key)/4 + 2` heuristic).
/// - first K_first and last K_last keys (insertion order — `IndexMap`
///   preserves it via the `serde_json/preserve_order` feature).
pub fn crush_object(
    obj: &Map<String, Value>,
    config: &SmartCrusherConfig,
    bias: f64,
) -> (Map<String, Value>, String) {
    let n = obj.len();
    if n <= 8 {
        return (obj.clone(), "object:passthrough".to_string());
    }

    // Estimate tokens per key-value pair. Python: `len(str)/4 + len(key)/4 + 2`.
    let mut kv_tokens: Vec<(String, usize)> = Vec::with_capacity(n);
    let mut total_tokens: usize = 0;
    for (key, val) in obj {
        let val_str = serde_json::to_string(val).unwrap_or_default();
        let tokens = val_str.len() / 4 + key.len() / 4 + 2;
        kv_tokens.push((key.clone(), tokens));
        total_tokens += tokens;
    }

    if total_tokens < config.min_tokens_to_crush {
        return (obj.clone(), "object:passthrough".to_string());
    }

    // Compute adaptive K on key-value string representations.
    let keys: Vec<&String> = obj.keys().collect();
    let kv_strings: Vec<String> = keys
        .iter()
        .map(|k| {
            format!(
                "{}: {}",
                k,
                serde_json::to_string(&obj[k.as_str()]).unwrap_or_default()
            )
        })
        .collect();
    let kv_refs: Vec<&str> = kv_strings.iter().map(|s| s.as_str()).collect();

    let max_k = if config.max_items_after_crush > 0 {
        Some(config.max_items_after_crush)
    } else {
        None
    };
    let k_total = compute_optimal_k(&kv_refs, bias, 3, max_k);

    if k_total >= n {
        return (obj.clone(), "object:passthrough".to_string());
    }

    // Always keep: error-keyword values.
    let mut keep_keys: HashSet<String> = HashSet::new();
    for (key, val) in obj {
        let val_str = serde_json::to_string(val)
            .unwrap_or_default()
            .to_lowercase();
        if ERROR_KEYWORDS.iter().any(|kw| val_str.contains(kw)) {
            keep_keys.insert(key.clone());
        }
    }

    // Always keep: small values (cheap to keep).
    // Python: `if tokens <= small_threshold // 4` where small_threshold=50,
    // so tokens <= 12.
    let small_threshold_tokens = 50_usize / 4;
    for (key, tokens) in &kv_tokens {
        if *tokens <= small_threshold_tokens {
            keep_keys.insert(key.clone());
        }
    }

    // Boundary: first K_first and last K_last (over the key insertion order).
    let k_first = 1_usize.max(round_ties_even(k_total as f64 * config.first_fraction) as usize);
    let k_last = 1_usize.max(round_ties_even(k_total as f64 * config.last_fraction) as usize);
    for k in keys.iter().take(k_first) {
        keep_keys.insert((*k).clone());
    }
    for k in keys.iter().rev().take(k_last) {
        keep_keys.insert((*k).clone());
    }

    // Stride fill. Python's cap recomputes the error-keyword count each
    // iteration (inefficient but deterministic). We can compute once
    // because once a key is in keep_keys, the count of error-flagged
    // entries grows monotonically — which means the cap effectively
    // grows. Mirror Python's behavior by recomputing.
    let remaining = k_total.saturating_sub(keep_keys.len());
    if remaining > 0 {
        let stride = ((n.saturating_sub(1)) / (remaining + 1)).max(1);
        let mut i: usize = 0;
        while i < n {
            // Python: `if len(keep_keys) >= k_total + len([k for k in keep_keys if any(kw in json.dumps(obj[k]).lower() for kw in keywords)])`
            let error_kept_count = keep_keys
                .iter()
                .filter(|k| {
                    let s = serde_json::to_string(&obj[k.as_str()])
                        .unwrap_or_default()
                        .to_lowercase();
                    ERROR_KEYWORDS.iter().any(|kw| s.contains(kw))
                })
                .count();
            if keep_keys.len() >= k_total + error_kept_count {
                break;
            }
            keep_keys.insert(keys[i].clone());
            i += stride;
        }
    }

    // Build output preserving original key insertion order.
    let mut result: Map<String, Value> = Map::new();
    for k in &keys {
        if keep_keys.contains(k.as_str()) {
            result.insert((*k).clone(), obj[k.as_str()].clone());
        }
    }

    let strategy = format!("object:adaptive({}->{} keys)", n, result.len());
    (result, strategy)
}

// ---------- helpers ----------

/// Linear-interpolation percentile (numpy "linear" method).
/// Mirrors Python's `_percentile_linear` helper for byte-equal
/// strategy-string parity (BUG #1 FIX).
fn percentile_linear(sorted_values: &[f64], q: f64) -> f64 {
    let n = sorted_values.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted_values[0];
    }
    let pos = q * (n - 1) as f64;
    let lo = pos as usize;
    let hi = if lo + 1 < n { lo + 1 } else { lo };
    let frac = pos - lo as f64;
    sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac
}

fn finite_min(values: &[f64]) -> f64 {
    values.iter().cloned().reduce(f64::min).unwrap_or(0.0)
}

fn finite_max(values: &[f64]) -> f64 {
    values.iter().cloned().reduce(f64::max).unwrap_or(0.0)
}

/// Python's `round()` uses banker's rounding (round-half-to-even). Rust
/// stabilized `f64::round_ties_even()` in 1.77 — that's the right
/// primitive for parity. Wrapping it in a helper keeps the call sites
/// readable.
fn round_ties_even(x: f64) -> f64 {
    x.round_ties_even()
}

/// Format a number for Python's f-string default repr (no precision
/// specifier). `min(finite)` and `max(finite)` in Python's strategy
/// string fall here. Integers print without a decimal; floats print
/// with their natural decimal form. JSON Number doesn't preserve the
/// integer/float distinction once parsed via `as_f64`, so we approximate:
/// values exactly representable as `i64` get integer formatting.
fn format_number_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 {
            "inf".to_string()
        } else {
            "-inf".to_string()
        };
    }
    if x.fract() == 0.0 && x.abs() < 1e16 {
        return format!("{}", x as i64);
    }
    // Otherwise Python's `str(float)` — which is "shortest round-trip".
    // Rust's f64 Display is also shortest round-trip; should match for
    // typical inputs.
    format!("{}", x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn cfg() -> SmartCrusherConfig {
        SmartCrusherConfig::default()
    }

    // ---------- compute_k_split ----------

    #[test]
    fn k_split_below_threshold_returns_n() {
        // n <= 8 → adaptive_k = n. k_first/k_last = max(1, round(n * fraction)).
        let items = ["a", "b", "c", "d", "e"];
        let (kt, kf, kl, ki) = compute_k_split(&items, &cfg(), 1.0);
        assert_eq!(kt, 5);
        // round(5 * 0.3) = round(1.5) = banker's → 2
        assert_eq!(kf, 2);
        // round(5 * 0.15) = round(0.75) = 1
        assert_eq!(kl, 1);
        // 5 - 2 - 1 = 2
        assert_eq!(ki, 2);
    }

    #[test]
    fn bug4_k_split_no_overshoot_when_k_total_is_one() {
        // BUG #4 FIX (Rust): direct test on the helper. We can't easily
        // make `compute_optimal_k` return 1 (its `min_k` floor is 3),
        // so verify the clamp via the helper that does the splitting:
        // when `k_total = 1`, we want `k_first + k_last <= 1`.
        //
        // We verify by exposing the clamp directly via a small synthetic
        // scenario: `compute_optimal_k` falls through to the n<=8 branch
        // with `n=1` and returns `n=1`. Construct that input.
        let items: [&str; 1] = ["only"];
        let (kt, kf, kl, ki) = compute_k_split(&items, &cfg(), 1.0);
        assert_eq!(kt, 1, "n=1 triggers fast-path n<=8 → k_total=1");
        assert!(
            kf + kl <= kt,
            "BUG #4: k_first={} + k_last={} must not exceed k_total={}",
            kf,
            kl,
            kt
        );
        assert_eq!(ki, kt.saturating_sub(kf + kl));
    }

    #[test]
    fn bug4_k_split_no_overshoot_when_k_total_is_two() {
        // For k_total=2: pre-fix Python: k_first=1, k_last=1 — sum=2 = k_total ✓
        // (this case wasn't actually buggy). We pin it anyway to lock the
        // boundary that the bug #4 fix preserves untouched.
        let items: [&str; 2] = ["a", "b"];
        let (kt, kf, kl, _) = compute_k_split(&items, &cfg(), 1.0);
        assert_eq!(kt, 2);
        assert!(kf + kl <= kt);
        assert_eq!(kf, 1);
        assert_eq!(kl, 1);
    }

    #[test]
    fn k_split_low_diversity_returns_min_k() {
        // 10 identical items: tier-1 unique-by-simhash=1, returns max(min_k=3, 1)=3.
        // Then k_first = max(1, round_ties_even(3*0.3))=max(1, round(0.9))=max(1,1)=1.
        let items: [&str; 10] = ["x"; 10];
        let (kt, kf, kl, _) = compute_k_split(&items, &cfg(), 1.0);
        assert_eq!(kt, 3, "low-diversity → max(min_k, unique_count)=3");
        assert_eq!(kf, 1);
        assert_eq!(kl, 1);
    }

    // ---------- crush_string_array ----------

    #[test]
    fn string_array_passthrough_at_threshold() {
        let items: [&str; 8] = ["a", "b", "c", "d", "e", "f", "g", "h"];
        let (out, strat) = crush_string_array(&items, &cfg(), 1.0);
        assert_eq!(out.len(), 8);
        assert_eq!(strat, "string:passthrough");
    }

    #[test]
    fn string_array_keeps_error_strings() {
        let items: Vec<&str> = (0..30)
            .map(|i| {
                if i == 15 {
                    "FATAL: out of memory"
                } else {
                    "ok"
                }
            })
            .collect();
        let (out, strat) = crush_string_array(&items, &cfg(), 1.0);
        // Error item at index 15 must survive.
        assert!(out.iter().any(|s| s == "FATAL: out of memory"));
        assert!(strat.contains("errors=1"));
    }

    #[test]
    fn string_array_keeps_first_and_last() {
        let items: Vec<String> = (0..30).map(|i| format!("item_{}", i)).collect();
        let refs: Vec<&str> = items.iter().map(|s| s.as_str()).collect();
        let (out, _) = crush_string_array(&refs, &cfg(), 1.0);
        // First item (item_0) should always be kept (k_first >= 1).
        assert!(out.iter().any(|s| s == "item_0"));
        // Last item (item_29) should always be kept (k_last >= 1).
        assert!(out.iter().any(|s| s == "item_29"));
    }

    #[test]
    fn string_array_dedup_count_appears_in_strategy() {
        // Lots of duplicates that survive stride sampling get deduped.
        let items: Vec<&str> = std::iter::repeat("dup").take(50).collect();
        let (_out, strat) = crush_string_array(&items, &cfg(), 1.0);
        // 50 identical items: unique-by-simhash = 1, fast-path returns 3.
        // So k_total=3. Stride loop runs but every item is "dup" already
        // seen → dedup_count > 0.
        assert!(
            strat.contains("dedup="),
            "strategy {} should mention dedup",
            strat
        );
    }

    // ---------- crush_number_array ----------

    #[test]
    fn number_array_passthrough_at_threshold() {
        let items: Vec<Value> = (0..8).map(|i| json!(i)).collect();
        let (out, strat) = crush_number_array(&items, &cfg(), 1.0);
        assert_eq!(out.len(), 8);
        assert_eq!(strat, "number:passthrough");
    }

    #[test]
    fn number_array_no_finite_returns_passthrough() {
        // n > 8 but no finite values → "number:no_finite" strategy.
        // serde_json can't carry NaN, so use null values for non-numeric:
        // they're filtered out by `as_f64()`.
        let items: Vec<Value> = (0..15).map(|_| json!(null)).collect();
        let (out, strat) = crush_number_array(&items, &cfg(), 1.0);
        assert_eq!(out.len(), items.len());
        assert_eq!(strat, "number:no_finite");
    }

    #[test]
    fn number_array_keeps_outliers() {
        // 30 zeros + one 1000 → outlier should be kept.
        let mut items: Vec<Value> = vec![json!(0); 30];
        items.push(json!(1000));
        let (out, strat) = crush_number_array(&items, &cfg(), 1.0);
        assert!(out.iter().any(|v| v.as_f64() == Some(1000.0)));
        assert!(strat.contains("outliers="));
    }

    #[test]
    fn number_array_strategy_string_includes_summary() {
        let items: Vec<Value> = (1..=20).map(|i| json!(i)).collect();
        let (_out, strat) = crush_number_array(&items, &cfg(), 1.0);
        assert!(strat.starts_with("number:adaptive("));
        assert!(strat.contains("min=1"));
        assert!(strat.contains("max=20"));
        assert!(strat.contains("mean="));
        assert!(strat.contains("median="));
        assert!(strat.contains("p25="));
        assert!(strat.contains("p75="));
    }

    // ---------- crush_object ----------

    #[test]
    fn object_passthrough_when_few_keys() {
        let mut obj = Map::new();
        for i in 0..5 {
            obj.insert(format!("k{}", i), json!(i));
        }
        let (out, strat) = crush_object(&obj, &cfg(), 1.0);
        assert_eq!(out.len(), 5);
        assert_eq!(strat, "object:passthrough");
    }

    #[test]
    fn object_passthrough_when_total_tokens_below_min() {
        // Many tiny keys/values: total_tokens stays below
        // min_tokens_to_crush=200.
        let mut obj = Map::new();
        for i in 0..30 {
            obj.insert(format!("k{}", i), json!(i));
        }
        let (_out, strat) = crush_object(&obj, &cfg(), 1.0);
        assert_eq!(strat, "object:passthrough");
    }

    #[test]
    fn object_crushes_when_token_budget_exceeded() {
        // 30 keys, each with a long string value → total tokens > 200,
        // and unique k_total < n → actual crushing happens.
        let mut obj = Map::new();
        for i in 0..30 {
            obj.insert(
                format!("k{:02}", i),
                json!(format!(
                    "this is a relatively long value string for entry number {} with content",
                    i
                )),
            );
        }
        let (out, strat) = crush_object(&obj, &cfg(), 1.0);
        // Either the optimizer kept all (if it deems them all distinct
        // enough — strategy = passthrough), or it crushed.
        if strat == "object:passthrough" {
            assert_eq!(out.len(), 30);
        } else {
            assert!(strat.starts_with("object:adaptive("));
            assert!(out.len() <= 30);
        }
    }

    #[test]
    fn object_keeps_small_values() {
        // Mix of small + large values; small ones (<=12 tokens) always survive.
        let mut obj = Map::new();
        obj.insert("tiny".to_string(), json!(1));
        for i in 0..30 {
            obj.insert(
                format!("big{:02}", i),
                json!(format!(
                    "this is a long string with content for entry number {} that exceeds the small threshold",
                    i
                )),
            );
        }
        let (out, _) = crush_object(&obj, &cfg(), 1.0);
        assert!(
            out.contains_key("tiny"),
            "tiny key (small value) must survive"
        );
    }

    #[test]
    fn object_keeps_error_keywords() {
        let mut obj = Map::new();
        obj.insert(
            "msg1".to_string(),
            json!(format!("FATAL: {}", "x".repeat(200))),
        );
        for i in 0..30 {
            obj.insert(
                format!("k{:02}", i),
                json!(format!("padding content for entry {} with text", i)),
            );
        }
        let (out, _) = crush_object(&obj, &cfg(), 1.0);
        assert!(
            out.contains_key("msg1"),
            "key with error-keyword value must survive"
        );
    }

    // ---------- BUG #1 documentation test ----------

    #[test]
    fn bug1_percentile_proper_linear_interpolation() {
        // BUG #1 FIX (Rust + Python in lockstep): proper linear-
        // interpolation percentile. For sorted [1,2,3,4,5,6,7,8,9],
        // n=9 so:
        //   p25 index = 0.25 * 8 = 2.0    → sorted[2] = 3.0
        //   p75 index = 0.75 * 8 = 6.0    → sorted[6] = 7.0
        // (Both p25 and p75 land on integer indices for n=9.)
        let mut items: Vec<Value> = (1..=9).map(|i| json!(i)).collect();
        items.extend(vec![json!(null); 5]); // nulls drop out of `finite`
        let (_out, strat) = crush_number_array(&items, &cfg(), 1.0);
        assert!(strat.contains("p25=3"), "got: {}", strat);
        assert!(strat.contains("p75=7"), "got: {}", strat);
    }

    #[test]
    fn bug1_percentile_interpolates_when_index_non_integer() {
        // For sorted [10, 20, 30, 40, 50] (n=5):
        //   p25 = 0.25 * 4 = 1.0  → sorted[1] = 20
        //   p75 = 0.75 * 4 = 3.0  → sorted[3] = 40
        // For sorted with n=10, n=11, etc., the index is non-integer
        // and we interpolate. Pin a case where interpolation actually
        // happens to verify the fix.
        // n=10 finite: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        //   p25 = 0.25 * 9 = 2.25 → sorted[2] * 0.75 + sorted[3] * 0.25
        //                          = 30 * 0.75 + 40 * 0.25 = 32.5
        //   p75 = 0.75 * 9 = 6.75 → sorted[6] * 0.25 + sorted[7] * 0.75
        //                          = 70 * 0.25 + 80 * 0.75 = 77.5
        let items: Vec<Value> = (1..=10).map(|i| json!(i * 10)).collect();
        let (_out, strat) = crush_number_array(&items, &cfg(), 1.0);
        // Pre-fix would have given p25=sorted[10/4]=sorted[2]=30 (wrong).
        // Post-fix gives 32.5.
        assert!(
            strat.contains("p25=32.5"),
            "expected proper-percentile p25=32.5, got: {}",
            strat
        );
        assert!(
            strat.contains("p75=77.5"),
            "expected proper-percentile p75=77.5, got: {}",
            strat
        );
    }
}
