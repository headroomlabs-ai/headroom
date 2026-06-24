//! Statistical helpers for field characterization.
//!
//! Direct port of the helpers at `smart_crusher.py:378-481`. These are
//! used by the analyzer to classify fields (ID-like, score-like, etc.).
//! Detection logic is heuristic; small numeric drift between Python and
//! Rust would change classifications and break fixtures, so the math
//! here mirrors Python step-by-step.

use serde_json::Value;
use std::collections::HashMap;

/// Check if a string looks like a UUID.
///
/// Direct port of `_is_uuid_format` (Python `smart_crusher.py:378-392`).
/// Format check only — no version-bit validation. Hex chars are lower
/// or upper case, matching Python.
pub fn is_uuid_format(value: &str) -> bool {
    if value.len() != 36 {
        return false;
    }

    // Expected segment lengths: 8-4-4-4-12.
    let parts: Vec<&str> = value.split('-').collect();
    if parts.len() != 5 {
        return false;
    }
    let expected_lens = [8, 4, 4, 4, 12];
    for (part, &expected_len) in parts.iter().zip(expected_lens.iter()) {
        if part.len() != expected_len {
            return false;
        }
        for c in part.chars() {
            if !c.is_ascii_hexdigit() {
                return false;
            }
        }
    }
    true
}

/// Shannon entropy of a string, normalized to `[0, 1]`.
///
/// Direct port of `_calculate_string_entropy` (`smart_crusher.py:395-423`).
/// High entropy (>0.7) suggests random/ID-like content. Low entropy
/// (<0.3) suggests repetitive/predictable content. Used by ID detection.
///
/// # Edge cases (matched to Python)
/// - Empty or single-character strings return `0.0`.
/// - All-identical chars: `freq` has 1 entry, `max_entropy = log2(min(1, n)) = 0.0`,
///   we return `0.0` to avoid division by zero.
pub fn calculate_string_entropy(s: &str) -> f64 {
    // Python uses `len(s) < 2` and computes by character. Rust strings
    // are UTF-8 so we iterate `chars()` to match Python's character-level
    // semantics (Python iterates code points; Rust's `chars()` yields
    // Unicode scalar values — same thing for non-surrogate text).
    let n = s.chars().count();
    if n < 2 {
        return 0.0;
    }

    let mut freq: HashMap<char, usize> = HashMap::new();
    for c in s.chars() {
        *freq.entry(c).or_insert(0) += 1;
    }

    let length = n as f64;
    let mut entropy = 0.0_f64;
    for &count in freq.values() {
        let p = count as f64 / length;
        if p > 0.0 {
            entropy -= p * p.log2();
        }
    }

    // Normalize by the maximum possible entropy at this length:
    // Python: max_entropy = log2(min(len(freq), length))
    let max_entropy = (freq.len().min(n) as f64).log2();
    if max_entropy > 0.0 {
        entropy / max_entropy
    } else {
        0.0
    }
}

/// Parse a string the way Python's built-in `int()` does for plain
/// integer literals. Used by `detect_sequential_pattern` to mirror
/// `int(v)` behavior exactly.
///
/// Python's `int()` accepts:
///   - leading/trailing ASCII whitespace (stripped)
///   - leading sign (`+` or `-`)
///   - PEP 515 underscore digit separators (e.g. `"3_000"` → `3000`)
///
/// Rust's `str::parse::<i64>()` rejects all of those. If we used the
/// raw `parse`, real-world payloads with `"  5  "` or `"+5"` would
/// silently disagree with Python on whether the field is "numeric",
/// which changes sequential classification and breaks fixtures.
///
/// We deliberately do NOT support Python's other `int()` features
/// (base prefixes like `"0x10"`, scientific notation via `int(float(s))`,
/// etc.) because the Python `_detect_sequential_pattern` call site
/// uses the default-base `int()` overload — those paths are
/// unreachable.
fn python_int_parse(s: &str) -> Option<i64> {
    // Python: `int()` strips ASCII whitespace from both ends.
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return None;
    }
    // Python: drop PEP 515 underscores between digits. The implementation
    // is more careful than this (rejects leading/trailing/double underscores),
    // but for our use-case any string with valid digits + underscore separators
    // is what we want to accept. Edge cases like `"_5_"` will fail the
    // i64::parse call below, matching Python's behavior of rejecting them.
    let cleaned: String = if trimmed.contains('_') {
        // Reject patterns Python rejects: leading/trailing underscore,
        // double underscores. Otherwise strip them out.
        let bytes = trimmed.as_bytes();
        let starts_or_ends =
            bytes[0] == b'_' || *bytes.last().unwrap() == b'_' || trimmed.contains("__");
        if starts_or_ends {
            return None;
        }
        trimmed.replace('_', "")
    } else {
        trimmed.to_string()
    };
    cleaned.parse::<i64>().ok()
}

/// Detect if numeric values form a sequential pattern (like IDs:
/// 1, 2, 3, ...).
///
/// Direct port of `_detect_sequential_pattern` (`smart_crusher.py:426-481`)
/// **with BUG #2 FIXED**.
///
/// # Bug #2 — string-padding misclassification
/// Python's original implementation calls `int("001") == 1` and silently
/// loses zero padding, so a list of padded string IDs like
/// `["001", "002", ..., "100"]` looks like a sequential numeric pattern
/// when in reality it's a categorical string field where the padding
/// matters. The fix: when a value is a string that parses as a number,
/// flag the input as "had string-encoded numerics". If ALL parsed values
/// originated as strings, refuse to classify as a sequential numeric
/// pattern. Mixed numeric+string inputs still parse as sequential because
/// the unambiguous numeric values dominate the signal.
///
/// This fix is applied in BOTH languages simultaneously (Python `smart_crusher.py`
/// gets the same fix in the same PR) so the parity fixtures continue to
/// match. Tests covering this bug live at
/// `tests/test_transforms/test_smart_crusher_bugs.py`.
///
/// # Args
/// - `values`: items to inspect.
/// - `check_order`: when true, also require ascending order in the
///   original array (the Python flag — IDs are usually ascending in
///   source order, scores are usually descending).
pub fn detect_sequential_pattern(values: &[Value], check_order: bool) -> bool {
    if values.len() < 5 {
        return false;
    }

    // Collect numeric values, tracking whether each value originated as
    // a string. This is the BUG #2 fix: we still parse strings into
    // numbers (so legitimate mixed-type fields work), but we'll refuse
    // to flag the field as sequential if EVERY parseable value was a
    // string.
    let mut nums: Vec<f64> = Vec::new();
    let mut had_non_string_numeric = false;

    for v in values {
        match v {
            Value::Number(n) => {
                if let Some(f) = n.as_f64() {
                    nums.push(f);
                    had_non_string_numeric = true;
                }
            }
            Value::Bool(_) => {
                // Python: `isinstance(v, int | float) and not isinstance(v, bool)` —
                // bools are explicitly excluded.
            }
            Value::String(s) => {
                // Python: `try: nums.append(int(v))`. `int("3.14")` raises
                // and Rust's plain `parse::<i64>` differs from `int()` on
                // edges like leading whitespace and PEP 515 underscores.
                // `python_int_parse` mirrors Python exactly — see fn doc.
                if let Some(parsed) = python_int_parse(s) {
                    nums.push(parsed as f64);
                    // BUG #2 fix: do NOT set had_non_string_numeric.
                    // If we later find this is the ONLY source of numeric
                    // values, we refuse to call it sequential.
                }
            }
            _ => {}
        }
    }

    if nums.len() < 5 {
        return false;
    }

    // BUG #2 fix gate: if every numeric value originated as a string,
    // the field is categorical (e.g. zero-padded codes); not sequential.
    if !had_non_string_numeric {
        return false;
    }

    // Need at least 2 elements for pairwise comparison. (Python checks
    // this redundantly after `len(nums) < 5`.)
    if nums.len() < 2 {
        return false;
    }

    // Sort and compute pairwise diffs.
    let mut sorted_nums = nums.clone();
    sorted_nums.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let diffs: Vec<f64> = sorted_nums.windows(2).map(|w| w[1] - w[0]).collect();
    if diffs.is_empty() {
        return false;
    }

    let avg_diff: f64 = diffs.iter().sum::<f64>() / diffs.len() as f64;
    if !(0.5..=2.0).contains(&avg_diff) {
        return false;
    }

    // Most diffs in [0.5, 2.0] => sequential candidate.
    let consistent_count = diffs.iter().filter(|&&d| (0.5..=2.0).contains(&d)).count();
    let is_sequential = consistent_count as f64 / diffs.len() as f64 > 0.8;
    if !is_sequential {
        return false;
    }

    if check_order {
        // Python: ascending count over original (not-sorted) sequence.
        // IDs ascend in array order; scores typically descend.
        let ascending_count = nums.windows(2).filter(|w| w[0] <= w[1]).count();
        let n_pairs = nums.len() - 1;
        let is_ascending = ascending_count as f64 / n_pairs as f64 > 0.7;
        return is_ascending;
    }

    is_sequential
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ---------- is_uuid_format ----------

    #[test]
    fn uuid_format_canonical_lowercase() {
        assert!(is_uuid_format("550e8400-e29b-41d4-a716-446655440000"));
    }

    #[test]
    fn uuid_format_uppercase() {
        assert!(is_uuid_format("550E8400-E29B-41D4-A716-446655440000"));
    }

    #[test]
    fn uuid_format_wrong_length_rejected() {
        assert!(!is_uuid_format("550e8400-e29b-41d4-a716-44665544000")); // 1 short
        assert!(!is_uuid_format("550e8400-e29b-41d4-a716-4466554400000")); // 1 long
    }

    #[test]
    fn uuid_format_wrong_segment_count() {
        assert!(!is_uuid_format("550e8400e29b41d4a716446655440000"));
    }

    #[test]
    fn uuid_format_non_hex_rejected() {
        assert!(!is_uuid_format("550e8400-e29b-41d4-a716-44665544000z"));
    }

    #[test]
    fn uuid_format_empty_rejected() {
        assert!(!is_uuid_format(""));
    }

    // ---------- calculate_string_entropy ----------

    #[test]
    fn entropy_empty_string_is_zero() {
        assert_eq!(calculate_string_entropy(""), 0.0);
    }

    #[test]
    fn entropy_single_char_is_zero() {
        assert_eq!(calculate_string_entropy("a"), 0.0);
    }

    #[test]
    fn entropy_all_same_chars_is_zero() {
        // "aaaa" — freq has 1 entry, max_entropy = log2(1) = 0.0,
        // we return 0.0 from the guard.
        assert_eq!(calculate_string_entropy("aaaa"), 0.0);
    }

    #[test]
    fn entropy_perfectly_uniform_normalized_to_one() {
        // Two distinct chars, 50/50: raw entropy = 1.0, max = log2(2) = 1.0,
        // normalized = 1.0.
        let e = calculate_string_entropy("ab");
        assert!((e - 1.0).abs() < 1e-9);
    }

    #[test]
    fn entropy_mostly_repeated_low() {
        // "aaaaaab" — 6/7 'a', 1/7 'b' — should be small.
        let e = calculate_string_entropy("aaaaaab");
        assert!(e < 0.7);
    }

    #[test]
    fn entropy_high_for_random_looking_string() {
        // Approximation of a UUID-ish hex string. Should be > 0.7.
        let e = calculate_string_entropy("a3f7b2c9d8e1f4a7");
        assert!(e > 0.7);
    }

    // ---------- detect_sequential_pattern ----------

    #[test]
    fn sequential_simple_int_ascending() {
        let v: Vec<Value> = (1..=10).map(|i| json!(i)).collect();
        assert!(detect_sequential_pattern(&v, true));
    }

    #[test]
    fn sequential_too_few_values() {
        let v = vec![json!(1), json!(2), json!(3)];
        assert!(!detect_sequential_pattern(&v, true));
    }

    #[test]
    fn sequential_random_numbers_not_detected() {
        let v: Vec<Value> = vec![
            json!(100),
            json!(2),
            json!(85),
            json!(7),
            json!(43),
            json!(17),
        ];
        assert!(!detect_sequential_pattern(&v, true));
    }

    #[test]
    fn sequential_descending_with_check_order_rejected() {
        // Descending sequence — if check_order=true, must NOT be flagged
        // as sequential (Python: scores descend, IDs ascend).
        let v: Vec<Value> = (1..=10).rev().map(|i| json!(i)).collect();
        assert!(!detect_sequential_pattern(&v, true));
    }

    #[test]
    fn sequential_descending_without_check_order_accepted() {
        let v: Vec<Value> = (1..=10).rev().map(|i| json!(i)).collect();
        assert!(detect_sequential_pattern(&v, false));
    }

    #[test]
    fn bug2_zero_padded_strings_no_longer_misclassified() {
        // BUG #2: ["001", "002", ..., "010"] — Python's original code
        // parsed each via int() and called this sequential. Fixed: every
        // numeric value here originated as a string, so we refuse.
        let v: Vec<Value> = (1..=10).map(|i| json!(format!("{:03}", i))).collect();
        assert!(
            !detect_sequential_pattern(&v, true),
            "BUG #2 fix: zero-padded string IDs must not be classified as sequential"
        );
    }

    #[test]
    fn bug2_mixed_string_and_int_still_detected() {
        // Sanity check the fix doesn't break the legitimate case: a
        // field that has BOTH genuine ints AND string-encoded ints
        // should still be detected (the unambiguous ints dominate the
        // signal).
        let v = vec![json!(1), json!(2), json!("3"), json!(4), json!(5), json!(6)];
        assert!(detect_sequential_pattern(&v, true));
    }

    #[test]
    fn sequential_bools_excluded() {
        // Python: `isinstance(v, int | float) and not isinstance(v, bool)`
        // — bools never count as numeric.
        let v = vec![
            json!(true),
            json!(false),
            json!(true),
            json!(false),
            json!(true),
            json!(false),
        ];
        assert!(!detect_sequential_pattern(&v, true));
    }

    #[test]
    fn sequential_floats_with_unit_step() {
        let v: Vec<Value> = (1..=10).map(|i| json!(i as f64)).collect();
        assert!(detect_sequential_pattern(&v, true));
    }

    #[test]
    fn sequential_fractional_unit_step() {
        // Floats with non-integer values but constant unit step. avg_diff
        // = 1.0, all diffs in [0.5, 2.0], should be sequential. (Suggestion
        // S6 in code review — pins float arithmetic doesn't drift.)
        let v: Vec<Value> = vec![json!(1.5), json!(2.5), json!(3.5), json!(4.5), json!(5.5)];
        assert!(detect_sequential_pattern(&v, true));
    }

    #[test]
    fn bug2_all_unparseable_strings_returns_false() {
        // S3 in code review: explicit test for the all-strings case where
        // none parse. Falls out of `nums.len() < 5` already, but pinning
        // the behavior protects against future refactors.
        let v: Vec<Value> = vec![
            json!("abc"),
            json!("def"),
            json!("ghi"),
            json!("jkl"),
            json!("mno"),
        ];
        assert!(!detect_sequential_pattern(&v, true));
    }

    #[test]
    fn bug2_single_int_among_strings_still_detects() {
        // S3 in code review: validates that the BUG #2 gate fires on
        // "ANY non-string numeric", not "majority". One real int among
        // string-encoded numerics should be enough to count as sequential.
        let v: Vec<Value> = vec![
            json!("001"),
            json!("002"),
            json!(3), // <-- the unambiguous numeric
            json!("004"),
            json!("005"),
            json!("006"),
        ];
        assert!(detect_sequential_pattern(&v, true));
    }

    // ---------- python_int_parse ----------

    #[test]
    fn python_int_parse_basic() {
        assert_eq!(python_int_parse("5"), Some(5));
        assert_eq!(python_int_parse("-5"), Some(-5));
        assert_eq!(python_int_parse("+5"), Some(5));
    }

    #[test]
    fn python_int_parse_strips_whitespace() {
        // Python: `int("  5  ") == 5`. Rust's plain parse fails on this.
        assert_eq!(python_int_parse("  5  "), Some(5));
        assert_eq!(python_int_parse("\t-3\n"), Some(-3));
    }

    #[test]
    fn python_int_parse_underscores() {
        // PEP 515 — Python: `int("3_000") == 3000`.
        assert_eq!(python_int_parse("3_000"), Some(3000));
        assert_eq!(python_int_parse("1_000_000"), Some(1_000_000));
    }

    #[test]
    fn python_int_parse_underscore_edge_cases_rejected() {
        // Python rejects these (raises ValueError); we mirror by
        // returning None.
        assert_eq!(python_int_parse("_5"), None);
        assert_eq!(python_int_parse("5_"), None);
        assert_eq!(python_int_parse("3__000"), None);
    }

    #[test]
    fn python_int_parse_rejects_floats() {
        // Python: `int("3.14")` raises. Mirror by returning None.
        assert_eq!(python_int_parse("3.14"), None);
    }

    #[test]
    fn python_int_parse_rejects_non_numeric() {
        assert_eq!(python_int_parse("abc"), None);
        assert_eq!(python_int_parse(""), None);
        assert_eq!(python_int_parse("   "), None);
    }

    #[test]
    fn sequential_with_whitespace_padded_strings_via_python_int_parse() {
        // I1 fix in code review: real fixtures may carry whitespace-padded
        // numeric strings. With the python_int_parse helper, mixed real-int
        // + whitespace-padded-string fields still detect correctly.
        let v: Vec<Value> = vec![
            json!(1),
            json!("  2  "),
            json!(3),
            json!(" 4 "),
            json!(5),
            json!(6),
        ];
        assert!(detect_sequential_pattern(&v, true));
    }
}
