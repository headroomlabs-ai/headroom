//! Legacy regex-based query anchor extraction.
//!
//! Direct port of `extract_query_anchors` and `item_matches_anchors`
//! (`smart_crusher.py:99-168`). The Python doc-comment marks both as
//! DEPRECATED in favor of `RelevanceScorer`, but they're still called
//! by the live SmartCrusher path on every invocation, so we port them
//! faithfully.
//!
//! # Why regex parity matters
//!
//! These regexes drive which array items survive compression. A subtle
//! difference between Python's `re` engine and Rust's `regex` crate
//! (e.g. word-boundary behavior on Unicode, or repetition greediness)
//! would silently change which anchors are detected and which items
//! survive. The patterns below are pinned to lowercase ASCII inputs
//! and use only ASCII-safe constructs to keep behavior identical.

use regex::Regex;
use serde_json::Value;
use std::collections::HashSet;
use std::sync::LazyLock;

// ---------------------------------------------------------------
// Pattern definitions — direct ports of the module-level Python regexes
// at `smart_crusher.py:85-93`. `std::sync::LazyLock` (stable since Rust
// 1.80) is the modern equivalent of `once_cell::sync::Lazy`, mirroring
// Python's `re.compile` at module import time.
// ---------------------------------------------------------------

/// `\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b`
static UUID_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
        .expect("UUID_PATTERN")
});

/// 4+ digit numbers (likely IDs). Python: `r"\b\d{4,}\b"`.
static NUMERIC_ID_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b\d{4,}\b").expect("NUMERIC_ID_PATTERN"));

/// Hostname pattern. Matches `host.tld` with optional `.tld2`. Python:
/// `r"\b[a-zA-Z0-9][-a-zA-Z0-9]*\.[a-zA-Z0-9][-a-zA-Z0-9]*(?:\.[a-zA-Z]{2,})?\b"`.
static HOSTNAME_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b[a-zA-Z0-9][-a-zA-Z0-9]*\.[a-zA-Z0-9][-a-zA-Z0-9]*(?:\.[a-zA-Z]{2,})?\b")
        .expect("HOSTNAME_PATTERN")
});

/// Short quoted strings (single OR double quotes), 1-50 chars between
/// quotes. Python: `r"['\"]([^'\"]{1,50})['\"]"`.
static QUOTED_STRING_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"['"]([^'"]{1,50})['"]"#).expect("QUOTED_STRING_PATTERN"));

/// Email addresses. Python: `r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"`.
/// (Note Python's `[A-Z|a-z]` includes a literal `|` in the character
/// class — almost certainly a typo, but we faithfully port it for
/// parity. Real-world impact is nil since `|` doesn't appear in TLDs.)
static EMAIL_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b").expect("EMAIL_PATTERN")
});

/// Hostname false-positive blocklist. Python uses a set literal at
/// `smart_crusher.py:137`. We mirror exactly — these strings get
/// dropped from anchor results.
const HOSTNAME_FALSE_POSITIVES: &[&str] = &["e.g", "i.e", "etc."];

/// Extract query anchors from user text. **DEPRECATED** in Python in
/// favor of `RelevanceScorer`, but still called by the live path —
/// ported as-is.
///
/// Output is a set of lowercased anchor strings. Order is not
/// significant (Python returns `set[str]`).
pub fn extract_query_anchors(text: &str) -> HashSet<String> {
    let mut anchors = HashSet::new();

    if text.is_empty() {
        return anchors;
    }

    // UUIDs — lowercase the match.
    for m in UUID_PATTERN.find_iter(text) {
        anchors.insert(m.as_str().to_lowercase());
    }

    // Numeric IDs — Python keeps original case (digits, no transform needed).
    for m in NUMERIC_ID_PATTERN.find_iter(text) {
        anchors.insert(m.as_str().to_string());
    }

    // Hostnames — lowercase, filter false positives.
    for m in HOSTNAME_PATTERN.find_iter(text) {
        let lc = m.as_str().to_lowercase();
        if !HOSTNAME_FALSE_POSITIVES.contains(&lc.as_str()) {
            anchors.insert(lc);
        }
    }

    // Quoted strings — capture group 1 (the content between quotes),
    // require trim().len() >= 2 (Python's `if len(match.strip()) >= 2`).
    for caps in QUOTED_STRING_PATTERN.captures_iter(text) {
        if let Some(inner) = caps.get(1) {
            if inner.as_str().trim().len() >= 2 {
                anchors.insert(inner.as_str().to_lowercase());
            }
        }
    }

    // Emails — lowercase.
    for m in EMAIL_PATTERN.find_iter(text) {
        anchors.insert(m.as_str().to_lowercase());
    }

    anchors
}

/// Serialize a `serde_json::Value` to a string matching Python's
/// `str()` of the equivalent native value.
///
/// Used by `item_matches_anchors` because Python compares anchors via
/// `anchor in str(item).lower()` and `str(dict)` differs from
/// `json.dumps(dict)` in three ways that affect substring matching:
///
/// | Aspect           | Python `str(dict)`           | `serde_json::to_string` |
/// |------------------|------------------------------|-------------------------|
/// | String quotes    | single `'`                   | double `"`              |
/// | Booleans / null  | `True`, `False`, `None`      | `true`, `false`, `null` |
/// | Spacing          | `key: value`, `a, b`         | `key:value`, `a,b`      |
///
/// All three matter for anchor matching:
/// - An anchor `"name': 'a"` extracted from a user phrase like
///   `find {'name': 'alice'}` would match Python's serialization but
///   never the JSON form.
/// - An anchor `"true"` (lowercased from `"True"`) matches both, but
///   the unlowercased version `"True"` is in Python output and not
///   JSON. Lowercasing both sides handles this.
/// - An anchor `"name: alice"` (with the space) would match Python
///   but never JSON.
///
/// Output is then lowercased upstream (matching Python's `.lower()`)
/// so True/False/None case is normalized away after that step.
fn python_repr(value: &Value) -> String {
    let mut out = String::new();
    write_python_repr(&mut out, value);
    out
}

fn write_python_repr(out: &mut String, value: &Value) {
    match value {
        Value::Null => out.push_str("None"),
        Value::Bool(true) => out.push_str("True"),
        Value::Bool(false) => out.push_str("False"),
        Value::Number(n) => {
            // Python `str(int)` and `str(float)` produce minimal forms.
            // `serde_json::Number`'s `Display` matches Python for ints
            // (`5`) but for floats it can write `1.0` while Python may
            // write `1.0` too — close enough for substring matching
            // since anchor strings rarely contain numeric literals
            // beyond the digit prefix.
            out.push_str(&n.to_string());
        }
        Value::String(s) => {
            // Python `repr(s)` chooses single or double quotes
            // depending on content. Default preference is single
            // quotes; switches to double if the string contains a
            // single quote and no double. We emit single quotes
            // always — this matches the dominant case (no quotes in
            // the string) and is what Python does for `str(dict)` of
            // most realistic data. The rare case where Python would
            // switch to double quotes is documented as a known parity
            // gap in `python_repr_string_with_single_quote_drift`.
            out.push('\'');
            out.push_str(s);
            out.push('\'');
        }
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_python_repr(out, item);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            // Python preserves insertion order in `dict.__str__` (since
            // Python 3.7). We require the workspace `serde_json` to be
            // built with `preserve_order` so `serde_json::Map` uses
            // `IndexMap` instead of the default `BTreeMap` — see the
            // comment on `serde_json` in the workspace `Cargo.toml`.
            // Without that feature, this iteration is sorted-by-key
            // and silently diverges from Python on every multi-key
            // object.
            for (i, (k, v)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push('\'');
                out.push_str(k);
                out.push('\'');
                out.push_str(": ");
                write_python_repr(out, v);
            }
            out.push('}');
        }
    }
}

/// Check if a JSON value matches any query anchors.
///
/// Direct port of `item_matches_anchors` (Python `smart_crusher.py:152-168`).
/// Python uses `str(item).lower()` which produces Python's repr-like
/// representation. We mirror that via `python_repr` rather than
/// `serde_json::to_string` so substring matching has the same surface
/// as Python (single quotes, `True`/`False`/`None`, spaced commas/colons).
pub fn item_matches_anchors(item: &Value, anchors: &HashSet<String>) -> bool {
    if anchors.is_empty() {
        return false;
    }

    // Python: `str(item).lower()`. `python_repr` produces the same
    // single-quoted, space-after-colon, `True`/`False`/`None` form
    // that Python's `str()` does; lowercase normalizes the bool/null
    // case to match Python's downstream `.lower()` call.
    let item_str = python_repr(item).to_lowercase();
    anchors.iter().any(|a| item_str.contains(a))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn empty_text_no_anchors() {
        assert!(extract_query_anchors("").is_empty());
    }

    #[test]
    fn extracts_uuid_lowercased() {
        let anchors = extract_query_anchors("see id 550E8400-E29B-41D4-A716-446655440000 plz");
        assert!(anchors.contains("550e8400-e29b-41d4-a716-446655440000"));
    }

    #[test]
    fn extracts_numeric_id_unchanged() {
        let anchors = extract_query_anchors("user 12345 reported issue");
        assert!(anchors.contains("12345"));
    }

    #[test]
    fn three_digit_number_not_anchor() {
        // Pattern requires 4+ digits.
        let anchors = extract_query_anchors("user 123 reported issue");
        assert!(!anchors.iter().any(|a| a == "123"));
    }

    #[test]
    fn extracts_hostname() {
        let anchors = extract_query_anchors("connect to api.example.com asap");
        assert!(anchors.contains("api.example.com"));
    }

    #[test]
    fn hostname_false_positive_filtered() {
        // "e.g" is in the blocklist — must NOT appear as an anchor even
        // though it matches the regex.
        let anchors = extract_query_anchors("test e.g.com endpoint");
        // "e.g" is filtered, but "e.g.com" or other longer matches may
        // pass; we only assert "e.g" itself is gone.
        assert!(!anchors.contains("e.g"));
    }

    #[test]
    fn extracts_quoted_string_double() {
        let anchors = extract_query_anchors(r#"find the "user_name" field"#);
        assert!(anchors.contains("user_name"));
    }

    #[test]
    fn extracts_quoted_string_single() {
        let anchors = extract_query_anchors("find the 'user_name' field");
        assert!(anchors.contains("user_name"));
    }

    #[test]
    fn very_short_quoted_skipped() {
        // Less than 2 chars after trim — skipped.
        let anchors = extract_query_anchors(r#"the "x" thing"#);
        assert!(!anchors.contains("x"));
    }

    #[test]
    fn extracts_email() {
        let anchors = extract_query_anchors("contact USER@example.COM please");
        assert!(anchors.contains("user@example.com"));
    }

    #[test]
    fn item_matches_anchors_empty_set() {
        let empty = HashSet::new();
        assert!(!item_matches_anchors(&json!({"a": 1}), &empty));
    }

    #[test]
    fn item_matches_anchor_in_value() {
        let anchors: HashSet<String> = ["alice".to_string()].into_iter().collect();
        assert!(item_matches_anchors(&json!({"name": "Alice"}), &anchors));
    }

    #[test]
    fn item_matches_anchor_in_key() {
        let anchors: HashSet<String> = ["status".to_string()].into_iter().collect();
        // The anchor "status" appears in the JSON-serialized key.
        assert!(item_matches_anchors(&json!({"status": "ok"}), &anchors));
    }

    #[test]
    fn item_no_match_with_unrelated_anchor() {
        let anchors: HashSet<String> = ["xyz123".to_string()].into_iter().collect();
        assert!(!item_matches_anchors(&json!({"a": "b"}), &anchors));
    }

    #[test]
    fn hostname_blocklist_drops_e_g() {
        // S5 in code review: pin that "e.g" in input doesn't surface as
        // an anchor. Direct match against the regex confirms "e.g" itself
        // matches before the blocklist filters it.
        let anchors = extract_query_anchors("see e.g for example");
        assert!(!anchors.contains("e.g"));
        // Sanity: a normal hostname still passes through.
        let anchors = extract_query_anchors("connect to api.example.com");
        assert!(anchors.contains("api.example.com"));
    }

    #[test]
    fn email_typo_pattern_still_matches_real_emails() {
        // S4 in code review: the Python `[A-Z|a-z]` typo doesn't break
        // real email matching — pin that explicitly.
        let anchors = extract_query_anchors("contact alice@example.com today");
        assert!(anchors.contains("alice@example.com"));
        let anchors = extract_query_anchors("ping bob@SUB.EXAMPLE.IO");
        assert!(anchors.contains("bob@sub.example.io"));
    }

    // ---------- python_repr (used by item_matches_anchors) ----------

    #[test]
    fn python_repr_matches_python_str_for_dict() {
        // Python: `str({'name': 'Alice', 'ok': True, 'count': 5, 'val': None})`
        // = `"{'name': 'Alice', 'ok': True, 'count': 5, 'val': None}"`
        // (insertion order — Python's dict preserves it since 3.7).
        //
        // Workspace `Cargo.toml` enables serde_json's `preserve_order`
        // feature, so `json!` macro and `serde_json::from_str` both
        // preserve key insertion order. Without that feature the test
        // below would fail.
        let v = json!({"name": "Alice", "ok": true, "count": 5, "val": null});
        let r = python_repr(&v);
        assert_eq!(r, "{'name': 'Alice', 'ok': True, 'count': 5, 'val': None}");
    }

    #[test]
    fn python_repr_list_uses_space_after_comma() {
        // Python: `str([1, 2, 'abc', True])` = `"[1, 2, 'abc', True]"`.
        let v = json!([1, 2, "abc", true]);
        assert_eq!(python_repr(&v), "[1, 2, 'abc', True]");
    }

    #[test]
    fn python_repr_nested() {
        let v = json!({"a": [1, {"b": "c"}]});
        assert_eq!(python_repr(&v), "{'a': [1, {'b': 'c'}]}");
    }

    #[test]
    fn item_matches_anchor_with_python_none_form() {
        // I3 fix in review: Python `str({'val': None}).lower()` produces
        // `{'val': none}`. With the old JSON-based matcher, the same
        // input would serialize as `{"val":null}` and an anchor "none"
        // would never match. With `python_repr` the serialization is
        // `{'val': None}` → lowercased to `{'val': none}` → contains "none".
        let anchors: HashSet<String> = ["none".to_string()].into_iter().collect();
        assert!(item_matches_anchors(&json!({"val": null}), &anchors));
    }

    #[test]
    fn item_matches_anchor_avoids_json_null_token() {
        // Inverse of the above: an anchor "null" must NOT match a Python-
        // null repr (which writes `none`). Pre-fix code would erroneously
        // match because of `serde_json::to_string`'s `null` literal.
        let anchors: HashSet<String> = ["null".to_string()].into_iter().collect();
        assert!(!item_matches_anchors(&json!({"val": null}), &anchors));
    }

    #[test]
    fn python_repr_string_with_single_quote_drift() {
        // Documented parity gap: Python's `repr` switches to double
        // quotes if the string contains a single quote. We always use
        // single quotes. Pin the gap so future changes are intentional.
        let v = json!({"k": "it's fine"});
        // Our output: `{'k': 'it's fine'}` (broken Python repr — Python
        // would emit `{'k': "it's fine"}`).
        assert_eq!(python_repr(&v), "{'k': 'it's fine'}");
        // Substring matching for typical anchors still works because
        // they don't reference the quote chars themselves.
    }
}
