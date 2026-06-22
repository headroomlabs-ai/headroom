//! Per-cell classification for the compaction pipeline.
//!
//! Given a JSON value, decide what kind of compaction treatment it needs.
//! The classifier is intentionally conservative — when in doubt, return
//! [`CellClass::Scalar`] so the cell is rendered verbatim.
//!
//! # Detection priorities
//!
//! 1. **Object / array** — pass through to caller, who decides whether to
//!    flatten (uniform-nested) or recurse ([`CellClass::JsonObject`],
//!    [`CellClass::JsonArray`]).
//! 2. **Stringified-JSON** — strings that parse to a JSON object/array.
//!    Common in tool-output payloads where one field is a serialized
//!    sub-structure ([`CellClass::StringifiedJson`]).
//! 3. **Opaque blob** — strings above a length threshold the classifier
//!    couldn't otherwise place. Sub-classified into base64 / HTML /
//!    plain long-string for telemetry ([`CellClass::Opaque`]).
//! 4. **Scalar** — everything else, rendered verbatim.

use serde_json::Value;

use super::ir::OpaqueKind;

/// Per-cell classification result.
#[derive(Debug, Clone, PartialEq)]
pub enum CellClass {
    /// Number, bool, null, short string — render verbatim.
    Scalar,
    /// Cell is a JSON object. Caller decides flatten-vs-recurse based
    /// on schema uniformity across rows.
    JsonObject,
    /// Cell is a JSON array. Caller may recurse with TabularCompactor.
    JsonArray,
    /// String that parses to a JSON object/array. The parsed value is
    /// returned so the caller doesn't re-parse.
    StringifiedJson(Value),
    /// Long string the classifier judged opaque. Sub-classified for
    /// telemetry only — all variants get CCR-substituted.
    Opaque(OpaqueKind),
}

/// Config controlling classification thresholds.
///
/// Defaults are tuned for typical tool-output payloads. Override via
/// builder if a workload has different characteristics (e.g. an API
/// that always emits 500-char status descriptions shouldn't have those
/// CCR-substituted).
#[derive(Debug, Clone)]
pub struct ClassifyConfig {
    /// Strings strictly longer than this become candidates for opaque
    /// classification. Default: 256 bytes.
    pub opaque_min_bytes: usize,
    /// Base64-alphabet ratio threshold. Strings whose chars are at
    /// least this fraction in `[A-Za-z0-9+/=_-]` and longer than 64
    /// bytes are tagged base64. Default: 0.95.
    pub base64_alphabet_ratio: f64,
    /// `<` count above which a long string is considered HTML-ish.
    /// Default: 3.
    pub html_min_open_brackets: usize,
    /// When false, long strings are NOT classified as opaque — they stay
    /// `Scalar` and render verbatim, so output is marker-free and
    /// guaranteed-lossless. Mirrors the row-drop path's `enable_ccr_marker`
    /// gate (see `crusher.rs`). Default: true.
    pub emit_opaque_markers: bool,
}

impl Default for ClassifyConfig {
    fn default() -> Self {
        Self {
            opaque_min_bytes: 256,
            base64_alphabet_ratio: 0.95,
            html_min_open_brackets: 3,
            emit_opaque_markers: true,
        }
    }
}

/// Classify a single cell value.
pub fn classify_cell(value: &Value, cfg: &ClassifyConfig) -> CellClass {
    match value {
        Value::Object(_) => CellClass::JsonObject,
        Value::Array(_) => CellClass::JsonArray,
        Value::String(s) => classify_string(s, cfg),
        _ => CellClass::Scalar,
    }
}

fn classify_string(s: &str, cfg: &ClassifyConfig) -> CellClass {
    // Stringified-JSON check first. Cheap fast-path: must start with
    // `{` or `[` (after optional whitespace) — skip strings that
    // can't possibly be JSON containers. Parsing `"123"` would
    // technically succeed as JSON-the-number, but that's a scalar,
    // not a recursion target.
    let trimmed = s.trim_start();
    if matches!(trimmed.chars().next(), Some('{') | Some('[')) {
        if let Ok(parsed) = serde_json::from_str::<Value>(s) {
            if matches!(parsed, Value::Object(_) | Value::Array(_)) {
                return CellClass::StringifiedJson(parsed);
            }
        }
    }

    // Opaque-blob check — only for strings above the byte threshold, and
    // only when opaque markers are enabled. With markers off, keep the full
    // string verbatim (Scalar) so the output stays lossless and marker-free.
    if s.len() <= cfg.opaque_min_bytes || !cfg.emit_opaque_markers {
        return CellClass::Scalar;
    }

    if looks_like_base64(s, cfg.base64_alphabet_ratio) {
        return CellClass::Opaque(OpaqueKind::Base64Blob);
    }

    if looks_like_html(s, cfg.html_min_open_brackets) {
        return CellClass::Opaque(OpaqueKind::HtmlChunk);
    }

    CellClass::Opaque(OpaqueKind::LongString)
}

fn looks_like_base64(s: &str, ratio_threshold: f64) -> bool {
    if s.len() < 64 {
        return false;
    }
    // Disqualifying signals — these instantly rule out base64.
    if s.contains('<') || s.contains('>') {
        return false;
    }
    if s.chars().any(|c| c.is_whitespace()) {
        return false;
    }

    let total = s.len();
    let alphabet = s
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '/' | '=' | '_' | '-'))
        .count();
    if (alphabet as f64) / (total as f64) < ratio_threshold {
        return false;
    }

    // Diversity filter: real base64-encoded random bytes use most of
    // their 64-character alphabet. Strings with < 16 unique characters
    // are almost certainly not base64 (typical false-positive: brace-
    // wrapped repeated characters like `{xxxx...}`).
    let mut unique = std::collections::HashSet::new();
    for c in s.chars() {
        unique.insert(c);
        if unique.len() >= 16 {
            return true;
        }
    }
    false
}

fn looks_like_html(s: &str, min_open_brackets: usize) -> bool {
    let opens = s.chars().filter(|c| *c == '<').count();
    if opens < min_open_brackets {
        return false;
    }
    // Cheap signal: opens are followed by an alpha char or `/` reasonably
    // often. Avoids false-positives on math-heavy strings ("a < b").
    let bytes = s.as_bytes();
    let mut tag_starts = 0usize;
    for (i, b) in bytes.iter().enumerate() {
        if *b == b'<' {
            if let Some(next) = bytes.get(i + 1) {
                if next.is_ascii_alphabetic() || *next == b'/' || *next == b'!' {
                    tag_starts += 1;
                }
            }
        }
    }
    tag_starts >= min_open_brackets
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn cfg() -> ClassifyConfig {
        ClassifyConfig::default()
    }

    #[test]
    fn scalars_are_scalars() {
        assert_eq!(classify_cell(&json!(1), &cfg()), CellClass::Scalar);
        assert_eq!(classify_cell(&json!(1.5), &cfg()), CellClass::Scalar);
        assert_eq!(classify_cell(&json!(true), &cfg()), CellClass::Scalar);
        assert_eq!(classify_cell(&json!(null), &cfg()), CellClass::Scalar);
        assert_eq!(classify_cell(&json!("short"), &cfg()), CellClass::Scalar);
    }

    #[test]
    fn objects_and_arrays_pass_through() {
        assert_eq!(
            classify_cell(&json!({"a": 1}), &cfg()),
            CellClass::JsonObject
        );
        assert_eq!(classify_cell(&json!([1, 2]), &cfg()), CellClass::JsonArray);
    }

    #[test]
    fn stringified_json_object_is_parsed() {
        let v = json!(r#"{"x":1,"y":2}"#);
        match classify_cell(&v, &cfg()) {
            CellClass::StringifiedJson(parsed) => {
                assert_eq!(parsed, json!({"x": 1, "y": 2}));
            }
            other => panic!("expected StringifiedJson, got {other:?}"),
        }
    }

    #[test]
    fn stringified_json_array_is_parsed() {
        let v = json!(r#"[1,2,3]"#);
        match classify_cell(&v, &cfg()) {
            CellClass::StringifiedJson(parsed) => {
                assert_eq!(parsed, json!([1, 2, 3]));
            }
            other => panic!("expected StringifiedJson, got {other:?}"),
        }
    }

    #[test]
    fn stringified_scalar_is_not_recursed() {
        // "123" parses as a JSON number, but we don't recurse on scalars.
        assert_eq!(classify_cell(&json!("123"), &cfg()), CellClass::Scalar);
        // Same for booleans, nulls.
        assert_eq!(classify_cell(&json!("true"), &cfg()), CellClass::Scalar);
    }

    #[test]
    fn malformed_brace_string_is_long_opaque_or_scalar() {
        let short = json!("{not json}");
        assert_eq!(classify_cell(&short, &cfg()), CellClass::Scalar);
        let long = "{".to_string() + &"x".repeat(300) + "}";
        match classify_cell(&Value::String(long), &cfg()) {
            CellClass::Opaque(OpaqueKind::LongString) => {}
            other => panic!("expected LongString, got {other:?}"),
        }
    }

    #[test]
    fn long_string_stays_scalar_when_opaque_markers_disabled() {
        // #1091: with opaque markers disabled, a long string must NOT be
        // classified Opaque (which would emit a `<<ccr:>>` marker); it stays
        // Scalar and renders verbatim, so the output is lossless.
        let v = Value::String("x".repeat(512));
        // Default config classifies it Opaque.
        assert!(matches!(classify_cell(&v, &cfg()), CellClass::Opaque(_)));
        // Markers disabled → Scalar (verbatim).
        let no_markers = ClassifyConfig {
            emit_opaque_markers: false,
            ..ClassifyConfig::default()
        };
        assert_eq!(classify_cell(&v, &no_markers), CellClass::Scalar);
    }

    #[test]
    fn base64_blob_detected() {
        let s = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/==".repeat(5);
        match classify_cell(&Value::String(s), &cfg()) {
            CellClass::Opaque(OpaqueKind::Base64Blob) => {}
            other => panic!("expected Base64Blob, got {other:?}"),
        }
    }

    #[test]
    fn html_chunk_detected() {
        let s = "<html><body><p>".to_string() + &"x".repeat(300) + "</p></body></html>";
        match classify_cell(&Value::String(s), &cfg()) {
            CellClass::Opaque(OpaqueKind::HtmlChunk) => {}
            other => panic!("expected HtmlChunk, got {other:?}"),
        }
    }

    #[test]
    fn long_plain_string_is_long_opaque() {
        let s = "the quick brown fox ".repeat(20);
        match classify_cell(&Value::String(s), &cfg()) {
            CellClass::Opaque(OpaqueKind::LongString) => {}
            other => panic!("expected LongString, got {other:?}"),
        }
    }

    #[test]
    fn math_with_lt_is_not_html() {
        let s = "a < b but not really ".repeat(20);
        match classify_cell(&Value::String(s), &cfg()) {
            CellClass::Opaque(OpaqueKind::LongString) => {}
            other => panic!("expected LongString, got {other:?}"),
        }
    }

    #[test]
    fn config_threshold_respected() {
        let mut c = cfg();
        c.opaque_min_bytes = 10;
        let s = json!("hello world this is long");
        match classify_cell(&s, &c) {
            CellClass::Opaque(_) => {}
            other => panic!("expected Opaque under low threshold, got {other:?}"),
        }
    }
}
