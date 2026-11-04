//! `JsonMinifier` — a [`ReformatTransform`] that strips insignificant
//! whitespace from validated JSON without reserializing its values.
//!
//! Strings, escapes, number literals, key order and duplicate keys remain
//! byte-identical. No CCR involvement — reformat transforms never need it.
//!
//! [`ReformatTransform`]: crate::transforms::pipeline::traits::ReformatTransform

use crate::transforms::pipeline::traits::{ReformatOutput, ReformatTransform, TransformError};
use crate::transforms::ContentType;

const NAME: &str = "json_minifier";

/// Whitespace-stripping JSON minifier. Handles both arrays and objects
/// since `JsonObject` and `JsonArray` are both registered content types.
#[derive(Debug, Default, Clone, Copy)]
pub struct JsonMinifier;

impl JsonMinifier {
    /// Return a smaller data-equivalent JSON representation, or no rewrite.
    pub fn minify(&self, content: &str) -> Result<Option<String>, TransformError> {
        if content.trim().is_empty() {
            return Err(TransformError::skipped(NAME, "empty input"));
        }
        let _: &serde_json::value::RawValue = serde_json::from_str(content)
            .map_err(|e| TransformError::invalid_input(NAME, e.to_string()))?;
        let mut output: Option<String> = None;
        let mut segment_start = 0;
        let mut quoted = false;
        let mut escaped = false;
        for (index, byte) in content.bytes().enumerate() {
            if quoted {
                if escaped {
                    escaped = false;
                } else if byte == b'\\' {
                    escaped = true;
                } else if byte == b'"' {
                    quoted = false;
                }
            } else if byte == b'"' {
                quoted = true;
            } else if matches!(byte, b' ' | b'\n' | b'\r' | b'\t') {
                output
                    .get_or_insert_with(|| String::with_capacity(content.len() - 1))
                    .push_str(&content[segment_start..index]);
                segment_start = index + 1;
            }
        }
        if let Some(output) = &mut output {
            output.push_str(&content[segment_start..]);
        }
        Ok(output)
    }
}

impl ReformatTransform for JsonMinifier {
    fn name(&self) -> &'static str {
        NAME
    }

    fn applies_to(&self) -> &[ContentType] {
        // The detector folds both arrays and objects into `JsonArray`
        // (the umbrella tag for "JSON the structural layer recognized");
        // the minifier itself doesn't care about top-level shape.
        &[ContentType::JsonArray]
    }

    fn apply(&self, content: &str) -> Result<ReformatOutput, TransformError> {
        let output = self.minify(content)?.unwrap_or_else(|| content.to_string());
        Ok(ReformatOutput::from_lengths(content.len(), output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn minification_preserves_json_literals() {
        let input =
            r#" { "z": 123456789012345678901234567890, "a": 1.2300e+400, "z": "\u0061 \" b 🌍" } "#;
        let compact =
            r#"{"z":123456789012345678901234567890,"a":1.2300e+400,"z":"\u0061 \" b 🌍"}"#;
        assert_eq!(
            JsonMinifier.minify(input).unwrap().as_deref(),
            Some(compact)
        );
        assert_eq!(JsonMinifier.minify(compact).unwrap(), None);
    }

    #[test]
    fn pretty_object_minifies() {
        let pretty = "{\n  \"a\": 1,\n  \"b\": 2\n}";
        let r = JsonMinifier.apply(pretty).expect("parses");
        assert_eq!(r.output, r#"{"a":1,"b":2}"#);
        assert!(r.bytes_saved > 0);
    }

    #[test]
    fn pretty_array_minifies() {
        let pretty = "[\n  1,\n  2,\n  3\n]";
        let r = JsonMinifier.apply(pretty).expect("parses");
        assert_eq!(r.output, "[1,2,3]");
        assert!(r.bytes_saved > 0);
    }

    #[test]
    fn already_compact_yields_zero_savings() {
        let compact = r#"{"a":1,"b":2}"#;
        let r = JsonMinifier.apply(compact).expect("parses");
        assert_eq!(r.output, compact);
        assert_eq!(r.bytes_saved, 0);
    }

    #[test]
    fn invalid_json_errors_with_invalid_input() {
        let bad = "{not: valid";
        let err = JsonMinifier.apply(bad).expect_err("must fail");
        match err {
            TransformError::InvalidInput { transform, .. } => {
                assert_eq!(transform, "json_minifier")
            }
            _ => panic!("expected InvalidInput, got {err:?}"),
        }
    }

    #[test]
    fn empty_input_skipped() {
        let err = JsonMinifier.apply("").expect_err("empty must skip");
        match err {
            TransformError::Skipped { transform, .. } => assert_eq!(transform, "json_minifier"),
            _ => panic!("expected Skipped, got {err:?}"),
        }
    }

    #[test]
    fn whitespace_only_skipped() {
        let err = JsonMinifier
            .apply("   \n\t  ")
            .expect_err("ws-only must skip");
        match err {
            TransformError::Skipped { .. } => {}
            _ => panic!("expected Skipped"),
        }
    }

    #[test]
    fn nested_structure_round_trips_semantically() {
        let pretty = r#"
        {
          "users": [
            {"id": 1, "name": "alice", "active": true},
            {"id": 2, "name": "bob",   "active": false}
          ],
          "count": 2
        }
        "#;
        let r = JsonMinifier.apply(pretty).expect("parses");
        // Re-parse the output and verify structural equivalence.
        let original_val: serde_json::Value = serde_json::from_str(pretty).unwrap();
        let output_val: serde_json::Value = serde_json::from_str(&r.output).unwrap();
        assert_eq!(original_val, output_val);
        assert!(r.bytes_saved > 0);
    }

    #[test]
    fn minifier_never_grows_output() {
        // Defensive contract: even if a caller hands us compact JSON
        // with embedded escaped strings that re-emit longer, we hand
        // the original back rather than inflating.
        let inputs = [
            r#"{}"#,
            r#"[]"#,
            r#"null"#,
            r#"42"#,
            r#""string""#,
            r#"{"k":"value with spaces"}"#,
        ];
        for input in inputs {
            let r = JsonMinifier.apply(input).expect("valid");
            assert!(
                r.output.len() <= input.len(),
                "minifier grew output for {input:?}: {} -> {}",
                input.len(),
                r.output.len()
            );
        }
    }

    #[test]
    fn unicode_round_trips() {
        let pretty = r#"{ "msg": "héllo 🌍 wörld" }"#;
        let r = JsonMinifier.apply(pretty).expect("parses");
        let v: serde_json::Value = serde_json::from_str(&r.output).unwrap();
        assert_eq!(v["msg"], "héllo 🌍 wörld");
    }
}
