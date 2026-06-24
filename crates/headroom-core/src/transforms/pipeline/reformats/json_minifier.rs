//! `JsonMinifier` — a [`ReformatTransform`] that strips insignificant
//! whitespace from JSON via `serde_json` round-trip.
//!
//! Lossless by construction: parse to `serde_json::Value`, re-emit
//! compactly, return whichever is shorter. No CCR involvement —
//! reformat transforms never need it.
//!
//! [`ReformatTransform`]: crate::transforms::pipeline::traits::ReformatTransform

use crate::transforms::pipeline::traits::{ReformatOutput, ReformatTransform, TransformError};
use crate::transforms::ContentType;

const NAME: &str = "json_minifier";

/// Whitespace-stripping JSON minifier. Handles both arrays and objects
/// since `JsonObject` and `JsonArray` are both registered content types.
#[derive(Debug, Default, Clone, Copy)]
pub struct JsonMinifier;

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
        let trimmed = content.trim();
        if trimmed.is_empty() {
            return Err(TransformError::skipped(NAME, "empty input"));
        }

        let value: serde_json::Value = serde_json::from_str(trimmed)
            .map_err(|e| TransformError::invalid_input(NAME, e.to_string()))?;

        let minified = serde_json::to_string(&value)
            .map_err(|e| TransformError::internal(NAME, e.to_string()))?;

        // Defensive: if minification grew the byte count (e.g. caller
        // already passed compact JSON, or escaping rules added bytes),
        // hand the original back so we never inflate the wire output.
        if minified.len() >= content.len() {
            return Ok(ReformatOutput::from_lengths(
                content.len(),
                content.to_string(),
            ));
        }

        Ok(ReformatOutput::from_lengths(content.len(), minified))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_and_applies_to() {
        let m = JsonMinifier;
        assert_eq!(m.name(), "json_minifier");
        assert_eq!(m.applies_to(), &[ContentType::JsonArray]);
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
