//! Vertex publisher envelope parser.
//!
//! Vertex's Anthropic publisher path takes a body shaped almost
//! identically to the Anthropic Messages API, with two differences:
//!
//! 1. The body MUST contain `anthropic_version` (e.g. `"vertex-2023-10-16"`).
//! 2. The body MUST NOT contain `model` — the model id travels in
//!    the URL path.
//!
//! We treat any other shape as wire-format drift: log loudly and
//! forward unmodified. This module's only job is to detect/confirm
//! the envelope shape so the handler decides whether the live-zone
//! Anthropic dispatcher should run. We never strip or rewrite
//! `anthropic_version`.
//!
//! # Performance note
//!
//! The envelope check uses `serde_json::from_slice::<Value>` once.
//! That's the same cost the live-zone dispatcher already pays; we're
//! not adding a parse, we're hoisting one read of two top-level keys
//! up before dispatching. No body clone — the parsed `Value` is
//! discarded and the dispatcher re-parses for byte-faithful surgery
//! (which uses `RawValue` to preserve exact bytes).

use serde_json::Value;
use thiserror::Error;

/// Errors detecting the Vertex envelope. None are panics; the handler
/// surfaces these as structured log events plus an HTTP error response.
#[derive(Debug, Error)]
pub enum VertexEnvelopeError {
    #[error("body is not valid JSON: {0}")]
    NotJson(#[from] serde_json::Error),
    #[error("body is not a JSON object")]
    NotObject,
    #[error("body missing required field `anthropic_version`")]
    MissingAnthropicVersion,
    #[error(
        "body has unexpected `model` field; Vertex carries the model in the URL path \
         (got model={got:?})"
    )]
    UnexpectedModelField { got: String },
}

/// Parsed-once view of the envelope's two distinguishing fields. The
/// dispatcher does not read the rest of the body through this struct —
/// it re-parses with `RawValue` for byte-faithful surgery.
#[derive(Debug, Clone)]
pub struct ParsedEnvelope {
    /// Value of the `anthropic_version` field as it appeared in the
    /// body. We do NOT validate against an allowlist — Vertex may
    /// roll new versions without our involvement.
    pub anthropic_version: String,
    /// `true` if the body has at least one `messages[*]` entry.
    /// Diagnostic only; the dispatcher walks `messages` independently.
    pub has_messages: bool,
}

/// Parse and validate the envelope. On success returns the
/// distinguishing-field view. On failure returns a structured error
/// the handler converts to a log event + HTTP response.
pub fn parse(body: &[u8]) -> Result<ParsedEnvelope, VertexEnvelopeError> {
    let parsed: Value = serde_json::from_slice(body)?;
    let obj = match parsed {
        Value::Object(map) => map,
        _ => return Err(VertexEnvelopeError::NotObject),
    };

    let anthropic_version = match obj.get("anthropic_version") {
        Some(Value::String(s)) => s.clone(),
        Some(other) => {
            // Wire format says string; surface drift loudly. We still
            // accept by stringifying so we don't reject novel
            // representations Vertex might roll out — but log a
            // structured event upstream of this call so operators
            // see it. The handler is responsible for the warn log.
            other.to_string()
        }
        None => return Err(VertexEnvelopeError::MissingAnthropicVersion),
    };

    if let Some(model) = obj.get("model") {
        let got = match model {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        return Err(VertexEnvelopeError::UnexpectedModelField { got });
    }

    let has_messages = matches!(obj.get("messages"), Some(Value::Array(arr)) if !arr.is_empty());

    Ok(ParsedEnvelope {
        anthropic_version,
        has_messages,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_minimal_envelope() {
        let body = json!({
            "anthropic_version": "vertex-2023-10-16",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 64,
        });
        let bytes = serde_json::to_vec(&body).unwrap();
        let env = parse(&bytes).expect("parse");
        assert_eq!(env.anthropic_version, "vertex-2023-10-16");
        assert!(env.has_messages);
    }

    #[test]
    fn rejects_missing_anthropic_version() {
        let body = json!({
            "messages": [{"role": "user", "content": "hi"}],
        });
        let bytes = serde_json::to_vec(&body).unwrap();
        let err = parse(&bytes).unwrap_err();
        assert!(matches!(err, VertexEnvelopeError::MissingAnthropicVersion));
    }

    #[test]
    fn rejects_model_field_present() {
        let body = json!({
            "anthropic_version": "vertex-2023-10-16",
            "model": "claude-3-5-sonnet",
            "messages": [],
        });
        let bytes = serde_json::to_vec(&body).unwrap();
        let err = parse(&bytes).unwrap_err();
        match err {
            VertexEnvelopeError::UnexpectedModelField { got } => {
                assert_eq!(got, "claude-3-5-sonnet");
            }
            other => panic!("wrong error: {other:?}"),
        }
    }

    #[test]
    fn rejects_non_object_body() {
        let bytes = b"[1,2,3]";
        let err = parse(bytes).unwrap_err();
        assert!(matches!(err, VertexEnvelopeError::NotObject));
    }

    #[test]
    fn rejects_invalid_json() {
        let bytes = b"not json at all {{";
        let err = parse(bytes).unwrap_err();
        assert!(matches!(err, VertexEnvelopeError::NotJson(_)));
    }

    #[test]
    fn empty_messages_array_marks_has_messages_false() {
        let body = json!({
            "anthropic_version": "vertex-2023-10-16",
            "messages": [],
        });
        let bytes = serde_json::to_vec(&body).unwrap();
        let env = parse(&bytes).unwrap();
        assert!(!env.has_messages);
    }
}
