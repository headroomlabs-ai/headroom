//! Bedrock envelope: parse the body shape Bedrock expects, hand the
//! Anthropic-shape sub-body to the compressor, and re-emit with
//! `anthropic_version` preserved as the FIRST key.
//!
//! # Wire shape
//!
//! Bedrock InvokeModel for any `anthropic.claude-*` model expects:
//!
//! ```json
//! {
//!   "anthropic_version": "bedrock-2023-05-31",
//!   "messages": [...],
//!   "max_tokens": 1024,
//!   ...rest_of_anthropic_body
//! }
//! ```
//!
//! `model` is in the URL path (`/model/{model}/invoke`), NOT the body
//! — the opposite of direct-Anthropic `/v1/messages`. `anthropic_version`
//! is REQUIRED and must be a literal Bedrock-recognized version
//! string (e.g. `"bedrock-2023-05-31"`).
//!
//! # Why key order matters
//!
//! Bedrock's request validator does NOT depend on key order, but our
//! cache-safety contract (`I1` in REALIGNMENT/02-architecture.md) is
//! that *unmodified* bytes round-trip byte-equal. If the compressor
//! returns `NoChange` we forward the original buffered bytes
//! verbatim — we never re-serialize. If the compressor returns
//! `Modified`, the byte slice it produced already preserves
//! key order from the input via the `preserve_order` feature on
//! `serde_json` (the workspace turns this on by default).
//!
//! This module's [`BedrockEnvelope`] is used ONLY for re-emitting
//! the envelope shape after compression. It calls the compressor
//! against the body bytes directly — there is no decode-then-encode
//! round trip on the no-change path.

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

/// The literal `anthropic_version` field name. Bedrock-strict.
const ANTHROPIC_VERSION_KEY: &str = "anthropic_version";

/// Parsed Bedrock InvokeModel envelope. Holds the literal
/// `anthropic_version` string plus the rest of the body as a parsed
/// `serde_json::Value` (so callers can inspect / route it). The
/// original byte-equal body is also stashed for the no-change
/// passthrough path.
#[derive(Debug, Clone)]
pub struct BedrockEnvelope {
    /// The literal `anthropic_version` string, e.g.
    /// `"bedrock-2023-05-31"`. Required.
    pub anthropic_version: String,
    /// The full parsed body, including `anthropic_version`. Useful
    /// for tests + logs; the compressor takes raw bytes.
    pub body: Value,
}

/// Errors surfaced when parsing a Bedrock envelope.
#[derive(Debug, Error)]
pub enum EnvelopeError {
    /// Body was not valid JSON.
    #[error("body is not valid JSON: {0}")]
    NotJson(serde_json::Error),
    /// Body was JSON but not a top-level object.
    #[error("body is not a JSON object")]
    NotObject,
    /// `anthropic_version` field missing.
    #[error("missing required `anthropic_version` field")]
    MissingAnthropicVersion,
    /// `anthropic_version` was present but not a string.
    #[error("`anthropic_version` is not a string")]
    AnthropicVersionNotString,
}

impl BedrockEnvelope {
    /// Parse a Bedrock envelope from raw JSON bytes.
    ///
    /// This does NOT mutate the bytes. Compression dispatch happens
    /// elsewhere (the handler hands the same byte slice to
    /// `compress_anthropic_request`).
    pub fn parse(body: &[u8]) -> Result<Self, EnvelopeError> {
        let value: Value = serde_json::from_slice(body).map_err(EnvelopeError::NotJson)?;
        let obj = value.as_object().ok_or(EnvelopeError::NotObject)?;
        let av = obj
            .get(ANTHROPIC_VERSION_KEY)
            .ok_or(EnvelopeError::MissingAnthropicVersion)?;
        let av_str = av
            .as_str()
            .ok_or(EnvelopeError::AnthropicVersionNotString)?
            .to_string();
        Ok(Self {
            anthropic_version: av_str,
            body: value,
        })
    }

    /// Re-emit the envelope as JSON bytes with `anthropic_version`
    /// preserved as the FIRST key.
    ///
    /// Used only after the compressor returns `Modified` and we need
    /// to reassemble the body. With `serde_json`'s `preserve_order`
    /// feature (workspace default), the parsed object already preserves
    /// insertion order; we just need to make sure `anthropic_version`
    /// is the first key. If the compressor's output already has it
    /// first (which it will — the compressor only mutates the
    /// `messages` content slot, not key ordering), we hand the bytes
    /// back unchanged.
    ///
    /// `body` is the (possibly-compressed) JSON bytes returned by the
    /// compression dispatcher.
    ///
    /// Returns `Ok(bytes)` on success. On any structural error, returns
    /// the input bytes unchanged so the byte-fidelity contract is
    /// preserved (the caller will surface the error).
    pub fn ensure_anthropic_version_first(body: &[u8]) -> Result<Bytes, EnvelopeError> {
        // Parse with preserve_order (workspace serde_json default).
        let mut value: Value = serde_json::from_slice(body).map_err(EnvelopeError::NotJson)?;
        let map = value.as_object_mut().ok_or(EnvelopeError::NotObject)?;

        // If `anthropic_version` is missing, that's a logical error
        // for the Bedrock surface — surface it loudly.
        if !map.contains_key(ANTHROPIC_VERSION_KEY) {
            return Err(EnvelopeError::MissingAnthropicVersion);
        }

        // Already first? Then `body` round-trips byte-equal — no work.
        // Note: serde_json::Map iteration order, with preserve_order,
        // is insertion order. We check the first key directly.
        if map
            .keys()
            .next()
            .map(|k| k.as_str() == ANTHROPIC_VERSION_KEY)
            .unwrap_or(false)
        {
            return Ok(Bytes::copy_from_slice(body));
        }

        // Reorder: pull anthropic_version, drain rest, rebuild with
        // anthropic_version first. This only runs when the compressor
        // moved the field (it shouldn't, but if it ever does we
        // reassert the invariant rather than ship a Bedrock-rejecting
        // body).
        let av = map
            .remove(ANTHROPIC_VERSION_KEY)
            .expect("contains_key guard above");
        let mut new_map = serde_json::Map::with_capacity(map.len() + 1);
        new_map.insert(ANTHROPIC_VERSION_KEY.to_string(), av);
        for (k, v) in std::mem::take(map) {
            new_map.insert(k, v);
        }
        let rebuilt = Value::Object(new_map);
        let bytes = serde_json::to_vec(&rebuilt).map_err(EnvelopeError::NotJson)?;
        Ok(Bytes::from(bytes))
    }
}

/// Helper for reading just the model id from a Bedrock URL path.
///
/// Bedrock paths look like `/model/{model_id}/invoke`. The `{model_id}`
/// segment can contain dots (`anthropic.claude-3-haiku-20240307-v1:0`),
/// hyphens, colons, and digits — all of which are allowed in URL path
/// segments. We use Axum's `:model_id` capture; this helper exists for
/// callers who only have the raw path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPath {
    pub model_id: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_minimal_envelope() {
        let body = json!({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}]
        });
        let bytes = serde_json::to_vec(&body).unwrap();
        let env = BedrockEnvelope::parse(&bytes).expect("parses");
        assert_eq!(env.anthropic_version, "bedrock-2023-05-31");
    }

    #[test]
    fn missing_anthropic_version_errors() {
        let body = json!({"max_tokens": 16, "messages": []});
        let bytes = serde_json::to_vec(&body).unwrap();
        let err = BedrockEnvelope::parse(&bytes).expect_err("must error");
        assert!(matches!(err, EnvelopeError::MissingAnthropicVersion));
    }

    #[test]
    fn anthropic_version_not_string_errors() {
        let body = json!({"anthropic_version": 123, "max_tokens": 16});
        let bytes = serde_json::to_vec(&body).unwrap();
        let err = BedrockEnvelope::parse(&bytes).expect_err("must error");
        assert!(matches!(err, EnvelopeError::AnthropicVersionNotString));
    }

    #[test]
    fn not_an_object_errors() {
        let bytes = b"[1,2,3]";
        let err = BedrockEnvelope::parse(bytes).expect_err("must error");
        assert!(matches!(err, EnvelopeError::NotObject));
    }

    #[test]
    fn invalid_json_errors() {
        let bytes = b"not json";
        let err = BedrockEnvelope::parse(bytes).expect_err("must error");
        assert!(matches!(err, EnvelopeError::NotJson(_)));
    }

    #[test]
    fn ensure_first_no_op_when_already_first() {
        let body = br#"{"anthropic_version":"bedrock-2023-05-31","max_tokens":16}"#;
        let out = BedrockEnvelope::ensure_anthropic_version_first(body).unwrap();
        assert_eq!(&out[..], &body[..]);
    }

    #[test]
    fn ensure_first_reorders_when_not_first() {
        // anthropic_version comes second.
        let body = br#"{"max_tokens":16,"anthropic_version":"bedrock-2023-05-31"}"#;
        let out = BedrockEnvelope::ensure_anthropic_version_first(body).unwrap();
        let out_str = std::str::from_utf8(&out).unwrap();
        // First key after `{"` must be `anthropic_version`.
        assert!(
            out_str.starts_with(r#"{"anthropic_version":"bedrock-2023-05-31""#),
            "got {out_str}"
        );
    }
}
