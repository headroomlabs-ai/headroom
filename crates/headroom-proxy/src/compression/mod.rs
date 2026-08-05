//! Compression interceptor for LLM-shaped requests.
//!
//! # Phase A lockdown (PR-A1)
//!
//! Per `REALIGNMENT/03-phase-A-lockdown.md`, the
//! `IntelligentContextManager`-driven path that previously ran on
//! every `/v1/messages` request is gone. Today this module is a
//! tracking shell: it owns the path-matcher (`is_compressible_path`)
//! and the Anthropic decision stub (`compress_anthropic_request`)
//! that always returns `Outcome::NoCompression`.
//!
//! Phase B PR-B2 reintroduces real compression, but with two
//! invariants the deleted code violated:
//!
//! 1. The cache hot zone (system, tools, historical messages,
//!    reasoning items, thinking signatures, redacted_thinking,
//!    compaction items) is never modified.
//! 2. Compression is append-only: only the live zone is rewritten.
//!
//! # Provider matrix (current + planned)
//!
//! | Provider     | Path                  | Status |
//! |--------------|-----------------------|--------|
//! | Anthropic    | `POST /v1/messages`   | passthrough (PR-A1) → live-zone (PR-B2) |
//! | OpenAI       | `POST /v1/chat/completions` | follow-up |
//! | Google       | `POST /v1beta/...`    | follow-up |
//! | Bedrock      | varied                | follow-up |
//!
//! # Failure-mode contract
//!
//! Compression must NEVER break a request. Even when Phase B brings
//! a real dispatcher back, every error path falls through to the
//! original body being forwarded unchanged.

pub mod anthropic;
pub mod live_zone_anthropic;
pub mod live_zone_openai;
pub mod live_zone_responses;
pub mod model_limits;

// PR-A4 helper for cache-control floor derivation lives on the
// passthrough-stub module so PR-B2's live-zone dispatcher can call
// it without dragging in the rest of `anthropic.rs`. The stub
// itself stays through B1 → B2 transition for parallel review;
// `compress_anthropic_request` is sourced from the live-zone module.
pub use anthropic::resolve_frozen_count;
pub use live_zone_anthropic::{
    compress_anthropic_request, Outcome, PassthroughReason, PerStrategyTokens,
};
pub use live_zone_openai::{
    compress_openai_chat_request, should_skip_compression, SkipCompressionReason,
};
pub use live_zone_responses::compress_openai_responses_request;

/// Which provider's compression dispatcher should run for a request
/// path. PR-C2 wired `/v1/chat/completions`; PR-C3 adds
/// `/v1/responses`. Future PRs add Gemini etc. Returning an enum
/// (rather than a bare bool + string later) keeps the routing
/// explicit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressibleEndpoint {
    /// Anthropic `/v1/messages`.
    AnthropicMessages,
    /// OpenAI Chat Completions `/v1/chat/completions`.
    OpenAiChatCompletions,
    /// OpenAI Responses `/v1/responses`.
    OpenAiResponses,
}

/// Does this request path target an LLM endpoint we know how to
/// compress? Cheap pre-filter before buffering the body.
pub fn is_compressible_path(path: &str) -> bool {
    classify_compressible_path(path).is_some()
}

/// Classify a request path to its compression dispatcher (or `None`
/// if no compressor handles it). Single match arm per provider keeps
/// the cache scope explicit.
pub fn classify_compressible_path(path: &str) -> Option<CompressibleEndpoint> {
    match path {
        "/v1/messages" => Some(CompressibleEndpoint::AnthropicMessages),
        "/v1/chat/completions" => Some(CompressibleEndpoint::OpenAiChatCompletions),
        "/v1/responses" => Some(CompressibleEndpoint::OpenAiResponses),
        _ => None,
    }
}

/// Strip the `[1m]` context-window tier suffix that the Headroom
/// CLI appends to Anthropic model IDs (e.g. `glm-5.2[1m]`,
/// `claude-3-7-sonnet[1m]`) before forwarding to the upstream
/// Anthropic API. The upstream does not recognize the suffix and
/// rejects requests with 400.
///
/// This mirrors the Python proxy's `sanitize_anthropic_model_id()`
/// (Python PR #1840, fixes issue #1812). The Rust proxy
/// intentionally scopes this to **Anthropic** requests only — the
/// `[1m]` marker is an Anthropic / Claude Code compatibility
/// signal emitted by the CLI, and we must not silently mutate
/// OpenAI-compatible request model IDs. The caller is responsible
/// for gating on `CompressibleEndpoint::AnthropicMessages` before
/// invoking this function; doing so at the call site (not inside
/// this function) keeps the helper cheap and unambiguous.
///
/// Returns the original `body` byte-for-byte on every "no
/// sanitization" path:
/// - body is not valid JSON
/// - body has no top-level `model` field
/// - top-level `model` is not a string
/// - the string does not end in `[1m]`
/// - re-serialization fails (extremely unusual; we have already
///   parsed the body, but `serde_json` could in theory reject a
///   value shape it accepted on parse)
pub fn sanitize_anthropic_model_id_in_body(body: bytes::Bytes) -> bytes::Bytes {
    let Ok(mut parsed) = serde_json::from_slice::<serde_json::Value>(&body) else {
        return body;
    };

    let Some(model_value) = parsed.get_mut("model") else {
        return body;
    };

    let serde_json::Value::String(model) = model_value else {
        return body;
    };

    let sanitized = trim_anthropic_model_id_suffix(model);
    if sanitized == *model {
        // Either no `[1m]` suffix to strip, or the suffix is not at
        // the tail (e.g. `claude-3-7-sonnet[1m]-thinking` — not a
        // CLI-emitted shape, but a defensive no-op). Byte-equal
        // passthrough preserves the cache-safety invariant.
        return body;
    }

    *model = sanitized;
    match serde_json::to_vec(&parsed) {
        Ok(buf) => bytes::Bytes::from(buf),
        Err(_) => body,
    }
}

/// Pure helper: strip a trailing `[1m]` from a model ID string.
/// Exposed for unit tests; production callers should use
/// [`sanitize_anthropic_model_id_in_body`].
fn trim_anthropic_model_id_suffix(model: &str) -> String {
    model.trim_end_matches("[1m]").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anthropic_messages_path_matches() {
        assert!(is_compressible_path("/v1/messages"));
        assert_eq!(
            classify_compressible_path("/v1/messages"),
            Some(CompressibleEndpoint::AnthropicMessages)
        );
    }

    #[test]
    fn openai_chat_path_matches() {
        assert!(is_compressible_path("/v1/chat/completions"));
        assert_eq!(
            classify_compressible_path("/v1/chat/completions"),
            Some(CompressibleEndpoint::OpenAiChatCompletions)
        );
    }

    #[test]
    fn openai_responses_path_matches() {
        assert!(is_compressible_path("/v1/responses"));
        assert_eq!(
            classify_compressible_path("/v1/responses"),
            Some(CompressibleEndpoint::OpenAiResponses)
        );
    }

    #[test]
    fn other_paths_skip() {
        assert!(!is_compressible_path("/v1/messages/123"));
        assert!(!is_compressible_path("/v1/responses/123"));
        assert!(!is_compressible_path("/healthz"));
        assert!(!is_compressible_path("/"));
        assert!(!is_compressible_path(""));
    }

    // ─── [1m] Anthropic model-suffix sanitizer (PR #2027) ────────────
    //
    // PR #2027 review feedback (JerrettDavis): the sanitizer must
    // strip the `[1m]` CLI suffix from Anthropic `/v1/messages`
    // request bodies without mutating OpenAI-shaped bodies. These
    // unit tests pin the pure helper; integration coverage for
    // /v1/messages vs /v1/chat/completions vs /v1/responses lives
    // in `tests/integration_anthropic_model_sanitize.rs`.

    #[test]
    fn sanitizer_strips_trailing_1m_suffix() {
        let body = br#"{"model":"glm-5.2[1m]","max_tokens":1024,"messages":[]}"#;
        let out = sanitize_anthropic_model_id_in_body(bytes::Bytes::copy_from_slice(body));
        let parsed: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(parsed["model"], "glm-5.2");
        // Other fields round-trip unchanged.
        assert_eq!(parsed["max_tokens"], 1024);
        assert!(parsed["messages"].is_array());
    }

    #[test]
    fn sanitizer_strips_suffix_from_claude_model() {
        let body = br#"{"model":"claude-3-7-sonnet[1m]","max_tokens":1024}"#;
        let out = sanitize_anthropic_model_id_in_body(bytes::Bytes::copy_from_slice(body));
        let parsed: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(parsed["model"], "claude-3-7-sonnet");
    }

    #[test]
    fn sanitizer_passthrough_when_no_suffix() {
        let body = br#"{"model":"claude-3-7-sonnet","max_tokens":1024}"#;
        let original = bytes::Bytes::copy_from_slice(body);
        let out = sanitize_anthropic_model_id_in_body(original.clone());
        // Byte-equal — required to keep the cache-safety invariant
        // for the (very common) no-suffix case.
        assert_eq!(out, original);
    }

    #[test]
    fn sanitizer_passthrough_when_model_not_string() {
        // Anthropic `/v1/messages` requires a string model, but a
        // malformed body must round-trip unchanged — we never
        // mutate non-string `model` values.
        let body = br#"{"model":42,"max_tokens":1024}"#;
        let original = bytes::Bytes::copy_from_slice(body);
        let out = sanitize_anthropic_model_id_in_body(original.clone());
        assert_eq!(out, original);
    }

    #[test]
    fn sanitizer_passthrough_when_no_model_field() {
        let body = br#"{"max_tokens":1024,"messages":[]}"#;
        let original = bytes::Bytes::copy_from_slice(body);
        let out = sanitize_anthropic_model_id_in_body(original.clone());
        assert_eq!(out, original);
    }

    #[test]
    fn sanitizer_passthrough_for_non_json_body() {
        // A non-JSON body (e.g. SSE-rewritten or streaming chunk)
        // must round-trip unchanged. The compressible-path gate in
        // proxy.rs already filters by Content-Type=application/json,
        // so this is a belt-and-braces check.
        let body = b"not json at all";
        let original = bytes::Bytes::copy_from_slice(body);
        let out = sanitize_anthropic_model_id_in_body(original.clone());
        assert_eq!(out, original);
    }

    #[test]
    fn sanitizer_passthrough_for_1m_not_at_tail() {
        // Defensive: a `[1m]` mid-string is not a CLI suffix and
        // must not be mutated. Real CLI behavior only ever appends
        // `[1m]` at the tail, but a future wire format should not
        // silently corrupt legitimate `model` strings.
        let body = br#"{"model":"claude-3-7-sonnet[1m]-thinking","max_tokens":1024}"#;
        let original = bytes::Bytes::copy_from_slice(body);
        let out = sanitize_anthropic_model_id_in_body(original.clone());
        assert_eq!(out, original);
    }

    #[test]
    fn trim_helper_pure() {
        // The pure helper is a thin wrapper around
        // `str::trim_end_matches`; pin its behavior directly so
        // refactors that swap the implementation can't silently
        // change the strip semantics.
        assert_eq!(trim_anthropic_model_id_suffix("glm-5.2[1m]"), "glm-5.2");
        assert_eq!(trim_anthropic_model_id_suffix("claude[1m]"), "claude");
        assert_eq!(trim_anthropic_model_id_suffix("claude"), "claude");
        // `trim_end_matches` is greedy — back-to-back suffixes are
        // all stripped. The CLI never emits this, but pinning the
        // behavior makes the helper a pure function with a stable
        // contract.
        assert_eq!(trim_anthropic_model_id_suffix("claude[1m][1m]"), "claude");
    }
}
