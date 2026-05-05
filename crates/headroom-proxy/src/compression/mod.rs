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
pub use live_zone_anthropic::{compress_anthropic_request, Outcome, PassthroughReason};
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
}
