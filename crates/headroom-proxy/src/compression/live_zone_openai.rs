//! OpenAI Chat Completions `/v1/chat/completions` request compression
//! — live-zone dispatcher entry point.
//!
//! # Provider scope
//!
//! Sibling of [`super::live_zone_anthropic`]. Same per-content-type
//! compressor backend, same byte-threshold gate, same tokenizer-validated
//! rejection check, same byte-range surgery. The differences from
//! Anthropic are walker-shape:
//!
//! - **Live zone:** the latest `role == "tool"` message's `content`
//!   AND the latest `role == "user"` message's text content. Earlier
//!   tool/user messages are frozen (cached prefix); never touched.
//! - **No `frozen_message_count`:** OpenAI doesn't expose a
//!   provider-level `cache_control` marker scheme like Anthropic.
//!   Cache safety is enforced purely by the live-zone walker — only
//!   the *latest* tool / user messages are candidates.
//! - **`n > 1` passthrough:** when the request asks for multiple
//!   completions, we don't compress; the handler short-circuits
//!   before calling this module.
//! - **`tools` and `tool_choice` are never mutated.** Mutating tool
//!   definitions would bust per-tool-schema cache; the dispatcher
//!   doesn't read or rewrite either field.
//!
//! Failure-mode contract matches the Anthropic side: every error path
//! returns the original body unchanged (the proxy forwards verbatim).
//! Per `feedback_no_silent_fallbacks.md`: per-block compressor errors
//! are surfaced via the manifest at warn-level; only the failing
//! block reverts, not the whole request.

use bytes::Bytes;
use headroom_core::transforms::live_zone::DEFAULT_MODEL;
use headroom_core::transforms::{
    compress_openai_chat_live_zone, AuthMode, BlockAction, LiveZoneError, LiveZoneOutcome,
};

use crate::compression::{Outcome, PassthroughReason};
use crate::config::CompressionMode;

/// OpenAI Chat Completions live-zone compression entry point.
///
/// # Behaviour
///
/// - `mode == Off` → [`Outcome::Passthrough { ModeOff }`].
/// - Body parses but `messages` is missing/non-array → `Passthrough { NoMessages }`.
/// - Body doesn't parse → `Passthrough { NotJson }`.
/// - `n > 1` (caller-detected) is *not* this module's responsibility;
///   the handler skips this call. The dispatcher always assumes the
///   caller has already gated the non-determinism case.
/// - Latest user message body or latest tool message body is large
///   enough to compress → [`Outcome::Compressed`] (proxy forwards
///   the new body).
/// - Otherwise → [`Outcome::NoCompression`] (proxy forwards original).
pub fn compress_openai_chat_request(
    body: &Bytes,
    mode: CompressionMode,
    request_id: &str,
) -> Outcome {
    if matches!(mode, CompressionMode::Off) {
        tracing::info!(
            event = "compression_decision",
            request_id = %request_id,
            path = "/v1/chat/completions",
            method = "POST",
            compression_mode = mode.as_str(),
            decision = "passthrough",
            reason = "mode_off",
            body_bytes = body.len(),
            "openai chat compression decision"
        );
        return Outcome::Passthrough {
            reason: PassthroughReason::ModeOff,
        };
    }

    // Inspect the body shape only enough to gate. The dispatcher does
    // its own parse — keeping the gate lightweight (just `messages`
    // existence + `n` + `stream` flags) avoids double-walking the
    // tree for the common LiveZone/no-compression case.
    let parsed: serde_json::Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(_) => {
            tracing::warn!(
                event = "compression_decision",
                request_id = %request_id,
                path = "/v1/chat/completions",
                method = "POST",
                compression_mode = mode.as_str(),
                decision = "passthrough",
                reason = "not_json",
                body_bytes = body.len(),
                "openai chat compression decision"
            );
            return Outcome::Passthrough {
                reason: PassthroughReason::NotJson,
            };
        }
    };

    if parsed.get("messages").and_then(|v| v.as_array()).is_none() {
        tracing::info!(
            event = "compression_decision",
            request_id = %request_id,
            path = "/v1/chat/completions",
            method = "POST",
            compression_mode = mode.as_str(),
            decision = "passthrough",
            reason = "no_messages",
            body_bytes = body.len(),
            "openai chat compression decision"
        );
        return Outcome::Passthrough {
            reason: PassthroughReason::NoMessages,
        };
    }

    let model = parsed
        .get("model")
        .and_then(serde_json::Value::as_str)
        .unwrap_or(DEFAULT_MODEL);

    match compress_openai_chat_live_zone(body, AuthMode::Payg, model) {
        Ok(LiveZoneOutcome::NoChange { manifest }) => {
            tracing::info!(
                event = "compression_decision",
                request_id = %request_id,
                path = "/v1/chat/completions",
                method = "POST",
                compression_mode = mode.as_str(),
                decision = "no_change",
                reason = "no_block_compressed",
                body_bytes = body.len(),
                messages_total = manifest.messages_total,
                latest_user_message_index = ?manifest.latest_user_message_index,
                live_zone_blocks = manifest.block_outcomes.len(),
                model = model,
                "openai chat live-zone dispatch"
            );
            Outcome::NoCompression
        }
        Ok(LiveZoneOutcome::Modified { new_body, manifest }) => {
            // Aggregate manifest stats. Mirrors the Anthropic
            // module — same metric shape so dashboards don't need
            // to special-case the provider.
            let mut original_bytes_total: usize = 0;
            let mut compressed_bytes_total: usize = 0;
            let mut original_tokens_total: usize = 0;
            let mut compressed_tokens_total: usize = 0;
            let mut strategies: Vec<&'static str> = Vec::new();
            let mut had_compressor_error = false;
            for entry in &manifest.block_outcomes {
                match entry.action {
                    BlockAction::Compressed {
                        strategy,
                        original_bytes,
                        compressed_bytes,
                        original_tokens,
                        compressed_tokens,
                    } => {
                        original_bytes_total += original_bytes;
                        compressed_bytes_total += compressed_bytes;
                        original_tokens_total += original_tokens;
                        compressed_tokens_total += compressed_tokens;
                        if !strategies.contains(&strategy) {
                            strategies.push(strategy);
                        }
                    }
                    BlockAction::CompressorError {
                        strategy,
                        ref error,
                    } => {
                        had_compressor_error = true;
                        tracing::error!(
                            event = "compression_error",
                            request_id = %request_id,
                            path = "/v1/chat/completions",
                            strategy = strategy,
                            error = %error,
                            "openai chat compressor error on a block; that block reverts to original"
                        );
                    }
                    _ => {}
                }
            }
            let body_bytes_in = body.len();
            let new_body_bytes = Bytes::copy_from_slice(new_body.get().as_bytes());
            let body_bytes_out = new_body_bytes.len();
            tracing::info!(
                event = "compression_decision",
                request_id = %request_id,
                path = "/v1/chat/completions",
                method = "POST",
                compression_mode = mode.as_str(),
                decision = "compressed",
                reason = "live_zone_blocks_rewritten",
                body_bytes_in = body_bytes_in,
                body_bytes_out = body_bytes_out,
                bytes_freed = body_bytes_in.saturating_sub(body_bytes_out),
                messages_total = manifest.messages_total,
                latest_user_message_index = ?manifest.latest_user_message_index,
                live_zone_blocks = manifest.block_outcomes.len(),
                live_zone_strategies = ?strategies,
                live_zone_block_original_bytes = original_bytes_total,
                live_zone_block_compressed_bytes = compressed_bytes_total,
                live_zone_block_original_tokens = original_tokens_total,
                live_zone_block_compressed_tokens = compressed_tokens_total,
                had_compressor_error = had_compressor_error,
                model = model,
                "openai chat live-zone dispatch"
            );
            Outcome::Compressed {
                body: new_body_bytes,
                tokens_before: original_tokens_total,
                tokens_after: compressed_tokens_total,
                strategies_applied: strategies,
                markers_inserted: Vec::new(),
            }
        }
        Err(LiveZoneError::BodyNotJson(_)) => {
            tracing::warn!(
                event = "compression_decision",
                request_id = %request_id,
                path = "/v1/chat/completions",
                "openai chat live-zone dispatcher rejected JSON body; falling back to passthrough"
            );
            Outcome::Passthrough {
                reason: PassthroughReason::NotJson,
            }
        }
        Err(LiveZoneError::NoMessagesArray) => {
            tracing::info!(
                event = "compression_decision",
                request_id = %request_id,
                path = "/v1/chat/completions",
                method = "POST",
                compression_mode = mode.as_str(),
                decision = "passthrough",
                reason = "no_messages",
                body_bytes = body.len(),
                "openai chat compression decision"
            );
            Outcome::Passthrough {
                reason: PassthroughReason::NoMessages,
            }
        }
    }
}

/// Inspect a Chat Completions request body and return `true` if the
/// proxy should skip live-zone compression entirely.
///
/// PR-C2 conditions (any matched → skip):
///
/// - `n > 1` (multiple completions; non-determinism semantics —
///   compressing some user/tool blocks while requesting many
///   completions confuses cache invariants and may mask bugs).
///
/// `tool_choice` and `stream_options` are NOT skip conditions: they
/// don't affect what we'd touch (the dispatcher never reads or
/// rewrites tool definitions or stream options). They round-trip
/// byte-equal as a side effect of byte-range surgery.
pub fn should_skip_compression(body: &Bytes) -> SkipCompressionReason {
    let parsed: serde_json::Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        // Don't skip on bad JSON — let the dispatcher surface
        // `Passthrough { NotJson }` itself so the decision is logged
        // through one path.
        Err(_) => return SkipCompressionReason::DoNotSkip,
    };

    if let Some(n) = parsed.get("n").and_then(|v| v.as_u64()) {
        if n > 1 {
            return SkipCompressionReason::NGreaterThanOne(n);
        }
    }

    SkipCompressionReason::DoNotSkip
}

/// Reason the proxy chose to skip Chat Completions live-zone compression
/// pre-dispatch. `DoNotSkip` is the common case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipCompressionReason {
    /// Run the live-zone dispatcher.
    DoNotSkip,
    /// `n > 1` was set on the request — multiple completions imply
    /// non-determinism scenarios; passthrough preserves byte-fidelity.
    NGreaterThanOne(u64),
}

impl SkipCompressionReason {
    pub fn is_skip(self) -> bool {
        !matches!(self, SkipCompressionReason::DoNotSkip)
    }

    pub fn as_log_str(self) -> &'static str {
        match self {
            SkipCompressionReason::DoNotSkip => "do_not_skip",
            SkipCompressionReason::NGreaterThanOne(_) => "n_greater_than_one",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn body_of(value: serde_json::Value) -> Bytes {
        Bytes::from(serde_json::to_vec(&value).unwrap())
    }

    #[test]
    fn mode_off_short_circuits() {
        let body = Bytes::from_static(b"not valid json");
        let out = compress_openai_chat_request(&body, CompressionMode::Off, "req-1");
        assert!(matches!(
            out,
            Outcome::Passthrough {
                reason: PassthroughReason::ModeOff
            }
        ));
    }

    #[test]
    fn invalid_json_passthrough() {
        let body = Bytes::from_static(b"\x01\x02 not json");
        let out = compress_openai_chat_request(&body, CompressionMode::LiveZone, "req-2");
        assert!(matches!(
            out,
            Outcome::Passthrough {
                reason: PassthroughReason::NotJson
            }
        ));
    }

    #[test]
    fn no_messages_passthrough() {
        let body = body_of(json!({"model": "gpt-4o"}));
        let out = compress_openai_chat_request(&body, CompressionMode::LiveZone, "req-3");
        assert!(matches!(
            out,
            Outcome::Passthrough {
                reason: PassthroughReason::NoMessages
            }
        ));
    }

    #[test]
    fn small_body_no_change() {
        let body = body_of(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}]
        }));
        let out = compress_openai_chat_request(&body, CompressionMode::LiveZone, "req-4");
        assert!(matches!(out, Outcome::NoCompression));
    }

    #[test]
    fn n_eq_three_skip_predicate() {
        let body = body_of(json!({
            "model": "gpt-4o",
            "n": 3,
            "messages": [{"role": "user", "content": "hi"}]
        }));
        let r = should_skip_compression(&body);
        assert_eq!(r, SkipCompressionReason::NGreaterThanOne(3));
        assert!(r.is_skip());
    }

    #[test]
    fn n_eq_one_no_skip() {
        let body = body_of(json!({
            "model": "gpt-4o",
            "n": 1,
            "messages": [{"role": "user", "content": "hi"}]
        }));
        let r = should_skip_compression(&body);
        assert_eq!(r, SkipCompressionReason::DoNotSkip);
    }

    #[test]
    fn n_absent_no_skip() {
        let body = body_of(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}]
        }));
        let r = should_skip_compression(&body);
        assert_eq!(r, SkipCompressionReason::DoNotSkip);
    }
}
