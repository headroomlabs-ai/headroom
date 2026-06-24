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
use headroom_core::auth_mode::AuthMode as RequestAuthMode;
use headroom_core::transforms::live_zone::DEFAULT_MODEL;
use headroom_core::transforms::{
    compress_openai_chat_live_zone, BlockAction, LiveZoneError, LiveZoneOutcome,
};
use serde_json::Value;

use crate::cache_stabilization::tool_def_normalize::{
    any_tool_has_cache_control, sort_schema_keys_recursive, sort_tools_deterministically,
};
use crate::compression::{Outcome, PassthroughReason, PerStrategyTokens};
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
    auth_mode: RequestAuthMode,
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

    // ── Phase E PR-E1: tool-array deterministic sort ────────────
    // Same gate logic as the Anthropic walker (see that module's
    // `normalize_tool_definitions` for rationale). PAYG-only,
    // skipped when any tool already carries `cache_control`.
    let (dispatch_body, normalization_applied) =
        normalize_tool_definitions_openai_chat(body, &parsed, auth_mode, request_id);

    // F2.1 c2/6: forward F1's classified auth_mode into the dispatcher
    // instead of the hard-coded `Payg`. See live_zone_anthropic.rs for
    // the rationale — same wiring on the OpenAI chat path.
    match compress_openai_chat_live_zone(&dispatch_body, auth_mode.into(), model) {
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
            if normalization_applied.any() {
                return Outcome::Compressed {
                    body: dispatch_body,
                    tokens_before: 0,
                    tokens_after: 0,
                    strategies_applied: normalization_applied.strategies(),
                    markers_inserted: Vec::new(),
                    per_strategy_tokens: Vec::new(),
                };
            }
            Outcome::NoCompression
        }
        Ok(LiveZoneOutcome::Modified { new_body, manifest }) => {
            // Aggregate manifest stats. Mirrors the Anthropic
            // module — same metric shape so dashboards don't need
            // to special-case the provider.
            //
            // H1 + C5 remediation: per-strategy token accumulation
            // for the proxy's per-strategy compression-ratio metric +
            // every rejected-not-smaller block bumps the dedicated
            // counter.
            let mut original_bytes_total: usize = 0;
            let mut compressed_bytes_total: usize = 0;
            let mut original_tokens_total: usize = 0;
            let mut compressed_tokens_total: usize = 0;
            let mut strategies: Vec<&'static str> = Vec::new();
            let mut per_strategy_tokens: Vec<PerStrategyTokens> = Vec::new();
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
                        if let Some(slot) = per_strategy_tokens
                            .iter_mut()
                            .find(|s| s.strategy == strategy)
                        {
                            slot.original_tokens += original_tokens;
                            slot.compressed_tokens += compressed_tokens;
                        } else {
                            per_strategy_tokens.push(PerStrategyTokens {
                                strategy,
                                original_tokens,
                                compressed_tokens,
                            });
                        }
                    }
                    BlockAction::RejectedNotSmaller { strategy, .. } => {
                        crate::observability::record_compression_rejected_by_token_check(strategy);
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
            // Stitch in PR-E1 strategy tags so dashboards see the
            // tool-array sort separately from live-zone compressors.
            for strategy in normalization_applied.strategies() {
                if !strategies.contains(&strategy) {
                    strategies.push(strategy);
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
                per_strategy_tokens,
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

/// Tracks which Phase E normalization steps mutated the dispatch
/// body for the OpenAI Chat path. Mirrors the Anthropic walker's
/// `NormalizationApplied`.
#[derive(Debug, Clone, Copy, Default)]
struct NormalizationApplied {
    e1_tool_sort: bool,
    e2_schema_sort: bool,
}

impl NormalizationApplied {
    fn any(self) -> bool {
        self.e1_tool_sort || self.e2_schema_sort
    }

    fn strategies(self) -> Vec<&'static str> {
        let mut out = Vec::new();
        if self.e1_tool_sort {
            out.push("tool_array_sort");
        }
        if self.e2_schema_sort {
            out.push("schema_key_sort");
        }
        out
    }
}

/// Apply PR-E1 (tool-array sort) and PR-E2 (schema-key sort) to an
/// OpenAI Chat Completions body when the auth-mode + marker gates
/// clear. Mirrors the Anthropic walker's `normalize_tool_definitions`
/// — same gates, same outcome shape. Module-private; only the
/// dispatcher above calls this.
///
/// OpenAI tools nest the schema at `tool.function.parameters` (not
/// `tool.input_schema` like Anthropic) — this is the only walker
/// shape difference between the two providers.
fn normalize_tool_definitions_openai_chat(
    body: &Bytes,
    parsed: &Value,
    auth_mode: RequestAuthMode,
    request_id: &str,
) -> (Bytes, NormalizationApplied) {
    if !matches!(auth_mode, RequestAuthMode::Payg) {
        tracing::info!(
            event = "e1_skipped",
            request_id = %request_id,
            path = "/v1/chat/completions",
            reason = "auth_mode",
            auth_mode = auth_mode.as_str(),
            "tool-array sort skipped: non-PAYG auth mode passes through byte-equal"
        );
        tracing::info!(
            event = "e2_skipped",
            request_id = %request_id,
            path = "/v1/chat/completions",
            reason = "auth_mode",
            auth_mode = auth_mode.as_str(),
            "schema-key sort skipped: non-PAYG auth mode passes through byte-equal"
        );
        return (body.clone(), NormalizationApplied::default());
    }

    let Some(tools_in) = parsed.get("tools").and_then(Value::as_array) else {
        return (body.clone(), NormalizationApplied::default());
    };
    if tools_in.is_empty() {
        return (body.clone(), NormalizationApplied::default());
    }

    let marker_present = any_tool_has_cache_control(tools_in);
    if marker_present {
        tracing::info!(
            event = "e1_skipped",
            request_id = %request_id,
            path = "/v1/chat/completions",
            reason = "marker_present",
            tool_count = tools_in.len(),
            "tool-array sort skipped: customer cache_control marker present \
             on at least one tool; preserving customer-intentional order"
        );
    }

    let mut working = parsed.clone();
    let tools = working
        .get_mut("tools")
        .and_then(Value::as_array_mut)
        .expect("tools array verified above");

    let mut applied = NormalizationApplied::default();

    if !marker_present {
        applied.e1_tool_sort = sort_tools_deterministically(tools);
        if applied.e1_tool_sort {
            tracing::info!(
                event = "e1_applied",
                request_id = %request_id,
                path = "/v1/chat/completions",
                tool_count = tools.len(),
                "tool-array sort applied: tools reordered alphabetically by name"
            );
        }
    }

    // PR-E2: OpenAI tool schema lives at `tool.function.parameters`.
    for tool in tools.iter_mut() {
        let Some(parameters) = tool
            .get_mut("function")
            .and_then(|f| f.get_mut("parameters"))
        else {
            continue;
        };
        let before = serde_json::to_vec(parameters).unwrap_or_default();
        sort_schema_keys_recursive(parameters);
        let after = serde_json::to_vec(parameters).unwrap_or_default();
        if before != after {
            applied.e2_schema_sort = true;
        }
    }
    if applied.e2_schema_sort {
        tracing::info!(
            event = "e2_applied",
            request_id = %request_id,
            path = "/v1/chat/completions",
            tool_count = tools.len(),
            "schema-key sort applied: function.parameters keys rewritten in alphabetic order"
        );
    }

    if !applied.any() {
        return (body.clone(), applied);
    }

    match serde_json::to_vec(&working) {
        Ok(bytes) => (Bytes::from(bytes), applied),
        Err(e) => {
            tracing::warn!(
                event = "tool_def_normalize_serialize_failed",
                request_id = %request_id,
                path = "/v1/chat/completions",
                error = %e,
                "tool-def normalization failed at re-serialize; falling back \
                 to original body bytes"
            );
            (body.clone(), NormalizationApplied::default())
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
        let out = compress_openai_chat_request(
            &body,
            CompressionMode::Off,
            RequestAuthMode::Payg,
            "req-1",
        );
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
        let out = compress_openai_chat_request(
            &body,
            CompressionMode::LiveZone,
            RequestAuthMode::Payg,
            "req-2",
        );
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
        let out = compress_openai_chat_request(
            &body,
            CompressionMode::LiveZone,
            RequestAuthMode::Payg,
            "req-3",
        );
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
        let out = compress_openai_chat_request(
            &body,
            CompressionMode::LiveZone,
            RequestAuthMode::Payg,
            "req-4",
        );
        assert!(matches!(out, Outcome::NoCompression));
    }

    #[test]
    fn e1_sorts_tools_when_payg() {
        let body = body_of(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {"type": "function", "function": {"name": "zebra"}},
                {"type": "function", "function": {"name": "apple"}},
            ],
        }));
        let out = compress_openai_chat_request(
            &body,
            CompressionMode::LiveZone,
            RequestAuthMode::Payg,
            "req-e1",
        );
        match out {
            Outcome::Compressed {
                strategies_applied, ..
            } => assert!(
                strategies_applied.contains(&"tool_array_sort"),
                "expected tool_array_sort, got {strategies_applied:?}",
            ),
            other => panic!("expected Compressed, got {other:?}"),
        }
    }

    #[test]
    fn e1_passes_through_when_oauth() {
        let body = body_of(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {"type": "function", "function": {"name": "zebra"}},
                {"type": "function", "function": {"name": "apple"}},
            ],
        }));
        let out = compress_openai_chat_request(
            &body,
            CompressionMode::LiveZone,
            RequestAuthMode::OAuth,
            "req-e1-oauth",
        );
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
