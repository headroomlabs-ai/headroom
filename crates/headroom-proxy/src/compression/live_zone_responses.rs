//! OpenAI Responses `/v1/responses` request compression — live-zone
//! dispatcher entry point (Phase C PR-C3).
//!
//! # Provider scope
//!
//! Sibling of [`super::live_zone_openai`] (Chat Completions) and
//! [`super::live_zone_anthropic`] (Messages). Same per-content-type
//! compressor backend, same byte-threshold gate, same
//! tokenizer-validated rejection check, same byte-range surgery.
//!
//! Differences from the Chat Completions dispatcher:
//!
//! - Request shape: items are keyed under `input` (canonical) or
//!   `messages` (legacy alias) and are explicitly typed by the
//!   `type` field, not role-tagged.
//! - Live zone: latest of each compressible kind —
//!   `function_call_output`, `local_shell_call_output`,
//!   `apply_patch_call_output`, plus the latest `message` (user role)
//!   text content. Earlier *_output items are FROZEN.
//! - Output items must clear a 2 KiB minimum BEFORE the
//!   per-content-type byte threshold even runs (per spec PR-C3
//!   §scope, line 167 of the realignment plan).
//! - Cache hot zone: every other item type passes through verbatim.
//!   This includes `reasoning.encrypted_content`, `compaction.*`,
//!   MCP / computer-use / web-search / file-search /
//!   code-interpreter / image-generation / tool-search /
//!   custom-tool calls, and any future-unknown `type` value.
//!
//! Failure-mode contract matches every other live-zone dispatcher:
//! every error path returns the original body unchanged. Per-block
//! compressor errors surface via the manifest at warn-level; only the
//! failing block reverts.

use bytes::Bytes;
use headroom_core::transforms::live_zone::DEFAULT_MODEL;
use headroom_core::transforms::{
    compress_openai_responses_live_zone, AuthMode, BlockAction, LiveZoneError, LiveZoneOutcome,
};

use crate::compression::{Outcome, PassthroughReason};
use crate::config::CompressionMode;

/// OpenAI Responses live-zone compression entry point.
///
/// # Behaviour
///
/// - `mode == Off` → [`Outcome::Passthrough { ModeOff }`].
/// - Body parses but neither `input` nor `messages` is an array →
///   `Passthrough { NoMessages }`.
/// - Body doesn't parse → `Passthrough { NotJson }`.
/// - At least one live-zone block compressed → [`Outcome::Compressed`].
/// - Otherwise → [`Outcome::NoCompression`].
pub fn compress_openai_responses_request(
    body: &Bytes,
    mode: CompressionMode,
    request_id: &str,
) -> Outcome {
    if matches!(mode, CompressionMode::Off) {
        tracing::info!(
            event = "compression_decision",
            request_id = %request_id,
            path = "/v1/responses",
            method = "POST",
            compression_mode = mode.as_str(),
            decision = "passthrough",
            reason = "mode_off",
            body_bytes = body.len(),
            "openai responses compression decision"
        );
        return Outcome::Passthrough {
            reason: PassthroughReason::ModeOff,
        };
    }

    // Lightweight gate before the full dispatcher walk: parse only
    // enough to determine `input` (or `messages`) shape and the
    // model name. The dispatcher does its own parse — keeping this
    // gate light avoids double-walking the tree on the common
    // no-compression path.
    let parsed: serde_json::Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(_) => {
            tracing::warn!(
                event = "compression_decision",
                request_id = %request_id,
                path = "/v1/responses",
                method = "POST",
                compression_mode = mode.as_str(),
                decision = "passthrough",
                reason = "not_json",
                body_bytes = body.len(),
                "openai responses compression decision"
            );
            return Outcome::Passthrough {
                reason: PassthroughReason::NotJson,
            };
        }
    };

    let has_array_field = parsed
        .get("input")
        .or_else(|| parsed.get("messages"))
        .and_then(|v| v.as_array())
        .is_some();
    if !has_array_field {
        tracing::info!(
            event = "compression_decision",
            request_id = %request_id,
            path = "/v1/responses",
            method = "POST",
            compression_mode = mode.as_str(),
            decision = "passthrough",
            reason = "no_messages",
            body_bytes = body.len(),
            "openai responses compression decision"
        );
        return Outcome::Passthrough {
            reason: PassthroughReason::NoMessages,
        };
    }

    // Walk every item once for telemetry — log unknown item types at
    // warn level (no-silent-fallbacks) and redact image_data fields
    // from the logged shape (no PII / no megabytes of base64). The
    // upstream-bound bytes are NEVER mutated by this loop; the body
    // is forwarded byte-for-byte as the live-zone dispatcher decides.
    log_item_telemetry(&parsed, request_id);

    let model = parsed
        .get("model")
        .and_then(serde_json::Value::as_str)
        .unwrap_or(DEFAULT_MODEL);

    match compress_openai_responses_live_zone(body, AuthMode::Payg, model) {
        Ok(LiveZoneOutcome::NoChange { manifest }) => {
            tracing::info!(
                event = "compression_decision",
                request_id = %request_id,
                path = "/v1/responses",
                method = "POST",
                compression_mode = mode.as_str(),
                decision = "no_change",
                reason = "no_block_compressed",
                body_bytes = body.len(),
                items_total = manifest.messages_total,
                latest_user_message_index = ?manifest.latest_user_message_index,
                live_zone_blocks = manifest.block_outcomes.len(),
                model = model,
                "openai responses live-zone dispatch"
            );
            Outcome::NoCompression
        }
        Ok(LiveZoneOutcome::Modified { new_body, manifest }) => {
            // Aggregate per-block savings for the structured log.
            // Mirrors the Chat Completions sibling so dashboards
            // don't need provider-specific shapes.
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
                            path = "/v1/responses",
                            strategy = strategy,
                            error = %error,
                            "openai responses compressor error on a block; that block reverts to original"
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
                path = "/v1/responses",
                method = "POST",
                compression_mode = mode.as_str(),
                decision = "compressed",
                reason = "live_zone_blocks_rewritten",
                body_bytes_in = body_bytes_in,
                body_bytes_out = body_bytes_out,
                bytes_freed = body_bytes_in.saturating_sub(body_bytes_out),
                items_total = manifest.messages_total,
                latest_user_message_index = ?manifest.latest_user_message_index,
                live_zone_blocks = manifest.block_outcomes.len(),
                live_zone_strategies = ?strategies,
                live_zone_block_original_bytes = original_bytes_total,
                live_zone_block_compressed_bytes = compressed_bytes_total,
                live_zone_block_original_tokens = original_tokens_total,
                live_zone_block_compressed_tokens = compressed_tokens_total,
                had_compressor_error = had_compressor_error,
                model = model,
                "openai responses live-zone dispatch"
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
                path = "/v1/responses",
                "openai responses live-zone dispatcher rejected JSON body; falling back to passthrough"
            );
            Outcome::Passthrough {
                reason: PassthroughReason::NotJson,
            }
        }
        Err(LiveZoneError::NoMessagesArray) => {
            tracing::info!(
                event = "compression_decision",
                request_id = %request_id,
                path = "/v1/responses",
                method = "POST",
                compression_mode = mode.as_str(),
                decision = "passthrough",
                reason = "no_messages",
                body_bytes = body.len(),
                "openai responses compression decision"
            );
            Outcome::Passthrough {
                reason: PassthroughReason::NoMessages,
            }
        }
    }
}

/// Walk the items array once and emit per-item telemetry. Recognised
/// item types are tallied; unknown `type` values trigger a
/// `tracing::warn!` `event = responses_unknown_item_type` but never
/// alter the upstream-bound bytes. `image_generation_call.image_data`
/// is never logged verbatim — only its byte length, per spec.
fn log_item_telemetry(parsed: &serde_json::Value, request_id: &str) {
    let items = match parsed
        .get("input")
        .or_else(|| parsed.get("messages"))
        .and_then(|v| v.as_array())
    {
        Some(items) => items,
        None => return,
    };

    use crate::responses_items::{classify_items, ResponseItem};
    use serde_json::value::RawValue;

    // Build a `RawValue` from the items array so we can use the
    // typed classifier. We're already past the gate; one additional
    // serialize is fine (telemetry path, not hot path for body bytes).
    let items_string = match serde_json::to_string(items) {
        Ok(s) => s,
        Err(_) => return,
    };
    let items_raw = match RawValue::from_string(items_string) {
        Ok(r) => r,
        Err(_) => return,
    };
    let classified = match classify_items(&items_raw) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(
                event = "responses_classify_error",
                request_id = %request_id,
                error = %e,
                "could not classify Responses items array; passthrough preserves bytes"
            );
            return;
        }
    };

    let mut by_type: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for c in &classified {
        match &c.typed {
            None => {
                // No-silent-fallbacks: log the unknown type at warn,
                // preserving the type tag so operators can grep for it.
                tracing::warn!(
                    event = "responses_unknown_item_type",
                    request_id = %request_id,
                    type_tag = %c.type_tag,
                    raw_bytes = c.raw.get().len(),
                    "responses item with unknown `type` — preserving verbatim"
                );
                *by_type.entry("unknown").or_insert(0) += 1;
            }
            Some(item) => {
                let tag = item.type_tag();
                *by_type.entry(tag).or_insert(0) += 1;
                // Image-generation log redaction. The upstream-bound
                // body is NOT mutated; this only keeps `image_data`
                // out of the structured-log path. We log the tag and
                // a size estimate (the raw item byte length).
                if matches!(item, ResponseItem::ImageGenerationCall { .. }) {
                    tracing::debug!(
                        event = "responses_image_generation_call",
                        request_id = %request_id,
                        item_bytes = c.raw.get().len(),
                        // image_data is intentionally omitted —
                        // base64 image payloads can be megabytes.
                        "image_generation_call seen (image bytes redacted from log)"
                    );
                }
            }
        }
    }
    tracing::info!(
        event = "responses_item_summary",
        request_id = %request_id,
        items_total = classified.len(),
        breakdown = ?by_type,
        "responses item type breakdown"
    );
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
        let out = compress_openai_responses_request(&body, CompressionMode::Off, "req-1");
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
        let out = compress_openai_responses_request(&body, CompressionMode::LiveZone, "req-2");
        assert!(matches!(
            out,
            Outcome::Passthrough {
                reason: PassthroughReason::NotJson
            }
        ));
    }

    #[test]
    fn no_input_passthrough() {
        let body = body_of(json!({"model": "gpt-4o"}));
        let out = compress_openai_responses_request(&body, CompressionMode::LiveZone, "req-3");
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
            "input": [
                {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": "hi"}]}
            ]
        }));
        let out = compress_openai_responses_request(&body, CompressionMode::LiveZone, "req-4");
        assert!(matches!(out, Outcome::NoCompression));
    }
}
