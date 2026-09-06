//! Response-usage capture — Phase H. Shared by every lane.
//!
//! Recording a request's savings needs the *response's* token usage,
//! which arrives in four wire shapes:
//!
//! - JSON bodies (non-streaming Anthropic / OpenAI / Bedrock /
//!   Vertex responses),
//! - Anthropic-style SSE streams (direct, Vertex, and Bedrock's
//!   SSE-translated mode) — already accumulated by
//!   [`crate::sse::anthropic::AnthropicStreamState`],
//! - Converse-native frames whose usage keys are camelCase and carry
//!   no Anthropic `type` (the state machine drops them),
//! - Bedrock binary EventStream passthrough, where nothing parses
//!   the frames today.
//!
//! This module provides one [`PendingRecord`] (a request's
//! half-built ledger event plus the deferred-finalization contract:
//! **record only after the response is fully observed**, so
//! streaming token counts are real instead of zero) and the capture
//! tasks that feed it. Every capture path is a bounded, best-effort
//! tee off the client byte path — a stalled or hostile response can
//! delay telemetry, never the client.
//!
//! Best-effort has one deliberate edge: capture tasks are plain
//! `tokio::spawn`s with no shutdown drain, so requests still
//! in-flight when the process exits are dropped without recording.
//! Tracking them against shutdown would couple the byte path to
//! telemetry lifetime — the wrong trade for a stats ledger.

use std::sync::Arc;
use std::time::Instant;

use base64::Engine as _;
use serde_json::Value;

use super::ledger::{Ledger, RequestEvent, Usage};

/// Cap for buffered non-streaming response bodies awaiting a usage
/// parse. Bigger bodies finalize with zero usage (tokens/USD from
/// compression estimates only) rather than growing memory.
const JSON_CAPTURE_CAP: usize = 2 * 1024 * 1024;

/// Queue depth for every telemetry tee (capture bodies and the SSE
/// parser feeds in proxy/bedrock/vertex). Bounded so a stalled
/// parser task can never make the byte path wait: senders `try_send`
/// and drop on a full queue.
pub(crate) const TEE_QUEUE_DEPTH: usize = 256;

/// A request's ledger event, waiting for its response to finish.
///
/// Constructed when the request is forwarded; finalized exactly once
/// when the response is fully observed (stream closed / body ended /
/// transport failed). Latency is measured here so every lane reports
/// the same thing: time from forwarding decision to response fully
/// observed.
pub struct PendingRecord {
    ledger: Arc<Ledger>,
    event: RequestEvent,
    started: Instant,
}

impl PendingRecord {
    pub fn new(ledger: Arc<Ledger>, event: RequestEvent, started: Instant) -> Self {
        Self {
            ledger,
            event,
            started,
        }
    }

    /// Mark the upstream outcome failed (non-2xx or transport error)
    /// before finalization.
    pub fn mark_failed(&mut self) {
        self.event.failed = true;
    }

    /// Supply the model id the request side knows (path parameter,
    /// or the buffered request body when compression is on).
    pub fn set_model(&mut self, model: &str) {
        self.event.set_model(model);
    }

    /// Supply the model id learned from the *response*. Recording is
    /// not gated on the compression buffer (compression defaults
    /// off), so on a pure passthrough request this is the only place
    /// the model is known. Never overrides a request-side value.
    pub fn set_model_if_unknown(&mut self, model: &str) {
        if self.event.model_is_unknown() {
            self.event.set_model(model);
        }
    }

    /// Attach the compression dispatcher's before/after token
    /// estimate and the strategies it applied.
    pub fn set_compression(&mut self, before: u64, after: u64, transforms: Vec<String>) {
        self.event.tokens_before = before;
        self.event.tokens_after = after;
        self.event.transforms = transforms;
    }

    /// Record with the captured usage. Consumes the record — each
    /// request records exactly once.
    pub fn finalize(mut self, usage: Usage) {
        self.event.usage = usage;
        self.event.latency_ms = self.started.elapsed().as_millis() as u64;
        self.ledger.record(self.event);
    }

    /// Record a failure (zero usage, no savings — the ledger's
    /// failed-path accounting).
    pub fn finalize_failed(mut self) {
        self.mark_failed();
        self.finalize(Usage::default());
    }
}

/// Build the pending ledger event for a request entering a handler
/// (`None` when `--stats` is off). One call site per lane keeps the
/// "constructed at handler entry so every exit records exactly once"
/// rule easy to audit.
pub fn start_pending(
    state: &crate::proxy::AppState,
    provider: &'static str,
    model: &str,
    request_id: &str,
) -> Option<PendingRecord> {
    state.config.stats.then(|| {
        PendingRecord::new(
            state.stats.clone(),
            RequestEvent::new(provider, model, request_id),
            Instant::now(),
        )
    })
}

/// Finalize a pending record as failed on a proxy-side rejection
/// (bad path, missing credentials, signing failure, oversized body,
/// …). Sugar for handler early-return paths, so "every request that
/// enters a handler produces exactly one record" survives rejects: a
/// credentials outage must show in `/stats` as a failure spike, not
/// as zero traffic.
pub fn finalize_rejected(pending: Option<PendingRecord>) {
    if let Some(p) = pending {
        p.finalize_failed();
    }
}

/// Which JSON body shape to expect when parsing a non-streaming
/// response for usage.
#[derive(Debug, Clone, Copy)]
pub enum ResponseShape {
    Anthropic,
    OpenAiChat,
    OpenAiResponses,
    /// Bedrock sync responses: InvokeModel bodies are Anthropic
    /// snake_case, Converse bodies are camelCase — try both.
    Bedrock,
}

/// Extract usage from a complete response body. `None` when the
/// body has no recognisable usage block (error envelopes, non-JSON).
fn usage_from_response_json(shape: ResponseShape, v: &Value) -> Option<Usage> {
    usage_from_usage_block(shape, v.get("usage")?)
}

/// Parse a bare `usage` object (already extracted from its
/// envelope) — the SSE state machines hold usage in this form.
pub fn usage_from_usage_block(shape: ResponseShape, usage: &Value) -> Option<Usage> {
    match shape {
        ResponseShape::Anthropic => snake_usage(usage),
        ResponseShape::OpenAiChat => openai_chat_usage(usage),
        ResponseShape::OpenAiResponses => openai_responses_usage(usage),
        ResponseShape::Bedrock => snake_usage(usage).or_else(|| camel_usage(usage)),
    }
}

fn u64_at(v: &Value, key: &str) -> Option<u64> {
    v.get(key).and_then(Value::as_u64)
}

/// Anthropic `usage` block (also Bedrock InvokeModel for Anthropic
/// models): `input_tokens` already EXCLUDES cache reads/writes.
fn snake_usage(usage: &Value) -> Option<Usage> {
    let input = u64_at(usage, "input_tokens");
    let output = u64_at(usage, "output_tokens");
    if input.is_none() && output.is_none() {
        return None;
    }
    Some(Usage {
        input_tokens: input.unwrap_or(0),
        output_tokens: output.unwrap_or(0),
        cache_read_tokens: u64_at(usage, "cache_read_input_tokens").unwrap_or(0),
        cache_write_tokens: u64_at(usage, "cache_creation_input_tokens").unwrap_or(0),
    })
}

/// Bedrock Converse `usage` block: camelCase, and `inputTokens`
/// likewise excludes cache tokens.
///
/// The cache fields fall back to the `*InputTokenCount` spelling
/// after `*InputTokens`. Reading only one name would silently report
/// zero cache tokens against a response that uses the other, and a
/// miss costs one map lookup — cheap insurance on a field whose
/// whole job is to be believed.
fn camel_usage(usage: &Value) -> Option<Usage> {
    let input = u64_at(usage, "inputTokens");
    let output = u64_at(usage, "outputTokens");
    if input.is_none() && output.is_none() {
        return None;
    }
    Some(Usage {
        input_tokens: input.unwrap_or(0),
        output_tokens: output.unwrap_or(0),
        cache_read_tokens: u64_at(usage, "cacheReadInputTokens")
            .or_else(|| u64_at(usage, "cacheReadInputTokenCount"))
            .unwrap_or(0),
        cache_write_tokens: u64_at(usage, "cacheWriteInputTokens")
            .or_else(|| u64_at(usage, "cacheWriteInputTokenCount"))
            .unwrap_or(0),
    })
}

/// OpenAI Chat Completions: `prompt_tokens` INCLUDES cached tokens —
/// normalise to the ledger's Anthropic semantics (input = uncached).
fn openai_chat_usage(usage: &Value) -> Option<Usage> {
    let prompt = u64_at(usage, "prompt_tokens")?;
    let cached = usage
        .get("prompt_tokens_details")
        .and_then(|d| u64_at(d, "cached_tokens"))
        .unwrap_or(0)
        .min(prompt);
    Some(Usage {
        input_tokens: prompt - cached,
        output_tokens: u64_at(usage, "completion_tokens").unwrap_or(0),
        cache_read_tokens: cached,
        cache_write_tokens: 0,
    })
}

/// OpenAI Responses: `input_tokens` INCLUDES cached tokens — same
/// normalisation.
fn openai_responses_usage(usage: &Value) -> Option<Usage> {
    let input = u64_at(usage, "input_tokens")?;
    let cached = usage
        .get("input_tokens_details")
        .and_then(|d| u64_at(d, "cached_tokens"))
        .unwrap_or(0)
        .min(input);
    Some(Usage {
        input_tokens: input - cached,
        output_tokens: u64_at(usage, "output_tokens").unwrap_or(0),
        cache_read_tokens: cached,
        cache_write_tokens: 0,
    })
}

/// Usage accumulated by the shared Anthropic SSE state machine.
pub fn usage_from_anthropic_state(state: &crate::sse::anthropic::AnthropicStreamState) -> Usage {
    Usage {
        input_tokens: state.usage.input_tokens,
        output_tokens: state.usage.output_tokens,
        cache_read_tokens: state.usage.cache_read_input_tokens,
        cache_write_tokens: state.usage.cache_creation_input_tokens,
    }
}

/// Fold one streaming frame's JSON payload into a usage accumulator.
/// Handles every Bedrock/Anthropic stream shape (fields are monotone
/// within a stream, so `max` merge is safe against repeats):
///
/// - Anthropic SSE events: `message_start.message.usage`,
///   `message_delta.usage` (snake_case),
/// - Converse-native frames: bare `usage` with camelCase keys
///   (`metadata` events — no Anthropic `type` field at all),
/// - Bedrock InvokeModel EventStream chunks: `{"bytes": "<base64>"}`
///   wrapping an Anthropic SSE event (recursed after decode).
pub fn merge_stream_usage(acc: &mut Usage, payload: &Value) {
    merge_stream_usage_depth(acc, payload, 0);
}

/// Real Bedrock wraps exactly once (`{"bytes": <b64 of the Anthropic
/// event>}`); the recursion exists only for that unwrap. Cap it so a
/// crafted deeply-nested base64 payload can't blow the stack — a
/// stack overflow aborts the whole process, not just this task.
const MAX_BYTES_UNWRAP_DEPTH: u8 = 4;

fn merge_stream_usage_depth(acc: &mut Usage, payload: &Value, depth: u8) {
    // InvokeModel EventStream chunk: base64-wrapped inner event.
    if let Some(b64) = payload.get("bytes").and_then(Value::as_str) {
        if depth >= MAX_BYTES_UNWRAP_DEPTH {
            tracing::warn!(
                event = "stats_capture_bytes_depth_exceeded",
                depth = depth,
                "nested {{\"bytes\": …}} wrapping exceeded the unwrap cap; \
                 ignoring the payload (real Bedrock wraps exactly once)"
            );
            return;
        }
        if let Ok(decoded) = base64::engine::general_purpose::STANDARD.decode(b64) {
            if let Ok(inner) = serde_json::from_slice::<Value>(&decoded) {
                merge_stream_usage_depth(acc, &inner, depth + 1);
            }
        }
        return;
    }
    // message_start carries usage nested under `message`.
    if let Some(u) = payload
        .get("message")
        .and_then(|m| m.get("usage"))
        .and_then(snake_usage)
    {
        merge_max(acc, u);
    }
    // message_delta (and some terminal frames) carry a bare `usage`.
    if let Some(usage) = payload.get("usage") {
        if let Some(u) = snake_usage(usage).or_else(|| camel_usage(usage)) {
            merge_max(acc, u);
        }
    }
}

/// Take the field-wise maximum of two usage snapshots (usage fields
/// are monotone within a stream, so `max` merge is repeat-safe).
pub(crate) fn merge_max(acc: &mut Usage, u: Usage) {
    acc.input_tokens = acc.input_tokens.max(u.input_tokens);
    acc.output_tokens = acc.output_tokens.max(u.output_tokens);
    acc.cache_read_tokens = acc.cache_read_tokens.max(u.cache_read_tokens);
    acc.cache_write_tokens = acc.cache_write_tokens.max(u.cache_write_tokens);
}

/// Bounded tee → JSON-body usage capture. Returns the sender to tee
/// response chunks into; finalizes `pending` when the channel
/// closes (i.e. the response body finished or the client hung up).
pub fn spawn_json_usage_capture(
    shape: ResponseShape,
    mut pending: PendingRecord,
) -> tokio::sync::mpsc::Sender<bytes::Bytes> {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<bytes::Bytes>(TEE_QUEUE_DEPTH);
    tokio::spawn(async move {
        let mut buf: Vec<u8> = Vec::new();
        let mut truncated = false;
        while let Some(chunk) = rx.recv().await {
            if truncated {
                continue; // keep draining so the tee never backpressures
            }
            if buf.len() + chunk.len() > JSON_CAPTURE_CAP {
                // No silent fallbacks: the request still records, but
                // its usage will read zero — say why, once.
                tracing::warn!(
                    event = "stats_capture_body_too_large",
                    cap_bytes = JSON_CAPTURE_CAP,
                    "response body exceeded the usage-capture buffer; \
                     this request records with compression-side numbers \
                     only (tokens/USD from usage read as 0)"
                );
                truncated = true;
                buf.clear();
                continue;
            }
            buf.extend_from_slice(&chunk);
        }
        let mut usage = Usage::default();
        if !truncated {
            match serde_json::from_slice::<Value>(&buf) {
                Ok(v) => {
                    if let Some(model) = v.get("model").and_then(Value::as_str) {
                        pending.set_model_if_unknown(model);
                    }
                    // A 2xx from an LLM endpoint carries a usage block
                    // in every shape we know. A miss means the wire
                    // format moved (or this isn't the body we think it
                    // is), and the request would silently record $0 —
                    // the phantom-zero this module exists to avoid.
                    // Same no-silent-fallbacks rule as the cap above.
                    match usage_from_response_json(shape, &v) {
                        Some(u) => usage = u,
                        None => tracing::warn!(
                            event = "stats_capture_no_usage_block",
                            shape = ?shape,
                            "response parsed as JSON but carried no recognisable \
                             usage block; recording this request with zero tokens \
                             and $0 spend"
                        ),
                    }
                }
                Err(e) => tracing::warn!(
                    event = "stats_capture_body_not_json",
                    error = %e,
                    "response body did not parse as JSON; recording this request \
                     with zero tokens and $0 spend"
                ),
            }
        }
        pending.finalize(usage);
    });
    tx
}

/// Bounded tee → Bedrock binary EventStream usage capture, for the
/// passthrough mode where no other parser sees the frames. CRC
/// validation is off — this is the telemetry side; enforcement
/// happens (or not, per config) on the byte path.
pub fn spawn_eventstream_usage_capture(
    pending: PendingRecord,
) -> tokio::sync::mpsc::Sender<bytes::Bytes> {
    use crate::bedrock::eventstream::{CrcValidation, EventStreamParser};
    let (tx, mut rx) = tokio::sync::mpsc::channel::<bytes::Bytes>(TEE_QUEUE_DEPTH);
    tokio::spawn(async move {
        let mut parser = EventStreamParser::new().with_crc_validation(CrcValidation::No);
        let mut usage = Usage::default();
        'outer: while let Some(chunk) = rx.recv().await {
            parser.push(&chunk);
            loop {
                match parser.next_message() {
                    Ok(Some(msg)) => {
                        if let Ok(payload) = serde_json::from_slice::<Value>(&msg.payload) {
                            merge_stream_usage(&mut usage, &payload);
                        }
                    }
                    Ok(None) => break,
                    Err(_) => {
                        // Telemetry parser out of sync — stop and
                        // record what accumulated so far. The channel
                        // is simply dropped; the tee's `try_send` on a
                        // closed channel is a no-op, so the byte path
                        // never notices.
                        break 'outer;
                    }
                }
            }
        }
        pending.finalize(usage);
    });
    tx
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn anthropic_json_usage_parses() {
        let v = json!({"usage": {"input_tokens": 10, "output_tokens": 5,
            "cache_read_input_tokens": 7, "cache_creation_input_tokens": 3}});
        let u = usage_from_response_json(ResponseShape::Anthropic, &v).unwrap();
        assert_eq!(
            u,
            Usage {
                input_tokens: 10,
                output_tokens: 5,
                cache_read_tokens: 7,
                cache_write_tokens: 3
            }
        );
    }

    #[test]
    fn openai_chat_normalises_cached_out_of_prompt() {
        let v = json!({"usage": {"prompt_tokens": 100, "completion_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 60}}});
        let u = usage_from_response_json(ResponseShape::OpenAiChat, &v).unwrap();
        assert_eq!(u.input_tokens, 40, "prompt minus cached");
        assert_eq!(u.cache_read_tokens, 60);
        assert_eq!(u.output_tokens, 20);
        // Pathological cached > prompt clamps instead of underflowing.
        let v = json!({"usage": {"prompt_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 60}}});
        let u = usage_from_response_json(ResponseShape::OpenAiChat, &v).unwrap();
        assert_eq!(u.input_tokens, 0);
        assert_eq!(u.cache_read_tokens, 10);
    }

    #[test]
    fn openai_responses_normalises_cached_out_of_input() {
        let v = json!({"usage": {"input_tokens": 80, "output_tokens": 9,
            "input_tokens_details": {"cached_tokens": 50}}});
        let u = usage_from_response_json(ResponseShape::OpenAiResponses, &v).unwrap();
        assert_eq!(u.input_tokens, 30);
        assert_eq!(u.cache_read_tokens, 50);
    }

    #[test]
    fn bedrock_shape_accepts_snake_and_camel() {
        let snake = json!({"usage": {"input_tokens": 10, "output_tokens": 5}});
        let camel = json!({"usage": {"inputTokens": 11, "outputTokens": 6,
            "cacheReadInputTokens": 4, "cacheWriteInputTokens": 2}});
        assert_eq!(
            usage_from_response_json(ResponseShape::Bedrock, &snake)
                .unwrap()
                .input_tokens,
            10
        );
        let u = usage_from_response_json(ResponseShape::Bedrock, &camel).unwrap();
        assert_eq!(u.input_tokens, 11);
        assert_eq!(u.cache_read_tokens, 4);
        assert_eq!(u.cache_write_tokens, 2);
    }

    /// The cache fields accept the `*InputTokenCount` spelling too.
    /// Untested, this fallback reads as dead code and invites
    /// deletion — at which point any response using that spelling
    /// silently reports zero cache tokens.
    #[test]
    fn bedrock_camel_usage_accepts_token_count_suffix() {
        let v = json!({"usage": {"inputTokens": 11, "outputTokens": 6,
            "cacheReadInputTokenCount": 4, "cacheWriteInputTokenCount": 2}});
        let u = usage_from_response_json(ResponseShape::Bedrock, &v).unwrap();
        assert_eq!(u.cache_read_tokens, 4);
        assert_eq!(u.cache_write_tokens, 2);

        // The unsuffixed spelling wins when both are present.
        let both = json!({"usage": {"inputTokens": 11, "outputTokens": 6,
            "cacheReadInputTokens": 9, "cacheReadInputTokenCount": 4}});
        let u = usage_from_response_json(ResponseShape::Bedrock, &both).unwrap();
        assert_eq!(u.cache_read_tokens, 9);
    }

    #[test]
    fn error_envelope_yields_none() {
        let v = json!({"error": {"type": "overloaded_error"}});
        assert!(usage_from_response_json(ResponseShape::Anthropic, &v).is_none());
        let v = json!({"usage": {"totalTokens": 5}});
        assert!(usage_from_response_json(ResponseShape::Bedrock, &v).is_none());
    }

    #[test]
    fn stream_merge_handles_all_three_wire_shapes() {
        let mut acc = Usage::default();
        // 1. Anthropic message_start (nested under message).
        merge_stream_usage(
            &mut acc,
            &json!({"type": "message_start",
                "message": {"usage": {"input_tokens": 25, "output_tokens": 1,
                    "cache_read_input_tokens": 9}}}),
        );
        // 2. Anthropic message_delta (bare usage, growing output).
        merge_stream_usage(
            &mut acc,
            &json!({"type": "message_delta", "usage": {"output_tokens": 42}}),
        );
        // 3. Converse metadata (camelCase, no `type`).
        merge_stream_usage(
            &mut acc,
            &json!({"usage": {"inputTokens": 25, "outputTokens": 42,
                "cacheWriteInputTokens": 3}}),
        );
        assert_eq!(acc.input_tokens, 25);
        assert_eq!(acc.output_tokens, 42);
        assert_eq!(acc.cache_read_tokens, 9);
        assert_eq!(acc.cache_write_tokens, 3);
    }

    /// A crafted payload nesting `{"bytes": …}` inside itself must
    /// hit the unwrap cap instead of recursing until the stack blows
    /// (a stack overflow aborts the whole process). Real Bedrock
    /// wraps exactly once.
    #[test]
    fn stream_merge_bounds_nested_base64_depth() {
        let engine = &base64::engine::general_purpose::STANDARD;
        // usage buried 10 wraps deep — far past the cap.
        let mut v = json!({"type": "message_delta", "usage": {"output_tokens": 99}});
        for _ in 0..10 {
            v = json!({"bytes": engine.encode(v.to_string())});
        }
        let mut acc = Usage::default();
        merge_stream_usage(&mut acc, &v); // must return, not overflow
        assert_eq!(
            acc.output_tokens, 0,
            "usage beyond the unwrap cap is ignored, not chased"
        );
        // …while the real single-wrap shape still works.
        let single = json!({"bytes": engine.encode(
            json!({"type": "message_delta", "usage": {"output_tokens": 7}}).to_string()
        )});
        merge_stream_usage(&mut acc, &single);
        assert_eq!(acc.output_tokens, 7);
    }

    #[test]
    fn stream_merge_decodes_invoke_model_base64_chunks() {
        let inner = json!({"type": "message_delta",
            "usage": {"output_tokens": 17, "input_tokens": 12}});
        let b64 =
            base64::engine::general_purpose::STANDARD.encode(serde_json::to_vec(&inner).unwrap());
        let mut acc = Usage::default();
        merge_stream_usage(&mut acc, &json!({"bytes": b64, "p": "pad"}));
        assert_eq!(acc.output_tokens, 17);
        assert_eq!(acc.input_tokens, 12);
        // Garbage base64 is ignored, never panics.
        merge_stream_usage(&mut acc, &json!({"bytes": "!!!not-base64!!!"}));
        assert_eq!(acc.output_tokens, 17);
    }
}
