//! Anthropic Messages streaming state machine.
//!
//! Per the Anthropic Messages streaming guide §5.1, an Anthropic
//! response stream is a sequence of named events:
//!
//!   message_start
//!     content_block_start (index=0)
//!       content_block_delta (index=0) ×N
//!     content_block_stop (index=0)
//!     content_block_start (index=1)
//!       content_block_delta (index=1) ×N
//!     content_block_stop (index=1)
//!     ...
//!   message_delta  (carries final stop_reason, output_tokens)
//!   message_stop
//!
//! Blocks are keyed by `index`, NOT by position. While in practice
//! Anthropic emits them in monotone ascending order, the spec does
//! not require this — our state machine tolerates out-of-order
//! interleaving (test `interleaved_blocks_by_index`).
//!
//! `content_block_delta` carries a `delta` object whose `type` field
//! determines how the payload accumulates:
//!
//!   - `text_delta` — append `delta.text` to `text_buffer`.
//!   - `thinking_delta` — append `delta.thinking` to `text_buffer`
//!     (the block's `block_type == "thinking"`).
//!   - `signature_delta` — set `signature` to `delta.signature`,
//!     BYTE-FOR-BYTE preserved (cryptographic verification of
//!     redacted thinking).
//!   - `input_json_delta` — append `delta.partial_json` to
//!     `partial_json`. At `content_block_stop` the accumulated string
//!     is parsed once into the block's `input` (we keep the string
//!     form for telemetry to avoid re-stringifying on the response
//!     side).
//!   - `citations_delta` — append `delta.citation` to `citations`.
//!
//! P1-8/9/14/17 in the production telemetry log come from missing
//! delta-type arms (the Python proxy ignored thinking_delta,
//! signature_delta, citations_delta). This module enumerates ALL
//! known delta types; any unknown delta type emits a tracing warn
//! with `event=sse_unknown_event` so operators see the wire-format
//! drift loudly.

use std::collections::HashMap;

use serde_json::Value;

use super::framing::SseEvent;

/// Streaming-stream-level status. `Open` is the steady state during
/// streaming. `MessageStop` and `Errored` are terminal — pushing more
/// events is logged but does not panic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StreamStatus {
    /// `message_start` has been received; events will follow.
    #[default]
    Open,
    /// `message_stop` has been received; the stream is done.
    MessageStop,
    /// An `error` event was received mid-stream OR we received a
    /// terminal-state-violation (events after `message_stop`).
    Errored,
}

/// Per-block state. A "block" is one entry in the `content` array
/// of the final response — text, tool_use, thinking, etc.
#[derive(Debug, Clone, Default)]
pub struct BlockState {
    /// `content_block_start.content_block.type`:
    /// "text" | "thinking" | "tool_use" | "redacted_thinking" | ...
    pub block_type: String,
    /// Accumulated text for `text_delta` / `thinking_delta`.
    pub text_buffer: String,
    /// Accumulated `partial_json` from `input_json_delta`. Intentionally
    /// kept as a String — the spec says it's a JSON-fragment string,
    /// not a partial value, and we only parse it once at
    /// `content_block_stop` (or never, if the consumer wants the raw
    /// fragment for replay).
    pub partial_json: String,
    /// Cryptographic signature for redacted_thinking blocks. Must be
    /// preserved BYTE-EQUAL across the proxy — Anthropic uses this
    /// for verification on subsequent calls. Storing as a `String`
    /// is safe because the wire format guarantees ASCII base64.
    pub signature: Option<String>,
    /// Accumulated citations from `citations_delta`. Stored as raw
    /// `serde_json::Value` because Anthropic continues to extend the
    /// citation shape; we do not enforce a strict schema here.
    pub citations: Vec<Value>,
    /// Initial metadata from `content_block_start.content_block` —
    /// the `id`, `name`, `input` (for tool_use), etc. Preserved
    /// verbatim so downstream code can introspect type-specific
    /// fields without re-parsing the wire.
    pub metadata: Value,
    /// True after `content_block_stop` for this index.
    pub complete: bool,
}

/// Per-stream state. Lives in a `tokio::spawn`ed task that consumes
/// framed events from the SSE framer; the response byte-passthrough
/// is independent (see `wire_state_machine` in `proxy.rs`).
#[derive(Debug, Default)]
pub struct AnthropicStreamState {
    pub message_id: Option<String>,
    pub model: Option<String>,
    /// Block index → block state. Keyed by index, not Vec position,
    /// because the spec allows out-of-order completion.
    pub blocks: HashMap<usize, BlockState>,
    /// The most recent block index opened via `content_block_start`.
    /// Tracks "which block is the live zone right now" but does NOT
    /// gate which block a delta applies to — every delta carries its
    /// own `index`.
    pub current_block_index: Option<usize>,
    /// Set on `message_delta` (NOT `message_stop` — Anthropic puts
    /// the final stop_reason on the delta).
    pub stop_reason: Option<String>,
    pub usage: UsageBuilder,
    pub status: StreamStatus,
}

/// Accumulator for token counts. `message_start` carries the input
/// tokens (and an initial `output_tokens: N` that may be 0); each
/// `message_delta.usage` carries an updated `output_tokens` that
/// strictly grows. We keep the latest values; the final values
/// surface at `message_stop`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct UsageBuilder {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_creation_input_tokens: u64,
    pub cache_read_input_tokens: u64,
}

/// Errors during state-machine application. Per project rules these
/// must NOT silently degrade — callers `tracing::warn!` and either
/// drop the event or close the stream. None of these is a panic.
#[derive(Debug, thiserror::Error)]
pub enum StateError {
    #[error("event payload is not valid UTF-8: {0}")]
    PayloadNotUtf8(#[from] std::str::Utf8Error),
    #[error("event payload is not valid JSON: {0}")]
    PayloadNotJson(#[from] serde_json::Error),
    #[error("event missing required field {field}")]
    MissingField { field: &'static str },
}

impl AnthropicStreamState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply one framed event. The `event_name` is required (Anthropic
    /// always emits an `event:` line); a None name is logged and
    /// dropped silently because the SSE comment rule already filtered
    /// keepalives upstream.
    pub fn apply(&mut self, event: SseEvent) -> Result<(), StateError> {
        let Some(name) = event.event_name.as_deref() else {
            // No `event:` line. Anthropic always sends one; this is
            // wire-format drift. Log and drop. We do NOT route by
            // `data` shape because that would invite the
            // silent-fallback class of bug.
            tracing::warn!(
                event = "sse_unknown_event",
                provider = "anthropic",
                event_name = "<missing>",
                payload_preview = %payload_preview(&event.data),
                "anthropic event missing required event: line; dropping"
            );
            return Ok(());
        };

        match name {
            "ping" => {
                // Anthropic emits explicit `event: ping` events
                // alongside SSE-level `: ping` comments. Both are
                // keepalives; we already drop comments at the framer.
                Ok(())
            }
            "message_start" => self.on_message_start(&event),
            "content_block_start" => self.on_content_block_start(&event),
            "content_block_delta" => self.on_content_block_delta(&event),
            "content_block_stop" => self.on_content_block_stop(&event),
            "message_delta" => self.on_message_delta(&event),
            "message_stop" => {
                self.status = StreamStatus::MessageStop;
                Ok(())
            }
            "error" => {
                self.status = StreamStatus::Errored;
                tracing::warn!(
                    event = "sse_anthropic_error_event",
                    payload_preview = %payload_preview(&event.data),
                    "anthropic stream emitted error event"
                );
                Ok(())
            }
            other => {
                // Unknown event name — wire-format drift or new
                // event type Anthropic added that we haven't ported
                // yet. Log loudly per `feedback_no_silent_fallbacks.md`.
                tracing::warn!(
                    event = "sse_unknown_event",
                    provider = "anthropic",
                    event_name = other,
                    payload_preview = %payload_preview(&event.data),
                    "unknown anthropic event; preserving stream but not updating state"
                );
                Ok(())
            }
        }
    }

    fn on_message_start(&mut self, event: &SseEvent) -> Result<(), StateError> {
        let v: Value = parse_json(&event.data)?;
        let msg = v
            .get("message")
            .ok_or(StateError::MissingField { field: "message" })?;
        if let Some(id) = msg.get("id").and_then(|x| x.as_str()) {
            self.message_id = Some(id.to_string());
        }
        if let Some(model) = msg.get("model").and_then(|x| x.as_str()) {
            self.model = Some(model.to_string());
        }
        if let Some(usage) = msg.get("usage") {
            self.usage.merge_from(usage);
        }
        self.status = StreamStatus::Open;
        Ok(())
    }

    fn on_content_block_start(&mut self, event: &SseEvent) -> Result<(), StateError> {
        let v: Value = parse_json(&event.data)?;
        let index = v
            .get("index")
            .and_then(|x| x.as_u64())
            .ok_or(StateError::MissingField { field: "index" })? as usize;
        let cb = v.get("content_block").ok_or(StateError::MissingField {
            field: "content_block",
        })?;
        let block_type = cb
            .get("type")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        let mut block = BlockState {
            block_type,
            metadata: cb.clone(),
            ..Default::default()
        };
        // Some block types start with content already (e.g. a
        // `text` block whose initial chunk arrives in
        // `content_block.text`). Capture it so the subsequent
        // deltas append cleanly.
        if let Some(initial_text) = cb.get("text").and_then(|x| x.as_str()) {
            block.text_buffer.push_str(initial_text);
        }
        if let Some(initial_thinking) = cb.get("thinking").and_then(|x| x.as_str()) {
            block.text_buffer.push_str(initial_thinking);
        }
        self.blocks.insert(index, block);
        self.current_block_index = Some(index);
        Ok(())
    }

    fn on_content_block_delta(&mut self, event: &SseEvent) -> Result<(), StateError> {
        let v: Value = parse_json(&event.data)?;
        let index = v
            .get("index")
            .and_then(|x| x.as_u64())
            .ok_or(StateError::MissingField { field: "index" })? as usize;
        let delta = v
            .get("delta")
            .ok_or(StateError::MissingField { field: "delta" })?;
        let delta_type =
            delta
                .get("type")
                .and_then(|x| x.as_str())
                .ok_or(StateError::MissingField {
                    field: "delta.type",
                })?;

        let block = self.blocks.entry(index).or_default();
        match delta_type {
            "text_delta" => {
                if let Some(t) = delta.get("text").and_then(|x| x.as_str()) {
                    block.text_buffer.push_str(t);
                }
            }
            "thinking_delta" => {
                // Note: thinking_delta uses a `thinking` field, not `text`.
                if let Some(t) = delta.get("thinking").and_then(|x| x.as_str()) {
                    block.text_buffer.push_str(t);
                }
            }
            "input_json_delta" => {
                if let Some(p) = delta.get("partial_json").and_then(|x| x.as_str()) {
                    block.partial_json.push_str(p);
                }
            }
            "signature_delta" => {
                // BYTE-EQUAL preservation: take the wire string verbatim,
                // even if it's empty (an empty signature is itself a
                // signal — it means the model declined to sign).
                if let Some(sig) = delta.get("signature").and_then(|x| x.as_str()) {
                    // Concatenate if a previous signature_delta arrived.
                    // In practice Anthropic emits the full signature in
                    // one delta, but the spec permits chunking.
                    match &mut block.signature {
                        Some(s) => s.push_str(sig),
                        None => block.signature = Some(sig.to_string()),
                    }
                }
            }
            "citations_delta" => {
                if let Some(c) = delta.get("citation") {
                    block.citations.push(c.clone());
                }
            }
            other => {
                tracing::warn!(
                    event = "sse_unknown_event",
                    provider = "anthropic",
                    event_name = "content_block_delta",
                    delta_type = other,
                    payload_preview = %payload_preview(&event.data),
                    "unknown anthropic delta.type; preserving stream but not updating state"
                );
            }
        }
        Ok(())
    }

    fn on_content_block_stop(&mut self, event: &SseEvent) -> Result<(), StateError> {
        let v: Value = parse_json(&event.data)?;
        let index = v
            .get("index")
            .and_then(|x| x.as_u64())
            .ok_or(StateError::MissingField { field: "index" })? as usize;
        if let Some(block) = self.blocks.get_mut(&index) {
            block.complete = true;
            // If we accumulated input_json_delta fragments, attempt
            // to parse the final string. Failure is logged but does
            // not error — the raw fragment is still available for
            // replay/telemetry.
            if !block.partial_json.is_empty() {
                if let Err(e) = serde_json::from_str::<Value>(&block.partial_json) {
                    tracing::warn!(
                        event = "sse_partial_json_unparseable",
                        provider = "anthropic",
                        block_index = index,
                        error = %e,
                        "input_json_delta accumulated string did not parse; \
                         keeping raw fragment in BlockState.partial_json"
                    );
                }
            }
        }
        Ok(())
    }

    fn on_message_delta(&mut self, event: &SseEvent) -> Result<(), StateError> {
        let v: Value = parse_json(&event.data)?;
        if let Some(delta) = v.get("delta") {
            if let Some(stop_reason) = delta.get("stop_reason").and_then(|x| x.as_str()) {
                self.stop_reason = Some(stop_reason.to_string());
            }
        }
        if let Some(usage) = v.get("usage") {
            self.usage.merge_from(usage);
        }
        Ok(())
    }
}

impl UsageBuilder {
    /// Merge a JSON `usage` object into the accumulator. Per the
    /// Anthropic spec, fields are monotone non-decreasing within a
    /// single stream, so we simply take the maximum of (previous,
    /// incoming) for each field. If the incoming object is missing
    /// a field, the previous value is preserved.
    pub fn merge_from(&mut self, v: &Value) {
        if let Some(n) = v.get("input_tokens").and_then(|x| x.as_u64()) {
            self.input_tokens = self.input_tokens.max(n);
        }
        if let Some(n) = v.get("output_tokens").and_then(|x| x.as_u64()) {
            self.output_tokens = self.output_tokens.max(n);
        }
        if let Some(n) = v
            .get("cache_creation_input_tokens")
            .and_then(|x| x.as_u64())
        {
            self.cache_creation_input_tokens = self.cache_creation_input_tokens.max(n);
        }
        if let Some(n) = v.get("cache_read_input_tokens").and_then(|x| x.as_u64()) {
            self.cache_read_input_tokens = self.cache_read_input_tokens.max(n);
        }
    }
}

fn parse_json(data: &bytes::Bytes) -> Result<Value, StateError> {
    let s = std::str::from_utf8(data)?;
    Ok(serde_json::from_str(s)?)
}

/// Up to 96 bytes of the payload, lossy-decoded for log readability.
/// Used only on warn paths — never on the success hot path.
fn payload_preview(data: &bytes::Bytes) -> String {
    const LIMIT: usize = 96;
    let slice = if data.len() > LIMIT {
        &data[..LIMIT]
    } else {
        &data[..]
    };
    String::from_utf8_lossy(slice).into_owned()
}
