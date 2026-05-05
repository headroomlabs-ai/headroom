//! OpenAI Chat Completions streaming state machine.
//!
//! Per OpenAI's streaming spec (and §5.2 of the Headroom realignment
//! guide), a Chat Completions stream is a sequence of `data:` lines
//! with no `event:` field. Each `data:` payload is a JSON `chunk`
//! object whose shape mirrors the non-streaming response, except
//! that:
//!
//! - Each `choices[*]` carries a `delta` instead of a `message`.
//! - The first chunk for a choice carries `delta.role = "assistant"`.
//! - Subsequent chunks carry `delta.content` (string fragments to
//!   concatenate) and/or `delta.tool_calls` (per-tool-call delta
//!   objects keyed by `index`).
//! - For tool calls: ONLY the first chunk for a given index carries
//!   `id` and `function.name`. Subsequent chunks carry just
//!   `function.arguments` to concatenate. The Python proxy didn't
//!   handle this and silently overwrote `arguments` when a chunk
//!   omitted `id` (P4-48 telemetry).
//! - The literal `[DONE]` sentinel marks end-of-stream; no event
//!   payload is parsed.
//! - When `stream_options.include_usage = true`, OpenAI emits a
//!   final chunk with `choices: []` and a populated `usage` object.
//!   Without that flag, no usage is emitted on the stream — the
//!   client must use the non-streaming endpoint for token counts.
//! - The `refusal` field, used by GPT-4o-class safety responses,
//!   carries fragments to concatenate just like `content`.

use serde_json::Value;
use std::collections::HashMap;

use super::framing::SseEvent;

/// Stream-level status mirroring the Anthropic shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StreamStatus {
    #[default]
    Open,
    Done,
    Errored,
}

#[derive(Debug, Default, Clone)]
pub struct ChunkState {
    pub id: Option<String>,
    pub model: Option<String>,
    pub system_fingerprint: Option<String>,
    pub choices: HashMap<usize, ChoiceState>,
    pub usage: Option<Value>,
    pub status: StreamStatus,
}

#[derive(Debug, Default, Clone)]
pub struct ChoiceState {
    pub role: Option<String>,
    pub content: String,
    pub refusal: String,
    pub finish_reason: Option<String>,
    /// Tool calls, keyed by `index` from the wire (NOT vec position).
    pub tool_calls: HashMap<usize, ToolCallState>,
}

#[derive(Debug, Default, Clone)]
pub struct ToolCallState {
    pub id: Option<String>,
    pub call_type: Option<String>,
    pub function_name: Option<String>,
    /// Accumulated `function.arguments` string. Stays as a string
    /// (NOT parsed JSON) because OpenAI's tool-call argument shape
    /// is producer-defined and may legitimately not be JSON-parseable
    /// on partial-stream snapshots.
    pub function_arguments: String,
}

#[derive(Debug, thiserror::Error)]
pub enum StateError {
    #[error("event payload is not valid UTF-8: {0}")]
    PayloadNotUtf8(#[from] std::str::Utf8Error),
    #[error("event payload is not valid JSON: {0}")]
    PayloadNotJson(#[from] serde_json::Error),
}

impl ChunkState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply one framed event. Chat Completions has no `event:` line;
    /// callers feed every framed event in here.
    pub fn apply(&mut self, event: SseEvent) -> Result<(), StateError> {
        if event.is_done_sentinel() {
            self.status = StreamStatus::Done;
            return Ok(());
        }
        // We do warn on `event:` being set because Chat Completions
        // doesn't use it — if a producer (or a reverse proxy in
        // front of us) added one, that's wire-format drift we want
        // to surface, not paper over.
        if let Some(name) = event.event_name.as_deref() {
            tracing::warn!(
                event = "sse_unknown_event",
                provider = "openai_chat",
                event_name = name,
                payload_preview = %payload_preview(&event.data),
                "openai chat event has an event: line; spec says it shouldn't"
            );
        }

        let v: Value = parse_json(&event.data)?;

        // Top-level fields: id, model, system_fingerprint, choices, usage.
        if let Some(id) = v.get("id").and_then(|x| x.as_str()) {
            // Only set on first chunk; later chunks repeat the same id.
            if self.id.is_none() {
                self.id = Some(id.to_string());
            }
        }
        if let Some(m) = v.get("model").and_then(|x| x.as_str()) {
            if self.model.is_none() {
                self.model = Some(m.to_string());
            }
        }
        if let Some(fp) = v.get("system_fingerprint").and_then(|x| x.as_str()) {
            if self.system_fingerprint.is_none() {
                self.system_fingerprint = Some(fp.to_string());
            }
        }

        if let Some(choices) = v.get("choices").and_then(|x| x.as_array()) {
            for choice in choices {
                self.apply_choice(choice);
            }
        }

        // Usage (final chunk when include_usage is set).
        if let Some(usage) = v.get("usage") {
            if !usage.is_null() {
                self.usage = Some(usage.clone());
            }
        }
        Ok(())
    }

    fn apply_choice(&mut self, choice: &Value) {
        let Some(index) = choice.get("index").and_then(|x| x.as_u64()) else {
            // Spec mandates `index`; missing → log and drop the
            // choice (don't fall back to choice[0] silently).
            tracing::warn!(
                event = "sse_unknown_event",
                provider = "openai_chat",
                event_name = "choice_missing_index",
                "choice missing index; dropping"
            );
            return;
        };
        let index = index as usize;
        let cs = self.choices.entry(index).or_default();

        if let Some(delta) = choice.get("delta") {
            if let Some(role) = delta.get("role").and_then(|x| x.as_str()) {
                // Set on first delta only; subsequent deltas omit it.
                if cs.role.is_none() {
                    cs.role = Some(role.to_string());
                }
            }
            if let Some(text) = delta.get("content").and_then(|x| x.as_str()) {
                cs.content.push_str(text);
            }
            if let Some(text) = delta.get("refusal").and_then(|x| x.as_str()) {
                cs.refusal.push_str(text);
            }
            if let Some(tcs) = delta.get("tool_calls").and_then(|x| x.as_array()) {
                for tc in tcs {
                    apply_tool_call_delta(cs, tc);
                }
            }
        }

        if let Some(fr) = choice.get("finish_reason").and_then(|x| x.as_str()) {
            cs.finish_reason = Some(fr.to_string());
        }
    }
}

fn apply_tool_call_delta(cs: &mut ChoiceState, tc: &Value) {
    let Some(idx) = tc.get("index").and_then(|x| x.as_u64()) else {
        tracing::warn!(
            event = "sse_unknown_event",
            provider = "openai_chat",
            event_name = "tool_call_missing_index",
            "tool_call delta missing index; dropping"
        );
        return;
    };
    let entry = cs.tool_calls.entry(idx as usize).or_default();
    // id and function.name only on first chunk for this tool call.
    if let Some(id) = tc.get("id").and_then(|x| x.as_str()) {
        if entry.id.is_none() {
            entry.id = Some(id.to_string());
        }
    }
    if let Some(t) = tc.get("type").and_then(|x| x.as_str()) {
        if entry.call_type.is_none() {
            entry.call_type = Some(t.to_string());
        }
    }
    if let Some(func) = tc.get("function") {
        if let Some(n) = func.get("name").and_then(|x| x.as_str()) {
            if entry.function_name.is_none() {
                entry.function_name = Some(n.to_string());
            }
        }
        if let Some(args) = func.get("arguments").and_then(|x| x.as_str()) {
            entry.function_arguments.push_str(args);
        }
    }
}

fn parse_json(data: &bytes::Bytes) -> Result<Value, StateError> {
    let s = std::str::from_utf8(data)?;
    Ok(serde_json::from_str(s)?)
}

fn payload_preview(data: &bytes::Bytes) -> String {
    const LIMIT: usize = 96;
    let slice = if data.len() > LIMIT {
        &data[..LIMIT]
    } else {
        &data[..]
    };
    String::from_utf8_lossy(slice).into_owned()
}
