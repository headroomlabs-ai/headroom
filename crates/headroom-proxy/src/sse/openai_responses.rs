//! OpenAI Responses streaming state machine.
//!
//! Per OpenAI's Responses streaming spec (and §5.3 of the Headroom
//! realignment guide), the Responses stream uses **named events**
//! (an `event:` SSE line per event), unlike Chat Completions. Each
//! event corresponds to a structured update on the response object.
//!
//! Top-level events we handle:
//!
//!   response.created                 — initial envelope; carries `response.id`.
//!   response.in_progress             — periodic status; usually no state change.
//!   output_item.added                — a new output item appears at `item.id`.
//!   output_item.done                 — that item is complete (final shape in
//!                                      `item`). Items can complete in any order
//!                                      (parallel reasoning + function_call).
//!   content_part.added               — a content_part entry under an item.
//!   content_part.done                — that content_part is complete.
//!   output_text.delta                — text fragment to append to a message item.
//!   output_text.done                 — text is finished.
//!   function_call_arguments.delta    — fragment of a function_call's `arguments`
//!                                      string. STAYS A STRING end-to-end.
//!   function_call_arguments.done     — done; `arguments` final value.
//!   reasoning_summary.delta          — fragment of a reasoning summary.
//!   reasoning_summary.done           — summary complete.
//!   response.completed               — final usage + status.
//!   response.failed                  — error; status = Errored.
//!   response.incomplete              — incomplete (usually max_output_tokens).
//!
//! P1-17 telemetry: items keyed by position broke when OpenAI's spec
//! permitted out-of-order completion (item 1 completed before item 0).
//! This module keys EVERYTHING by `item.id` (a string like
//! `msg_abc123`), never by position. Test
//! `out_of_order_item_completion_by_id` enforces this.

use serde_json::Value;
use std::collections::HashMap;

use super::framing::SseEvent;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StreamStatus {
    #[default]
    Open,
    Completed,
    Failed,
    Incomplete,
}

#[derive(Debug, Default, Clone)]
pub struct ResponseState {
    pub response_id: Option<String>,
    pub model: Option<String>,
    /// Items keyed by their wire `id` field (NOT position). Out-of-
    /// order completion is allowed by spec.
    pub items: HashMap<String, ItemState>,
    pub usage: Option<Value>,
    pub status: StreamStatus,
}

#[derive(Debug, Default, Clone)]
pub struct ItemState {
    pub item_type: String,
    /// Concatenated `output_text` for message-type items.
    pub output_text: String,
    /// Concatenated `reasoning_summary` deltas.
    pub reasoning_summary: String,
    /// Concatenated `function_call_arguments`. Stays as a STRING —
    /// arguments may be JSON but the proxy never re-parses it.
    pub function_call_arguments: String,
    /// Initial metadata captured at output_item.added.
    pub metadata: Value,
    /// True after output_item.done for this id.
    pub complete: bool,
    /// Per-content_part state, keyed by part's `index`. We do not
    /// presently surface fields from the part beyond what the Anthropic
    /// state-machine surfaces — this keeps the structure small.
    pub content_parts: HashMap<usize, ContentPartState>,
}

#[derive(Debug, Default, Clone)]
pub struct ContentPartState {
    pub part_type: String,
    pub text: String,
    pub complete: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum StateError {
    #[error("event payload is not valid UTF-8: {0}")]
    PayloadNotUtf8(#[from] std::str::Utf8Error),
    #[error("event payload is not valid JSON: {0}")]
    PayloadNotJson(#[from] serde_json::Error),
    #[error("event missing required field {field}")]
    MissingField { field: &'static str },
}

impl ResponseState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply one framed event. The `event:` line is required (the
    /// Responses spec mandates one per event); a None name is logged
    /// and dropped.
    pub fn apply(&mut self, event: SseEvent) -> Result<(), StateError> {
        let Some(name) = event.event_name.as_deref() else {
            tracing::warn!(
                event = "sse_unknown_event",
                provider = "openai_responses",
                event_name = "<missing>",
                payload_preview = %payload_preview(&event.data),
                "openai responses event missing event: line; dropping"
            );
            return Ok(());
        };

        let v: Value = parse_json(&event.data)?;
        match name {
            "response.created" => self.on_response_created(&v),
            "response.in_progress" => Ok(()),
            "output_item.added" => self.on_output_item_added(&v),
            "output_item.done" => self.on_output_item_done(&v),
            "content_part.added" => self.on_content_part_added(&v),
            "content_part.done" => self.on_content_part_done(&v),
            "response.output_text.delta" | "output_text.delta" => self.on_output_text_delta(&v),
            "response.output_text.done" | "output_text.done" => self.on_output_text_done(&v),
            "response.function_call_arguments.delta" | "function_call_arguments.delta" => {
                self.on_function_call_arguments_delta(&v)
            }
            "response.function_call_arguments.done" | "function_call_arguments.done" => {
                self.on_function_call_arguments_done(&v)
            }
            "response.reasoning_summary.delta"
            | "response.reasoning_summary_text.delta"
            | "reasoning_summary.delta" => self.on_reasoning_summary_delta(&v),
            "response.reasoning_summary.done"
            | "response.reasoning_summary_text.done"
            | "reasoning_summary.done" => self.on_reasoning_summary_done(&v),
            "response.completed" => self.on_response_completed(&v),
            "response.failed" => {
                self.status = StreamStatus::Failed;
                Ok(())
            }
            "response.incomplete" => {
                self.status = StreamStatus::Incomplete;
                Ok(())
            }
            other => {
                tracing::warn!(
                    event = "sse_unknown_event",
                    provider = "openai_responses",
                    event_name = other,
                    payload_preview = %payload_preview(&event.data),
                    "unknown openai responses event; preserving stream but not updating state"
                );
                Ok(())
            }
        }
    }

    fn on_response_created(&mut self, v: &Value) -> Result<(), StateError> {
        let resp = v
            .get("response")
            .ok_or(StateError::MissingField { field: "response" })?;
        if let Some(id) = resp.get("id").and_then(|x| x.as_str()) {
            self.response_id = Some(id.to_string());
        }
        if let Some(m) = resp.get("model").and_then(|x| x.as_str()) {
            self.model = Some(m.to_string());
        }
        Ok(())
    }

    fn on_output_item_added(&mut self, v: &Value) -> Result<(), StateError> {
        let item = v
            .get("item")
            .ok_or(StateError::MissingField { field: "item" })?;
        let id = item
            .get("id")
            .and_then(|x| x.as_str())
            .ok_or(StateError::MissingField { field: "item.id" })?
            .to_string();
        let item_type = item
            .get("type")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        self.items.insert(
            id,
            ItemState {
                item_type,
                metadata: item.clone(),
                ..Default::default()
            },
        );
        Ok(())
    }

    fn on_output_item_done(&mut self, v: &Value) -> Result<(), StateError> {
        let item = v
            .get("item")
            .ok_or(StateError::MissingField { field: "item" })?;
        let id = item
            .get("id")
            .and_then(|x| x.as_str())
            .ok_or(StateError::MissingField { field: "item.id" })?;
        // Out-of-order completion: the item may not have an
        // output_item.added if the producer is exotic; insert-or-update.
        let entry = self.items.entry(id.to_string()).or_default();
        entry.complete = true;
        // Refresh metadata to the final shape — the `done` payload
        // is authoritative.
        entry.metadata = item.clone();
        if entry.item_type.is_empty() {
            if let Some(t) = item.get("type").and_then(|x| x.as_str()) {
                entry.item_type = t.to_string();
            }
        }
        Ok(())
    }

    fn on_content_part_added(&mut self, v: &Value) -> Result<(), StateError> {
        let item_id = v
            .get("item_id")
            .and_then(|x| x.as_str())
            .ok_or(StateError::MissingField { field: "item_id" })?
            .to_string();
        let part_index = v
            .get("content_index")
            .or_else(|| v.get("part_index"))
            .and_then(|x| x.as_u64())
            .ok_or(StateError::MissingField {
                field: "content_index",
            })? as usize;
        let part = v
            .get("part")
            .ok_or(StateError::MissingField { field: "part" })?;
        let part_type = part
            .get("type")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        let item = self.items.entry(item_id).or_default();
        item.content_parts.insert(
            part_index,
            ContentPartState {
                part_type,
                ..Default::default()
            },
        );
        Ok(())
    }

    fn on_content_part_done(&mut self, v: &Value) -> Result<(), StateError> {
        let item_id = v
            .get("item_id")
            .and_then(|x| x.as_str())
            .ok_or(StateError::MissingField { field: "item_id" })?;
        let part_index = v
            .get("content_index")
            .or_else(|| v.get("part_index"))
            .and_then(|x| x.as_u64())
            .ok_or(StateError::MissingField {
                field: "content_index",
            })? as usize;
        if let Some(item) = self.items.get_mut(item_id) {
            if let Some(part) = item.content_parts.get_mut(&part_index) {
                part.complete = true;
            }
        }
        Ok(())
    }

    fn on_output_text_delta(&mut self, v: &Value) -> Result<(), StateError> {
        let item_id = v
            .get("item_id")
            .and_then(|x| x.as_str())
            .ok_or(StateError::MissingField { field: "item_id" })?
            .to_string();
        let delta = v
            .get("delta")
            .and_then(|x| x.as_str())
            .ok_or(StateError::MissingField { field: "delta" })?;
        let item = self.items.entry(item_id).or_default();
        item.output_text.push_str(delta);
        // Also append to the addressed content_part if present.
        if let Some(part_index) = v
            .get("content_index")
            .or_else(|| v.get("part_index"))
            .and_then(|x| x.as_u64())
        {
            if let Some(part) = item.content_parts.get_mut(&(part_index as usize)) {
                part.text.push_str(delta);
            }
        }
        Ok(())
    }

    fn on_output_text_done(&mut self, _v: &Value) -> Result<(), StateError> {
        // The `done` event carries the final aggregated `text`. We
        // could overwrite output_text from this, but the delta-sum
        // is already authoritative and matches the wire — we keep it
        // for robustness.
        Ok(())
    }

    fn on_function_call_arguments_delta(&mut self, v: &Value) -> Result<(), StateError> {
        let item_id = v
            .get("item_id")
            .and_then(|x| x.as_str())
            .ok_or(StateError::MissingField { field: "item_id" })?
            .to_string();
        let delta = v
            .get("delta")
            .and_then(|x| x.as_str())
            .ok_or(StateError::MissingField { field: "delta" })?;
        let item = self.items.entry(item_id).or_default();
        item.function_call_arguments.push_str(delta);
        Ok(())
    }

    fn on_function_call_arguments_done(&mut self, v: &Value) -> Result<(), StateError> {
        // The done event carries the final `arguments` string. Replace
        // our accumulator with it iff present and nonempty — guarantees
        // a producer that sends ONLY the done event (no deltas) still
        // ends up with the right value.
        let item_id = v
            .get("item_id")
            .and_then(|x| x.as_str())
            .ok_or(StateError::MissingField { field: "item_id" })?
            .to_string();
        if let Some(args) = v.get("arguments").and_then(|x| x.as_str()) {
            let item = self.items.entry(item_id).or_default();
            // Only overwrite if our delta accumulator is empty;
            // otherwise the deltas are authoritative (they may
            // include trailing content the `done` shape elides).
            if item.function_call_arguments.is_empty() {
                item.function_call_arguments.push_str(args);
            }
        }
        Ok(())
    }

    fn on_reasoning_summary_delta(&mut self, v: &Value) -> Result<(), StateError> {
        let item_id = v
            .get("item_id")
            .and_then(|x| x.as_str())
            .ok_or(StateError::MissingField { field: "item_id" })?
            .to_string();
        let delta = v
            .get("delta")
            .and_then(|x| x.as_str())
            .ok_or(StateError::MissingField { field: "delta" })?;
        let item = self.items.entry(item_id).or_default();
        item.reasoning_summary.push_str(delta);
        Ok(())
    }

    fn on_reasoning_summary_done(&mut self, _v: &Value) -> Result<(), StateError> {
        Ok(())
    }

    fn on_response_completed(&mut self, v: &Value) -> Result<(), StateError> {
        self.status = StreamStatus::Completed;
        if let Some(resp) = v.get("response") {
            if let Some(usage) = resp.get("usage") {
                if !usage.is_null() {
                    self.usage = Some(usage.clone());
                }
            }
        }
        Ok(())
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
