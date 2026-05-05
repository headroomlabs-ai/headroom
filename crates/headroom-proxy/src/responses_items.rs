//! Per-item-type parsing for the OpenAI Responses API
//! (`/v1/responses`) — Phase C PR-C3.
//!
//! # Why an explicit enum?
//!
//! The `input` array of a `/v1/responses` request carries items whose
//! shapes diverge sharply by `type`. The Python proxy currently
//! flattens these into Chat-Completions-shape via
//! `headroom/proxy/responses_converter.py` — every new OpenAI item
//! type (Codex `phase`, encrypted reasoning, MCP server-side tools,
//! `apply_patch_call`, V4A diffs, …) silently breaks the converter
//! until someone updates it.
//!
//! C3 ports the request path to Rust with **first-class per-item-type
//! handling**. The rules are:
//!
//! - Items whose payloads are *opaque to the proxy* (encrypted
//!   reasoning, compaction blobs, MCP / computer / web-search /
//!   file-search / code-interpreter call results, image generation
//!   results) are **passthrough** — the bytes flow upstream unchanged
//!   and the proxy never re-serializes them. Re-serializing risks
//!   busting whitespace / key-order / Unicode-escape invariants the
//!   provider's prompt cache may already have keyed against.
//! - Items whose payloads are *output strings* of stateful tool calls
//!   (`function_call_output`, `local_shell_call_output`,
//!   `apply_patch_call_output`) are eligible for live-zone
//!   compression — but only the *latest* of each kind, only above the
//!   2 KiB output-item floor, and only when the per-content-type
//!   compressor agrees the result shrinks the token count.
//! - **Unknown item types** are logged at warn level and preserved
//!   byte-for-byte via `serde_json::value::RawValue`. This is the
//!   no-silent-fallbacks contract — we never strip an item we don't
//!   recognise; a future OpenAI release that lands a new `type` value
//!   keeps flowing through this proxy without any code change.
//!
//! # `RawValue` strategy
//!
//! `serde(other)` on an enum *does* drop the data (it only stores the
//! tag). For byte-faithful preservation we deserialize each item in
//! two passes: first as `&RawValue` so we hold the original byte
//! slice, then as a typed `ResponseItem<'a>` against the same slice.
//! When we want to emit the unknown-warning log we still hold the
//! `RawValue` alongside it — see [`ClassifiedItem`].
//!
//! Per `feedback_realignment_build_constraints.md`: the parser uses
//! serde, not regex; the dispatcher honors per-content-type thresholds
//! from `transforms::live_zone`; structured `tracing` logs name every
//! decision.

use std::borrow::Cow;

use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use serde_json::Value;

/// Wire-shape `local_shell_call.action` payload. The `command` is the
/// argv array — preserving the array structure (rather than joining
/// into a string) is load-bearing for execution-side parity with how
/// the Codex CLI actually invokes processes.
///
/// Note: this typed struct is for **telemetry / decision-making**,
/// not byte preservation. Byte fidelity is provided by the
/// accompanying `&RawValue` slice in [`ClassifiedItem`]. We use
/// `serde_json::Value` for the nested object/array fields here
/// (instead of `RawValue`) because `RawValue`-as-a-struct-field has
/// finicky deserializer-token requirements when nested two levels
/// deep; `Value` always works and the typed struct is never
/// re-serialized to the wire (the outer `RawValue` is what flows
/// upstream).
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct LocalShellAction<'a> {
    /// Always `"exec"` today; future shells (e.g. `"powershell"`) will
    /// land as new variants and we must not collapse them.
    #[serde(default, borrow)]
    pub r#type: Option<Cow<'a, str>>,
    /// argv of the process to launch. **Must remain a JSON array**
    /// upstream — joining into a string changes shell-quoting
    /// semantics. Stored here as `Value` (typed-only); the original
    /// bytes flow through the parent `RawValue`.
    #[serde(default)]
    pub command: Option<Value>,
    /// Working directory.
    #[serde(default, borrow)]
    pub working_directory: Option<Cow<'a, str>>,
    /// Timeout in milliseconds (Codex sets ~5 min default).
    #[serde(default)]
    pub timeout_ms: Option<u64>,
    /// Environment variables, key/value object.
    #[serde(default)]
    pub env: Option<Value>,
    /// Catch-all for forward-compatibility — any new field on
    /// `action` is preserved through the parent's `RawValue`.
    #[serde(default, rename = "with")]
    pub with: Option<Value>,
}

/// `apply_patch_call.operation` carries a V4A unified diff string. The
/// only field we name is `diff`; everything else round-trips via the
/// parent `RawValue`. Concretely, `diff` is the V4A patch body
/// **verbatim** — re-serializing it would change indentation and break
/// the apply.
///
/// As with [`LocalShellAction`], this typed struct is for telemetry;
/// byte preservation comes from the parent `RawValue`.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ApplyPatchOperation<'a> {
    /// Currently always `"apply_patch"`.
    #[serde(default, borrow)]
    pub r#type: Option<Cow<'a, str>>,
    /// V4A diff payload as a JSON string.
    #[serde(default, borrow)]
    pub diff: Option<Cow<'a, str>>,
}

/// Typed view over a `/v1/responses` request input item. The variants
/// we name are the ones whose handling diverges; everything else falls
/// to a [`ClassifiedItem`] with `typed = None` and is preserved
/// byte-for-byte via the accompanying `RawValue`.
///
/// Notes:
/// - `arguments` on `function_call` is a JSON-encoded **string** on
///   the wire; never parse it as JSON inside the proxy. The model
///   built it; the model parses it.
/// - `output` on `*_output` items is a string. Compressors run only
///   on the latest of each kind, and only above the 2 KiB output-item
///   floor (see [`OUTPUT_ITEM_MIN_BYTES`]).
/// - String fields use `Cow<'a, str>` so escape-bearing JSON values
///   (e.g. `"{\"q\":\"hello\"}"`) succeed without allocation on the
///   non-escaped common path.
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type")]
pub enum ResponseItem<'a> {
    /// Conversational message item. `phase` (Codex) is preserved
    /// verbatim — values like `"commentary"` and `"final_answer"` are
    /// load-bearing for Codex routing.
    #[serde(rename = "message")]
    Message {
        #[serde(default, borrow)]
        role: Option<Cow<'a, str>>,
        #[serde(default, borrow)]
        phase: Option<Cow<'a, str>>,
        /// Stringly-typed `content` is rare on Responses; arrays of
        /// content parts are the common shape. Telemetry-only here;
        /// the byte path uses the parent `RawValue`.
        #[serde(default)]
        content: Option<Value>,
        #[serde(default, borrow)]
        id: Option<Cow<'a, str>>,
        #[serde(default, borrow)]
        status: Option<Cow<'a, str>>,
    },

    /// Reasoning item. `encrypted_content` is opaque — passthrough
    /// only. We do not even peek inside.
    #[serde(rename = "reasoning")]
    Reasoning {
        #[serde(default, borrow)]
        id: Option<Cow<'a, str>>,
        #[serde(default, borrow)]
        encrypted_content: Option<Cow<'a, str>>,
        #[serde(default)]
        summary: Option<Value>,
    },

    /// Function tool call. `arguments` is a **string** the model
    /// emitted; never JSON-parsed by the proxy.
    #[serde(rename = "function_call")]
    FunctionCall {
        #[serde(default, borrow)]
        id: Option<Cow<'a, str>>,
        #[serde(default, borrow)]
        call_id: Option<Cow<'a, str>>,
        #[serde(default, borrow)]
        name: Option<Cow<'a, str>>,
        #[serde(default, borrow)]
        arguments: Option<Cow<'a, str>>,
        #[serde(default, borrow)]
        status: Option<Cow<'a, str>>,
    },

    /// Function tool output. `output` is the string the proxy may
    /// compress when this is the latest `function_call_output` and
    /// the bytes exceed the 2 KiB floor.
    #[serde(rename = "function_call_output")]
    FunctionCallOutput {
        #[serde(default, borrow)]
        id: Option<Cow<'a, str>>,
        #[serde(default, borrow)]
        call_id: Option<Cow<'a, str>>,
        #[serde(default, borrow)]
        output: Option<Cow<'a, str>>,
        #[serde(default, borrow)]
        status: Option<Cow<'a, str>>,
    },

    /// Local shell call. `action.command` is an argv array; the
    /// dispatcher must NOT join it into a string.
    #[serde(rename = "local_shell_call")]
    LocalShellCall {
        #[serde(default, borrow)]
        id: Option<Cow<'a, str>>,
        #[serde(default, borrow)]
        call_id: Option<Cow<'a, str>>,
        #[serde(default, borrow)]
        action: Option<LocalShellAction<'a>>,
        #[serde(default, borrow)]
        status: Option<Cow<'a, str>>,
    },

    /// Local shell output (stdout / stderr / exit).
    #[serde(rename = "local_shell_call_output")]
    LocalShellCallOutput {
        #[serde(default, borrow)]
        id: Option<Cow<'a, str>>,
        #[serde(default, borrow)]
        call_id: Option<Cow<'a, str>>,
        #[serde(default, borrow)]
        output: Option<Cow<'a, str>>,
        #[serde(default, borrow)]
        status: Option<Cow<'a, str>>,
    },

    /// Apply-patch call (V4A unified-diff). The diff bytes must NEVER
    /// be re-serialized.
    #[serde(rename = "apply_patch_call")]
    ApplyPatchCall {
        #[serde(default, borrow)]
        id: Option<Cow<'a, str>>,
        #[serde(default, borrow)]
        call_id: Option<Cow<'a, str>>,
        #[serde(default, borrow)]
        operation: Option<ApplyPatchOperation<'a>>,
        #[serde(default, borrow)]
        status: Option<Cow<'a, str>>,
    },

    /// Apply-patch output (the result string after applying the
    /// patch — typically the new file content or an error message).
    #[serde(rename = "apply_patch_call_output")]
    ApplyPatchCallOutput {
        #[serde(default, borrow)]
        id: Option<Cow<'a, str>>,
        #[serde(default, borrow)]
        call_id: Option<Cow<'a, str>>,
        #[serde(default, borrow)]
        output: Option<Cow<'a, str>>,
        #[serde(default, borrow)]
        status: Option<Cow<'a, str>>,
    },

    /// Compaction blob — encrypted, opaque, sticky to cache.
    #[serde(rename = "compaction")]
    Compaction {
        #[serde(default, borrow)]
        id: Option<Cow<'a, str>>,
        #[serde(default, borrow)]
        encrypted_content: Option<Cow<'a, str>>,
    },

    /// Server-side MCP call (function-shaped). Passthrough.
    #[serde(rename = "mcp_call")]
    McpCall {
        #[serde(default, borrow)]
        id: Option<Cow<'a, str>>,
    },

    /// Server-side MCP list-tools call. Passthrough.
    #[serde(rename = "mcp_list_tools")]
    McpListTools {
        #[serde(default, borrow)]
        id: Option<Cow<'a, str>>,
    },

    /// Server-side MCP approval request. Passthrough.
    #[serde(rename = "mcp_approval_request")]
    McpApprovalRequest {
        #[serde(default, borrow)]
        id: Option<Cow<'a, str>>,
    },

    /// Computer-use call — passthrough.
    #[serde(rename = "computer_call")]
    ComputerCall {
        #[serde(default, borrow)]
        id: Option<Cow<'a, str>>,
    },

    /// Computer-use call output — screenshot + status. Passthrough
    /// on the wire.
    #[serde(rename = "computer_call_output")]
    ComputerCallOutput {
        #[serde(default, borrow)]
        id: Option<Cow<'a, str>>,
    },

    /// Hosted web-search tool call. Passthrough.
    #[serde(rename = "web_search_call")]
    WebSearchCall {
        #[serde(default, borrow)]
        id: Option<Cow<'a, str>>,
    },

    /// Hosted file-search tool call. Passthrough.
    #[serde(rename = "file_search_call")]
    FileSearchCall {
        #[serde(default, borrow)]
        id: Option<Cow<'a, str>>,
    },

    /// Hosted code-interpreter tool call. Passthrough.
    #[serde(rename = "code_interpreter_call")]
    CodeInterpreterCall {
        #[serde(default, borrow)]
        id: Option<Cow<'a, str>>,
    },

    /// Hosted image-generation tool call. Passthrough on the wire;
    /// log path redacts `image_data` (size-only) — see
    /// `compression::live_zone_responses::log_item_telemetry`.
    #[serde(rename = "image_generation_call")]
    ImageGenerationCall {
        #[serde(default, borrow)]
        id: Option<Cow<'a, str>>,
    },

    /// Hosted tool-search tool call. Passthrough.
    #[serde(rename = "tool_search_call")]
    ToolSearchCall {
        #[serde(default, borrow)]
        id: Option<Cow<'a, str>>,
    },

    /// Customer-defined custom tool call. Passthrough — we don't
    /// know the argument schema.
    #[serde(rename = "custom_tool_call")]
    CustomToolCall {
        #[serde(default, borrow)]
        id: Option<Cow<'a, str>>,
    },
}

impl<'a> ResponseItem<'a> {
    /// Tag of the variant — useful for structured logs.
    pub fn type_tag(&self) -> &'static str {
        match self {
            ResponseItem::Message { .. } => "message",
            ResponseItem::Reasoning { .. } => "reasoning",
            ResponseItem::FunctionCall { .. } => "function_call",
            ResponseItem::FunctionCallOutput { .. } => "function_call_output",
            ResponseItem::LocalShellCall { .. } => "local_shell_call",
            ResponseItem::LocalShellCallOutput { .. } => "local_shell_call_output",
            ResponseItem::ApplyPatchCall { .. } => "apply_patch_call",
            ResponseItem::ApplyPatchCallOutput { .. } => "apply_patch_call_output",
            ResponseItem::Compaction { .. } => "compaction",
            ResponseItem::McpCall { .. } => "mcp_call",
            ResponseItem::McpListTools { .. } => "mcp_list_tools",
            ResponseItem::McpApprovalRequest { .. } => "mcp_approval_request",
            ResponseItem::ComputerCall { .. } => "computer_call",
            ResponseItem::ComputerCallOutput { .. } => "computer_call_output",
            ResponseItem::WebSearchCall { .. } => "web_search_call",
            ResponseItem::FileSearchCall { .. } => "file_search_call",
            ResponseItem::CodeInterpreterCall { .. } => "code_interpreter_call",
            ResponseItem::ImageGenerationCall { .. } => "image_generation_call",
            ResponseItem::ToolSearchCall { .. } => "tool_search_call",
            ResponseItem::CustomToolCall { .. } => "custom_tool_call",
        }
    }

    /// Is this an `*_output` item the live-zone dispatcher considers
    /// for compression?
    pub fn is_output_item(&self) -> bool {
        matches!(
            self,
            ResponseItem::FunctionCallOutput { .. }
                | ResponseItem::LocalShellCallOutput { .. }
                | ResponseItem::ApplyPatchCallOutput { .. }
        )
    }
}

/// Per-item-type minimum bytes before the live-zone dispatcher even
/// inspects an `*_output` payload. Per spec PR-C3 §scope: 2 KiB.
/// Per-content-type thresholds from `transforms::live_zone` still
/// apply on top of this floor (e.g. logs at 512 B → still skipped
/// because output items must clear 2 KiB first).
pub const OUTPUT_ITEM_MIN_BYTES: usize = 2 * 1024;

/// Two-pass result: a typed view alongside the byte-faithful raw
/// slice. Lifetime ties to the underlying request body. Always
/// preserve the `raw` alongside the typed view; emitting the raw
/// upstream is what guarantees byte-fidelity for unknown / opaque
/// items.
#[derive(Debug)]
pub struct ClassifiedItem<'a> {
    /// Typed parse, when the `type` tag matches a known variant.
    /// `None` for unknown / future item types — those keep flowing
    /// upstream via [`Self::raw`].
    pub typed: Option<ResponseItem<'a>>,
    /// Type tag string (the literal `"type"` field on the JSON
    /// object). Used to log unknown variants by name.
    pub type_tag: &'a str,
    /// Original byte slice for the item. Owned by the request body.
    pub raw: &'a RawValue,
}

/// Helper: extract the `type` tag from a JSON object slice without
/// fully parsing the payload. Returns the borrowed string slice into
/// the input.
#[derive(Deserialize)]
struct TypeOnly<'a> {
    #[serde(borrow, default)]
    r#type: Option<&'a str>,
}

/// Two-pass classification: each item is parsed first as `&RawValue`,
/// then we read the `type` tag and try the typed parse. Errors on the
/// typed parse demote to `Unknown` (we still have the raw slice).
///
/// # Errors
///
/// Returns an error only when the `items` array shape is wrong (not a
/// JSON array) or we couldn't even pluck the `type` tag — in that
/// case the caller falls through to passthrough (no compression).
pub fn classify_items<'a>(
    items_raw: &'a RawValue,
) -> Result<Vec<ClassifiedItem<'a>>, ClassifyError> {
    let raw_items: Vec<&'a RawValue> =
        serde_json::from_str(items_raw.get()).map_err(|_| ClassifyError::ItemsNotArray)?;

    let mut out = Vec::with_capacity(raw_items.len());
    for raw in raw_items {
        let type_only: TypeOnly<'a> =
            serde_json::from_str(raw.get()).map_err(|_| ClassifyError::ItemMissingTypeTag)?;
        let type_tag = type_only.r#type.unwrap_or("");
        // Try typed parse. If it fails (unknown type, or known type
        // with a malformed payload), demote to typed=None — the raw
        // slice still flows upstream verbatim.
        let typed: Option<ResponseItem<'a>> = serde_json::from_str(raw.get()).ok();
        out.push(ClassifiedItem {
            typed,
            type_tag,
            raw,
        });
    }
    Ok(out)
}

/// Classification failure. Both variants imply the proxy falls back
/// to passthrough — we never strip items we can't classify.
#[derive(Debug, thiserror::Error)]
pub enum ClassifyError {
    /// `input` (or `items`) is not a JSON array.
    #[error("response items field is not a JSON array")]
    ItemsNotArray,
    /// One of the items is missing the `type` tag entirely. Without
    /// a tag we cannot even log a useful warn message.
    #[error("response item missing `type` tag")]
    ItemMissingTypeTag,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn raw(v: serde_json::Value) -> Box<RawValue> {
        RawValue::from_string(v.to_string()).unwrap()
    }

    #[test]
    fn function_call_arguments_string() {
        let r = raw(json!({
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "search",
            "arguments": "{\"q\":\"hello\"}",
        }));
        let parsed: ResponseItem = serde_json::from_str(r.get()).unwrap();
        match parsed {
            ResponseItem::FunctionCall { arguments, .. } => {
                // Critical: arguments stays as STRING (the model's
                // serialized JSON, not parsed).
                assert_eq!(arguments.as_deref(), Some("{\"q\":\"hello\"}"));
            }
            other => panic!("expected FunctionCall, got {other:?}"),
        }
    }

    #[test]
    fn local_shell_command_array_preserved() {
        // Wire-shape preservation: the BYTES on the wire keep the
        // command as a JSON array. The typed view exposes a Value
        // (telemetry path); the byte path uses ClassifiedItem.raw.
        let r = raw(json!({
            "type": "local_shell_call",
            "id": "ls_1",
            "call_id": "call_1",
            "action": {
                "type": "exec",
                "command": ["bash", "-c", "ls -la"],
            }
        }));
        // Byte-level: r.get() must contain the command array
        // verbatim (not stringified).
        let bytes = r.get();
        assert!(bytes.contains(r#""command":["bash","-c","ls -la"]"#));
        // Typed-level: the command is parsed as a JSON array.
        let parsed: ResponseItem = serde_json::from_str(bytes).unwrap();
        match parsed {
            ResponseItem::LocalShellCall { action, .. } => {
                let cmd = action.unwrap().command.unwrap();
                assert!(cmd.is_array(), "command must be a JSON array, got: {cmd}");
                let arr = cmd.as_array().unwrap();
                assert_eq!(arr.len(), 3);
                assert_eq!(arr[0], json!("bash"));
                assert_eq!(arr[1], json!("-c"));
                assert_eq!(arr[2], json!("ls -la"));
            }
            other => panic!("expected LocalShellCall, got {other:?}"),
        }
    }

    #[test]
    fn unknown_type_demotes_to_none() {
        let body = json!({
            "input": [
                {"type": "future_item_type_v2", "novel_field": "value"}
            ]
        });
        let raw_input = raw(body["input"].clone());
        let classified = classify_items(&raw_input).unwrap();
        assert_eq!(classified.len(), 1);
        assert_eq!(classified[0].type_tag, "future_item_type_v2");
        assert!(classified[0].typed.is_none());
    }

    #[test]
    fn classify_message_with_phase() {
        let body = json!({
            "input": [
                {"type": "message", "role": "assistant", "phase": "commentary",
                 "content": [{"type": "output_text", "text": "thinking"}]}
            ]
        });
        let raw_input = raw(body["input"].clone());
        let classified = classify_items(&raw_input).unwrap();
        match classified[0].typed.as_ref().unwrap() {
            ResponseItem::Message { phase, .. } => {
                assert_eq!(phase.as_deref(), Some("commentary"));
            }
            _ => panic!("expected message"),
        }
    }

    #[test]
    fn is_output_item_correct() {
        let r = raw(json!({"type": "function_call_output", "output": "x"}));
        let p: ResponseItem = serde_json::from_str(r.get()).unwrap();
        assert!(p.is_output_item());

        let r = raw(json!({"type": "reasoning"}));
        let p: ResponseItem = serde_json::from_str(r.get()).unwrap();
        assert!(!p.is_output_item());
    }
}
