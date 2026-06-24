//! Tool-pair atomicity rules for the live-zone-only compression
//! architecture.
//!
//! Live-zone compression never drops messages — it operates on
//! content blocks within messages. So the old "which indices are
//! safe to drop" question goes away. What remains, and what this
//! module provides, is the rule that an `assistant.tool_use` and
//! its matching `tool_result` must be treated as one unit when
//! deciding what to compress: compressing one but not the other
//! desynchronizes the conversation and a re-replay of the tool
//! response will mismatch the call id, producing 400s upstream.
//!
//! For the live-zone block dispatcher (`PR-B2`+), this module
//! exposes [`tool_pair_indices`] — given a slice of messages, it
//! returns the index pairs that must be co-treated.
//!
//! Both OpenAI and Anthropic tool-call shapes are recognized:
//!
//! - **OpenAI**: `assistant.tool_calls[i].id` ↔ `tool.tool_call_id`.
//! - **Anthropic**: `assistant.content[].type=="tool_use".id`
//!   ↔ `user.content[].type=="tool_result".tool_use_id`.

use std::collections::{HashMap, HashSet};

use serde_json::Value;

/// One paired (assistant_tool_use_index, tool_response_index) entry.
/// Either index may appear in multiple pairs if a single assistant
/// message issues multiple `tool_use` blocks that resolve in the
/// same following user message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ToolPair {
    pub assistant_index: usize,
    pub response_index: usize,
}

/// Return every (assistant, response) tool-call pair in the message
/// list. Used by the live-zone dispatcher to keep tool_use and its
/// tool_result on the same compression decision.
pub fn tool_pair_indices(messages: &[Value]) -> Vec<ToolPair> {
    // Map from tool_call id → assistant index that announced it.
    let mut announced: HashMap<String, usize> = HashMap::new();
    for (i, msg) in messages.iter().enumerate() {
        if msg.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        for id in collect_assistant_tool_call_ids(msg) {
            announced.insert(id, i);
        }
    }

    let mut pairs: Vec<ToolPair> = Vec::new();
    let mut seen: HashSet<(usize, usize)> = HashSet::new();
    for (i, msg) in messages.iter().enumerate() {
        let role = msg.get("role").and_then(Value::as_str);

        // OpenAI shape: role=tool, single tool_call_id.
        if role == Some("tool") {
            if let Some(tcid) = msg.get("tool_call_id").and_then(Value::as_str) {
                if let Some(&assistant_index) = announced.get(tcid) {
                    if seen.insert((assistant_index, i)) {
                        pairs.push(ToolPair {
                            assistant_index,
                            response_index: i,
                        });
                    }
                }
            }
        }

        // Anthropic shape: role=user, content blocks may contain
        // multiple tool_result entries pointing back to multiple
        // tool_use ids on potentially different assistant messages.
        if role == Some("user") {
            if let Some(blocks) = msg.get("content").and_then(Value::as_array) {
                for block in blocks {
                    if block.get("type").and_then(Value::as_str) == Some("tool_result") {
                        if let Some(tuid) = block.get("tool_use_id").and_then(Value::as_str) {
                            if let Some(&assistant_index) = announced.get(tuid) {
                                if seen.insert((assistant_index, i)) {
                                    pairs.push(ToolPair {
                                        assistant_index,
                                        response_index: i,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    pairs
}

/// Extract the set of tool_call ids announced by an assistant message,
/// handling both OpenAI and Anthropic shapes.
///
/// - **OpenAI**: `{role:assistant, tool_calls: [{id, ...}, ...]}`
/// - **Anthropic**: `{role:assistant, content: [{type:tool_use, id, ...}]}`
fn collect_assistant_tool_call_ids(assistant: &Value) -> HashSet<String> {
    let mut ids: HashSet<String> = HashSet::new();

    if let Some(arr) = assistant.get("tool_calls").and_then(Value::as_array) {
        for tc in arr {
            if let Some(id) = tc.get("id").and_then(Value::as_str) {
                ids.insert(id.to_string());
            }
        }
    }

    if let Some(blocks) = assistant.get("content").and_then(Value::as_array) {
        for block in blocks {
            if block.get("type").and_then(Value::as_str) == Some("tool_use") {
                if let Some(id) = block.get("id").and_then(Value::as_str) {
                    ids.insert(id.to_string());
                }
            }
        }
    }

    ids
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn openai_pair_is_detected() {
        let msgs = vec![
            json!({"role": "user", "content": "go"}),
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_1", "type": "function",
                                 "function": {"name": "f", "arguments": "{}"}}]
            }),
            json!({"role": "tool", "tool_call_id": "call_1", "content": "result"}),
            json!({"role": "user", "content": "thanks"}),
        ];
        let pairs = tool_pair_indices(&msgs);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].assistant_index, 1);
        assert_eq!(pairs[0].response_index, 2);
    }

    #[test]
    fn anthropic_pair_is_detected() {
        let msgs = vec![
            json!({"role": "user", "content": "go"}),
            json!({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "thinking..."},
                    {"type": "tool_use", "id": "tu_1", "name": "f", "input": {}}
                ]
            }),
            json!({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "ok"}]
            }),
        ];
        let pairs = tool_pair_indices(&msgs);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].assistant_index, 1);
        assert_eq!(pairs[0].response_index, 2);
    }

    #[test]
    fn unmatched_tool_response_is_dropped() {
        let msgs = vec![
            json!({"role": "user", "content": "go"}),
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_known", "function": {"name": "f"}}]
            }),
            json!({"role": "tool", "tool_call_id": "call_orphan", "content": "?"}),
        ];
        let pairs = tool_pair_indices(&msgs);
        assert!(pairs.is_empty());
    }

    #[test]
    fn multiple_anthropic_tool_results_in_one_user_message() {
        let msgs = vec![
            json!({
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tu_a", "name": "f"},
                    {"type": "tool_use", "id": "tu_b", "name": "g"}
                ]
            }),
            json!({
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu_a", "content": "a"},
                    {"type": "tool_result", "tool_use_id": "tu_b", "content": "b"}
                ]
            }),
        ];
        let pairs = tool_pair_indices(&msgs);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].assistant_index, 0);
        assert_eq!(pairs[0].response_index, 1);
    }

    #[test]
    fn empty_messages_yields_no_pairs() {
        assert!(tool_pair_indices(&[]).is_empty());
    }
}
