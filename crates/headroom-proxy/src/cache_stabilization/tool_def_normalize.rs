//! PR-E1: tool array deterministic sort.
//!
//! Many client SDKs accumulate `tools[]` from a Python `set()` or a
//! `dict` whose iteration order is hash-randomized between processes.
//! The proxy sees a different tool order on every restart even though
//! the customer's source code never changed. Each shuffle busts every
//! prompt-cache hit on the cached prefix that contains the tools
//! definition, because cache hits require byte-identical bytes.
//!
//! This module provides a single mutation: sort `tools[]` alphabetically
//! by `tool["name"]`. Idempotent — re-sorting an already-sorted array
//! is a no-op (we still pay for the JSON walk but produce the same
//! bytes).
//!
//! # Cache-safety contract
//!
//! Mutating the request body is only safe under three preconditions
//! (the caller checks all three):
//!
//! 1. **PAYG auth mode.** OAuth and Subscription clients are
//!    passthrough-prefer; reordering bytes for a subscription client
//!    can look like cache-evasion to the upstream and trigger
//!    revocation. The caller gates with [`AuthMode::Payg`].
//! 2. **No `cache_control` marker on any tool.** When the customer has
//!    explicitly placed a marker on a tool object, reordering the
//!    array shifts what is "before" their marker and silently changes
//!    cache scope. Their intentional layout wins. See
//!    [`any_tool_has_cache_control`].
//! 3. **Idempotency.** Re-running on already-sorted input must yield
//!    byte-identical bytes; the walker uses a stable sort and rebuilds
//!    objects with `serde_json::Map` (which preserves insertion order
//!    via the workspace `preserve_order` feature, so the second sort
//!    sees the same input as the first).
//!
//! # Why no regex
//!
//! Per `feedback_realignment_build_constraints.md` (Realignment build
//! policy): no regex for parsing. Marker detection here is a structured
//! key lookup (`tool.get("cache_control")`), not a pattern match against
//! serialized JSON.

use md5::{Digest, Md5};
use serde_json::Value;

/// Sort `tools[]` deterministically by name, in place.
///
/// Sort key: `tool["name"]` as a string. For tools missing a name (rare;
/// the API requires it but malformed inputs do reach the proxy), the
/// fallback key is the MD5 hex digest of the canonical-JSON serialization
/// of the tool object. MD5 is sufficient — the value is opaque, used
/// only for in-process ordering, never persisted, never compared across
/// hosts.
///
/// Returns `true` if the sort changed the order, `false` if the array
/// was already sorted (idempotent signal). The caller emits a structured
/// event using this signal so dashboards can see how often the policy
/// fires.
///
/// # Stability
///
/// Uses a stable sort (`sort_by_key`): equal keys preserve original
/// order. Two unnamed tools that happen to MD5-collide will keep their
/// original relative order — collision is astronomically rare for any
/// realistic input but the contract still holds.
///
/// The slice signature (`&mut [Value]`) is preferred over `&mut
/// Vec<Value>` per clippy's `ptr_arg` guidance: callers can pass
/// either a `Vec` or any `&mut [Value]`, and we don't need
/// `Vec`-specific operations.
pub fn sort_tools_deterministically(tools: &mut [Value]) -> bool {
    // Capture the pre-sort key sequence so the return-value contract
    // (`true` iff anything moved) is exact. We compare keys, not full
    // values, because the sort is by key — equal-key swaps would not
    // affect cache bytes.
    let before: Vec<String> = tools.iter().map(sort_key).collect();
    tools.sort_by_key(sort_key);
    let after: Vec<String> = tools.iter().map(sort_key).collect();
    before != after
}

/// Build the deterministic sort key for a tool. Public only inside
/// this module; the public API is [`sort_tools_deterministically`].
///
/// Looks for the name at two known locations:
///
///   1. `tool["name"]` — Anthropic shape (`{"name": "...",
///      "input_schema": ...}`).
///   2. `tool["function"]["name"]` — OpenAI Chat Completions shape
///      (`{"type": "function", "function": {"name": "...",
///      "parameters": ...}}`).
///
/// Both providers carry the tool name in exactly one of these
/// positions; tools that match neither are rare malformed inputs that
/// fall back to the MD5-of-canonical-JSON fallback.
fn sort_key(tool: &Value) -> String {
    if let Some(name) = tool.get("name").and_then(Value::as_str) {
        return name.to_string();
    }
    if let Some(name) = tool
        .get("function")
        .and_then(|f| f.get("name"))
        .and_then(Value::as_str)
    {
        return name.to_string();
    }
    // Fallback for unnamed tools: MD5 of canonical-JSON
    // serialization. `serde_json::to_vec` is deterministic for a
    // given `Value` because the `preserve_order` workspace feature
    // pins object key order to insertion order.
    let serialized = serde_json::to_vec(tool).unwrap_or_default();
    let mut hasher = Md5::new();
    hasher.update(&serialized);
    let digest = hasher.finalize();
    // Hex-encode by hand to keep the dep surface tiny — `format!`
    // with `{:02x}` produces the same lowercase hex `hex::encode`
    // would.
    let mut out = String::with_capacity(32);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// Return `true` if any tool object carries a `cache_control` field at
/// its top level.
///
/// The Anthropic API places `cache_control` on the tool object itself
/// (e.g. `{"name": "x", "cache_control": {"type": "ephemeral"}, ...}`).
/// The customer uses this to mark a cache breakpoint that depends on
/// the tool's *position* in the array — reordering tools would shift
/// what's "before" the marker and silently change cache scope, voiding
/// their intent. So when any tool has the marker, we skip the sort.
///
/// This function only checks the top-level field. Markers nested inside
/// `input_schema` (none of the public APIs put one there) would not be
/// caught — but they would also not be position-dependent, so the
/// safety contract still holds.
pub fn any_tool_has_cache_control(tools: &[Value]) -> bool {
    tools.iter().any(|tool| tool.get("cache_control").is_some())
}

/// Recursively sort the keys of every JSON object node in `value`,
/// in place. PR-E2.
///
/// JSON Schema permits arbitrary key order, but cache hits require
/// byte-identical bytes. Different SDK serializers emit keys in
/// different orders (some sort, some preserve insertion, some are
/// hash-randomized). This walker rewrites every `Value::Object` with
/// keys in alphabetic order so the same logical schema serializes
/// to the same bytes regardless of upstream serializer behaviour.
///
/// # Array semantics
///
/// JSON Schema arrays are ordered. `oneOf`, `anyOf`, `allOf`,
/// `prefixItems`, and `enum` all carry semantic meaning in the
/// element order — so this walker recurses into arrays element by
/// element but does NOT reorder the arrays themselves. (Reordering
/// `oneOf` would be a no-op semantically but would still change
/// bytes; we preserve customer order to honour intent.)
///
/// # Idempotency
///
/// Sorting an already-sorted map yields byte-identical output: we
/// rebuild a fresh `serde_json::Map` populated in alphabetic order,
/// and `Map` (with the workspace `preserve_order` feature) emits
/// keys in insertion order. So the second pass produces the same
/// `Map` literal.
///
/// # Marker safety
///
/// Unlike PR-E1, this function has no `cache_control`-marker check.
/// The Anthropic API places `cache_control` on the tool *object* itself
/// (`{"name": ..., "cache_control": {...}, "input_schema": {...}}`),
/// not inside `input_schema`. Sorting keys inside `input_schema` does
/// not move the marker, so the customer's cache-breakpoint intent is
/// preserved either way. The caller is therefore free to pass a
/// schema for any tool, marker-bearing or not.
pub fn sort_schema_keys_recursive(value: &mut Value) {
    match value {
        Value::Object(map) => {
            // Recurse first so children are normalized before we
            // rebuild the parent. Order of recursion doesn't affect
            // correctness (each child is independent) but doing it
            // first means the parent's sorted Map is built once over
            // already-sorted children — no repeated work.
            for (_k, v) in map.iter_mut() {
                sort_schema_keys_recursive(v);
            }
            // Collect existing entries, sort by key, rebuild the
            // map. Cloning is unavoidable: `serde_json::Map` does
            // not expose an in-place key reorder. The clone is a
            // shallow Value clone — children were already mutated
            // in place above so we don't lose the recursive sort.
            let mut entries: Vec<(String, Value)> =
                map.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            map.clear();
            for (k, v) in entries {
                map.insert(k, v);
            }
        }
        Value::Array(items) => {
            // Preserve array order — JSON Schema arrays are ordered.
            // Recurse into each element so nested objects inside the
            // array still get key-sorted.
            for item in items.iter_mut() {
                sort_schema_keys_recursive(item);
            }
        }
        // Strings, numbers, booleans, null have no keys to sort.
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use serde_json::json;

    // ─── E1: sort_tools_deterministically ─────────────────────────

    #[test]
    fn sort_alphabetic_by_name() {
        let mut tools = vec![
            json!({"name": "B"}),
            json!({"name": "A"}),
            json!({"name": "C"}),
        ];
        let changed = sort_tools_deterministically(&mut tools);
        assert!(changed, "out-of-order input should report a reorder");
        let names: Vec<&str> = tools
            .iter()
            .map(|t| t.get("name").and_then(Value::as_str).unwrap())
            .collect();
        assert_eq!(names, vec!["A", "B", "C"]);
    }

    #[test]
    fn idempotent_resort_no_change() {
        let mut tools = vec![
            json!({"name": "A"}),
            json!({"name": "B"}),
            json!({"name": "C"}),
        ];
        let changed = sort_tools_deterministically(&mut tools);
        assert!(!changed, "already-sorted input must report no reorder");
        let names: Vec<&str> = tools
            .iter()
            .map(|t| t.get("name").and_then(Value::as_str).unwrap())
            .collect();
        assert_eq!(names, vec!["A", "B", "C"]);
    }

    #[test]
    fn byte_stable_across_runs() {
        // Two independently-shuffled inputs produce byte-identical
        // serialized output after sort. This is the core invariant:
        // upstream sees the same bytes regardless of upstream client
        // tool-collection order.
        let mut input_a = vec![
            json!({"name": "search", "description": "x"}),
            json!({"name": "fetch", "description": "y"}),
            json!({"name": "edit", "description": "z"}),
        ];
        let mut input_b = vec![
            json!({"name": "edit", "description": "z"}),
            json!({"name": "search", "description": "x"}),
            json!({"name": "fetch", "description": "y"}),
        ];
        sort_tools_deterministically(&mut input_a);
        sort_tools_deterministically(&mut input_b);
        let a_bytes = serde_json::to_vec(&input_a).unwrap();
        let b_bytes = serde_json::to_vec(&input_b).unwrap();
        assert_eq!(
            a_bytes, b_bytes,
            "different inputs with same tool set must serialize identically after sort"
        );
    }

    #[test]
    fn sort_alphabetic_by_openai_function_name() {
        // OpenAI Chat shape: name lives at `tool.function.name`.
        let mut tools = vec![
            json!({"type": "function", "function": {"name": "Z_tool"}}),
            json!({"type": "function", "function": {"name": "A_tool"}}),
            json!({"type": "function", "function": {"name": "M_tool"}}),
        ];
        let changed = sort_tools_deterministically(&mut tools);
        assert!(changed);
        let names: Vec<&str> = tools
            .iter()
            .map(|t| {
                t.get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(Value::as_str)
                    .unwrap()
            })
            .collect();
        assert_eq!(names, vec!["A_tool", "M_tool", "Z_tool"]);
    }

    #[test]
    fn unnamed_tool_uses_md5_fallback() {
        // Two unnamed tools — the MD5 of canonical JSON breaks ties
        // deterministically. The serialized output must be stable
        // across runs.
        let mut tools = vec![
            json!({"description": "second"}),
            json!({"description": "first"}),
        ];
        let _ = sort_tools_deterministically(&mut tools);
        let bytes_run1 = serde_json::to_vec(&tools).unwrap();

        let mut tools2 = vec![
            json!({"description": "first"}),
            json!({"description": "second"}),
        ];
        let _ = sort_tools_deterministically(&mut tools2);
        let bytes_run2 = serde_json::to_vec(&tools2).unwrap();

        assert_eq!(
            bytes_run1, bytes_run2,
            "unnamed-tool MD5 fallback must produce stable byte output"
        );
    }

    #[test]
    fn cache_control_detection_finds_marker() {
        let with_marker = vec![
            json!({"name": "A"}),
            json!({"name": "B", "cache_control": {"type": "ephemeral"}}),
            json!({"name": "C"}),
        ];
        assert!(any_tool_has_cache_control(&with_marker));

        let without_marker = vec![
            json!({"name": "A"}),
            json!({"name": "B"}),
            json!({"name": "C"}),
        ];
        assert!(!any_tool_has_cache_control(&without_marker));
    }

    #[test]
    fn cache_control_detection_returns_false_on_empty_tools() {
        let empty: Vec<Value> = Vec::new();
        assert!(!any_tool_has_cache_control(&empty));
    }

    proptest! {
        /// Sort is a permutation: no tools added, no tools removed,
        /// for any reasonable mix of named / unnamed tools.
        #[test]
        fn sort_is_permutation(
            names in prop::collection::vec(
                prop::option::of("[a-zA-Z][a-zA-Z0-9_]{0,15}"),
                0..16,
            )
        ) {
            let mut tools: Vec<Value> = names
                .iter()
                .map(|maybe_name| match maybe_name {
                    Some(n) => json!({"name": n, "description": "x"}),
                    None => json!({"description": "unnamed"}),
                })
                .collect();
            let len_before = tools.len();
            sort_tools_deterministically(&mut tools);
            prop_assert_eq!(tools.len(), len_before);
        }
    }

    // ─── E2: sort_schema_keys_recursive ───────────────────────────

    #[test]
    fn sorts_top_level_object_keys() {
        let mut value = json!({
            "type": "object",
            "properties": {},
            "required": [],
        });
        sort_schema_keys_recursive(&mut value);
        let serialized = serde_json::to_string(&value).unwrap();
        let p_pos = serialized.find("\"properties\"").unwrap();
        let r_pos = serialized.find("\"required\"").unwrap();
        let t_pos = serialized.find("\"type\"").unwrap();
        assert!(
            p_pos < r_pos && r_pos < t_pos,
            "expected alphabetic order properties < required < type, got: {serialized}"
        );
    }

    #[test]
    fn sorts_nested_property_keys() {
        let mut value = json!({
            "type": "object",
            "properties": {
                "z_field": {"type": "string"},
                "a_field": {"type": "integer"},
                "m_field": {"type": "boolean"},
            },
        });
        sort_schema_keys_recursive(&mut value);
        let serialized = serde_json::to_string(&value).unwrap();
        let a_pos = serialized.find("\"a_field\"").unwrap();
        let m_pos = serialized.find("\"m_field\"").unwrap();
        let z_pos = serialized.find("\"z_field\"").unwrap();
        assert!(
            a_pos < m_pos && m_pos < z_pos,
            "nested property keys must be sorted alphabetically; got: {serialized}"
        );
    }

    #[test]
    fn preserves_array_order_in_oneof() {
        let mut value = json!({
            "oneOf": [
                {"const": "third"},
                {"const": "first"},
                {"const": "second"},
            ],
        });
        sort_schema_keys_recursive(&mut value);
        let arr = value.get("oneOf").and_then(Value::as_array).unwrap();
        let consts: Vec<&str> = arr
            .iter()
            .map(|v| v.get("const").and_then(Value::as_str).unwrap())
            .collect();
        assert_eq!(
            consts,
            vec!["third", "first", "second"],
            "JSON Schema arrays (oneOf) must preserve element order"
        );
    }

    #[test]
    fn idempotent_resort_schema() {
        let mut value = json!({
            "type": "object",
            "properties": {
                "z": {"type": "integer", "default": 1, "description": "z field"},
                "a": {"type": "string", "minLength": 1, "default": "x"},
            },
            "additionalProperties": false,
            "required": ["a", "z"],
        });
        sort_schema_keys_recursive(&mut value);
        let bytes_first = serde_json::to_vec(&value).unwrap();
        sort_schema_keys_recursive(&mut value);
        let bytes_second = serde_json::to_vec(&value).unwrap();
        assert_eq!(
            bytes_first, bytes_second,
            "second sort over already-sorted schema must be a byte-equal no-op"
        );
    }

    #[test]
    fn does_not_alter_arrays_within_arrays() {
        // Nested arrays preserve order at every level.
        let mut value = json!({
            "examples": [
                [3, 1, 2],
                ["c", "a", "b"],
            ],
        });
        sort_schema_keys_recursive(&mut value);
        let outer = value.get("examples").and_then(Value::as_array).unwrap();
        let inner_nums: Vec<i64> = outer[0]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        assert_eq!(inner_nums, vec![3, 1, 2]);
        let inner_strs: Vec<&str> = outer[1]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(inner_strs, vec!["c", "a", "b"]);
    }

    #[test]
    fn handles_deeply_nested_schemas() {
        let mut value = json!({
            "type": "object",
            "properties": {
                "outer": {
                    "type": "object",
                    "properties": {
                        "inner": {
                            "type": "object",
                            "properties": {
                                "z_deep": {"type": "string"},
                                "a_deep": {"type": "integer"},
                            },
                            "required": ["z_deep", "a_deep"],
                        },
                    },
                },
            },
        });
        sort_schema_keys_recursive(&mut value);
        let serialized = serde_json::to_string(&value).unwrap();
        let a_pos = serialized.find("\"a_deep\"").unwrap();
        let z_pos = serialized.find("\"z_deep\"").unwrap();
        assert!(
            a_pos < z_pos,
            "deeply-nested keys must be sorted alphabetically; got: {serialized}"
        );
    }
}
