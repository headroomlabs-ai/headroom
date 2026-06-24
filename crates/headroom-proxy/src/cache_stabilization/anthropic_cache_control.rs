//! PR-E3: Anthropic `cache_control` auto-placement.
//!
//! Anthropic's prompt cache is opt-in per content block: nothing is
//! cached unless the request body explicitly carries
//! `cache_control: {"type": "ephemeral"}` on at least one block.
//! Sophisticated clients (e.g. Claude Code) place those markers
//! themselves; less-sophisticated callers (hand-rolled SDK code,
//! smaller agents like Aider/Continue, plain `curl`) typically don't
//! even know `cache_control` exists. For *those* clients we can
//! get them cache hits at zero learning-cost by inserting one
//! marker on the highest-reuse content block in the request.
//!
//! # Safety contract
//!
//! This module is the only Phase E surface that **mutates request
//! bytes** — that's the whole point. To stay inside the Phase A
//! "passthrough is sacred" invariant we layer three independent
//! gates:
//!
//! 1. **Auth-mode gate (caller's responsibility).** Mutating bytes
//!    on an OAuth-bearing or subscription-bound request risks
//!    looking like cache-evasion to the upstream and can trigger
//!    subscription revocation. The caller MUST classify the
//!    inbound auth mode via [`headroom_core::auth_mode::classify`]
//!    and only call this function when the mode is
//!    [`headroom_core::auth_mode::AuthMode::Payg`]. The caller
//!    emits a structured `event = "e3_skipped", reason =
//!    "auth_mode"` log line on the non-PAYG path.
//! 2. **Customer-placement-wins gate.** If *any* `cache_control`
//!    marker is found anywhere in the body the caller hands us we
//!    return [`AutoPlaceOutcome::Skipped { reason:
//!    SkipReason::MarkerPresent }`] and never mutate. This walks
//!    `system` (string OR array of blocks), `messages[].content`
//!    (string OR array of blocks), and `tools[]` (top-level on
//!    each tool). The customer's cache layout is theirs to own.
//! 3. **Idempotency.** Re-running on a body that already carries
//!    the markers we'd add is a no-op via gate (2) — the
//!    previously-placed marker becomes the customer-placement-wins
//!    signal on the next pass.
//!
//! Every skip / apply emits a structured `tracing::info!` with
//! `event = "e3_skipped"` or `event = "e3_applied"` so production
//! telemetry can confirm the gates fire as designed.
//!
//! # Placement strategy (first ship: ONE marker only)
//!
//! Anthropic's `cache_control` semantics: a marker on a block caches
//! *that block + everything before it* in the canonical request
//! order (`system → tools → messages`). Each cached prefix lasts
//! 5 minutes. A request may carry up to **4** markers (Anthropic's
//! hard limit).
//!
//! Future placement priority (not yet enabled — requires production
//! telemetry to validate before we mutate further bytes per request):
//!
//!   1. **Last tool definition (top-level).** Caches the entire
//!      `tools` array — highest reuse across turns.
//!   2. **Last block of the system prompt.** Caches `system + tools`.
//!      Only fires when `system` is already an array of blocks; we
//!      do **not** convert a plain-string `system` to an array on
//!      first ship — that's a bigger surgery and we'd want telemetry
//!      first.
//!   3. **Last user message's last block** when the conversation
//!      already has ≥ 2 turns of history (caches everything except
//!      the live tool_result tail).
//!   4. Fourth slot reserved — left unplaced for safety.
//!
//! **First ship default:** place exactly one marker on the last tool
//! definition's top-level. Highest-value, lowest-risk, easiest to
//! revert. Slots 2/3/4 require production telemetry to enable.
//!
//! # Marker shape
//!
//! ```json
//! {"cache_control": {"type": "ephemeral"}}
//! ```
//!
//! No TTL field — the 5-minute default is the right one for the
//! first-ship "last tool" placement (tools rarely turn over within
//! 5 minutes; longer TTLs require careful auth-mode and per-tenant
//! sizing we don't yet have).

use serde_json::{json, Value};

/// Result of an auto-placement attempt.
///
/// Returned by [`auto_place_anthropic_cache_control`]. The caller
/// uses the variant + count to emit structured telemetry. Variants
/// stay narrow — one happy path with a count, one skip with a
/// machine-readable reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AutoPlaceOutcome {
    /// At least one `cache_control` marker was inserted. `placed_count`
    /// is the number of markers added in this call. With the
    /// first-ship policy this is always `1` when the body has tools
    /// and `0` when it doesn't (the latter still returns `Applied`
    /// with `placed_count = 0` so the caller can log the
    /// "we-tried-but-no-target" branch).
    Applied {
        /// Number of markers added on this call.
        placed_count: usize,
        /// JSON-pointer-style locations the markers were placed at.
        /// Stable identifiers used by dashboards to spot which slots
        /// fire most. Example: `"tools[3]"`.
        locations: Vec<String>,
    },
    /// We did not mutate. `reason` tells dashboards which gate fired.
    Skipped {
        /// Why we skipped — see [`SkipReason`] for the closed set.
        reason: SkipReason,
    },
}

/// Why E3 declined to place a marker.
///
/// Closed enum so the structured `event = "e3_skipped"` log carries
/// a stable `reason` field. Dashboards filter on these strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipReason {
    /// Caller's auth-mode gate fired — request was not classified as
    /// PAYG, so the safety contract bans mutation. The caller is the
    /// one who detects this; surfaced here so the function's outcome
    /// shape stays uniform across both gates.
    AuthMode,
    /// At least one `cache_control` marker was already present in the
    /// body (`system`, any message block, any tool top-level).
    /// Customer placement wins.
    MarkerPresent,
}

impl SkipReason {
    /// Stable string for structured-log `reason` field. Dashboards
    /// filter on these — do not change without a deprecation note.
    pub fn as_str(self) -> &'static str {
        match self {
            SkipReason::AuthMode => "auth_mode",
            SkipReason::MarkerPresent => "marker_present",
        }
    }
}

/// Walk `body` and return `true` if any `cache_control` field appears
/// anywhere. Public so the caller can early-skip without going through
/// the full mutating path (e.g. when it wants to log a different
/// `reason` field on its own gate).
///
/// The walker inspects the three places Anthropic's request schema
/// allows `cache_control`:
///
/// - `body.system` — when it's an array of content blocks (string
///   form cannot carry a marker).
/// - `body.messages[].content` — when it's an array of blocks.
/// - `body.tools[]` — top-level on each tool definition.
///
/// We do **not** descend into arbitrary nested objects. The only
/// shape Anthropic recognises `cache_control` on is the documented
/// surface above; descending into tool `input_schema` etc. would
/// false-positive on customer JSON Schemas that happen to mention
/// the field name as a property key.
pub fn any_anthropic_cache_control(body: &Value) -> bool {
    // ── system: string OR array of blocks ─────────────────────────
    // Only the array form can carry markers — string form is
    // explicitly skipped (cannot carry a `cache_control` field).
    if let Some(Value::Array(blocks)) = body.get("system") {
        for block in blocks {
            if block_has_cache_control(block) {
                return true;
            }
        }
    }

    // ── messages[].content: string OR array of blocks ─────────────
    if let Some(Value::Array(messages)) = body.get("messages") {
        for msg in messages {
            if let Some(Value::Array(blocks)) = msg.get("content") {
                for block in blocks {
                    if block_has_cache_control(block) {
                        return true;
                    }
                }
            }
            // string form: cannot carry a marker — skip.
        }
    }

    // ── tools[]: top-level field on each tool ─────────────────────
    if let Some(Value::Array(tools)) = body.get("tools") {
        for tool in tools {
            if block_has_cache_control(tool) {
                return true;
            }
        }
    }

    false
}

/// Auto-place up to one Anthropic `cache_control` marker on the
/// last tool definition.
///
/// **Caller contract:** caller MUST gate on auth_mode == PAYG before
/// invoking. This function does NOT classify auth mode itself; the
/// split lets unit tests exercise marker detection without
/// constructing a real `HeaderMap`.
///
/// **Behaviour:**
///
/// - If any `cache_control` marker is already present anywhere in
///   the body (`system` blocks, message blocks, or top-level on any
///   tool), returns [`AutoPlaceOutcome::Skipped { reason:
///   SkipReason::MarkerPresent }`] and does not mutate.
/// - If `body.tools[]` is missing, empty, or not an array, returns
///   [`AutoPlaceOutcome::Applied { placed_count: 0, locations: vec![] }`]
///   and does not mutate. (Distinguishes "we ran and found nothing
///   to do" from "we declined to run".)
/// - Otherwise, inserts a `cache_control: {"type": "ephemeral"}` field
///   on the last tool definition (top-level) and returns
///   [`AutoPlaceOutcome::Applied { placed_count: 1, locations: ["tools[N-1]"] }`].
///
/// **Idempotency:** running this twice on the same body is a no-op.
/// The first call inserts the marker; the second call sees it via
/// the customer-placement-wins gate and returns `Skipped`.
///
/// **First-ship scope:** only the "last tool" slot is enabled. The
/// system-prompt and message-history slots are documented in the
/// module-level docs but require production telemetry to enable.
pub fn auto_place_anthropic_cache_control(body: &mut Value) -> AutoPlaceOutcome {
    // Gate 2: any pre-existing marker → customer wins, full skip.
    if any_anthropic_cache_control(body) {
        return AutoPlaceOutcome::Skipped {
            reason: SkipReason::MarkerPresent,
        };
    }

    // First-ship default: place ONE marker on the last tool.
    let tools = match body.get_mut("tools") {
        Some(Value::Array(t)) if !t.is_empty() => t,
        _ => {
            // No tools array, or empty, or not an array. We "ran"
            // but had no target — return Applied{0} so the caller
            // can log "no_targets_present" for telemetry.
            return AutoPlaceOutcome::Applied {
                placed_count: 0,
                locations: Vec::new(),
            };
        }
    };
    let last_idx = tools.len() - 1;
    let last_tool = &mut tools[last_idx];
    if !insert_cache_control_on_object(last_tool) {
        // The last "tool" is not an object (Anthropic's schema
        // requires it to be — but we never panic on a malformed
        // body). Skip the slot rather than crashing.
        return AutoPlaceOutcome::Applied {
            placed_count: 0,
            locations: Vec::new(),
        };
    }
    AutoPlaceOutcome::Applied {
        placed_count: 1,
        locations: vec![format!("tools[{last_idx}]")],
    }
}

/// Does this content block carry a `cache_control` field at its top
/// level? Used by both the read-only walker
/// ([`any_anthropic_cache_control`]) and (transitively) the
/// idempotency gate.
fn block_has_cache_control(block: &Value) -> bool {
    block.get("cache_control").is_some()
}

/// Insert `"cache_control": {"type": "ephemeral"}` on `value` if it
/// is a JSON object. Returns `true` on insert, `false` if `value`
/// was not an object (so the caller can decline to claim a slot).
fn insert_cache_control_on_object(value: &mut Value) -> bool {
    match value.as_object_mut() {
        Some(map) => {
            map.insert("cache_control".to_string(), json!({"type": "ephemeral"}));
            true
        }
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Fresh body with one tool, one short user message, plain-string
    /// system. Used as the seed for "happy-path placement" tests.
    fn body_one_tool_no_markers() -> Value {
        json!({
            "model": "claude-3-5-sonnet-20241022",
            "system": "You are helpful.",
            "tools": [
                {
                    "name": "search",
                    "description": "search the web",
                    "input_schema": {"type": "object", "properties": {}}
                }
            ],
            "messages": [
                {"role": "user", "content": "hi"}
            ],
        })
    }

    #[test]
    fn places_cache_control_on_last_tool_when_payg_and_no_markers() {
        let mut body = body_one_tool_no_markers();
        let outcome = auto_place_anthropic_cache_control(&mut body);
        match outcome {
            AutoPlaceOutcome::Applied {
                placed_count,
                locations,
            } => {
                assert_eq!(placed_count, 1);
                assert_eq!(locations, vec!["tools[0]"]);
            }
            other => panic!("expected Applied{{1}}, got {other:?}"),
        }
        // Marker visible at the right path.
        let cc = body
            .pointer("/tools/0/cache_control")
            .expect("marker inserted on tools[0]");
        assert_eq!(cc, &json!({"type": "ephemeral"}));
    }

    #[test]
    fn places_on_last_tool_when_multiple_tools() {
        // With multiple tools, the marker must go on the last one.
        let mut body = json!({
            "tools": [
                {"name": "a", "description": "a"},
                {"name": "b", "description": "b"},
                {"name": "c", "description": "c"}
            ],
            "messages": [{"role": "user", "content": "hi"}],
        });
        let outcome = auto_place_anthropic_cache_control(&mut body);
        match outcome {
            AutoPlaceOutcome::Applied {
                placed_count,
                locations,
            } => {
                assert_eq!(placed_count, 1);
                assert_eq!(locations, vec!["tools[2]"]);
            }
            other => panic!("expected Applied{{1}}, got {other:?}"),
        }
        assert!(body.pointer("/tools/0/cache_control").is_none());
        assert!(body.pointer("/tools/1/cache_control").is_none());
        assert!(body.pointer("/tools/2/cache_control").is_some());
    }

    #[test]
    fn skips_when_any_tool_already_has_marker() {
        // Customer placed a marker on the FIRST tool (not the slot
        // we'd pick). Customer-placement-wins still skips us.
        let mut body = json!({
            "tools": [
                {
                    "name": "search",
                    "description": "search",
                    "cache_control": {"type": "ephemeral"}
                },
                {"name": "fetch", "description": "fetch"}
            ],
            "messages": [{"role": "user", "content": "hi"}],
        });
        let before = body.clone();
        let outcome = auto_place_anthropic_cache_control(&mut body);
        assert_eq!(
            outcome,
            AutoPlaceOutcome::Skipped {
                reason: SkipReason::MarkerPresent
            }
        );
        assert_eq!(body, before, "skip path must not mutate");
    }

    #[test]
    fn skips_when_system_block_already_has_marker() {
        // Customer used the array-form `system` and placed their own
        // marker. Customer-placement-wins.
        let mut body = json!({
            "system": [
                {"type": "text", "text": "you are helpful"},
                {
                    "type": "text",
                    "text": "cite sources",
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            "tools": [{"name": "search", "description": "search"}],
            "messages": [{"role": "user", "content": "hi"}],
        });
        let before = body.clone();
        let outcome = auto_place_anthropic_cache_control(&mut body);
        assert_eq!(
            outcome,
            AutoPlaceOutcome::Skipped {
                reason: SkipReason::MarkerPresent
            }
        );
        assert_eq!(body, before, "skip path must not mutate");
    }

    #[test]
    fn skips_when_message_block_already_has_marker() {
        // Customer placed a marker mid-conversation. Skip everything.
        let mut body = json!({
            "tools": [{"name": "search", "description": "search"}],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "remember this",
                            "cache_control": {"type": "ephemeral"}
                        }
                    ]
                },
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "now what?"}
            ],
        });
        let before = body.clone();
        let outcome = auto_place_anthropic_cache_control(&mut body);
        assert_eq!(
            outcome,
            AutoPlaceOutcome::Skipped {
                reason: SkipReason::MarkerPresent
            }
        );
        assert_eq!(body, before, "skip path must not mutate");
    }

    #[test]
    fn idempotent_when_we_already_placed_marker_last_run() {
        // Run once: marker placed. Run again: customer-placement-wins
        // gate fires (the marker we placed last time IS now a
        // customer-side marker as far as gate 2 is concerned).
        let mut body = body_one_tool_no_markers();
        let first = auto_place_anthropic_cache_control(&mut body);
        assert!(matches!(
            first,
            AutoPlaceOutcome::Applied {
                placed_count: 1,
                ..
            }
        ));
        let after_first = body.clone();
        let second = auto_place_anthropic_cache_control(&mut body);
        assert_eq!(
            second,
            AutoPlaceOutcome::Skipped {
                reason: SkipReason::MarkerPresent
            }
        );
        assert_eq!(body, after_first, "second run must not mutate");
    }

    #[test]
    fn does_nothing_when_no_tools_present() {
        // Body with no tools field. Returns Applied{0}, no mutation.
        let mut body = json!({
            "model": "claude-3-5-sonnet-20241022",
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "hi"}],
        });
        let before = body.clone();
        let outcome = auto_place_anthropic_cache_control(&mut body);
        assert_eq!(
            outcome,
            AutoPlaceOutcome::Applied {
                placed_count: 0,
                locations: Vec::new(),
            }
        );
        assert_eq!(body, before, "Applied{{0}} path must not mutate");
    }

    #[test]
    fn does_nothing_when_tools_array_is_empty() {
        let mut body = json!({
            "tools": [],
            "messages": [{"role": "user", "content": "hi"}],
        });
        let before = body.clone();
        let outcome = auto_place_anthropic_cache_control(&mut body);
        assert_eq!(
            outcome,
            AutoPlaceOutcome::Applied {
                placed_count: 0,
                locations: Vec::new(),
            }
        );
        assert_eq!(body, before, "empty-tools path must not mutate");
    }

    #[test]
    fn system_string_form_does_not_get_converted_to_array() {
        // Conservative first-ship policy: a plain-string `system`
        // stays a plain string. We only place on tools.
        let mut body = json!({
            "system": "You are helpful. Cite sources.",
            "tools": [{"name": "search", "description": "search"}],
            "messages": [{"role": "user", "content": "hi"}],
        });
        let outcome = auto_place_anthropic_cache_control(&mut body);
        assert!(matches!(
            outcome,
            AutoPlaceOutcome::Applied {
                placed_count: 1,
                ..
            }
        ));
        // System is still a plain string.
        assert_eq!(
            body.get("system"),
            Some(&json!("You are helpful. Cite sources.")),
            "string-form `system` must stay untouched on first ship",
        );
        // Marker landed on tools[0] instead.
        assert_eq!(
            body.pointer("/tools/0/cache_control"),
            Some(&json!({"type": "ephemeral"})),
        );
    }

    #[test]
    fn applies_only_one_marker_in_first_ship_default() {
        // Multi-turn conversation + multiple tools + array-form system.
        // First-ship default places exactly ONE marker (last tool).
        let mut body = json!({
            "system": [
                {"type": "text", "text": "rule 1"},
                {"type": "text", "text": "rule 2"}
            ],
            "tools": [
                {"name": "a", "description": "a"},
                {"name": "b", "description": "b"}
            ],
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "second"},
                {"role": "user", "content": "third"},
                {"role": "assistant", "content": "fourth"},
                {"role": "user", "content": "fifth"}
            ],
        });
        let outcome = auto_place_anthropic_cache_control(&mut body);
        assert_eq!(
            outcome,
            AutoPlaceOutcome::Applied {
                placed_count: 1,
                locations: vec!["tools[1]".to_string()],
            }
        );
        // Walk every other slot to confirm it stayed clean.
        let system_blocks = body.pointer("/system").unwrap().as_array().unwrap();
        for block in system_blocks {
            assert!(
                !block_has_cache_control(block),
                "first-ship default must not place on system blocks: {block:?}"
            );
        }
        for (i, msg) in body
            .pointer("/messages")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            // Messages with string-form content can't carry markers
            // anyway, but assert no marker placed regardless.
            if let Some(Value::Array(blocks)) = msg.get("content") {
                for block in blocks {
                    assert!(
                        !block_has_cache_control(block),
                        "message[{i}] block carries unexpected marker: {block:?}"
                    );
                }
            }
        }
        // Only tools[1] carries a marker.
        assert!(body.pointer("/tools/0/cache_control").is_none());
        assert_eq!(
            body.pointer("/tools/1/cache_control"),
            Some(&json!({"type": "ephemeral"})),
        );
    }

    #[test]
    fn applied_path_preserves_other_tool_fields() {
        // Property: post-placement body matches input body modulo
        // the `tools[last].cache_control` key. Sort-stable, no
        // collateral mutation elsewhere.
        let original = body_one_tool_no_markers();
        let mut body = original.clone();
        let _ = auto_place_anthropic_cache_control(&mut body);

        // Strip the marker we placed and compare to the original.
        let tools = body
            .get_mut("tools")
            .and_then(Value::as_array_mut)
            .expect("tools array");
        for tool in tools {
            if let Some(map) = tool.as_object_mut() {
                map.remove("cache_control");
            }
        }
        assert_eq!(
            body, original,
            "Applied path must mutate ONLY the cache_control field on the chosen slot",
        );
    }

    #[test]
    fn skip_reason_strings_are_stable() {
        // Dashboards filter on these strings. Pin them.
        assert_eq!(SkipReason::AuthMode.as_str(), "auth_mode");
        assert_eq!(SkipReason::MarkerPresent.as_str(), "marker_present");
    }

    #[test]
    fn any_marker_walker_scans_system_array_form_only() {
        // String-form `system` cannot carry a marker — walker should
        // not flag it even when the string contains the substring
        // "cache_control".
        let body = json!({
            "system": "Note: cache_control is an Anthropic concept.",
            "messages": [],
        });
        assert!(!any_anthropic_cache_control(&body));

        // Array-form `system` with a marker — walker DOES flag it.
        let body_with_marker = json!({
            "system": [
                {"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [],
        });
        assert!(any_anthropic_cache_control(&body_with_marker));
    }

    #[test]
    fn any_marker_walker_does_not_descend_into_input_schema() {
        // Customer's tool input_schema happens to declare a property
        // literally named `cache_control`. That's NOT a real Anthropic
        // marker — it's a JSON Schema property name. Walker must NOT
        // false-positive.
        let body = json!({
            "tools": [{
                "name": "configure",
                "description": "configure something",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "cache_control": {"type": "string"}
                    }
                }
            }],
            "messages": [{"role": "user", "content": "hi"}],
        });
        assert!(
            !any_anthropic_cache_control(&body),
            "walker must scope to documented Anthropic surfaces; \
             schema property keys do not count as markers",
        );
    }

    #[test]
    fn malformed_tool_entry_does_not_panic() {
        // Defensive: an Anthropic request with a non-object tool would
        // get a 400 from upstream, but we never panic on a malformed
        // body. Skip the slot instead.
        let mut body = json!({
            "tools": ["not-an-object"],
            "messages": [{"role": "user", "content": "hi"}],
        });
        let before = body.clone();
        let outcome = auto_place_anthropic_cache_control(&mut body);
        assert_eq!(
            outcome,
            AutoPlaceOutcome::Applied {
                placed_count: 0,
                locations: Vec::new(),
            }
        );
        assert_eq!(body, before, "malformed-tool path must not mutate");
    }
}
