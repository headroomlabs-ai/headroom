//! Customer `cache_control` marker walker for Anthropic `/v1/messages`
//! request bodies.
//!
//! # Why this exists
//!
//! Anthropic prompt caching pins a prefix of the request: every block
//! up to and including the last `cache_control` marker is part of
//! the cache key. The provider returns `cache_read_input_tokens` for
//! that prefix on subsequent requests; that's the customer's primary
//! lever for cost reduction.
//!
//! Headroom's compressor must **never** modify any byte that's part
//! of that prefix — doing so changes the cache key, drops the hit
//! rate to 0, and silently torches the customer's bill. Phase A
//! lockdown PR-A1 made `/v1/messages` a passthrough so we couldn't
//! cause this damage; PR-A4 (this module) computes the floor below
//! which Phase B's live-zone dispatcher must not touch.
//!
//! # Contract
//!
//! For an Anthropic request body parsed into `serde_json::Value`,
//! [`compute_frozen_count`] returns the smallest `N` such that
//! `messages[i]` is in the cache hot zone for every `i < N`.
//! Specifically:
//!
//! - For each marker found in `messages[i].content[*].cache_control`,
//!   the function bumps `frozen_count` to at least `i + 1`. The "+1"
//!   makes the floor exclusive: `messages[i]` itself is part of the
//!   cached prefix, so it's frozen.
//! - For markers in `system` (string OR block list) or `tools[*]`,
//!   the function does NOT bump `frozen_count`. Those fields are
//!   unconditionally part of the cache hot zone (see invariant I2 in
//!   `REALIGNMENT/02-architecture.md` §2.2); they're never touched by
//!   the compressor regardless of marker placement, so they don't
//!   affect the message-index floor.
//! - Returns `0` when there are no markers anywhere in `messages[*]`.
//!
//! # Why no regex
//!
//! Per the realignment build constraints
//! (`feedback_realignment_build_constraints.md` rule 3), pattern
//! detection uses parsers, not regex. We walk the parsed JSON tree
//! via `serde_json` accessors only — that's both safer (no
//! pattern-string typo risk) and faster (no compilation cost on
//! the hot path).
//!
//! # TTL ordering
//!
//! Per the Anthropic prompt-caching guide §2.19, when both `5m` and
//! `1h` markers appear, `1h` markers must precede `5m`. We compute
//! `frozen_count` correctly regardless of ordering, but emit a
//! `tracing::warn!` for the operator's benefit when the customer's
//! request violates the rule. We do NOT reject the request: it's the
//! customer's choice to make and Anthropic itself accepts both
//! orderings (just with potentially-suboptimal cache eviction).
//!
//! # Source priority
//!
//! Configurable on/off via `Config::cache_control_auto_frozen`
//! (CLI flag `--cache-control-auto-frozen` / env var
//! `HEADROOM_PROXY_CACHE_CONTROL_AUTO_FROZEN`). When `disabled`, the
//! caller bypasses [`compute_frozen_count`] entirely and treats every
//! message as live-zone. The function itself is config-agnostic; the
//! gate lives in the caller (the live-zone dispatcher in Phase B).

use serde_json::Value;

/// TTL marker value for the Anthropic 1-hour cache extension.
///
/// Per guide §2.19, the literal string `"1h"` selects the hour-long
/// ephemeral cache lane. We keep the constant here (rather than
/// inlining `"1h"` literal at use sites) so any future rename or
/// case-change is a single-edit affair. Avoiding magic strings is
/// rule 2 of the realignment build constraints.
const CACHE_TTL_1H: &str = "1h";

/// TTL marker value for the Anthropic 5-minute cache lane (the
/// default). Matches `cache_control.ttl == "5m"` literal.
const CACHE_TTL_5M: &str = "5m";

/// Walk the parsed Anthropic request body and return the smallest
/// `frozen_message_count` that respects every customer-set
/// `cache_control` marker.
///
/// # Arguments
///
/// - `parsed`: the request body as a `serde_json::Value`. The walker
///   reads `parsed.get("messages")`, `parsed.get("system")`, and
///   `parsed.get("tools")`. Other top-level fields are ignored.
///
/// # Returns
///
/// The lowest message index `N` such that `messages[i]` is frozen
/// for every `i < N`. Specifically:
/// - `0` when no markers are found in `messages[*]` (the live-zone
///   dispatcher is then free to compress every message).
/// - `i + 1` for the highest `i` whose `messages[i].content[*]`
///   contains a `cache_control` marker.
///
/// Markers in `system` and `tools[*]` do NOT raise the message-index
/// floor (those fields are unconditionally cache-hot; see module docs).
///
/// # Logging
///
/// Emits `tracing::debug!` for every marker found. Emits
/// `tracing::warn!` for any TTL ordering violation per guide §2.19
/// (a `5m` marker preceding a `1h` marker in the same field-list).
/// Returns the correct value regardless of ordering.
pub fn compute_frozen_count(parsed: &Value) -> usize {
    // Highest message-index marker seen so far. Tracked as `Option`
    // so a missing-vs-zero state is unambiguous: `None` means "no
    // marker observed", `Some(i)` means "saw a marker on index i".
    let mut highest_message_index: Option<usize> = None;

    // Walk `messages[*]` — the only field that affects the return
    // value. We log `system` and `tools` markers below for parity
    // with the design doc, but they don't bump the floor.
    walk_messages(parsed, &mut highest_message_index);

    // Walk `system` blocks for logging + TTL-ordering check only.
    // These markers never bump `frozen_count`; the system field is
    // always part of the cache hot zone independently.
    walk_system(parsed);

    // Walk `tools[*]` blocks for logging + TTL-ordering check only.
    walk_tools(parsed);

    // Translate the highest-marker index into a frozen-count floor.
    // The "+1" makes the floor exclusive: `messages[i]` itself is
    // part of the cached prefix, so the live-zone dispatcher must
    // not touch any index up to and including `i`.
    highest_message_index.map(|i| i + 1).unwrap_or(0)
}

/// Walk `parsed.messages[*].content[*]` and update
/// `highest_message_index` for any `cache_control` marker found.
///
/// The walker tolerates two content shapes Anthropic accepts:
/// - String content: `messages[i].content` is a JSON string. No
///   block list, so no `cache_control` marker possible.
/// - Block list: `messages[i].content` is an array. Each block is
///   an object that MAY have a top-level `cache_control` field.
fn walk_messages(parsed: &Value, highest_message_index: &mut Option<usize>) {
    let Some(messages) = parsed.get("messages").and_then(Value::as_array) else {
        return;
    };

    // Track per-message-list TTL ordering: across the entire
    // messages[*].content[*] sequence, every `1h` marker must
    // precede every `5m` marker. We log a single warning if the
    // rule is violated, regardless of how many violations there are
    // — the customer just needs to know once.
    let mut ttl_walk = TtlOrderingWalk::new();

    for (i, message) in messages.iter().enumerate() {
        let Some(content) = message.get("content") else {
            continue;
        };
        let Some(blocks) = content.as_array() else {
            // String content: no block list, no markers possible.
            continue;
        };
        for block in blocks {
            if let Some(marker) = block.get("cache_control") {
                let ttl = extract_ttl(marker);
                tracing::debug!(
                    field = "messages",
                    message_index = i,
                    ttl = ttl.as_deref().unwrap_or("default"),
                    "cache_control marker found"
                );
                ttl_walk.observe(ttl.as_deref());
                // Bump the floor.
                *highest_message_index = Some(match highest_message_index {
                    Some(prev) => (*prev).max(i),
                    None => i,
                });
            }
        }
    }

    ttl_walk.warn_if_violated("messages");
}

/// Walk `parsed.system` for `cache_control` markers. Logs at
/// `tracing::debug!` per marker; emits TTL-ordering warning. Does NOT
/// affect `frozen_count` — the system field is unconditionally
/// cache-hot.
///
/// `system` may be a string (no markers possible) or an array of
/// blocks. Mirrors the `messages[*].content` shape rules.
fn walk_system(parsed: &Value) {
    let Some(system) = parsed.get("system") else {
        return;
    };
    let Some(blocks) = system.as_array() else {
        // String system prompt: no block list, no markers.
        return;
    };
    let mut ttl_walk = TtlOrderingWalk::new();
    for block in blocks {
        if let Some(marker) = block.get("cache_control") {
            let ttl = extract_ttl(marker);
            tracing::debug!(
                field = "system",
                ttl = ttl.as_deref().unwrap_or("default"),
                "cache_control marker found"
            );
            ttl_walk.observe(ttl.as_deref());
        }
    }
    ttl_walk.warn_if_violated("system");
}

/// Walk `parsed.tools[*].cache_control` markers. Logs at
/// `tracing::debug!`; emits TTL-ordering warning. Does NOT affect
/// `frozen_count` — `tools` is unconditionally cache-hot.
fn walk_tools(parsed: &Value) {
    let Some(tools) = parsed.get("tools").and_then(Value::as_array) else {
        return;
    };
    let mut ttl_walk = TtlOrderingWalk::new();
    for (i, tool) in tools.iter().enumerate() {
        if let Some(marker) = tool.get("cache_control") {
            let ttl = extract_ttl(marker);
            tracing::debug!(
                field = "tools",
                tool_index = i,
                ttl = ttl.as_deref().unwrap_or("default"),
                "cache_control marker found"
            );
            ttl_walk.observe(ttl.as_deref());
        }
    }
    ttl_walk.warn_if_violated("tools");
}

/// Pull the optional `ttl` string out of a `cache_control` marker.
///
/// The marker is shaped `{"type": "ephemeral", "ttl": "1h"}` — the
/// `type` field is always `"ephemeral"` (Anthropic's only legal
/// value today) and `ttl` is optional, defaulting to `5m`. We
/// deliberately don't normalise here: returning `None` lets callers
/// distinguish "default 5m" from "explicit 5m" if they ever need to.
///
/// Returns `None` when `marker` isn't an object, when there's no
/// `ttl` key, or when `ttl` isn't a string.
fn extract_ttl(marker: &Value) -> Option<String> {
    marker.get("ttl")?.as_str().map(str::to_owned)
}

/// State machine for the TTL-ordering check (guide §2.19).
///
/// As we walk a sequence of markers, we record whether we've seen a
/// `5m` marker yet. If we then see a `1h` marker after that, the
/// rule is violated: `1h` markers must precede `5m`.
///
/// We accept all orderings (the customer's request, not ours to
/// reject) but emit one `tracing::warn!` per field-list when a
/// violation is detected, scoped to the field name passed to
/// `warn_if_violated` (e.g. `"messages"`, `"system"`, `"tools"`).
struct TtlOrderingWalk {
    seen_5m: bool,
    violated: bool,
}

impl TtlOrderingWalk {
    fn new() -> Self {
        Self {
            seen_5m: false,
            violated: false,
        }
    }

    /// Record one observed marker. `ttl` is `Some("1h")`, `Some("5m")`,
    /// `Some(other)`, or `None` (defaulting to 5m semantics). Only
    /// `1h`/`5m` participate in the ordering rule; unknown TTL values
    /// don't affect the state machine.
    fn observe(&mut self, ttl: Option<&str>) {
        // Default TTL is "5m" per guide §2.19. Treat None and
        // `Some("5m")` identically for ordering purposes.
        let is_5m = matches!(ttl, None | Some(CACHE_TTL_5M));
        let is_1h = matches!(ttl, Some(CACHE_TTL_1H));

        if is_5m {
            self.seen_5m = true;
        } else if is_1h && self.seen_5m {
            self.violated = true;
        }
    }

    fn warn_if_violated(&self, field: &'static str) {
        if self.violated {
            tracing::warn!(
                field = field,
                rule = "anthropic_prompt_caching_guide_2_19",
                "cache_control TTL ordering violation: 1h marker appears after 5m marker; \
                 cache eviction may be suboptimal but request is forwarded"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn no_markers_yields_zero() {
        let body = json!({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
        });
        assert_eq!(compute_frozen_count(&body), 0);
    }

    #[test]
    fn marker_at_message_zero_yields_one() {
        let body = json!({
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "first", "cache_control": {"type": "ephemeral"}},
                ]},
                {"role": "assistant", "content": "second"},
            ],
        });
        assert_eq!(compute_frozen_count(&body), 1);
    }

    #[test]
    fn marker_in_system_does_not_bump() {
        let body = json!({
            "system": [
                {"type": "text", "text": "you are helpful", "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [
                {"role": "user", "content": "hi"},
            ],
        });
        assert_eq!(compute_frozen_count(&body), 0);
    }

    #[test]
    fn marker_in_tools_does_not_bump() {
        let body = json!({
            "tools": [
                {"name": "search", "description": "search", "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [
                {"role": "user", "content": "hi"},
            ],
        });
        assert_eq!(compute_frozen_count(&body), 0);
    }

    #[test]
    fn missing_messages_yields_zero() {
        let body = json!({"model": "claude"});
        assert_eq!(compute_frozen_count(&body), 0);
    }

    #[test]
    fn string_content_yields_zero() {
        // String-shaped content can't carry a cache_control marker;
        // the walker must skip over it without panicking.
        let body = json!({
            "messages": [
                {"role": "user", "content": "plain string"},
                {"role": "assistant", "content": "another string"},
            ],
        });
        assert_eq!(compute_frozen_count(&body), 0);
    }

    #[test]
    fn ttl_extracted_when_present() {
        let m = json!({"type": "ephemeral", "ttl": "1h"});
        assert_eq!(extract_ttl(&m).as_deref(), Some("1h"));
    }

    #[test]
    fn ttl_missing_returns_none() {
        let m = json!({"type": "ephemeral"});
        assert_eq!(extract_ttl(&m), None);
    }

    #[test]
    fn ttl_walker_accepts_1h_before_5m() {
        let mut w = TtlOrderingWalk::new();
        w.observe(Some("1h"));
        w.observe(Some("5m"));
        assert!(!w.violated);
    }

    #[test]
    fn ttl_walker_flags_5m_before_1h() {
        let mut w = TtlOrderingWalk::new();
        w.observe(Some("5m"));
        w.observe(Some("1h"));
        assert!(w.violated);
    }

    #[test]
    fn ttl_walker_treats_default_as_5m() {
        let mut w = TtlOrderingWalk::new();
        w.observe(None);
        w.observe(Some("1h"));
        assert!(w.violated);
    }
}
