//! PR-E5: volatile-content detector.
//!
//! Scans inbound LLM request bodies for substrings that are known to
//! bust prompt-cache hits when they appear inside the cached prefix
//! (system prompt, tool definitions, historical messages):
//!
//!   1. **ISO-8601 timestamps** (`YYYY-MM-DDTHH:MM:SS...`) — almost
//!      always rendered freshly per request, so any cache hit on a
//!      prefix containing one is accidental.
//!   2. **UUID v4** (`xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx`) — the
//!      version-4 nibble at position 14 distinguishes UUIDs the
//!      caller is generating per-request from random hex strings
//!      (build hashes are usually v0, fixed identifiers wouldn't
//!      change between requests at all).
//!   3. **ID-named JSON fields** — keys whose name matches one of
//!      the known volatile field names (`request_id`, `trace_id`,
//!      `session_id`, `correlation_id`). Even non-empty UUIDs
//!      embedded in a normal text field are caught by (1)/(2);
//!      this rule catches values that the volatile-substring scan
//!      would miss (e.g. integer trace IDs, custom slug formats).
//!
//! # Non-mutation invariant
//!
//! This module **never** mutates the request body. It takes
//! `&serde_json::Value` (already parsed by the proxy gate above)
//! and walks read-only. The Phase A cache-safety invariant —
//! bytes-in == bytes-out for any non-modifying request — still
//! holds; the caller's `debug_assert_eq!` on length and the
//! integration tests' SHA-256 byte-equality continue to gate
//! regressions.
//!
//! # Detection policy
//!
//! - **No regex.** Realignment build-constraints policy bans regex
//!   for parsing (it hides intent and slows cold start). Each
//!   pattern is recognized via explicit byte-position checks.
//! - **Cap findings at 10 per request.** A noisy customer payload
//!   (think: a CSV pasted into the system prompt) could otherwise
//!   produce hundreds of warnings, drowning the logs. The cap is
//!   conservative; in practice the first 1-3 findings are the
//!   ones the customer will act on.
//! - **Sample truncated to 80 chars.** We log a small slice so the
//!   customer can locate the offending content, but never log
//!   bulk customer data.
//! - **Path scoping.** OpenAI and Anthropic shape their JSON
//!   bodies differently; the [`ApiKind`] enum picks the right
//!   walker. Bedrock / Vertex / etc. follow in a Phase E
//!   follow-up — this PR keeps the surface tight.

use serde_json::Value;

use crate::compression::CompressibleEndpoint;

/// Maximum findings reported per request. See module docs for rationale.
pub const MAX_FINDINGS_PER_REQUEST: usize = 10;

/// Maximum bytes of `sample` we log per finding. Lifts a small
/// excerpt so the operator can find the offending content without
/// exposing bulk customer data.
pub const SAMPLE_TRUNCATE_BYTES: usize = 80;

/// JSON field names that are conventionally per-request unique IDs.
/// Matched case-insensitively against the *substring* of a key —
/// a key named `"x_request_id"` or `"meta.session_id"` is caught.
const ID_FIELD_NEEDLES: &[&str] = &["request_id", "trace_id", "session_id", "correlation_id"];

/// What kind of volatile content we found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolatileKind {
    /// ISO-8601 timestamp shape: positions 4=`-`, 7=`-`, 10=`T`, 13=`:`, 16=`:`.
    Timestamp,
    /// UUID v4 shape: 36 chars, hex, hyphens at 8/13/18/23, version
    /// nibble `4` at position 14.
    Uuid,
    /// JSON key whose name contains one of the conventionally
    /// per-request ID needles (`request_id`, `trace_id`,
    /// `session_id`, `correlation_id`).
    IdField,
}

impl VolatileKind {
    /// Stable string representation for structured logging. The
    /// detection rules in dashboards filter on this; do not change
    /// the strings without a deprecation note.
    pub fn as_str(self) -> &'static str {
        match self {
            VolatileKind::Timestamp => "iso8601_timestamp",
            VolatileKind::Uuid => "uuid_v4",
            VolatileKind::IdField => "id_field",
        }
    }
}

/// One volatile-content finding.
///
/// `location` is a JSON-pointer-style path so the customer can map
/// the warning back to the exact field in their request shape (e.g.
/// `system[2].text`, `messages[0].content[1].text`,
/// `tools[0].input_schema.properties.session_id`). Sample is a
/// truncated excerpt; [`SAMPLE_TRUNCATE_BYTES`] caps it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VolatileFinding {
    pub kind: VolatileKind,
    pub location: String,
    pub sample: String,
}

/// Which provider's body shape to walk. Selected by the caller
/// from the request path — see [`from_endpoint`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiKind {
    /// Anthropic `/v1/messages` shape: top-level `system` (string
    /// or content blocks), `messages[].content` (string or
    /// blocks), `tools[].description` + `tools[].input_schema`.
    Anthropic,
    /// OpenAI Chat Completions / Responses shape: top-level
    /// `messages[].content`, `tools[].function.description` +
    /// `tools[].function.parameters`.
    OpenAi,
}

impl ApiKind {
    /// Map a [`CompressibleEndpoint`] (already classified by the
    /// proxy gate) to the corresponding [`ApiKind`]. Bedrock /
    /// Vertex / etc. are not yet wired — their walkers land in a
    /// follow-up PR.
    pub fn from_endpoint(endpoint: CompressibleEndpoint) -> Self {
        match endpoint {
            CompressibleEndpoint::AnthropicMessages => ApiKind::Anthropic,
            CompressibleEndpoint::OpenAiChatCompletions | CompressibleEndpoint::OpenAiResponses => {
                ApiKind::OpenAi
            }
        }
    }
}

/// Public detection entry point.
///
/// Walks the parsed body for the given API shape and returns up to
/// [`MAX_FINDINGS_PER_REQUEST`] findings. Caller is responsible for
/// passing the *parsed* body — re-parsing on the hot path here
/// would double the JSON cost.
pub fn detect_volatile_content(body: &Value, kind: ApiKind) -> Vec<VolatileFinding> {
    let mut findings: Vec<VolatileFinding> = Vec::new();
    match kind {
        ApiKind::Anthropic => walk_anthropic(body, &mut findings),
        ApiKind::OpenAi => walk_openai(body, &mut findings),
    }
    findings
}

/// Emit one `tracing::warn!` per finding with a stable structured
/// shape. Operators / customers consume `event="volatile_content_detected"`
/// in their log search to surface cache-busting content.
pub fn emit_volatile_warnings(findings: &[VolatileFinding], request_id: &str) {
    for finding in findings {
        tracing::warn!(
            event = "volatile_content_detected",
            request_id = %request_id,
            kind = finding.kind.as_str(),
            location = %finding.location,
            sample = %finding.sample,
            "volatile content in cached prefix will bust prompt-cache hits; \
             move per-request IDs/timestamps to message metadata or post-prefix \
             fields"
        );
    }
}

// ─── Anthropic walker ──────────────────────────────────────────────────

fn walk_anthropic(body: &Value, out: &mut Vec<VolatileFinding>) {
    if !out.is_empty() && out.len() >= MAX_FINDINGS_PER_REQUEST {
        return;
    }
    // system: string | array of content blocks
    if let Some(system) = body.get("system") {
        scan_value_for_strings(system, "system", out);
    }
    // messages[].content: string | array of blocks
    if let Some(Value::Array(messages)) = body.get("messages") {
        for (i, msg) in messages.iter().enumerate() {
            if out.len() >= MAX_FINDINGS_PER_REQUEST {
                return;
            }
            if let Some(content) = msg.get("content") {
                let loc = format!("messages[{i}].content");
                scan_value_for_strings(content, &loc, out);
            }
        }
    }
    // tools[].description + tools[].input_schema
    if let Some(Value::Array(tools)) = body.get("tools") {
        for (i, tool) in tools.iter().enumerate() {
            if out.len() >= MAX_FINDINGS_PER_REQUEST {
                return;
            }
            if let Some(Value::String(desc)) = tool.get("description") {
                let loc = format!("tools[{i}].description");
                scan_string(desc, &loc, out);
            }
            if let Some(schema) = tool.get("input_schema") {
                let loc = format!("tools[{i}].input_schema");
                scan_value_recursive(schema, &loc, out);
            }
        }
    }
}

// ─── OpenAI walker ─────────────────────────────────────────────────────

fn walk_openai(body: &Value, out: &mut Vec<VolatileFinding>) {
    // messages[].content: string | array of parts
    if let Some(Value::Array(messages)) = body.get("messages") {
        for (i, msg) in messages.iter().enumerate() {
            if out.len() >= MAX_FINDINGS_PER_REQUEST {
                return;
            }
            if let Some(content) = msg.get("content") {
                let loc = format!("messages[{i}].content");
                scan_value_for_strings(content, &loc, out);
            }
        }
    }
    // tools[].function.description + tools[].function.parameters
    if let Some(Value::Array(tools)) = body.get("tools") {
        for (i, tool) in tools.iter().enumerate() {
            if out.len() >= MAX_FINDINGS_PER_REQUEST {
                return;
            }
            if let Some(function) = tool.get("function") {
                if let Some(Value::String(desc)) = function.get("description") {
                    let loc = format!("tools[{i}].function.description");
                    scan_string(desc, &loc, out);
                }
                if let Some(params) = function.get("parameters") {
                    let loc = format!("tools[{i}].function.parameters");
                    scan_value_recursive(params, &loc, out);
                }
            }
        }
    }
}

// ─── Generic walkers ───────────────────────────────────────────────────

/// Scan a [`Value`] that may be a string, an array of content
/// blocks, or some other shape. Strings are scanned for volatile
/// substrings; objects / arrays of blocks are recursed.
fn scan_value_for_strings(v: &Value, location: &str, out: &mut Vec<VolatileFinding>) {
    if out.len() >= MAX_FINDINGS_PER_REQUEST {
        return;
    }
    match v {
        Value::String(s) => scan_string(s, location, out),
        Value::Array(items) => {
            for (i, item) in items.iter().enumerate() {
                if out.len() >= MAX_FINDINGS_PER_REQUEST {
                    return;
                }
                let nested = format!("{location}[{i}]");
                scan_value_recursive(item, &nested, out);
            }
        }
        Value::Object(_) => scan_value_recursive(v, location, out),
        _ => {}
    }
}

/// Walk a [`Value`] recursively for both string-content scanning
/// and ID-named-key detection. This is the only walker that
/// inspects keys: tool input_schemas / function parameters / nested
/// content blocks all flow through here.
fn scan_value_recursive(v: &Value, location: &str, out: &mut Vec<VolatileFinding>) {
    if out.len() >= MAX_FINDINGS_PER_REQUEST {
        return;
    }
    match v {
        Value::String(s) => scan_string(s, location, out),
        Value::Array(items) => {
            for (i, item) in items.iter().enumerate() {
                if out.len() >= MAX_FINDINGS_PER_REQUEST {
                    return;
                }
                let nested = format!("{location}[{i}]");
                scan_value_recursive(item, &nested, out);
            }
        }
        Value::Object(map) => {
            for (k, sub) in map.iter() {
                if out.len() >= MAX_FINDINGS_PER_REQUEST {
                    return;
                }
                if is_id_named_key(k) && !is_value_empty(sub) {
                    out.push(VolatileFinding {
                        kind: VolatileKind::IdField,
                        location: format!("{location}.{k}"),
                        sample: truncate_sample(&value_to_sample(sub)),
                    });
                    if out.len() >= MAX_FINDINGS_PER_REQUEST {
                        return;
                    }
                }
                let nested = format!("{location}.{k}");
                scan_value_recursive(sub, &nested, out);
            }
        }
        _ => {}
    }
}

/// Scan a string for ISO-8601 timestamps and UUID v4 substrings.
/// Multiple occurrences in the same string each produce a finding,
/// up to the global cap.
fn scan_string(s: &str, location: &str, out: &mut Vec<VolatileFinding>) {
    let bytes = s.as_bytes();
    let len = bytes.len();
    // ISO-8601 minimum window: `YYYY-MM-DDTHH:MM:SS` = 19 bytes.
    // UUID v4 window: 36 bytes.
    // Walk by byte index; both checks are pure byte-position lookups.
    let mut i = 0usize;
    while i < len {
        if out.len() >= MAX_FINDINGS_PER_REQUEST {
            return;
        }
        // Try ISO-8601 first (shorter window means fewer false-misses
        // when the string ends mid-UUID).
        if i + 19 <= len && looks_like_iso8601(&bytes[i..i + 19]) {
            let end = (i + 19).min(len);
            out.push(VolatileFinding {
                kind: VolatileKind::Timestamp,
                location: location.to_string(),
                sample: truncate_sample(&s[i..end]),
            });
            i += 19;
            continue;
        }
        if i + 36 <= len && looks_like_uuid_v4(&bytes[i..i + 36]) {
            out.push(VolatileFinding {
                kind: VolatileKind::Uuid,
                location: location.to_string(),
                sample: truncate_sample(&s[i..i + 36]),
            });
            i += 36;
            continue;
        }
        i += 1;
    }
}

/// Is the 19-byte window an ISO-8601 timestamp prefix?
/// Positions 0..4: 4 ASCII digits (year).
/// Position 4: `-`.
/// Positions 5..7: 2 ASCII digits (month).
/// Position 7: `-`.
/// Positions 8..10: 2 ASCII digits (day).
/// Position 10: `T` or `t` or ` ` (space — RFC 3339 §5.6 allows it).
/// Positions 11..13: 2 ASCII digits (hour).
/// Position 13: `:`.
/// Positions 14..16: 2 ASCII digits (minute).
/// Position 16: `:`.
/// Positions 17..19: 2 ASCII digits (second).
fn looks_like_iso8601(window: &[u8]) -> bool {
    if window.len() < 19 {
        return false;
    }
    let digits_in =
        |range: std::ops::Range<usize>| -> bool { window[range].iter().all(u8::is_ascii_digit) };
    digits_in(0..4)
        && window[4] == b'-'
        && digits_in(5..7)
        && window[7] == b'-'
        && digits_in(8..10)
        && (window[10] == b'T' || window[10] == b't' || window[10] == b' ')
        && digits_in(11..13)
        && window[13] == b':'
        && digits_in(14..16)
        && window[16] == b':'
        && digits_in(17..19)
}

/// Is the 36-byte window a UUID v4?
/// Hyphens at 8, 13, 18, 23.
/// Position 14 = `4` (version nibble).
/// Position 19 in `{8, 9, a, b, A, B}` (variant nibble per RFC 4122 §4.4).
/// All other positions: ASCII hex.
fn looks_like_uuid_v4(window: &[u8]) -> bool {
    if window.len() < 36 {
        return false;
    }
    if window[8] != b'-' || window[13] != b'-' || window[18] != b'-' || window[23] != b'-' {
        return false;
    }
    if window[14] != b'4' {
        return false;
    }
    match window[19] {
        b'8' | b'9' | b'a' | b'b' | b'A' | b'B' => {}
        _ => return false,
    }
    for (i, &c) in window.iter().enumerate().take(36) {
        if i == 8 || i == 13 || i == 18 || i == 23 {
            continue;
        }
        if !c.is_ascii_hexdigit() {
            return false;
        }
    }
    true
}

/// Does the JSON key name match one of the conventional per-request
/// ID needles? Case-insensitive substring match.
fn is_id_named_key(key: &str) -> bool {
    let lowered = key.to_ascii_lowercase();
    ID_FIELD_NEEDLES
        .iter()
        .any(|needle| lowered.contains(needle))
}

/// Treat empty strings, empty arrays/objects, and null as "no value
/// present" so the ID-field rule doesn't false-positive on schemas
/// that *declare* a `request_id` field but pass it as `""`.
fn is_value_empty(v: &Value) -> bool {
    match v {
        Value::Null => true,
        Value::String(s) => s.is_empty(),
        Value::Array(a) => a.is_empty(),
        Value::Object(m) => m.is_empty(),
        _ => false,
    }
}

/// Render a JSON value into a short sample string. Strings flow
/// through verbatim (then truncated); other primitives go through
/// `to_string`. Objects/arrays render as their compact JSON.
fn value_to_sample(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        // Compact JSON keeps the sample small; the truncation step
        // below caps it regardless.
        _ => v.to_string(),
    }
}

/// Truncate `s` to at most [`SAMPLE_TRUNCATE_BYTES`] bytes,
/// respecting UTF-8 boundaries (truncating mid-codepoint would
/// produce invalid UTF-8 and panic later when written to logs).
fn truncate_sample(s: &str) -> String {
    if s.len() <= SAMPLE_TRUNCATE_BYTES {
        return s.to_string();
    }
    // Find the largest char boundary <= SAMPLE_TRUNCATE_BYTES.
    let mut cut = SAMPLE_TRUNCATE_BYTES;
    while cut > 0 && !s.is_char_boundary(cut) {
        cut -= 1;
    }
    let mut out = String::with_capacity(cut + 1);
    out.push_str(&s[..cut]);
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn detects_iso8601_timestamp_in_system_prompt() {
        let body = json!({
            "system": "Today is 2026-05-04T14:30:00Z. Be concise.",
            "messages": [],
        });
        let findings = detect_volatile_content(&body, ApiKind::Anthropic);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].kind, VolatileKind::Timestamp);
        assert_eq!(findings[0].location, "system");
        assert!(
            findings[0].sample.starts_with("2026-05-04T14:30:00"),
            "sample should be the ISO-8601 substring, got {:?}",
            findings[0].sample
        );
    }

    #[test]
    fn detects_uuid_v4_in_user_message() {
        let body = json!({
            "messages": [
                {"role": "user", "content": "trace=550e8400-e29b-41d4-a716-446655440000"},
            ],
        });
        let findings = detect_volatile_content(&body, ApiKind::Anthropic);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].kind, VolatileKind::Uuid);
        assert_eq!(findings[0].location, "messages[0].content");
        assert_eq!(findings[0].sample, "550e8400-e29b-41d4-a716-446655440000");
    }

    #[test]
    fn detects_request_id_field_in_nested_object() {
        // Tools input_schema with a nested `request_id` field whose
        // value is a non-UUID string. The volatile-substring scan
        // would miss this; the ID-field-name rule catches it.
        let body = json!({
            "tools": [{
                "name": "lookup",
                "description": "Look up a user.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                        "request_id": "req-2026-abc-12345"
                    }
                }
            }],
            "messages": [],
        });
        let findings = detect_volatile_content(&body, ApiKind::Anthropic);
        let id_field = findings
            .iter()
            .find(|f| f.kind == VolatileKind::IdField)
            .expect("expected an IdField finding");
        assert!(
            id_field.location.ends_with(".request_id"),
            "location should end with .request_id, got {:?}",
            id_field.location
        );
        assert!(id_field.sample.contains("req-2026-abc-12345"));
    }

    #[test]
    fn stable_content_yields_zero_findings() {
        // Plain prose with no timestamps, no UUIDs, no ID-named keys.
        let body = json!({
            "system": "You are a helpful assistant. Be concise.",
            "messages": [
                {"role": "user", "content": "Summarize the document below."},
                {"role": "assistant", "content": "Sure — please paste it."},
            ],
            "tools": [{
                "name": "search",
                "description": "Search the corpus.",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}}
                }
            }],
        });
        let findings = detect_volatile_content(&body, ApiKind::Anthropic);
        assert!(
            findings.is_empty(),
            "expected zero findings on stable content, got {findings:?}",
        );
    }

    #[test]
    fn caps_findings_at_ten() {
        // Build a body with many UUID-bearing user messages so the
        // detector would otherwise emit > 10 findings.
        let mut messages = Vec::new();
        for i in 0..30 {
            messages.push(json!({
                "role": "user",
                "content": format!("turn {i}: 550e8400-e29b-41d4-a716-446655440000"),
            }));
        }
        let body = json!({"messages": messages});
        let findings = detect_volatile_content(&body, ApiKind::Anthropic);
        assert_eq!(
            findings.len(),
            MAX_FINDINGS_PER_REQUEST,
            "detector must cap findings at {MAX_FINDINGS_PER_REQUEST}",
        );
    }

    #[test]
    fn does_not_mutate_input() {
        let body = json!({
            "system": "Today is 2026-05-04T14:30:00Z.",
            "messages": [{
                "role": "user",
                "content": "trace=550e8400-e29b-41d4-a716-446655440000",
            }],
            "tools": [{
                "name": "lookup",
                "description": "Look up a user.",
                "input_schema": {
                    "type": "object",
                    "properties": {"request_id": "req-abc"}
                }
            }],
        });
        let before = serde_json::to_vec(&body).expect("serialize before");
        let _findings = detect_volatile_content(&body, ApiKind::Anthropic);
        let after = serde_json::to_vec(&body).expect("serialize after");
        assert_eq!(before, after, "detector must NOT mutate input body bytes",);
    }

    #[test]
    fn apikind_anthropic_scans_correct_paths() {
        // Anthropic shape: tools[].description + tools[].input_schema.
        let body = json!({
            "tools": [{
                "name": "lookup",
                "description": "scheduled at 2026-05-04T10:00:00Z",
                "input_schema": {"type": "object"}
            }],
        });
        let anthropic_findings = detect_volatile_content(&body, ApiKind::Anthropic);
        assert_eq!(anthropic_findings.len(), 1);
        assert_eq!(anthropic_findings[0].kind, VolatileKind::Timestamp);
        assert_eq!(
            anthropic_findings[0].location, "tools[0].description",
            "Anthropic shape: tools[].description (NOT tools[].function.description)",
        );

        // OpenAI shape on the same body should find nothing — it
        // expects tools[].function.description, not tools[].description.
        let openai_findings = detect_volatile_content(&body, ApiKind::OpenAi);
        assert!(
            openai_findings.is_empty(),
            "OpenAI walker must not match Anthropic shape, got {openai_findings:?}",
        );

        // OpenAI-shape body matches the OpenAI walker.
        let openai_body = json!({
            "tools": [{
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "scheduled at 2026-05-04T10:00:00Z",
                    "parameters": {"type": "object"}
                }
            }],
        });
        let openai_findings = detect_volatile_content(&openai_body, ApiKind::OpenAi);
        assert_eq!(openai_findings.len(), 1);
        assert_eq!(openai_findings[0].kind, VolatileKind::Timestamp);
        assert_eq!(openai_findings[0].location, "tools[0].function.description",);
    }

    #[test]
    fn id_field_with_empty_value_does_not_fire() {
        // request_id present but empty — schemas / clients that
        // declare the field but don't fill it shouldn't trigger.
        let body = json!({
            "tools": [{
                "input_schema": {
                    "properties": {"request_id": ""}
                }
            }],
        });
        let findings = detect_volatile_content(&body, ApiKind::Anthropic);
        assert!(
            findings.iter().all(|f| f.kind != VolatileKind::IdField),
            "empty ID-field values must not trigger; got {findings:?}",
        );
    }

    #[test]
    fn iso8601_with_space_separator_recognized() {
        // RFC 3339 §5.6 allows a space in place of `T`. Ops logs
        // commonly render it that way; we accept it to keep the
        // detector helpful.
        let body = json!({"system": "started at 2026-05-04 14:30:00"});
        let findings = detect_volatile_content(&body, ApiKind::Anthropic);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].kind, VolatileKind::Timestamp);
    }

    #[test]
    fn random_hex_without_v4_nibble_is_not_a_uuid() {
        // 36-char shape with hyphens but version nibble != 4 (here
        // the position-14 char is `0`, e.g. a synthesised legacy
        // identifier). Must NOT be flagged as UUID.
        let body = json!({
            "messages": [{
                "role": "user",
                "content": "id=550e8400-e29b-01d4-a716-446655440000",
            }],
        });
        let findings = detect_volatile_content(&body, ApiKind::Anthropic);
        assert!(
            findings.iter().all(|f| f.kind != VolatileKind::Uuid),
            "non-v4 UUID-shaped string must not match v4 detector; got {findings:?}",
        );
    }

    #[test]
    fn truncate_sample_respects_utf8_boundaries() {
        // 80 bytes of ASCII followed by a multi-byte codepoint at
        // the cut. Must not panic and must not produce invalid UTF-8.
        let mut s = "a".repeat(SAMPLE_TRUNCATE_BYTES);
        s.push('é'); // 2 bytes
        let out = truncate_sample(&s);
        // Round-trip through String (would panic on invalid UTF-8).
        let _ = out.as_bytes();
        assert!(out.ends_with('…'));
    }

    #[test]
    fn from_endpoint_maps_correctly() {
        assert_eq!(
            ApiKind::from_endpoint(CompressibleEndpoint::AnthropicMessages),
            ApiKind::Anthropic,
        );
        assert_eq!(
            ApiKind::from_endpoint(CompressibleEndpoint::OpenAiChatCompletions),
            ApiKind::OpenAi,
        );
        assert_eq!(
            ApiKind::from_endpoint(CompressibleEndpoint::OpenAiResponses),
            ApiKind::OpenAi,
        );
    }
}
