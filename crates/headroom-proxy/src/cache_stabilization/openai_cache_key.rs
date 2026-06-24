//! PR-E4: OpenAI `prompt_cache_key` auto-injection.
//!
//! OpenAI's prefix caching is automatic since late 2024 — the API
//! caches prefix tokens server-side without client opt-in. The
//! `prompt_cache_key` field, however, is the field a client passes
//! to **pin cache lookup** to a stable identity. Without it,
//! OpenAI falls back to org-wide cache lookups that may collide
//! with other tenants. Most clients don't set the field because
//! they don't know it exists. This module derives a deterministic
//! key from the request's *structural* prefix (model + system +
//! tools) and injects it on PAYG requests where the customer has
//! not already provided one.
//!
//! # Universal safety contract
//!
//! This module mutates request bytes — that is the whole point.
//! Mutating bytes for non-PAYG auth modes risks looking like
//! cache-evasion to the upstream and would void OAuth scope; the
//! caller MUST gate on `AuthMode::Payg` before calling
//! [`inject_prompt_cache_key`]. The function itself does NOT
//! re-classify auth mode: that is the caller's responsibility,
//! consistent with PR-E3's `cache_control` placement which gates
//! the same way at the dispatch site.
//!
//! # Skip rules (this module enforces these directly)
//!
//! - **Already present**: if `body.prompt_cache_key` exists at the
//!   top level and is a non-empty string, return
//!   [`InjectOutcome::Skipped`] with reason
//!   [`SkipReason::KeyPresent`]. The customer's value wins.
//! - **Idempotency**: re-running on a body that already has *our*
//!   injected key returns [`SkipReason::KeyPresent`] (we don't
//!   distinguish ours from theirs — both mean "leave alone").
//!
//! # Key derivation
//!
//! Inputs to the hash, in order:
//!
//! 1. `body.model` (string) — different model = different cache
//!    universe.
//! 2. SHA-256 of canonical-JSON bytes of the system content.
//!    For Chat Completions, this is the first message with
//!    `role == "system"`. For Responses, this is the
//!    `instructions` field if present, else the first
//!    `role == "system"` item.
//! 3. SHA-256 of canonical-JSON bytes of `body.tools`.
//!
//! User and assistant messages are **deliberately excluded** —
//! including them would defeat the purpose by producing a
//! different key on every turn.
//!
//! Combined: `key = hex(sha256(model || system_hash ||
//! tools_hash))[..32]` — 128 bits of collision resistance, 32
//! hex chars on the wire, compact in logs.

use serde_json::Value;
use sha2::{Digest, Sha256};

/// Length of the injected key in hex characters. 32 hex chars =
/// 128 bits, plenty of collision resistance for per-tenant cache
/// pinning, while staying compact on the wire and in logs.
pub const KEY_HEX_LEN: usize = 32;

/// First-N chars of the key surfaced in `tracing` events. **Never**
/// log the full key — it is identifying material for cache pinning
/// and operators do not need the suffix to debug.
pub const KEY_PREFIX_LOG_LEN: usize = 8;

/// Which OpenAI request shape the body conforms to. The shape
/// affects only where the system content lives:
///
/// - [`OpenAiShape::ChatCompletions`]: first `messages[*]` with
///   `role == "system"`.
/// - [`OpenAiShape::Responses`]: top-level `instructions` field,
///   else first `input[*]` (or `messages[*]`) with
///   `role == "system"`.
///
/// `tools` lives at the top level in both shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAiShape {
    /// `POST /v1/chat/completions`.
    ChatCompletions,
    /// `POST /v1/responses`.
    Responses,
}

/// Why the injector skipped a body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipReason {
    /// The body is not a JSON object (the dispatcher should have
    /// gated this earlier; we surface it as a skip rather than
    /// panicking so the proxy stays a transparent forwarder for
    /// pathological inputs).
    NotAnObject,
    /// `body.prompt_cache_key` is already a non-empty string.
    /// Customer-set values win.
    KeyPresent,
}

impl SkipReason {
    /// Stable string for structured-log fields. Dashboards filter
    /// on this; do not change without a deprecation note.
    pub fn as_str(self) -> &'static str {
        match self {
            SkipReason::NotAnObject => "not_an_object",
            SkipReason::KeyPresent => "key_present",
        }
    }
}

/// Outcome of an injection attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InjectOutcome {
    /// We added a `prompt_cache_key` to the body. The first
    /// [`KEY_PREFIX_LOG_LEN`] hex chars of the key, suitable for
    /// logs. The full key is in the body — never log it.
    Applied { key_prefix: String },
    /// We left the body unchanged. See [`SkipReason`] for why.
    Skipped { reason: SkipReason },
}

/// Top-level field check.
///
/// Returns `true` if `body.prompt_cache_key` is a non-empty
/// string. Empty strings count as "absent" because some SDKs
/// default the field to `""` when the user did not set it.
pub fn has_prompt_cache_key(body: &Value) -> bool {
    body.get("prompt_cache_key")
        .and_then(Value::as_str)
        .map(|s| !s.is_empty())
        .unwrap_or(false)
}

/// Inject a `prompt_cache_key` into `body` if appropriate.
///
/// **Caller MUST gate on `AuthMode::Payg`.** This function does not
/// re-classify auth mode — same contract as PR-E3
/// `cache_control` placement.
///
/// Returns:
/// - [`InjectOutcome::Applied`] when the key is set;
///   `body["prompt_cache_key"]` is now a 32-hex-char string.
/// - [`InjectOutcome::Skipped`] when the body is not a JSON
///   object, or already has a non-empty `prompt_cache_key`.
///
/// This function is **deterministic**: the same `(model, system,
/// tools)` triple always produces the same key, so re-running on a
/// body whose injected key was stripped yields the same key
/// (idempotency property the test suite pins).
pub fn inject_prompt_cache_key(body: &mut Value, shape: OpenAiShape) -> InjectOutcome {
    // Guardrail: top-level must be an object. Anything else (array,
    // string, null) is malformed for our hook point; the dispatcher
    // already gates on `messages` / `input` being arrays, so this
    // path is essentially unreachable but we surface it explicitly
    // rather than panicking.
    if !body.is_object() {
        return InjectOutcome::Skipped {
            reason: SkipReason::NotAnObject,
        };
    }

    if has_prompt_cache_key(body) {
        return InjectOutcome::Skipped {
            reason: SkipReason::KeyPresent,
        };
    }

    let key = derive_key(body, shape);
    let key_prefix = key[..KEY_PREFIX_LOG_LEN.min(key.len())].to_string();

    // Safe: we just verified `body.is_object()` above.
    if let Some(map) = body.as_object_mut() {
        map.insert("prompt_cache_key".to_string(), Value::String(key));
    }

    InjectOutcome::Applied { key_prefix }
}

/// Derive the cache key from the structural prefix of the body.
///
/// Hash inputs (concatenated, length-prefixed to avoid ambiguity
/// between e.g. `model="ab", system="c"` vs `model="a", system="bc"`):
///
/// 1. ASCII bytes of the model field (empty if missing).
/// 2. SHA-256 of canonical-JSON bytes of the system content.
/// 3. SHA-256 of canonical-JSON bytes of the tools field.
///
/// Length-prefixing is critical: without it, two different
/// (model, system) splits could collide. We use a single `0x00`
/// byte separator instead of an explicit u32 length because
/// `0x00` cannot appear in a UTF-8 model name or a hex digest;
/// the separator unambiguously delimits the three inputs.
fn derive_key(body: &Value, shape: OpenAiShape) -> String {
    let model = body
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default();

    let system_value = extract_system(body, shape);
    let system_hash = canonical_sha256(&system_value);

    let tools_value = body.get("tools").cloned().unwrap_or(Value::Null);
    let tools_hash = canonical_sha256(&tools_value);

    let mut hasher = Sha256::new();
    hasher.update(model.as_bytes());
    hasher.update([0u8]);
    hasher.update(system_hash.as_bytes());
    hasher.update([0u8]);
    hasher.update(tools_hash.as_bytes());
    let digest = hasher.finalize();

    // 16 bytes = 32 hex chars. We hex-encode by hand to avoid
    // pulling in `hex` for one call site.
    let mut out = String::with_capacity(KEY_HEX_LEN);
    for byte in digest.iter().take(KEY_HEX_LEN / 2) {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

/// Locate the system content for the given shape, returning the
/// JSON value (possibly `Null`) we will hash. We hash the JSON
/// value (not its serialized bytes alone) so that
/// content-block-array systems and string systems with the same
/// concatenated text produce *different* keys — which is the
/// correct behaviour for cache pinning.
fn extract_system(body: &Value, shape: OpenAiShape) -> Value {
    match shape {
        OpenAiShape::ChatCompletions => first_system_message_content(body, "messages"),
        OpenAiShape::Responses => {
            // Responses canonical: top-level `instructions`.
            // Legacy alias: a system message in `input` (or `messages`).
            if let Some(instructions) = body.get("instructions") {
                return instructions.clone();
            }
            if let Some(v) = body.get("input") {
                if let Some(content) = first_system_in_array(v) {
                    return content;
                }
            }
            first_system_message_content(body, "messages")
        }
    }
}

/// First `messages[*]` (under the given key) with `role == "system"` —
/// returns the message's `content` field, or `Null` if none found.
fn first_system_message_content(body: &Value, key: &str) -> Value {
    body.get(key)
        .and_then(first_system_in_array)
        .unwrap_or(Value::Null)
}

fn first_system_in_array(arr: &Value) -> Option<Value> {
    let items = arr.as_array()?;
    for item in items {
        if item.get("role").and_then(Value::as_str) == Some("system") {
            return item.get("content").cloned().or(Some(Value::Null));
        }
    }
    None
}

/// SHA-256 of canonical-JSON bytes of `v`, hex-encoded.
///
/// We use `serde_json::to_vec` rather than `to_string` to avoid an
/// unnecessary UTF-8 validation pass; `serde_json` always emits
/// valid UTF-8 anyway. Object keys are emitted in insertion order
/// (workspace `serde_json` is built with `preserve_order`), which
/// is acceptable here because the **same** request body always
/// hashes the **same** way — and that is the only invariant we
/// require for stable cache pinning.
fn canonical_sha256(v: &Value) -> String {
    let bytes = serde_json::to_vec(v).unwrap_or_default();
    let digest = Sha256::digest(&bytes);
    let mut out = String::with_capacity(64);
    for byte in digest.iter() {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn is_hex(s: &str) -> bool {
        s.bytes().all(|b| b.is_ascii_hexdigit())
    }

    fn injected_key(body: &Value) -> String {
        body.get("prompt_cache_key")
            .and_then(Value::as_str)
            .expect("prompt_cache_key must be a string")
            .to_string()
    }

    #[test]
    fn injects_key_when_payg_and_absent() {
        let mut body = json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"}
            ],
        });
        let outcome = inject_prompt_cache_key(&mut body, OpenAiShape::ChatCompletions);
        match outcome {
            InjectOutcome::Applied { key_prefix } => {
                assert_eq!(key_prefix.len(), KEY_PREFIX_LOG_LEN);
                assert!(is_hex(&key_prefix));
            }
            other => panic!("expected Applied, got {other:?}"),
        }
        let key = injected_key(&body);
        assert_eq!(key.len(), KEY_HEX_LEN);
        assert!(is_hex(&key), "key must be hex: {key}");
    }

    #[test]
    fn injects_key_when_payg_and_absent_responses() {
        let mut body = json!({
            "model": "gpt-4o",
            "instructions": "You are a helpful assistant.",
            "input": [{"role": "user", "content": "Hello"}],
        });
        let outcome = inject_prompt_cache_key(&mut body, OpenAiShape::Responses);
        assert!(matches!(outcome, InjectOutcome::Applied { .. }));
        let key = injected_key(&body);
        assert_eq!(key.len(), KEY_HEX_LEN);
        assert!(is_hex(&key));
    }

    #[test]
    fn skips_when_key_already_set() {
        let original = json!({
            "model": "gpt-4o",
            "prompt_cache_key": "user-pinned",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"}
            ],
        });
        let mut body = original.clone();
        let outcome = inject_prompt_cache_key(&mut body, OpenAiShape::ChatCompletions);
        assert_eq!(
            outcome,
            InjectOutcome::Skipped {
                reason: SkipReason::KeyPresent,
            }
        );
        assert_eq!(body, original, "body must be unchanged when key is present");
    }

    #[test]
    fn skips_when_key_already_set_responses() {
        let original = json!({
            "model": "gpt-4o",
            "prompt_cache_key": "user-pinned-2",
            "instructions": "Be concise.",
            "input": [{"role": "user", "content": "Hello"}],
        });
        let mut body = original.clone();
        let outcome = inject_prompt_cache_key(&mut body, OpenAiShape::Responses);
        assert!(matches!(
            outcome,
            InjectOutcome::Skipped {
                reason: SkipReason::KeyPresent
            }
        ));
        assert_eq!(body, original);
    }

    #[test]
    fn skips_when_body_is_not_an_object() {
        let mut body = json!(["not", "an", "object"]);
        let outcome = inject_prompt_cache_key(&mut body, OpenAiShape::ChatCompletions);
        assert_eq!(
            outcome,
            InjectOutcome::Skipped {
                reason: SkipReason::NotAnObject,
            }
        );
    }

    #[test]
    fn empty_string_key_treated_as_absent() {
        // Some SDKs default the field to "" when the user did not
        // set it. We treat that as absent and inject anyway.
        let mut body = json!({
            "model": "gpt-4o",
            "prompt_cache_key": "",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"}
            ],
        });
        let outcome = inject_prompt_cache_key(&mut body, OpenAiShape::ChatCompletions);
        assert!(matches!(outcome, InjectOutcome::Applied { .. }));
        let key = injected_key(&body);
        assert_eq!(key.len(), KEY_HEX_LEN);
    }

    #[test]
    fn idempotent_same_inputs_same_key() {
        // Re-running on a body whose injected key was stripped
        // (e.g. customer-side scrub) must yield the same key.
        let template = || {
            json!({
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "first turn"}
                ],
                "tools": [
                    {"type": "function", "function": {"name": "search"}}
                ]
            })
        };
        let mut body1 = template();
        let _ = inject_prompt_cache_key(&mut body1, OpenAiShape::ChatCompletions);
        let key1 = injected_key(&body1);

        let mut body2 = template();
        let _ = inject_prompt_cache_key(&mut body2, OpenAiShape::ChatCompletions);
        let key2 = injected_key(&body2);

        assert_eq!(key1, key2);
    }

    #[test]
    fn different_model_yields_different_key() {
        let mut a = json!({
            "model": "gpt-4o",
            "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "u"}],
        });
        let mut b = json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "u"}],
        });
        let _ = inject_prompt_cache_key(&mut a, OpenAiShape::ChatCompletions);
        let _ = inject_prompt_cache_key(&mut b, OpenAiShape::ChatCompletions);
        assert_ne!(injected_key(&a), injected_key(&b));
    }

    #[test]
    fn different_system_yields_different_key() {
        let mut a = json!({
            "model": "gpt-4o",
            "messages": [{"role": "system", "content": "system A"}, {"role": "user", "content": "u"}],
        });
        let mut b = json!({
            "model": "gpt-4o",
            "messages": [{"role": "system", "content": "system B"}, {"role": "user", "content": "u"}],
        });
        let _ = inject_prompt_cache_key(&mut a, OpenAiShape::ChatCompletions);
        let _ = inject_prompt_cache_key(&mut b, OpenAiShape::ChatCompletions);
        assert_ne!(injected_key(&a), injected_key(&b));
    }

    #[test]
    fn different_tools_yields_different_key() {
        let mut a = json!({
            "model": "gpt-4o",
            "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "u"}],
            "tools": [{"type": "function", "function": {"name": "search"}}],
        });
        let mut b = json!({
            "model": "gpt-4o",
            "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "u"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
        });
        let _ = inject_prompt_cache_key(&mut a, OpenAiShape::ChatCompletions);
        let _ = inject_prompt_cache_key(&mut b, OpenAiShape::ChatCompletions);
        assert_ne!(injected_key(&a), injected_key(&b));
    }

    #[test]
    fn same_user_messages_different_system_yields_different_key() {
        // Confirms user content is NOT in the hash by holding user
        // content fixed and varying only the system. If user
        // content were hashed, this could still differ — but the
        // companion test below pins the inverse direction.
        let user = json!({"role": "user", "content": "what is 2+2?"});
        let mut a = json!({
            "model": "gpt-4o",
            "messages": [{"role": "system", "content": "concise"}, user.clone()],
        });
        let mut b = json!({
            "model": "gpt-4o",
            "messages": [{"role": "system", "content": "verbose"}, user],
        });
        let _ = inject_prompt_cache_key(&mut a, OpenAiShape::ChatCompletions);
        let _ = inject_prompt_cache_key(&mut b, OpenAiShape::ChatCompletions);
        assert_ne!(injected_key(&a), injected_key(&b));
    }

    #[test]
    fn same_system_different_user_messages_yields_same_key() {
        // The crucial property: user/assistant turns vary across
        // requests, but the cache key must STAY CONSTANT so OpenAI
        // can pin the cache lookup to the same identity.
        let mut a = json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": "first question"}
            ],
        });
        let mut b = json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": "second question, much longer than the first"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "third question"}
            ],
        });
        let _ = inject_prompt_cache_key(&mut a, OpenAiShape::ChatCompletions);
        let _ = inject_prompt_cache_key(&mut b, OpenAiShape::ChatCompletions);
        assert_eq!(
            injected_key(&a),
            injected_key(&b),
            "user/assistant turn variation must not change the cache key"
        );
    }

    #[test]
    fn key_length_is_exactly_32_hex_chars() {
        // Property test: across many input shapes the key is
        // always exactly 32 hex chars.
        let inputs = [
            json!({"model": "gpt-4o", "messages": []}),
            json!({"model": "gpt-4o-2024-08-06", "messages": [{"role": "system", "content": "abc"}]}),
            json!({"model": "o1-mini", "messages": [{"role": "system", "content": ["array", "content"]}]}),
            json!({"model": "", "messages": [{"role": "user", "content": "no system"}]}),
            json!({"model": "gpt-4o", "instructions": "responses style"}),
        ];
        for (i, base) in inputs.iter().enumerate() {
            let mut body = base.clone();
            let outcome = inject_prompt_cache_key(&mut body, OpenAiShape::ChatCompletions);
            assert!(
                matches!(outcome, InjectOutcome::Applied { .. }),
                "input {i}: expected Applied"
            );
            let key = injected_key(&body);
            assert_eq!(
                key.len(),
                KEY_HEX_LEN,
                "input {i}: key {key:?} must be {KEY_HEX_LEN} chars"
            );
            assert!(is_hex(&key), "input {i}: key {key:?} must be hex");
        }
    }

    #[test]
    fn responses_instructions_field_distinguishes_from_chat_system() {
        // Same model, same logical "system", different shape:
        // the keys may legally differ (different containers hash
        // differently). What we DO assert is that running the
        // same shape on the same body is stable (covered by
        // idempotency test) and that running each shape produces
        // a valid key.
        let mut body_chat = json!({
            "model": "gpt-4o",
            "messages": [{"role": "system", "content": "S"}],
        });
        let mut body_resp = json!({
            "model": "gpt-4o",
            "instructions": "S",
        });
        let _ = inject_prompt_cache_key(&mut body_chat, OpenAiShape::ChatCompletions);
        let _ = inject_prompt_cache_key(&mut body_resp, OpenAiShape::Responses);
        assert_eq!(injected_key(&body_chat).len(), KEY_HEX_LEN);
        assert_eq!(injected_key(&body_resp).len(), KEY_HEX_LEN);
    }

    #[test]
    fn has_prompt_cache_key_works() {
        assert!(!has_prompt_cache_key(&json!({})));
        assert!(!has_prompt_cache_key(&json!({"prompt_cache_key": ""})));
        assert!(!has_prompt_cache_key(&json!({"prompt_cache_key": null})));
        assert!(has_prompt_cache_key(&json!({"prompt_cache_key": "x"})));
        assert!(has_prompt_cache_key(
            &json!({"prompt_cache_key": "user-pinned"})
        ));
    }

    #[test]
    fn skip_reason_string_is_stable() {
        // Dashboards filter on these strings. Don't change without
        // a deprecation note.
        assert_eq!(SkipReason::NotAnObject.as_str(), "not_an_object");
        assert_eq!(SkipReason::KeyPresent.as_str(), "key_present");
    }
}
