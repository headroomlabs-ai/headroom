//! Model name → context window (tokens) lookup, sourced from LiteLLM.
//!
//! # Why LiteLLM and not a hand-rolled table
//!
//! Earlier drafts of this module hardcoded a small `if/else` chain
//! covering Claude / GPT-4o / GPT-3.5. Two failure modes that cost us:
//!
//! 1. **Static rot.** Anthropic and OpenAI ship new models monthly.
//!    A hardcoded table goes stale between Headroom releases; new
//!    models silently fall through to a default that is often wrong.
//! 2. **Long tail.** Bedrock, Cohere, Mistral, Gemini, AzureML — each
//!    has dozens of model variants. We aren't going to maintain
//!    a comprehensive table by hand.
//!
//! [LiteLLM] maintains `model_prices_and_context_window.json` as a
//! community-curated source of truth: ~2000 chat models, refreshed
//! weekly, with `max_input_tokens` (the field we care about) plus
//! pricing, output limits, capability flags. Every other LLM-tool
//! ecosystem (Portkey, Helicone, OpenRouter, Continue) pulls from it.
//!
//! [LiteLLM]: https://github.com/BerriAI/litellm
//!
//! # Vendoring strategy
//!
//! We **check the JSON into the repo** at
//! `crates/headroom-proxy/data/model_prices_and_context_window.json`
//! and `include_str!` it at compile time. Reasons over alternatives:
//!
//! - **Build-time fetch (`build.rs` + curl):** breaks for offline /
//!   air-gapped builds — bad for BYOC where customers may not allow
//!   outbound network during install.
//! - **Runtime fetch:** same problem, plus a startup-failure surface
//!   we don't need.
//!
//! Refresh is operator-driven: `scripts/refresh_model_limits.sh`
//! re-pulls and validates the JSON, the diff lands in a regular PR.
//! We trade "always fresh" for "deterministic, offline-buildable,
//! auditable in version control" — the right trade for a deploy
//! artifact that's expected to run in customer VPCs.
//!
//! # Performance
//!
//! The 1.4MB JSON parses in ~10ms one-time on first lookup. Parsed
//! result is cached in a `OnceLock<HashMap>` so subsequent lookups
//! are O(1). When `--compression` is off, the JSON is in the binary
//! image but never parsed — zero runtime cost.
//!
//! # Default for unknown models
//!
//! When a model isn't in the table, we return a conservative 128K
//! and emit a `tracing::warn!` (once per unknown model id). 128K is
//! the dominant context window across modern frontier models; being
//! wrong here means we either over-compress (safe — we just trim
//! unnecessarily, the request still works) or under-compress (the
//! upstream rejects it with `context_length_exceeded`, which is
//! recoverable by the client).

use std::collections::HashMap;
use std::sync::OnceLock;

/// Conservative default for unknown models. Modern frontier models
/// almost universally have ≥128K context; we err on the side of
/// over-compressing (safe) rather than under-compressing (broken).
pub(crate) const DEFAULT_CONTEXT_WINDOW: u32 = 128_000;

/// LiteLLM's vendored model price + context-window table. Refreshed
/// via `scripts/refresh_model_limits.sh`. ~1.4MB; embedded into the
/// binary so the proxy ships with no startup network dependency.
const VENDORED_JSON: &str = include_str!("../../data/model_prices_and_context_window.json");

/// Parsed lookup: model id → max input tokens. Built lazily on
/// first call; subsequent calls reuse the same `HashMap`.
static TABLE: OnceLock<HashMap<String, u32>> = OnceLock::new();

/// Looks up `max_input_tokens` for `model`. Returns
/// [`DEFAULT_CONTEXT_WINDOW`] when the model isn't in the table.
///
/// Lookup is exact-match by model id. We deliberately do NOT do
/// prefix matching — model versions are semantically distinct
/// (`claude-3-5-sonnet-20241022` was 200K, but a hypothetical
/// `claude-3-5-sonnet-mini` may be different) and a prefix rule
/// would cause silent wrong answers.
pub fn context_window_for(model: &str) -> u32 {
    let table = TABLE.get_or_init(parse_vendored);
    if let Some(&n) = table.get(model) {
        return n;
    }
    // Unknown model. We don't log here on every miss — that's
    // per-request noise. The caller (compression::anthropic) logs
    // once, with the model id, when this happens. Just return the
    // default and let the caller handle observability.
    DEFAULT_CONTEXT_WINDOW
}

/// Walk the LiteLLM JSON and extract the chat-model context windows.
///
/// LiteLLM's schema: top-level object whose keys are model ids.
/// Values may be:
/// - `sample_spec` — a template entry; skipped.
/// - Image / audio / embedding entries — `mode != "chat"`; skipped.
/// - Chat entries — have `max_input_tokens` (preferred) or
///   `max_tokens` (legacy fallback).
///
/// The 79-ish chat entries in the current snapshot that lack BOTH
/// fields are skipped silently — they'd hit `DEFAULT_CONTEXT_WINDOW`
/// at lookup time anyway.
fn parse_vendored() -> HashMap<String, u32> {
    let raw: serde_json::Value = serde_json::from_str(VENDORED_JSON)
        .expect("vendored LiteLLM JSON must parse at build time");
    let obj = raw
        .as_object()
        .expect("LiteLLM JSON must be a top-level object");

    // Slight over-allocation; better than reallocating during the walk.
    let mut out: HashMap<String, u32> = HashMap::with_capacity(obj.len());
    for (key, val) in obj {
        if key == "sample_spec" {
            continue;
        }
        let entry = match val.as_object() {
            Some(o) => o,
            None => continue,
        };
        // Only chat-mode models are relevant for our compressor.
        // Image / audio / embedding endpoints don't have a "messages"
        // array we can compress.
        if entry.get("mode").and_then(|m| m.as_str()) != Some("chat") {
            continue;
        }

        // Prefer max_input_tokens. Fall back to max_tokens (older
        // entries used max_tokens as a synonym for input window).
        let n = entry
            .get("max_input_tokens")
            .and_then(|v| v.as_u64())
            .or_else(|| entry.get("max_tokens").and_then(|v| v.as_u64()));
        let Some(n) = n else { continue };

        // u32 fits every realistic context window. The largest known
        // today is ~10M (Magic.dev, hypothetical) — still under
        // 4 billion. If a future model crosses u32::MAX we have
        // larger problems than this `as`.
        out.insert(key.clone(), n.min(u32::MAX as u64) as u32);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vendored_json_parses_at_runtime() {
        // Calling this once forces parse via OnceLock. If the JSON
        // is malformed, this test panics with a useful error before
        // any lookup test runs. Subsequent tests in this module
        // share the same parsed table.
        let table = TABLE.get_or_init(parse_vendored);
        assert!(
            table.len() > 100,
            "expected >100 chat models in LiteLLM snapshot, got {}",
            table.len()
        );
    }

    #[test]
    fn current_claude_models_present() {
        // Lock against the snapshot rotting silently. If LiteLLM
        // renames the canonical Claude entry we want a test failure
        // — not a silent fall-through to DEFAULT_CONTEXT_WINDOW.
        // Pick a model we expect to remain stable: claude-sonnet-4-5
        // (current as of the snapshot fetch).
        let n = context_window_for("claude-sonnet-4-5-20250929");
        assert_eq!(n, 200_000, "claude-sonnet-4-5 should be 200K input window");
    }

    #[test]
    fn current_gpt_models_present() {
        assert_eq!(context_window_for("gpt-4o-mini"), 128_000);
        assert_eq!(context_window_for("gpt-4-turbo"), 128_000);
    }

    #[test]
    fn unknown_model_returns_default() {
        assert_eq!(
            context_window_for("definitely-not-a-real-model-2099"),
            DEFAULT_CONTEXT_WINDOW
        );
        assert_eq!(context_window_for(""), DEFAULT_CONTEXT_WINDOW);
    }

    #[test]
    fn empty_or_garbage_string_does_not_panic() {
        // The lookup must not panic on adversarial input — bad
        // model strings come from the wire and we forward unknown
        // ones rather than failing the request.
        let _ = context_window_for("");
        let _ = context_window_for("\0\0\0");
        let _ = context_window_for(&"x".repeat(10_000));
    }

    #[test]
    fn sample_spec_entry_is_excluded() {
        // LiteLLM's JSON includes a "sample_spec" template entry
        // documenting the schema. It must not appear as a real
        // model in our lookup — a request specifying it would
        // otherwise get a bogus context window.
        let table = TABLE.get_or_init(parse_vendored);
        assert!(!table.contains_key("sample_spec"));
    }
}
