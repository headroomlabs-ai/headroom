//! Live-zone block dispatcher — Phase B.
//!
//! # The mental model
//!
//! After Phase B PR-B1 retired the message-dropping machinery, all
//! compression happens *within* messages, never *between* them. The
//! live-zone dispatcher walks the request body and identifies the
//! *live zone*: the blocks the model will emit a response *against*,
//! which are the only ones whose bytes can mutate without busting the
//! provider's prompt cache.
//!
//! # Provider scope
//!
//! Phase B ships ONE dispatcher entry point —
//! [`compress_anthropic_live_zone`] — that handles the Anthropic
//! Messages API shape (`/v1/messages`). Other providers (OpenAI
//! Chat Completions, OpenAI Responses, Google Gemini, Bedrock with
//! native payloads, …) need their own dispatchers because their
//! request shapes diverge in load-bearing ways:
//!
//! - OpenAI Chat Completions puts tool results in their own
//!   `role: "tool"` messages, not nested in user messages.
//! - OpenAI Responses uses `input` (not `messages`) with item types
//!   like `function_call_output` and `reasoning`.
//! - Gemini uses `contents`/`parts`/`function_response`.
//!
//! Phase C (`REALIGNMENT/05-phase-C-rust-proxy.md`) introduces
//! `compress_openai_chat_live_zone`, `compress_openai_responses_live_zone`,
//! and friends. They share this module's provider-agnostic types
//! ([`LiveZoneOutcome`], [`BlockAction`], [`CompressionManifest`])
//! and the per-content-type compressor backend, but each owns its
//! own walker.
//!
//! For Anthropic `/v1/messages`, the live zone is bounded by:
//!
//! - **Floor:** `frozen_message_count` (computed by
//!   [`crate::compute_frozen_count`] from explicit `cache_control`
//!   markers; passed in here). Indices below the floor are in the
//!   prompt cache and MUST be byte-identical.
//! - **Ceiling:** the latest user message. The latest assistant
//!   message (if any) is part of the cache hot zone too — it's what
//!   the next response continues from. We never touch it.
//! - **Inside the latest user message:** every block is a candidate.
//!   The most common compressible block type is `tool_result`
//!   (because tool outputs dominate token budgets); `text` blocks
//!   are also eligible (e.g. user pastes a long log).
//!
//! # Phase B build-up
//!
//! - **PR-B2** shipped the dispatcher *skeleton*: identify live-zone
//!   blocks, route to no-op compressors, always return `NoChange`.
//! - **PR-B3** (this PR) wires per-content-type compressors:
//!   `JsonArray` → SmartCrusher; `BuildOutput` → LogCompressor;
//!   `SearchResults` → SearchCompressor; `GitDiff` → DiffCompressor;
//!   `SourceCode` / `PlainText` / `Html` → no-op (B4 + a Rust
//!   code-compressor port follow-up).
//! - **PR-B4** adds the tokenizer-validation gate (per-block
//!   `compressed.tokens >= original.tokens` → fall back) and the
//!   per-content-type byte threshold below which compression is
//!   skipped.
//! - **PR-B7** wires CCR retrieval-marker injection.
//!
//! # Cache safety invariant
//!
//! Bytes outside the live zone are NEVER touched. PR-B3 writes new
//! bodies via **byte-range surgery**: we locate each rewritten block
//! by pointer arithmetic on `serde_json::value::RawValue` borrowed
//! slices (which retain their offset into the original buffer), then
//! splice the replacement into the output. Concretely:
//!
//! ```text
//!     out = body[..block_start] || replacement || body[block_end..]
//! ```
//!
//! The bytes outside the rewritten ranges are *literally copied*
//! from the input, never re-serialized. This is how we guarantee
//! the SHA-256 of the prefix and suffix are byte-identical to the
//! input — Phase A's fixtures and B3's `byte_fidelity_outside_compressed_block`
//! test pin this in CI.
//!
//! Why byte-range surgery and not "deserialize → mutate → serialize"?
//! Re-serializing a JSON `Value` does not preserve original
//! whitespace, key order subtleties, or numeric formatting that the
//! provider may have already cached against. Byte-faithful copy of
//! everything we don't touch is the only way to guarantee
//! cache stability — see `project_compression_realignment_2026_05`.
//!
//! # AuthMode
//!
//! The `AuthMode` parameter is taken in B3 but unused — Phase F
//! PR-F2 wires the gate (PAYG/OAuth/Subscription each demand
//! different policies; see project memory
//! `project_auth_mode_compression_nuances.md`). Keeping the
//! parameter in the signature now means later PRs are pure
//! implementation swaps, not signature redesigns.

use std::sync::OnceLock;

use serde::Deserialize;
use serde_json::value::RawValue;
use serde_json::Value;
use thiserror::Error;

use super::content_detector::{detect_content_type, ContentType};
use super::diff_compressor::{DiffCompressor, DiffCompressorConfig};
use super::log_compressor::{LogCompressor, LogCompressorConfig};
use super::search_compressor::{SearchCompressor, SearchCompressorConfig};
use super::smart_crusher::{SmartCrusher, SmartCrusherConfig};
use crate::ccr::{compute_key, marker_for, CcrStore};
use crate::tokenizer::get_tokenizer;

// ─── Tunable constants (no magic numbers in the dispatch logic) ────────

/// Strategy tag emitted when SmartCrusher rewrote a JSON-array block.
const STRATEGY_SMART_CRUSHER: &str = "smart_crusher";
/// Strategy tag emitted when LogCompressor rewrote a build-output / log block.
const STRATEGY_LOG_COMPRESSOR: &str = "log_compressor";
/// Strategy tag emitted when SearchCompressor rewrote a grep / ripgrep block.
const STRATEGY_SEARCH_COMPRESSOR: &str = "search_compressor";
/// Strategy tag emitted when DiffCompressor rewrote a unified-diff block.
const STRATEGY_DIFF_COMPRESSOR: &str = "diff_compressor";

/// Empty query context passed to compressors that take a relevance
/// query string. PR-B3 dispatcher does not yet plumb the user's last
/// prompt through; PR-F3 will.
const EMPTY_QUERY: &str = "";
/// Default relevance bias passed to scoring-aware compressors. Mirrors
/// the OSS-default behaviour ("no bias").
const DEFAULT_BIAS: f64 = 0.0;

/// Default model name handed to the tokenizer registry when the proxy
/// could not extract `body["model"]`. Matches the most-common
/// production Claude model — chars-per-token estimator for `claude-*`
/// is calibrated to 3.5 cpt; using a non-Claude model here would
/// silently pick a different estimator density. PR-F3 will plumb the
/// actual model from `body["model"]`; PR-B4 just establishes the
/// signature.
pub const DEFAULT_MODEL: &str = "claude-3-5-sonnet-20241022";

// ─── Per-content-type byte thresholds ──────────────────────────────────
//
// Below these byte sizes the dispatcher does not even attempt
// compression — the per-block overhead (tokenizer count, dispatcher
// bookkeeping, log lines) costs more than the marginal token savings,
// and tiny inputs almost never compress at all.
//
// Sourced from the spec (`REALIGNMENT/04-phase-B-live-zone.md::PR-B4`).
// Pinned as `const` rather than a hard-coded `match` so the values are
// grep-able and reviewable in one place.

/// JSON-array tool_results below this size route to no-op (1 KiB).
const THRESHOLD_JSON_ARRAY: usize = 1024;
/// Build / log output below this size routes to no-op (512 B). Logs
/// are the most repetitive content type so the threshold is the
/// lowest of the bunch.
const THRESHOLD_BUILD_OUTPUT: usize = 512;
/// Search-result blocks below this size route to no-op (1 KiB).
const THRESHOLD_SEARCH_RESULTS: usize = 1024;
/// Git-diff blocks below this size route to no-op (1 KiB).
const THRESHOLD_GIT_DIFF: usize = 1024;
/// Source-code blocks below this size route to no-op (2 KiB). Pinned
/// for the future Rust code-compressor port — currently unused
/// because `ContentType::SourceCode` short-circuits to no-op above
/// the dispatch (see `dispatch_compressor`).
const THRESHOLD_SOURCE_CODE: usize = 2048;
/// Plain-text blocks below this size route to no-op (5 KiB). Pinned
/// for the future Kompress wiring (PR-B7 follow-up); currently unused.
const THRESHOLD_PLAIN_TEXT: usize = 5120;
/// HTML blocks have no compressor; threshold matches plain text so
/// when an HTML compressor lands the value is already pinned.
const THRESHOLD_HTML: usize = 5120;

/// Map a content type to its byte threshold. Returning `usize` rather
/// than an `Option` because every variant has a sensible default;
/// `Html` is a no-op anyway so the threshold check never fires.
fn threshold_for(content_type: ContentType) -> usize {
    match content_type {
        ContentType::JsonArray => THRESHOLD_JSON_ARRAY,
        ContentType::BuildOutput => THRESHOLD_BUILD_OUTPUT,
        ContentType::SearchResults => THRESHOLD_SEARCH_RESULTS,
        ContentType::GitDiff => THRESHOLD_GIT_DIFF,
        ContentType::SourceCode => THRESHOLD_SOURCE_CODE,
        ContentType::PlainText => THRESHOLD_PLAIN_TEXT,
        ContentType::Html => THRESHOLD_HTML,
    }
}

// ─── Public types ──────────────────────────────────────────────────────

/// Authentication mode of the originating request. Passed through to
/// the dispatcher so PR-F2 can vary policy without re-shaping the
/// public API. PR-B3 ignores the value (always treated as `Payg`).
///
/// Also reused by [`super::recommendations`] (PR-B5) as the lookup
/// key prefix — keeping one canonical enum avoids drift between the
/// dispatcher's auth slice and the published recommendations'.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AuthMode {
    /// Pay-as-you-go API key. Most aggressive compression budget —
    /// every saved token is real money for the customer.
    Payg,
    /// OAuth-bearing client (e.g. Anthropic.com OAuth). Compression
    /// must not break the per-account routing the OAuth header pins;
    /// otherwise behaves like PAYG.
    OAuth,
    /// Subscription seat (e.g. Claude.ai usage). The provider
    /// already counts tokens against a fixed quota; aggressive
    /// compression is less compelling and may interact badly with
    /// rate-limit accounting.
    Subscription,
    /// Auth slice not yet detected. Matches the Python TOIN publish
    /// CLI's "unknown" default. Used by the recommendations loader
    /// (PR-B5) when an aggregation row didn't carry an auth tag.
    Unknown,
}

impl AuthMode {
    /// String form used as the recommendations-store lookup key.
    /// Mirrors the Python publish CLI tag values.
    pub fn as_str(self) -> &'static str {
        match self {
            AuthMode::Payg => "payg",
            AuthMode::OAuth => "oauth",
            AuthMode::Subscription => "subscription",
            AuthMode::Unknown => "unknown",
        }
    }
}

/// Per-block decision recorded for observability. Independent of
/// whether the body was actually rewritten.
#[derive(Debug, Clone)]
pub struct BlockOutcome {
    /// Index into the `messages` array.
    pub message_index: usize,
    /// Index into the message's `content` array. `None` when the
    /// content is a plain string (Anthropic accepts both shapes).
    pub block_index: Option<usize>,
    /// Block kind detected on this slot. `text`, `tool_result`,
    /// `tool_use`, `image`, ... or `string_content` for the
    /// string-shaped fallback.
    pub block_type: String,
    /// What the dispatcher decided.
    pub action: BlockAction,
}

/// Disposition of one block.
#[derive(Debug, Clone)]
pub enum BlockAction {
    /// Content type was inspected, no compressor was applicable.
    /// Examples: `PlainText` (Kompress wires in PR-B4), `SourceCode`
    /// (Rust code-compressor port pending), `Html` (no compressor),
    /// `Image` (binary), unknown shapes.
    NoCompressionApplied {
        /// String form of the detected content type — `"text"`,
        /// `"source_code"`, `"html"`, `"image"`, `"unknown"`, etc.
        content_type: String,
    },
    /// A compressor ran and produced a smaller output (in tokens, as
    /// counted by the model's tokenizer) that was spliced into the
    /// body. Both byte and token counts are reported so the proxy
    /// can log the savings ratio in either currency.
    Compressed {
        /// Identifier of the compressor (`"smart_crusher"`,
        /// `"log_compressor"`, ...). Static so the manifest is
        /// allocation-light.
        strategy: &'static str,
        /// Bytes of the original block content (the JSON string
        /// value, after unescaping).
        original_bytes: usize,
        /// Bytes of the replacement block content.
        compressed_bytes: usize,
        /// Tokens in the original block content (per the model's
        /// tokenizer).
        original_tokens: usize,
        /// Tokens in the replacement block content. Always strictly
        /// less than `original_tokens` for this variant — the
        /// tokenizer-validated rejection gate (PR-B4) maps the
        /// `>=` case to `RejectedNotSmaller`.
        compressed_tokens: usize,
    },
    /// A compressor was tried but failed loudly. Per project memory
    /// `feedback_no_silent_fallbacks.md`: surface the error in the
    /// manifest; the proxy logs warn-level and forwards the original
    /// bytes for that block (other blocks in the same body still get
    /// compressed normally).
    CompressorError {
        /// Identifier of the compressor that failed.
        strategy: &'static str,
        /// Human-readable error string (from `Display`).
        error: String,
    },
    /// A compressor ran but produced output that did not shrink the
    /// token count. Cache safety + "don't make it worse" → keep the
    /// original. PR-B4 wired the tokenizer-validated check; both
    /// byte and token counts are reported for observability.
    RejectedNotSmaller {
        /// Identifier of the compressor that was rejected.
        strategy: &'static str,
        /// Original block-content size, bytes.
        original_bytes: usize,
        /// Would-be compressed-block-content size, bytes.
        compressed_bytes: usize,
        /// Original block-content size, tokens.
        original_tokens: usize,
        /// Would-be compressed-block-content size, tokens. Always
        /// `>= original_tokens` (otherwise this would be
        /// `Compressed`).
        compressed_tokens: usize,
    },
    /// The block content was below the per-content-type byte
    /// threshold; no compressor was invoked. The dispatcher does
    /// not even spin up the tokenizer for these — they're below the
    /// per-call overhead so the marginal savings are negative.
    BelowByteThreshold {
        /// Detected content type — string tag matches
        /// `ContentType::as_str`.
        content_type: &'static str,
        /// Bytes in the block content.
        byte_count: usize,
        /// Threshold (in bytes) the content failed to clear.
        threshold_bytes: usize,
    },
    /// Block type is intentionally outside the live zone (e.g.
    /// `tool_use` → cache hot zone) and is excluded from dispatch.
    Excluded { reason: ExclusionReason },
}

/// Why a block was not eligible for compression.
#[derive(Debug, Clone, Copy)]
pub enum ExclusionReason {
    /// Block is in a message at index `< frozen_message_count`.
    BelowFrozenFloor,
    /// Block belongs to a message above the latest user message
    /// boundary (e.g. an older assistant turn).
    AboveLiveZone,
    /// Block type is on the cache-hot list (e.g. `tool_use`,
    /// `thinking`, `redacted_thinking`).
    HotZoneBlockType,
}

/// Aggregated per-request manifest. Always populated, regardless of
/// whether any bytes were written.
#[derive(Debug, Clone)]
pub struct CompressionManifest {
    /// Total messages in the input array. Matches
    /// `body.messages.len()`.
    pub messages_total: usize,
    /// Messages with index `< frozen_message_count`. Untouched.
    pub messages_below_frozen_floor: usize,
    /// Index of the latest user message in the live zone, if any.
    pub latest_user_message_index: Option<usize>,
    /// Per-block outcomes for the latest user message. Empty when
    /// the live zone has no eligible blocks (or the body has no
    /// messages).
    pub block_outcomes: Vec<BlockOutcome>,
}

impl CompressionManifest {
    fn empty() -> Self {
        Self {
            messages_total: 0,
            messages_below_frozen_floor: 0,
            latest_user_message_index: None,
            block_outcomes: Vec::new(),
        }
    }

    /// True when at least one block was actually rewritten by a
    /// compressor (used to discriminate the `Modified` arm from
    /// `NoChange`).
    fn has_compressed_block(&self) -> bool {
        self.block_outcomes
            .iter()
            .any(|b| matches!(b.action, BlockAction::Compressed { .. }))
    }
}

/// Outcome of dispatching the live zone.
#[derive(Debug)]
pub enum LiveZoneOutcome {
    /// No bytes were rewritten. The caller must forward the original
    /// buffered request body byte-for-byte.
    NoChange { manifest: CompressionManifest },
    /// The dispatcher rewrote at least one block and emitted a fresh
    /// body. The caller forwards `new_body` upstream.
    Modified {
        new_body: Box<RawValue>,
        manifest: CompressionManifest,
    },
}

/// Dispatcher errors. Every variant is recoverable by the caller —
/// the proxy turns each into a structured warn-level log and
/// falls back to forwarding the original bytes.
#[derive(Debug, Error)]
pub enum LiveZoneError {
    /// The request body is not valid JSON.
    #[error("request body is not valid JSON: {0}")]
    BodyNotJson(serde_json::Error),
    /// `messages` field is missing or not a JSON array.
    #[error("body has no `messages` array")]
    NoMessagesArray,
}

/// Block types the live-zone dispatcher considers "in the cache hot
/// zone" even when they appear inside a live-zone message. Listed
/// explicitly (no string-prefix matching) so the cache-safety
/// surface is grep-able.
const HOT_ZONE_BLOCK_TYPES: &[&str] = &[
    "tool_use",
    "thinking",
    "redacted_thinking",
    // Anthropic compaction items — once injected they're sticky to
    // the cache as much as `tool_use` is.
    "compaction",
];

// ─── Compressor singletons ─────────────────────────────────────────────
//
// Each compressor's struct holds its config + (for SmartCrusher) the
// scoring infrastructure. Allocating one per request would be
// wasteful and (in SmartCrusher's case) defeats the purpose of the
// builder. Hold one instance per process behind `OnceLock`; cheap to
// clone the &reference each call.

fn smart_crusher() -> &'static SmartCrusher {
    static INSTANCE: OnceLock<SmartCrusher> = OnceLock::new();
    INSTANCE.get_or_init(|| SmartCrusher::new(SmartCrusherConfig::default()))
}

fn log_compressor() -> &'static LogCompressor {
    static INSTANCE: OnceLock<LogCompressor> = OnceLock::new();
    INSTANCE.get_or_init(|| LogCompressor::new(LogCompressorConfig::default()))
}

fn search_compressor() -> &'static SearchCompressor {
    static INSTANCE: OnceLock<SearchCompressor> = OnceLock::new();
    INSTANCE.get_or_init(|| SearchCompressor::new(SearchCompressorConfig::default()))
}

fn diff_compressor() -> &'static DiffCompressor {
    static INSTANCE: OnceLock<DiffCompressor> = OnceLock::new();
    INSTANCE.get_or_init(|| DiffCompressor::new(DiffCompressorConfig::default()))
}

// ─── Public entry point ────────────────────────────────────────────────

/// Inspect a buffered Anthropic `/v1/messages` body and decide which
/// blocks (if any) to rewrite.
///
/// # Provider scope (Phase B)
///
/// This function only handles the Anthropic Messages API shape:
///
/// - `messages: [{role, content}]`, with `content` either a JSON
///   string or an array of typed blocks (`text`, `tool_result`,
///   `tool_use`, `thinking`, `image`, …).
/// - The "live zone" is the latest `role == "user"` message at or
///   above `frozen_message_count`. Earlier messages are in the
///   prompt cache hot zone and are byte-preserved.
///
/// **Other providers need their own dispatchers** because their
/// request shapes diverge:
///
/// - **OpenAI Chat Completions** (`/v1/chat/completions`) — tool
///   results live in their own `role: "tool"` messages, not nested
///   in user messages. The live zone is the trailing run of
///   `tool` messages plus the latest `user` message.
/// - **OpenAI Responses API** (`/v1/responses`) — the request is
///   keyed under `input` (not `messages`) with item types like
///   `function_call_output` and `reasoning`; live zone is the
///   trailing function-call-output items since the last `message`
///   or `reasoning` item.
/// - **Google Gemini** (`/v1beta/.../:generateContent`) — request
///   is keyed under `contents` (not `messages`), with
///   `function_response` parts (not `tool_result`). Function
///   responses can be either string or structured object.
/// - **Bedrock InvokeModel** — the embedded payload follows the
///   model's native format (Anthropic, Llama, Cohere, …); route
///   to the matching dispatcher.
///
/// Phase C (`REALIGNMENT/05-phase-C-rust-proxy.md`) introduces the
/// per-provider dispatchers. Each will live as
/// `compress_<provider>_live_zone` and share the cache-safety
/// invariants and the per-content-type compressor backend
/// (SmartCrusher / LogCompressor / SearchCompressor /
/// DiffCompressor / Code) from this module. The
/// [`LiveZoneOutcome`], [`BlockAction`], and
/// [`CompressionManifest`] types are intentionally
/// provider-agnostic so the per-provider dispatchers can return
/// them unchanged.
///
/// # Arguments
///
/// - `body_raw`: the buffered request body as bytes. Must be valid
///   UTF-8 JSON; non-JSON returns [`LiveZoneError::BodyNotJson`].
/// - `frozen_message_count`: hot-zone floor. Indices `< floor` are
///   excluded from dispatch.
/// - `_auth_mode`: reserved for PR-F2; B3 ignores it.
/// - `model`: the upstream model name (e.g. `"claude-3-5-sonnet-20241022"`).
///   Routes the tokenizer registry to the right backend for the
///   per-block token-count check (PR-B4). Pass [`DEFAULT_MODEL`] when
///   the proxy could not extract `body["model"]`.
///
/// # Returns
///
/// - [`LiveZoneOutcome::NoChange`] when no block was rewritten
///   (either nothing was eligible, or every compressor declined /
///   failed / produced larger output).
/// - [`LiveZoneOutcome::Modified`] when at least one block was
///   rewritten — the proxy forwards the new body.
pub fn compress_anthropic_live_zone(
    body_raw: &[u8],
    frozen_message_count: usize,
    auth_mode: AuthMode,
    model: &str,
) -> Result<LiveZoneOutcome, LiveZoneError> {
    compress_anthropic_live_zone_with_ccr(body_raw, frozen_message_count, auth_mode, model, None)
}

/// Same as [`compress_anthropic_live_zone`] but with an optional
/// [`CcrStore`] for retrieval-marker injection (PR-B7).
///
/// When `ccr_store` is `Some(_)` and a compressor produces a strictly
/// smaller block, the dispatcher:
///
/// 1. Computes `hash = compute_key(original_bytes)` (BLAKE3 → 24 hex
///    chars).
/// 2. Stores the original block content in the backend under that hash.
/// 3. Appends the marker `<<ccr:HASH>>` to the compressed block content
///    (newline-separated) so the model can later call
///    `headroom_retrieve(hash="HASH")` to recover the original bytes.
///
/// When `ccr_store` is `None` (default for tests, default for the old
/// `compress_anthropic_live_zone` shim), the dispatcher behaves
/// identically to PR-B4 — no markers, no put.
pub fn compress_anthropic_live_zone_with_ccr(
    body_raw: &[u8],
    frozen_message_count: usize,
    _auth_mode: AuthMode,
    model: &str,
    ccr_store: Option<&dyn CcrStore>,
) -> Result<LiveZoneOutcome, LiveZoneError> {
    let parsed: Value = serde_json::from_slice(body_raw).map_err(LiveZoneError::BodyNotJson)?;
    let messages = parsed
        .get("messages")
        .and_then(Value::as_array)
        .ok_or(LiveZoneError::NoMessagesArray)?;

    if messages.is_empty() {
        return Ok(LiveZoneOutcome::NoChange {
            manifest: CompressionManifest::empty(),
        });
    }

    let messages_total = messages.len();
    let messages_below_frozen_floor = frozen_message_count.min(messages_total);

    // Latest user message index, restricted to the live zone (>= floor).
    let latest_user_message_index = find_latest_user_message_index(messages, frozen_message_count);

    let Some(target_idx) = latest_user_message_index else {
        return Ok(LiveZoneOutcome::NoChange {
            manifest: CompressionManifest {
                messages_total,
                messages_below_frozen_floor,
                latest_user_message_index: None,
                block_outcomes: Vec::new(),
            },
        });
    };

    // Resolve block ranges (byte offsets into `body_raw`) by walking
    // the body via `RawValue` borrowed slices. The Vec<Replacement>
    // produced here is the surgery plan; we do *not* mutate `body_raw`
    // while computing it.
    let plan = match plan_block_replacements(body_raw, target_idx) {
        Ok(p) => p,
        Err(_) => {
            // Body shape doesn't match what we expect (e.g. content
            // is not a string and not an array, or messages is shaped
            // unexpectedly). Treat as no-change; the proxy forwards
            // the original bytes verbatim.
            let block_outcomes =
                inspect_latest_user_blocks_value(&messages[target_idx], target_idx)
                    .unwrap_or_default();
            return Ok(LiveZoneOutcome::NoChange {
                manifest: CompressionManifest {
                    messages_total,
                    messages_below_frozen_floor,
                    latest_user_message_index: Some(target_idx),
                    block_outcomes,
                },
            });
        }
    };

    let mut block_outcomes: Vec<BlockOutcome> = Vec::with_capacity(plan.len());
    let mut replacements: Vec<Replacement> = Vec::new();
    // One tokenizer per request — `get_tokenizer` is cheap (it
    // returns a `Box<dyn Tokenizer>` over either a tiktoken-rs handle
    // or an estimator) but counting once per block is a hot path.
    // PR-B4 only invokes the tokenizer on blocks that actually
    // produced compressed output; the byte-threshold gate filters
    // sub-threshold content first.
    let tokenizer = get_tokenizer(model);

    for slot in plan {
        let outcome = match slot.kind {
            SlotKind::HotZone(block_type) => BlockOutcome {
                message_index: target_idx,
                block_index: Some(slot.block_index),
                block_type,
                action: BlockAction::Excluded {
                    reason: ExclusionReason::HotZoneBlockType,
                },
            },
            SlotKind::Compressible {
                block_type,
                content_text,
                content_byte_range,
            } => {
                let detected = detect_content_type(&content_text);
                let outcome: BlockOutcome = compress_one_block(
                    &content_text,
                    detected.content_type,
                    content_byte_range,
                    target_idx,
                    Some(slot.block_index),
                    block_type,
                    tokenizer.as_ref(),
                    &mut replacements,
                    ccr_store,
                );
                outcome
            }
            SlotKind::StringContent {
                content_text,
                content_byte_range,
            } => {
                let detected = detect_content_type(&content_text);
                compress_one_block(
                    &content_text,
                    detected.content_type,
                    content_byte_range,
                    target_idx,
                    None,
                    "string_content".to_string(),
                    tokenizer.as_ref(),
                    &mut replacements,
                    ccr_store,
                )
            }
        };
        block_outcomes.push(outcome);
    }

    let manifest = CompressionManifest {
        messages_total,
        messages_below_frozen_floor,
        latest_user_message_index: Some(target_idx),
        block_outcomes,
    };

    if !manifest.has_compressed_block() || replacements.is_empty() {
        return Ok(LiveZoneOutcome::NoChange { manifest });
    }

    // Build the new body via byte-range surgery. Replacements are
    // produced in ascending block order; sort defensively.
    let new_bytes = apply_replacements(body_raw, &mut replacements);

    // The output is always still valid JSON: every replacement is a
    // JSON string slot replaced by another JSON string slot. We could
    // round-trip-verify with `serde_json::from_slice` and bail out to
    // NoChange on failure, but that doubles parse cost on the hot
    // path. Rely on type discipline; the byte_fidelity test in
    // `live_zone_dispatch.rs` pins correctness.
    let new_body_str = match std::str::from_utf8(&new_bytes) {
        Ok(s) => s,
        Err(_) => {
            // Should be impossible: input was valid JSON (UTF-8) and
            // every replacement was a JSON-encoded string (also UTF-8).
            // Fall back rather than risk shipping malformed bytes.
            return Ok(LiveZoneOutcome::NoChange { manifest });
        }
    };
    let raw = match RawValue::from_string(new_body_str.to_string()) {
        Ok(r) => r,
        Err(_) => {
            // Same defensive bail-out; should not happen.
            return Ok(LiveZoneOutcome::NoChange { manifest });
        }
    };

    Ok(LiveZoneOutcome::Modified {
        new_body: raw,
        manifest,
    })
}

// ─── Internal helpers ──────────────────────────────────────────────────

/// Per-block dispatch shared by the array-of-blocks slot and the
/// legacy string-content slot. Encapsulates the PR-B4 sequence:
///
/// 1. Per-content-type byte threshold — sub-threshold content is
///    tagged `BelowByteThreshold` and the dispatcher does not even
///    invoke a compressor.
/// 2. Type-aware dispatch (`dispatch_compressor`).
/// 3. Tokenizer-validated rejection — if `compressed_tokens >=
///    original_tokens` keep the original and tag `RejectedNotSmaller`
///    (note: tokens, not bytes, drive the gate).
/// 4. Otherwise record the replacement and tag `Compressed`.
#[allow(clippy::too_many_arguments)]
fn compress_one_block(
    content_text: &str,
    content_type: ContentType,
    content_byte_range: (usize, usize),
    message_index: usize,
    block_index: Option<usize>,
    block_type: String,
    tokenizer: &dyn crate::tokenizer::Tokenizer,
    replacements: &mut Vec<Replacement>,
    ccr_store: Option<&dyn CcrStore>,
) -> BlockOutcome {
    // 1. Byte-threshold gate. Empty content always falls through to
    //    `dispatch_compressor` (which short-circuits on empty), so
    //    only check when the slot has real bytes — this preserves
    //    the existing "tool_result with no inner content" pathway.
    if !content_text.is_empty() && content_text.len() < threshold_for(content_type) {
        return BlockOutcome {
            message_index,
            block_index,
            block_type,
            action: BlockAction::BelowByteThreshold {
                content_type: content_type.as_str(),
                byte_count: content_text.len(),
                threshold_bytes: threshold_for(content_type),
            },
        };
    }

    match dispatch_compressor(content_text, content_type) {
        DispatchResult::NoOp { content_type } => BlockOutcome {
            message_index,
            block_index,
            block_type,
            action: BlockAction::NoCompressionApplied {
                content_type: content_type.to_string(),
            },
        },
        DispatchResult::Compressed {
            strategy,
            compressed,
        } => {
            let original_bytes = content_text.len();
            // PR-B7: when a CCR store is wired, persist the original
            // block content keyed by `BLAKE3(original)[..24]` and append
            // the `<<ccr:HASH>>` marker to the compressed string. The
            // marker stays on a fresh trailing line so it is easy for
            // the model to spot and so that the per-content-type
            // compressors (which already produce trailing summary
            // lines) keep their final newline before the marker.
            //
            // The token-validation gate (step 3) is computed against
            // the marker-augmented string so the saved-token check
            // stays honest — the marker costs ~6 tokens and we'd
            // rather forward the original than ship a bigger payload
            // for a 5-byte block.
            let (compressed_for_replacement, ccr_hash_emitted) =
                maybe_inject_ccr_marker(content_text, &compressed, ccr_store);
            let compressed_bytes = compressed_for_replacement.len();
            // 3. Tokenizer-validated rejection. Per PR-B4 spec we
            //    count both the original and compressed strings
            //    using the model's tokenizer; the compression is
            //    accepted only when it shrinks the token count.
            //    Bytes-shrinking-but-tokens-growing happens for
            //    pathological inputs (e.g. dense base64 → tokenizer
            //    fragments more aggressively after a transform).
            let original_tokens = tokenizer.count_text(content_text);
            let compressed_tokens = tokenizer.count_text(&compressed_for_replacement);
            if compressed_tokens >= original_tokens {
                BlockOutcome {
                    message_index,
                    block_index,
                    block_type,
                    action: BlockAction::RejectedNotSmaller {
                        strategy,
                        original_bytes,
                        compressed_bytes,
                        original_tokens,
                        compressed_tokens,
                    },
                }
            } else {
                // Only persist to the CCR store once the rejection
                // gate has admitted the compression — otherwise we
                // populate the store with hashes whose markers
                // never reach the wire (still correct, but wastes
                // storage capacity).
                if let (Some(store), Some(hash)) = (ccr_store, ccr_hash_emitted.as_deref()) {
                    store.put(hash, content_text);
                }
                let replacement_bytes = serde_json::to_vec(&compressed_for_replacement)
                    .expect("string is always JSON-encodable");
                replacements.push(Replacement {
                    range: content_byte_range,
                    replacement: replacement_bytes,
                });
                BlockOutcome {
                    message_index,
                    block_index,
                    block_type,
                    action: BlockAction::Compressed {
                        strategy,
                        original_bytes,
                        compressed_bytes,
                        original_tokens,
                        compressed_tokens,
                    },
                }
            }
        }
        DispatchResult::Error { strategy, error } => BlockOutcome {
            message_index,
            block_index,
            block_type,
            action: BlockAction::CompressorError { strategy, error },
        },
    }
}

/// Walk `messages` from the back, returning the index of the latest
/// `role == "user"` message. Restricted to indices `>= floor`; if
/// the latest user message lies in the cache hot zone we return
/// `None` (it's out of bounds for live-zone work).
fn find_latest_user_message_index(messages: &[Value], floor: usize) -> Option<usize> {
    let start = floor.min(messages.len());
    for (offset, msg) in messages.iter().enumerate().rev() {
        if offset < start {
            return None;
        }
        if msg.get("role").and_then(Value::as_str) == Some("user") {
            return Some(offset);
        }
    }
    None
}

/// Body-shape view used to find byte ranges.
///
/// `&'a RawValue` borrows are pointer-equal to slices into the input
/// buffer; we use this to compute exact byte offsets via the
/// `bytes_offset_of` helper. The struct intentionally only captures
/// the path we need; everything else is left unparsed.
#[derive(Deserialize)]
struct BodyView<'a> {
    #[serde(borrow)]
    messages: Vec<&'a RawValue>,
}

#[derive(Deserialize)]
struct MessageView<'a> {
    #[serde(borrow, default)]
    content: Option<&'a RawValue>,
}

#[derive(Deserialize)]
struct BlockHeader<'a> {
    #[serde(borrow, default)]
    r#type: Option<&'a str>,
    #[serde(borrow, default)]
    content: Option<&'a RawValue>,
}

/// Per-block dispatch slot the planner emits.
struct PlanSlot {
    block_index: usize,
    kind: SlotKind,
}

enum SlotKind {
    /// Content is a JSON string the dispatcher may compress in place.
    Compressible {
        block_type: String,
        content_text: String,
        content_byte_range: (usize, usize),
    },
    /// String-shaped message content (Anthropic legacy shape: the
    /// whole message's `content` is a JSON string, no per-block
    /// array).
    StringContent {
        content_text: String,
        content_byte_range: (usize, usize),
    },
    /// Block type is on the cache-hot list — record but do not
    /// dispatch.
    HotZone(String),
}

/// Walk the buffered body, return one `PlanSlot` per block in the
/// latest user message. Errors out on shapes the dispatcher does not
/// support (e.g. structured-array `content` inside a tool_result —
/// rare; we degrade to NoChange in that case).
fn plan_block_replacements(
    body_raw: &[u8],
    target_msg_idx: usize,
) -> Result<Vec<PlanSlot>, PlanError> {
    // `serde_json::from_slice` requires UTF-8; we re-validate here
    // explicitly so the pointer-arithmetic helper can take a `&str`
    // without unsafe.
    let body_str = std::str::from_utf8(body_raw).map_err(|_| PlanError::ParseFailed)?;
    let body: BodyView<'_> = serde_json::from_str(body_str).map_err(|_| PlanError::ParseFailed)?;
    let target_msg_raw = body
        .messages
        .get(target_msg_idx)
        .ok_or(PlanError::TargetOutOfBounds)?;

    let msg_view: MessageView<'_> =
        serde_json::from_str(target_msg_raw.get()).map_err(|_| PlanError::ParseFailed)?;

    let Some(content_raw) = msg_view.content else {
        return Ok(Vec::new());
    };

    // Compute the byte offset of the message's `content` value into
    // `body_raw`. The target_msg_raw points into body_raw; content_raw
    // points into target_msg_raw's bytes (which are the same backing
    // memory).
    let content_offset_in_msg =
        bytes_offset_of(target_msg_raw.get(), content_raw.get()).ok_or(PlanError::OffsetMissing)?;
    let msg_offset_in_body =
        bytes_offset_of(body_str, target_msg_raw.get()).ok_or(PlanError::OffsetMissing)?;
    let content_offset_in_body = msg_offset_in_body + content_offset_in_msg;

    let content_str = content_raw.get();

    // Case 1: content is a JSON string (Anthropic legacy shape for
    // user messages).
    if content_str.starts_with('"') {
        let unescaped: String =
            serde_json::from_str(content_str).map_err(|_| PlanError::ParseFailed)?;
        return Ok(vec![PlanSlot {
            block_index: 0,
            kind: SlotKind::StringContent {
                content_text: unescaped,
                content_byte_range: (
                    content_offset_in_body,
                    content_offset_in_body + content_str.len(),
                ),
            },
        }]);
    }

    // Case 2: content is an array of blocks. Borrow each block as a
    // &RawValue so we can compute its byte range too.
    let blocks: Vec<&RawValue> =
        serde_json::from_str(content_str).map_err(|_| PlanError::ParseFailed)?;

    let mut slots = Vec::with_capacity(blocks.len());
    for (block_idx, block_raw) in blocks.iter().enumerate() {
        let block_offset_in_content =
            bytes_offset_of(content_str, block_raw.get()).ok_or(PlanError::OffsetMissing)?;
        let block_offset_in_body = content_offset_in_body + block_offset_in_content;

        let header: BlockHeader<'_> =
            serde_json::from_str(block_raw.get()).map_err(|_| PlanError::ParseFailed)?;
        let block_type = header.r#type.unwrap_or("unknown").to_string();

        if HOT_ZONE_BLOCK_TYPES.iter().any(|t| *t == block_type) {
            slots.push(PlanSlot {
                block_index: block_idx,
                kind: SlotKind::HotZone(block_type),
            });
            continue;
        }

        // Find the inner `content` field's byte range. For tool_result
        // blocks this is the field we'd compress. For text blocks
        // it's a `text` field — we read that instead.
        let (inner_field_str, inner_field_offset_in_block) = match block_type.as_str() {
            "tool_result" => {
                let Some(field_raw) = header.content else {
                    // tool_result with no content — skip dispatch.
                    slots.push(PlanSlot {
                        block_index: block_idx,
                        kind: SlotKind::Compressible {
                            block_type,
                            content_text: String::new(),
                            content_byte_range: (block_offset_in_body, block_offset_in_body),
                        },
                    });
                    continue;
                };
                let off = bytes_offset_of(block_raw.get(), field_raw.get())
                    .ok_or(PlanError::OffsetMissing)?;
                (field_raw.get(), off)
            }
            "text" => {
                #[derive(Deserialize)]
                struct TextHeader<'a> {
                    #[serde(borrow, default)]
                    text: Option<&'a RawValue>,
                }
                let h: TextHeader<'_> =
                    serde_json::from_str(block_raw.get()).map_err(|_| PlanError::ParseFailed)?;
                let Some(text_raw) = h.text else {
                    slots.push(PlanSlot {
                        block_index: block_idx,
                        kind: SlotKind::Compressible {
                            block_type,
                            content_text: String::new(),
                            content_byte_range: (block_offset_in_body, block_offset_in_body),
                        },
                    });
                    continue;
                };
                let off = bytes_offset_of(block_raw.get(), text_raw.get())
                    .ok_or(PlanError::OffsetMissing)?;
                (text_raw.get(), off)
            }
            _ => {
                // image, document, etc. — record as compressible
                // block-type but with empty content so no compressor
                // runs.
                slots.push(PlanSlot {
                    block_index: block_idx,
                    kind: SlotKind::Compressible {
                        block_type,
                        content_text: String::new(),
                        content_byte_range: (block_offset_in_body, block_offset_in_body),
                    },
                });
                continue;
            }
        };

        // The compressors expect a plain string, not a JSON-quoted
        // string. `tool_result.content` and `text.text` are
        // either a JSON string or a structured array; we only
        // compress the string shape (B3). Structured-array shape
        // falls through to no-op.
        if !inner_field_str.starts_with('"') {
            slots.push(PlanSlot {
                block_index: block_idx,
                kind: SlotKind::Compressible {
                    block_type,
                    content_text: String::new(),
                    content_byte_range: (block_offset_in_body, block_offset_in_body),
                },
            });
            continue;
        }
        let unescaped: String =
            serde_json::from_str(inner_field_str).map_err(|_| PlanError::ParseFailed)?;

        let inner_field_start_in_body = block_offset_in_body + inner_field_offset_in_block;
        let inner_field_end_in_body = inner_field_start_in_body + inner_field_str.len();

        slots.push(PlanSlot {
            block_index: block_idx,
            kind: SlotKind::Compressible {
                block_type,
                content_text: unescaped,
                content_byte_range: (inner_field_start_in_body, inner_field_end_in_body),
            },
        });
    }

    Ok(slots)
}

#[derive(Debug)]
enum PlanError {
    /// JSON parse failure on a body-shape view we expected to succeed.
    ParseFailed,
    /// Pointer-arithmetic could not locate a sub-slice's offset.
    /// Should not happen for valid JSON; surfacing rather than
    /// silently degrading.
    OffsetMissing,
    /// Latest-user-message index points past the end of `messages`.
    /// The caller already validated this — surfacing for safety.
    TargetOutOfBounds,
}

/// Compute the byte offset of `child` within `parent` when both are
/// `&str` views into the same backing memory. Returns `None` when
/// `child` does not lie strictly inside `parent`.
///
/// We rely on this trick because `serde_json` does not expose the
/// byte offset of a `&RawValue`; the `RawValue::get()` slice points
/// into the input buffer when `from_slice` / `from_str` was used,
/// so pointer arithmetic recovers it.
fn bytes_offset_of(parent: &str, child: &str) -> Option<usize> {
    let parent_start = parent.as_ptr() as usize;
    let parent_end = parent_start + parent.len();
    let child_start = child.as_ptr() as usize;
    if child_start < parent_start || child_start + child.len() > parent_end {
        return None;
    }
    Some(child_start - parent_start)
}

/// One byte-range replacement to apply. Sorted in ascending `range.0`
/// before splicing.
struct Replacement {
    range: (usize, usize),
    replacement: Vec<u8>,
}

/// Apply all `replacements` to `original`, returning the new buffer.
/// `replacements` are sorted in-place by ascending start offset; the
/// caller may inspect them post-call (they remain valid).
fn apply_replacements(original: &[u8], replacements: &mut [Replacement]) -> Vec<u8> {
    replacements.sort_by_key(|r| r.range.0);

    // Pre-size: original_len - sum(removed) + sum(replacement_len).
    let removed: usize = replacements.iter().map(|r| r.range.1 - r.range.0).sum();
    let added: usize = replacements.iter().map(|r| r.replacement.len()).sum();
    let mut out = Vec::with_capacity(original.len().saturating_sub(removed) + added);

    let mut cursor = 0usize;
    for r in replacements.iter() {
        out.extend_from_slice(&original[cursor..r.range.0]);
        out.extend_from_slice(&r.replacement);
        cursor = r.range.1;
    }
    out.extend_from_slice(&original[cursor..]);
    out
}

/// PR-B7: append a `<<ccr:HASH>>` retrieval marker to the compressed
/// block content when a CCR store is wired. Returns the
/// (possibly-augmented) compressed string and the hash that was
/// emitted (so the caller can decide whether to put the original into
/// the store after the rejection gate). When `ccr_store` is `None`,
/// returns the input compressed string unchanged with `None`.
///
/// The marker is appended on its own line — `\n<<ccr:HASH>>` — so:
///
/// 1. The marker is unambiguously after the compressor's last byte,
///    even if that byte was a newline already (we only add one).
/// 2. Markers are easy to detect in human-readable diffs / logs.
/// 3. The Python `inject_ccr_retrieve_tool` regex in
///    `headroom/ccr/tool_injection.py` keeps working — it matches
///    `[a-f0-9]{24}` anywhere in the text.
fn maybe_inject_ccr_marker(
    original: &str,
    compressed: &str,
    ccr_store: Option<&dyn CcrStore>,
) -> (String, Option<String>) {
    if ccr_store.is_none() {
        return (compressed.to_string(), None);
    }
    let hash = compute_key(original.as_bytes());
    let marker = marker_for(&hash);
    let augmented = if compressed.ends_with('\n') {
        format!("{compressed}{marker}")
    } else {
        format!("{compressed}\n{marker}")
    };
    (augmented, Some(hash))
}

/// Per-block dispatch result — whether any compressor ran and what
/// it produced.
enum DispatchResult {
    /// No compressor was applicable for this content type.
    NoOp { content_type: &'static str },
    /// A compressor ran and produced a candidate replacement string.
    Compressed {
        strategy: &'static str,
        compressed: String,
    },
    /// A compressor ran and failed loudly. The error string is
    /// surfaced via the manifest; the proxy logs it.
    #[allow(dead_code)]
    Error {
        strategy: &'static str,
        error: String,
    },
}

/// Map `(text, content_type)` to the compressor result.
///
/// Per spec PR-B3:
///
/// - `JsonArray` (with `is_dict_array=true`) → SmartCrusher
/// - `BuildOutput` → LogCompressor
/// - `SearchResults` → SearchCompressor
/// - `GitDiff` → DiffCompressor
/// - `SourceCode` → no-op (Rust port pending; see TODO below)
/// - `PlainText` → no-op (PR-B4 wires Kompress)
/// - `Html` → no-op (no compressor)
fn dispatch_compressor(text: &str, content_type: ContentType) -> DispatchResult {
    if text.is_empty() {
        return DispatchResult::NoOp {
            content_type: content_type.as_str(),
        };
    }

    match content_type {
        ContentType::JsonArray => {
            // The detector classifies arrays-of-scalars as JsonArray
            // too (confidence 0.8). SmartCrusher's `crush` is safe to
            // call on those — it parses, finds no compressible
            // arrays, and returns the input.
            let result = smart_crusher().crush(text, EMPTY_QUERY, DEFAULT_BIAS);
            if !result.was_modified {
                return DispatchResult::NoOp {
                    content_type: content_type.as_str(),
                };
            }
            DispatchResult::Compressed {
                strategy: STRATEGY_SMART_CRUSHER,
                compressed: result.compressed,
            }
        }
        ContentType::BuildOutput => {
            let (result, _stats) = log_compressor().compress(text, DEFAULT_BIAS);
            if result.compressed == result.original {
                return DispatchResult::NoOp {
                    content_type: content_type.as_str(),
                };
            }
            DispatchResult::Compressed {
                strategy: STRATEGY_LOG_COMPRESSOR,
                compressed: result.compressed,
            }
        }
        ContentType::SearchResults => {
            let (result, _stats) = search_compressor().compress(text, EMPTY_QUERY, DEFAULT_BIAS);
            if result.compressed == result.original {
                return DispatchResult::NoOp {
                    content_type: content_type.as_str(),
                };
            }
            DispatchResult::Compressed {
                strategy: STRATEGY_SEARCH_COMPRESSOR,
                compressed: result.compressed,
            }
        }
        ContentType::GitDiff => {
            let result = diff_compressor().compress(text, EMPTY_QUERY);
            if result.compressed == text {
                return DispatchResult::NoOp {
                    content_type: content_type.as_str(),
                };
            }
            DispatchResult::Compressed {
                strategy: STRATEGY_DIFF_COMPRESSOR,
                compressed: result.compressed,
            }
        }
        // TODO(PR-B4 / Rust code-compressor port): Python has a
        // CodeAwareCompressor; the Rust port is not yet shipped. Once
        // that crate lands, `ContentType::SourceCode` routes here
        // exactly as the others above.
        ContentType::SourceCode => DispatchResult::NoOp {
            content_type: content_type.as_str(),
        },
        // TODO(PR-B4): wire Kompress (lossless prose compressor) for
        // PlainText. For now, leave untouched.
        ContentType::PlainText => DispatchResult::NoOp {
            content_type: content_type.as_str(),
        },
        // No HTML compressor on the Rust side; pages are handled by
        // upstream extractors, not the proxy.
        ContentType::Html => DispatchResult::NoOp {
            content_type: content_type.as_str(),
        },
    }
}

/// Fallback when byte-range planning fails: still record per-block
/// outcomes so observability covers the request. Mirrors PR-B2's
/// observation-only path.
fn inspect_latest_user_blocks_value(
    message: &Value,
    message_index: usize,
) -> Option<Vec<BlockOutcome>> {
    let content = message.get("content")?;

    if content.as_str().is_some() {
        return Some(vec![BlockOutcome {
            message_index,
            block_index: None,
            block_type: "string_content".to_string(),
            action: BlockAction::NoCompressionApplied {
                content_type: "text".to_string(),
            },
        }]);
    }

    let blocks = content.as_array()?;
    let mut outcomes = Vec::with_capacity(blocks.len());
    for (idx, block) in blocks.iter().enumerate() {
        let block_type = block
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string();
        let action = if HOT_ZONE_BLOCK_TYPES.iter().any(|t| *t == block_type) {
            BlockAction::Excluded {
                reason: ExclusionReason::HotZoneBlockType,
            }
        } else {
            BlockAction::NoCompressionApplied {
                content_type: "unknown".to_string(),
            }
        };
        outcomes.push(BlockOutcome {
            message_index,
            block_index: Some(idx),
            block_type,
            action,
        });
    }
    Some(outcomes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn body(value: Value) -> Vec<u8> {
        serde_json::to_vec(&value).unwrap()
    }

    fn outcome_block_actions(o: &LiveZoneOutcome) -> Vec<&BlockAction> {
        let manifest = match o {
            LiveZoneOutcome::NoChange { manifest } => manifest,
            LiveZoneOutcome::Modified { manifest, .. } => manifest,
        };
        manifest.block_outcomes.iter().map(|b| &b.action).collect()
    }

    #[test]
    fn empty_messages_yields_no_change() {
        let b = body(json!({"model": "claude", "messages": []}));
        let out = compress_anthropic_live_zone(&b, 0, AuthMode::Payg, DEFAULT_MODEL).unwrap();
        match out {
            LiveZoneOutcome::NoChange { manifest } => {
                assert_eq!(manifest.messages_total, 0);
                assert_eq!(manifest.latest_user_message_index, None);
                assert!(manifest.block_outcomes.is_empty());
            }
            _ => panic!("expected NoChange"),
        }
    }

    #[test]
    fn no_messages_field_errors() {
        let b = body(json!({"model": "claude"}));
        let err = compress_anthropic_live_zone(&b, 0, AuthMode::Payg, DEFAULT_MODEL).unwrap_err();
        assert!(matches!(err, LiveZoneError::NoMessagesArray));
    }

    #[test]
    fn invalid_json_errors() {
        let err = compress_anthropic_live_zone(b"not json", 0, AuthMode::Payg, DEFAULT_MODEL)
            .unwrap_err();
        assert!(matches!(err, LiveZoneError::BodyNotJson(_)));
    }

    #[test]
    fn dispatches_only_to_latest_user_message() {
        // Two user messages; the dispatcher must pick the second (index 2).
        let b = body(json!({
            "messages": [
                {"role": "user", "content": "first user"},
                {"role": "assistant", "content": "first asst"},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "result"},
                    {"type": "text", "text": "summarize"}
                ]},
            ]
        }));
        let out = compress_anthropic_live_zone(&b, 0, AuthMode::Payg, DEFAULT_MODEL).unwrap();
        let manifest = match &out {
            LiveZoneOutcome::NoChange { manifest } => manifest,
            LiveZoneOutcome::Modified { manifest, .. } => manifest,
        };
        assert_eq!(manifest.latest_user_message_index, Some(2));
        let block_msg_indices: Vec<usize> = manifest
            .block_outcomes
            .iter()
            .map(|b| b.message_index)
            .collect();
        assert!(
            block_msg_indices.iter().all(|i| *i == 2),
            "all block outcomes must reference the latest user message; got {block_msg_indices:?}"
        );
    }

    #[test]
    fn respects_frozen_message_count() {
        // Latest user message is at index 1; floor is 2 → live zone is empty.
        let b = body(json!({
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "user", "content": [{"type": "text", "text": "second"}]},
            ]
        }));
        let out = compress_anthropic_live_zone(&b, 2, AuthMode::Payg, DEFAULT_MODEL).unwrap();
        let manifest = match &out {
            LiveZoneOutcome::NoChange { manifest } => manifest,
            _ => panic!("expected NoChange"),
        };
        assert_eq!(manifest.latest_user_message_index, None);
        assert!(manifest.block_outcomes.is_empty());
        assert_eq!(manifest.messages_below_frozen_floor, 2);
    }

    #[test]
    fn excludes_hot_zone_block_types() {
        let b = body(json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t", "content": "x"},
                    {"type": "thinking", "thinking": "...", "signature": "sig"},
                    {"type": "text", "text": "ok"},
                ]
            }]
        }));
        let out = compress_anthropic_live_zone(&b, 0, AuthMode::Payg, DEFAULT_MODEL).unwrap();
        let actions = outcome_block_actions(&out);
        assert_eq!(actions.len(), 3);
        // tool_result with tiny content → BelowByteThreshold (1 byte < 5 KiB plain-text threshold).
        assert!(matches!(actions[0], BlockAction::BelowByteThreshold { .. }));
        assert!(matches!(
            actions[1],
            BlockAction::Excluded {
                reason: ExclusionReason::HotZoneBlockType
            }
        ));
        // text block with "ok" → BelowByteThreshold (2 bytes < 5 KiB plain-text threshold).
        assert!(matches!(actions[2], BlockAction::BelowByteThreshold { .. }));
    }

    #[test]
    fn string_content_message_records_synthetic_block() {
        let b = body(json!({
            "messages": [{"role": "user", "content": "just a string"}]
        }));
        let out = compress_anthropic_live_zone(&b, 0, AuthMode::Payg, DEFAULT_MODEL).unwrap();
        let manifest = match &out {
            LiveZoneOutcome::NoChange { manifest } => manifest,
            LiveZoneOutcome::Modified { manifest, .. } => manifest,
        };
        assert_eq!(manifest.block_outcomes.len(), 1);
        assert_eq!(manifest.block_outcomes[0].block_type, "string_content");
        // 13 bytes of plain text is well below the 5 KiB plain-text threshold.
        assert!(matches!(
            manifest.block_outcomes[0].action,
            BlockAction::BelowByteThreshold { .. }
        ));
    }

    #[test]
    fn no_user_message_in_live_zone_returns_no_blocks() {
        let b = body(json!({
            "messages": [{"role": "assistant", "content": "hi"}]
        }));
        let out = compress_anthropic_live_zone(&b, 0, AuthMode::Payg, DEFAULT_MODEL).unwrap();
        let manifest = match &out {
            LiveZoneOutcome::NoChange { manifest } => manifest,
            _ => panic!("expected NoChange"),
        };
        assert_eq!(manifest.latest_user_message_index, None);
        assert!(manifest.block_outcomes.is_empty());
    }

    #[test]
    fn auth_mode_does_not_affect_b3_outcome_for_short_input() {
        // Trivial input → every mode behaves identically.
        let b = body(json!({
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        }));
        let payg = compress_anthropic_live_zone(&b, 0, AuthMode::Payg, DEFAULT_MODEL).unwrap();
        let oauth = compress_anthropic_live_zone(&b, 0, AuthMode::OAuth, DEFAULT_MODEL).unwrap();
        let sub =
            compress_anthropic_live_zone(&b, 0, AuthMode::Subscription, DEFAULT_MODEL).unwrap();
        for o in [&payg, &oauth, &sub] {
            assert!(matches!(o, LiveZoneOutcome::NoChange { .. }));
        }
    }

    #[test]
    fn no_change_when_input_already_minimal_returns_original_semantics() {
        // tiny tool_result → detected as plain text, no-op
        // dispatch → NoChange.
        let b = body(json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t", "content": "x"},
                ]
            }]
        }));
        let out = compress_anthropic_live_zone(&b, 0, AuthMode::Payg, DEFAULT_MODEL).unwrap();
        assert!(matches!(out, LiveZoneOutcome::NoChange { .. }));
    }

    #[test]
    fn manifest_records_messages_below_floor() {
        let b = body(json!({
            "messages": [
                {"role": "user", "content": "frozen"},
                {"role": "assistant", "content": "frozen"},
                {"role": "user", "content": "live"},
            ]
        }));
        let out = compress_anthropic_live_zone(&b, 2, AuthMode::Payg, DEFAULT_MODEL).unwrap();
        let manifest = match &out {
            LiveZoneOutcome::NoChange { manifest } => manifest,
            LiveZoneOutcome::Modified { manifest, .. } => manifest,
        };
        assert_eq!(manifest.messages_total, 3);
        assert_eq!(manifest.messages_below_frozen_floor, 2);
        assert_eq!(manifest.latest_user_message_index, Some(2));
    }

    #[test]
    fn frozen_count_above_messages_clamps() {
        let b = body(json!({
            "messages": [{"role": "user", "content": "x"}]
        }));
        let out = compress_anthropic_live_zone(&b, 99, AuthMode::Payg, DEFAULT_MODEL).unwrap();
        let manifest = match &out {
            LiveZoneOutcome::NoChange { manifest } => manifest,
            _ => panic!("expected NoChange"),
        };
        assert_eq!(manifest.messages_below_frozen_floor, 1);
        assert_eq!(manifest.latest_user_message_index, None);
    }
}

// ─── OpenAI Chat Completions live-zone dispatcher (Phase C PR-C2) ────────
//
// Sibling of `compress_anthropic_live_zone`. Same compressor backend,
// same per-content-type byte thresholds, same tokenizer-validated
// rejection gate, same byte-range-surgery rewrite strategy. The
// difference is the walker: Chat Completions defines the live zone as
// the LATEST `role: "tool"` message and the LATEST `role: "user"`
// message (separately, not as a contiguous run). All earlier `tool` /
// `user` messages are part of the cache hot zone — never touched.
//
// Tool messages have shape `{role: "tool", tool_call_id, content}`
// where `content` is either a JSON string (the common case) or an
// array of content parts (rarer; only the string shape is compressible).
// User messages have shape `{role: "user", content}` where `content`
// is either a JSON string or an array of `{type: "text", text}` /
// `{type: "image_url", ...}` blocks; only the text-blocks are eligible.
//
// `n > 1` (multiple completions) is gated *outside* this function by
// the proxy handler — we keep the dispatcher pure and unaware of
// non-determinism semantics.

/// Compress live-zone blocks of an OpenAI Chat Completions request.
///
/// # Provider scope
///
/// `/v1/chat/completions` only. The body shape is:
///
/// ```json
/// { "model": "...", "messages": [ {"role": "...", "content": "..."}, ... ] }
/// ```
///
/// Live zone = the latest `tool` role message's `content` plus the
/// latest `user` role message's text content. Earlier `tool` and
/// `user` messages are frozen (cached prefix); never rewritten.
///
/// Cache-safety invariant matches the Anthropic dispatcher: bytes
/// outside the rewritten ranges are *literally copied* from the input,
/// never re-serialized. PR-C2 integration tests pin SHA-256 byte
/// equality on the prefix and suffix.
pub fn compress_openai_chat_live_zone(
    body_raw: &[u8],
    _auth_mode: AuthMode,
    model: &str,
) -> Result<LiveZoneOutcome, LiveZoneError> {
    let parsed: Value = serde_json::from_slice(body_raw).map_err(LiveZoneError::BodyNotJson)?;
    let messages = parsed
        .get("messages")
        .and_then(Value::as_array)
        .ok_or(LiveZoneError::NoMessagesArray)?;

    if messages.is_empty() {
        return Ok(LiveZoneOutcome::NoChange {
            manifest: CompressionManifest::empty(),
        });
    }

    let messages_total = messages.len();

    // Latest tool / user message indices in the live zone.
    let latest_tool_idx = find_latest_role_index(messages, "tool");
    let latest_user_idx = find_latest_role_index(messages, "user");

    // No live-zone candidates → NoChange.
    if latest_tool_idx.is_none() && latest_user_idx.is_none() {
        return Ok(LiveZoneOutcome::NoChange {
            manifest: CompressionManifest {
                messages_total,
                messages_below_frozen_floor: 0,
                latest_user_message_index: latest_user_idx,
                block_outcomes: Vec::new(),
            },
        });
    }

    // Plan replacements for both targets. Each plan returns slots for
    // its own message; we stitch them together into a single
    // replacement vec keyed by ascending byte offset (apply_replacements
    // sorts defensively too).
    let mut all_slots: Vec<(usize, OpenAiPlanSlot)> = Vec::new();
    if let Some(idx) = latest_tool_idx {
        // Body shape doesn't match what we expect → skip planning
        // for the tool message but keep going for the user message.
        if let Ok(slot) = plan_openai_tool_message(body_raw, idx) {
            all_slots.push((idx, slot));
        }
    }
    if let Some(idx) = latest_user_idx {
        if let Ok(slots) = plan_openai_user_message(body_raw, idx) {
            for s in slots {
                all_slots.push((idx, s));
            }
        }
    }

    if all_slots.is_empty() {
        return Ok(LiveZoneOutcome::NoChange {
            manifest: CompressionManifest {
                messages_total,
                messages_below_frozen_floor: 0,
                latest_user_message_index: latest_user_idx,
                block_outcomes: Vec::new(),
            },
        });
    }

    let tokenizer = get_tokenizer(model);
    let mut block_outcomes: Vec<BlockOutcome> = Vec::with_capacity(all_slots.len());
    let mut replacements: Vec<Replacement> = Vec::new();

    for (msg_idx, slot) in all_slots {
        let detected = detect_content_type(&slot.content_text);
        let outcome = compress_one_block(
            &slot.content_text,
            detected.content_type,
            slot.content_byte_range,
            msg_idx,
            slot.block_index,
            slot.block_type,
            tokenizer.as_ref(),
            &mut replacements,
            None, // PR-C2: no CCR store yet on the OpenAI path.
        );
        block_outcomes.push(outcome);
    }

    let manifest = CompressionManifest {
        messages_total,
        messages_below_frozen_floor: 0,
        latest_user_message_index: latest_user_idx,
        block_outcomes,
    };

    if !manifest.has_compressed_block() || replacements.is_empty() {
        return Ok(LiveZoneOutcome::NoChange { manifest });
    }

    let new_bytes = apply_replacements(body_raw, &mut replacements);
    let new_body_str = match std::str::from_utf8(&new_bytes) {
        Ok(s) => s,
        Err(_) => return Ok(LiveZoneOutcome::NoChange { manifest }),
    };
    let raw = match RawValue::from_string(new_body_str.to_string()) {
        Ok(r) => r,
        Err(_) => return Ok(LiveZoneOutcome::NoChange { manifest }),
    };

    Ok(LiveZoneOutcome::Modified {
        new_body: raw,
        manifest,
    })
}

/// Find the highest index of a message with `role == role`. `None` if
/// no such message exists.
fn find_latest_role_index(messages: &[Value], role: &str) -> Option<usize> {
    for (idx, msg) in messages.iter().enumerate().rev() {
        if msg.get("role").and_then(Value::as_str) == Some(role) {
            return Some(idx);
        }
    }
    None
}

/// One OpenAI live-zone plan slot. Mirrors `PlanSlot` but emits the
/// `block_index` and `block_type` shape `compress_one_block` expects.
struct OpenAiPlanSlot {
    block_index: Option<usize>,
    block_type: String,
    content_text: String,
    content_byte_range: (usize, usize),
}

/// Plan a replacement slot for the tool message at `msg_idx`. Tool
/// messages carry `content` as either a string (compressible) or an
/// array of parts (rare; not compressed in PR-C2 — falls through).
fn plan_openai_tool_message(body_raw: &[u8], msg_idx: usize) -> Result<OpenAiPlanSlot, PlanError> {
    let body_str = std::str::from_utf8(body_raw).map_err(|_| PlanError::ParseFailed)?;
    let body: BodyView<'_> = serde_json::from_str(body_str).map_err(|_| PlanError::ParseFailed)?;
    let msg_raw = body
        .messages
        .get(msg_idx)
        .ok_or(PlanError::TargetOutOfBounds)?;

    let msg_view: MessageView<'_> =
        serde_json::from_str(msg_raw.get()).map_err(|_| PlanError::ParseFailed)?;
    let content_raw = msg_view.content.ok_or(PlanError::ParseFailed)?;

    let content_offset_in_msg =
        bytes_offset_of(msg_raw.get(), content_raw.get()).ok_or(PlanError::OffsetMissing)?;
    let msg_offset_in_body =
        bytes_offset_of(body_str, msg_raw.get()).ok_or(PlanError::OffsetMissing)?;
    let content_offset_in_body = msg_offset_in_body + content_offset_in_msg;

    let content_str = content_raw.get();
    if !content_str.starts_with('"') {
        // Non-string content (array of parts). PR-C2 doesn't walk
        // these — treat as not-planned and let the dispatcher record
        // no slot. This is a planner-level skip, not a parse error.
        return Err(PlanError::ParseFailed);
    }

    let unescaped: String =
        serde_json::from_str(content_str).map_err(|_| PlanError::ParseFailed)?;

    Ok(OpenAiPlanSlot {
        block_index: None,
        block_type: "tool_content".to_string(),
        content_text: unescaped,
        content_byte_range: (
            content_offset_in_body,
            content_offset_in_body + content_str.len(),
        ),
    })
}

/// Plan replacement slots for the user message at `msg_idx`. User
/// content can be:
///
/// - A JSON string → compressible as a single slot.
/// - An array of parts where each `{type: "text", text}` is a
///   compressible slot. `{type: "image_url", ...}` and other
///   non-text parts are skipped.
fn plan_openai_user_message(
    body_raw: &[u8],
    msg_idx: usize,
) -> Result<Vec<OpenAiPlanSlot>, PlanError> {
    let body_str = std::str::from_utf8(body_raw).map_err(|_| PlanError::ParseFailed)?;
    let body: BodyView<'_> = serde_json::from_str(body_str).map_err(|_| PlanError::ParseFailed)?;
    let msg_raw = body
        .messages
        .get(msg_idx)
        .ok_or(PlanError::TargetOutOfBounds)?;

    let msg_view: MessageView<'_> =
        serde_json::from_str(msg_raw.get()).map_err(|_| PlanError::ParseFailed)?;
    let Some(content_raw) = msg_view.content else {
        return Ok(Vec::new());
    };

    let content_offset_in_msg =
        bytes_offset_of(msg_raw.get(), content_raw.get()).ok_or(PlanError::OffsetMissing)?;
    let msg_offset_in_body =
        bytes_offset_of(body_str, msg_raw.get()).ok_or(PlanError::OffsetMissing)?;
    let content_offset_in_body = msg_offset_in_body + content_offset_in_msg;

    let content_str = content_raw.get();

    // Case 1: content is a JSON string.
    if content_str.starts_with('"') {
        let unescaped: String =
            serde_json::from_str(content_str).map_err(|_| PlanError::ParseFailed)?;
        return Ok(vec![OpenAiPlanSlot {
            block_index: None,
            block_type: "user_string".to_string(),
            content_text: unescaped,
            content_byte_range: (
                content_offset_in_body,
                content_offset_in_body + content_str.len(),
            ),
        }]);
    }

    // Case 2: content is an array of typed parts.
    let parts: Vec<&RawValue> =
        serde_json::from_str(content_str).map_err(|_| PlanError::ParseFailed)?;

    let mut slots = Vec::with_capacity(parts.len());
    for (part_idx, part_raw) in parts.iter().enumerate() {
        let header: BlockHeader<'_> =
            serde_json::from_str(part_raw.get()).map_err(|_| PlanError::ParseFailed)?;
        let block_type = header.r#type.unwrap_or("unknown").to_string();
        if block_type != "text" {
            // Skip image_url / other non-text parts.
            continue;
        }

        // Extract the `text` field byte range.
        #[derive(Deserialize)]
        struct TextHeader<'a> {
            #[serde(borrow, default)]
            text: Option<&'a RawValue>,
        }
        let h: TextHeader<'_> =
            serde_json::from_str(part_raw.get()).map_err(|_| PlanError::ParseFailed)?;
        let Some(text_raw) = h.text else {
            continue;
        };

        let part_offset_in_content =
            bytes_offset_of(content_str, part_raw.get()).ok_or(PlanError::OffsetMissing)?;
        let part_offset_in_body = content_offset_in_body + part_offset_in_content;
        let text_offset_in_part =
            bytes_offset_of(part_raw.get(), text_raw.get()).ok_or(PlanError::OffsetMissing)?;

        let text_str = text_raw.get();
        if !text_str.starts_with('"') {
            continue;
        }
        let unescaped: String =
            serde_json::from_str(text_str).map_err(|_| PlanError::ParseFailed)?;

        let text_start_in_body = part_offset_in_body + text_offset_in_part;
        let text_end_in_body = text_start_in_body + text_str.len();

        slots.push(OpenAiPlanSlot {
            block_index: Some(part_idx),
            block_type: "user_text".to_string(),
            content_text: unescaped,
            content_byte_range: (text_start_in_body, text_end_in_body),
        });
    }

    Ok(slots)
}

#[cfg(test)]
mod openai_chat_tests {
    use super::*;
    use serde_json::json;

    fn body(value: Value) -> Vec<u8> {
        serde_json::to_vec(&value).unwrap()
    }

    #[test]
    fn empty_messages_yields_no_change() {
        let b = body(json!({"model": "gpt-4o", "messages": []}));
        let out = compress_openai_chat_live_zone(&b, AuthMode::Payg, DEFAULT_MODEL).unwrap();
        assert!(matches!(out, LiveZoneOutcome::NoChange { .. }));
    }

    #[test]
    fn no_messages_field_errors() {
        let b = body(json!({"model": "gpt-4o"}));
        let err = compress_openai_chat_live_zone(&b, AuthMode::Payg, DEFAULT_MODEL).unwrap_err();
        assert!(matches!(err, LiveZoneError::NoMessagesArray));
    }

    #[test]
    fn invalid_json_errors() {
        let err =
            compress_openai_chat_live_zone(b"not json", AuthMode::Payg, DEFAULT_MODEL).unwrap_err();
        assert!(matches!(err, LiveZoneError::BodyNotJson(_)));
    }

    #[test]
    fn no_user_or_tool_yields_no_change() {
        let b = body(json!({
            "messages": [{"role": "system", "content": "you are helpful"}]
        }));
        let out = compress_openai_chat_live_zone(&b, AuthMode::Payg, "gpt-4o").unwrap();
        assert!(matches!(out, LiveZoneOutcome::NoChange { .. }));
    }

    #[test]
    fn tiny_tool_content_below_threshold_no_change() {
        let b = body(json!({
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "doing tool"},
                {"role": "tool", "tool_call_id": "t1", "content": "ok"},
            ]
        }));
        let out = compress_openai_chat_live_zone(&b, AuthMode::Payg, "gpt-4o").unwrap();
        match &out {
            LiveZoneOutcome::NoChange { manifest } => {
                // Both latest tool (idx 2) and latest user (idx 0)
                // contributed a slot; both below threshold.
                assert!(manifest
                    .block_outcomes
                    .iter()
                    .all(|b| matches!(b.action, BlockAction::BelowByteThreshold { .. })));
            }
            _ => panic!("expected NoChange"),
        }
    }

    #[test]
    fn user_array_text_parts_planned() {
        // User content as array of {type: text} + {type: image_url}.
        // Only the text part is planned.
        let b = body(json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ]
            }]
        }));
        let out = compress_openai_chat_live_zone(&b, AuthMode::Payg, "gpt-4o").unwrap();
        match &out {
            LiveZoneOutcome::NoChange { manifest } => {
                assert_eq!(manifest.block_outcomes.len(), 1);
                assert_eq!(manifest.block_outcomes[0].block_type, "user_text");
            }
            _ => panic!("expected NoChange"),
        }
    }

    #[test]
    fn picks_latest_tool_only() {
        // Two tool messages; only the latest is in the live zone.
        let b = body(json!({
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "tool", "tool_call_id": "t1", "content": "early"},
                {"role": "user", "content": "again"},
                {"role": "tool", "tool_call_id": "t2", "content": "late"},
            ]
        }));
        let out = compress_openai_chat_live_zone(&b, AuthMode::Payg, "gpt-4o").unwrap();
        let manifest = match &out {
            LiveZoneOutcome::NoChange { manifest } => manifest,
            LiveZoneOutcome::Modified { manifest, .. } => manifest,
        };
        // Tool block should reference message index 3 (latest tool),
        // user block index 2 (latest user).
        let tool_block = manifest
            .block_outcomes
            .iter()
            .find(|b| b.block_type == "tool_content")
            .expect("tool block recorded");
        assert_eq!(tool_block.message_index, 3);
        let user_block = manifest
            .block_outcomes
            .iter()
            .find(|b| b.block_type == "user_string")
            .expect("user block recorded");
        assert_eq!(user_block.message_index, 2);
    }
}

// ─── OpenAI Responses live-zone dispatcher (Phase C PR-C3) ────────────
//
// Sibling of `compress_openai_chat_live_zone`. The Responses API
// (`/v1/responses`) keys the request under `input` rather than
// `messages`, and the array carries explicitly-typed items (not
// role-tagged messages).
//
// Live zone, per spec PR-C3 (`REALIGNMENT/05-phase-C-rust-proxy.md`):
//
//   - latest `function_call_output.output`
//   - latest `local_shell_call_output.output`
//   - latest `apply_patch_call_output.output`
//   - latest `message` (text content) OR `user`-role message
//
// Earlier `*_output` items are FROZEN (cached prefix) — never touched.
// All other item types (`reasoning`, `compaction`, `mcp_*`,
// `computer_*`, `web_search_call`, `file_search_call`,
// `code_interpreter_call`, `image_generation_call`, `tool_search_call`,
// `custom_tool_call`, `function_call`, `local_shell_call`,
// `apply_patch_call`, future-unknown) are passthrough — the dispatcher
// records a `NoCompressionApplied` outcome but never plans a
// replacement.
//
// Output items must additionally clear a 2 KiB minimum (per spec line
// 167) before the per-content-type byte threshold even runs.

/// Output-item floor below which the Responses dispatcher does not
/// even attempt compression. Per spec PR-C3 §scope. Matches
/// `responses_items::OUTPUT_ITEM_MIN_BYTES`; pinned here too because
/// `headroom-core` is independent of the proxy crate.
const RESPONSES_OUTPUT_MIN_BYTES: usize = 2 * 1024;

/// Compress live-zone blocks of an OpenAI Responses request.
///
/// # Provider scope
///
/// `/v1/responses` only. The body shape is:
///
/// ```json
/// {
///   "model": "...",
///   "input": [
///     {"type": "message", "role": "user", "content": "..."},
///     {"type": "function_call", "call_id": "c1", "name": "...", "arguments": "..."},
///     {"type": "function_call_output", "call_id": "c1", "output": "..."},
///     {"type": "local_shell_call", ...},
///     {"type": "apply_patch_call", "operation": {...}},
///     ...
///   ]
/// }
/// ```
///
/// Live zone = the latest item of each compressible kind:
/// `function_call_output`, `local_shell_call_output`,
/// `apply_patch_call_output`, plus the latest `message` text. Earlier
/// items of those kinds are frozen (cached prefix); never rewritten.
/// All other item types pass through verbatim.
///
/// Cache-safety invariant matches the Anthropic / Chat dispatchers:
/// bytes outside the rewritten ranges are *literally copied* from the
/// input, never re-serialized.
pub fn compress_openai_responses_live_zone(
    body_raw: &[u8],
    _auth_mode: AuthMode,
    model: &str,
) -> Result<LiveZoneOutcome, LiveZoneError> {
    let parsed: Value = serde_json::from_slice(body_raw).map_err(LiveZoneError::BodyNotJson)?;

    // Responses uses `input`. We accept both `input` and `messages`
    // for forward-compat (some clients alias) — but `input` is the
    // canonical name. If neither field is present, surface
    // `NoMessagesArray` so the proxy can passthrough with a named
    // reason.
    let items = parsed
        .get("input")
        .or_else(|| parsed.get("messages"))
        .and_then(Value::as_array)
        .ok_or(LiveZoneError::NoMessagesArray)?;

    if items.is_empty() {
        return Ok(LiveZoneOutcome::NoChange {
            manifest: CompressionManifest::empty(),
        });
    }

    let items_total = items.len();

    // Walk items from the back, tagging the first occurrence of each
    // compressible kind. This naturally yields "latest" semantics.
    let mut latest_function_output: Option<usize> = None;
    let mut latest_local_shell_output: Option<usize> = None;
    let mut latest_apply_patch_output: Option<usize> = None;
    let mut latest_message: Option<usize> = None;

    for (idx, item) in items.iter().enumerate().rev() {
        let type_tag = item.get("type").and_then(Value::as_str).unwrap_or("");
        match type_tag {
            "function_call_output" if latest_function_output.is_none() => {
                latest_function_output = Some(idx);
            }
            "local_shell_call_output" if latest_local_shell_output.is_none() => {
                latest_local_shell_output = Some(idx);
            }
            "apply_patch_call_output" if latest_apply_patch_output.is_none() => {
                latest_apply_patch_output = Some(idx);
            }
            // Only consider user-role messages for compression.
            // Assistant messages are part of the cache hot zone
            // (next-turn continuation context).
            "message"
                if latest_message.is_none()
                    && item.get("role").and_then(Value::as_str) == Some("user") =>
            {
                latest_message = Some(idx);
            }
            _ => {}
        }
        // Early-exit if we've found everything we care about.
        if latest_function_output.is_some()
            && latest_local_shell_output.is_some()
            && latest_apply_patch_output.is_some()
            && latest_message.is_some()
        {
            break;
        }
    }

    let candidates: &[(Option<usize>, &str)] = &[
        (latest_function_output, "function_call_output"),
        (latest_local_shell_output, "local_shell_call_output"),
        (latest_apply_patch_output, "apply_patch_call_output"),
        (latest_message, "message"),
    ];

    if candidates.iter().all(|(idx, _)| idx.is_none()) {
        return Ok(LiveZoneOutcome::NoChange {
            manifest: CompressionManifest {
                messages_total: items_total,
                messages_below_frozen_floor: 0,
                latest_user_message_index: latest_message,
                block_outcomes: Vec::new(),
            },
        });
    }

    // Plan replacements per candidate kind. Each plan returns at most
    // one slot (output items have a single string field; messages
    // have a single text content slot).
    let mut all_slots: Vec<(usize, ResponsesPlanSlot)> = Vec::new();
    for (maybe_idx, kind_tag) in candidates {
        let Some(idx) = maybe_idx else { continue };
        match plan_responses_item(body_raw, *idx, kind_tag) {
            Ok(Some(slot)) => all_slots.push((*idx, slot)),
            Ok(None) => {}
            Err(_) => {
                // Body shape doesn't match what we expect for this
                // item — skip it but keep going for the others.
                continue;
            }
        }
    }

    if all_slots.is_empty() {
        return Ok(LiveZoneOutcome::NoChange {
            manifest: CompressionManifest {
                messages_total: items_total,
                messages_below_frozen_floor: 0,
                latest_user_message_index: latest_message,
                block_outcomes: Vec::new(),
            },
        });
    }

    let tokenizer = get_tokenizer(model);
    let mut block_outcomes: Vec<BlockOutcome> = Vec::with_capacity(all_slots.len());
    let mut replacements: Vec<Replacement> = Vec::new();

    for (msg_idx, slot) in all_slots {
        // Output items must clear the 2 KiB floor BEFORE the
        // per-content-type threshold even runs. This is on top of the
        // existing per-block byte-threshold gate.
        if slot.is_output_item && slot.content_text.len() < RESPONSES_OUTPUT_MIN_BYTES {
            block_outcomes.push(BlockOutcome {
                message_index: msg_idx,
                block_index: slot.block_index,
                block_type: slot.block_type.clone(),
                action: BlockAction::BelowByteThreshold {
                    content_type: "output_item",
                    byte_count: slot.content_text.len(),
                    threshold_bytes: RESPONSES_OUTPUT_MIN_BYTES,
                },
            });
            continue;
        }
        let detected = detect_content_type(&slot.content_text);
        let outcome = compress_one_block(
            &slot.content_text,
            detected.content_type,
            slot.content_byte_range,
            msg_idx,
            slot.block_index,
            slot.block_type,
            tokenizer.as_ref(),
            &mut replacements,
            None, // PR-C3: no CCR store on the Responses path yet.
        );
        block_outcomes.push(outcome);
    }

    let manifest = CompressionManifest {
        messages_total: items_total,
        messages_below_frozen_floor: 0,
        latest_user_message_index: latest_message,
        block_outcomes,
    };

    if !manifest.has_compressed_block() || replacements.is_empty() {
        return Ok(LiveZoneOutcome::NoChange { manifest });
    }

    let new_bytes = apply_replacements(body_raw, &mut replacements);
    let new_body_str = match std::str::from_utf8(&new_bytes) {
        Ok(s) => s,
        Err(_) => return Ok(LiveZoneOutcome::NoChange { manifest }),
    };
    let raw = match RawValue::from_string(new_body_str.to_string()) {
        Ok(r) => r,
        Err(_) => return Ok(LiveZoneOutcome::NoChange { manifest }),
    };

    Ok(LiveZoneOutcome::Modified {
        new_body: raw,
        manifest,
    })
}

/// Per-kind plan slot for the Responses dispatcher. Mirrors
/// `OpenAiPlanSlot` but tracks whether the slot is an `*_output` item
/// (so the 2 KiB floor only applies there, not to `message` text).
struct ResponsesPlanSlot {
    block_index: Option<usize>,
    block_type: String,
    content_text: String,
    content_byte_range: (usize, usize),
    /// True when the slot is one of `function_call_output`,
    /// `local_shell_call_output`, `apply_patch_call_output`. Used to
    /// gate the 2 KiB output-item floor.
    is_output_item: bool,
}

/// Body view for the Responses request; accepts both `input` (canonical)
/// and `messages` (alias).
#[derive(Deserialize)]
struct ResponsesBodyView<'a> {
    #[serde(borrow, default)]
    input: Option<Vec<&'a RawValue>>,
    #[serde(borrow, default)]
    messages: Option<Vec<&'a RawValue>>,
}

impl<'a> ResponsesBodyView<'a> {
    fn items(&self) -> Option<&Vec<&'a RawValue>> {
        self.input.as_ref().or(self.messages.as_ref())
    }
}

#[derive(Deserialize)]
struct OutputItemView<'a> {
    #[serde(borrow, default)]
    output: Option<&'a RawValue>,
}

#[derive(Deserialize)]
struct MessageItemView<'a> {
    #[serde(borrow, default)]
    content: Option<&'a RawValue>,
}

/// Plan a single replacement slot for a Responses item at index
/// `item_idx`. Returns `Ok(None)` when the item exists but has no
/// compressible payload (e.g. message with array content where every
/// part is non-text).
fn plan_responses_item(
    body_raw: &[u8],
    item_idx: usize,
    kind_tag: &str,
) -> Result<Option<ResponsesPlanSlot>, PlanError> {
    let body_str = std::str::from_utf8(body_raw).map_err(|_| PlanError::ParseFailed)?;
    let body: ResponsesBodyView<'_> =
        serde_json::from_str(body_str).map_err(|_| PlanError::ParseFailed)?;
    let items = body.items().ok_or(PlanError::ParseFailed)?;
    let item_raw = items.get(item_idx).ok_or(PlanError::TargetOutOfBounds)?;
    let item_offset_in_body =
        bytes_offset_of(body_str, item_raw.get()).ok_or(PlanError::OffsetMissing)?;

    match kind_tag {
        "function_call_output" | "local_shell_call_output" | "apply_patch_call_output" => {
            let view: OutputItemView<'_> =
                serde_json::from_str(item_raw.get()).map_err(|_| PlanError::ParseFailed)?;
            let Some(output_raw) = view.output else {
                return Ok(None);
            };
            let output_offset_in_item = bytes_offset_of(item_raw.get(), output_raw.get())
                .ok_or(PlanError::OffsetMissing)?;
            let output_offset_in_body = item_offset_in_body + output_offset_in_item;
            let output_str = output_raw.get();
            // `output` must be a JSON string for compression to apply.
            // Nested-object `output` (rare) falls through.
            if !output_str.starts_with('"') {
                return Ok(None);
            }
            let unescaped: String =
                serde_json::from_str(output_str).map_err(|_| PlanError::ParseFailed)?;
            Ok(Some(ResponsesPlanSlot {
                block_index: None,
                block_type: kind_tag.to_string(),
                content_text: unescaped,
                content_byte_range: (
                    output_offset_in_body,
                    output_offset_in_body + output_str.len(),
                ),
                is_output_item: true,
            }))
        }
        "message" => {
            let view: MessageItemView<'_> =
                serde_json::from_str(item_raw.get()).map_err(|_| PlanError::ParseFailed)?;
            let Some(content_raw) = view.content else {
                return Ok(None);
            };
            let content_offset_in_item = bytes_offset_of(item_raw.get(), content_raw.get())
                .ok_or(PlanError::OffsetMissing)?;
            let content_offset_in_body = item_offset_in_body + content_offset_in_item;
            let content_str = content_raw.get();

            // Case A: stringly-typed content.
            if content_str.starts_with('"') {
                let unescaped: String =
                    serde_json::from_str(content_str).map_err(|_| PlanError::ParseFailed)?;
                return Ok(Some(ResponsesPlanSlot {
                    block_index: None,
                    block_type: "message_string".to_string(),
                    content_text: unescaped,
                    content_byte_range: (
                        content_offset_in_body,
                        content_offset_in_body + content_str.len(),
                    ),
                    is_output_item: false,
                }));
            }

            // Case B: array of typed content parts. The Responses
            // spec uses `{type: "input_text", text: "..."}` and
            // `{type: "output_text", text: "..."}`. Both are
            // compressible. Anything else (image, file, etc.) is
            // skipped.
            let parts: Vec<&RawValue> =
                serde_json::from_str(content_str).map_err(|_| PlanError::ParseFailed)?;

            // Pick the first text-shaped part for compression. The
            // common Codex shape has exactly one input_text per
            // user message; the assistant final-answer shape has
            // exactly one output_text. If a future shape carries
            // multiple, we compress the first only — the rest still
            // round-trip byte-equal because we never plan a second
            // slot.
            for (part_idx, part_raw) in parts.iter().enumerate() {
                let header: BlockHeader<'_> =
                    serde_json::from_str(part_raw.get()).map_err(|_| PlanError::ParseFailed)?;
                let block_type = header.r#type.unwrap_or("unknown");
                let is_text = block_type == "input_text"
                    || block_type == "output_text"
                    || block_type == "text";
                if !is_text {
                    continue;
                }

                #[derive(Deserialize)]
                struct TextHeader<'a> {
                    #[serde(borrow, default)]
                    text: Option<&'a RawValue>,
                }
                let h: TextHeader<'_> =
                    serde_json::from_str(part_raw.get()).map_err(|_| PlanError::ParseFailed)?;
                let Some(text_raw) = h.text else { continue };

                let part_offset_in_content =
                    bytes_offset_of(content_str, part_raw.get()).ok_or(PlanError::OffsetMissing)?;
                let part_offset_in_body = content_offset_in_body + part_offset_in_content;
                let text_offset_in_part = bytes_offset_of(part_raw.get(), text_raw.get())
                    .ok_or(PlanError::OffsetMissing)?;

                let text_str = text_raw.get();
                if !text_str.starts_with('"') {
                    continue;
                }
                let unescaped: String =
                    serde_json::from_str(text_str).map_err(|_| PlanError::ParseFailed)?;

                let text_start_in_body = part_offset_in_body + text_offset_in_part;
                let text_end_in_body = text_start_in_body + text_str.len();

                return Ok(Some(ResponsesPlanSlot {
                    block_index: Some(part_idx),
                    block_type: format!("message_{block_type}"),
                    content_text: unescaped,
                    content_byte_range: (text_start_in_body, text_end_in_body),
                    is_output_item: false,
                }));
            }
            Ok(None)
        }
        _ => Ok(None),
    }
}

#[cfg(test)]
mod openai_responses_tests {
    use super::*;
    use serde_json::json;

    fn body(value: Value) -> Vec<u8> {
        serde_json::to_vec(&value).unwrap()
    }

    #[test]
    fn empty_input_yields_no_change() {
        let b = body(json!({"model": "gpt-4o", "input": []}));
        let out = compress_openai_responses_live_zone(&b, AuthMode::Payg, DEFAULT_MODEL).unwrap();
        assert!(matches!(out, LiveZoneOutcome::NoChange { .. }));
    }

    #[test]
    fn no_input_field_errors() {
        let b = body(json!({"model": "gpt-4o"}));
        let err =
            compress_openai_responses_live_zone(&b, AuthMode::Payg, DEFAULT_MODEL).unwrap_err();
        assert!(matches!(err, LiveZoneError::NoMessagesArray));
    }

    #[test]
    fn invalid_json_errors() {
        let err = compress_openai_responses_live_zone(b"not json", AuthMode::Payg, DEFAULT_MODEL)
            .unwrap_err();
        assert!(matches!(err, LiveZoneError::BodyNotJson(_)));
    }

    #[test]
    fn output_below_2kb_skipped() {
        // 1 KB output → below the output-item floor.
        let small = "x".repeat(1024);
        let b = body(json!({
            "model": "gpt-4o",
            "input": [
                {"type": "function_call_output", "call_id": "c1", "output": small}
            ]
        }));
        let out = compress_openai_responses_live_zone(&b, AuthMode::Payg, "gpt-4o").unwrap();
        match &out {
            LiveZoneOutcome::NoChange { manifest } => {
                assert_eq!(manifest.block_outcomes.len(), 1);
                match &manifest.block_outcomes[0].action {
                    BlockAction::BelowByteThreshold {
                        content_type,
                        byte_count,
                        threshold_bytes,
                    } => {
                        assert_eq!(*content_type, "output_item");
                        assert_eq!(*byte_count, 1024);
                        assert_eq!(*threshold_bytes, 2048);
                    }
                    other => panic!("expected BelowByteThreshold, got {other:?}"),
                }
            }
            _ => panic!("expected NoChange"),
        }
    }

    #[test]
    fn picks_latest_function_output_only() {
        // Two function_call_output items; only the latest is in the
        // live zone. Both small, so neither compresses, but the
        // manifest must show one slot, not two.
        let b = body(json!({
            "model": "gpt-4o",
            "input": [
                {"type": "function_call_output", "call_id": "c1", "output": "early"},
                {"type": "function_call", "call_id": "c2", "name": "f", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c2", "output": "late"},
            ]
        }));
        let out = compress_openai_responses_live_zone(&b, AuthMode::Payg, "gpt-4o").unwrap();
        let manifest = match &out {
            LiveZoneOutcome::NoChange { manifest } => manifest,
            LiveZoneOutcome::Modified { manifest, .. } => manifest,
        };
        let outputs: Vec<_> = manifest
            .block_outcomes
            .iter()
            .filter(|b| b.block_type == "function_call_output")
            .collect();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].message_index, 2);
    }

    #[test]
    fn unknown_item_types_passthrough_no_slot() {
        // Items the dispatcher doesn't compress — no replacement.
        let b = body(json!({
            "model": "gpt-4o",
            "input": [
                {"type": "reasoning", "id": "r1", "encrypted_content": "opaque"},
                {"type": "compaction", "id": "k1", "encrypted_content": "opaque"},
                {"type": "future_item_v2", "novel": true},
            ]
        }));
        let out = compress_openai_responses_live_zone(&b, AuthMode::Payg, "gpt-4o").unwrap();
        match &out {
            LiveZoneOutcome::NoChange { manifest } => {
                assert!(manifest.block_outcomes.is_empty());
            }
            _ => panic!("expected NoChange"),
        }
    }

    #[test]
    fn large_log_output_compressed() {
        // Compressible build-output style log block with repeated
        // template lines. Above 2 KB so the output floor passes;
        // LogCompressor handles BuildOutput content type.
        let mut log = String::new();
        for i in 0..400 {
            log.push_str(&format!(
                "[2024-01-01 00:00:00] INFO compile.rs:42 building module foo_{i}\n"
            ));
        }
        assert!(log.len() > 2048);
        let b = body(json!({
            "model": "gpt-4o",
            "input": [
                {"type": "local_shell_call_output", "call_id": "c1", "output": log}
            ]
        }));
        let out = compress_openai_responses_live_zone(&b, AuthMode::Payg, "gpt-4o").unwrap();
        match &out {
            LiveZoneOutcome::Modified { new_body, manifest } => {
                let new = new_body.get();
                assert!(new.len() < b.len());
                assert!(manifest
                    .block_outcomes
                    .iter()
                    .any(|b| matches!(b.action, BlockAction::Compressed { .. })));
            }
            LiveZoneOutcome::NoChange { manifest } => {
                // RejectedNotSmaller is also an acceptable outcome
                // for the test fixture; what matters is that the
                // dispatcher *attempted* the compression.
                let attempted = manifest.block_outcomes.iter().any(|b| {
                    matches!(
                        b.action,
                        BlockAction::Compressed { .. } | BlockAction::RejectedNotSmaller { .. }
                    )
                });
                assert!(
                    attempted,
                    "expected dispatcher to attempt compression on a 2KB+ log fixture: {manifest:?}"
                );
            }
        }
    }

    #[test]
    fn message_user_content_planned() {
        let b = body(json!({
            "model": "gpt-4o",
            "input": [
                {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": "describe this"}]}
            ]
        }));
        let out = compress_openai_responses_live_zone(&b, AuthMode::Payg, "gpt-4o").unwrap();
        match &out {
            LiveZoneOutcome::NoChange { manifest } => {
                assert_eq!(manifest.block_outcomes.len(), 1);
                assert_eq!(manifest.block_outcomes[0].block_type, "message_input_text");
            }
            _ => panic!("expected NoChange"),
        }
    }

    #[test]
    fn assistant_message_not_in_live_zone() {
        // Only user messages are eligible. An assistant `message`
        // item is never planned.
        let b = body(json!({
            "model": "gpt-4o",
            "input": [
                {"type": "message", "role": "assistant",
                 "content": [{"type": "output_text", "text": "answer"}]}
            ]
        }));
        let out = compress_openai_responses_live_zone(&b, AuthMode::Payg, "gpt-4o").unwrap();
        match &out {
            LiveZoneOutcome::NoChange { manifest } => {
                assert!(manifest.block_outcomes.is_empty());
            }
            _ => panic!("expected NoChange"),
        }
    }
}
