//! Anthropic `/v1/messages` request compression — live-zone
//! dispatcher entry point.
//!
//! # Provider scope
//!
//! This module is **Anthropic-only**. The proxy gates compression
//! on `path == "/v1/messages"` (see `compression::is_compressible_path`).
//! OpenAI Chat Completions, OpenAI Responses, and Google Gemini
//! each get their own sibling module in Phase C — they share
//! [`headroom_core::transforms::LiveZoneOutcome`] and the
//! per-content-type compressor backend, but the walkers are
//! provider-specific because the request shapes diverge.
//!
//! # Pipeline
//!
//! 1. Resolve `frozen_message_count` from the request body via
//!    [`crate::compression::resolve_frozen_count`] (PR-A4 helper).
//!    The proxy's `cache_control_auto_frozen` config gates whether
//!    the body is parsed at all — when disabled, floor=0 without
//!    inspection.
//! 2. Hand the buffered body bytes to
//!    [`headroom_core::transforms::compress_anthropic_live_zone`].
//!    The dispatcher inspects the live zone (latest user message),
//!    detects per-block content type, dispatches each block to the
//!    matching compressor (SmartCrusher / LogCompressor /
//!    SearchCompressor / DiffCompressor), and rewrites the body
//!    via byte-range surgery so unmodified bytes round-trip
//!    byte-equal.
//! 3. Translate [`LiveZoneOutcome::Modified`] →
//!    [`Outcome::Compressed`] (caller forwards the new body) or
//!    [`LiveZoneOutcome::NoChange`] → [`Outcome::NoCompression`]
//!    (caller forwards the original body verbatim).
//!
//! # Cache-safety invariant
//!
//! Bytes outside the rewritten block round-trip byte-equal. The
//! `byte_fidelity_outside_compressed_block` integration test in
//! `crates/headroom-core/tests/live_zone_dispatch.rs` pins the
//! SHA-256 prefix-and-suffix invariant in CI.

use bytes::Bytes;
use headroom_core::auth_mode::AuthMode as RequestAuthMode;
use headroom_core::transforms::live_zone::DEFAULT_MODEL;
use headroom_core::transforms::{
    compress_anthropic_live_zone, BlockAction, ExclusionReason, LiveZoneError, LiveZoneOutcome,
};
use serde_json::Value;

use crate::cache_stabilization::anthropic_cache_control::{
    auto_place_anthropic_cache_control, AutoPlaceOutcome, SkipReason,
};
use crate::cache_stabilization::tool_def_normalize::{
    any_tool_has_cache_control, sort_schema_keys_recursive, sort_tools_deterministically,
};
use crate::compression::resolve_frozen_count;
use crate::config::{CacheControlAutoFrozen, CompressionMode};

/// Per-strategy aggregate token counts for the
/// `proxy_compression_ratio_by_strategy` metric. One entry per
/// distinct `strategy` tag observed in the manifest's
/// `BlockAction::Compressed` blocks. `H1` remediation: the proxy
/// previously emitted the same aggregate ratio per strategy when
/// multiple strategies ran on one body — meaning Phase H per-
/// strategy dashboards read garbage. This struct surfaces the
/// genuine per-strategy values so the emit loop reports the right
/// numbers.
#[derive(Debug, Clone, Copy)]
pub struct PerStrategyTokens {
    pub strategy: &'static str,
    pub original_tokens: usize,
    pub compressed_tokens: usize,
}

/// What happened. The caller uses the variant to decide whether to
/// forward the original bytes (everything PR-B2 lands on) or a
/// modified body (PR-B3+).
#[derive(Debug)]
pub enum Outcome {
    /// Body was not compressed. Caller forwards the original
    /// buffered bytes byte-equal. Always returned in PR-B2.
    NoCompression,
    /// Reserved for PR-B3+: live-zone compression actually ran and
    /// produced a (smaller) body.
    #[allow(dead_code)]
    Compressed {
        body: Bytes,
        tokens_before: usize,
        tokens_after: usize,
        strategies_applied: Vec<&'static str>,
        markers_inserted: Vec<String>,
        /// H1 remediation: per-strategy `(before, after)` aggregate
        /// for the `proxy_compression_ratio_by_strategy` metric.
        /// Empty when the proxy compressed via a non-block path
        /// (e.g. Phase E normalization that doesn't have per-strategy
        /// token accounting) — emit-site falls back to one
        /// aggregate-labelled sample.
        per_strategy_tokens: Vec<PerStrategyTokens>,
    },
    /// Dispatcher opted out for a reason we can name.
    Passthrough { reason: PassthroughReason },
}

/// Reason the live-zone dispatcher fell through. Each variant is
/// logged at warn level by the proxy.
#[derive(Debug, Clone, Copy)]
pub enum PassthroughReason {
    /// Body was not valid JSON — never our job to fix that, but we
    /// log so operators know which requests opted out.
    NotJson,
    /// `messages` was missing or not a JSON array — the upstream
    /// API will reject with a 400 anyway; we're just bystanders.
    NoMessages,
    /// The compression-mode config is `Off`. The dispatcher is not
    /// invoked.
    ModeOff,
}

/// Live-zone compression entry point for Anthropic `/v1/messages`.
///
/// Returns one of:
///
/// - [`Outcome::NoCompression`] — proxy forwards the original
///   buffered body verbatim. PR-B2 always lands here.
/// - [`Outcome::Compressed`] — PR-B3+ produces this when at least
///   one block was rewritten.
/// - [`Outcome::Passthrough`] — invalid body shape; proxy forwards
///   the original bytes anyway.
///
/// # Arguments
///
/// - `body`: the buffered request body. Owned by the caller for the
///   lifetime of the upstream request — we only borrow.
/// - `mode`: configured compression mode. `Off` short-circuits to
///   [`Outcome::Passthrough { reason: ModeOff }`]; `LiveZone` runs
///   the dispatcher.
/// - `cache_control_policy`: gates auto-derivation of
///   `frozen_message_count` from explicit `cache_control` markers
///   in the body. Disabled → floor=0 (everything is in the live
///   zone).
/// - `auth_mode`: F1's [`RequestAuthMode`] classification of the
///   inbound request. Gates every Phase E byte-mutating pass —
///   PR-E1 (tool-array sort), PR-E2 (JSON Schema key sort), and
///   PR-E3 (`cache_control` auto-placement) — on `Payg` only.
///   OAuth and Subscription modes pass through byte-equal because
///   mutating their bytes risks looking like cache-evasion to the
///   upstream. The live-zone dispatcher itself still runs on every
///   mode in PR-B/C; the auth-mode gate is local to Phase E.
/// - `request_id`: per-request id used for log correlation.
pub fn compress_anthropic_request(
    body: &Bytes,
    mode: CompressionMode,
    cache_control_policy: CacheControlAutoFrozen,
    auth_mode: RequestAuthMode,
    request_id: &str,
) -> Outcome {
    if matches!(mode, CompressionMode::Off) {
        tracing::info!(
            request_id = %request_id,
            path = "/v1/messages",
            method = "POST",
            compression_mode = mode.as_str(),
            decision = "passthrough",
            reason = "mode_off",
            body_bytes = body.len(),
            "anthropic compression decision"
        );
        return Outcome::Passthrough {
            reason: PassthroughReason::ModeOff,
        };
    }

    // Mode is LiveZone. Resolve the cache-hot floor first; this is
    // the only place the body is parsed at all when the policy is
    // Disabled (resolve_frozen_count short-circuits).
    let mut parsed: serde_json::Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(_) => {
            tracing::warn!(
                request_id = %request_id,
                path = "/v1/messages",
                method = "POST",
                compression_mode = mode.as_str(),
                decision = "passthrough",
                reason = "not_json",
                body_bytes = body.len(),
                "anthropic compression decision"
            );
            return Outcome::Passthrough {
                reason: PassthroughReason::NotJson,
            };
        }
    };

    let frozen_count = resolve_frozen_count(&parsed, cache_control_policy, request_id);

    // ── Phase E byte-mutating passes ──────────────────────────────
    //
    // Three PAYG-gated passes run on the same parsed body, in this
    // order, before the live-zone dispatcher:
    //
    //   1. PR-E1 — sort `tools[]` alphabetically by name.
    //      Skipped if any tool carries a `cache_control` marker.
    //   2. PR-E2 — recursively sort JSON Schema object keys inside
    //      every tool's `input_schema`. No marker check (markers
    //      live on the tool object, not inside the schema, so
    //      sorting schema keys never moves the marker).
    //   3. PR-E3 — auto-place a `cache_control` marker on the
    //      (now-sorted) last tool. Skipped if any marker is already
    //      present anywhere in the body.
    //
    // Why this order: E1 must run before E3 so E3 places its marker
    // on the deterministic "last tool after sort". If E3 ran first,
    // E1 would correctly skip on `marker_present` but the marker
    // would be on a non-deterministic tool. E2 sits between E1 and
    // E3 because schema key order doesn't interact with marker
    // placement at all.
    //
    // OAuth and Subscription auth modes pass through byte-equal —
    // mutating their bytes can look like cache-evasion to the
    // upstream and trigger revocation.
    //
    // Each gate skip emits a structured `eN_skipped` event so
    // dashboards can see how often each policy fires in production.
    // Each apply emits `eN_applied` with diagnostic fields.

    // PR-E1 + PR-E2: sort tools[] and schema keys in-place on the
    // parsed value.
    let normalization_applied = normalize_tool_definitions(&mut parsed, auth_mode, request_id);

    // PR-E3: auto-place anthropic cache_control on the last tool.
    let mut e3_locations: Vec<String> = Vec::new();
    let mut e3_applied: bool = false;
    let e3_skipped: bool;
    if matches!(auth_mode, RequestAuthMode::Payg) {
        match auto_place_anthropic_cache_control(&mut parsed) {
            AutoPlaceOutcome::Applied {
                placed_count,
                locations,
            } => {
                e3_skipped = false;
                if placed_count > 0 {
                    tracing::info!(
                        event = "e3_applied",
                        request_id = %request_id,
                        path = "/v1/messages",
                        placed_count = placed_count,
                        locations = ?locations,
                        "auto-placed anthropic cache_control marker(s)"
                    );
                    e3_applied = true;
                    e3_locations = locations;
                } else {
                    // Applied with placed_count = 0 means "ran but
                    // nothing to do" (no tools array, empty array,
                    // or the last tool wasn't an object). Emit a
                    // distinct event so dashboards can spot the
                    // we-tried-but-no-target branch.
                    tracing::info!(
                        event = "e3_no_target",
                        request_id = %request_id,
                        path = "/v1/messages",
                        "auto-placement ran but found no tool slot to mark"
                    );
                }
            }
            AutoPlaceOutcome::Skipped {
                reason: SkipReason::MarkerPresent,
            } => {
                e3_skipped = true;
                tracing::info!(
                    event = "e3_skipped",
                    request_id = %request_id,
                    path = "/v1/messages",
                    reason = SkipReason::MarkerPresent.as_str(),
                    "customer-placed cache_control marker(s) present; auto-placement skipped"
                );
            }
            AutoPlaceOutcome::Skipped {
                reason: SkipReason::AuthMode,
            } => {
                // The function never returns AuthMode itself — that
                // gate lives in this caller. Defensive arm so the
                // match is exhaustive across the public enum.
                e3_skipped = true;
            }
        }
    } else {
        e3_skipped = true;
        tracing::info!(
            event = "e3_skipped",
            request_id = %request_id,
            path = "/v1/messages",
            reason = SkipReason::AuthMode.as_str(),
            auth_mode = auth_mode.as_str(),
            "non-PAYG auth mode; cache_control auto-placement skipped"
        );
    }
    // Suppress dead-code warnings on the local; we keep the variable
    // so future telemetry can surface the OAuth/Subscription pass
    // counts without re-deriving them.
    let _ = e3_skipped;

    // Re-serialize the parsed value once if any Phase E pass mutated
    // it. The live-zone dispatcher will re-parse internally — this
    // costs one extra serialize on the (rare) mutated path; on the
    // all-skipped path we don't touch the bytes at all.
    let dispatch_body: Bytes = if normalization_applied.any() || e3_applied {
        match serde_json::to_vec(&parsed) {
            Ok(v) => Bytes::from(v),
            Err(err) => {
                // We just parsed successfully; serialize failure is
                // unreachable in practice. If it ever fires, fall
                // back to the original body bytes — never poison the
                // request. Loud log so operators notice.
                tracing::error!(
                    event = "phase_e_serialize_failed",
                    request_id = %request_id,
                    path = "/v1/messages",
                    error = %err,
                    "Phase E pass(es) mutated parsed body but \
                     serialize-back failed; forwarding original bytes"
                );
                body.clone()
            }
        }
    } else {
        body.clone()
    };

    // PR-B4: extract `body["model"]` so the live-zone dispatcher can
    // route the tokenizer registry to the right backend for the
    // per-block token-count rejection gate. Anthropic
    // `/v1/messages` always carries a `model` string per the API
    // schema, but the proxy never breaks on a missing field — we
    // fall back to `DEFAULT_MODEL` (a Claude name, so the
    // chars-per-token estimator picks the calibrated 3.5 cpt
    // density) and continue.
    let model = parsed
        .get("model")
        .and_then(serde_json::Value::as_str)
        .unwrap_or(DEFAULT_MODEL);

    // Run the live-zone dispatcher. PR-B3 wires per-type compressors:
    // SmartCrusher / LogCompressor / SearchCompressor / DiffCompressor.
    // PR-B4 added the per-content-type byte-threshold gate and the
    // tokenizer-validated rejection check. The dispatcher returns
    // `Modified` whenever at least one block was rewritten and
    // `NoChange` otherwise (live zone empty, every compressor
    // declined, or every compressor produced output whose token
    // count was not strictly less than the input's).
    // F2.1 c2/6: forward F1's classified auth_mode into the dispatcher
    // instead of the hard-coded `Payg`. The dispatcher itself doesn't
    // change behaviour by mode in F2.1 (live-zone compression runs for
    // every mode — closing #327/#388 means subscription users keep
    // getting compression, not losing it). The plumbing here lets
    // F2.2 vary per-block thresholds by mode without touching this
    // call site again.
    match compress_anthropic_live_zone(&dispatch_body, frozen_count, auth_mode.into(), model) {
        Ok(LiveZoneOutcome::NoChange { manifest }) => {
            let block_count = manifest.block_outcomes.len();
            let blocks_excluded = manifest
                .block_outcomes
                .iter()
                .filter(|b| {
                    matches!(
                        b.action,
                        BlockAction::Excluded {
                            reason: ExclusionReason::HotZoneBlockType
                        }
                    )
                })
                .count();
            tracing::info!(
                request_id = %request_id,
                path = "/v1/messages",
                method = "POST",
                compression_mode = mode.as_str(),
                decision = "no_change",
                reason = "no_block_compressed",
                body_bytes = body.len(),
                frozen_message_count = frozen_count,
                messages_total = manifest.messages_total,
                latest_user_message_index = ?manifest.latest_user_message_index,
                live_zone_blocks = block_count,
                live_zone_blocks_excluded = blocks_excluded,
                "anthropic live-zone dispatch"
            );
            // The live-zone dispatcher made no change — but if any
            // Phase E pass (E1 sort, E3 cache_control auto-placement)
            // rewrote bytes, the proxy must still forward the new
            // bytes. Surface as `Compressed` with the union of
            // strategies and markers so the outer log/metrics layer
            // attributes the byte change correctly.
            if normalization_applied.any() || e3_applied {
                let mut strategies = normalization_applied.strategies();
                if e3_applied {
                    strategies.push("e3_anthropic_cache_control");
                }
                Outcome::Compressed {
                    body: dispatch_body,
                    tokens_before: 0,
                    tokens_after: 0,
                    strategies_applied: strategies,
                    markers_inserted: e3_locations,
                    // H1 remediation: Phase E normalization passes
                    // (E1 sort, E3 cache_control auto-placement)
                    // mutate bytes but don't have per-strategy
                    // token accounting. Empty vec → emit-site falls
                    // back to one aggregate sample.
                    per_strategy_tokens: Vec::new(),
                }
            } else {
                Outcome::NoCompression
            }
        }
        Ok(LiveZoneOutcome::Modified { new_body, manifest }) => {
            // Aggregate manifest into the proxy's `Compressed` payload.
            // PR-B4 reports token counts via the same tokenizer the
            // dispatcher used to gate per-block acceptance — so the
            // saving the proxy logs is the saving the cache will
            // actually see.
            //
            // H1 + C5 remediation:
            //   - Per-strategy `(before, after)` aggregate populated
            //     from the manifest so the proxy emits one
            //     `proxy_compression_ratio_by_strategy` sample with
            //     the right numbers per strategy (instead of the
            //     same aggregate ratio per strategy, which was the
            //     pre-fix behavior).
            //   - Every `BlockAction::RejectedNotSmaller` increments
            //     the `proxy_compression_rejected_by_token_check_total`
            //     counter so dashboards can attribute "compressor ran
            //     but kept the original" cases.
            let mut original_bytes_total: usize = 0;
            let mut compressed_bytes_total: usize = 0;
            let mut original_tokens_total: usize = 0;
            let mut compressed_tokens_total: usize = 0;
            let mut strategies: Vec<&'static str> = Vec::new();
            let mut per_strategy_tokens: Vec<PerStrategyTokens> = Vec::new();
            for entry in &manifest.block_outcomes {
                match entry.action {
                    BlockAction::Compressed {
                        strategy,
                        original_bytes,
                        compressed_bytes,
                        original_tokens,
                        compressed_tokens,
                    } => {
                        original_bytes_total += original_bytes;
                        compressed_bytes_total += compressed_bytes;
                        original_tokens_total += original_tokens;
                        compressed_tokens_total += compressed_tokens;
                        if !strategies.contains(&strategy) {
                            strategies.push(strategy);
                        }
                        // H1: accumulate per-strategy tokens (one
                        // entry per strategy; multiple blocks of
                        // the same strategy sum).
                        if let Some(slot) = per_strategy_tokens
                            .iter_mut()
                            .find(|s| s.strategy == strategy)
                        {
                            slot.original_tokens += original_tokens;
                            slot.compressed_tokens += compressed_tokens;
                        } else {
                            per_strategy_tokens.push(PerStrategyTokens {
                                strategy,
                                original_tokens,
                                compressed_tokens,
                            });
                        }
                    }
                    BlockAction::RejectedNotSmaller { strategy, .. } => {
                        // C5: surface the tokenizer-validated
                        // rejection in the dedicated counter.
                        crate::observability::record_compression_rejected_by_token_check(strategy);
                    }
                    _ => {}
                }
            }
            // Stitch in the PR-E1 / PR-E2 / PR-E3 strategy tags so
            // downstream log/metrics layers attribute the
            // normalization / auto-placement to its distinct
            // cache-stabilization surface rather than to a live-zone
            // compressor that didn't actually run.
            for strategy in normalization_applied.strategies() {
                if !strategies.contains(&strategy) {
                    strategies.push(strategy);
                }
            }
            if e3_applied {
                let s = "e3_anthropic_cache_control";
                if !strategies.contains(&s) {
                    strategies.push(s);
                }
            }
            let body_bytes_in = body.len();
            let new_body_bytes = Bytes::copy_from_slice(new_body.get().as_bytes());
            let body_bytes_out = new_body_bytes.len();
            let block_count = manifest.block_outcomes.len();
            tracing::info!(
                request_id = %request_id,
                path = "/v1/messages",
                method = "POST",
                compression_mode = mode.as_str(),
                decision = "compressed",
                reason = "live_zone_blocks_rewritten",
                body_bytes_in = body_bytes_in,
                body_bytes_out = body_bytes_out,
                bytes_freed = body_bytes_in.saturating_sub(body_bytes_out),
                frozen_message_count = frozen_count,
                messages_total = manifest.messages_total,
                latest_user_message_index = ?manifest.latest_user_message_index,
                live_zone_blocks = block_count,
                live_zone_strategies = ?strategies,
                live_zone_block_original_bytes = original_bytes_total,
                live_zone_block_compressed_bytes = compressed_bytes_total,
                live_zone_block_original_tokens = original_tokens_total,
                live_zone_block_compressed_tokens = compressed_tokens_total,
                model = model,
                "anthropic live-zone dispatch"
            );
            Outcome::Compressed {
                body: new_body_bytes,
                tokens_before: original_tokens_total,
                tokens_after: compressed_tokens_total,
                strategies_applied: strategies,
                // PR-E3 surfaces tool-slot location(s); PR-B7 will
                // append CCR retrieval markers when wired.
                markers_inserted: e3_locations,
                per_strategy_tokens,
            }
        }
        Err(LiveZoneError::BodyNotJson(_)) => {
            // We already parsed successfully above; the dispatcher's
            // independent parse can only fail on a state we missed.
            // Pass through with the same byte-faithful guarantee.
            tracing::warn!(
                request_id = %request_id,
                path = "/v1/messages",
                "live-zone dispatcher rejected JSON body that this layer parsed; \
                 falling back to passthrough"
            );
            Outcome::Passthrough {
                reason: PassthroughReason::NotJson,
            }
        }
        Err(LiveZoneError::NoMessagesArray) => {
            tracing::info!(
                request_id = %request_id,
                path = "/v1/messages",
                method = "POST",
                compression_mode = mode.as_str(),
                decision = "passthrough",
                reason = "no_messages",
                body_bytes = body.len(),
                "anthropic compression decision"
            );
            Outcome::Passthrough {
                reason: PassthroughReason::NoMessages,
            }
        }
    }
}

/// Tracks which Phase E normalization steps actually mutated the
/// dispatch body. Each `bool` is `true` only when the gate cleared AND
/// the operation produced a byte-different result. Used by the caller
/// to attribute strategies on the `Outcome::Compressed` payload.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct NormalizationApplied {
    pub e1_tool_sort: bool,
    pub e2_schema_sort: bool,
}

impl NormalizationApplied {
    pub(super) fn any(self) -> bool {
        self.e1_tool_sort || self.e2_schema_sort
    }

    pub(super) fn strategies(self) -> Vec<&'static str> {
        let mut out = Vec::new();
        if self.e1_tool_sort {
            out.push("tool_array_sort");
        }
        if self.e2_schema_sort {
            out.push("schema_key_sort");
        }
        out
    }
}

/// Apply PR-E1 (tool-array sort) and PR-E2 (schema-key sort) in-place
/// on the parsed body when the auth-mode + marker gates clear.
///
/// The caller owns re-serialization (because PR-E3 may also mutate
/// the same parsed value before bytes are produced). Returns a flag
/// set indicating which Phase E normalization step actually ran.
///
/// PR-E1 (sort) is additionally skipped when any tool already carries
/// a `cache_control` marker; PR-E2 still runs in that case because
/// sorting schema keys never moves the marker.
///
/// Every gate skip emits a structured `tracing::info!` event so
/// dashboards can see how often each policy fires in production.
pub(super) fn normalize_tool_definitions(
    parsed: &mut Value,
    auth_mode: RequestAuthMode,
    request_id: &str,
) -> NormalizationApplied {
    // Auth-mode gate first — both PR-E1 and PR-E2 mutate request
    // bytes, which is only safe under PAYG. OAuth and Subscription
    // clients pass through byte-equal so the proxy never looks
    // like a cache-evasion intermediary to the upstream.
    if !matches!(auth_mode, RequestAuthMode::Payg) {
        tracing::info!(
            event = "e1_skipped",
            request_id = %request_id,
            path = "/v1/messages",
            reason = "auth_mode",
            auth_mode = auth_mode.as_str(),
            "tool-array sort skipped: non-PAYG auth mode passes through byte-equal"
        );
        tracing::info!(
            event = "e2_skipped",
            request_id = %request_id,
            path = "/v1/messages",
            reason = "auth_mode",
            auth_mode = auth_mode.as_str(),
            "schema-key sort skipped: non-PAYG auth mode passes through byte-equal"
        );
        return NormalizationApplied::default();
    }

    // The body must carry a `tools` array for any normalization to
    // be possible. Missing / non-array `tools` → no work; this is
    // not a "skip" event because it is the customer's request shape,
    // not a policy gate firing.
    let Some(tools_in) = parsed.get("tools").and_then(Value::as_array) else {
        return NormalizationApplied::default();
    };
    if tools_in.is_empty() {
        return NormalizationApplied::default();
    }

    // PR-E1 marker check. Reordering tools when any tool already
    // carries `cache_control` shifts what's "before" the marker and
    // silently changes cache scope. Skip the SORT (E1); E2 still
    // runs because sorting schema keys does not move the marker
    // (which lives on the tool object itself, not inside the schema).
    let marker_present = any_tool_has_cache_control(tools_in);
    if marker_present {
        tracing::info!(
            event = "e1_skipped",
            request_id = %request_id,
            path = "/v1/messages",
            reason = "marker_present",
            tool_count = tools_in.len(),
            "tool-array sort skipped: customer cache_control marker present \
             on at least one tool; preserving customer-intentional order"
        );
    }

    let tools = parsed
        .get_mut("tools")
        .and_then(Value::as_array_mut)
        .expect("tools array verified above");

    let mut applied = NormalizationApplied::default();

    if !marker_present {
        applied.e1_tool_sort = sort_tools_deterministically(tools);
        if applied.e1_tool_sort {
            tracing::info!(
                event = "e1_applied",
                request_id = %request_id,
                path = "/v1/messages",
                tool_count = tools.len(),
                "tool-array sort applied: tools reordered alphabetically by name"
            );
        }
    }

    // PR-E2: sort each tool's `input_schema` keys recursively.
    // Anthropic schema lives at `tool.input_schema`. Tools without
    // an `input_schema` field are silently skipped — that is a
    // valid Anthropic shape (e.g. zero-argument tools).
    for tool in tools.iter_mut() {
        let Some(schema) = tool.get_mut("input_schema") else {
            continue;
        };
        // We compare bytes before / after to detect whether the
        // sort actually moved any keys (idempotent re-runs report
        // `false` and the caller surfaces no event for the no-op).
        let before = serde_json::to_vec(schema).unwrap_or_default();
        sort_schema_keys_recursive(schema);
        let after = serde_json::to_vec(schema).unwrap_or_default();
        if before != after {
            applied.e2_schema_sort = true;
        }
    }
    if applied.e2_schema_sort {
        tracing::info!(
            event = "e2_applied",
            request_id = %request_id,
            path = "/v1/messages",
            tool_count = tools.len(),
            "schema-key sort applied: input_schema keys rewritten in alphabetic order"
        );
    }

    applied
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body_of(value: serde_json::Value) -> Bytes {
        Bytes::from(serde_json::to_vec(&value).unwrap())
    }

    #[test]
    fn mode_off_short_circuits_without_parsing() {
        // Invalid JSON — would fail parse — but mode=Off must not
        // attempt to parse, and instead Passthrough{ModeOff}.
        let body = Bytes::from_static(b"not valid json");
        let out = compress_anthropic_request(
            &body,
            CompressionMode::Off,
            CacheControlAutoFrozen::Disabled,
            RequestAuthMode::Payg,
            "req-1",
        );
        match out {
            Outcome::Passthrough {
                reason: PassthroughReason::ModeOff,
            } => {}
            other => panic!("expected Passthrough{{ModeOff}}, got {other:?}"),
        }
    }

    #[test]
    fn live_zone_mode_with_no_messages_field_passthrough() {
        let body = body_of(serde_json::json!({"model": "claude"}));
        let out = compress_anthropic_request(
            &body,
            CompressionMode::LiveZone,
            CacheControlAutoFrozen::Enabled,
            RequestAuthMode::Payg,
            "req-2",
        );
        match out {
            Outcome::Passthrough {
                reason: PassthroughReason::NoMessages,
            } => {}
            other => panic!("expected Passthrough{{NoMessages}}, got {other:?}"),
        }
    }

    #[test]
    fn live_zone_mode_with_invalid_json_passthrough() {
        let body = Bytes::from_static(b"\x01\x02 not json");
        let out = compress_anthropic_request(
            &body,
            CompressionMode::LiveZone,
            CacheControlAutoFrozen::Enabled,
            RequestAuthMode::Payg,
            "req-3",
        );
        match out {
            Outcome::Passthrough {
                reason: PassthroughReason::NotJson,
            } => {}
            other => panic!("expected Passthrough{{NotJson}}, got {other:?}"),
        }
    }

    #[test]
    fn live_zone_mode_with_valid_body_returns_no_compression_pr_b2() {
        // PR-B2 invariant: every well-formed body returns NoCompression.
        let body = body_of(serde_json::json!({
            "model": "claude",
            "messages": [
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t", "content": "hello"}
                ]}
            ]
        }));
        let out = compress_anthropic_request(
            &body,
            CompressionMode::LiveZone,
            CacheControlAutoFrozen::Disabled,
            RequestAuthMode::Payg,
            "req-4",
        );
        match out {
            Outcome::NoCompression => {}
            other => panic!("expected NoCompression, got {other:?}"),
        }
    }

    #[test]
    fn empty_body_with_live_zone_mode_passthrough_not_json() {
        let body = Bytes::new();
        let out = compress_anthropic_request(
            &body,
            CompressionMode::LiveZone,
            CacheControlAutoFrozen::Enabled,
            RequestAuthMode::Payg,
            "req-5",
        );
        match out {
            Outcome::Passthrough {
                reason: PassthroughReason::NotJson,
            } => {}
            other => panic!("expected Passthrough{{NotJson}}, got {other:?}"),
        }
    }

    #[test]
    fn cache_control_disabled_yields_floor_zero() {
        // With auto-derivation Disabled, frozen floor is 0 even
        // though the body marks every message as cached. The
        // dispatcher will treat the entire array as live zone.
        // (PR-B2: still returns NoCompression — this test pins the
        // policy plumbing rather than compression behaviour.)
        //
        // The body carries a cache_control marker on a message
        // block, so PR-E3's `MarkerPresent` gate also fires —
        // result: still NoCompression (no E3 placement, no
        // live-zone change).
        let body = body_of(serde_json::json!({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}
                    ]
                }
            ]
        }));
        let out = compress_anthropic_request(
            &body,
            CompressionMode::LiveZone,
            CacheControlAutoFrozen::Disabled,
            RequestAuthMode::Payg,
            "req-6",
        );
        match out {
            Outcome::NoCompression => {}
            other => panic!("expected NoCompression, got {other:?}"),
        }
    }

    // ─── PR-E3 cache_control auto-placement: unit tests ──────────

    #[test]
    fn pr_e3_payg_with_tools_and_no_markers_returns_compressed_with_marker() {
        // PR-E3 happy path: PAYG body with one tool and no markers
        // anywhere → dispatcher inserts a marker on the last tool
        // and returns Compressed with the new bytes. With one tool,
        // E1 sort is a no-op so the only mutation is E3.
        let original = serde_json::json!({
            "model": "claude-3-5-sonnet-20241022",
            "tools": [
                {"name": "search", "description": "search the web"}
            ],
            "messages": [
                {"role": "user", "content": "hi"}
            ],
        });
        let body = body_of(original);
        let out = compress_anthropic_request(
            &body,
            CompressionMode::LiveZone,
            CacheControlAutoFrozen::Disabled,
            RequestAuthMode::Payg,
            "req-e3-1",
        );
        match out {
            Outcome::Compressed {
                body: new_body,
                strategies_applied,
                markers_inserted,
                ..
            } => {
                assert!(
                    strategies_applied.contains(&"e3_anthropic_cache_control"),
                    "expected e3_anthropic_cache_control strategy, got: {strategies_applied:?}",
                );
                assert_eq!(markers_inserted, vec!["tools[0]".to_string()]);
                let parsed: serde_json::Value =
                    serde_json::from_slice(&new_body).expect("re-parse new body");
                assert_eq!(
                    parsed.pointer("/tools/0/cache_control"),
                    Some(&serde_json::json!({"type": "ephemeral"})),
                    "marker must be present on last tool",
                );
            }
            other => panic!("expected Compressed{{e3_…}}, got {other:?}"),
        }
    }

    #[test]
    fn pr_e3_oauth_skips_auto_placement() {
        // OAuth → mutating bytes is unsafe → never auto-place. With
        // no other reason for the dispatcher to mutate, we get
        // NoCompression.
        let body = body_of(serde_json::json!({
            "model": "claude-3-5-sonnet-20241022",
            "tools": [{"name": "search", "description": "search"}],
            "messages": [{"role": "user", "content": "hi"}],
        }));
        let out = compress_anthropic_request(
            &body,
            CompressionMode::LiveZone,
            CacheControlAutoFrozen::Disabled,
            RequestAuthMode::OAuth,
            "req-e3-2",
        );
        match out {
            Outcome::NoCompression => {}
            other => panic!("expected NoCompression on OAuth, got {other:?}"),
        }
    }

    #[test]
    fn pr_e3_subscription_skips_auto_placement() {
        let body = body_of(serde_json::json!({
            "model": "claude-3-5-sonnet-20241022",
            "tools": [{"name": "search", "description": "search"}],
            "messages": [{"role": "user", "content": "hi"}],
        }));
        let out = compress_anthropic_request(
            &body,
            CompressionMode::LiveZone,
            CacheControlAutoFrozen::Disabled,
            RequestAuthMode::Subscription,
            "req-e3-3",
        );
        match out {
            Outcome::NoCompression => {}
            other => panic!("expected NoCompression on Subscription, got {other:?}"),
        }
    }

    #[test]
    fn pr_e3_payg_with_existing_marker_skips() {
        // Customer placed a marker on the only tool. Skip E3.
        let body = body_of(serde_json::json!({
            "model": "claude-3-5-sonnet-20241022",
            "tools": [
                {
                    "name": "search",
                    "description": "search",
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            "messages": [{"role": "user", "content": "hi"}],
        }));
        let out = compress_anthropic_request(
            &body,
            CompressionMode::LiveZone,
            CacheControlAutoFrozen::Disabled,
            RequestAuthMode::Payg,
            "req-e3-4",
        );
        match out {
            Outcome::NoCompression => {}
            other => panic!("expected NoCompression on customer-placed marker, got {other:?}"),
        }
    }

    #[test]
    fn pr_e3_payg_no_tools_returns_no_compression() {
        // PAYG, no markers, no tools → E3 has nothing to place.
        // No bytes mutated → NoCompression.
        let body = body_of(serde_json::json!({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hi"}],
        }));
        let out = compress_anthropic_request(
            &body,
            CompressionMode::LiveZone,
            CacheControlAutoFrozen::Disabled,
            RequestAuthMode::Payg,
            "req-e3-5",
        );
        match out {
            Outcome::NoCompression => {}
            other => panic!("expected NoCompression on no-tools PAYG body, got {other:?}"),
        }
    }

    // ─── PR-E1 tool-array sort: unit tests ────────────────────────

    #[test]
    fn e1_sorts_tools_when_payg_and_no_marker() {
        // PAYG, tools out of order, no `cache_control` marker → sort
        // should fire. Live-zone dispatcher sees the same `messages`
        // structure (no compressible blocks), so we expect
        // `Outcome::Compressed` with `tool_array_sort` strategy.
        // E3 also fires (no customer marker); after E1 sort, the
        // last tool is "zebra" → marker lands on tools[2].
        let body = body_of(serde_json::json!({
            "model": "claude",
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "hi"}
                ]}
            ],
            "tools": [
                {"name": "zebra"},
                {"name": "apple"},
                {"name": "mango"},
            ],
        }));
        let out = compress_anthropic_request(
            &body,
            CompressionMode::LiveZone,
            CacheControlAutoFrozen::Disabled,
            RequestAuthMode::Payg,
            "req-e1-1",
        );
        match out {
            Outcome::Compressed {
                body: new_body,
                strategies_applied,
                ..
            } => {
                assert!(
                    strategies_applied.contains(&"tool_array_sort"),
                    "expected tool_array_sort strategy, got: {strategies_applied:?}",
                );
                let parsed: Value = serde_json::from_slice(&new_body).unwrap();
                let tools = parsed.get("tools").and_then(Value::as_array).unwrap();
                let names: Vec<&str> = tools
                    .iter()
                    .map(|t| t.get("name").and_then(Value::as_str).unwrap())
                    .collect();
                assert_eq!(names, vec!["apple", "mango", "zebra"]);
            }
            other => panic!("expected Compressed with sort, got {other:?}"),
        }
    }

    #[test]
    fn e1_passes_through_when_oauth() {
        // Same body shape; auth_mode=OAuth → byte-equal passthrough.
        let body = body_of(serde_json::json!({
            "model": "claude",
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "hi"}
                ]}
            ],
            "tools": [
                {"name": "zebra"},
                {"name": "apple"},
            ],
        }));
        let out = compress_anthropic_request(
            &body,
            CompressionMode::LiveZone,
            CacheControlAutoFrozen::Disabled,
            RequestAuthMode::OAuth,
            "req-e1-2",
        );
        // Non-PAYG → no normalization → live-zone dispatcher sees
        // no compressible block → NoCompression.
        match out {
            Outcome::NoCompression => {}
            other => panic!("expected NoCompression for OAuth, got {other:?}"),
        }
    }

    #[test]
    fn e1_passes_through_when_subscription() {
        let body = body_of(serde_json::json!({
            "model": "claude",
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "hi"}
                ]}
            ],
            "tools": [
                {"name": "zebra"},
                {"name": "apple"},
            ],
        }));
        let out = compress_anthropic_request(
            &body,
            CompressionMode::LiveZone,
            CacheControlAutoFrozen::Disabled,
            RequestAuthMode::Subscription,
            "req-e1-3",
        );
        match out {
            Outcome::NoCompression => {}
            other => panic!("expected NoCompression for Subscription, got {other:?}"),
        }
    }

    #[test]
    fn e1_skips_when_marker_present() {
        // PAYG, but customer placed `cache_control` on a tool →
        // skip the sort, byte-equal passthrough.
        let body = body_of(serde_json::json!({
            "model": "claude",
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "hi"}
                ]}
            ],
            "tools": [
                {"name": "zebra"},
                {"name": "apple", "cache_control": {"type": "ephemeral"}},
            ],
        }));
        let out = compress_anthropic_request(
            &body,
            CompressionMode::LiveZone,
            CacheControlAutoFrozen::Disabled,
            RequestAuthMode::Payg,
            "req-e1-4",
        );
        match out {
            Outcome::NoCompression => {}
            other => panic!("expected NoCompression when marker present, got {other:?}"),
        }
    }

    #[test]
    fn e1_skips_when_no_tools_field() {
        // PAYG, no `tools` field at all → no normalization, no sort
        // event, byte-equal passthrough.
        let body = body_of(serde_json::json!({
            "model": "claude",
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "hi"}
                ]}
            ],
        }));
        let out = compress_anthropic_request(
            &body,
            CompressionMode::LiveZone,
            CacheControlAutoFrozen::Disabled,
            RequestAuthMode::Payg,
            "req-e1-5",
        );
        match out {
            Outcome::NoCompression => {}
            other => panic!("expected NoCompression with no tools, got {other:?}"),
        }
    }

    #[test]
    fn e2_sorts_input_schema_keys_when_payg() {
        // PAYG, single tool with shuffled input_schema keys → sort
        // should fire (e2 strategy).
        let body = body_of(serde_json::json!({
            "model": "claude",
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "hi"}
                ]}
            ],
            "tools": [
                {
                    "name": "search",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "filters": {"type": "object"},
                        },
                        "required": ["query"],
                    },
                },
            ],
        }));
        let out = compress_anthropic_request(
            &body,
            CompressionMode::LiveZone,
            CacheControlAutoFrozen::Disabled,
            RequestAuthMode::Payg,
            "req-e2-1",
        );
        match out {
            Outcome::Compressed {
                body: new_body,
                strategies_applied,
                ..
            } => {
                assert!(
                    strategies_applied.contains(&"schema_key_sort"),
                    "expected schema_key_sort strategy, got: {strategies_applied:?}",
                );
                let parsed: Value = serde_json::from_slice(&new_body).unwrap();
                let schema = &parsed["tools"][0]["input_schema"];
                // Inspect the top-level key sequence directly via the
                // serde_json::Map so we don't accidentally match nested
                // `"type"` occurrences in `find()` calls.
                let map = schema.as_object().unwrap();
                let keys: Vec<&str> = map.keys().map(String::as_str).collect();
                assert_eq!(
                    keys,
                    vec!["properties", "required", "type"],
                    "input_schema top-level keys must be alphabetic; got: {keys:?}"
                );
            }
            other => panic!("expected Compressed with schema sort, got {other:?}"),
        }
    }

    #[test]
    fn e2_runs_even_when_marker_blocks_e1() {
        // PAYG, marker present (E1 skipped), but E2 still runs and
        // mutates schema keys. The dispatch surface should report
        // only `schema_key_sort`, never `tool_array_sort`.
        let body = body_of(serde_json::json!({
            "model": "claude",
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "hi"}
                ]}
            ],
            "tools": [
                {
                    "name": "search",
                    "cache_control": {"type": "ephemeral"},
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                        },
                    },
                },
            ],
        }));
        let out = compress_anthropic_request(
            &body,
            CompressionMode::LiveZone,
            CacheControlAutoFrozen::Disabled,
            RequestAuthMode::Payg,
            "req-e2-2",
        );
        match out {
            Outcome::Compressed {
                strategies_applied, ..
            } => {
                assert!(strategies_applied.contains(&"schema_key_sort"));
                assert!(
                    !strategies_applied.contains(&"tool_array_sort"),
                    "E1 must skip when marker is present; got {strategies_applied:?}",
                );
            }
            other => panic!("expected Compressed with schema sort, got {other:?}"),
        }
    }

    #[test]
    fn e1_already_sorted_idempotent() {
        // Tools in alphabetic order already — E1 sort is a no-op.
        // E3 still fires (no customer marker, PAYG, has tools), so
        // we still get Outcome::Compressed but only with the
        // `e3_anthropic_cache_control` strategy — NOT with
        // `tool_array_sort`.
        let body = body_of(serde_json::json!({
            "model": "claude",
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "hi"}
                ]}
            ],
            "tools": [
                {"name": "apple"},
                {"name": "mango"},
                {"name": "zebra"},
            ],
        }));
        let out = compress_anthropic_request(
            &body,
            CompressionMode::LiveZone,
            CacheControlAutoFrozen::Disabled,
            RequestAuthMode::Payg,
            "req-e1-6",
        );
        match out {
            Outcome::Compressed {
                strategies_applied, ..
            } => {
                assert!(
                    !strategies_applied.contains(&"tool_array_sort"),
                    "expected NO tool_array_sort strategy on already-sorted tools, got: \
                     {strategies_applied:?}",
                );
                assert!(
                    strategies_applied.contains(&"e3_anthropic_cache_control"),
                    "expected e3_anthropic_cache_control on already-sorted PAYG tools, got: \
                     {strategies_applied:?}",
                );
            }
            other => {
                panic!("expected Compressed (E3 fires) for already-sorted tools, got {other:?}")
            }
        }
    }
}
