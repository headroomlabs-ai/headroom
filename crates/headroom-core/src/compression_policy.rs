//! Per-auth-mode compression policy — Phase F PR-F2.1, extended in F2.2.
//!
//! F1 (`auth_mode.rs`) classifies each inbound request into one of
//! `{Payg, OAuth, Subscription}`. Phase F2.1 turns that classification
//! into a `CompressionPolicy` that downstream pipeline stages read to
//! decide whether they run. F2.2 extends the same struct with per-mode
//! tuning fields so the same call sites also read *how aggressively*
//! to run.
//!
//! Why a struct instead of `match auth_mode { ... }` everywhere?
//! Two reasons:
//!
//! 1. **Centralisation.** Without a policy struct, the per-mode
//!    decision is duplicated at every gate (E3 cache_control, E4
//!    prompt_cache_key, the new live-zone gate, the new cache_aligner
//!    gate, …). When F2.2 wants to tune (e.g. allow OAuth users a
//!    relaxed live-zone gate but stricter volatile-detector threshold)
//!    we'd need to find every site. The struct is the one place to
//!    edit; call sites just read `policy.field`.
//!
//! 2. **Test surface.** `for_mode(AuthMode) -> CompressionPolicy` is
//!    pure and trivial to property-test. Asserting per-mode values
//!    against the struct catches regressions cheaply, whereas asserting
//!    end-to-end behaviour against the dispatcher requires a full
//!    request fixture.
//!
//! ## Field semantics
//!
//! ### F2.1 fields (load-bearing for closing #327 / #388)
//!
//! - **`live_zone_only`**: when `true`, downstream stages MUST NOT
//!   modify bytes outside the post-cache-marker live zone. Phase B's
//!   Rust dispatcher is *already* live-zone-only by construction, so
//!   this flag is effectively a no-op on the Rust path and exists for
//!   the Python `TransformPipeline`'s `CacheAligner` / `ContentRouter`
//!   gates. Storing it on the canonical struct keeps the cross-
//!   language parity tests honest — Python and Rust must agree on the
//!   field map even when only one side acts on a value.
//!
//! - **`cache_aligner_enabled`**: when `false`, the Python
//!   `CacheAligner` transform's `should_apply` MUST return `False`.
//!   `CacheAligner` is the load-bearing fix for the cache-instability
//!   complaints — historically it has been mutating cached prefixes
//!   and writing into `_previous_prefix_hash` per pipeline instance,
//!   which is what destabilised Subscription users' prompt caches.
//!   Disabling it for Subscription is the user-visible win of F2.1.
//!
//! ### F2.2 tuning fields (CONSERVATIVE defaults pending bake telemetry)
//!
//! - **`volatile_token_threshold`**: per-mode token-count threshold
//!   below which content is treated as cache-stable (i.e. not flagged
//!   as volatile). Subscription is conservative (low threshold → flag
//!   more aggressively → keep prompts stable) while PAYG is aggressive
//!   (higher threshold → tolerate more volatile noise before warning).
//!   F2.1 had no such threshold; F2.2 introduces the field plumbed
//!   through the struct so future detector code can pick it up. NOTE:
//!   no current detector consumes this value — it lands plumbed-but-
//!   unconsumed in F2.2 (intentional; the volatile detector in
//!   `cache_aligner.py` is shape-based, not token-count-based, and
//!   wiring it would force a detector refactor outside F2.2 scope).
//!
//! - **`max_lossy_ratio`**: per-mode upper bound on how aggressive
//!   lossy compression can be, expressed as the fraction of original
//!   tokens that may be dropped (`0.0` = no lossy compression allowed,
//!   `1.0` = unlimited). Subscription is conservative (`0.25`) so cache
//!   prefixes stay stable, PAYG aggressive (`0.45`). NOTE: no current
//!   compressor consumes this value — it lands plumbed-but-unconsumed
//!   in F2.2 (the `target_ratio` runtime kwarg in `content_router.py`
//!   is a separate, caller-driven knob; wiring `max_lossy_ratio` as a
//!   policy-driven cap is F2.2-followup once telemetry decides whether
//!   to gate lossy paths or just observe them).
//!
//! - **`toin_read_only`**: when `true`, TOIN serves cached
//!   recommendations but never *writes* new pattern observations from
//!   this request. Subscription requests pay for prompt-cache stability,
//!   so we don't want their compression events to mutate the global
//!   learning pool — consistency over learning. PAYG/OAuth still write
//!   so the network effect keeps growing.
//!
//! ## Per-mode F2.2 values (CONSERVATIVE; F2.2-followup will tune)
//!
//! | Mode         | live_zone_only | cache_aligner_enabled | volatile_token_threshold | max_lossy_ratio | toin_read_only |
//! |--------------|----------------|-----------------------|--------------------------|-----------------|----------------|
//! | Payg         | false          | true                  | 128                      | 0.45            | false          |
//! | OAuth        | false          | true (= PAYG today)   | 128 (= PAYG today)       | 0.45 (= PAYG)   | false (= PAYG) |
//! | Subscription | true           | false                 | 32                       | 0.25            | true           |
//!
//! OAuth starts identical to PAYG. F2.2-followup will divide them once
//! telemetry from F2.1's bake on `main` shows what each mode actually
//! costs / saves.
//!
//! ## What this struct does NOT replace
//!
//! Phase E's existing PAYG-only gates (cache_control auto-placement,
//! prompt_cache_key injection) keep matching `auth_mode == Payg`
//! directly. Migrating those to the policy struct is F2.2 cleanup
//! work. Doing it in F2.1 would balloon the diff and the existing
//! gates already produce the correct per-mode behaviour — there's no
//! user-visible reason to refactor them now.

use crate::auth_mode::AuthMode;

// ── F2.2 per-mode default values (CONSERVATIVE pending bake telemetry) ──
// Centralised constants instead of inlining in the match arms so a
// follow-up tune lands in one place. Each constant is `pub(crate)` so
// the unit tests can assert against the same source of truth — if a
// caller drifts the per-mode value, the assertion fails.
//
// Per the realignment build constraints (project memory
// `feedback_realignment_build_constraints.md`): "configurable / no
// hardcoded values". The configuration *is* the per-mode default — we
// deliberately do NOT add a separate env var per field. Operators tune
// by editing these constants and shipping a new build, which is the
// same pattern the F2.1 fields use.

/// PAYG: aggressive — let volatile content noise up to ~128 tokens slip
/// before flagging. Higher than Subscription because PAYG users opt in
/// to aggressive compression.
pub(crate) const VOLATILE_TOKEN_THRESHOLD_PAYG: u32 = 128;

/// Subscription: conservative — flag volatile content earlier (32
/// tokens) so cache prefixes stay stable.
pub(crate) const VOLATILE_TOKEN_THRESHOLD_SUBSCRIPTION: u32 = 32;

/// PAYG: cap lossy compression at 45% of original tokens. Aggressive
/// but bounded — F2.1 had no cap (effectively `1.0`), F2.2 introduces
/// one.
pub(crate) const MAX_LOSSY_RATIO_PAYG: f32 = 0.45;

/// Subscription: conservative cap at 25%. Cache stability over savings.
pub(crate) const MAX_LOSSY_RATIO_SUBSCRIPTION: f32 = 0.25;

/// Anthropic prompt-cache write multiplier: a `cache_creation` token
/// costs 1.25× a plain input token (5-minute TTL tier). Input to the
/// net-cost mutation formula (#856).
pub const CACHE_WRITE_MULTIPLIER: f32 = 1.25;

/// Anthropic prompt-cache read multiplier: a `cache_read` token costs
/// 0.1× a plain input token. Input to the net-cost mutation formula
/// (#856).
pub const CACHE_READ_MULTIPLIER: f32 = 0.1;

/// Per-auth-mode policy that downstream compression stages consult.
///
/// `Copy` because the struct is small POD (two `bool`s + a `u32` + an
/// `f32` + a `bool`) — passing by value is cheaper than passing a
/// reference and the call sites all want owned copies anyway.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompressionPolicy {
    /// When `true`, transforms MUST NOT modify bytes outside the
    /// post-cache-marker live zone. See module docs.
    pub live_zone_only: bool,

    /// When `false`, the `CacheAligner` transform MUST be skipped.
    /// See module docs.
    pub cache_aligner_enabled: bool,

    /// F2.2: per-mode threshold (in tokens) below which content is
    /// treated as cache-stable. Subscription is conservative
    /// (`32`); PAYG aggressive (`128`). See module docs.
    ///
    /// NOT consumed by any detector in F2.2 — plumbed through the
    /// struct so the volatile detector refactor in a follow-up PR
    /// has a stable hook to read from.
    pub volatile_token_threshold: u32,

    /// F2.2: per-mode upper bound on lossy compression aggressiveness,
    /// expressed as the fraction of original tokens that may be
    /// dropped (`0.0`–`1.0`). Subscription `0.25`, PAYG `0.45`.
    /// See module docs.
    ///
    /// NOT consumed by any compressor in F2.2 — plumbed through the
    /// struct as a stable hook for a follow-up PR that gates lossy
    /// paths on the cap. Distinct from the caller-driven
    /// `target_ratio` kwarg in the Python ContentRouter.
    pub max_lossy_ratio: f32,

    /// F2.2: when `true`, TOIN serves cached recommendations but
    /// never writes new pattern observations from this request.
    /// Subscription `true` (consistency over learning), PAYG/OAuth
    /// `false` (network effect keeps growing).
    pub toin_read_only: bool,
}

// `f32` doesn't impl `Eq`, so the derived `Eq` would be invalid. Two
// `f32`s in this struct are never NaN by construction (we only set
// them from finite literal constants), so `PartialEq` is sufficient.
// The unit tests assert structural equality via `assert_eq!`.

impl CompressionPolicy {
    /// Resolve the F2.1+F2.2 policy for an auth mode. See module docs
    /// for per-mode rationale.
    pub fn for_mode(mode: AuthMode) -> Self {
        match mode {
            AuthMode::Payg => Self {
                live_zone_only: false,
                cache_aligner_enabled: true,
                volatile_token_threshold: VOLATILE_TOKEN_THRESHOLD_PAYG,
                max_lossy_ratio: MAX_LOSSY_RATIO_PAYG,
                toin_read_only: false,
            },
            // OAuth identical to PAYG in F2.1+F2.2. F2.2-followup may
            // diverge once telemetry shows what OAuth users actually
            // need.
            AuthMode::OAuth => Self {
                live_zone_only: false,
                cache_aligner_enabled: true,
                volatile_token_threshold: VOLATILE_TOKEN_THRESHOLD_PAYG,
                max_lossy_ratio: MAX_LOSSY_RATIO_PAYG,
                toin_read_only: false,
            },
            // The user-visible win of F2.1: subscription users stop
            // seeing cache instability because CacheAligner no longer
            // touches their prefix. F2.2 extends that protection: the
            // volatile threshold is tighter, the lossy cap is lower,
            // and TOIN won't mutate the learning pool from these
            // requests.
            AuthMode::Subscription => Self {
                live_zone_only: true,
                cache_aligner_enabled: false,
                volatile_token_threshold: VOLATILE_TOKEN_THRESHOLD_SUBSCRIPTION,
                max_lossy_ratio: MAX_LOSSY_RATIO_SUBSCRIPTION,
                toin_read_only: true,
            },
        }
    }

    /// Whether the live-zone dispatcher should run at all for this
    /// policy. Always `true` in F2.1 — every mode still gets live-zone
    /// compression (closing #327/#388 requires Subscription to KEEP
    /// compressing the live zone, just stop destabilising the cache).
    /// F2.2 may flip Subscription to `false` if telemetry shows the
    /// live-zone savings aren't worth the latency.
    pub fn live_zone_compression_enabled(&self) -> bool {
        true
    }

    /// Net gain (in plain-input-token cost units) of a mutation that
    /// removes `delta_t` tokens from a message whose cached suffix is
    /// `suffix_tokens` long (#856).
    ///
    /// Mutating message K invalidates every cached token after it: the
    /// suffix is re-written once at the write multiplier instead of
    /// being read at the read multiplier, costing
    /// `P_alive · (w − r) · S`. In exchange, `delta_t` tokens are gone
    /// from the current write and every one of the `expected_reads`
    /// remaining reads of the chain, saving `ΔT · (w + r·(R − 1))`.
    ///
    ///   gain = ΔT · (w + r·(R − 1))  −  P_alive · (w − r) · S
    ///
    /// Sanity anchors (Anthropic w=1.25, r=0.1), matching the unit
    /// tests below: a 2K shave under a 50K warm suffix needs ~276
    /// remaining reads to pay off (rarely profitable); a 50K shave
    /// under a 10K suffix is profitable from the first write (its
    /// break-even read count is negative); a live-zone edit (S = 0)
    /// is always profitable.
    ///
    /// Takes `&self` so a follow-up can apply per-mode margins; today
    /// the arithmetic is mode-independent. Inputs are clamped:
    /// `expected_reads` to `>= 0` (NaN → 0), `p_alive` to `[0, 1]`
    /// (NaN → 1, the conservative full-penalty assumption).
    pub fn net_mutation_gain(
        &self,
        delta_t: u32,
        suffix_tokens: u32,
        expected_reads: f32,
        p_alive: f32,
    ) -> f32 {
        let w = CACHE_WRITE_MULTIPLIER;
        let r = CACHE_READ_MULTIPLIER;
        // f32::max ignores NaN (returns the other operand), so NaN reads
        // land on 0.0; clamp would propagate NaN, so guard alive explicitly.
        let reads = expected_reads.max(0.0);
        let alive = if p_alive.is_nan() {
            1.0
        } else {
            p_alive.clamp(0.0, 1.0)
        };
        (delta_t as f32) * (w + r * (reads - 1.0)) - alive * (w - r) * (suffix_tokens as f32)
    }

    /// Decision form of [`Self::net_mutation_gain`]: mutate iff the
    /// gain is strictly positive.
    pub fn should_mutate_deep(
        &self,
        delta_t: u32,
        suffix_tokens: u32,
        expected_reads: f32,
        p_alive: f32,
    ) -> bool {
        self.net_mutation_gain(delta_t, suffix_tokens, expected_reads, p_alive) > 0.0
    }

    /// Remaining-read count at which a warm-cache (P_alive = 1)
    /// mutation breaks even:
    ///
    ///   R = ((w − r) / r) · (S/ΔT − 1)   ≈ 11.5 · S/ΔT  for S ≫ ΔT
    ///
    /// Useful for decision telemetry ("this edit pays off if the
    /// session lasts N more turns"). Returns 0 when `delta_t` is 0
    /// (no savings — callers gate on `delta_t > 0`).
    pub fn break_even_reads(&self, delta_t: u32, suffix_tokens: u32) -> f32 {
        if delta_t == 0 {
            return 0.0;
        }
        let w = CACHE_WRITE_MULTIPLIER;
        let r = CACHE_READ_MULTIPLIER;
        ((w - r) / r) * ((suffix_tokens as f32) / (delta_t as f32) - 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payg_is_aggressive() {
        let p = CompressionPolicy::for_mode(AuthMode::Payg);
        assert!(!p.live_zone_only, "PAYG can touch outside live zone");
        assert!(p.cache_aligner_enabled, "PAYG runs cache aligner");
        assert!(p.live_zone_compression_enabled());
    }

    #[test]
    fn payg_tuning_fields_aggressive() {
        // F2.2: per-mode tuning fields. PAYG values are the aggressive
        // end of the conservative-defaults spectrum — F2.2-followup may
        // raise them once bake telemetry confirms savings.
        let p = CompressionPolicy::for_mode(AuthMode::Payg);
        assert_eq!(
            p.volatile_token_threshold, 128,
            "PAYG volatile threshold is the relaxed default; F2.2-followup will tune"
        );
        assert!(
            (p.max_lossy_ratio - 0.45).abs() < f32::EPSILON,
            "PAYG max_lossy_ratio caps lossy paths at 0.45; F2.2-followup will tune"
        );
        assert!(
            !p.toin_read_only,
            "PAYG keeps TOIN write-enabled — network effect feeds on PAYG traffic"
        );
    }

    #[test]
    fn oauth_matches_payg_today() {
        // Canary: when F2.2-followup diverges OAuth from PAYG, this test
        // fails and forces a deliberate update — which is the point.
        // Covers ALL fields (F2.1 + F2.2) so a future field-level
        // divergence (e.g. OAuth gets stricter `max_lossy_ratio` than
        // PAYG) trips the assertion just as loudly as a flag flip.
        let oauth = CompressionPolicy::for_mode(AuthMode::OAuth);
        let payg = CompressionPolicy::for_mode(AuthMode::Payg);
        assert_eq!(
            oauth, payg,
            "F2.1+F2.2 ship OAuth=PAYG; F2.2-followup will diverge based on telemetry"
        );
    }

    #[test]
    fn subscription_disables_cache_aligner() {
        let p = CompressionPolicy::for_mode(AuthMode::Subscription);
        assert!(p.live_zone_only, "Subscription is live-zone-only");
        assert!(
            !p.cache_aligner_enabled,
            "Subscription MUST skip cache aligner — load-bearing for #327/#388"
        );
        assert!(
            p.live_zone_compression_enabled(),
            "Subscription still gets live-zone compression — closing the cache complaint must NOT mean shipping zero compression"
        );
    }

    #[test]
    fn subscription_tuning_fields_conservative() {
        // F2.2: per-mode tuning fields. Subscription is the conservative
        // end — tighter threshold, lower lossy cap, TOIN read-only — so
        // cache prefixes stay stable and the learning pool isn't
        // mutated from cache-stability-sensitive traffic.
        let p = CompressionPolicy::for_mode(AuthMode::Subscription);
        assert_eq!(
            p.volatile_token_threshold, 32,
            "Subscription volatile threshold flags content earlier (cache stability)"
        );
        assert!(
            (p.max_lossy_ratio - 0.25).abs() < f32::EPSILON,
            "Subscription max_lossy_ratio caps lossy paths at 0.25 (conservative)"
        );
        assert!(
            p.toin_read_only,
            "Subscription MUST be TOIN read-only — load-bearing for keeping the learning pool consistent across cache-sensitive traffic"
        );
    }

    #[test]
    fn max_lossy_ratio_in_unit_interval() {
        // Defensive: every per-mode `max_lossy_ratio` MUST be in `[0.0,
        // 1.0]` because it expresses a fraction. A tune that drifts
        // outside the unit interval is a bug — catch it cheaply here
        // rather than at the eventual consumer site.
        for mode in [AuthMode::Payg, AuthMode::OAuth, AuthMode::Subscription] {
            let r = CompressionPolicy::for_mode(mode).max_lossy_ratio;
            assert!(
                (0.0..=1.0).contains(&r),
                "max_lossy_ratio for {mode:?} = {r} is outside [0.0, 1.0]"
            );
        }
    }

    // --- Net-cost mutation formula (#856). Scenario values are golden:
    // tests/test_compression_policy.py asserts the identical numbers
    // against the Python hand-mirror, so a drift in either side trips
    // the parity pair loudly.

    #[test]
    fn net_gain_small_shave_deep_suffix_is_loss() {
        // Shave 2K under a 50K warm suffix at R=10 remaining reads:
        // 2000·(1.25 + 0.1·9) − 1.0·1.15·50000 = 4300 − 57500 = −53200.
        let p = CompressionPolicy::for_mode(AuthMode::Payg);
        let gain = p.net_mutation_gain(2_000, 50_000, 10.0, 1.0);
        assert!((gain - (-53_200.0)).abs() < 1.0, "gain = {gain}");
        assert!(!p.should_mutate_deep(2_000, 50_000, 10.0, 1.0));
    }

    #[test]
    fn net_gain_big_shave_shallow_suffix_is_win() {
        // Shave 50K under a 10K warm suffix at R=3:
        // 50000·(1.25 + 0.1·2) − 1.0·1.15·10000 = 72500 − 11500 = 61000.
        let p = CompressionPolicy::for_mode(AuthMode::Payg);
        let gain = p.net_mutation_gain(50_000, 10_000, 3.0, 1.0);
        assert!((gain - 61_000.0).abs() < 1.0, "gain = {gain}");
        assert!(p.should_mutate_deep(50_000, 10_000, 3.0, 1.0));
    }

    #[test]
    fn net_gain_live_zone_edit_always_profitable() {
        // S = 0 derives the existing Subscription live-zone policy as a
        // special case: nothing cached is invalidated, so any positive
        // shave wins even at R=0 (gain = ΔT·(w − r) > 0).
        let p = CompressionPolicy::for_mode(AuthMode::Subscription);
        assert!(p.should_mutate_deep(1, 0, 0.0, 1.0));
        assert!(p.should_mutate_deep(2_000, 0, 0.0, 1.0));
    }

    #[test]
    fn net_gain_cold_cache_ignores_suffix() {
        // P_alive = 0 (TTL lapsed): no warm suffix to lose, so even the
        // worst shave/suffix ratio is profitable. This is the idle-timer
        // compaction window.
        let p = CompressionPolicy::for_mode(AuthMode::Payg);
        assert!(p.should_mutate_deep(2_000, 50_000, 0.0, 0.0));
    }

    #[test]
    fn net_gain_clamps_out_of_range_inputs() {
        let p = CompressionPolicy::for_mode(AuthMode::Payg);
        // Negative reads clamp to 0; p_alive > 1 clamps to 1.
        let clamped = p.net_mutation_gain(2_000, 50_000, -5.0, 7.0);
        let reference = p.net_mutation_gain(2_000, 50_000, 0.0, 1.0);
        assert!((clamped - reference).abs() < f32::EPSILON);
    }

    #[test]
    fn net_gain_guards_nan_inputs() {
        // NaN reads → 0, NaN p_alive → 1: gain stays finite and matches
        // the conservative reference instead of poisoning the decision.
        let p = CompressionPolicy::for_mode(AuthMode::Payg);
        let guarded = p.net_mutation_gain(2_000, 50_000, f32::NAN, f32::NAN);
        assert!(guarded.is_finite());
        let reference = p.net_mutation_gain(2_000, 50_000, 0.0, 1.0);
        assert!((guarded - reference).abs() < f32::EPSILON);
    }

    #[test]
    fn break_even_reads_matches_research_anchor() {
        // R = 11.5·(S/ΔT − 1): 2K shave / 50K suffix → 11.5·24 = 276
        // (rarely profitable); 50K shave / 10K suffix →
        // 11.5·(0.2 − 1) < 0 → profitable from the first read.
        let p = CompressionPolicy::for_mode(AuthMode::Payg);
        let r = p.break_even_reads(2_000, 50_000);
        assert!((r - 276.0).abs() < 0.5, "break-even = {r}");
        assert!(p.break_even_reads(50_000, 10_000) < 0.0);
        assert_eq!(p.break_even_reads(0, 10_000), 0.0);
    }
}
