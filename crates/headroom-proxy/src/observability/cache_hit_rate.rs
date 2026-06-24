//! Per-session cache-hit-rate observability — Phase G PR-G3.
//!
//! # The H-blocker metric
//!
//! Phase H ("retire the Python proxy") depends on this metric to
//! confirm parity between the Rust proxy and the soon-to-retire
//! Python proxy during canary. The acceptance gate in
//! `REALIGNMENT/10-phase-H-python-retirement.md:11-12` reads
//! `cache_hit_rate ≥ Python baseline; no 5xx regressions in 24h`.
//! That assertion is meaningless without this histogram.
//!
//! # Definition
//!
//! For every completed request session we observe one sample on the
//! `proxy_cache_hit_rate_per_session` histogram whose value is:
//!
//! ```text
//!   cache_read_input_tokens
//!   ───────────────────────
//!         total_input_tokens
//! ```
//!
//! `total_input_tokens` is the full denominator each provider
//! exposes on the `usage` payload (input + cache_read +
//! cache_creation), NOT the "billable" input that excludes cache
//! hits. Defining the denominator this way means the metric stays
//! meaningful across providers and stays in [0, 1].
//!
//! # When called
//!
//! - Anthropic: at `message_delta` (which carries the final `usage`).
//! - OpenAI Chat: on the final usage chunk (when
//!   `stream_options.include_usage = true`; otherwise no sample is
//!   emitted — the absence is itself a signal).
//! - OpenAI Responses: at `response.completed`.
//!
//! # Cardinality
//!
//! One label only: `provider ∈ {anthropic, openai_chat, openai_responses}`.
//! The metric is *intentionally* low-cardinality — the H1 canary
//! looks at fleet-wide hit rate, not per-model. Per-model breakdown
//! lives in the per-provider info logs we already emit.
//!
//! # Bucket selection
//!
//! Bucket boundaries are picked so the histogram discriminates the
//! "no cache" → "cache landed" transition with high resolution near
//! 0 and the "high-hit-rate" regime (where cache is paying for
//! itself) with high resolution near 1. The middle is coarser because
//! ~50% hit-rate is operationally uninteresting — either the cache
//! is working or it isn't.

use std::sync::OnceLock;

use prometheus::{HistogramOpts, HistogramVec, Registry};

use super::metric_names::{
    LABEL_PROVIDER, METRIC_PROXY_CACHE_HIT_RATE_PER_SESSION,
    METRIC_PROXY_CACHE_HIT_RATE_PER_SESSION_HELP,
};

/// Histogram buckets in [0, 1]. Tighter near 0 (so a "barely any
/// cache hit" stays distinguishable from a true zero) and near 1
/// (so a "near-perfect cache" stays distinguishable from a real
/// 100% hit). The middle band is intentionally coarser.
pub(super) const CACHE_HIT_RATE_BUCKETS: &[f64] =
    &[0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0];

/// Provider label vocabulary. Kept here (rather than scattered
/// across emit sites) so the cardinality budget is reviewable in one
/// place per realignment build-constraint "configurable + scalable".
pub mod provider {
    pub const ANTHROPIC: &str = "anthropic";
    pub const OPENAI_CHAT: &str = "openai_chat";
    pub const OPENAI_RESPONSES: &str = "openai_responses";
}

/// `proxy_cache_hit_rate_per_session{provider}` — initialised on
/// first call from `register_in`.
pub fn histogram(registry: &Registry) -> &'static HistogramVec {
    static HIST: OnceLock<HistogramVec> = OnceLock::new();
    HIST.get_or_init(|| {
        let opts = HistogramOpts::new(
            METRIC_PROXY_CACHE_HIT_RATE_PER_SESSION,
            METRIC_PROXY_CACHE_HIT_RATE_PER_SESSION_HELP,
        )
        .buckets(CACHE_HIT_RATE_BUCKETS.to_vec());
        let hist = HistogramVec::new(opts, &[LABEL_PROVIDER])
            .expect("proxy_cache_hit_rate_per_session descriptor is well-formed");
        registry
            .register(Box::new(hist.clone()))
            .expect("proxy_cache_hit_rate_per_session registers exactly once");
        hist
    })
}

/// Compute the per-session cache hit rate given the three counters
/// most providers expose on `usage`. Returns `None` when the
/// denominator is zero (no input tokens at all — typically a
/// degenerate request such as an empty messages array), so callers
/// can choose to skip the observation rather than divide-by-zero.
///
/// Per realignment build-constraint "no silent fallbacks", a
/// zero-denominator request is *NOT* coerced to `0.0`; it returns
/// `None` so the emit-site can log + skip instead of polluting the
/// histogram with synthesised samples.
pub fn compute_hit_rate(
    input_tokens: u64,
    cache_read_input_tokens: u64,
    cache_creation_input_tokens: u64,
) -> Option<f64> {
    let denom = input_tokens
        .saturating_add(cache_read_input_tokens)
        .saturating_add(cache_creation_input_tokens);
    if denom == 0 {
        return None;
    }
    Some(cache_read_input_tokens as f64 / denom as f64)
}

/// H2 gate: should we observe a cache-hit-rate sample for this
/// Anthropic session?
///
/// Returns `Some(rate)` ONLY when the stream completed cleanly
/// (`state.status == MessageStop`) AND the denominator is non-zero.
/// A client disconnect mid-stream closes the channel too — without
/// this gate we'd observe a half-finished session that has only
/// `message_start` usage and pollute the histogram with garbage.
///
/// Extracted from `proxy.rs::run_sse_state_machine` so the H2
/// contract is unit-testable independent of the global Prometheus
/// registry (which parallel tests share).
pub fn compute_anthropic_session_hit_rate(
    state: &crate::sse::anthropic::AnthropicStreamState,
) -> Option<f64> {
    if state.status != crate::sse::anthropic::StreamStatus::MessageStop {
        return None;
    }
    compute_hit_rate(
        state.usage.input_tokens,
        state.usage.cache_read_input_tokens,
        state.usage.cache_creation_input_tokens,
    )
}

/// Observe one per-session sample.
///
/// `provider` MUST be one of the [`provider`] constants — callers
/// from the SSE state machine pass the matching string verbatim. We
/// do not validate the label here because the cardinality is bounded
/// by the static call sites; an invalid label would be a bug, not a
/// runtime mismatch.
///
/// M3 fix: NaN / non-finite inputs are loud-skipped instead of
/// silently observed. `f64::clamp(0.0, 1.0)` returns NaN when the
/// input is NaN, so the prior implementation could pollute the
/// histogram with NaN samples in release builds (the debug_assert
/// was compiled out). Per "no silent fallbacks", an unexpected NaN
/// surfaces in the logs rather than getting eaten.
pub fn observe(provider: &'static str, request_id: &str, hit_rate: f64) {
    if !hit_rate.is_finite() {
        tracing::warn!(
            event = "cache_hit_rate_non_finite",
            metric = METRIC_PROXY_CACHE_HIT_RATE_PER_SESSION,
            provider = provider,
            request_id = %request_id,
            hit_rate = hit_rate,
            "refusing to observe a non-finite cache hit rate; this is a caller bug"
        );
        return;
    }
    debug_assert!(
        (0.0..=1.0).contains(&hit_rate),
        "cache hit rate must be in [0.0, 1.0]; got {hit_rate}"
    );
    // After the is_finite guard, clamp can only normalise legitimate
    // edge-of-range floats (1.0 + epsilon etc.) and never produces NaN.
    let clamped = hit_rate.clamp(0.0, 1.0);
    histogram(super::prometheus::registry())
        .with_label_values(&[provider])
        .observe(clamped);
    tracing::debug!(
        event = "metric_recorded",
        metric = METRIC_PROXY_CACHE_HIT_RATE_PER_SESSION,
        provider = provider,
        request_id = %request_id,
        hit_rate = clamped,
        "observed proxy_cache_hit_rate_per_session"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hit_rate_zero_when_no_cache_reads() {
        let r = compute_hit_rate(100, 0, 0).unwrap();
        assert!((r - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn hit_rate_one_when_all_reads_are_cache_hits() {
        // Edge case: input_tokens == 0 means upstream charged us
        // nothing new — every input came from cache.
        let r = compute_hit_rate(0, 1000, 0).unwrap();
        assert!((r - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn hit_rate_split_three_ways() {
        // 50 fresh input, 50 cache_read, 50 cache_creation → 50/150.
        let r = compute_hit_rate(50, 50, 50).unwrap();
        assert!((r - (1.0 / 3.0)).abs() < 1e-9);
    }

    #[test]
    fn hit_rate_none_on_empty_request() {
        // Degenerate: no tokens at all. Caller should skip the
        // observation, not coerce to 0.0.
        assert!(compute_hit_rate(0, 0, 0).is_none());
    }

    #[test]
    fn observe_nan_skipped_loudly() {
        // M3: a NaN input must NOT reach the histogram. The
        // pre-fix code clamped via `f64::clamp(0.0, 1.0)` which
        // returns NaN for NaN input — a NaN sample in release
        // builds was the bug. After the fix we log + skip.
        // The histogram count for this synthetic provider stays
        // at whatever it was before the call.
        let label = "test_nan_provider_v1";
        // Drive a guaranteed-clean session count via a real
        // observation, then push a NaN and assert no count move.
        observe(label, "req-nan-baseline", 0.5);
        let before = histogram(super::super::prometheus::registry())
            .with_label_values(&[label])
            .get_sample_count();
        observe(label, "req-nan-attempt", f64::NAN);
        let after = histogram(super::super::prometheus::registry())
            .with_label_values(&[label])
            .get_sample_count();
        assert_eq!(
            before, after,
            "NaN must not be observed; expected count unchanged from {before} to {after}"
        );
    }

    #[test]
    fn h2_aborted_anthropic_stream_returns_none() {
        // H2: a stream that closes without `message_stop` (Open
        // state) MUST NOT produce a sample, regardless of usage
        // values.
        use crate::sse::anthropic::{AnthropicStreamState, StreamStatus, UsageBuilder};
        let state = AnthropicStreamState {
            status: StreamStatus::Open,
            usage: UsageBuilder {
                input_tokens: 800,
                cache_read_input_tokens: 200,
                output_tokens: 50,
                cache_creation_input_tokens: 0,
            },
            ..Default::default()
        };
        assert!(
            compute_anthropic_session_hit_rate(&state).is_none(),
            "H2 gate must skip emission for non-completed stream"
        );
    }

    #[test]
    fn h2_errored_anthropic_stream_returns_none() {
        // H2: an Errored stream is also not "completed cleanly" —
        // skip the observation.
        use crate::sse::anthropic::{AnthropicStreamState, StreamStatus, UsageBuilder};
        let state = AnthropicStreamState {
            status: StreamStatus::Errored,
            usage: UsageBuilder {
                input_tokens: 800,
                cache_read_input_tokens: 200,
                output_tokens: 50,
                cache_creation_input_tokens: 0,
            },
            ..Default::default()
        };
        assert!(
            compute_anthropic_session_hit_rate(&state).is_none(),
            "H2 gate must skip emission for errored stream"
        );
    }

    #[test]
    fn h2_completed_anthropic_stream_returns_rate() {
        // H2 positive case: MessageStop + non-zero usage → return
        // the rate so the caller can observe it.
        use crate::sse::anthropic::{AnthropicStreamState, StreamStatus, UsageBuilder};
        let state = AnthropicStreamState {
            status: StreamStatus::MessageStop,
            usage: UsageBuilder {
                input_tokens: 800,
                cache_read_input_tokens: 200,
                output_tokens: 50,
                cache_creation_input_tokens: 0,
            },
            ..Default::default()
        };
        let rate = compute_anthropic_session_hit_rate(&state).expect("completed stream emits");
        // 200 / (800 + 200 + 0) = 0.2
        assert!((rate - 0.2).abs() < 1e-9);
    }

    #[test]
    fn h2_completed_but_zero_denominator_returns_none() {
        // Even on a completed stream, a zero-token request returns
        // None — per "no silent fallbacks", no synthesised 0.0.
        use crate::sse::anthropic::{AnthropicStreamState, StreamStatus, UsageBuilder};
        let state = AnthropicStreamState {
            status: StreamStatus::MessageStop,
            usage: UsageBuilder::default(),
            ..Default::default()
        };
        assert!(compute_anthropic_session_hit_rate(&state).is_none());
    }

    #[test]
    fn observe_infinity_skipped_loudly() {
        // Same contract for +/- infinity.
        let label = "test_inf_provider_v1";
        observe(label, "req-inf-baseline", 0.5);
        let before = histogram(super::super::prometheus::registry())
            .with_label_values(&[label])
            .get_sample_count();
        observe(label, "req-pos-inf", f64::INFINITY);
        observe(label, "req-neg-inf", f64::NEG_INFINITY);
        let after = histogram(super::super::prometheus::registry())
            .with_label_values(&[label])
            .get_sample_count();
        assert_eq!(before, after);
    }
}
