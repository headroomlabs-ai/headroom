//! Per-block compression-ratio observability — Phase G PR-G3.
//!
//! # Definition
//!
//! For every block the live-zone dispatcher successfully shrinks, we
//! observe one sample on the `proxy_compression_ratio_by_strategy`
//! histogram. The sample value is the ratio
//! `compressed_tokens / original_tokens` (in [0, 1)), so smaller =
//! better. The metric is labelled by:
//!
//! - `strategy`  — the compressor identifier (`smart_crusher`,
//!   `log_compressor`, `search_compressor`, `diff_compressor`,
//!   `e3_anthropic_cache_control`, `e1_tool_array_sort`, …). The
//!   label vocabulary is bounded by the static `&'static str` set
//!   the compressors return on their `BlockAction::Compressed`
//!   manifests — never a customer-controlled value.
//! - `content_type` — the detection-tier output for the block
//!   (`source_code`, `log`, `search_output`, `diff`, `text`, …).
//!   Cardinality bounded by `headroom_core::transforms::ContentType`.
//!
//! Counter twin: `proxy_compression_rejected_by_token_check_total{strategy}`
//! captures the "compressor ran but didn't shrink the token count"
//! case so dashboards can attribute the absence of histogram samples
//! to legitimate rejection rather than to a missing call site.
//!
//! # Bucket selection
//!
//! Buckets are tighter near 0 (the "aggressive shrink" tail dashboard
//! operators care about), wider in the middle. Anything above 1 would
//! be a tokenizer regression — `BlockAction::Compressed` enforces
//! `compressed_tokens < original_tokens`, so the histogram should
//! never see a sample > 0.99 in practice.

use std::sync::OnceLock;

use prometheus::{HistogramOpts, HistogramVec, IntCounterVec, Opts, Registry};

use super::metric_names::{
    LABEL_CONTENT_TYPE, LABEL_STRATEGY, METRIC_PROXY_COMPRESSION_RATIO_BY_STRATEGY,
    METRIC_PROXY_COMPRESSION_RATIO_BY_STRATEGY_HELP,
    METRIC_PROXY_COMPRESSION_REJECTED_BY_TOKEN_CHECK_TOTAL,
    METRIC_PROXY_COMPRESSION_REJECTED_BY_TOKEN_CHECK_TOTAL_HELP,
};

/// Histogram buckets in [0, 1). Tight at the aggressive end (≤0.25
/// means we kept ≤25% of the original tokens), coarser at the upper
/// end where a measurement is operationally indistinguishable from
/// "barely any shrink".
pub(super) const COMPRESSION_RATIO_BUCKETS: &[f64] =
    &[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 0.99];

/// `proxy_compression_ratio_by_strategy{strategy, content_type}`.
pub fn ratio_histogram(registry: &Registry) -> &'static HistogramVec {
    static HIST: OnceLock<HistogramVec> = OnceLock::new();
    HIST.get_or_init(|| {
        let opts = HistogramOpts::new(
            METRIC_PROXY_COMPRESSION_RATIO_BY_STRATEGY,
            METRIC_PROXY_COMPRESSION_RATIO_BY_STRATEGY_HELP,
        )
        .buckets(COMPRESSION_RATIO_BUCKETS.to_vec());
        let hist = HistogramVec::new(opts, &[LABEL_STRATEGY, LABEL_CONTENT_TYPE])
            .expect("proxy_compression_ratio_by_strategy descriptor is well-formed");
        registry
            .register(Box::new(hist.clone()))
            .expect("proxy_compression_ratio_by_strategy registers exactly once");
        hist
    })
}

/// `proxy_compression_rejected_by_token_check_total{strategy}`.
pub fn rejected_counter(registry: &Registry) -> &'static IntCounterVec {
    static COUNTER: OnceLock<IntCounterVec> = OnceLock::new();
    COUNTER.get_or_init(|| {
        let opts = Opts::new(
            METRIC_PROXY_COMPRESSION_REJECTED_BY_TOKEN_CHECK_TOTAL,
            METRIC_PROXY_COMPRESSION_REJECTED_BY_TOKEN_CHECK_TOTAL_HELP,
        );
        let counter = IntCounterVec::new(opts, &[LABEL_STRATEGY])
            .expect("proxy_compression_rejected_by_token_check_total descriptor is well-formed");
        registry
            .register(Box::new(counter.clone()))
            .expect("proxy_compression_rejected_by_token_check_total registers exactly once");
        counter
    })
}

/// Observe one compression-ratio sample. `strategy` and `content_type`
/// must be `&'static str` (bounded by compiler-known compressor +
/// content-type enum sets).
pub fn observe_ratio(
    strategy: &'static str,
    content_type: &str,
    original_tokens: usize,
    compressed_tokens: usize,
) {
    if original_tokens == 0 {
        // Per realignment build-constraint "no silent fallbacks": a
        // zero-denominator block would synthesise a `NaN` sample we
        // can't observe meaningfully. Loud-log and skip.
        tracing::warn!(
            event = "compression_ratio_zero_denominator",
            strategy = strategy,
            content_type = %content_type,
            compressed_tokens = compressed_tokens,
            "skipping proxy_compression_ratio_by_strategy observation: \
             original_tokens == 0 (would divide by zero)"
        );
        return;
    }
    let ratio = (compressed_tokens as f64) / (original_tokens as f64);
    ratio_histogram(super::prometheus::registry())
        .with_label_values(&[strategy, content_type])
        .observe(ratio);
    tracing::debug!(
        event = "metric_recorded",
        metric = METRIC_PROXY_COMPRESSION_RATIO_BY_STRATEGY,
        strategy = strategy,
        content_type = %content_type,
        original_tokens = original_tokens,
        compressed_tokens = compressed_tokens,
        ratio = ratio,
        "observed proxy_compression_ratio_by_strategy"
    );
}

/// Increment the rejection counter for a compressor that ran but did
/// not produce a token-smaller output.
pub fn record_rejected_by_token_check(strategy: &'static str) {
    rejected_counter(super::prometheus::registry())
        .with_label_values(&[strategy])
        .inc();
    tracing::debug!(
        event = "metric_recorded",
        metric = METRIC_PROXY_COMPRESSION_REJECTED_BY_TOKEN_CHECK_TOTAL,
        strategy = strategy,
        "incremented proxy_compression_rejected_by_token_check_total"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_original_tokens_skips_observation() {
        // Calling observe with original_tokens=0 must not panic and
        // must not divide-by-zero; the warn log is sufficient.
        observe_ratio("smart_crusher", "text", 0, 0);
    }

    #[test]
    fn ratio_50_percent_recorded() {
        // Drive the metric so the family appears in scrapes; we
        // exercise the path here, not the value (which is asserted
        // via integration tests).
        observe_ratio("test_strategy_50pct", "text", 200, 100);
    }
}
