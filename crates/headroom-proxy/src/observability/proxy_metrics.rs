//! Phase G PR-G3 proxy-side observability metrics: rate-limit gauges,
//! the passthrough-bytes-modified alarm counter, service-tier +
//! response-status counters, and the image-base64 redaction counter.
//!
//! Grouped in one file because each metric is a thin
//! `OnceLock` + 1–2 emit helpers; splitting per metric would dilute
//! the call sites without adding test surface. Heavier metrics with
//! their own validation logic (`cache_hit_rate`, `compression_ratio`)
//! live in their own modules.

use std::sync::OnceLock;

use prometheus::{IntCounterVec, IntGaugeVec, Opts, Registry};

use super::metric_names::{
    LABEL_PATH, LABEL_PROVIDER, LABEL_STATUS, LABEL_TIER,
    METRIC_PROXY_PASSTHROUGH_BYTES_MODIFIED_TOTAL,
    METRIC_PROXY_PASSTHROUGH_BYTES_MODIFIED_TOTAL_HELP,
    METRIC_PROXY_RATE_LIMIT_REMAINING_INPUT_TOKENS,
    METRIC_PROXY_RATE_LIMIT_REMAINING_INPUT_TOKENS_HELP,
    METRIC_PROXY_RATE_LIMIT_REMAINING_OUTPUT_TOKENS,
    METRIC_PROXY_RATE_LIMIT_REMAINING_OUTPUT_TOKENS_HELP,
    METRIC_PROXY_RATE_LIMIT_REMAINING_REQUESTS, METRIC_PROXY_RATE_LIMIT_REMAINING_REQUESTS_HELP,
    METRIC_PROXY_RATE_LIMIT_REMAINING_TOKENS, METRIC_PROXY_RATE_LIMIT_REMAINING_TOKENS_HELP,
    METRIC_PROXY_RESPONSE_STATUS_COUNT_TOTAL, METRIC_PROXY_RESPONSE_STATUS_COUNT_TOTAL_HELP,
    METRIC_PROXY_SERVICE_TIER_COUNT_TOTAL, METRIC_PROXY_SERVICE_TIER_COUNT_TOTAL_HELP,
};

// ---------- proxy_passthrough_bytes_modified_total{path} ----------

/// Counter (not gauge) so the metric obeys Prometheus' `_total`
/// convention while still meeting the spec's "must stay 0" alarm
/// requirement: dashboards alert on `rate(...[5m]) > 0`. A counter
/// stays at 0 forever until something actually modifies passthrough
/// bytes — which is the alarmable event.
pub fn passthrough_bytes_modified_counter(registry: &Registry) -> &'static IntCounterVec {
    static COUNTER: OnceLock<IntCounterVec> = OnceLock::new();
    COUNTER.get_or_init(|| {
        let opts = Opts::new(
            METRIC_PROXY_PASSTHROUGH_BYTES_MODIFIED_TOTAL,
            METRIC_PROXY_PASSTHROUGH_BYTES_MODIFIED_TOTAL_HELP,
        );
        let counter = IntCounterVec::new(opts, &[LABEL_PATH])
            .expect("proxy_passthrough_bytes_modified_total descriptor is well-formed");
        registry
            .register(Box::new(counter.clone()))
            .expect("proxy_passthrough_bytes_modified_total registers exactly once");
        counter
    })
}

/// Add `bytes` modified on a path that was supposed to be byte-equal
/// passthrough. The increment value is the byte delta — operators
/// then `rate(...)` to see "bytes/sec of policy violation".
pub fn record_passthrough_bytes_modified(path: &str, bytes: u64, request_id: &str) {
    passthrough_bytes_modified_counter(super::prometheus::registry())
        .with_label_values(&[path])
        .inc_by(bytes);
    tracing::warn!(
        event = "passthrough_bytes_modified",
        metric = METRIC_PROXY_PASSTHROUGH_BYTES_MODIFIED_TOTAL,
        path = %path,
        bytes = bytes,
        request_id = %request_id,
        "passthrough path modified bytes; this is the cache-safety alarm condition"
    );
}

// ---------- proxy_rate_limit_remaining_* gauges ----------

pub fn rate_limit_remaining_requests_gauge(registry: &Registry) -> &'static IntGaugeVec {
    static GAUGE: OnceLock<IntGaugeVec> = OnceLock::new();
    GAUGE.get_or_init(|| {
        let opts = Opts::new(
            METRIC_PROXY_RATE_LIMIT_REMAINING_REQUESTS,
            METRIC_PROXY_RATE_LIMIT_REMAINING_REQUESTS_HELP,
        );
        let gauge = IntGaugeVec::new(opts, &[LABEL_PROVIDER])
            .expect("proxy_rate_limit_remaining_requests descriptor is well-formed");
        registry
            .register(Box::new(gauge.clone()))
            .expect("proxy_rate_limit_remaining_requests registers exactly once");
        gauge
    })
}

pub fn rate_limit_remaining_tokens_gauge(registry: &Registry) -> &'static IntGaugeVec {
    static GAUGE: OnceLock<IntGaugeVec> = OnceLock::new();
    GAUGE.get_or_init(|| {
        let opts = Opts::new(
            METRIC_PROXY_RATE_LIMIT_REMAINING_TOKENS,
            METRIC_PROXY_RATE_LIMIT_REMAINING_TOKENS_HELP,
        );
        let gauge = IntGaugeVec::new(opts, &[LABEL_PROVIDER])
            .expect("proxy_rate_limit_remaining_tokens descriptor is well-formed");
        registry
            .register(Box::new(gauge.clone()))
            .expect("proxy_rate_limit_remaining_tokens registers exactly once");
        gauge
    })
}

pub fn rate_limit_remaining_input_tokens_gauge(registry: &Registry) -> &'static IntGaugeVec {
    static GAUGE: OnceLock<IntGaugeVec> = OnceLock::new();
    GAUGE.get_or_init(|| {
        let opts = Opts::new(
            METRIC_PROXY_RATE_LIMIT_REMAINING_INPUT_TOKENS,
            METRIC_PROXY_RATE_LIMIT_REMAINING_INPUT_TOKENS_HELP,
        );
        let gauge = IntGaugeVec::new(opts, &[LABEL_PROVIDER])
            .expect("proxy_rate_limit_remaining_input_tokens descriptor is well-formed");
        registry
            .register(Box::new(gauge.clone()))
            .expect("proxy_rate_limit_remaining_input_tokens registers exactly once");
        gauge
    })
}

pub fn rate_limit_remaining_output_tokens_gauge(registry: &Registry) -> &'static IntGaugeVec {
    static GAUGE: OnceLock<IntGaugeVec> = OnceLock::new();
    GAUGE.get_or_init(|| {
        let opts = Opts::new(
            METRIC_PROXY_RATE_LIMIT_REMAINING_OUTPUT_TOKENS,
            METRIC_PROXY_RATE_LIMIT_REMAINING_OUTPUT_TOKENS_HELP,
        );
        let gauge = IntGaugeVec::new(opts, &[LABEL_PROVIDER])
            .expect("proxy_rate_limit_remaining_output_tokens descriptor is well-formed");
        registry
            .register(Box::new(gauge.clone()))
            .expect("proxy_rate_limit_remaining_output_tokens registers exactly once");
        gauge
    })
}

/// Snapshot of upstream rate-limit headers extracted from one
/// response. None-fields are headers the upstream did not include
/// (per realignment build-constraint "no silent fallbacks": we do not
/// fabricate a value, we just don't emit on that gauge).
#[derive(Debug, Default, Clone, Copy)]
pub struct RateLimitSnapshot {
    pub remaining_requests: Option<i64>,
    pub remaining_tokens: Option<i64>,
    pub remaining_input_tokens: Option<i64>,
    pub remaining_output_tokens: Option<i64>,
}

/// Extract a `RateLimitSnapshot` from a HeaderMap. Accepts both
/// Anthropic (`anthropic-ratelimit-*`) and OpenAI (`x-ratelimit-*`)
/// header families. Missing headers stay `None`.
///
/// `Retry-After` is intentionally NOT parsed here — that header is
/// upstream-bounded by 429s, not steady-state telemetry, and lives in
/// the existing structured 429 log line.
pub fn extract_rate_limit_snapshot(headers: &http::HeaderMap) -> RateLimitSnapshot {
    let parse_i64 = |name: &str| -> Option<i64> {
        headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.trim().parse::<i64>().ok())
    };
    RateLimitSnapshot {
        remaining_requests: parse_i64("anthropic-ratelimit-requests-remaining")
            .or_else(|| parse_i64("x-ratelimit-remaining-requests")),
        remaining_tokens: parse_i64("anthropic-ratelimit-tokens-remaining")
            .or_else(|| parse_i64("x-ratelimit-remaining-tokens")),
        remaining_input_tokens: parse_i64("anthropic-ratelimit-input-tokens-remaining"),
        remaining_output_tokens: parse_i64("anthropic-ratelimit-output-tokens-remaining"),
    }
}

/// Set all four gauges that the snapshot populates.
pub fn record_rate_limit_snapshot(
    provider: &'static str,
    snapshot: &RateLimitSnapshot,
    request_id: &str,
) {
    let registry = super::prometheus::registry();
    if let Some(v) = snapshot.remaining_requests {
        rate_limit_remaining_requests_gauge(registry)
            .with_label_values(&[provider])
            .set(v);
    }
    if let Some(v) = snapshot.remaining_tokens {
        rate_limit_remaining_tokens_gauge(registry)
            .with_label_values(&[provider])
            .set(v);
    }
    if let Some(v) = snapshot.remaining_input_tokens {
        rate_limit_remaining_input_tokens_gauge(registry)
            .with_label_values(&[provider])
            .set(v);
    }
    if let Some(v) = snapshot.remaining_output_tokens {
        rate_limit_remaining_output_tokens_gauge(registry)
            .with_label_values(&[provider])
            .set(v);
    }
    tracing::debug!(
        event = "metric_recorded",
        metric = "proxy_rate_limit_remaining_*",
        provider = provider,
        request_id = %request_id,
        remaining_requests = ?snapshot.remaining_requests,
        remaining_tokens = ?snapshot.remaining_tokens,
        remaining_input_tokens = ?snapshot.remaining_input_tokens,
        remaining_output_tokens = ?snapshot.remaining_output_tokens,
        "recorded proxy_rate_limit_remaining_* gauges"
    );
}

// ---------- proxy_service_tier_count_total{tier} ----------

pub fn service_tier_counter(registry: &Registry) -> &'static IntCounterVec {
    static COUNTER: OnceLock<IntCounterVec> = OnceLock::new();
    COUNTER.get_or_init(|| {
        let opts = Opts::new(
            METRIC_PROXY_SERVICE_TIER_COUNT_TOTAL,
            METRIC_PROXY_SERVICE_TIER_COUNT_TOTAL_HELP,
        );
        let counter = IntCounterVec::new(opts, &[LABEL_TIER])
            .expect("proxy_service_tier_count_total descriptor is well-formed");
        registry
            .register(Box::new(counter.clone()))
            .expect("proxy_service_tier_count_total registers exactly once");
        counter
    })
}

pub fn record_service_tier(tier: &str, request_id: &str) {
    service_tier_counter(super::prometheus::registry())
        .with_label_values(&[tier])
        .inc();
    tracing::debug!(
        event = "metric_recorded",
        metric = METRIC_PROXY_SERVICE_TIER_COUNT_TOTAL,
        tier = %tier,
        request_id = %request_id,
        "incremented proxy_service_tier_count_total"
    );
}

// ---------- proxy_response_status_count_total{status} ----------

pub fn response_status_counter(registry: &Registry) -> &'static IntCounterVec {
    static COUNTER: OnceLock<IntCounterVec> = OnceLock::new();
    COUNTER.get_or_init(|| {
        let opts = Opts::new(
            METRIC_PROXY_RESPONSE_STATUS_COUNT_TOTAL,
            METRIC_PROXY_RESPONSE_STATUS_COUNT_TOTAL_HELP,
        );
        let counter = IntCounterVec::new(opts, &[LABEL_STATUS])
            .expect("proxy_response_status_count_total descriptor is well-formed");
        registry
            .register(Box::new(counter.clone()))
            .expect("proxy_response_status_count_total registers exactly once");
        counter
    })
}

/// Record a Responses terminal status. `reason` is the
/// `incomplete_details.reason` field on `incomplete` responses (or
/// any other side-channel info worth pairing with the metric). It is
/// emitted in the structured log alongside the counter increment but
/// is NOT used as a label (would blow up cardinality).
pub fn record_response_status(status: &str, reason: Option<&str>, request_id: &str) {
    response_status_counter(super::prometheus::registry())
        .with_label_values(&[status])
        .inc();
    // Optional-3: aligned with the peer `record_*` helpers in this
    // module which all use `debug!` for the metric-correlation log
    // line. INFO was inconsistent and produced extra log volume
    // during normal Responses traffic.
    tracing::debug!(
        event = "metric_recorded",
        metric = METRIC_PROXY_RESPONSE_STATUS_COUNT_TOTAL,
        status = %status,
        reason = reason.unwrap_or(""),
        request_id = %request_id,
        "incremented proxy_response_status_count_total"
    );
}

// Phase G PR-G3 remediation (C3 + C4): the image-redacted counter
// and the wrap_rtk_invocations counter were originally registered
// here but neither had a production emit site that crossed the
// Python/Rust boundary. Both have moved Python-side
// (`headroom.proxy.request_logger::redactions_total` and
// `headroom.cli.wrap_rtk_metrics::rtk_invocation_counts`) and the
// Python proxy's `/metrics` exporter surfaces them — see
// `docs/observability.md` for the placement decision. Keeping a
// dead Rust counter would (a) violate the "no dead metrics
// registered" review finding and (b) mislead Phase H canary
// dashboards into expecting two scrape sources for what is really
// one Python-side counter.

#[cfg(test)]
mod tests {
    use super::*;
    use http::{HeaderMap, HeaderValue};

    #[test]
    fn extract_rate_limit_snapshot_anthropic() {
        let mut h = HeaderMap::new();
        h.insert(
            "anthropic-ratelimit-requests-remaining",
            HeaderValue::from_static("499"),
        );
        h.insert(
            "anthropic-ratelimit-tokens-remaining",
            HeaderValue::from_static("99000"),
        );
        h.insert(
            "anthropic-ratelimit-input-tokens-remaining",
            HeaderValue::from_static("80000"),
        );
        h.insert(
            "anthropic-ratelimit-output-tokens-remaining",
            HeaderValue::from_static("16000"),
        );
        let snap = extract_rate_limit_snapshot(&h);
        assert_eq!(snap.remaining_requests, Some(499));
        assert_eq!(snap.remaining_tokens, Some(99000));
        assert_eq!(snap.remaining_input_tokens, Some(80000));
        assert_eq!(snap.remaining_output_tokens, Some(16000));
    }

    #[test]
    fn extract_rate_limit_snapshot_openai() {
        let mut h = HeaderMap::new();
        h.insert(
            "x-ratelimit-remaining-requests",
            HeaderValue::from_static("1000"),
        );
        h.insert(
            "x-ratelimit-remaining-tokens",
            HeaderValue::from_static("250000"),
        );
        let snap = extract_rate_limit_snapshot(&h);
        assert_eq!(snap.remaining_requests, Some(1000));
        assert_eq!(snap.remaining_tokens, Some(250000));
        // OpenAI does not split input/output buckets.
        assert_eq!(snap.remaining_input_tokens, None);
        assert_eq!(snap.remaining_output_tokens, None);
    }

    #[test]
    fn extract_rate_limit_snapshot_no_headers() {
        let h = HeaderMap::new();
        let snap = extract_rate_limit_snapshot(&h);
        assert_eq!(snap.remaining_requests, None);
        assert_eq!(snap.remaining_tokens, None);
        assert_eq!(snap.remaining_input_tokens, None);
        assert_eq!(snap.remaining_output_tokens, None);
    }

    #[test]
    fn extract_rate_limit_snapshot_unparseable_value_is_none() {
        let mut h = HeaderMap::new();
        // Junk value — must not panic; must surface as None per
        // "no silent fallback" (the absence itself is the signal).
        h.insert(
            "anthropic-ratelimit-requests-remaining",
            HeaderValue::from_static("not-a-number"),
        );
        let snap = extract_rate_limit_snapshot(&h);
        assert_eq!(snap.remaining_requests, None);
    }
}
