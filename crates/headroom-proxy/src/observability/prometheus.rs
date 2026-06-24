//! Prometheus instrumentation for the Bedrock route — Phase D PR-D3.
//!
//! # Registered metrics
//!
//! - `bedrock_invoke_count_total{model, region, auth_mode}` — Counter.
//!   One increment per `/model/{model}/invoke` (or `/converse` or
//!   `/invoke-with-response-stream`) request that successfully
//!   passed the path-parameter extractor and reached the handler
//!   body. Failures BEFORE the handler runs (router 404s, axum
//!   extractor errors) do not increment.
//! - `bedrock_invoke_latency_seconds{model, region}` — Histogram.
//!   Observed at request completion (whether the upstream call
//!   succeeded or returned 5xx). Buckets target typical Bedrock
//!   latencies (50ms → 60s) so p50/p99 land in distinct buckets at
//!   typical throughput. The `auth_mode` label is intentionally
//!   absent: the cost of cross-multiplying it with `model` would
//!   triple the per-model cardinality with little operator value
//!   (auth-mode breakdown lives in the count metric instead).
//! - `bedrock_eventstream_message_count_total{model, region, event_type}`
//!   — Counter. One increment per parsed binary EventStream message
//!   in the streaming handler. `event_type` is the
//!   `:event-type` header from the message (`chunk`, `metadata`,
//!   `internalServerException`, etc.). The set is bounded by
//!   AWS's documented event-type vocabulary, not customer input —
//!   see `crates/headroom-proxy/src/bedrock/eventstream.rs` for
//!   the parsed shape.
//!
//! # Wiring
//!
//! Every counter / histogram is created exactly once at first call
//! via `OnceLock`. Per-request work is `inc_with_label_values` /
//! `observe`, which is `O(1)` and lock-free in the common case
//! (the underlying `prometheus` crate uses a sharded RwLock per
//! metric vector). Total D3 overhead per request: a few hundred ns
//! plus one `Instant::elapsed()` for the latency histogram.
//!
//! # Logs paired with every increment
//!
//! Per the realignment build-constraint "comprehensive structured
//! logs", every metric increment in this module emits a
//! `tracing::debug!` with `event = "metric_recorded"` so operators
//! can correlate scrape values with log lines during incidents.
//! The cardinality of these debug logs is the same as the metric
//! itself (bounded by `model × region × auth_mode`), so leaving
//! them at `debug` level avoids per-request log volume in normal
//! operation while still being available under
//! `RUST_LOG=headroom_proxy::observability=debug`.

use std::sync::OnceLock;

use axum::body::Body;
use axum::http::{header, StatusCode};
use axum::response::Response;
use prometheus::{
    Encoder, HistogramOpts, HistogramVec, IntCounterVec, Opts, Registry, TextEncoder,
};

use headroom_core::auth_mode::AuthMode;

/// Latency-histogram buckets in seconds. Chosen to discriminate
/// across typical Bedrock latencies: cold-start (~1-2s), warm
/// streaming-start (~100-500ms), small completions (~50-200ms),
/// long completions (5-60s). Mirrors the bucket layout the AWS
/// CloudWatch sample dashboards use.
const LATENCY_BUCKETS_SECONDS: &[f64] = &[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0];

/// Lazy singleton registry. Borrowed by `handle_metrics` for
/// scrape rendering and by every metric registration helper.
///
/// Visibility note: `pub(super)` is intentional — the Phase G PR-G3
/// metric modules (`cache_hit_rate`, `compression_ratio`,
/// `proxy_metrics`) live alongside this one and reach for the shared
/// singleton at register time. External callers should NOT touch the
/// registry directly; they go through the per-metric `record_*`
/// helpers, which keeps registration centralised and emit sites
/// uniform.
pub(super) fn registry() -> &'static Registry {
    static REGISTRY: OnceLock<Registry> = OnceLock::new();
    REGISTRY.get_or_init(Registry::new)
}

/// `bedrock_invoke_count_total{model, region, auth_mode}` —
/// initialised on first call.
fn invoke_counter() -> &'static IntCounterVec {
    static COUNTER: OnceLock<IntCounterVec> = OnceLock::new();
    COUNTER.get_or_init(|| {
        let opts = Opts::new(
            "bedrock_invoke_count_total",
            "Total Bedrock invoke requests handled by the Rust proxy, \
             broken down by model id, AWS region, and inbound auth mode.",
        );
        let counter = IntCounterVec::new(opts, &["model", "region", "auth_mode"])
            .expect("bedrock_invoke_count_total descriptor is well-formed");
        registry()
            .register(Box::new(counter.clone()))
            .expect("bedrock_invoke_count_total registers exactly once");
        counter
    })
}

/// `bedrock_invoke_latency_seconds{model, region}` — initialised
/// on first call.
fn invoke_latency() -> &'static HistogramVec {
    static HIST: OnceLock<HistogramVec> = OnceLock::new();
    HIST.get_or_init(|| {
        let opts = HistogramOpts::new(
            "bedrock_invoke_latency_seconds",
            "Latency in seconds of Bedrock invoke requests as observed at the \
             Rust proxy entry boundary. Includes upstream call time plus any \
             pre-/post-compression and SigV4 signing.",
        )
        .buckets(LATENCY_BUCKETS_SECONDS.to_vec());
        let hist = HistogramVec::new(opts, &["model", "region"])
            .expect("bedrock_invoke_latency_seconds descriptor is well-formed");
        registry()
            .register(Box::new(hist.clone()))
            .expect("bedrock_invoke_latency_seconds registers exactly once");
        hist
    })
}

/// `bedrock_eventstream_message_count_total{model, region, event_type}`
/// — initialised on first call.
fn eventstream_counter() -> &'static IntCounterVec {
    static COUNTER: OnceLock<IntCounterVec> = OnceLock::new();
    COUNTER.get_or_init(|| {
        let opts = Opts::new(
            "bedrock_eventstream_message_count_total",
            "Total Bedrock binary EventStream messages parsed by the Rust proxy, \
             broken down by model id, AWS region, and the message's :event-type header \
             (chunk, metadata, modelStreamErrorException, etc.).",
        );
        let counter = IntCounterVec::new(opts, &["model", "region", "event_type"])
            .expect("bedrock_eventstream_message_count_total descriptor is well-formed");
        registry()
            .register(Box::new(counter.clone()))
            .expect("bedrock_eventstream_message_count_total registers exactly once");
        counter
    })
}

/// Record a single Bedrock invoke (non-streaming or streaming).
///
/// Pure increment + a paired `tracing::debug!` so operators can
/// correlate this metric with the request's structured log line via
/// the `request_id` field (callers thread that through).
pub fn record_bedrock_invoke(model: &str, region: &str, auth_mode: AuthMode) {
    invoke_counter()
        .with_label_values(&[model, region, auth_mode.as_str()])
        .inc();
    tracing::debug!(
        event = "metric_recorded",
        metric = "bedrock_invoke_count_total",
        model = %model,
        region = %region,
        auth_mode = auth_mode.as_str(),
        "incremented bedrock_invoke_count_total"
    );
}

/// Observe latency at the END of an invoke. The duration must be
/// computed by the caller via `Instant::elapsed()` — passing the
/// duration in (rather than the start time) keeps this helper
/// free of `Instant` types so unit tests can assert on synthetic
/// values.
pub fn observe_bedrock_invoke_latency(model: &str, region: &str, seconds: f64) {
    invoke_latency()
        .with_label_values(&[model, region])
        .observe(seconds);
    tracing::debug!(
        event = "metric_recorded",
        metric = "bedrock_invoke_latency_seconds",
        model = %model,
        region = %region,
        seconds = seconds,
        "observed bedrock_invoke_latency_seconds"
    );
}

/// Record a single parsed EventStream message in the streaming
/// path. The `event_type` argument is the `:event-type` header of
/// the message (or `unknown` when the message header was missing,
/// which itself is loud-logged at the call site).
pub fn record_bedrock_eventstream_message(model: &str, region: &str, event_type: &str) {
    eventstream_counter()
        .with_label_values(&[model, region, event_type])
        .inc();
    tracing::debug!(
        event = "metric_recorded",
        metric = "bedrock_eventstream_message_count_total",
        model = %model,
        region = %region,
        event_type = %event_type,
        "incremented bedrock_eventstream_message_count_total"
    );
}

/// Axum handler for `GET /metrics`. Renders the registry in the
/// Prometheus text format. Per Phase D acceptance: the scrape MUST
/// include each metric family as soon as it has been touched at
/// least once with a label set. The `prometheus` v0.13 crate skips
/// empty `MetricVec` families from `gather()`, so until a counter
/// has incremented (or a histogram has observed) once, neither its
/// HELP/TYPE nor any row appears — a documented quirk we lean on
/// for the "must stay 0" alarm-able `proxy_passthrough_bytes_modified_total`
/// surface: its absence FROM the scrape is itself the proof the
/// alarm is silent.
pub async fn handle_metrics() -> Response {
    // Force lazy registration so the HELP/TYPE descriptor lines
    // appear in the scrape even before any request has hit the
    // Bedrock route. Operators who curl /metrics on a fresh boot
    // see the metric names already advertised — surprises are
    // worse than the cost of three function calls.
    let _ = invoke_counter();
    let _ = invoke_latency();
    let _ = eventstream_counter();

    // Phase G PR-G3: same idea for the new proxy-wide metric families.
    // Lazy `OnceLock`-backed singletons; touching each here forces
    // registration on first scrape.
    //
    // H3 fix: registration alone is NOT enough — the prometheus
    // crate's v0.13 `gather()` skips empty MetricVec families
    // entirely (no HELP/TYPE lines either). Operators who curl
    // /metrics on a fresh boot would otherwise see NO trace of the
    // catalogue. We force-touch each counter / gauge MetricVec
    // with a sentinel `__init__` label so HELP/TYPE + a zero row
    // appear from boot and dashboards/alarms have a predictable
    // scrape shape. Counters with the `__init__` label increment
    // by 0, so the alarm-able "must stay 0" semantic of
    // `proxy_passthrough_bytes_modified_total` is preserved
    // (the family becomes visible, the rate stays 0).
    //
    // Histograms are NOT force-zeroed: a synthetic `observe(0.0)`
    // would pollute the per-label distribution. The two histogram
    // families (`proxy_cache_hit_rate_per_session` and
    // `proxy_compression_ratio_by_strategy`) only surface in the
    // scrape after the first real session, by design.
    let reg = registry();
    let _ = super::cache_hit_rate::histogram(reg);
    let _ = super::compression_ratio::ratio_histogram(reg);
    let rejected_counter = super::compression_ratio::rejected_counter(reg);
    let passthrough_counter = super::proxy_metrics::passthrough_bytes_modified_counter(reg);
    let rl_requests_gauge = super::proxy_metrics::rate_limit_remaining_requests_gauge(reg);
    let rl_tokens_gauge = super::proxy_metrics::rate_limit_remaining_tokens_gauge(reg);
    let rl_input_gauge = super::proxy_metrics::rate_limit_remaining_input_tokens_gauge(reg);
    let rl_output_gauge = super::proxy_metrics::rate_limit_remaining_output_tokens_gauge(reg);
    let tier_counter = super::proxy_metrics::service_tier_counter(reg);
    let status_counter = super::proxy_metrics::response_status_counter(reg);

    const INIT_SENTINEL: &str = "__init__";
    rejected_counter
        .with_label_values(&[INIT_SENTINEL])
        .inc_by(0);
    passthrough_counter
        .with_label_values(&[INIT_SENTINEL])
        .inc_by(0);
    rl_requests_gauge.with_label_values(&[INIT_SENTINEL]).set(0);
    rl_tokens_gauge.with_label_values(&[INIT_SENTINEL]).set(0);
    rl_input_gauge.with_label_values(&[INIT_SENTINEL]).set(0);
    rl_output_gauge.with_label_values(&[INIT_SENTINEL]).set(0);
    tier_counter.with_label_values(&[INIT_SENTINEL]).inc_by(0);
    status_counter.with_label_values(&[INIT_SENTINEL]).inc_by(0);

    let metric_families = registry().gather();
    let mut buffer = Vec::with_capacity(2048);
    let encoder = TextEncoder::new();
    if let Err(e) = encoder.encode(&metric_families, &mut buffer) {
        tracing::error!(
            event = "metrics_encode_failed",
            error = %e,
            "failed to encode Prometheus metrics scrape"
        );
        return Response::builder()
            .status(StatusCode::INTERNAL_SERVER_ERROR)
            .body(Body::from(format!("metrics encode error: {e}")))
            .expect("static error response");
    }
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, encoder.format_type())
        .body(Body::from(buffer))
        .unwrap_or_else(|e| {
            Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(Body::from(format!("metrics response build error: {e}")))
                .expect("static error response")
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: render the registry to a String for assertions.
    fn scrape() -> String {
        let mf = registry().gather();
        let encoder = TextEncoder::new();
        let mut buf = Vec::new();
        encoder.encode(&mf, &mut buf).expect("encode");
        String::from_utf8(buf).expect("utf8")
    }

    #[test]
    fn invoke_counter_advertises_metric_in_scrape() {
        // The `prometheus` crate only emits HELP/TYPE for vectors
        // that have AT LEAST ONE row (`gather()` skips empty
        // vectors), so we must fire one increment with a unique
        // label set this test owns to assert the metric family
        // appears in the scrape.
        invoke_counter()
            .with_label_values(&[
                "anthropic.unit-test-advertise-v1:0",
                "us-test-advertise-1",
                "oauth",
            ])
            .inc();
        let body = scrape();
        assert!(
            body.contains("bedrock_invoke_count_total"),
            "scrape missing bedrock_invoke_count_total: {body}"
        );
        assert!(
            body.contains("# TYPE bedrock_invoke_count_total counter"),
            "scrape missing TYPE line: {body}"
        );
    }

    #[test]
    fn invoke_increment_appears_with_labels() {
        record_bedrock_invoke(
            "anthropic.claude-3-haiku-20240307-v1:0",
            "us-east-1",
            AuthMode::OAuth,
        );
        let body = scrape();
        // The label-set rendering uses lexical ordering of label
        // names — auth_mode, model, region — so we assert on the
        // values without locking in the ordering of the columns.
        assert!(
            body.contains("auth_mode=\"oauth\""),
            "scrape missing auth_mode label: {body}"
        );
        assert!(
            body.contains("model=\"anthropic.claude-3-haiku-20240307-v1:0\""),
            "scrape missing model label: {body}"
        );
        assert!(
            body.contains("region=\"us-east-1\""),
            "scrape missing region label: {body}"
        );
    }

    #[test]
    fn latency_histogram_records_observation() {
        observe_bedrock_invoke_latency("anthropic.claude-3-haiku-20240307-v1:0", "us-east-1", 0.42);
        let body = scrape();
        assert!(
            body.contains("bedrock_invoke_latency_seconds_bucket"),
            "histogram bucket lines missing: {body}"
        );
        assert!(
            body.contains("bedrock_invoke_latency_seconds_sum"),
            "histogram sum line missing: {body}"
        );
        assert!(
            body.contains("bedrock_invoke_latency_seconds_count"),
            "histogram count line missing: {body}"
        );
    }

    #[test]
    fn eventstream_counter_records_event_type_label() {
        record_bedrock_eventstream_message(
            "anthropic.claude-3-haiku-20240307-v1:0",
            "us-east-1",
            "chunk",
        );
        record_bedrock_eventstream_message(
            "anthropic.claude-3-haiku-20240307-v1:0",
            "us-east-1",
            "metadata",
        );
        let body = scrape();
        assert!(
            body.contains("event_type=\"chunk\""),
            "scrape missing event_type=chunk: {body}"
        );
        assert!(
            body.contains("event_type=\"metadata\""),
            "scrape missing event_type=metadata: {body}"
        );
    }
}
