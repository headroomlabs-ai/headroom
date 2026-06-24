//! Centralised metric-name + label-key constants — Phase G PR-G3.
//!
//! Realignment build-constraint "configurable: every metric name + label
//! vocabulary defined in one place" applies here. The Bedrock D3
//! metrics in [`super::prometheus`] predate this module; they keep
//! their inline literals (a churn-cost decision documented in the
//! PR-G3 commit) but every PR-G3 metric and its labels live here.
//!
//! # Naming convention
//!
//! - `METRIC_*` — the wire-name string. Prometheus convention:
//!   `_total` for counters, `_seconds` / no suffix for histograms.
//! - `METRIC_*_HELP` — the HELP-line text used at registration. Kept
//!   alongside the wire name so a rename catches both in one diff.
//! - `LABEL_*` — the label-key string. Reuse across metrics where
//!   the dimension is the same (`provider`, `strategy`, …).

// ---------- proxy_cache_hit_rate_per_session ----------

pub const METRIC_PROXY_CACHE_HIT_RATE_PER_SESSION: &str = "proxy_cache_hit_rate_per_session";
pub const METRIC_PROXY_CACHE_HIT_RATE_PER_SESSION_HELP: &str =
    "Per-session cache hit rate observed at the Rust proxy from \
     usage.cache_read_input_tokens / (input + cache_read + cache_creation). \
     Phase H canary gate: parity with the Python proxy baseline.";

// ---------- proxy_compression_ratio_by_strategy ----------

pub const METRIC_PROXY_COMPRESSION_RATIO_BY_STRATEGY: &str = "proxy_compression_ratio_by_strategy";
pub const METRIC_PROXY_COMPRESSION_RATIO_BY_STRATEGY_HELP: &str =
    "Compression ratio (compressed_tokens / original_tokens) observed \
     per block that was actually shrunk by the live-zone dispatcher. \
     Labelled by strategy (smart_crusher/log_compressor/…) and \
     detected content_type.";

// ---------- proxy_compression_rejected_by_token_check_total ----------

pub const METRIC_PROXY_COMPRESSION_REJECTED_BY_TOKEN_CHECK_TOTAL: &str =
    "proxy_compression_rejected_by_token_check_total";
pub const METRIC_PROXY_COMPRESSION_REJECTED_BY_TOKEN_CHECK_TOTAL_HELP: &str =
    "Count of compressor runs whose output failed the tokenizer-validated \
     shrink check (compressed_tokens >= original_tokens). Surfaces 'we ran \
     but kept the original' cases that would otherwise be invisible.";

// ---------- proxy_passthrough_bytes_modified_total ----------

pub const METRIC_PROXY_PASSTHROUGH_BYTES_MODIFIED_TOTAL: &str =
    "proxy_passthrough_bytes_modified_total";
pub const METRIC_PROXY_PASSTHROUGH_BYTES_MODIFIED_TOTAL_HELP: &str =
    "Bytes modified on a path that is supposed to passthrough verbatim. \
     MUST stay 0 outside the compression-on hot path. Any non-zero rate \
     fires the cache-safety alarm.";

// ---------- proxy_rate_limit_remaining_* ----------

pub const METRIC_PROXY_RATE_LIMIT_REMAINING_REQUESTS: &str = "proxy_rate_limit_remaining_requests";
pub const METRIC_PROXY_RATE_LIMIT_REMAINING_REQUESTS_HELP: &str =
    "Upstream-reported remaining requests for the current window, \
     extracted from rate-limit response headers (anthropic-ratelimit-* \
     or x-ratelimit-*). Per-provider, per-window-bucket gauge.";

pub const METRIC_PROXY_RATE_LIMIT_REMAINING_TOKENS: &str = "proxy_rate_limit_remaining_tokens";
pub const METRIC_PROXY_RATE_LIMIT_REMAINING_TOKENS_HELP: &str =
    "Upstream-reported remaining tokens for the current window, extracted \
     from rate-limit response headers (anthropic-ratelimit-*-tokens or \
     x-ratelimit-remaining-tokens).";

pub const METRIC_PROXY_RATE_LIMIT_REMAINING_INPUT_TOKENS: &str =
    "proxy_rate_limit_remaining_input_tokens";
pub const METRIC_PROXY_RATE_LIMIT_REMAINING_INPUT_TOKENS_HELP: &str =
    "Upstream-reported remaining INPUT tokens for the current window. \
     Anthropic separates input and output token budgets in its \
     ratelimit headers; this gauge tracks the input bucket.";

pub const METRIC_PROXY_RATE_LIMIT_REMAINING_OUTPUT_TOKENS: &str =
    "proxy_rate_limit_remaining_output_tokens";
pub const METRIC_PROXY_RATE_LIMIT_REMAINING_OUTPUT_TOKENS_HELP: &str =
    "Upstream-reported remaining OUTPUT tokens for the current window. \
     Anthropic-only header on present providers.";

// ---------- proxy_service_tier_count_total ----------

pub const METRIC_PROXY_SERVICE_TIER_COUNT_TOTAL: &str = "proxy_service_tier_count_total";
pub const METRIC_PROXY_SERVICE_TIER_COUNT_TOTAL_HELP: &str =
    "Count of requests/responses observed at the proxy, labelled by the \
     OpenAI Responses service_tier the request resolved into (auto, \
     default, flex, on_demand, priority).";

// ---------- proxy_response_status_count_total ----------

pub const METRIC_PROXY_RESPONSE_STATUS_COUNT_TOTAL: &str = "proxy_response_status_count_total";
pub const METRIC_PROXY_RESPONSE_STATUS_COUNT_TOTAL_HELP: &str =
    "Count of OpenAI Responses outcomes labelled by terminal status \
     (completed, incomplete, failed, cancelled, in_progress). \
     'incomplete' detail lands in the structured log paired with each \
     increment.";

// Phase G PR-G3 remediation (C3 + C4): the metric-name constants
// for `proxy_image_generation_call_log_redacted_total`,
// `wrap_rtk_invocations_total`, and `wrap_rtk_tokens_saved_per_session`
// were removed because the underlying counters had no production
// emit site on the Rust side. The same metrics are exported by the
// Python proxy (`headroom/proxy/prometheus_metrics.py`) which is the
// natural owner: image redaction is a Python-proxy operation and RTK
// invocation tracking lives in the wrap CLI, both Python-side
// surfaces. See `docs/observability.md`.

// ---------- shared label keys ----------

pub const LABEL_PROVIDER: &str = "provider";
pub const LABEL_STRATEGY: &str = "strategy";
pub const LABEL_CONTENT_TYPE: &str = "content_type";
pub const LABEL_PATH: &str = "path";
pub const LABEL_TIER: &str = "tier";
pub const LABEL_STATUS: &str = "status";

// ---------- bounded label vocabularies ----------

/// OpenAI service-tier values per the Responses API spec
/// (`service_tier` field on the response object). The metric label
/// vocabulary is **strictly** this set plus a `"scale"` value
/// (documented in OpenAI's tier-pricing page) and a sentinel
/// `"other"` bucket for anything else, so a malicious client posting
/// `{"service_tier":"<random>"}` per request cannot blow up
/// cardinality.
pub mod service_tier {
    pub const AUTO: &str = "auto";
    pub const DEFAULT: &str = "default";
    pub const FLEX: &str = "flex";
    pub const ON_DEMAND: &str = "on_demand";
    pub const PRIORITY: &str = "priority";
    pub const SCALE: &str = "scale";
    /// Sentinel for any unknown / unrecognised tier value. Prevents
    /// label-cardinality DoS from arbitrary inbound JSON.
    pub const OTHER: &str = "other";

    /// Validate an inbound `service_tier` string against the bounded
    /// vocabulary. Returns the matching `&'static` constant or
    /// [`OTHER`] for any unrecognised value (with a tracing::warn so
    /// wire-format drift is loud rather than silently bucketed).
    ///
    /// The matching is case-sensitive — the OpenAI spec is
    /// case-sensitive on these strings; a case-different value is
    /// treated as drift, not as the same tier.
    pub fn validate(raw: &str) -> &'static str {
        match raw {
            AUTO => AUTO,
            DEFAULT => DEFAULT,
            FLEX => FLEX,
            ON_DEMAND => ON_DEMAND,
            PRIORITY => PRIORITY,
            SCALE => SCALE,
            _ => {
                tracing::warn!(
                    event = "service_tier_unknown",
                    raw = %raw,
                    bucket = OTHER,
                    "unknown service_tier value bucketed to 'other' to bound cardinality"
                );
                OTHER
            }
        }
    }
}

/// OpenAI Responses terminal-status vocabulary. `in_progress` is the
/// non-terminal entry — included so observers see a request that
/// closed mid-stream (we increment on the last status seen).
pub mod response_status {
    pub const COMPLETED: &str = "completed";
    pub const INCOMPLETE: &str = "incomplete";
    pub const FAILED: &str = "failed";
    pub const CANCELLED: &str = "cancelled";
    pub const IN_PROGRESS: &str = "in_progress";
}
