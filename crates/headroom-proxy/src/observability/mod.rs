//! Proxy observability surface — Phase D PR-D3.
//!
//! Centralises all Prometheus instrumentation in one place so that
//! metric names, label keys, and the global registry stay
//! co-located and discoverable. The Phase D acceptance criterion
//! (`Prometheus scrape includes Bedrock metrics`) demands a single
//! `/metrics` endpoint that serves the registry; that endpoint is
//! mounted by [`crate::proxy::build_app`] when the observability
//! module is in scope.
//!
//! # Module layout
//!
//! - [`prometheus`] — registry construction (lazy via `OnceLock`),
//!   Bedrock-scoped counters / histograms, and the `/metrics`
//!   text-format scrape handler. Per the realignment build
//!   constraint "elegant + scalable" we keep one module per
//!   concern; future Phase F / Phase H additions (auth-mode
//!   counters, OpenAI request totals) live alongside the
//!   Bedrock-prefixed ones below — never sprinkled across handlers.
//!
//! # Cardinality discipline
//!
//! Every label is bounded by infrastructure config, NOT by request
//! input. `model` comes from the axum path parameter (Bedrock vendor
//! prefix is enforced upstream of the metric increment); `region`
//! comes from `Config::bedrock_region`; `auth_mode` comes from the
//! `headroom_core::auth_mode::AuthMode` enum (3 variants total).
//! There is no path where a malicious client can drive label
//! cardinality unbounded — see `bedrock::invoke::handle_invoke` for
//! the call site.
//!
//! # Why not `metrics-rs`?
//!
//! `metrics-rs` is the more idiomatic Rust choice but it requires a
//! separate exporter binary. The Phase D scope is observability for
//! a single proxy binary; the simpler `prometheus` crate (with the
//! global default registry pinned in a `OnceLock`) keeps the
//! footprint small and the scrape endpoint trivial. Phase F may
//! revisit if multi-process aggregation lands.

pub mod cache_hit_rate;
pub mod compression_ratio;
pub mod metric_names;
pub mod prometheus;
pub mod proxy_metrics;

pub use prometheus::{
    handle_metrics, observe_bedrock_invoke_latency, record_bedrock_eventstream_message,
    record_bedrock_invoke,
};

// Phase G PR-G3 — re-export the canonical record_* helpers so call
// sites in `crate::sse`, `crate::handlers`, and `crate::proxy` get a
// flat `observability::record_x` import surface rather than a deep
// `observability::proxy_metrics::record_x` one. The deeper modules
// stay reachable for tests asserting on the metric vectors directly.
pub use cache_hit_rate::{
    compute_hit_rate as compute_cache_hit_rate, observe as observe_cache_hit_rate,
    provider as cache_hit_rate_provider,
};
pub use compression_ratio::{
    observe_ratio as observe_compression_ratio,
    record_rejected_by_token_check as record_compression_rejected_by_token_check,
};
pub use proxy_metrics::{
    extract_rate_limit_snapshot, record_passthrough_bytes_modified, record_rate_limit_snapshot,
    record_response_status, record_service_tier, RateLimitSnapshot,
};
