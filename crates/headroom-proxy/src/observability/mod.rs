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

pub mod prometheus;

pub use prometheus::{
    handle_metrics, observe_bedrock_invoke_latency, record_bedrock_eventstream_message,
    record_bedrock_invoke,
};
