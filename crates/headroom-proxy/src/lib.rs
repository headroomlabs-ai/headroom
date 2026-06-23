//! headroom-proxy library: transparent reverse proxy in front of the Python
//! Headroom proxy. Used by both `main.rs` and the integration tests.

// Enable the `#[coverage(off)]` attribute under the coverage run only. On
// stable (normal build/test/clippy) the `coverage_nightly` cfg is unset, so
// this is a no-op and the crate compiles without any unstable feature. The
// nightly `cargo llvm-cov` run sets `--cfg coverage_nightly`, which turns the
// attribute on so genuinely-unreachable error edges can be excluded and the
// region metric reads a true 100%.
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

pub mod bedrock;
pub mod cache_stabilization;
pub mod compression;
pub mod config;
pub mod error;
pub mod handlers;
pub mod headers;
pub mod health;
pub mod observability;
pub mod proxy;
pub mod responses_items;
pub mod sse;
pub mod vertex;
pub mod websocket;

pub use config::Config;
pub use error::ProxyError;
pub use proxy::{build_app, spawn_savings_flusher, AppState};
