//! headroom-core: foundation crate for the Rust port of Headroom.

pub mod auth_mode;
pub mod cache_control;
pub mod ccr;
pub mod compression_policy;
pub mod relevance;
pub mod signals;
pub mod tokenizer;
pub mod transforms;

// Re-exports for the live-zone dispatcher (Phase B PR-B2 consumes this).
// Hoisted to the crate root so the proxy crate gets one stable import
// path: `use headroom_core::compute_frozen_count;`. Keeping the
// `cache_control` module public too means downstream code can reach
// the helper types directly when needed.
pub use cache_control::compute_frozen_count;

/// Identity stub used by downstream crates and the Python binding to verify
/// linkage end-to-end.
pub fn hello() -> &'static str {
    "headroom-core"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hello_returns_crate_name() {
        assert_eq!(hello(), "headroom-core");
    }
}
