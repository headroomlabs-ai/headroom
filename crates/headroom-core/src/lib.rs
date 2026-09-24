//! headroom-core: foundation crate for the Rust port of Headroom.

pub mod auth_mode;
pub mod cache_control;
pub mod ccr;
pub mod compression_policy;
pub mod offline;
#[cfg(feature = "ml")]
mod onnx_cpu;
pub mod relevance;
pub mod rollout;
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
pub(crate) mod test_support {
    use std::sync::{Mutex, MutexGuard};

    /// The process environment is global while `cargo test` runs tests in
    /// parallel threads inside one process, so a test that sets an env var can
    /// be observed by an unrelated sibling mid-assertion. Every test that
    /// mutates the environment holds this for its whole body. Same shape as
    /// `REGISTRY_LOCK` in `tokenizer::registry`.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Recovers from poisoning on purpose: one panicking test should not
    /// cascade into every later test that touches the environment.
    pub(crate) fn env_lock() -> MutexGuard<'static, ()> {
        ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hello_returns_crate_name() {
        assert_eq!(hello(), "headroom-core");
    }
}
