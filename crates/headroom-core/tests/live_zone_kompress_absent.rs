//! Model-absent NoOp — dedicated integration test file.
//!
//! The Kompress slot (`crates/headroom-core/src/transforms/live_zone.rs`)
//! latches process-globally: the first PlainText dispatch starts its
//! background initialization, whose cache lookup reads `HF_HUB_CACHE` /
//! `HF_HOME` / `HOME` / `USERPROFILE`. Environment variables are
//! process-global too, so forcing those four to a cold, empty cache dir
//! must happen in a test process that does nothing else — any other test
//! in the same binary that dispatched PlainText content first would
//! settle the slot against the *ambient* environment instead. That is
//! why this file holds exactly ONE `#[test]` fn.
//!
//! Do NOT add more tests to this file, and do NOT move this test into
//! `live_zone_dispatch.rs` (see
//! `plain_text_routes_to_kompress_when_model_cached` there, which relies
//! on the loader observing the *ambient* cache).
//!
//! On `set_var`: see the note in `live_zone_disable_arms.rs`. The same
//! caveat applies, for the same reason — the input under test is the
//! environment. An injectable cache root on `Kompress::from_cache` would
//! remove the need for this file; see the PR description.

mod common;

use common::{body_with_tool_result, dispatch, plain_prose, tool_result_action};
use headroom_core::transforms::{BlockAction, LiveZoneOutcome};

#[test]
fn plain_text_model_absent_is_deterministic_no_op() {
    // Force every cache root the loader consults to a fresh, empty temp
    // dir so the lookup deterministically misses, rather than picking up
    // whatever happens to be in the real user cache on this machine.
    let cold_dir = tempfile::tempdir().expect("create fresh temp dir for cold model cache");
    let cold_path = cold_dir.path().to_str().expect("temp dir path is UTF-8");
    for var in ["HF_HUB_CACHE", "HF_HOME", "HOME", "USERPROFILE"] {
        std::env::set_var(var, cold_path);
    }

    // > 5120 bytes so the PlainText byte threshold is cleared and the
    // dispatcher actually attempts the arm rather than short-circuiting
    // at `BelowByteThreshold`.
    let prose = plain_prose(5_200);
    assert!(
        prose.len() > 5120,
        "fixture must clear the PlainText byte threshold (5120); got {} bytes",
        prose.len()
    );

    let body = body_with_tool_result(&prose).0;
    let assert_no_op = |label: &str| {
        let out = dispatch(&body);
        let manifest = match &out {
            LiveZoneOutcome::NoChange { manifest } => manifest,
            LiveZoneOutcome::Modified { manifest, .. } => panic!(
                "cache-cold Kompress must not rewrite bytes ({label}); got Modified. \
                 manifest: {manifest:?}"
            ),
        };
        let action = tool_result_action(manifest);
        assert!(
            matches!(action, BlockAction::NoCompressionApplied { .. }),
            "cache-cold Kompress must degrade to a deterministic NoOp ({label}), \
             not an error: {action:?}"
        );
    };

    // First dispatch: a NoOp by the non-blocking contract — this call is
    // what starts the slot's background initialization.
    assert_no_op("first dispatch, slot unsettled");

    // Settle the slot off the request path and pin WHY it is empty: the
    // loader found nothing under the forced-cold roots.
    assert!(
        !headroom_core::transforms::live_zone::warm_live_zone_compressors(),
        "the loader reported a model ready under a forced-cold cache root"
    );

    // A post-settle dispatch is the same deterministic NoOp.
    assert_no_op("post-settle dispatch");
}
