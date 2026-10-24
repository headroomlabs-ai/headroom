//! Env-var kill-switch for the live-zone dispatch arms — dedicated
//! integration test file.
//!
//! `disabled_arms` (`crates/headroom-core/src/transforms/live_zone.rs`)
//! reads `HEADROOM_LIVE_ZONE_DISABLE_ARMS` on first dispatch and latches
//! the parsed set behind a process-global `OnceLock` (the determinism
//! invariant — a run's arm-disable set cannot change mid-flight).
//! Environment variables are process-global too, so setting the variable
//! must happen in a test process that does nothing else: any other test
//! in the same binary that dispatched first would freeze the `OnceLock`
//! against whatever it saw at that point. That is why this file holds
//! exactly ONE `#[test]` fn — a dedicated file is a dedicated test
//! binary, so no other test can race the initialization.
//!
//! Do NOT add more tests to this file.
//!
//! **On `set_var`.** `std::env::set_var` is unsound in a multi-threaded
//! process (it races any concurrent reader of the environment, including
//! ones inside libc) and becomes `unsafe` in edition 2024. It is used
//! here because the switch's input genuinely *is* the environment and
//! this binary is single-threaded at the point of the call. The parsing
//! contract itself — aliases, trimming, unknown tokens — is covered
//! without any environment mutation by
//! `live_zone_disable_arms_parsing.rs` against the pure
//! `parse_disabled_arms`. Threading an explicit config value into the
//! dispatcher would remove the need for this file entirely; see the PR
//! description.

mod common;

use common::{
    body_with_tool_result, dispatch, json_array_of_dicts, kompress_available, plain_prose,
    python_module_source, tool_result_action,
};
use headroom_core::transforms::{BlockAction, LiveZoneOutcome};

/// Assert the single `tool_result` block was left uncompressed.
fn assert_no_compression(label: &str, body: &[u8]) {
    let out = dispatch(body);
    let manifest = match &out {
        LiveZoneOutcome::NoChange { manifest } => manifest,
        LiveZoneOutcome::Modified { manifest, .. } => {
            panic!(
                "disabled {label} arm must not rewrite bytes; got Modified. manifest: {manifest:?}"
            )
        }
    };
    let action = tool_result_action(manifest);
    assert!(
        matches!(action, BlockAction::NoCompressionApplied { .. }),
        "disabled {label} arm must yield NoCompressionApplied, got {action:?}"
    );
}

#[test]
fn disabled_arms_route_to_no_op_others_unaffected() {
    // `source_code, plain_text, json_array, bogus_type` — the internal
    // spaces exercise the trim path and `bogus_type` the unknown-token
    // branch (logged and ignored, never a panic).
    std::env::set_var(
        "HEADROOM_LIVE_ZONE_DISABLE_ARMS",
        "source_code, plain_text, json_array, bogus_type",
    );

    // (a) SourceCode disabled: a >2048-byte Python tool_result must not
    // reach the CodeAwareCompressor. Deterministic on every host — the
    // code arm has no model-cache dependency.
    let code = python_module_source(10);
    assert!(
        code.len() > 2048,
        "fixture must clear the SourceCode byte threshold (2048); got {} bytes",
        code.len()
    );
    assert_no_compression("SourceCode", &body_with_tool_result(&code).0);

    // (b) JsonArray disabled: SmartCrusher must not fire on a shape it
    // would otherwise compress on any host. This is the case that
    // discriminates the kill switch unconditionally — unlike (c), it
    // cannot be satisfied by an absent model — and it also pins that the
    // switch is generic across arms rather than special-cased to the two
    // this PR wires.
    let payload = json_array_of_dicts(200);
    assert!(
        payload.len() > 512,
        "fixture must clear the JsonArray byte threshold (512); got {} bytes",
        payload.len()
    );
    assert_no_compression("JsonArray", &body_with_tool_result(&payload).0);

    // (c) PlainText disabled: a >5120-byte prose tool_result must not
    // reach Kompress. Two vacuity traps, not one:
    //
    // 1. The fixture MUST clear THRESHOLD_PLAIN_TEXT (5120) or the
    //    byte-threshold gate short-circuits before dispatch and the
    //    assertion passes regardless of the switch.
    // 2. On a cold HuggingFace cache the enabled arm's fall-open path
    //    returns the identical no-op, so this cannot discriminate there.
    //    `kompress_available()` says which case actually ran; (b) above
    //    carries the unconditional coverage.
    let prose = plain_prose(5_200);
    assert!(
        prose.len() > 5120,
        "fixture must clear the PlainText byte threshold (5120); got {} bytes",
        prose.len()
    );
    assert_no_compression("PlainText", &body_with_tool_result(&prose).0);
    if !kompress_available() {
        eprintln!(
            "NOTE: cold model cache — the PlainText case passed but could not discriminate \
             the kill switch from an absent model this run. Unconditional coverage comes \
             from the JsonArray case above."
        );
    }

    // (d) An arm NOT named in the list still compresses: build output
    // routes to LogCompressor as usual, proving the switch is selective
    // rather than a global off.
    let logs = "\
ERROR src/main.rs:42 connection refused
WARN  src/pool.rs:17 retrying in 250ms
ERROR src/main.rs:42 connection refused
WARN  src/pool.rs:17 retrying in 500ms
"
    .repeat(40);
    let out = dispatch(&body_with_tool_result(&logs).0);
    let manifest = match &out {
        LiveZoneOutcome::Modified { manifest, .. } => manifest,
        LiveZoneOutcome::NoChange { manifest } => panic!(
            "an arm absent from the disable list must be unaffected; got NoChange. \
             manifest: {manifest:?}"
        ),
    };
    match tool_result_action(manifest) {
        BlockAction::Compressed { strategy, .. } => assert_eq!(
            strategy, "log_compressor",
            "expected LogCompressor dispatch, unaffected by the disabled arms"
        ),
        other => panic!("expected BlockAction::Compressed via log_compressor, got {other:?}"),
    }
}
