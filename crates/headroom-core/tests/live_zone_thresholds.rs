//! PR-B4 byte-threshold gate — integration tests.
//!
//! The dispatcher must skip compression entirely for blocks whose
//! content is below the per-content-type byte threshold (1 KiB for
//! JSON arrays). PR-B4 spec, `REALIGNMENT/04-phase-B-live-zone.md`.

use headroom_core::transforms::live_zone::DEFAULT_MODEL;
use headroom_core::transforms::{
    compress_anthropic_live_zone, AuthMode, BlockAction, LiveZoneOutcome,
};
use serde_json::{json, Value};

fn body_of(value: Value) -> Vec<u8> {
    serde_json::to_vec(&value).unwrap()
}

/// Build a body whose latest user-message tool_result carries `text`.
fn body_with_tool_result(text: &str) -> Vec<u8> {
    body_of(json!({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 64,
        "messages": [{
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_threshold_test",
                "content": text,
            }],
        }],
    }))
}

fn dispatch(body: &[u8]) -> LiveZoneOutcome {
    compress_anthropic_live_zone(body, 0, AuthMode::Payg, DEFAULT_MODEL)
        .expect("dispatcher returns Ok on valid bodies")
}

fn first_tool_result_action(out: &LiveZoneOutcome) -> BlockAction {
    let manifest = match out {
        LiveZoneOutcome::NoChange { manifest } => manifest,
        LiveZoneOutcome::Modified { manifest, .. } => manifest,
    };
    manifest
        .block_outcomes
        .iter()
        .find(|b| b.block_type == "tool_result")
        .expect("tool_result block present in manifest")
        .action
        .clone()
}

#[test]
fn below_threshold_no_compression_attempted() {
    // 200 bytes of homogeneous JSON dicts — well below the 1 KiB
    // JsonArray threshold. The dispatcher must record
    // `BelowByteThreshold` and emit `NoChange`; no compressor runs.
    let small_array: Vec<Value> = (0..3).map(|i| json!({"id": i, "v": "x"})).collect();
    let payload = serde_json::to_string(&small_array).unwrap();
    assert!(
        payload.len() < 1024,
        "fixture must stay below the 1 KiB JsonArray threshold; got {}",
        payload.len()
    );

    let body = body_with_tool_result(&payload);
    let out = dispatch(&body);

    // Sub-threshold means no rewriting → NoChange.
    assert!(
        matches!(out, LiveZoneOutcome::NoChange { .. }),
        "below-threshold input must not produce a Modified body"
    );

    let action = first_tool_result_action(&out);
    match action {
        BlockAction::BelowByteThreshold {
            content_type,
            byte_count,
            threshold_bytes,
        } => {
            assert_eq!(content_type, "json_array");
            assert_eq!(byte_count, payload.len());
            assert_eq!(threshold_bytes, 1024);
        }
        other => panic!("expected BelowByteThreshold for sub-threshold JSON, got {other:?}"),
    }
}

#[test]
fn above_threshold_compression_attempted() {
    // 10 KB of homogeneous JSON dicts — comfortably above the 1 KiB
    // JsonArray threshold and SmartCrusher's bread-and-butter shape,
    // so the dispatcher SHOULD route through `dispatch_compressor`.
    // Either `Compressed` (SmartCrusher shrunk it — the typical
    // outcome) or `RejectedNotSmaller` (token gate vetoed it) is
    // acceptable; both prove the dispatcher attempted compression
    // rather than short-circuiting on the byte threshold.
    let big_array: Vec<Value> = (0..400)
        .map(|i| {
            json!({
                "id": i,
                "kind": "row",
                "status": "ok",
                "value": format!("repeat-{}", i % 5),
            })
        })
        .collect();
    let payload = serde_json::to_string(&big_array).unwrap();
    assert!(
        payload.len() >= 10_000,
        "fixture must be >= 10 KB to clear the JsonArray threshold; got {}",
        payload.len()
    );

    let body = body_with_tool_result(&payload);
    let out = dispatch(&body);
    let action = first_tool_result_action(&out);

    match action {
        BlockAction::Compressed { .. } | BlockAction::RejectedNotSmaller { .. } => {
            // Either outcome proves the byte-threshold gate did
            // NOT short-circuit and a compressor actually ran.
        }
        other => panic!(
            "expected Compressed or RejectedNotSmaller after byte-threshold pass, got {other:?}"
        ),
    }
}
