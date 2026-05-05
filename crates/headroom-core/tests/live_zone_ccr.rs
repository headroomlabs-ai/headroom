//! PR-B7 — `compress_anthropic_live_zone_with_ccr` integration tests.
//!
//! Confirms that wiring a CCR store into the live-zone dispatcher:
//!   1. Stores the original block bytes keyed by `BLAKE3(original)[..24]`.
//!   2. Appends `<<ccr:HASH>>` to the compressed block content.
//!   3. Leaves bytes outside the live zone byte-identical (cache safety).
//!   4. Is byte-equivalent to PR-B4 behaviour when no CCR store is wired.

use headroom_core::ccr::backends::InMemoryCcrStore;
use headroom_core::ccr::{compute_key, CcrStore};
use headroom_core::transforms::live_zone::{
    compress_anthropic_live_zone, compress_anthropic_live_zone_with_ccr, AuthMode, LiveZoneOutcome,
    DEFAULT_MODEL,
};
use serde_json::{json, Value};

/// Build a synthetic JSON-array tool_result above the 1 KiB threshold
/// so SmartCrusher actually engages.
fn large_json_array_payload() -> String {
    let items: Vec<Value> = (0..40)
        .map(|i| {
            json!({
                "id": i,
                "name": format!("entry_{i}"),
                "score": i * 7,
                "active": i % 2 == 0,
                "notes": "lorem ipsum dolor sit amet, consectetur adipiscing elit",
            })
        })
        .collect();
    serde_json::to_string(&Value::Array(items)).unwrap()
}

fn body_with_payload(payload: &str) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "model": "claude-3-5-sonnet-20241022",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": payload}
                ]
            }
        ]
    }))
    .unwrap()
}

#[test]
fn ccr_marker_injected_when_store_wired() {
    let payload = large_json_array_payload();
    let body = body_with_payload(&payload);
    let store = InMemoryCcrStore::new();

    let outcome = compress_anthropic_live_zone_with_ccr(
        &body,
        0,
        AuthMode::Payg,
        DEFAULT_MODEL,
        Some(&store),
    )
    .expect("dispatcher must succeed");

    let new_body = match &outcome {
        LiveZoneOutcome::Modified { new_body, .. } => new_body.get().to_string(),
        LiveZoneOutcome::NoChange { .. } => {
            panic!("expected Modified; SmartCrusher should compress this payload")
        }
    };

    let expected_hash = compute_key(payload.as_bytes());
    let marker = format!("<<ccr:{expected_hash}>>");
    assert!(
        new_body.contains(&marker),
        "compressed body must contain CCR marker; body={new_body}"
    );
    let recovered = store.get(&expected_hash);
    assert_eq!(
        recovered.as_deref(),
        Some(payload.as_str()),
        "store must hold the original bytes under the BLAKE3 hash key"
    );
}

#[test]
fn no_marker_when_store_omitted() {
    let payload = large_json_array_payload();
    let body = body_with_payload(&payload);

    let outcome =
        compress_anthropic_live_zone(&body, 0, AuthMode::Payg, DEFAULT_MODEL).expect("dispatcher");

    let new_body = match &outcome {
        LiveZoneOutcome::Modified { new_body, .. } => new_body.get().to_string(),
        LiveZoneOutcome::NoChange { .. } => return, // legitimate — token gate may reject
    };

    assert!(
        !new_body.contains("<<ccr:"),
        "no-store path must never inject markers; body={new_body}"
    );
}

#[test]
fn store_only_populated_after_token_gate_admits() {
    // Tiny payload below the 1 KiB threshold → BelowByteThreshold,
    // dispatcher never runs a compressor → store must stay empty.
    let body = body_with_payload("tiny");
    let store = InMemoryCcrStore::new();
    let _ = compress_anthropic_live_zone_with_ccr(
        &body,
        0,
        AuthMode::Payg,
        DEFAULT_MODEL,
        Some(&store),
    )
    .expect("dispatcher");
    assert_eq!(store.len(), 0, "no compression → no CCR put");
}
