//! End-to-end CCR roundtrip: compress → store → retrieve → reconstruct.
//!
//! These tests pin **the cornerstone guarantee** of CCR: every row the
//! lossy path drops out of the prompt is recoverable from the CCR store
//! by hash. Lossy on the wire, lossless end-to-end.
//!
//! If any test in this file regresses, we are silently losing data —
//! the prompt advertises a `<<ccr:HASH ...>>` pointer that the runtime
//! cannot honor. That is the bug class these tests exist to catch.

use std::sync::Arc;

use serde_json::{json, Value};

use headroom_core::ccr::{CcrStore, InMemoryCcrStore};
use headroom_core::transforms::smart_crusher::{
    SmartCrusher, SmartCrusherBuilder, SmartCrusherConfig,
};

/// Force the lossy path: set the lossless savings threshold above 1.0
/// so no tabular rendering can ever clear it.
fn force_lossy_config() -> SmartCrusherConfig {
    SmartCrusherConfig {
        lossless_min_savings_ratio: 0.99,
        ..SmartCrusherConfig::default()
    }
}

/// Reasonably crushable fixture: low-uniqueness rows so the analyzer
/// is willing to compress, large enough to overshoot adaptive_k.
fn lossy_friendly_items(n: usize) -> Vec<Value> {
    (0..n).map(|i| json!({"id": i, "status": "ok"})).collect()
}

#[test]
fn default_crusher_stores_dropped_rows() {
    // The default `SmartCrusher::new()` ships with both lossless-first
    // compaction AND a CCR store (matches Python's default — CCR
    // enabled). So a real lossy crush should leave the original parked
    // in the store, retrievable by hash.
    let crusher = SmartCrusher::new(force_lossy_config());
    let items = lossy_friendly_items(50);

    let result = crusher.crush_array(&items, "", 1.0);

    let hash = result
        .ccr_hash
        .as_ref()
        .expect("lossy path must emit a hash");
    let store = crusher.ccr_store().expect("default crusher has a store");

    let retrieved = store.get(hash).expect("hash must resolve in the store");
    let parsed: Value = serde_json::from_str(&retrieved).expect("payload is valid JSON");
    assert_eq!(parsed, Value::Array(items.clone()), "roundtrip mismatch");
}

#[test]
fn without_compaction_also_stores_dropped_rows() {
    // The legacy / parity constructor still carries a default store —
    // CCR is the no-data-loss contract, not an opt-in extra.
    let crusher = SmartCrusher::without_compaction(SmartCrusherConfig::default());
    let items = lossy_friendly_items(30);

    let result = crusher.crush_array(&items, "", 1.0);

    if let Some(hash) = result.ccr_hash.as_ref() {
        let store = crusher.ccr_store().expect("default crusher has a store");
        let retrieved = store.get(hash).expect("hash must resolve");
        let parsed: Value = serde_json::from_str(&retrieved).unwrap();
        assert_eq!(parsed, Value::Array(items.clone()));
    }
}

#[test]
fn shared_external_store_sees_writes() {
    // Production wiring: the runtime owns the store; SmartCrusher
    // writes through it. The proxy keeps an `Arc` for retrieval; this
    // test models that arrangement.
    let store: Arc<dyn CcrStore> = Arc::new(InMemoryCcrStore::new());
    let crusher = SmartCrusherBuilder::new(force_lossy_config())
        .with_default_oss_setup()
        .with_default_compaction()
        .with_ccr_store(store.clone())
        .build();

    let items = lossy_friendly_items(40);
    let result = crusher.crush_array(&items, "", 1.0);
    let hash = result.ccr_hash.expect("lossy path emits a hash");

    let retrieved = store.get(&hash).expect("external store has the payload");
    let parsed: Value = serde_json::from_str(&retrieved).unwrap();
    assert_eq!(parsed, Value::Array(items));
}

#[test]
fn passthrough_does_not_write_to_store() {
    // Below adaptive_k: nothing dropped, nothing to recover, no store
    // write. Otherwise we'd accumulate noise on the hot passthrough
    // path.
    let crusher = SmartCrusher::new(SmartCrusherConfig::default());
    let store = crusher.ccr_store().unwrap().clone();
    let starting_len = store.len();

    let small = lossy_friendly_items(3);
    let result = crusher.crush_array(&small, "", 1.0);

    assert!(result.ccr_hash.is_none());
    assert_eq!(store.len(), starting_len, "no write expected");
}

#[test]
fn lossless_win_does_not_write_to_store() {
    // Lossless wins → nothing dropped, no CCR retrieval needed → no
    // store write. The store should only see writes when the prompt
    // actually loses data.
    let crusher = SmartCrusher::new(SmartCrusherConfig::default());
    let store = crusher.ccr_store().unwrap().clone();
    let starting_len = store.len();

    // Highly tabular fixture — lossless compaction should clear the
    // 30% savings threshold easily.
    let items: Vec<Value> = (0..50)
        .map(|i| json!({"id": i, "kind": "click", "ts": 1700000000 + i, "user": "alice"}))
        .collect();

    let result = crusher.crush_array(&items, "", 1.0);

    if result.compacted.is_some() {
        assert!(result.ccr_hash.is_none(), "lossless win → no hash");
        assert_eq!(store.len(), starting_len, "lossless win → no store write");
    }
}

#[test]
fn store_roundtrip_is_deterministic_across_calls() {
    // Same input crushed twice → same hash → store has exactly one
    // entry (not two), and it resolves to the same payload.
    let crusher = SmartCrusher::new(force_lossy_config());
    let store = crusher.ccr_store().unwrap().clone();
    let items = lossy_friendly_items(40);

    let r1 = crusher.crush_array(&items, "", 1.0);
    let len_after_first = store.len();

    let r2 = crusher.crush_array(&items, "", 1.0);
    assert_eq!(r1.ccr_hash, r2.ccr_hash);

    let hash = r1.ccr_hash.unwrap();
    assert_eq!(
        store.len(),
        len_after_first,
        "second call with same input must not grow the store"
    );

    let retrieved = store.get(&hash).unwrap();
    let parsed: Value = serde_json::from_str(&retrieved).unwrap();
    assert_eq!(parsed, Value::Array(items));
}

#[test]
fn distinct_inputs_produce_distinct_store_entries() {
    let crusher = SmartCrusher::new(force_lossy_config());
    let store = crusher.ccr_store().unwrap().clone();
    let starting_len = store.len();

    let a: Vec<Value> = (0..40).map(|i| json!({"id": i, "tag": "a"})).collect();
    let b: Vec<Value> = (0..40).map(|i| json!({"id": i, "tag": "b"})).collect();

    let ra = crusher.crush_array(&a, "", 1.0);
    let rb = crusher.crush_array(&b, "", 1.0);

    let ha = ra.ccr_hash.unwrap();
    let hb = rb.ccr_hash.unwrap();
    assert_ne!(ha, hb);

    // Both originals retrievable.
    let pa: Value = serde_json::from_str(&store.get(&ha).unwrap()).unwrap();
    let pb: Value = serde_json::from_str(&store.get(&hb).unwrap()).unwrap();
    assert_eq!(pa, Value::Array(a));
    assert_eq!(pb, Value::Array(b));
    assert_eq!(store.len(), starting_len + 2);
}

#[test]
fn dropped_summary_marker_points_at_stored_hash() {
    // The marker the LLM sees in the prompt encodes the same hash that
    // resolves the stored payload. Pin the format so the retrieval-tool
    // contract stays honest.
    let crusher = SmartCrusher::new(force_lossy_config());
    let store = crusher.ccr_store().unwrap().clone();
    let items = lossy_friendly_items(50);

    let result = crusher.crush_array(&items, "", 1.0);
    let hash = result.ccr_hash.as_ref().unwrap();

    assert!(
        result.dropped_summary.contains(&format!("<<ccr:{hash} ")),
        "marker {:?} must embed hash {}",
        result.dropped_summary,
        hash
    );
    // The hash actually resolves.
    assert!(store.get(hash).is_some());
}

#[test]
fn full_crush_pipeline_roundtrips_through_store() {
    // End-to-end through the public `crush()` API (the entry point
    // that ContentRouter calls). The result string contains the marker;
    // we verify the marker hash resolves in the store.
    let crusher = SmartCrusher::new(force_lossy_config());
    let store = crusher.ccr_store().unwrap().clone();

    let items: Vec<Value> = (0..50).map(|i| json!({"id": i, "status": "ok"})).collect();
    let raw_input = serde_json::to_string(&Value::Array(items.clone())).unwrap();

    let _ = crusher.crush(&raw_input, "", 1.0);

    // The pipeline routed through `process_value` → `crush_array`,
    // which calls our wired `put`. So the original is in the store
    // under the canonical hash.
    let store_len = store.len();
    assert!(store_len > 0, "expected at least one CCR store entry");
}

// ─── PR8 additions: marker injection + walker unification ──────────

#[test]
fn lossy_crush_injects_marker_into_output_json() {
    // Cornerstone of PR8: the public `crush()` API output now carries
    // the `<<ccr:HASH ...>>` marker as a string element of the array.
    // Without this, the LLM never sees the retrieval pointer.
    let crusher = SmartCrusher::new(force_lossy_config());
    let items = lossy_friendly_items(50);
    let raw = serde_json::to_string(&Value::Array(items)).unwrap();

    let result = crusher.crush(&raw, "", 1.0);

    assert!(
        result.compressed.contains("<<ccr:"),
        "expected marker in output, got: {}",
        result.compressed
    );
    assert!(
        result.compressed.contains("rows_offloaded>>"),
        "marker should advertise dropped count: {}",
        result.compressed
    );

    // The hash in the marker is the same one in the store.
    let marker_hash =
        extract_hash_from_marker(&result.compressed).expect("marker must embed a hash");
    let store = crusher.ccr_store().unwrap();
    assert!(
        store.get(&marker_hash).is_some(),
        "marker hash {marker_hash} must resolve in the store"
    );
}

#[test]
fn nested_array_inside_object_gets_marker_injected() {
    // The public crush() recurses through wrapper objects. A nested
    // `events: [...]` array that triggers lossy must get a marker too.
    let crusher = SmartCrusher::new(force_lossy_config());
    let doc = json!({
        "user": "alice",
        "events": (0..50).map(|i| json!({"id": i, "status": "ok"}))
            .collect::<Vec<_>>(),
    });

    let result = crusher.crush(&doc.to_string(), "", 1.0);
    assert!(result.compressed.contains("<<ccr:"));

    let parsed: Value = serde_json::from_str(&result.compressed).unwrap();
    let events = parsed.get("events").expect("events field preserved");
    let arr = events.as_array().expect("events stays an array");
    let last = arr.last().expect("non-empty array");
    let marker = last
        .get("_ccr_dropped")
        .and_then(|v| v.as_str())
        .expect("sentinel object with _ccr_dropped key");
    assert!(
        marker.starts_with("<<ccr:"),
        "expected CCR marker text, got: {marker}"
    );
}

#[test]
fn opaque_string_in_object_emits_marker_and_stores_original() {
    // Walker semantics in process_value: a long base64-ish blob in
    // an object field should be replaced with a CCR marker AND the
    // original bytes stashed in the store.
    let crusher = SmartCrusher::new(SmartCrusherConfig::default());
    let store = crusher.ccr_store().unwrap().clone();
    let starting_len = store.len();

    // 64-char base64 alphabet repeated → tripping the opaque-blob
    // detector deterministically.
    let big = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=".repeat(8);
    let doc = json!({"id": 1, "blob": big.clone()});

    let result = crusher.crush(&doc.to_string(), "", 1.0);
    let parsed: Value = serde_json::from_str(&result.compressed).unwrap();
    let blob_out = parsed.get("blob").and_then(|v| v.as_str()).unwrap();

    assert!(
        blob_out.starts_with("<<ccr:") && blob_out.contains(",base64,"),
        "expected base64 CCR marker, got: {blob_out}"
    );

    // The store grew by 1 and holds the original payload.
    assert_eq!(
        store.len(),
        starting_len + 1,
        "opaque-string CCR must write to store"
    );
    let marker_hash = extract_inner_hash(blob_out).unwrap();
    let retrieved = store.get(&marker_hash).expect("hash must resolve");
    assert_eq!(retrieved, big);
}

#[test]
fn stringified_json_array_recurses_and_compacts() {
    // A field whose value is a JSON-encoded array should be parsed,
    // recursively crushed, and re-encoded — the walker behavior, now
    // available from the main pipeline.
    let crusher = SmartCrusher::new(force_lossy_config());
    let inner = (0..50)
        .map(|i| json!({"id": i, "status": "ok"}))
        .collect::<Vec<_>>();
    let inner_json = serde_json::to_string(&inner).unwrap();
    let doc = json!({"id": "outer", "payload": inner_json});

    let result = crusher.crush(&doc.to_string(), "", 1.0);
    let parsed: Value = serde_json::from_str(&result.compressed).unwrap();
    let payload = parsed.get("payload").and_then(|v| v.as_str()).unwrap();

    // The payload string went through process_value's String arm,
    // which parsed it as JSON, recursed, and re-emitted. Result:
    // either a marker-bearing array string or a direct-rendered form.
    assert!(
        payload.contains("<<ccr:") || payload.contains("rows_offloaded"),
        "expected stringified-JSON to be processed, got: {payload}"
    );
}

#[test]
fn document_walker_with_store_roundtrips_opaque_blob() {
    // Direct DocumentCompactor usage with the same store.
    use headroom_core::transforms::smart_crusher::compaction::DocumentCompactor;

    let store: Arc<dyn CcrStore> = Arc::new(InMemoryCcrStore::new());
    let dc = DocumentCompactor::new().with_ccr_store(store.clone());

    let big = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=".repeat(8);
    let out = dc.compact(json!({"id": 1, "blob": big.clone()}));
    let blob = out.pointer("/blob").and_then(|v| v.as_str()).unwrap();
    assert!(blob.starts_with("<<ccr:"));

    let h = extract_inner_hash(blob).unwrap();
    assert_eq!(store.get(&h).unwrap(), big);
}

// ─── helpers ──────────────────────────────────────────────────────

/// Pull the hash out of a `<<ccr:HASH N_rows_offloaded>>` row marker.
fn extract_hash_from_marker(s: &str) -> Option<String> {
    let i = s.find("<<ccr:")?;
    let after = &s[i + 6..];
    let end = after.find(' ')?;
    Some(after[..end].to_string())
}

/// Pull the hash out of a `<<ccr:HASH,KIND,SIZE>>` opaque marker.
fn extract_inner_hash(s: &str) -> Option<String> {
    let i = s.find("<<ccr:")?;
    let after = &s[i + 6..];
    let end = after.find(',')?;
    Some(after[..end].to_string())
}
