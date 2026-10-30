//! Shared fixtures for the live-zone integration tests and their
//! dedicated single-test binaries.
//!
//! Integration test files are separate crates, so each `mod common;`
//! gets its own copy of this module — the standard Rust layout for
//! sharing test helpers (The Book, ch. 11.3). Before this module the
//! same body builders, Python/prose generators and HuggingFace-cache
//! probes were pasted into four targets, where the cache probe in
//! particular had to be kept byte-compatible with the production
//! loader by hand.
//!
//! Not every target uses every helper, so unused items here are
//! expected rather than a defect.
#![allow(dead_code)]

use headroom_core::transforms::live_zone::DEFAULT_MODEL;
use headroom_core::transforms::{
    compress_anthropic_live_zone, AuthMode, BlockAction, CompressionManifest, LiveZoneOutcome,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

/// Serialize a JSON value to a request body.
pub fn body_of(value: &Value) -> Vec<u8> {
    serde_json::to_vec(value).expect("fixture body serializes")
}

/// Run the public dispatcher entry point with no frozen prefix.
pub fn dispatch(body: &[u8]) -> LiveZoneOutcome {
    compress_anthropic_live_zone(body, 0, AuthMode::Payg, DEFAULT_MODEL)
        .expect("dispatcher returns Ok on valid bodies")
}

pub fn sha256(bytes: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().into()
}

/// Byte range of the first occurrence of `needle` in `haystack`,
/// half-open. Used to locate the JSON-encoded `content` slot for
/// byte-fidelity assertions.
pub fn find_byte_range(haystack: &[u8], needle: &[u8]) -> (usize, usize) {
    let pos = haystack
        .windows(needle.len())
        .position(|w| w == needle)
        .unwrap_or_else(|| {
            panic!(
                "needle of {} bytes not found in haystack of {} bytes",
                needle.len(),
                haystack.len()
            )
        });
    (pos, pos + needle.len())
}

/// A body with one user message holding one `tool_result` whose
/// `content` is `text`. Returns the body and the byte range of the
/// JSON-encoded content slot (quotes included) inside it.
pub fn body_with_tool_result(text: &str) -> (Vec<u8>, (usize, usize)) {
    let body = body_of(&json!({
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "system": "you are a helpful assistant",
        "messages": [{
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_live_zone_test",
                "content": text,
            }],
        }],
    }));
    // The encoded slot is exactly `to_vec(&text)`: serde uses the same
    // string encoding for the embedded value.
    let needle = serde_json::to_vec(&text).expect("text serializes");
    let range = find_byte_range(&body, &needle);
    (body, range)
}

/// The `tool_result` block's action from a manifest, cloned out.
pub fn tool_result_action(manifest: &CompressionManifest) -> BlockAction {
    manifest
        .block_outcomes
        .iter()
        .find(|b| b.block_type == "tool_result")
        .expect("tool_result block present in manifest")
        .action
        .clone()
}

/// Syntactically valid Python with `n` small functions, each with a
/// docstring and a body longer than `CodeCompressorConfig`'s default
/// `max_body_lines` (5) so body elision has something to trim. Real
/// Python: the compressor re-parses and reverts on syntax errors, so a
/// fixture that merely looks like code would silently pass through.
pub fn python_module_source(n: usize) -> String {
    use std::fmt::Write as _;

    let mut code = String::from(
        "\"\"\"Example data-processing module used by the live-zone tests.\"\"\"\n\n\
         import json\n\
         import os\n\
         from typing import Any, Optional\n\n\n",
    );
    for i in 0..n {
        let _ = write!(
            code,
            "def process_record_{i}(record: dict) -> dict:\n    \
             \"\"\"Normalize record {i} and compute its derived fields.\"\"\"\n    \
             result = dict(record)\n    \
             result[\"index\"] = {i}\n    \
             result[\"doubled\"] = record.get(\"value\", 0) * 2\n    \
             result[\"source\"] = \"batch\"\n    \
             if result[\"doubled\"] > 100:\n        \
             result[\"flag\"] = \"high\"\n    \
             else:\n        \
             result[\"flag\"] = \"low\"\n    \
             return result\n\n\n"
        );
    }
    code
}

/// Repetitive plain prose of at least `min_bytes`, varied enough to
/// classify as `PlainText`.
pub fn plain_prose(min_bytes: usize) -> String {
    use std::fmt::Write as _;

    let mut text = String::with_capacity(min_bytes + 256);
    let mut i = 0usize;
    while text.len() < min_bytes {
        let _ = write!(
            text,
            "City officials announced today that the downtown revitalization \
             project will proceed as planned despite budget concerns raised \
             during round {i} of public comment. "
        );
        i += 1;
    }
    text
}

/// A JSON array of dicts, `n` entries — SmartCrusher's shape.
pub fn json_array_of_dicts(n: usize) -> String {
    let rows: Vec<Value> = (0..n)
        .map(|i| {
            json!({
                "id": i,
                "status": "ok",
                "value": format!("repeat-pattern-{}", i % 3),
            })
        })
        .collect();
    serde_json::to_string(&rows).expect("fixture array serializes")
}

/// Whether the Kompress model is cache-resident and loaded, asked of the
/// production slot itself: this runs `live_zone`'s blocking warmup — the
/// proxy's own startup path — and reports what it found, rather than
/// re-deriving HuggingFace cache paths in a probe that could desync.
///
/// Call it BEFORE dispatching content that should reach the arm.
/// Dispatch never waits on model construction, so an unwarmed first
/// dispatch is a NoOp by design — the contract pinned by
/// `live_zone_kompress_async_init.rs`. Warming here both settles the
/// slot the dispatch under test will read and prices the probe at one
/// shared construction instead of a throwaway one.
pub fn kompress_available() -> bool {
    headroom_core::transforms::live_zone::warm_live_zone_compressors()
}
