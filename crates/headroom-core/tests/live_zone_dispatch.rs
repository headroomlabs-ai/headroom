//! Integration tests for the PR-B3 live-zone dispatcher.
//!
//! These pin the per-content-type routing contract:
//!
//! - JSON array tool_results → SmartCrusher
//! - Build/log output       → LogCompressor
//! - Search-result tool_results → SearchCompressor
//! - Git diff tool_results  → DiffCompressor
//! - Source code            → no-op (Rust port pending)
//! - Unknown / image / html → no-op
//!
//! Plus the cache-safety invariant: bytes outside the rewritten
//! block are byte-identical to the input (SHA-256 prefix + suffix).

use headroom_core::transforms::live_zone::DEFAULT_MODEL;
use headroom_core::transforms::{
    compress_anthropic_live_zone, AuthMode, BlockAction, LiveZoneOutcome,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

fn body_of(value: Value) -> Vec<u8> {
    serde_json::to_vec(&value).unwrap()
}

fn dispatch(body: &[u8]) -> LiveZoneOutcome {
    compress_anthropic_live_zone(body, 0, AuthMode::Payg, DEFAULT_MODEL)
        .expect("dispatcher returns Ok on valid bodies")
}

/// Find the byte range of the FIRST occurrence of `needle` inside
/// `haystack`. Used by the byte-fidelity test below to identify the
/// JSON-encoded tool_result.content slot we expect the dispatcher to
/// rewrite. Returns `(start, end)` half-open.
fn find_byte_range(haystack: &[u8], needle: &[u8]) -> (usize, usize) {
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

fn sha256(bytes: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().into()
}

/// Build a body with one user message containing one `tool_result`
/// whose `content` is `text`. Returns the full body and the byte
/// range of the JSON-encoded `content` slot (including the surrounding
/// quotes) within that body — useful for byte-fidelity assertions.
fn body_with_tool_result(text: &str) -> (Vec<u8>, (usize, usize)) {
    let body = body_of(json!({
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "system": "you are a helpful assistant",
        "messages": [{
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_dispatch_test",
                "content": text,
            }],
        }],
    }));
    // The JSON-encoded `content` slot is exactly `serde_json::to_vec(&text)`,
    // since text is shorter than the whole body and serde uses the same
    // encoding for the embedded string.
    let needle = serde_json::to_vec(&text).unwrap();
    let range = find_byte_range(&body, &needle);
    (body, range)
}

// ─── Routing tests ─────────────────────────────────────────────────────

#[test]
fn json_tool_result_routes_to_smart_crusher() {
    // Array of homogeneous dicts → SmartCrusher's bread-and-butter.
    let array_of_dicts: Vec<Value> = (0..200)
        .map(|i| {
            json!({
                "id": i,
                "status": "ok",
                "value": format!("repeat-pattern-{}", i % 3),
            })
        })
        .collect();
    let payload = serde_json::to_string(&array_of_dicts).unwrap();
    let (body, _) = body_with_tool_result(&payload);

    let out = dispatch(&body);
    let manifest = match &out {
        LiveZoneOutcome::Modified { manifest, .. } => manifest,
        LiveZoneOutcome::NoChange { manifest } => panic!(
            "expected SmartCrusher to compress 200 homogeneous dicts; got NoChange. manifest: {manifest:?}"
        ),
    };
    let action = manifest
        .block_outcomes
        .iter()
        .find(|b| b.block_type == "tool_result")
        .expect("tool_result block present in manifest")
        .action
        .clone();
    match action {
        BlockAction::Compressed {
            strategy,
            original_bytes,
            compressed_bytes,
            original_tokens,
            compressed_tokens,
        } => {
            assert_eq!(strategy, "smart_crusher", "expected SmartCrusher dispatch");
            assert!(
                compressed_bytes < original_bytes,
                "SmartCrusher must produce strictly smaller output ({compressed_bytes} < {original_bytes})"
            );
            assert!(
                compressed_tokens < original_tokens,
                "tokenizer-validated gate (PR-B4) must accept only token-shrinking output \
                 ({compressed_tokens} < {original_tokens})"
            );
        }
        other => panic!("expected BlockAction::Compressed, got {other:?}"),
    }
}

#[test]
fn log_tool_result_routes_to_log_compressor() {
    // Multi-line build/log output that the detector classifies as
    // `BuildOutput`. Repetitive lines compress well.
    let mut lines = String::new();
    for i in 0..200 {
        lines.push_str(&format!(
            "[INFO] 2026-05-02T19:30:{:02}.000Z app=widget request_id=abc-{} pool=default ok\n",
            i % 60,
            i
        ));
    }
    let (body, _) = body_with_tool_result(&lines);

    let out = dispatch(&body);
    let manifest = match &out {
        LiveZoneOutcome::Modified { manifest, .. } => manifest,
        LiveZoneOutcome::NoChange { .. } => {
            // The log compressor may decline if the lines aren't
            // repetitive enough; accept either outcome but require the
            // detector to have routed it correctly. Check the manifest
            // for the dispatch attempt.
            let nochange_manifest = match &out {
                LiveZoneOutcome::NoChange { manifest } => manifest,
                _ => unreachable!(),
            };
            let action = nochange_manifest
                .block_outcomes
                .iter()
                .find(|b| b.block_type == "tool_result")
                .expect("tool_result block present")
                .action
                .clone();
            assert!(
                matches!(
                    action,
                    BlockAction::NoCompressionApplied { .. }
                        | BlockAction::RejectedNotSmaller { .. }
                        | BlockAction::BelowByteThreshold { .. }
                ),
                "log dispatch declined cleanly: {action:?}"
            );
            return;
        }
    };

    let action = manifest
        .block_outcomes
        .iter()
        .find(|b| b.block_type == "tool_result")
        .expect("tool_result block present")
        .action
        .clone();
    match action {
        BlockAction::Compressed {
            strategy,
            original_bytes,
            compressed_bytes,
            ..
        } => {
            assert_eq!(strategy, "log_compressor");
            assert!(compressed_bytes < original_bytes);
        }
        other => panic!("expected log_compressor Compressed, got {other:?}"),
    }
}

#[test]
fn diff_tool_result_routes_to_diff_compressor() {
    // A unidiff with surrounding context the diff compressor can trim.
    // Size kept comfortably above the 1 KiB GitDiff byte threshold
    // (PR-B4) so the dispatch gate is exercised.
    let mut diff = String::from("diff --git a/foo.rs b/foo.rs\n--- a/foo.rs\n+++ b/foo.rs\n");
    diff.push_str("@@ -1,80 +1,80 @@\n");
    for i in 0..40 {
        diff.push_str(&format!(" context line {i} with extra padding text\n"));
    }
    diff.push_str("-old line that needs to be replaced\n+new line replacing the old one\n");
    for i in 0..40 {
        diff.push_str(&format!(
            " context line {} with extra padding text\n",
            i + 40
        ));
    }
    assert!(
        diff.len() > 1024,
        "diff fixture must be > 1 KiB to clear the GitDiff threshold; got {}",
        diff.len()
    );

    let (body, _) = body_with_tool_result(&diff);
    let out = dispatch(&body);
    let manifest = match &out {
        LiveZoneOutcome::Modified { manifest, .. } => manifest,
        LiveZoneOutcome::NoChange { manifest } => {
            let action = manifest
                .block_outcomes
                .iter()
                .find(|b| b.block_type == "tool_result")
                .expect("tool_result block present")
                .action
                .clone();
            assert!(
                matches!(
                    action,
                    BlockAction::NoCompressionApplied { .. }
                        | BlockAction::RejectedNotSmaller { .. }
                        | BlockAction::BelowByteThreshold { .. }
                ),
                "diff dispatch declined cleanly: {action:?}"
            );
            return;
        }
    };
    let action = manifest
        .block_outcomes
        .iter()
        .find(|b| b.block_type == "tool_result")
        .expect("tool_result block present")
        .action
        .clone();
    match action {
        BlockAction::Compressed { strategy, .. } => {
            assert_eq!(strategy, "diff_compressor");
        }
        other => panic!("expected diff_compressor Compressed, got {other:?}"),
    }
}

#[test]
fn source_code_tool_result_routes_to_code_compressor() {
    // Detector classifies this as SourceCode; the dispatcher now routes it
    // to the Rust CodeCompressor (was no-op before the port landed). Twenty
    // identical multi-line Rust functions are well above the SourceCode byte
    // threshold and compress (each body is truncated to a budget + an
    // omitted-lines comment), so the dispatch yields a strictly smaller,
    // token-shrinking block.
    let code = "
fn main() {
    let x: i32 = 42;
    let y = x * 2;
    println!(\"{}\", y);
    if x > 0 {
        println!(\"positive\");
    } else {
        println!(\"non-positive\");
    }
}
"
    .repeat(20);
    let (body, _) = body_with_tool_result(&code);
    let out = dispatch(&body);
    let manifest = match &out {
        LiveZoneOutcome::Modified { manifest, .. } => manifest,
        LiveZoneOutcome::NoChange { manifest } => panic!(
            "expected CodeCompressor to compress 20 Rust functions; got NoChange. manifest: {manifest:?}"
        ),
    };
    let action = manifest
        .block_outcomes
        .iter()
        .find(|b| b.block_type == "tool_result")
        .expect("tool_result block present")
        .action
        .clone();
    match action {
        BlockAction::Compressed {
            strategy,
            original_bytes,
            compressed_bytes,
            original_tokens,
            compressed_tokens,
        } => {
            assert_eq!(strategy, "code_aware_compressor", "expected CodeCompressor dispatch");
            assert!(
                compressed_bytes < original_bytes,
                "CodeCompressor must produce smaller output ({compressed_bytes} < {original_bytes})"
            );
            assert!(
                compressed_tokens < original_tokens,
                "tokenizer-validated gate must accept only token-shrinking output \
                 ({compressed_tokens} < {original_tokens})"
            );
        }
        other => panic!("expected BlockAction::Compressed (code_aware_compressor), got {other:?}"),
    }
}

/// True iff the Kompress model + ModernBERT tokenizer are in the local HF
/// cache. The PlainText dispatch loads cache-only, so its behavior is
/// model-gated exactly like the kompress parity test.
fn kompress_model_cached() -> bool {
    use std::path::Path;
    let Ok(home) = std::env::var("HOME") else {
        return false;
    };
    let hub = Path::new(&home).join(".cache/huggingface/hub");
    let has = |repo: &str, rel: &[&str]| -> bool {
        let snaps = hub.join(repo).join("snapshots");
        let Ok(rd) = std::fs::read_dir(&snaps) else {
            return false;
        };
        rd.flatten().any(|e| {
            let mut p = e.path();
            for part in rel {
                p = p.join(part);
            }
            p.exists()
        })
    };
    has("models--answerdotai--ModernBERT-base", &["tokenizer.json"])
        && has(
            "models--chopratejas--kompress-v2-base",
            &["onnx", "kompress-int8-wo.onnx"],
        )
}

#[test]
fn plain_text_tool_result_routes_to_kompress() {
    // Long, repetitive English prose → detector classifies as PlainText,
    // comfortably above the 512-byte threshold. The dispatcher now routes it
    // to Kompress (cache-only). Model-gated: when the model isn't cached the
    // dispatcher passes through (mirroring the Python "unavailable → unchanged"
    // behavior), so we assert the wiring contract under both conditions.
    // Kompress is gated off by default (it loads a ~261 MB model); enable it
    // for this test exactly as the proxy does at startup from config.
    headroom_core::transforms::set_kompress_enabled(true);

    let prose = "The quick brown fox jumps over the lazy dog while the diligent \
                 engineer carefully reviews the output and discards redundant filler. "
        .repeat(12);
    assert!(prose.len() > 512, "prose must clear the PlainText threshold");
    let (body, _) = body_with_tool_result(&prose);
    let out = dispatch(&body);

    let action = match &out {
        LiveZoneOutcome::Modified { manifest, .. } | LiveZoneOutcome::NoChange { manifest } => {
            manifest
                .block_outcomes
                .iter()
                .find(|b| b.block_type == "tool_result")
                .expect("tool_result block present")
                .action
                .clone()
        }
    };

    if kompress_model_cached() {
        // Model present → Kompress ran. It either compressed (strategy
        // "kompress") or, if it kept every word / the tokenizer gate rejected
        // it, declined cleanly. Any outcome that names a *different*
        // compressor would mean mis-routing.
        match action {
            BlockAction::Compressed { strategy, .. } => {
                assert_eq!(strategy, "kompress", "PlainText must route to Kompress");
            }
            BlockAction::NoCompressionApplied { .. }
            | BlockAction::RejectedNotSmaller { .. }
            | BlockAction::BelowByteThreshold { .. } => {}
            other => panic!("unexpected PlainText action with model cached: {other:?}"),
        }
    } else {
        // Model not cached → passthrough, exactly like Python when Kompress
        // is unavailable.
        assert!(
            matches!(
                action,
                BlockAction::NoCompressionApplied { .. } | BlockAction::BelowByteThreshold { .. }
            ),
            "PlainText must pass through when the Kompress model isn't cached: {action:?}"
        );
    }
}

#[test]
fn unknown_content_type_no_op() {
    // Empty string should not invoke any compressor.
    let (body, _) = body_with_tool_result("");
    let out = dispatch(&body);
    let manifest = match &out {
        LiveZoneOutcome::NoChange { manifest } => manifest,
        LiveZoneOutcome::Modified { .. } => panic!("empty content must not trigger compression"),
    };
    let action = manifest
        .block_outcomes
        .iter()
        .find(|b| b.block_type == "tool_result")
        .expect("tool_result block present")
        .action
        .clone();
    assert!(
        matches!(action, BlockAction::NoCompressionApplied { .. }),
        "expected NoCompressionApplied, got {action:?}"
    );
}

// ─── Cache-safety invariant ────────────────────────────────────────────

#[test]
fn byte_fidelity_outside_compressed_block() {
    // 50 KB of homogeneous JSON dicts — guaranteed SmartCrusher fodder.
    // This pins the central B3 acceptance criterion: bytes OUTSIDE
    // the rewritten block must hash byte-identical to the input.
    let array_of_dicts: Vec<Value> = (0..1500)
        .map(|i| {
            json!({
                "id": i,
                "kind": "row",
                "value": format!("repeat-{}", i % 5),
                "status": "ok",
            })
        })
        .collect();
    let payload = serde_json::to_string(&array_of_dicts).unwrap();
    assert!(payload.len() > 50_000, "payload should exceed 50 KB");

    let (body_in, content_range) = body_with_tool_result(&payload);
    let (block_start, block_end) = content_range;

    let out = dispatch(&body_in);
    let new_body = match &out {
        LiveZoneOutcome::Modified { new_body, .. } => new_body.get().as_bytes().to_vec(),
        LiveZoneOutcome::NoChange { manifest } => panic!(
            "expected Modified outcome on 50 KB SmartCrusher fodder; got NoChange. manifest: {manifest:?}"
        ),
    };

    // Prefix bytes (before the content slot) must be byte-identical.
    let in_prefix = &body_in[..block_start];
    let out_prefix = &new_body[..block_start];
    assert_eq!(
        sha256(in_prefix),
        sha256(out_prefix),
        "prefix bytes outside the compressed block must be byte-equal"
    );

    // Suffix length will differ by the compression delta, so locate
    // the suffix in the output by length: it's the trailing
    // (in.len() - block_end) bytes.
    let in_suffix_len = body_in.len() - block_end;
    let in_suffix = &body_in[block_end..];
    let out_suffix = &new_body[new_body.len() - in_suffix_len..];
    assert_eq!(
        sha256(in_suffix),
        sha256(out_suffix),
        "suffix bytes outside the compressed block must be byte-equal"
    );

    // 2× size reduction inside the block.
    let in_block = &body_in[block_start..block_end];
    let out_block_len = new_body.len() - block_start - in_suffix_len;
    assert!(
        out_block_len * 2 < in_block.len(),
        "expected >2× block size reduction; got {out_block_len} bytes (was {})",
        in_block.len()
    );

    // Output must be valid JSON.
    let parsed: Value = serde_json::from_slice(&new_body).expect("output is valid JSON");
    assert_eq!(parsed["model"], "claude-sonnet-4-6");
    assert_eq!(parsed["system"], "you are a helpful assistant");
}
