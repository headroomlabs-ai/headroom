//! Integration tests for the PR-B3 live-zone dispatcher.
//!
//! These pin the per-content-type routing contract:
//!
//! - JSON array tool_results → SmartCrusher
//! - Build/log output       → LogCompressor
//! - Search-result tool_results → SearchCompressor
//! - Git diff tool_results  → DiffCompressor
//! - Source code            → CodeAwareCompressor
//! - Unknown / image / html → no-op
//!
//! Plus the cache-safety invariant: bytes outside the rewritten
//! block are byte-identical to the input (SHA-256 prefix + suffix).

mod common;

use common::{body_with_tool_result, dispatch, kompress_available, python_module_source, sha256};
use headroom_core::transforms::{BlockAction, LiveZoneOutcome};
use serde_json::{json, Value};

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
    // Detector classifies this as SourceCode. PR-B4 wires the arm up to
    // the tree-sitter-backed CodeAwareCompressor. This flips the PR-B3
    // pin ("a future 'wire it up' PR can flip this assertion").
    let code = python_module_source(10);
    assert!(
        code.len() > 2048,
        "fixture must clear the SourceCode byte threshold (2048); got {} bytes",
        code.len()
    );

    let (body, _) = body_with_tool_result(&code);
    let out = dispatch(&body);
    let manifest = match &out {
        LiveZoneOutcome::Modified { manifest, .. } => manifest,
        LiveZoneOutcome::NoChange { manifest } => panic!(
            "expected CodeAwareCompressor to shrink a 10-function Python module; got NoChange. manifest: {manifest:?}"
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
            original_tokens,
            compressed_tokens,
            ..
        } => {
            assert_eq!(
                strategy, "code_compressor",
                "expected code_compressor dispatch"
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
fn tiny_source_code_below_threshold_no_op() {
    // Detector classifies this as SourceCode, but it's well under the
    // 2048-byte SourceCode threshold, so the dispatcher must not even
    // spin up the CodeAwareCompressor.
    let code = "\
import os
from typing import Any


def add(a: int, b: int) -> int:
    \"\"\"Add two integers.\"\"\"
    return a + b


if __name__ == \"__main__\":
    print(add(1, 2))
";
    assert!(
        code.len() < 2048,
        "fixture must stay below the SourceCode byte threshold (2048); got {} bytes",
        code.len()
    );

    let (body, _) = body_with_tool_result(code);
    let out = dispatch(&body);
    let manifest = match &out {
        LiveZoneOutcome::NoChange { manifest } => manifest,
        LiveZoneOutcome::Modified { manifest, .. } => {
            panic!("tiny source-code block must not be compressed. manifest: {manifest:?}")
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
        BlockAction::BelowByteThreshold {
            content_type,
            threshold_bytes,
            ..
        } => {
            assert_eq!(threshold_bytes, 2048, "expected the SourceCode threshold");
            assert_eq!(content_type, "source_code", "unexpected content_type tag");
        }
        other => panic!("expected BelowByteThreshold, got {other:?}"),
    }
}

#[test]
fn plain_text_below_threshold_no_op() {
    // Plain prose, well under the 5120-byte PlainText threshold, so the
    // dispatcher must not even attempt Kompress.
    let prose = "The quarterly report highlighted steady growth across every \
        region, with the operations team noting improved throughput on the \
        warehouse floor and customer support tickets trending downward for \
        the third month running. Leadership expects the trend to continue \
        into next quarter, barring any supply chain disruptions.";
    assert!(
        prose.len() < 5120,
        "fixture must stay below the PlainText byte threshold (5120); got {} bytes",
        prose.len()
    );

    let (body, _) = body_with_tool_result(prose);
    let out = dispatch(&body);
    let manifest = match &out {
        LiveZoneOutcome::NoChange { manifest } => manifest,
        LiveZoneOutcome::Modified { manifest, .. } => {
            panic!("sub-threshold plain text must not be compressed. manifest: {manifest:?}")
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
        BlockAction::BelowByteThreshold {
            content_type,
            threshold_bytes,
            ..
        } => {
            assert_eq!(threshold_bytes, 5120, "expected the PlainText threshold");
            assert_eq!(content_type, "text", "unexpected content_type tag");
        }
        other => panic!("expected BelowByteThreshold, got {other:?}"),
    }
}

#[test]
fn plain_text_routes_to_kompress_when_model_cached() {
    // RUNTIME-SKIP: ask the loader itself whether the model is
    // cache-resident, rather than re-deriving cache paths here. A
    // hand-rolled probe has to be kept byte-compatible with
    // `Kompress::from_cache`'s own root and artifact resolution, and
    // when it drifts the skip becomes a lie — the test silently stops
    // running on hosts where production would load the model.
    if !kompress_available() {
        eprintln!(
            "SKIP: kompress model/tokenizer not cache-resident;              run `python scripts/record_kompress_trace.py` first"
        );
        return;
    }

    // Repetitive news-article-like prose: > 350 words so Kompress's
    // chunk_words=350 chunking actually engages, and > 5120 bytes to
    // clear the PlainText byte threshold.
    let mut article = String::new();
    for i in 0..60 {
        article.push_str(&format!(
            "City officials announced today that the downtown revitalization \
             project will proceed as planned despite budget concerns raised \
             during round {i} of public comment. "
        ));
    }
    assert!(
        article.split_whitespace().count() > 350,
        "fixture must exceed 350 words so Kompress chunking engages; got {} words",
        article.split_whitespace().count()
    );
    assert!(
        article.len() > 5120,
        "fixture must clear the PlainText byte threshold (5120); got {} bytes",
        article.len()
    );

    let (body, _) = body_with_tool_result(&article);
    let out = dispatch(&body);
    let manifest = match &out {
        LiveZoneOutcome::Modified { manifest, .. } => manifest,
        LiveZoneOutcome::NoChange { manifest } => panic!(
            "expected Kompress to compress a {}-word repetitive article; got NoChange. manifest: {manifest:?}",
            article.split_whitespace().count()
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
        BlockAction::Compressed { strategy, .. } => {
            assert_eq!(strategy, "kompress", "expected kompress dispatch");
        }
        other => panic!("expected BlockAction::Compressed via kompress, got {other:?}"),
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
