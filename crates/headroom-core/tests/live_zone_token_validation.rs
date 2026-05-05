//! PR-B4 token-validation gate — integration + property tests.
//!
//! After every per-block compression, the dispatcher counts tokens on
//! both the original and compressed text using the model's tokenizer.
//! When `compressed_tokens >= original_tokens` the candidate is
//! rejected and the original bytes are kept. This file pins the
//! contract for both directions of the gate, plus an invariant
//! property test: the dispatcher's emitted body has token-count <= the
//! input's token-count for any well-formed body.

use headroom_core::tokenizer::get_tokenizer;
use headroom_core::transforms::live_zone::DEFAULT_MODEL;
use headroom_core::transforms::{
    compress_anthropic_live_zone, AuthMode, BlockAction, LiveZoneOutcome,
};
use proptest::prelude::*;
use serde_json::{json, Value};

fn body_of(value: Value) -> Vec<u8> {
    serde_json::to_vec(&value).unwrap()
}

fn body_with_tool_result(text: &str) -> Vec<u8> {
    body_of(json!({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 64,
        "messages": [{
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_token_test",
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

/// Count the tokens of a body's serialized form via the same tokenizer
/// the dispatcher uses. Used by the property test below.
fn token_count_of(bytes: &[u8]) -> usize {
    let tok = get_tokenizer(DEFAULT_MODEL);
    let s = std::str::from_utf8(bytes).expect("body is UTF-8 JSON");
    tok.count_text(s)
}

#[test]
fn compressed_more_tokens_falls_back() {
    // Pathological input: a single dense base64 string wrapped in a
    // JSON array. The detector classifies it as JsonArray (which
    // routes to SmartCrusher), but the contents do not look like a
    // dict array — SmartCrusher will not modify it. We construct a
    // case where the content_type is JsonArray and SmartCrusher's
    // crush returns the input unchanged (`was_modified=false`),
    // which dispatches as NoOp / NoCompressionApplied — that does
    // not test the token gate.
    //
    // To exercise the token gate we need SmartCrusher to actually
    // produce *some* output. Build an array of dicts containing
    // already-minified base64 — SmartCrusher's column-extraction
    // can produce output with more tokens than the raw bytes, even
    // when `was_modified=true`, because dense base64 fragments
    // tokenize into many short BPE pieces post-rewrite.
    //
    // Empirically, for some pathological dict shapes SmartCrusher's
    // output is larger in tokens than the input. The dispatcher
    // must reject such cases via the token gate.
    //
    // The fixture below is deliberately constructed to be on the
    // boundary; if SmartCrusher decides to leave it alone (which it
    // is allowed to do) we still assert the gate behavior — by
    // checking either RejectedNotSmaller (gate fired) OR
    // NoCompressionApplied (compressor declined first). Both are
    // safe outcomes; the negative we want to catch is `Compressed`
    // with token-count growth, which the dispatcher cannot emit.

    // Build a 2 KiB array of single-key dicts holding base64 — large
    // enough to clear the 1 KiB JsonArray threshold but pathological
    // enough that the SmartCrusher's column rewrite is not strictly
    // smaller in tokens for a Claude-family chars-per-token estimator.
    let mut payload = String::from("[");
    for i in 0..40 {
        if i > 0 {
            payload.push(',');
        }
        // 50 bytes of dense base64-ish noise per entry.
        let blob: String = (0..50).map(|j| ((j + i) % 64) as u8 as char).collect();
        let blob_clean: String = blob
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { 'A' })
            .collect();
        payload.push_str(&format!("{{\"b\":\"{}\"}}", blob_clean));
    }
    payload.push(']');
    assert!(
        payload.len() >= 1024,
        "fixture must clear the 1 KiB JsonArray byte threshold; got {}",
        payload.len()
    );

    let body = body_with_tool_result(&payload);
    let out = dispatch(&body);
    let action = first_tool_result_action(&out);

    match action {
        BlockAction::RejectedNotSmaller {
            original_tokens,
            compressed_tokens,
            ..
        } => {
            assert!(
                compressed_tokens >= original_tokens,
                "RejectedNotSmaller invariant: compressed_tokens ({compressed_tokens}) \
                 must be >= original_tokens ({original_tokens})"
            );
        }
        // Compressor declined before producing output — also safe.
        BlockAction::NoCompressionApplied { .. } => {}
        // Compressor produced output AND it shrunk the token count —
        // also acceptable (this means our fixture wasn't actually
        // pathological for this tokenizer).
        BlockAction::Compressed {
            original_tokens,
            compressed_tokens,
            ..
        } => {
            assert!(
                compressed_tokens < original_tokens,
                "Compressed invariant: compressed_tokens ({compressed_tokens}) \
                 must be < original_tokens ({original_tokens})"
            );
        }
        BlockAction::BelowByteThreshold { .. } => {
            panic!(
                "fixture was supposed to clear the byte threshold; got BelowByteThreshold. \
                 payload len = {}",
                payload.len()
            );
        }
        other => panic!("unexpected dispatcher outcome: {other:?}"),
    }
}

#[test]
fn compressed_fewer_tokens_accepted() {
    // Well-formed JSON array of homogeneous dicts — SmartCrusher's
    // bread-and-butter shape. The token gate must accept this.
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

    let body = body_with_tool_result(&payload);
    let out = dispatch(&body);
    let action = first_tool_result_action(&out);

    match action {
        BlockAction::Compressed {
            strategy,
            original_tokens,
            compressed_tokens,
            ..
        } => {
            assert_eq!(strategy, "smart_crusher");
            assert!(
                compressed_tokens < original_tokens,
                "tokenizer-validated gate must produce strict token shrinkage \
                 ({compressed_tokens} < {original_tokens})"
            );
        }
        other => panic!(
            "expected token-shrinking Compressed for 200-dict SmartCrusher fodder, got {other:?}"
        ),
    }
}

proptest! {
    /// Invariant: for ANY well-formed Anthropic body the dispatcher
    /// either passes through unchanged or emits a body whose token
    /// count is <= the input's token count. There is no input shape
    /// for which the dispatcher must inflate tokens.
    ///
    /// The strategy generates JSON-array-of-dicts payloads whose
    /// dict shapes vary (matching the sort of tool-result content
    /// the dispatcher actually sees in production). Sub-threshold
    /// payloads short-circuit to `NoChange` and therefore trivially
    /// satisfy the invariant.
    #[test]
    fn live_zone_compression_token_count_non_increasing(
        n in 0usize..150,
        key_a in "[a-z]{1,8}",
        key_b in "[a-z]{1,8}",
        seed in any::<u32>(),
    ) {
        let arr: Vec<Value> = (0..n)
            .map(|i| {
                let v: u64 = (seed as u64).wrapping_add(i as u64);
                json!({
                    key_a.as_str(): v % 1000,
                    key_b.as_str(): format!("v{}", v % 7),
                })
            })
            .collect();
        let payload = serde_json::to_string(&arr).unwrap();
        let body = body_with_tool_result(&payload);
        let body_tokens_in = token_count_of(&body);

        let outcome = compress_anthropic_live_zone(
            &body,
            0,
            AuthMode::Payg,
            DEFAULT_MODEL,
        )
        .expect("dispatcher returns Ok on valid bodies");

        let body_tokens_out = match &outcome {
            LiveZoneOutcome::NoChange { .. } => body_tokens_in,
            LiveZoneOutcome::Modified { new_body, .. } => {
                token_count_of(new_body.get().as_bytes())
            }
        };

        prop_assert!(
            body_tokens_out <= body_tokens_in,
            "dispatcher must never inflate token count: \
             tokens_in={body_tokens_in}, tokens_out={body_tokens_out}, payload_len={}",
            payload.len(),
        );
    }
}
