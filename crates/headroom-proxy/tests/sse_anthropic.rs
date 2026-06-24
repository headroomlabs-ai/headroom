//! Integration tests for the Anthropic Messages SSE state machine.
//!
//! Each test feeds a curated event stream through `SseFramer +
//! AnthropicStreamState` and asserts the structured state matches what
//! the Anthropic Messages spec (and the Headroom realignment guide
//! §5.1) requires.
//!
//! These tests retire P1-8 (`thinking_delta`), P1-9 (`signature_delta`),
//! P1-14 (`citations_delta`) — wire-format quirks the Python proxy
//! mishandled in production telemetry.

use bytes::Bytes;
use headroom_proxy::sse::anthropic::{AnthropicStreamState, StreamStatus};
use headroom_proxy::sse::SseFramer;

/// Push raw bytes into a framer and drain all framed events through
/// the state machine. Test failure on any framing OR state-machine
/// error — the curated inputs in these tests are valid by construction.
fn run(state: &mut AnthropicStreamState, raw: &[u8]) {
    let mut framer = SseFramer::new();
    framer.push(raw);
    while let Some(r) = framer.next_event() {
        let ev = r.expect("framer must not fail on valid inputs");
        state
            .apply(ev)
            .expect("state machine must not fail on valid inputs");
    }
}

#[test]
fn four_event_dance_text_block() {
    let mut s = AnthropicStreamState::new();
    let raw = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-3-5-sonnet\",\"usage\":{\"input_tokens\":100,\"output_tokens\":0}}}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello, \"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"world!\"}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":42}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    );
    run(&mut s, raw.as_bytes());

    assert_eq!(s.message_id.as_deref(), Some("msg_1"));
    assert_eq!(s.model.as_deref(), Some("claude-3-5-sonnet"));
    assert_eq!(s.status, StreamStatus::MessageStop);
    let block = s.blocks.get(&0).expect("block 0 must exist");
    assert_eq!(block.block_type, "text");
    assert_eq!(block.text_buffer, "Hello, world!");
    assert!(block.complete);
    assert_eq!(s.stop_reason.as_deref(), Some("end_turn"));
    assert_eq!(s.usage.input_tokens, 100);
    assert_eq!(s.usage.output_tokens, 42);
}

#[test]
fn thinking_delta_accumulated() {
    let mut s = AnthropicStreamState::new();
    let raw = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_2\",\"model\":\"claude-3-5-sonnet\",\"usage\":{\"input_tokens\":50,\"output_tokens\":0}}}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"Let me think...\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\" about this.\"}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
    );
    run(&mut s, raw.as_bytes());

    let block = s.blocks.get(&0).expect("thinking block must exist");
    assert_eq!(block.block_type, "thinking");
    assert_eq!(block.text_buffer, "Let me think... about this.");
    assert!(block.complete);
}

#[test]
fn signature_delta_preserved_byte_equal() {
    // Cryptographic signatures must round-trip BYTE-EQUAL. We verify
    // by feeding a base64-shaped value with characters that would be
    // mangled by any naive Unicode normalization (the `+/=` triad).
    let signature = "EqQBCkYIBxgCKkBcXt9+abc==/+ABC";
    let mut s = AnthropicStreamState::new();
    let raw = format!(
        concat!(
            "event: message_start\n",
            "data: {{\"type\":\"message_start\",\"message\":{{\"id\":\"msg_3\",\"model\":\"claude-3-5-sonnet\",\"usage\":{{\"input_tokens\":10,\"output_tokens\":0}}}}}}\n\n",
            "event: content_block_start\n",
            "data: {{\"type\":\"content_block_start\",\"index\":0,\"content_block\":{{\"type\":\"redacted_thinking\"}}}}\n\n",
            "event: content_block_delta\n",
            "data: {{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{{\"type\":\"signature_delta\",\"signature\":\"{}\"}}}}\n\n",
            "event: content_block_stop\n",
            "data: {{\"type\":\"content_block_stop\",\"index\":0}}\n\n",
        ),
        signature
    );
    run(&mut s, raw.as_bytes());

    let block = s.blocks.get(&0).expect("redacted block must exist");
    assert_eq!(
        block.signature.as_deref(),
        Some(signature),
        "signature must be byte-equal preserved (no normalization, no escaping)"
    );
}

#[test]
fn input_json_delta_concatenated_parsed_at_stop() {
    let mut s = AnthropicStreamState::new();
    // Tool-use block: input_json_delta fragments accumulate into a
    // partial_json string. At content_block_stop, the proxy attempts
    // to parse it (failure logs but does not panic).
    let raw = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_4\",\"model\":\"claude-3-5-sonnet\",\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"calc\",\"input\":{}}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"a\\\":\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"42}\"}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
    );
    run(&mut s, raw.as_bytes());

    let block = s.blocks.get(&0).expect("tool_use block must exist");
    assert_eq!(block.block_type, "tool_use");
    assert_eq!(block.partial_json, r#"{"a":42}"#);
    // The accumulated string must parse as JSON now that all
    // fragments are concatenated. (The state machine doesn't store
    // the parsed value; we verify here.)
    let parsed: serde_json::Value =
        serde_json::from_str(&block.partial_json).expect("concatenated partial_json must parse");
    assert_eq!(parsed["a"], 42);
    assert!(block.complete);
}

#[test]
fn citations_delta_accumulated() {
    let mut s = AnthropicStreamState::new();
    let raw = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_5\",\"model\":\"claude-3-5-sonnet\",\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"citations_delta\",\"citation\":{\"type\":\"page_location\",\"start_page\":1,\"end_page\":2}}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"citations_delta\",\"citation\":{\"type\":\"page_location\",\"start_page\":3,\"end_page\":4}}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
    );
    run(&mut s, raw.as_bytes());

    let block = s.blocks.get(&0).expect("text block must exist");
    assert_eq!(block.citations.len(), 2);
    assert_eq!(block.citations[0]["start_page"], 1);
    assert_eq!(block.citations[1]["start_page"], 3);
}

#[test]
fn message_delta_finalizes_stop_reason_and_output_tokens() {
    let mut s = AnthropicStreamState::new();
    let raw = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_6\",\"model\":\"claude-3-5-sonnet\",\"usage\":{\"input_tokens\":7,\"output_tokens\":0,\"cache_creation_input_tokens\":3,\"cache_read_input_tokens\":2}}}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"max_tokens\"},\"usage\":{\"output_tokens\":1024}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    );
    run(&mut s, raw.as_bytes());

    assert_eq!(s.stop_reason.as_deref(), Some("max_tokens"));
    assert_eq!(s.usage.input_tokens, 7);
    assert_eq!(s.usage.output_tokens, 1024);
    assert_eq!(s.usage.cache_creation_input_tokens, 3);
    assert_eq!(s.usage.cache_read_input_tokens, 2);
    assert_eq!(s.status, StreamStatus::MessageStop);
}

#[test]
fn mid_stream_error_event_handled() {
    let mut s = AnthropicStreamState::new();
    let raw = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_7\",\"model\":\"claude-3-5-sonnet\",\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n",
        "event: error\n",
        "data: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n",
    );
    run(&mut s, raw.as_bytes());

    assert_eq!(
        s.status,
        StreamStatus::Errored,
        "error event must transition status to Errored"
    );
    assert_eq!(s.message_id.as_deref(), Some("msg_7"));
}

#[test]
fn interleaved_blocks_by_index() {
    // The spec permits blocks to interleave deltas (block 0 delta,
    // then block 1 delta, then block 0 delta, etc). Each block's
    // text_buffer must be independent — keyed by `index`, not by
    // arrival order.
    let mut s = AnthropicStreamState::new();
    let raw = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_8\",\"model\":\"claude-3-5-sonnet\",\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"A0 \"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"B0 \"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"A1\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"B1\"}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
    );
    run(&mut s, raw.as_bytes());

    assert_eq!(s.blocks.get(&0).unwrap().text_buffer, "A0 A1");
    assert_eq!(s.blocks.get(&1).unwrap().text_buffer, "B0 B1");
    assert!(s.blocks.get(&0).unwrap().complete);
    assert!(s.blocks.get(&1).unwrap().complete);
}

#[test]
fn split_chunks_preserve_event_boundaries() {
    // Feed the events one byte at a time. The state machine must
    // produce identical structured output regardless of how the bytes
    // arrive — this is the cache-safety invariant under degenerate TCP.
    let raw = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_9\",\"model\":\"claude-3-5-sonnet\",\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
    )
    .as_bytes();

    let mut framer = SseFramer::new();
    let mut s = AnthropicStreamState::new();
    for byte in raw {
        framer.push(std::slice::from_ref(byte));
        while let Some(r) = framer.next_event() {
            s.apply(r.unwrap()).unwrap();
        }
    }
    assert_eq!(s.blocks.get(&0).unwrap().text_buffer, "hi");
}

/// Reference SseEvent used to verify Bytes payload typing in this file.
#[allow(dead_code)]
fn _ref_event() -> headroom_proxy::sse::SseEvent {
    headroom_proxy::sse::SseEvent {
        event_name: None,
        data: Bytes::from_static(b""),
    }
}
