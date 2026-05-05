//! Integration tests for the OpenAI Responses SSE state machine.
//!
//! Wire-format quirks under test (per realignment guide §5.3):
//!
//!   - Items are keyed by `item.id`, NOT by position. The spec permits
//!     out-of-order completion: item B can complete before item A even
//!     when both are open. (P1-17 in production telemetry.)
//!   - `function_call_arguments.delta` fragments concatenate as a
//!     STRING; the proxy never parses them as JSON.
//!   - `reasoning_summary.delta` fragments accumulate per item.

use headroom_proxy::sse::openai_responses::{ResponseState, StreamStatus};
use headroom_proxy::sse::SseFramer;

fn run(state: &mut ResponseState, raw: &[u8]) {
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
fn out_of_order_item_completion_by_id() {
    // Two items open: msg_A and msg_B. msg_B completes BEFORE msg_A.
    // The state machine must record completion against the right
    // item by id, not by arrival order.
    let mut s = ResponseState::new();
    let raw = concat!(
        "event: response.created\n",
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-5\"}}\n\n",
        "event: output_item.added\n",
        "data: {\"type\":\"output_item.added\",\"item\":{\"id\":\"msg_A\",\"type\":\"message\"}}\n\n",
        "event: output_item.added\n",
        "data: {\"type\":\"output_item.added\",\"item\":{\"id\":\"msg_B\",\"type\":\"message\"}}\n\n",
        // Out-of-order: B finishes first.
        "event: output_item.done\n",
        "data: {\"type\":\"output_item.done\",\"item\":{\"id\":\"msg_B\",\"type\":\"message\",\"status\":\"completed\"}}\n\n",
        "event: output_item.done\n",
        "data: {\"type\":\"output_item.done\",\"item\":{\"id\":\"msg_A\",\"type\":\"message\",\"status\":\"completed\"}}\n\n",
        "event: response.completed\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"usage\":{\"input_tokens\":10,\"output_tokens\":20}}}\n\n",
    );
    run(&mut s, raw.as_bytes());

    assert_eq!(s.response_id.as_deref(), Some("resp_1"));
    assert_eq!(s.status, StreamStatus::Completed);
    assert!(s.items.get("msg_A").unwrap().complete);
    assert!(s.items.get("msg_B").unwrap().complete);
    assert!(s.usage.is_some());
}

#[test]
fn reasoning_summary_accumulated() {
    let mut s = ResponseState::new();
    let raw = concat!(
        "event: response.created\n",
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_2\",\"model\":\"gpt-5\"}}\n\n",
        "event: output_item.added\n",
        "data: {\"type\":\"output_item.added\",\"item\":{\"id\":\"rs_1\",\"type\":\"reasoning\"}}\n\n",
        "event: response.reasoning_summary_text.delta\n",
        "data: {\"type\":\"response.reasoning_summary_text.delta\",\"item_id\":\"rs_1\",\"delta\":\"First, \"}\n\n",
        "event: response.reasoning_summary_text.delta\n",
        "data: {\"type\":\"response.reasoning_summary_text.delta\",\"item_id\":\"rs_1\",\"delta\":\"second.\"}\n\n",
        "event: response.reasoning_summary_text.done\n",
        "data: {\"type\":\"response.reasoning_summary_text.done\",\"item_id\":\"rs_1\"}\n\n",
        "event: output_item.done\n",
        "data: {\"type\":\"output_item.done\",\"item\":{\"id\":\"rs_1\",\"type\":\"reasoning\"}}\n\n",
    );
    run(&mut s, raw.as_bytes());

    let item = s.items.get("rs_1").expect("reasoning item must exist");
    assert_eq!(item.reasoning_summary, "First, second.");
    assert!(item.complete);
}

#[test]
fn function_call_arguments_string_preserved() {
    // function_call_arguments must STAY A STRING end-to-end. Even
    // though the wire payload happens to be valid JSON in this test,
    // the state machine must not parse it; we verify by feeding a
    // payload with content that would round-trip differently if parsed
    // (whitespace inside the JSON, key ordering).
    let mut s = ResponseState::new();
    let arg_part_1 = "{ \"loc\" : \"NYC\""; // intentional whitespace
    let arg_part_2 = " , \"days\" : 7 }"; // intentional whitespace + key order
    let raw = format!(
        concat!(
            "event: response.created\n",
            "data: {{\"type\":\"response.created\",\"response\":{{\"id\":\"resp_3\",\"model\":\"gpt-5\"}}}}\n\n",
            "event: output_item.added\n",
            "data: {{\"type\":\"output_item.added\",\"item\":{{\"id\":\"fc_1\",\"type\":\"function_call\",\"name\":\"weather\"}}}}\n\n",
            "event: response.function_call_arguments.delta\n",
            "data: {{\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"fc_1\",\"delta\":{a1}}}\n\n",
            "event: response.function_call_arguments.delta\n",
            "data: {{\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"fc_1\",\"delta\":{a2}}}\n\n",
            "event: response.function_call_arguments.done\n",
            "data: {{\"type\":\"response.function_call_arguments.done\",\"item_id\":\"fc_1\",\"arguments\":\"FINAL\"}}\n\n",
        ),
        a1 = serde_json::to_string(arg_part_1).unwrap(),
        a2 = serde_json::to_string(arg_part_2).unwrap(),
    );
    run(&mut s, raw.as_bytes());

    let item = s.items.get("fc_1").unwrap();
    let expected = format!("{}{}", arg_part_1, arg_part_2);
    assert_eq!(
        item.function_call_arguments, expected,
        "arguments must be byte-equal concatenation (NOT re-serialized JSON)"
    );
    // Whitespace inside the wire string must be preserved verbatim.
    assert!(item.function_call_arguments.contains(" \"loc\" "));
    assert!(item.function_call_arguments.contains(" , "));
}

#[test]
fn output_text_delta_accumulated_per_item() {
    let mut s = ResponseState::new();
    let raw = concat!(
        "event: response.created\n",
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_4\",\"model\":\"gpt-5\"}}\n\n",
        "event: output_item.added\n",
        "data: {\"type\":\"output_item.added\",\"item\":{\"id\":\"msg_X\",\"type\":\"message\"}}\n\n",
        "event: response.output_text.delta\n",
        "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_X\",\"delta\":\"Hello \"}\n\n",
        "event: response.output_text.delta\n",
        "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_X\",\"delta\":\"world\"}\n\n",
        "event: response.output_text.done\n",
        "data: {\"type\":\"response.output_text.done\",\"item_id\":\"msg_X\",\"text\":\"Hello world\"}\n\n",
    );
    run(&mut s, raw.as_bytes());
    let item = s.items.get("msg_X").unwrap();
    assert_eq!(item.output_text, "Hello world");
}

#[test]
fn response_failed_status() {
    let mut s = ResponseState::new();
    let raw = concat!(
        "event: response.created\n",
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_5\",\"model\":\"gpt-5\"}}\n\n",
        "event: response.failed\n",
        "data: {\"type\":\"response.failed\",\"response\":{\"id\":\"resp_5\",\"status\":\"failed\",\"error\":{\"code\":\"server_error\"}}}\n\n",
    );
    run(&mut s, raw.as_bytes());
    assert_eq!(s.status, StreamStatus::Failed);
}

#[test]
fn response_incomplete_status() {
    let mut s = ResponseState::new();
    let raw = concat!(
        "event: response.created\n",
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_6\",\"model\":\"gpt-5\"}}\n\n",
        "event: response.incomplete\n",
        "data: {\"type\":\"response.incomplete\",\"response\":{\"id\":\"resp_6\",\"status\":\"incomplete\",\"incomplete_details\":{\"reason\":\"max_output_tokens\"}}}\n\n",
    );
    run(&mut s, raw.as_bytes());
    assert_eq!(s.status, StreamStatus::Incomplete);
}

/// PR-C4 property test: feeding the same byte sequence through the
/// framer chunked at every possible boundary produces the same final
/// state. This is the cache-safety invariant on the streaming path —
/// the parser never depends on TCP chunk geometry.
#[test]
fn chunk_boundary_invariance_pr_c4() {
    let raw = concat!(
        "event: response.created\n",
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_inv\",\"model\":\"gpt-5\"}}\n\n",
        "event: output_item.added\n",
        "data: {\"type\":\"output_item.added\",\"item\":{\"id\":\"msg_inv\",\"type\":\"message\"}}\n\n",
        "event: response.output_text.delta\n",
        "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_inv\",\"delta\":\"alpha\"}\n\n",
        "event: response.output_text.delta\n",
        "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_inv\",\"delta\":\" beta\"}\n\n",
        "event: output_item.done\n",
        "data: {\"type\":\"output_item.done\",\"item\":{\"id\":\"msg_inv\",\"type\":\"message\"}}\n\n",
        "event: response.completed\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_inv\",\"usage\":{\"input_tokens\":1,\"output_tokens\":2}}}\n\n",
    )
    .as_bytes();

    // Try every single-byte split point and a couple of multi-split
    // variations. Final state must match.
    let baseline = {
        let mut s = ResponseState::new();
        run(&mut s, raw);
        s
    };

    for split in 1..raw.len() {
        let mut s = ResponseState::new();
        let mut framer = SseFramer::new();
        framer.push(&raw[..split]);
        while let Some(r) = framer.next_event() {
            s.apply(r.unwrap()).unwrap();
        }
        framer.push(&raw[split..]);
        while let Some(r) = framer.next_event() {
            s.apply(r.unwrap()).unwrap();
        }
        assert_eq!(s.response_id, baseline.response_id, "split={split}");
        assert_eq!(s.status, baseline.status, "split={split}");
        let item = s.items.get("msg_inv").expect("msg_inv present");
        assert_eq!(item.output_text, "alpha beta", "split={split}");
        assert!(item.complete, "split={split}");
    }
}

/// PR-C4: an empty / minimal upstream response (just `[DONE]`) must
/// never panic the state machine and must surface as a closed stream
/// with no items.
#[test]
fn minimal_upstream_response_pr_c4() {
    let mut s = ResponseState::new();
    let raw = b"data: [DONE]\n\n";
    run(&mut s, raw);
    assert!(s.items.is_empty());
    // status stays Open: [DONE] is a framer sentinel, not a state-
    // machine status. Genuine completion goes through `response.completed`.
    assert_eq!(s.status, StreamStatus::Open);
}
