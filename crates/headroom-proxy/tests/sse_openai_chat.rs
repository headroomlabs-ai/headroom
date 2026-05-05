//! Integration tests for the OpenAI Chat Completions SSE state machine.
//!
//! Wire-format quirks under test (per realignment guide §5.2):
//!
//!   - Tool calls: `id` and `function.name` arrive ONLY on the first
//!     chunk per `index`. Subsequent chunks omit them; the proxy must
//!     NOT overwrite the cached values with `None` (P4-48).
//!   - `function.arguments` is concatenated as a STRING — never
//!     re-parsed as JSON mid-stream.
//!   - When `stream_options.include_usage = true`, the FINAL chunk
//!     carries `choices: []` and a populated `usage` object. Without
//!     that flag, `usage` is never sent over the stream.
//!   - The `refusal` field (GPT-4o safety-class responses) carries
//!     fragments to concatenate just like `content`.

use headroom_proxy::sse::openai_chat::{ChunkState, StreamStatus};
use headroom_proxy::sse::SseFramer;

fn run(state: &mut ChunkState, raw: &[u8]) {
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
fn tool_call_id_and_name_only_first_chunk() {
    let mut s = ChunkState::new();
    let raw = concat!(
        // First chunk: id + function.name + first arguments fragment.
        "data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call_abc\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"loc\\\":\"}}]}}]}\n\n",
        // Second chunk: NO id, NO function.name; just more arguments.
        // The Python proxy used to overwrite id with null here.
        "data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"NYC\\\"}\"}}]}}]}\n\n",
        "data: [DONE]\n\n",
    );
    run(&mut s, raw.as_bytes());

    let choice = s.choices.get(&0).expect("choice 0 must exist");
    let tc = choice.tool_calls.get(&0).expect("tool call 0 must exist");
    assert_eq!(
        tc.id.as_deref(),
        Some("call_abc"),
        "id must NOT be overwritten by the second chunk's missing id (P4-48)"
    );
    assert_eq!(tc.function_name.as_deref(), Some("get_weather"));
    assert_eq!(tc.call_type.as_deref(), Some("function"));
    assert_eq!(s.status, StreamStatus::Done);
}

#[test]
fn tool_call_arguments_concatenated() {
    let mut s = ChunkState::new();
    let raw = concat!(
        "data: {\"id\":\"chatcmpl-2\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"f\",\"arguments\":\"{\\\"a\\\":\"}}]}}]}\n\n",
        "data: {\"id\":\"chatcmpl-2\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"1,\"}}]}}]}\n\n",
        "data: {\"id\":\"chatcmpl-2\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"b\\\":2}\"}}]}}]}\n\n",
        "data: [DONE]\n\n",
    );
    run(&mut s, raw.as_bytes());

    let choice = s.choices.get(&0).unwrap();
    let tc = choice.tool_calls.get(&0).unwrap();
    assert_eq!(tc.function_arguments, r#"{"a":1,"b":2}"#);
    // The string MUST parse as JSON now that it's concatenated, but
    // the state machine itself doesn't parse mid-stream — that's the
    // contract we lock down here.
    let parsed: serde_json::Value =
        serde_json::from_str(&tc.function_arguments).expect("concatenated arguments must parse");
    assert_eq!(parsed["a"], 1);
    assert_eq!(parsed["b"], 2);
}

#[test]
fn usage_in_final_chunk_when_include_usage_set() {
    let mut s = ChunkState::new();
    let raw = concat!(
        // Body chunks with content fragments.
        "data: {\"id\":\"chatcmpl-3\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hi\"}}]}\n\n",
        "data: {\"id\":\"chatcmpl-3\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" there\"},\"finish_reason\":\"stop\"}]}\n\n",
        // Final usage-only chunk: choices is empty, usage is populated.
        "data: {\"id\":\"chatcmpl-3\",\"choices\":[],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":7,\"total_tokens\":19}}\n\n",
        "data: [DONE]\n\n",
    );
    run(&mut s, raw.as_bytes());

    let choice = s.choices.get(&0).unwrap();
    assert_eq!(choice.role.as_deref(), Some("assistant"));
    assert_eq!(choice.content, "hi there");
    assert_eq!(choice.finish_reason.as_deref(), Some("stop"));

    let usage = s.usage.as_ref().expect("usage must be set on final chunk");
    assert_eq!(usage["prompt_tokens"], 12);
    assert_eq!(usage["completion_tokens"], 7);
    assert_eq!(usage["total_tokens"], 19);
}

#[test]
fn refusal_field_handled() {
    // GPT-4o-class safety responses substitute `refusal` for `content`.
    // Both fields concatenate identically.
    let mut s = ChunkState::new();
    let raw = concat!(
        "data: {\"id\":\"chatcmpl-4\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"refusal\":\"I can't \"}}]}\n\n",
        "data: {\"id\":\"chatcmpl-4\",\"choices\":[{\"index\":0,\"delta\":{\"refusal\":\"help with that.\"},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n",
    );
    run(&mut s, raw.as_bytes());

    let choice = s.choices.get(&0).unwrap();
    assert_eq!(choice.refusal, "I can't help with that.");
    // Content stayed empty — refusal and content are mutually exclusive
    // in the wire format but both must be supported.
    assert_eq!(choice.content, "");
    assert_eq!(choice.finish_reason.as_deref(), Some("stop"));
}

#[test]
fn done_sentinel_terminates_stream_status() {
    let mut s = ChunkState::new();
    let raw = b"data: [DONE]\n\n";
    run(&mut s, raw);
    assert_eq!(s.status, StreamStatus::Done);
}

#[test]
fn multiple_choices_keyed_by_index() {
    // OpenAI's `n>1` mode emits multiple choices per chunk. Each must
    // be kept independent, keyed by `choice.index`.
    let mut s = ChunkState::new();
    let raw = concat!(
        "data: {\"id\":\"chatcmpl-5\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"A\"}},{\"index\":1,\"delta\":{\"role\":\"assistant\",\"content\":\"B\"}}]}\n\n",
        "data: {\"id\":\"chatcmpl-5\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"A2\"}},{\"index\":1,\"delta\":{\"content\":\"B2\"}}]}\n\n",
        "data: [DONE]\n\n",
    );
    run(&mut s, raw.as_bytes());

    assert_eq!(s.choices.get(&0).unwrap().content, "AA2");
    assert_eq!(s.choices.get(&1).unwrap().content, "BB2");
}
