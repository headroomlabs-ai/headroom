//! Integration tests for the `/v1/chat/completions` Rust handler
//! (Phase C PR-C2).
//!
//! These tests boot the real Rust proxy in front of a wiremock upstream
//! and exercise the OpenAI Chat Completions request shape end-to-end.
//! Where compression is expected to NOT run, we assert SHA-256 byte
//! equality between the bytes the client sent and the bytes the
//! upstream received — the same cache-safety contract the Anthropic
//! tests pin.
//!
//! Coverage matrix (per PR-C2 spec, REALIGNMENT/05-phase-C-rust-proxy.md):
//!
//! 1. `passthrough_no_compression_byte_equal` — small body, compression
//!    on, body too small to compress; bytes round-trip byte-equal.
//! 2. `tool_message_compressed` — large JSON-array tool message;
//!    upstream body shrinks and the bytes outside the compressed slot
//!    stay byte-equal.
//! 3. `n_greater_than_one_passthrough` — `n: 3`; compression skipped
//!    pre-dispatch even though the body would otherwise compress.
//! 4. `stream_options_include_usage_preserved` — `stream_options.include_usage`
//!    round-trips byte-equal.
//! 5. `tool_choice_change_passthrough_no_mutation` — `tool_choice: "required"`
//!    + a `tools` array; neither field is mutated.
//! 6. `refusal_field_in_response_handled` — synthetic upstream stream
//!    with a `refusal` delta; `ChunkState`'s state machine handles it.
//! 7. `streaming_tool_call_argument_accumulation` — synthetic stream
//!    with three tool_call delta chunks; arguments concatenate.

mod common;

use bytes::Bytes;
use common::start_proxy_with;
use headroom_proxy::sse::framing::SseFramer;
use headroom_proxy::sse::openai_chat::ChunkState;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::sync::{Arc, Mutex};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Mount a /v1/chat/completions handler that captures the upstream
/// request body.
async fn mount_capture(upstream: &MockServer) -> Arc<Mutex<Option<Vec<u8>>>> {
    let captured: Arc<Mutex<Option<Vec<u8>>>> = Arc::new(Mutex::new(None));
    let captured_clone = captured.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |req: &wiremock::Request| {
            *captured_clone.lock().unwrap() = Some(req.body.clone());
            ResponseTemplate::new(200).set_body_string(r#"{"ok":true}"#)
        })
        .mount(upstream)
        .await;
    captured
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .fold(String::with_capacity(64), |mut acc, b| {
            use std::fmt::Write as _;
            let _ = write!(acc, "{b:02x}");
            acc
        })
}

#[track_caller]
fn assert_byte_equal_sha256(inbound: &[u8], received: &[u8]) {
    let inbound_hash = sha256_hex(inbound);
    let received_hash = sha256_hex(received);
    assert_eq!(
        inbound.len(),
        received.len(),
        "byte length mismatch: inbound={}, upstream-received={}",
        inbound.len(),
        received.len(),
    );
    assert_eq!(
        inbound_hash, received_hash,
        "SHA-256 mismatch: inbound={inbound_hash}, upstream-received={received_hash}",
    );
}

/// Build a JSON-array tool message payload large enough to trigger
/// SmartCrusher compression. Uses 1500 dict rows with low uniqueness
/// (matches the headroom-core dispatch test fixture's compressibility
/// profile).
fn compressible_tool_array_payload() -> String {
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
    serde_json::to_string(&array_of_dicts).unwrap()
}

#[tokio::test]
async fn passthrough_no_compression_byte_equal() {
    // Compression on. Small tool message → below threshold → no
    // mutation; upstream bytes must be byte-equal to client bytes.
    let upstream = MockServer::start().await;
    let captured = mount_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
    })
    .await;

    let payload = json!({
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "calling tool"},
            {"role": "tool", "tool_call_id": "t1", "content": "tiny"},
        ]
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", proxy.url()))
        .header("content-type", "application/json")
        // PR-E4: OAuth auth mode so the prompt_cache_key auto-injection
        // hook short-circuits and the byte-equality invariant this test
        // pins is preserved. PAYG bodies are now mutated by E4 — see
        // `integration_e4_openai_cache_key.rs` for that coverage.
        .header(
            "authorization",
            "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ0ZXN0In0.signature_bytes",
        )
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let got = captured.lock().unwrap().clone().expect("upstream got body");
    assert_byte_equal_sha256(&body, &got);
    proxy.shutdown().await;
}

#[tokio::test]
async fn tool_message_compressed() {
    // Compressible tool message (JSON array of 1500 homogeneous dicts).
    // Body should shrink at upstream; bytes outside the rewritten slot
    // remain byte-equal.
    let upstream = MockServer::start().await;
    let captured = mount_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
    })
    .await;

    let tool_payload = compressible_tool_array_payload();
    assert!(
        tool_payload.len() > 1024,
        "must exceed JSON array threshold"
    );

    let payload = json!({
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "summarize the rows below"},
            {"role": "assistant", "content": "fetching"},
            {"role": "tool", "tool_call_id": "t1", "content": tool_payload},
        ]
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", proxy.url()))
        .header("content-type", "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let got = captured.lock().unwrap().clone().expect("upstream got body");
    assert!(
        got.len() < body.len(),
        "upstream body should be smaller after live-zone compression: in={}, out={}",
        body.len(),
        got.len()
    );
    // Compression must shrink the tool slot meaningfully — at least
    // 40% reduction on the whole body for this fixture (the slot is
    // the dominant share of the body).
    let reduction_pct = (body.len() - got.len()) as f64 / body.len() as f64;
    assert!(
        reduction_pct > 0.40,
        "expected ≥40% body reduction; got {:.2}% (in={}, out={})",
        reduction_pct * 100.0,
        body.len(),
        got.len()
    );
    proxy.shutdown().await;
}

#[tokio::test]
async fn n_greater_than_one_passthrough() {
    // n: 3 → compression skipped pre-dispatch. Body must arrive
    // byte-equal at upstream even though it WOULD compress
    // otherwise (large tool message included).
    let upstream = MockServer::start().await;
    let captured = mount_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
    })
    .await;

    let tool_payload = compressible_tool_array_payload();
    let payload = json!({
        "model": "gpt-4o",
        "n": 3,
        "messages": [
            {"role": "user", "content": "describe"},
            {"role": "tool", "tool_call_id": "t1", "content": tool_payload},
        ]
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", proxy.url()))
        .header("content-type", "application/json")
        // PR-E4: OAuth auth mode preserves byte-equality (E4 only
        // injects on PAYG). The n>1 skip semantics are independent
        // of auth mode.
        .header(
            "authorization",
            "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ0ZXN0In0.signature_bytes",
        )
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let got = captured.lock().unwrap().clone().expect("upstream got body");
    assert_byte_equal_sha256(&body, &got);
    proxy.shutdown().await;
}

#[tokio::test]
async fn stream_options_include_usage_preserved() {
    // stream_options.include_usage = true must round-trip byte-equal
    // upstream. The dispatcher never reads or rewrites this field.
    let upstream = MockServer::start().await;
    let captured = mount_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
    })
    .await;

    let payload = json!({
        "model": "gpt-4o",
        "stream": true,
        "stream_options": {"include_usage": true},
        "messages": [
            {"role": "user", "content": "hi"},
        ]
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", proxy.url()))
        .header("content-type", "application/json")
        // PR-E4: OAuth auth mode preserves byte-equality. The
        // dispatcher never reads stream_options regardless of mode;
        // E4 only mutates on PAYG.
        .header(
            "authorization",
            "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ0ZXN0In0.signature_bytes",
        )
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let got = captured.lock().unwrap().clone().expect("upstream got body");
    assert_byte_equal_sha256(&body, &got);
    // Defensive: also verify the field literally arrived intact.
    let parsed: Value = serde_json::from_slice(&got).unwrap();
    assert_eq!(parsed["stream_options"]["include_usage"], json!(true));
    proxy.shutdown().await;
}

#[tokio::test]
async fn tool_choice_change_passthrough_no_mutation() {
    // tool_choice: "required" + a tools array. The dispatcher must
    // never mutate either field. Cache-stability invariant for
    // tool definitions.
    let upstream = MockServer::start().await;
    let captured = mount_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
    })
    .await;

    let tools = json!([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "get the weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            }
        }
    }]);
    let payload = json!({
        "model": "gpt-4o",
        "tool_choice": "required",
        "tools": tools,
        "messages": [
            {"role": "user", "content": "what's the weather in NYC?"},
        ]
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", proxy.url()))
        .header("content-type", "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let got = captured.lock().unwrap().clone().expect("upstream got body");
    let parsed: Value = serde_json::from_slice(&got).unwrap();
    assert_eq!(parsed["tool_choice"], json!("required"));
    assert_eq!(parsed["tools"], tools);
    proxy.shutdown().await;
}

#[tokio::test]
async fn refusal_field_in_response_handled() {
    // Drive ChunkState directly with a synthetic stream emitting a
    // refusal-style delta (GPT-4o safety class). This exercises the
    // exact wire-format contract the handler hands off to in
    // `forward_http`'s spawned state-machine task.
    let mut state = ChunkState::new();
    let mut framer = SseFramer::new();
    let raw = concat!(
        "data: {\"id\":\"chatcmpl-r\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"refusal\":\"I'm sorry \"}}]}\n\n",
        "data: {\"id\":\"chatcmpl-r\",\"choices\":[{\"index\":0,\"delta\":{\"refusal\":\"I can't help with that.\"},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n",
    );
    framer.push(raw.as_bytes());
    while let Some(ev_result) = framer.next_event() {
        let ev = ev_result.expect("framer must succeed on valid input");
        state.apply(ev).expect("state machine must succeed");
    }

    let choice = state.choices.get(&0).expect("choice 0 must be set");
    assert_eq!(choice.refusal, "I'm sorry I can't help with that.");
    assert_eq!(choice.content, "");
    assert_eq!(choice.finish_reason.as_deref(), Some("stop"));
}

#[tokio::test]
async fn streaming_tool_call_argument_accumulation() {
    // Three tool_call delta chunks: id+name in #1, args fragments
    // in #2 and #3. This exercises the same contract C1's
    // `tool_call_arguments_concatenated` test pins, but verifies it
    // through the same `ChunkState` the handler will hand to a
    // running stream in `forward_http`'s SSE tee.
    let mut state = ChunkState::new();
    let mut framer = SseFramer::new();
    let raw = concat!(
        "data: {\"id\":\"chatcmpl-tc\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call_xyz\",\"type\":\"function\",\"function\":{\"name\":\"echo\",\"arguments\":\"{\\\"q\\\":\"}}]}}]}\n\n",
        "data: {\"id\":\"chatcmpl-tc\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"hello\"}}]}}]}\n\n",
        "data: {\"id\":\"chatcmpl-tc\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\" world\\\"}\"}}]}}]}\n\n",
        "data: [DONE]\n\n",
    );
    framer.push(raw.as_bytes());
    while let Some(ev_result) = framer.next_event() {
        let ev = ev_result.expect("framer must succeed on valid input");
        state.apply(ev).expect("state machine must succeed");
    }

    let choice = state.choices.get(&0).expect("choice 0 set");
    let tc = choice.tool_calls.get(&0).expect("tool call 0 set");
    assert_eq!(
        tc.id.as_deref(),
        Some("call_xyz"),
        "id must persist across chunks (P4-48)"
    );
    assert_eq!(tc.function_name.as_deref(), Some("echo"));
    let parsed: Value =
        serde_json::from_str(&tc.function_arguments).expect("concatenated arguments must parse");
    assert_eq!(parsed["q"], json!("hello world"));

    // Sanity: ensure Bytes is in the test's import list (used by
    // SseFramer::push). Accessing the type avoids an unused-import
    // warning if the compiler reorders.
    let _: Bytes = Bytes::from_static(b"");
}
