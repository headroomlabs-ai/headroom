//! Integration tests for the `/v1/responses` streaming pipeline
//! (Phase C PR-C4).
//!
//! Per spec PR-C4:
//!
//! - When a `/v1/responses` request carries
//!   `Accept: text/event-stream`, the proxy:
//!   1. Still runs the C3 request-side live-zone compression
//!      (request body is byte-equal upstream when no compression
//!      applies; smaller when it does).
//!   2. Engages the SSE state-machine telemetry tee on the
//!      response stream — bytes flow back to the client unchanged
//!      and the byte-level `SseFramer` + `ResponseState` machine
//!      observe events in a parallel task.
//! - The streaming pipeline can be toggled via
//!   `Config::enable_responses_streaming` (default `true`). When
//!   `false`, the SSE bytes still pass through but the parser is
//!   not spun up.
//!
//! These tests cover the request→upstream byte fidelity
//! (request-side) and the response→client byte fidelity
//! (response-side) under a real wiremock upstream. The state
//! machine itself is unit-tested in `tests/sse_openai_responses.rs`.

mod common;

use bytes::Bytes;
use common::start_proxy_with;
use futures_util::StreamExt;
use headroom_proxy::sse::{openai_responses::ResponseState, SseFramer};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use http_body_util::StreamBody;
use hyper::body::Frame;
use hyper::service::service_fn;
use hyper::{Request, Response};
use hyper_util::rt::TokioIo;
use tokio::sync::Mutex;

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
fn assert_byte_equal(inbound: &[u8], received: &[u8]) {
    assert_eq!(
        inbound.len(),
        received.len(),
        "byte length mismatch: client={}, upstream={}",
        inbound.len(),
        received.len()
    );
    assert_eq!(
        sha256_hex(inbound),
        sha256_hex(received),
        "SHA-256 mismatch (client vs. upstream-received)"
    );
}

/// Hand-rolled hyper upstream that emits a representative
/// OpenAI-Responses SSE stream and captures the request body.
/// We can't use wiremock here because it doesn't speak streaming
/// response bodies — we need actual chunked frames over time.
async fn responses_sse_upstream() -> (
    SocketAddr,
    Arc<Mutex<Option<Vec<u8>>>>,
    tokio::task::JoinHandle<()>,
) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let captured: Arc<Mutex<Option<Vec<u8>>>> = Arc::new(Mutex::new(None));
    let captured_for_task = captured.clone();
    let task = tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                break;
            };
            let captured = captured_for_task.clone();
            tokio::spawn(async move {
                let io = TokioIo::new(stream);
                let _ = hyper::server::conn::http1::Builder::new()
                    .serve_connection(
                        io,
                        service_fn(move |req: Request<hyper::body::Incoming>| {
                            let captured = captured.clone();
                            async move {
                                use http_body_util::BodyExt;
                                // Capture the entire request body.
                                let body_bytes =
                                    req.into_body().collect().await.unwrap().to_bytes();
                                *captured.lock().await = Some(body_bytes.to_vec());

                                let (tx, rx) = tokio::sync::mpsc::channel::<
                                    Result<Frame<Bytes>, std::io::Error>,
                                >(8);

                                tokio::spawn(async move {
                                    // A representative OpenAI Responses SSE stream.
                                    // Mixes named events (`event:` lines) with the
                                    // typical `[DONE]` sentinel some clients still see.
                                    let frames: &[&[u8]] = &[
                                        b"event: response.created\n",
                                        b"data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_test\",\"model\":\"gpt-5\"}}\n\n",
                                        b"event: output_item.added\n",
                                        b"data: {\"type\":\"output_item.added\",\"item\":{\"id\":\"msg_1\",\"type\":\"message\"}}\n\n",
                                        b"event: response.output_text.delta\n",
                                        b"data: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_1\",\"delta\":\"Hello\"}\n\n",
                                        b"event: response.output_text.delta\n",
                                        b"data: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_1\",\"delta\":\" world\"}\n\n",
                                        b"event: response.output_text.done\n",
                                        b"data: {\"type\":\"response.output_text.done\",\"item_id\":\"msg_1\"}\n\n",
                                        b"event: output_item.done\n",
                                        b"data: {\"type\":\"output_item.done\",\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"status\":\"completed\"}}\n\n",
                                        b"event: response.completed\n",
                                        b"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_test\",\"usage\":{\"input_tokens\":5,\"output_tokens\":2}}}\n\n",
                                    ];
                                    for f in frames {
                                        if tx
                                            .send(Ok(Frame::data(Bytes::from_static(f))))
                                            .await
                                            .is_err()
                                        {
                                            return;
                                        }
                                        tokio::time::sleep(Duration::from_millis(15)).await;
                                    }
                                });

                                let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
                                let body = StreamBody::new(stream);
                                Ok::<_, Infallible>(
                                    Response::builder()
                                        .status(200)
                                        .header("content-type", "text/event-stream")
                                        .header("cache-control", "no-cache")
                                        .body(body)
                                        .unwrap(),
                                )
                            }
                        }),
                    )
                    .await;
            });
        }
    });
    (addr, captured, task)
}

/// Tiny representative request body — the client sends this with
/// `Accept: text/event-stream`. Below the 2 KiB output-item floor,
/// so request-side compression is a no-op and bytes round-trip equal.
fn small_responses_payload() -> Vec<u8> {
    let payload = json!({
        "model": "gpt-5",
        "stream": true,
        "input": [
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "say hi"}]}
        ]
    });
    serde_json::to_vec(&payload).unwrap()
}

#[tokio::test]
async fn streaming_request_bytes_byte_equal_upstream() {
    let (addr, captured, _server) = responses_sse_upstream().await;
    let proxy = start_proxy_with(&format!("http://{addr}"), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
        // Default ON, but pin it explicitly so the test pins behaviour
        // even if the project default flips later.
        c.enable_responses_streaming = true;
    })
    .await;

    let body = small_responses_payload();
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/responses", proxy.url()))
        .header("content-type", "application/json")
        .header("accept", "text/event-stream")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    // Drain the response so the upstream task finishes and capture lands.
    let _ = resp.bytes().await.unwrap();

    let got = captured
        .lock()
        .await
        .clone()
        .expect("upstream must observe a request body");
    assert_byte_equal(&body, &got);
    proxy.shutdown().await;
}

#[tokio::test]
async fn streaming_response_round_trips_through_framer() {
    // Engage the streaming pipeline and verify the bytes the client
    // receives parse cleanly through the SAME `SseFramer` +
    // `ResponseState` the proxy spawns internally. This is the
    // round-trip property: any upstream sequence the framer accepts
    // must reach the client unmodified.
    let (addr, _captured, _server) = responses_sse_upstream().await;
    let proxy = start_proxy_with(&format!("http://{addr}"), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
        c.enable_responses_streaming = true;
    })
    .await;

    let body = small_responses_payload();
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/responses", proxy.url()))
        .header("content-type", "application/json")
        .header("accept", "text/event-stream")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.headers().get("content-type").unwrap(),
        "text/event-stream"
    );
    let mut stream = resp.bytes_stream();

    // Drain the body, feed each chunk into a real framer, and run
    // the same state machine the proxy uses. End-state must reflect
    // the upstream's emitted events (id, items, completed status).
    let mut framer = SseFramer::new();
    let mut state = ResponseState::new();
    let mut total_bytes = 0usize;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.expect("client byte stream must not error mid-response");
        total_bytes += chunk.len();
        framer.push(&chunk);
        while let Some(ev_result) = framer.next_event() {
            let ev = ev_result.expect("framer parses upstream-faithful bytes");
            state
                .apply(ev)
                .expect("state machine handles representative stream");
        }
    }
    // The upstream emitted ~1.2 KiB of SSE; assert non-trivial payload
    // arrived (no premature truncation) and the state machine reached
    // a terminal state.
    assert!(
        total_bytes > 200,
        "expected non-trivial response payload, got {total_bytes} bytes"
    );
    assert_eq!(state.response_id.as_deref(), Some("resp_test"));
    assert_eq!(
        state.status,
        headroom_proxy::sse::openai_responses::StreamStatus::Completed
    );
    assert!(state.items.contains_key("msg_1"));
    let item = state.items.get("msg_1").unwrap();
    assert!(item.complete, "msg_1 must be marked complete");
    assert_eq!(item.output_text, "Hello world");

    proxy.shutdown().await;
}

#[tokio::test]
async fn streaming_pipeline_disabled_still_passes_bytes() {
    // Emergency-rollback path: when the operator flips
    // `enable_responses_streaming=false`, the SSE state machine is
    // skipped (a structured-log breadcrumb says so in proxy.rs), but
    // the bytes still flow client-side. This test pins the
    // "rollback never breaks the byte path" contract.
    let (addr, _captured, _server) = responses_sse_upstream().await;
    let proxy = start_proxy_with(&format!("http://{addr}"), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
        c.enable_responses_streaming = false;
    })
    .await;

    let body = small_responses_payload();
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/responses", proxy.url()))
        .header("content-type", "application/json")
        .header("accept", "text/event-stream")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let mut stream = resp.bytes_stream();
    let mut all = Vec::new();
    while let Some(chunk) = stream.next().await {
        all.extend_from_slice(&chunk.unwrap());
    }
    // The upstream emitted recognisable event names; without parsing
    // we just need to see the wire bytes survive the rollback.
    let body_str = String::from_utf8_lossy(&all);
    assert!(body_str.contains("response.created"));
    assert!(body_str.contains("response.completed"));

    proxy.shutdown().await;
}

#[tokio::test]
async fn streaming_request_no_compression_when_input_below_threshold() {
    // Pin the C3-style invariant on the streaming path: a streaming
    // request whose input is below the 2 KiB floor MUST round-trip
    // byte-equal upstream, regardless of `Accept: text/event-stream`.
    let (addr, captured, _server) = responses_sse_upstream().await;
    let proxy = start_proxy_with(&format!("http://{addr}"), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
    })
    .await;

    let payload = json!({
        "model": "gpt-5",
        "stream": true,
        "input": [
            {"type": "function_call_output", "id": "fco_1", "call_id": "c1",
             "output": "tiny output"},
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "do the thing"}]}
        ]
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/responses", proxy.url()))
        .header("content-type", "application/json")
        .header("accept", "text/event-stream")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let _ = resp.bytes().await.unwrap();

    let got = captured.lock().await.clone().expect("upstream got body");
    assert_byte_equal(&body, &got);
    proxy.shutdown().await;
}
