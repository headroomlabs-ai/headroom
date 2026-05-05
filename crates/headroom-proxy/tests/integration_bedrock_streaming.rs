//! Integration tests for the native Bedrock streaming route
//! (Phase D PR-D2).
//!
//! These tests exercise the binary EventStream parser, the SSE
//! translator, and the full POST `/model/{model}/invoke-with-response-stream`
//! handler. The upstream is a wiremock server that serves
//! `application/vnd.amazon.eventstream` bytes — no real AWS.
//!
//! Coverage matrix (per PR-D2 spec, REALIGNMENT/06-phase-D-bedrock-vertex.md):
//!
//! 1. `eventstream_parses_correctly` — feed known-good binary bytes
//!    to the parser; assert message boundaries, CRCs, and headers.
//! 2. `eventstream_translated_to_sse` — drive the proxy with a binary
//!    upstream; assert it emits valid `data: ...\n\n` SSE frames.
//! 3. `usage_extracted_from_translated_stream` — assert
//!    `AnthropicStreamState` accumulates `input_tokens` /
//!    `output_tokens` from the translated stream (via log capture).
//! 4. `client_can_choose_eventstream_or_sse` — `Accept: vnd.amazon.eventstream`
//!    returns binary unchanged; default Accept returns SSE.
//! 5. `eventstream_parser_no_panic` — proptest property: arbitrary
//!    bytes must never panic the parser.

mod common;

use aws_credential_types::Credentials;
use bytes::{Bytes, BytesMut};
use common::start_proxy_with_state;
use headroom_proxy::bedrock::{
    parse_eventstream, CrcValidation, EventStreamParser, HeaderValue, MessageBuilder, ParseError,
};
use proptest::prelude::*;
use serde_json::json;
use url::Url;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const TEST_MODEL: &str = "anthropic.claude-3-haiku-20240307-v1:0";

fn test_credentials() -> Credentials {
    Credentials::new(
        "AKIAEXAMPLEAKIDFORTEST",
        "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        None,
        None,
        "test",
    )
}

/// Synthesise a minimal Bedrock-shape Anthropic stream as binary
/// EventStream bytes. The exact JSON payload mirrors what real
/// Bedrock emits for a 2-token completion.
fn synthesize_bedrock_stream() -> Bytes {
    let mut buf = BytesMut::new();
    let events = [
        json!({
            "type": "message_start",
            "message": {
                "id": "msg_01ABCDEF",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-haiku-20240307",
                "content": [],
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {"input_tokens": 7, "output_tokens": 1}
            }
        }),
        json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""}
        }),
        json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "OK"}
        }),
        json!({
            "type": "content_block_stop",
            "index": 0
        }),
        json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": null},
            "usage": {"output_tokens": 2}
        }),
        json!({
            "type": "message_stop"
        }),
    ];
    for ev in events {
        let payload = serde_json::to_string(&ev).unwrap();
        let bytes = MessageBuilder::new()
            .header_string(":event-type", "chunk")
            .header_string(":content-type", "application/json")
            .header_string(":message-type", "event")
            .payload(Bytes::from(payload))
            .build();
        buf.extend_from_slice(&bytes);
    }
    buf.freeze()
}

async fn bedrock_proxy(
    upstream: &MockServer,
    customize: impl FnOnce(&mut headroom_proxy::Config),
) -> common::ProxyHandle {
    let endpoint: Url = upstream.uri().parse().unwrap();
    start_proxy_with_state(
        &upstream.uri(),
        |c| {
            c.bedrock_endpoint = Some(endpoint);
            customize(c);
        },
        |s| s.with_bedrock_credentials(test_credentials()),
    )
    .await
}

// ─── Test 1: Parser unit-style integration ─────────────────────────

#[test]
fn eventstream_parses_correctly() {
    // Known-good 6-message stream — exercise the parser end to end.
    let bytes = synthesize_bedrock_stream();
    let mut parser = EventStreamParser::new();
    parser.push(&bytes);

    let mut messages = Vec::new();
    while let Some(msg) = parser.next_message().expect("parse ok") {
        messages.push(msg);
    }
    assert_eq!(messages.len(), 6, "must parse all 6 messages");

    // Headers + payload sanity:
    assert_eq!(messages[0].event_type(), Some("chunk"));
    assert_eq!(messages[0].message_type(), Some("event"));
    let p0 = std::str::from_utf8(&messages[0].payload).unwrap();
    assert!(
        p0.contains("\"message_start\""),
        "first payload must be message_start; got {p0}"
    );

    // Last payload is message_stop.
    let p_last = std::str::from_utf8(&messages[5].payload).unwrap();
    assert!(p_last.contains("\"message_stop\""));

    // Every message has the standard Bedrock headers.
    for m in &messages {
        assert!(
            matches!(
                m.headers.get(":event-type").unwrap(),
                HeaderValue::String(s) if s == "chunk"
            ),
            ":event-type must be chunk on every chunk frame"
        );
    }

    // Buffer drained exactly.
    assert_eq!(parser.buffered_len(), 0);
}

#[test]
fn eventstream_parses_correctly_one_byte_at_a_time() {
    // Same data, but drip-fed one byte at a time. Verifies the
    // parser is truly incremental (no internal "must have whole
    // chunk" assumption).
    let bytes = synthesize_bedrock_stream();
    let mut parser = EventStreamParser::new();
    let mut messages = Vec::new();
    for b in bytes.iter() {
        parser.push(std::slice::from_ref(b));
        while let Some(msg) = parser.next_message().expect("parse ok") {
            messages.push(msg);
        }
    }
    assert_eq!(messages.len(), 6);
}

#[test]
fn eventstream_crc_mismatch_surfaces_structured_error() {
    let mut bytes: Vec<u8> = synthesize_bedrock_stream().to_vec();
    bytes[8] ^= 0x01; // corrupt prelude CRC of first message
    let err = parse_eventstream(&bytes).unwrap_err();
    assert!(
        matches!(err, ParseError::PreludeCrcMismatch { .. }),
        "expected PreludeCrcMismatch; got {err:?}"
    );
}

#[test]
fn eventstream_validation_off_accepts_corrupt() {
    let mut bytes: Vec<u8> = synthesize_bedrock_stream().to_vec();
    bytes[8] ^= 0x01;
    let mut parser = EventStreamParser::new().with_crc_validation(CrcValidation::No);
    parser.push(&bytes);
    // Should still produce 6 messages even with corrupt CRC.
    let mut count = 0;
    while parser.next_message().unwrap().is_some() {
        count += 1;
    }
    assert_eq!(count, 6);
}

// ─── Test 2: End-to-end Bedrock binary → SSE translation ──────────

async fn mount_eventstream_upstream(upstream: &MockServer, body: Bytes) {
    Mock::given(method("POST"))
        .and(path(format!(
            "/model/{TEST_MODEL}/invoke-with-response-stream"
        )))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/vnd.amazon.eventstream")
                .set_body_bytes(body.to_vec()),
        )
        .mount(upstream)
        .await;
}

#[tokio::test]
async fn eventstream_translated_to_sse() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_test_writer()
        .try_init();
    let upstream = MockServer::start().await;
    let bedrock_bytes = synthesize_bedrock_stream();
    mount_eventstream_upstream(&upstream, bedrock_bytes).await;
    let proxy = bedrock_proxy(&upstream, |c| {
        c.compression_mode = headroom_proxy::config::CompressionMode::Off;
    })
    .await;

    let body = serde_json::to_vec(&json!({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 16,
        "messages": [{"role":"user","content":"hi"}]
    }))
    .unwrap();
    let resp = reqwest::Client::new()
        .post(format!(
            "{}/model/{TEST_MODEL}/invoke-with-response-stream",
            proxy.url()
        ))
        .header("content-type", "application/json")
        // Default Accept → SSE translation.
        .header("accept", "text/event-stream")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    assert!(
        ct.contains("text/event-stream"),
        "translated response must declare text/event-stream; got {ct}"
    );
    let body = resp.bytes().await.unwrap();
    let body_str = std::str::from_utf8(&body).unwrap();

    // Each message becomes its own `data: ...\n\n` frame.
    let frame_count = body_str.matches("data: ").count();
    assert_eq!(
        frame_count, 6,
        "expected 6 SSE frames (one per upstream chunk message); got {frame_count}: {body_str}"
    );

    // Frame ordering: message_start first, message_stop last.
    let first_frame_pos = body_str
        .find("\"message_start\"")
        .expect("message_start in stream");
    let last_frame_pos = body_str
        .find("\"message_stop\"")
        .expect("message_stop in stream");
    assert!(first_frame_pos < last_frame_pos);

    // Telemetry sanity: text_delta payload preserved verbatim.
    assert!(
        body_str.contains("\"text\":\"OK\""),
        "text_delta payload must round-trip through translation"
    );
    proxy.shutdown().await;
}

// ─── Test 3: Usage extraction from translated stream ──────────────

#[tokio::test]
async fn usage_extracted_from_translated_stream() {
    // The AnthropicStreamState runs in a spawned task on the
    // translated SSE stream. Its log line emits `output_tokens` at
    // stream close. We assert the byte-equal contract on the SSE
    // payload: input_tokens=7 and output_tokens=2 must show up in
    // the JSON the client received (so the state machine, which
    // reads the same bytes the client does, would extract them).
    let upstream = MockServer::start().await;
    let bedrock_bytes = synthesize_bedrock_stream();
    mount_eventstream_upstream(&upstream, bedrock_bytes).await;
    let proxy = bedrock_proxy(&upstream, |c| {
        c.compression_mode = headroom_proxy::config::CompressionMode::Off;
    })
    .await;

    let body = serde_json::to_vec(&json!({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 16,
        "messages": [{"role":"user","content":"hi"}]
    }))
    .unwrap();
    let resp = reqwest::Client::new()
        .post(format!(
            "{}/model/{TEST_MODEL}/invoke-with-response-stream",
            proxy.url()
        ))
        .header("content-type", "application/json")
        .header("accept", "text/event-stream")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body = resp.bytes().await.unwrap();
    let body_str = std::str::from_utf8(&body).unwrap();

    // Confirm the usage JSON survived translation:
    assert!(
        body_str.contains("\"input_tokens\":7"),
        "input_tokens must be preserved through translation: {body_str}"
    );
    assert!(
        body_str.contains("\"output_tokens\":2"),
        "final output_tokens must be preserved through translation: {body_str}"
    );

    // Drive the state machine ourselves over the wire bytes to
    // assert what the in-task instance would have computed. (We
    // can't easily intercept the spawned task's tracing output in
    // this test without a custom subscriber.)
    use headroom_proxy::sse::anthropic::AnthropicStreamState;
    use headroom_proxy::sse::framing::SseFramer;
    let mut framer = SseFramer::new();
    framer.push(&body);
    let mut state = AnthropicStreamState::new();
    while let Some(ev) = framer.next_event() {
        let ev = ev.expect("framer ok");
        state.apply(ev).expect("apply ok");
    }
    assert_eq!(state.usage.input_tokens, 7, "state input tokens");
    assert_eq!(state.usage.output_tokens, 2, "state output tokens");
    assert_eq!(state.stop_reason.as_deref(), Some("end_turn"));
    proxy.shutdown().await;
}

// ─── Test 4: Client picks eventstream vs sse via Accept ───────────

#[tokio::test]
async fn client_can_choose_eventstream_or_sse() {
    let upstream = MockServer::start().await;
    let bedrock_bytes = synthesize_bedrock_stream();
    mount_eventstream_upstream(&upstream, bedrock_bytes.clone()).await;
    let proxy = bedrock_proxy(&upstream, |c| {
        c.compression_mode = headroom_proxy::config::CompressionMode::Off;
    })
    .await;

    let body = serde_json::to_vec(&json!({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 16,
        "messages": [{"role":"user","content":"hi"}]
    }))
    .unwrap();

    // Case A: Accept: vnd.amazon.eventstream → byte-equal passthrough.
    let resp = reqwest::Client::new()
        .post(format!(
            "{}/model/{TEST_MODEL}/invoke-with-response-stream",
            proxy.url()
        ))
        .header("content-type", "application/json")
        .header("accept", "application/vnd.amazon.eventstream")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let resp_ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    assert!(
        resp_ct.contains("application/vnd.amazon.eventstream"),
        "passthrough mode must echo the eventstream content-type; got {resp_ct}"
    );
    let bin = resp.bytes().await.unwrap();
    assert_eq!(
        bin.as_ref(),
        bedrock_bytes.as_ref(),
        "binary passthrough must be byte-equal upstream"
    );

    // Case B: Accept: text/event-stream → translation.
    let resp_sse = reqwest::Client::new()
        .post(format!(
            "{}/model/{TEST_MODEL}/invoke-with-response-stream",
            proxy.url()
        ))
        .header("content-type", "application/json")
        .header("accept", "text/event-stream")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp_sse.status(), 200);
    let sse_ct = resp_sse
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    assert!(
        sse_ct.contains("text/event-stream"),
        "sse mode must declare text/event-stream; got {sse_ct}"
    );
    let sse_body = resp_sse.bytes().await.unwrap();
    let sse_str = std::str::from_utf8(&sse_body).unwrap();
    assert!(
        sse_str.contains("data: "),
        "sse output must contain data: frames"
    );
    assert!(!sse_str.starts_with("\0\0\0"), "must not be raw binary");

    // Case C: no Accept header → defaults to SSE.
    let resp_default = reqwest::Client::new()
        .post(format!(
            "{}/model/{TEST_MODEL}/invoke-with-response-stream",
            proxy.url()
        ))
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp_default.status(), 200);
    let default_ct = resp_default
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    assert!(
        default_ct.contains("text/event-stream"),
        "default Accept must select sse translation; got {default_ct}"
    );

    proxy.shutdown().await;
}

// ─── Test 5: Property test — never panic on adversarial bytes ─────

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 1024,
        max_shrink_iters: 1024,
        ..ProptestConfig::default()
    })]

    /// Per `feedback_realignment_build_constraints.md`, every parser
    /// in the project must terminate without panic on arbitrary input.
    /// The proxy faces TCP, which can deliver any byte sequence
    /// (corruption, truncation, deliberate fuzzing). The test pushes
    /// random bytes into the parser and drains until exhausted.
    /// Either Ok or Err is acceptable; what is NOT acceptable is a
    /// panic.
    #[test]
    fn eventstream_parser_no_panic(
        bytes in proptest::collection::vec(any::<u8>(), 0..4096)
    ) {
        let mut parser = EventStreamParser::new();
        parser.push(&bytes);
        loop {
            match parser.next_message() {
                Ok(None) => break,
                Ok(Some(_)) => continue,
                Err(_) => break,
            }
        }
        // The one-shot helper must also be panic-safe.
        let _ = parse_eventstream(&bytes);
    }

    /// Same input space but feeds the bytes one at a time. The
    /// parser must remain panic-safe even when chunk boundaries
    /// fall mid-prelude or mid-header.
    #[test]
    fn eventstream_parser_no_panic_one_byte_at_a_time(
        bytes in proptest::collection::vec(any::<u8>(), 0..1024)
    ) {
        let mut parser = EventStreamParser::new();
        for b in &bytes {
            parser.push(std::slice::from_ref(b));
            loop {
                match parser.next_message() {
                    Ok(None) => break,
                    Ok(Some(_)) => continue,
                    Err(_) => break,
                }
            }
        }
    }
}
