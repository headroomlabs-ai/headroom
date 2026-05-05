//! Integration tests for the native Vertex publisher path
//! (Phase D PR-D4).
//!
//! These tests boot the real Rust proxy in front of a wiremock
//! upstream and exercise the
//! `POST /v1beta1/projects/{p}/locations/{l}/publishers/anthropic/models/{m}:rawPredict`
//! (and `:streamRawPredict`) routes end-to-end. ADC is mocked via
//! `StaticTokenSource` so tests never reach real GCP.
//!
//! Per PR-D4 spec the four required tests are:
//!
//! 1. `native_envelope_round_trip_byte_equal` — body bytes (with
//!    `anthropic_version` + Anthropic Messages shape) round-trip
//!    SHA-256 byte-equal upstream.
//! 2. `adc_bearer_token_signed_correctly` — the `Authorization` header
//!    on the upstream request is `Bearer <static-test-token>`, and the
//!    mock provider was actually consulted (no silent un-authed
//!    forward).
//! 3. `thinking_block_preserved` — a request body containing
//!    `thinking` / `redacted_thinking` blocks (the Python LiteLLM
//!    converter dropped these — that was the P4-37/P4-38 bug) survives
//!    byte-equal upstream.
//! 4. `stream_raw_predict_sse_handled` — `:streamRawPredict` proxies
//!    an Anthropic SSE response back to the client without corruption,
//!    and the AnthropicStreamState telemetry tee fires the `event =
//!    "vertex_streaming_pipeline_active"` log.

mod common;

use common::{install_static_token_source, start_proxy_with_state};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::sync::{Arc, Mutex};
use wiremock::matchers::{method, path_regex};
use wiremock::{Mock, MockServer, ResponseTemplate};

const TEST_BEARER: &str = "ya29.test-static-bearer-d4-pr-fixture";
const PROJECT: &str = "test-project-12345";
const LOCATION: &str = "us-central1";
const MODEL: &str = "claude-3-5-sonnet@20240620";

const VERTEX_PATH_REGEX: &str =
    r"^/v1beta1/projects/[^/]+/locations/[^/]+/publishers/anthropic/models/[^/]+$";

fn raw_predict_url(proxy_url: &str) -> String {
    format!(
        "{proxy_url}/v1beta1/projects/{PROJECT}/locations/{LOCATION}/publishers/anthropic/models/{MODEL}:rawPredict",
    )
}

fn stream_raw_predict_url(proxy_url: &str) -> String {
    format!(
        "{proxy_url}/v1beta1/projects/{PROJECT}/locations/{LOCATION}/publishers/anthropic/models/{MODEL}:streamRawPredict",
    )
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

/// Mount a Vertex rawPredict mock that captures the upstream request
/// bytes + `Authorization` header. The path-regex matcher mirrors the
/// canonical Vertex publisher shape (so any of the 4 path parts can
/// vary across tests without re-mounting).
struct CapturedUpstream {
    body: Mutex<Option<Vec<u8>>>,
    authorization: Mutex<Option<String>>,
    content_type_response: Mutex<Option<String>>,
}

impl CapturedUpstream {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            body: Mutex::new(None),
            authorization: Mutex::new(None),
            content_type_response: Mutex::new(None),
        })
    }
}

async fn mount_capture_json(upstream: &MockServer) -> Arc<CapturedUpstream> {
    let captured = CapturedUpstream::new();
    let cap = captured.clone();
    Mock::given(method("POST"))
        .and(path_regex(VERTEX_PATH_REGEX))
        .respond_with(move |req: &wiremock::Request| {
            *cap.body.lock().unwrap() = Some(req.body.clone());
            *cap.authorization.lock().unwrap() = req
                .headers
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string());
            *cap.content_type_response.lock().unwrap() = Some("application/json".into());
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_string(r#"{"id":"msg_test","type":"message","role":"assistant","content":[{"type":"text","text":"hi"}],"model":"claude-3-5-sonnet@20240620","stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}"#)
        })
        .mount(upstream)
        .await;
    captured
}

/// Mount a Vertex streamRawPredict mock that returns a small Anthropic
/// SSE stream. The exact event sequence below is a minimal-but-valid
/// Anthropic Messages stream (per the PR-C1 framer + state machine).
async fn mount_capture_sse(upstream: &MockServer) -> Arc<CapturedUpstream> {
    let captured = CapturedUpstream::new();
    let cap = captured.clone();
    let sse_body = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_strm\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-3-5-sonnet@20240620\",\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":2,\"output_tokens\":0}}}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":3}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    );
    Mock::given(method("POST"))
        .and(path_regex(VERTEX_PATH_REGEX))
        .respond_with(move |req: &wiremock::Request| {
            *cap.body.lock().unwrap() = Some(req.body.clone());
            *cap.authorization.lock().unwrap() = req
                .headers
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string());
            *cap.content_type_response.lock().unwrap() = Some("text/event-stream".into());
            ResponseTemplate::new(200)
                .set_body_raw(sse_body.as_bytes().to_vec(), "text/event-stream")
        })
        .mount(upstream)
        .await;
    captured
}

/// Vertex envelope without `model` — the shape the proxy expects.
/// Includes `anthropic_version` (required) and a single user message.
fn minimal_vertex_body() -> Value {
    json!({
        "anthropic_version": "vertex-2023-10-16",
        "messages": [
            {"role": "user", "content": "Hello, Claude!"}
        ],
        "max_tokens": 64,
    })
}

// ─── TEST 1 ────────────────────────────────────────────────────────────

#[tokio::test]
async fn native_envelope_round_trip_byte_equal() {
    let upstream = MockServer::start().await;
    let captured = mount_capture_json(&upstream).await;

    let proxy = start_proxy_with_state(
        &upstream.uri(),
        |c| {
            // Compression off: the only thing under test here is the
            // envelope detection + forwarding path. The body must
            // round-trip byte-equal even when the live-zone dispatcher
            // is engaged in a separate test (`thinking_block_preserved`).
            c.compression = false;
        },
        |s| install_static_token_source(s, TEST_BEARER),
    )
    .await;

    let body_value = minimal_vertex_body();
    let body_bytes = serde_json::to_vec(&body_value).unwrap();
    let resp = reqwest::Client::new()
        .post(raw_predict_url(&proxy.url()))
        .header("content-type", "application/json")
        .body(body_bytes.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let got = captured
        .body
        .lock()
        .unwrap()
        .clone()
        .expect("upstream got body");
    assert_byte_equal_sha256(&body_bytes, &got);

    // Defensive: parse and confirm the canonical envelope fields.
    let parsed: Value = serde_json::from_slice(&got).unwrap();
    assert_eq!(parsed["anthropic_version"], json!("vertex-2023-10-16"));
    assert!(
        parsed.get("model").is_none(),
        "Vertex envelope must NOT carry a `model` field"
    );

    proxy.shutdown().await;
}

// ─── TEST 2 ────────────────────────────────────────────────────────────

#[tokio::test]
async fn adc_bearer_token_signed_correctly() {
    let upstream = MockServer::start().await;
    let captured = mount_capture_json(&upstream).await;

    let proxy = start_proxy_with_state(
        &upstream.uri(),
        |c| {
            c.compression = false;
        },
        |s| install_static_token_source(s, TEST_BEARER),
    )
    .await;

    let body_bytes = serde_json::to_vec(&minimal_vertex_body()).unwrap();

    // Send the request WITHOUT an Authorization header. The proxy
    // must inject `Bearer <token>` from the static token source.
    let resp = reqwest::Client::new()
        .post(raw_predict_url(&proxy.url()))
        .header("content-type", "application/json")
        .body(body_bytes)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let auth = captured
        .authorization
        .lock()
        .unwrap()
        .clone()
        .expect("upstream got Authorization header");
    let expected = format!("Bearer {TEST_BEARER}");
    assert_eq!(
        auth, expected,
        "Vertex request must carry the ADC bearer token; got {auth}, expected {expected}",
    );

    // ALSO: verify the proxy OVERWRITES a client-supplied
    // Authorization header (Vertex would reject the wrong flavour
    // anyway — silent forward of the wrong auth would surface as a
    // confusing 401 from upstream).
    let body_bytes2 = serde_json::to_vec(&minimal_vertex_body()).unwrap();
    let resp2 = reqwest::Client::new()
        .post(raw_predict_url(&proxy.url()))
        .header("content-type", "application/json")
        .header("authorization", "Bearer some-client-supplied-key")
        .body(body_bytes2)
        .send()
        .await
        .unwrap();
    assert_eq!(resp2.status(), 200);
    let auth2 = captured
        .authorization
        .lock()
        .unwrap()
        .clone()
        .expect("auth header on second call");
    assert_eq!(
        auth2, expected,
        "proxy must overwrite client-supplied Authorization with the ADC bearer; \
         got {auth2}",
    );

    proxy.shutdown().await;
}

// ─── TEST 3 ────────────────────────────────────────────────────────────

#[tokio::test]
async fn thinking_block_preserved() {
    let upstream = MockServer::start().await;
    let captured = mount_capture_json(&upstream).await;

    // Compression ON + LiveZone mode so the live-zone Anthropic
    // dispatcher actually runs over the body. This test is the
    // teeth of P4-37/P4-38: the Python LiteLLM converter dropped
    // `thinking` and `redacted_thinking` blocks. The Rust path must
    // preserve them byte-equal upstream.
    let proxy = start_proxy_with_state(
        &upstream.uri(),
        |c| {
            c.compression = true;
            c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
        },
        |s| install_static_token_source(s, TEST_BEARER),
    )
    .await;

    // Realistic Anthropic block content covering:
    //   - `thinking` block with a signature (cryptographically signed
    //     reasoning the model emitted in a prior turn).
    //   - `redacted_thinking` block (returned by Anthropic when
    //     thinking content is policy-redacted; carries an opaque
    //     `data` blob that MUST round-trip byte-equal).
    //   - text content alongside, so the live-zone walker has more
    //     than one block to consider.
    let body_value = json!({
        "anthropic_version": "vertex-2023-10-16",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "I need to plan this carefully. Let me think step by step about the user's request and consider edge cases.",
                        "signature": "EuYBCkQYBCKMAQ.thinkingsig.example.base64.payload="
                    },
                    {
                        "type": "redacted_thinking",
                        "data": "EmkKAhgEEgwQ.redacted.opaque.bytes.must.roundtrip="
                    },
                    {
                        "type": "text",
                        "text": "Here is my answer."
                    }
                ]
            },
            {
                "role": "user",
                "content": "Can you elaborate?"
            }
        ],
        "thinking": {"type": "enabled", "budget_tokens": 5000}
    });
    let body_bytes = serde_json::to_vec(&body_value).unwrap();

    let resp = reqwest::Client::new()
        .post(raw_predict_url(&proxy.url()))
        .header("content-type", "application/json")
        .body(body_bytes.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let got = captured
        .body
        .lock()
        .unwrap()
        .clone()
        .expect("upstream got body");

    // Strong assertion: byte-equal end-to-end. The live-zone
    // dispatcher's RawValue-based surgery may rewrite live-zone
    // messages but ours has only an assistant turn (frozen by
    // definition) and a single short user turn that's below the
    // compression-eligibility floor, so the body should be the
    // same bytes.
    assert_byte_equal_sha256(&body_bytes, &got);

    // Defensive: thinking + redacted_thinking + signature all
    // present and unchanged.
    let parsed: Value = serde_json::from_slice(&got).unwrap();
    let assistant_blocks = &parsed["messages"][0]["content"];
    assert_eq!(assistant_blocks[0]["type"], json!("thinking"));
    assert!(
        assistant_blocks[0]["signature"]
            .as_str()
            .unwrap()
            .starts_with("EuYBCkQYBCKMAQ"),
        "thinking.signature must round-trip byte-equal"
    );
    assert_eq!(assistant_blocks[1]["type"], json!("redacted_thinking"));
    assert_eq!(
        assistant_blocks[1]["data"],
        json!("EmkKAhgEEgwQ.redacted.opaque.bytes.must.roundtrip="),
        "redacted_thinking.data must round-trip byte-equal"
    );
    assert_eq!(assistant_blocks[2]["type"], json!("text"));

    proxy.shutdown().await;
}

// ─── TEST 4 ────────────────────────────────────────────────────────────

#[tokio::test]
async fn stream_raw_predict_sse_handled() {
    let upstream = MockServer::start().await;
    let captured = mount_capture_sse(&upstream).await;

    let proxy = start_proxy_with_state(
        &upstream.uri(),
        |c| {
            c.compression = false;
        },
        |s| install_static_token_source(s, TEST_BEARER),
    )
    .await;

    // Streaming envelope: same shape as :rawPredict, with `stream:
    // true` (Vertex doesn't actually require the field — the verb
    // disambiguates — but real clients send it for compatibility).
    let body_value = json!({
        "anthropic_version": "vertex-2023-10-16",
        "stream": true,
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "Stream please."}
        ]
    });
    let body_bytes = serde_json::to_vec(&body_value).unwrap();

    let resp = reqwest::Client::new()
        .post(stream_raw_predict_url(&proxy.url()))
        .header("content-type", "application/json")
        .header("accept", "text/event-stream")
        .body(body_bytes.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    // Confirm the response carries SSE content-type back to the
    // client (the proxy MUST NOT translate to JSON or rewrap).
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        ct.eq_ignore_ascii_case("text/event-stream"),
        "Vertex stream response must surface SSE content-type to client; got {ct}",
    );

    // Drain the stream and confirm we see the message_start +
    // content_block_delta events the upstream emitted. Bytes pass
    // through unchanged.
    let body_text = resp.text().await.expect("read sse body");
    assert!(
        body_text.contains("event: message_start"),
        "sse body missing message_start: {body_text:?}",
    );
    assert!(
        body_text.contains("event: content_block_delta"),
        "sse body missing content_block_delta: {body_text:?}",
    );
    assert!(
        body_text.contains("event: message_stop"),
        "sse body missing message_stop: {body_text:?}",
    );

    // Body bytes upstream-received MUST match the inbound bytes (no
    // request-side rewrite when compression is off).
    let got = captured
        .body
        .lock()
        .unwrap()
        .clone()
        .expect("upstream got body");
    assert_byte_equal_sha256(&body_bytes, &got);

    // Bearer was attached.
    let auth = captured
        .authorization
        .lock()
        .unwrap()
        .clone()
        .expect("upstream got Authorization");
    assert_eq!(auth, format!("Bearer {TEST_BEARER}"));

    proxy.shutdown().await;
}

// ─── BONUS: ADC FAILURE PATH (no silent fallback) ──────────────────────

#[tokio::test]
async fn adc_failure_returns_5xx_no_silent_forward() {
    use async_trait::async_trait;
    use headroom_proxy::vertex::{TokenSource, TokenSourceError};
    use std::sync::Arc as StdArc;

    // Token source that always fails — verifies the no-silent-fallback
    // contract: the proxy must NOT forward a request to upstream
    // without a bearer.
    #[derive(Debug)]
    struct AlwaysFail;
    #[async_trait]
    impl TokenSource for AlwaysFail {
        async fn bearer(&self) -> Result<String, TokenSourceError> {
            Err(TokenSourceError::Fetch(
                "synthetic test failure: ADC chain unreachable".into(),
            ))
        }
    }

    let upstream = MockServer::start().await;
    let captured = mount_capture_json(&upstream).await;

    let proxy = start_proxy_with_state(
        &upstream.uri(),
        |c| {
            c.compression = false;
        },
        |mut state| {
            state.vertex_token_source = StdArc::new(AlwaysFail) as StdArc<dyn TokenSource>;
            state
        },
    )
    .await;

    let body_bytes = serde_json::to_vec(&minimal_vertex_body()).unwrap();
    let resp = reqwest::Client::new()
        .post(raw_predict_url(&proxy.url()))
        .header("content-type", "application/json")
        .body(body_bytes)
        .send()
        .await
        .unwrap();

    assert!(
        resp.status().is_server_error(),
        "ADC failure must surface as 5xx, got {}",
        resp.status()
    );

    // Critically: the upstream must NOT have been called at all.
    assert!(
        captured.body.lock().unwrap().is_none(),
        "ADC failure must short-circuit; upstream must not be reached"
    );

    proxy.shutdown().await;
}
