//! Integration tests for the native Bedrock InvokeModel route
//! (Phase D PR-D1).
//!
//! These tests boot the real Rust proxy in front of a wiremock
//! upstream that pretends to be the Bedrock runtime endpoint
//! (`https://bedrock-runtime.{region}.amazonaws.com`). The proxy is
//! configured with `bedrock_endpoint = wiremock_url` so SigV4-signed
//! requests are routed to the mock instead of real AWS — no live
//! AWS dependency.
//!
//! Coverage matrix (per PR-D1 spec, REALIGNMENT/06-phase-D-bedrock-vertex.md):
//!
//! 1. `native_envelope_round_trip_byte_equal` — small body,
//!    compression-mode off; bytes round-trip byte-equal upstream.
//! 2. `sigv4_signed_correctly_after_compression` — confirms the
//!    `authorization` header arrives at upstream and the
//!    `x-amz-content-sha256` matches the (post-compression) body.
//! 3. `thinking_block_preserved_through_bedrock` — Anthropic
//!    `thinking` block round-trips byte-equal with compression off.
//!    Validates the live-zone dispatcher doesn't strip the block.
//! 4. `redacted_thinking_preserved` — `redacted_thinking` block
//!    round-trips byte-equal.
//! 5. `document_block_preserved` — `document` block round-trips.
//! 6. `tool_result_array_with_image_preserved` — `tool_result` content
//!    array containing a base64 `image` block round-trips byte-equal
//!    when compression is off.
//! 7. `stop_sequence_null_only_when_present` — mock the upstream to
//!    return a Bedrock-shape response that does NOT include
//!    `stop_sequence`; the proxy must not inject a `null` value for
//!    it. (This is an end-to-end check that Phase D doesn't regress
//!    P4-37's hardcoded null.)
//! 8. `tool_use_input_byte_equal_preserves_key_order` — `tool_use.input`
//!    object keys must arrive in the same order they were sent
//!    (the `serde_json::preserve_order` feature backs this).

mod common;

use aws_credential_types::Credentials;
use common::start_proxy_with_state;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::sync::{Arc, Mutex};
use url::Url;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// What we capture from each upstream request — body + the
/// authorization-shaped headers we care about.
#[derive(Default, Clone, Debug)]
struct CapturedRequest {
    body: Option<Vec<u8>>,
    authorization: Option<String>,
    x_amz_date: Option<String>,
    x_amz_content_sha256: Option<String>,
    host: Option<String>,
    content_type: Option<String>,
}

type Capture = Arc<Mutex<CapturedRequest>>;

const TEST_MODEL: &str = "anthropic.claude-3-haiku-20240307-v1:0";

async fn mount_capture_invoke(upstream: &MockServer, response_body: &str) -> Capture {
    let captured: Capture = Arc::new(Mutex::new(CapturedRequest::default()));
    let captured_clone = captured.clone();
    let response_body = response_body.to_string();
    let model = TEST_MODEL.to_string();
    Mock::given(method("POST"))
        .and(path(format!("/model/{model}/invoke")))
        .respond_with(move |req: &wiremock::Request| {
            let mut c = captured_clone.lock().unwrap();
            c.body = Some(req.body.clone());
            c.authorization = req
                .headers
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .map(str::to_string);
            c.x_amz_date = req
                .headers
                .get("x-amz-date")
                .and_then(|v| v.to_str().ok())
                .map(str::to_string);
            c.x_amz_content_sha256 = req
                .headers
                .get("x-amz-content-sha256")
                .and_then(|v| v.to_str().ok())
                .map(str::to_string);
            c.host = req
                .headers
                .get("host")
                .and_then(|v| v.to_str().ok())
                .map(str::to_string);
            c.content_type = req
                .headers
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .map(str::to_string);
            ResponseTemplate::new(200).set_body_string(response_body.clone())
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

fn test_credentials() -> Credentials {
    Credentials::new(
        "AKIAEXAMPLEAKIDFORTEST",
        "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        None,
        None,
        "test",
    )
}

/// Boot a proxy pointed at the wiremock upstream as the Bedrock
/// endpoint. The fake-upstream URL goes into `bedrock_endpoint`;
/// the regular `upstream` field is set to a sentinel because the
/// Bedrock route bypasses `forward_http`.
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

#[tokio::test]
async fn native_envelope_round_trip_byte_equal() {
    // Compression off → body must arrive byte-equal at upstream.
    let upstream = MockServer::start().await;
    let captured = mount_capture_invoke(&upstream, r#"{"id":"msg_x","content":[]}"#).await;
    let proxy = bedrock_proxy(&upstream, |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::Off;
    })
    .await;

    let payload = json!({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "hi"}
        ]
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/model/{TEST_MODEL}/invoke", proxy.url()))
        .header("content-type", "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let got = captured.lock().unwrap().clone();
    assert_byte_equal_sha256(&body, got.body.as_deref().unwrap());
    proxy.shutdown().await;
}

#[tokio::test]
async fn sigv4_signed_correctly_after_compression() {
    // The signature must cover the bytes that actually hit upstream.
    // We confirm: (a) `authorization` is present, (b)
    // `x-amz-content-sha256` matches sha256(body received by upstream).
    let upstream = MockServer::start().await;
    let captured = mount_capture_invoke(&upstream, r#"{"id":"msg_x","content":[]}"#).await;
    let proxy = bedrock_proxy(&upstream, |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
    })
    .await;

    let payload = json!({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}]
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/model/{TEST_MODEL}/invoke", proxy.url()))
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let got = captured.lock().unwrap().clone();
    let auth = got.authorization.expect("authorization header present");
    assert!(
        auth.starts_with("AWS4-HMAC-SHA256 "),
        "authorization must be SigV4-shape; got {auth}"
    );
    assert!(
        auth.contains("Credential=AKIAEXAMPLEAKIDFORTEST/"),
        "authorization must reference the test access key id; got {auth}"
    );
    assert!(
        auth.contains("/bedrock/aws4_request"),
        "authorization scope must reference the bedrock service; got {auth}"
    );
    assert!(got.x_amz_date.is_some(), "x-amz-date must be present");
    let body_received = got.body.expect("upstream got body");
    let expected_sha = sha256_hex(&body_received);
    assert_eq!(
        got.x_amz_content_sha256.as_deref(),
        Some(expected_sha.as_str()),
        "x-amz-content-sha256 must match sha256 of bytes the upstream received"
    );
    proxy.shutdown().await;
}

#[tokio::test]
async fn thinking_block_preserved_through_bedrock() {
    // Anthropic `thinking` block: cache hot zone item. With
    // compression OFF, body round-trips byte-equal upstream. (The
    // dispatcher only mutates the live zone — the latest user
    // message — so even with compression on, an assistant `thinking`
    // block is left alone. We pin the byte-equal contract to the
    // off-mode path because that's what the litellm Python shim
    // would have lost — P4-37 evidence.)
    let upstream = MockServer::start().await;
    let captured = mount_capture_invoke(&upstream, r#"{"id":"msg_x","content":[]}"#).await;
    let proxy = bedrock_proxy(&upstream, |c| {
        c.compression_mode = headroom_proxy::config::CompressionMode::Off;
    })
    .await;

    let payload = json!({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "What's 2+2?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "The user is asking a basic arithmetic question. 2+2=4.",
                        "signature": "EpYBCkYIBRgCKkAhello_world_signature_payload="
                    },
                    {"type": "text", "text": "4"}
                ]
            }
        ]
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/model/{TEST_MODEL}/invoke", proxy.url()))
        .header("content-type", "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let got = captured.lock().unwrap().clone();
    assert_byte_equal_sha256(&body, got.body.as_deref().unwrap());
    let parsed: Value = serde_json::from_slice(got.body.as_deref().unwrap()).unwrap();
    assert_eq!(parsed["messages"][1]["content"][0]["type"], "thinking");
    assert_eq!(
        parsed["messages"][1]["content"][0]["signature"],
        "EpYBCkYIBRgCKkAhello_world_signature_payload="
    );
    proxy.shutdown().await;
}

#[tokio::test]
async fn redacted_thinking_preserved() {
    // `redacted_thinking` blocks: opaque encrypted payloads from
    // the model. Must round-trip BYTE-EQUAL — the proxy never
    // inspects them.
    let upstream = MockServer::start().await;
    let captured = mount_capture_invoke(&upstream, r#"{"id":"msg_x","content":[]}"#).await;
    let proxy = bedrock_proxy(&upstream, |c| {
        c.compression_mode = headroom_proxy::config::CompressionMode::Off;
    })
    .await;

    let payload = json!({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "redacted_thinking",
                        "data": "EuYBCogBAaR_o9XJEnEx_3Q9d5z9_redacted_payload"
                    },
                    {"type": "text", "text": "Hello!"}
                ]
            }
        ]
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/model/{TEST_MODEL}/invoke", proxy.url()))
        .header("content-type", "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let got = captured.lock().unwrap().clone();
    assert_byte_equal_sha256(&body, got.body.as_deref().unwrap());
    let parsed: Value = serde_json::from_slice(got.body.as_deref().unwrap()).unwrap();
    assert_eq!(
        parsed["messages"][1]["content"][0]["type"],
        "redacted_thinking"
    );
    assert_eq!(
        parsed["messages"][1]["content"][0]["data"],
        "EuYBCogBAaR_o9XJEnEx_3Q9d5z9_redacted_payload"
    );
    proxy.shutdown().await;
}

#[tokio::test]
async fn document_block_preserved() {
    // `document` block (PDF or text-document attachment). Has nested
    // `source.media_type` + `source.data` (base64). The litellm
    // Python shim drops these silently; the Rust route must NOT.
    let upstream = MockServer::start().await;
    let captured = mount_capture_invoke(&upstream, r#"{"id":"msg_x","content":[]}"#).await;
    let proxy = bedrock_proxy(&upstream, |c| {
        c.compression_mode = headroom_proxy::config::CompressionMode::Off;
    })
    .await;

    let payload = json!({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 256,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": "JVBERi0xLjQKJfbk/N8KMSAwIG9iago8PAovVHlwZSAvQ2F0YWxvZwo+PgplbmRvYmoK"
                    },
                    "title": "Quarterly Report",
                    "context": "Q4 2025 financial summary"
                },
                {"type": "text", "text": "Summarize."}
            ]
        }]
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/model/{TEST_MODEL}/invoke", proxy.url()))
        .header("content-type", "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let got = captured.lock().unwrap().clone();
    assert_byte_equal_sha256(&body, got.body.as_deref().unwrap());
    let parsed: Value = serde_json::from_slice(got.body.as_deref().unwrap()).unwrap();
    let doc = &parsed["messages"][0]["content"][0];
    assert_eq!(doc["type"], "document");
    assert_eq!(doc["source"]["media_type"], "application/pdf");
    assert_eq!(doc["title"], "Quarterly Report");
    proxy.shutdown().await;
}

#[tokio::test]
async fn tool_result_array_with_image_preserved() {
    // `tool_result.content` is an array; one element is an `image`
    // block with base64 source. The litellm shim flattened these
    // to a single string (P4-37). The Rust route must keep the
    // array shape verbatim.
    let upstream = MockServer::start().await;
    let captured = mount_capture_invoke(&upstream, r#"{"id":"msg_x","content":[]}"#).await;
    let proxy = bedrock_proxy(&upstream, |c| {
        c.compression_mode = headroom_proxy::config::CompressionMode::Off;
    })
    .await;

    let payload = json!({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 256,
        "messages": [
            {"role": "user", "content": "Take a screenshot"},
            {
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": "toolu_xyz",
                    "name": "screenshot",
                    "input": {"region": "full"}
                }]
            },
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_xyz",
                    "content": [
                        {"type": "text", "text": "Captured at 2026-05-03T12:00:00Z"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP8z8DwHwAFAQH/9zJEHwAAAABJRU5ErkJggg=="
                            }
                        }
                    ]
                }]
            }
        ]
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/model/{TEST_MODEL}/invoke", proxy.url()))
        .header("content-type", "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let got = captured.lock().unwrap().clone();
    assert_byte_equal_sha256(&body, got.body.as_deref().unwrap());
    let parsed: Value = serde_json::from_slice(got.body.as_deref().unwrap()).unwrap();
    let tool_result = &parsed["messages"][2]["content"][0];
    assert_eq!(tool_result["type"], "tool_result");
    let inner = &tool_result["content"];
    assert!(inner.is_array(), "tool_result.content must remain an array");
    assert_eq!(inner[1]["type"], "image");
    assert_eq!(inner[1]["source"]["media_type"], "image/png");
    proxy.shutdown().await;
}

#[tokio::test]
async fn stop_sequence_null_only_when_present() {
    // Synthesise a Bedrock-shape response that does NOT include
    // `stop_sequence` and confirm the proxy doesn't add `null` for
    // it. P4-37: the litellm Python shim hardcoded `stop_sequence:
    // null` in the converted response shape; the Rust path must not
    // do that — Bedrock's response is forwarded verbatim.
    let upstream = MockServer::start().await;
    let response_no_stop_sequence = r#"{"id":"msg_a","type":"message","role":"assistant","model":"claude-3-haiku-20240307","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":3,"output_tokens":1}}"#;
    let _captured = mount_capture_invoke(&upstream, response_no_stop_sequence).await;
    let proxy = bedrock_proxy(&upstream, |c| {
        c.compression_mode = headroom_proxy::config::CompressionMode::Off;
    })
    .await;

    let payload = json!({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}]
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/model/{TEST_MODEL}/invoke", proxy.url()))
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let resp_text = resp.text().await.unwrap();
    let resp_parsed: Value = serde_json::from_str(&resp_text).unwrap();
    assert!(
        resp_parsed.get("stop_sequence").is_none(),
        "stop_sequence must NOT be present on the response when upstream omitted it; got {resp_text}"
    );
    assert_eq!(resp_parsed["stop_reason"], "end_turn");
    proxy.shutdown().await;
}

#[tokio::test]
async fn tool_use_input_byte_equal_preserves_key_order() {
    // `tool_use.input` is a JSON object whose key order must
    // round-trip exactly. P4-43: the litellm shim parsed
    // `function.arguments` into a dict and re-stringified, breaking
    // key order. The Rust path uses serde_json's preserve_order
    // feature throughout — confirm the bytes arrive byte-equal.
    let upstream = MockServer::start().await;
    let captured = mount_capture_invoke(&upstream, r#"{"id":"msg_x","content":[]}"#).await;
    let proxy = bedrock_proxy(&upstream, |c| {
        c.compression_mode = headroom_proxy::config::CompressionMode::Off;
    })
    .await;

    // Hand-craft the body bytes so we can pin exact key order.
    // BTreeMap-default serialization would alphabetize the keys
    // (city < country < units); we send the opposite so any
    // accidental re-encode shows up as a byte mismatch.
    let body = br#"{"anthropic_version":"bedrock-2023-05-31","max_tokens":64,"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"toolu_zoom","name":"get_weather","input":{"units":"metric","country":"FR","city":"Paris"}}]}]}"#;

    let resp = reqwest::Client::new()
        .post(format!("{}/model/{TEST_MODEL}/invoke", proxy.url()))
        .header("content-type", "application/json")
        .body(body.to_vec())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let got = captured.lock().unwrap().clone();
    let received = got.body.expect("upstream got body");
    assert_byte_equal_sha256(body, &received);

    // Defensive: sanity-check the input key order by looking at the
    // raw substring.
    let received_str = std::str::from_utf8(&received).unwrap();
    let units_pos = received_str.find("\"units\"").expect("units present");
    let country_pos = received_str.find("\"country\"").expect("country present");
    let city_pos = received_str.find("\"city\"").expect("city present");
    assert!(
        units_pos < country_pos && country_pos < city_pos,
        "tool_use.input key order must be units→country→city; got: {received_str}"
    );
    proxy.shutdown().await;
}
