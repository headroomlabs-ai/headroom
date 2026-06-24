//! Integration tests for PR-E2: recursive JSON Schema key sort.
//!
//! Boots a real Rust proxy in front of a wiremock upstream. Three
//! scenarios:
//!
//!   1. **PAYG path**: a tool's `input_schema` arrives at the upstream
//!      with keys sorted alphabetically at every nesting level. Array
//!      order in `oneOf` etc. is preserved.
//!   2. **OAuth path**: schema keys pass through verbatim — bytes the
//!      upstream sees match the bytes the client sent (SHA-256
//!      byte-equal).
//!   3. **PAYG, marker present**: PR-E1 (sort) is skipped, but PR-E2
//!      still runs on the schema. Tools array preserves customer
//!      order; schema keys are sorted.
//!
//! The Phase A cache-safety invariant — bytes-in == bytes-out for
//! any non-PAYG request — is the contract under test.

mod common;

use common::start_proxy_with;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::sync::{Arc, Mutex};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut s = String::with_capacity(64);
    for b in digest {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

async fn mount_anthropic_capture(upstream: &MockServer) -> Arc<Mutex<Option<Vec<u8>>>> {
    let captured: Arc<Mutex<Option<Vec<u8>>>> = Arc::new(Mutex::new(None));
    let captured_clone = captured.clone();
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(move |req: &wiremock::Request| {
            *captured_clone.lock().unwrap() = Some(req.body.clone());
            ResponseTemplate::new(200).set_body_string(r#"{"ok":true}"#)
        })
        .mount(upstream)
        .await;
    captured
}

/// PAYG: schema arrives with keys in a hash-randomized order; assert
/// upstream sees them sorted at every nesting level. Array order in
/// `oneOf` is preserved.
#[tokio::test]
async fn payg_request_with_shuffled_schema_keys_arrives_sorted() {
    let upstream = MockServer::start().await;
    let captured = mount_anthropic_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
    })
    .await;

    let payload = json!({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {
                "name": "search",
                "input_schema": {
                    // Top-level keys in non-alphabetic order.
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        // Nested keys also shuffled.
                        "z_filter": {"type": "object"},
                        "query": {"type": "string"},
                        "a_field": {"type": "integer"},
                    },
                    // Array semantics test: oneOf must stay in order.
                    "oneOf": [
                        {"const": "third"},
                        {"const": "first"},
                        {"const": "second"},
                    ],
                },
            },
        ],
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        .header("x-api-key", "sk-ant-api03-abc")
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let upstream_body = captured
        .lock()
        .unwrap()
        .clone()
        .expect("upstream should have captured");
    let parsed: Value = serde_json::from_slice(&upstream_body).expect("upstream body is JSON");
    let schema = &parsed["tools"][0]["input_schema"];

    // Top-level: oneOf, properties, required, type (alphabetic).
    let top_map = schema.as_object().unwrap();
    let top_keys: Vec<&str> = top_map.keys().map(String::as_str).collect();
    assert_eq!(top_keys, vec!["oneOf", "properties", "required", "type"]);

    // Nested properties: a_field, query, z_filter (alphabetic).
    let props = schema["properties"].as_object().unwrap();
    let prop_keys: Vec<&str> = props.keys().map(String::as_str).collect();
    assert_eq!(prop_keys, vec!["a_field", "query", "z_filter"]);

    // oneOf array order preserved (NOT sorted).
    let one_of = schema["oneOf"].as_array().unwrap();
    let consts: Vec<&str> = one_of
        .iter()
        .map(|v| v.get("const").and_then(Value::as_str).unwrap())
        .collect();
    assert_eq!(consts, vec!["third", "first", "second"]);

    proxy.shutdown().await;
}

/// OAuth: bytes pass through verbatim — SHA-256 byte-equal.
#[tokio::test]
async fn oauth_request_passes_schema_through_byte_equal() {
    let upstream = MockServer::start().await;
    let captured = mount_anthropic_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
    })
    .await;

    let payload = json!({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {
                "name": "search",
                "input_schema": {
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "z_filter": {"type": "object"},
                        "query": {"type": "string"},
                    },
                },
            },
        ],
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let body_hash = sha256_hex(&body);

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        .header("authorization", "Bearer sk-ant-oat-foo")
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let upstream_body = captured
        .lock()
        .unwrap()
        .clone()
        .expect("upstream should have captured");
    assert_eq!(
        sha256_hex(&upstream_body),
        body_hash,
        "OAuth path must pass schema bytes through unchanged"
    );

    proxy.shutdown().await;
}

/// PAYG, marker present: E1 (sort) is skipped → tools array order
/// preserved. E2 (schema sort) still runs because the marker lives on
/// the tool object, not inside the schema.
#[tokio::test]
async fn payg_with_marker_runs_e2_but_not_e1() {
    let upstream = MockServer::start().await;
    let captured = mount_anthropic_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
    })
    .await;

    let payload = json!({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {
                "name": "zebra",
                "cache_control": {"type": "ephemeral"},
                "input_schema": {
                    "type": "object",
                    "required": ["q"],
                    "properties": {"q": {"type": "string"}},
                },
            },
            {
                "name": "apple",
                "input_schema": {
                    "type": "object",
                    "required": ["x"],
                    "properties": {"x": {"type": "string"}},
                },
            },
        ],
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        .header("x-api-key", "sk-ant-api03-abc")
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let upstream_body = captured
        .lock()
        .unwrap()
        .clone()
        .expect("upstream should have captured");
    let parsed: Value = serde_json::from_slice(&upstream_body).unwrap();

    // E1 skipped: tools order preserved (zebra still first).
    let tools = parsed["tools"].as_array().unwrap();
    let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
    assert_eq!(names, vec!["zebra", "apple"]);

    // E2 ran: input_schema keys are sorted on every tool, including
    // the one carrying the marker.
    for (i, _) in tools.iter().enumerate() {
        let schema = &parsed["tools"][i]["input_schema"];
        let keys: Vec<&str> = schema
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            keys,
            vec!["properties", "required", "type"],
            "schema keys must be sorted on tools[{i}]"
        );
    }

    proxy.shutdown().await;
}
