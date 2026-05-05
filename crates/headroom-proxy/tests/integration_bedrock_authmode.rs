//! Integration tests for the Bedrock auth-mode middleware
//! (Phase D PR-D3).
//!
//! Coverage:
//!
//! 1. `bedrock_classified_as_oauth` — POST a Bedrock invoke request
//!    with no Authorization header (the most common SDK pattern when
//!    AWS credentials live downstream of the proxy). Assert the
//!    middleware coerces the result to `AuthMode::OAuth` per the
//!    Bedrock policy matrix and that the value lands in
//!    `request.extensions()` where downstream Phase F handlers can
//!    pick it up.
//! 2. `oauth_policy_passthrough_prefer` — fire a request with an
//!    Anthropic body containing NO `cache_control` markers; assert
//!    the upstream-bound body is byte-equal to the inbound body.
//!    The OAuth policy matrix forbids auto-injecting `cache_control`
//!    or `prompt_cache_key`; D3 wires the marker, F2 enforces the
//!    policy. Until F2 lands, the proof is the byte-equality (no
//!    mutation observed at the upstream boundary).

mod common;

use aws_credential_types::Credentials;
use axum::body::Body;
use axum::extract::{Extension, State};
use axum::http::StatusCode;
use axum::routing::post;
use axum::Router;
use bytes::Bytes;
use common::start_proxy_with_state;
use headroom_core::auth_mode::AuthMode;
use headroom_proxy::AppState;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use tokio::sync::oneshot;
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

#[derive(Default, Clone, Debug)]
struct CapturedRequest {
    body: Option<Vec<u8>>,
}

type Capture = Arc<Mutex<CapturedRequest>>;

async fn mount_capture_invoke(upstream: &MockServer, response_body: &str) -> Capture {
    let captured: Capture = Arc::new(Mutex::new(CapturedRequest::default()));
    let captured_clone = captured.clone();
    let response_body = response_body.to_string();
    Mock::given(method("POST"))
        .and(path(format!("/model/{TEST_MODEL}/invoke")))
        .respond_with(move |req: &wiremock::Request| {
            let mut c = captured_clone.lock().unwrap();
            c.body = Some(req.body.clone());
            ResponseTemplate::new(200).set_body_string(response_body.clone())
        })
        .mount(upstream)
        .await;
    captured
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

/// Test 1: With no Authorization header, the bedrock auth-mode
/// middleware classifies as OAuth (Bedrock policy matrix). We boot
/// a separate axum app that mounts the same middleware in front of
/// a probe handler; the probe reads the AuthMode out of
/// `request.extensions()` and echoes it back. This is the canonical
/// "extension was set" assertion the spec asks for.
#[tokio::test]
async fn bedrock_classified_as_oauth() {
    use headroom_proxy::bedrock::classify_and_attach_auth_mode;

    async fn probe(Extension(auth_mode): Extension<AuthMode>) -> String {
        auth_mode.as_str().to_string()
    }
    let app = Router::new()
        .route("/model/:model_id/invoke", post(probe))
        .route_layer(axum::middleware::from_fn(classify_and_attach_auth_mode));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = oneshot::channel::<()>();
    let task = tokio::spawn(async move {
        let _ = axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .with_graceful_shutdown(async move {
            let _ = rx.await;
        })
        .await;
    });

    // Bedrock SDK style: no Authorization header in the inbound
    // request to our proxy (the SDK signs at the egress side, or
    // the customer is using IAM-instance-credential downstream of
    // our hop). NO x-api-key. NO x-goog-api-key. F1 returns Payg by
    // default; the bedrock middleware must coerce to OAuth.
    let resp = reqwest::Client::new()
        .post(format!(
            "http://{addr}/model/{TEST_MODEL}/invoke",
            addr = addr,
            TEST_MODEL = TEST_MODEL,
        ))
        .header("content-type", "application/json")
        .body(r#"{"anthropic_version":"bedrock-2023-05-31","max_tokens":8,"messages":[]}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body_text = resp.text().await.unwrap();
    assert_eq!(
        body_text, "oauth",
        "bedrock route must classify as OAuth; saw {body_text}"
    );
    let _ = tx.send(());
    let _ = task.await;
}

/// Test 2: confirm the upstream-bound body is byte-equal to the
/// inbound body. The OAuth policy forbids auto-injecting
/// `cache_control`; D3's contribution is to MARK the request as
/// OAuth so PR-F2 can gate the cache-control walker. For now the
/// invariant is "no mutation visible at the upstream boundary"
/// when compression mode is `off`.
#[tokio::test]
async fn oauth_policy_passthrough_prefer() {
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
    let received = got.body.expect("upstream got body");
    // Byte-equality (sha256 hashes match).
    let inbound_hash = sha256_hex(&body);
    let received_hash = sha256_hex(&received);
    assert_eq!(
        inbound_hash, received_hash,
        "upstream body must be byte-equal to inbound body under OAuth policy: \
         inbound={inbound_hash}, received={received_hash}"
    );
    // Belt-and-braces: parse the upstream body and assert NO
    // cache_control marker was added to any message.
    let parsed: Value = serde_json::from_slice(&received).unwrap();
    let messages = parsed["messages"].as_array().expect("messages array");
    for (i, msg) in messages.iter().enumerate() {
        // `cache_control` may live on either the message itself or
        // on individual content blocks. Assert neither path got
        // synthesised by us.
        assert!(
            msg.get("cache_control").is_none(),
            "messages[{i}] gained a cache_control marker; OAuth policy forbids auto-injection"
        );
        if let Some(content) = msg.get("content").and_then(|v| v.as_array()) {
            for (j, block) in content.iter().enumerate() {
                assert!(
                    block.get("cache_control").is_none(),
                    "messages[{i}].content[{j}] gained a cache_control marker"
                );
            }
        }
    }
    // And NO prompt_cache_key at the top level.
    assert!(
        parsed.get("prompt_cache_key").is_none(),
        "top-level prompt_cache_key must NOT be auto-injected under OAuth"
    );
    proxy.shutdown().await;
}

/// Helper: SHA-256 hex of bytes. Mirrors `integration_bedrock_invoke.rs`.
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

/// Pin the unused-import lint silencers — these symbols are
/// referenced by the assertions but the linter is paranoid about
/// `axum::body::Body` and `AppState` only being used in a single
/// type-position.
#[allow(dead_code)]
fn _pin(_: Body, _: State<AppState>, _: Bytes, _: StatusCode) {}
