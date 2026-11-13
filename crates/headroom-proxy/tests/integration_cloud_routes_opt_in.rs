//! The native Bedrock and Vertex routes attach this host's cloud
//! credentials (SigV4 / Google ADC bearer) to whatever request reaches
//! them. They are mounted only on explicit opt-in, the Vertex bearer
//! goes only to a Google API host, and the ADC token source uses the
//! configured scope.

mod common;

use std::net::SocketAddr;

use axum::body::Body;
use axum::extract::ConnectInfo;
use axum::http::{Request, StatusCode};
use clap::Parser;
use headroom_proxy::config::CliArgs;
use headroom_proxy::{build_app, AppState, Config};
use tower::util::ServiceExt;
use wiremock::matchers::any;
use wiremock::{Mock, MockServer, ResponseTemplate};

const HOST_BEARER: &str = "ya29.host-adc-bearer";
const VERTEX_PREFIX: &str =
    "/v1beta1/projects/p/locations/us-central1/publishers/anthropic/models/m";
const BODY: &str = r#"{"anthropic_version":"vertex-2023-10-16","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}"#;

async fn upstream() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(any())
        .respond_with(ResponseTemplate::new(200).set_body_string("{}"))
        .mount(&server)
        .await;
    server
}

/// The production defaults, as `main` builds them from the command line.
fn default_config(upstream: &MockServer) -> Config {
    let args = CliArgs::try_parse_from(["headroom-proxy", "--upstream", &upstream.uri()])
        .expect("parse defaults");
    let mut config = Config::from_cli(args);
    // Keep any Bedrock signing on the mock, and stats in memory.
    config.bedrock_endpoint = Some(upstream.uri().parse().unwrap());
    config.stats_path = None;
    config
}

/// POST `path` with host AWS credentials and a host ADC bearer both
/// available to the proxy, as they would be on a cloud VM.
async fn post(config: Config, path: &str) -> StatusCode {
    let state = AppState::new(config)
        .expect("app state")
        .with_bedrock_credentials(common::test_credentials());
    let app = build_app(common::install_static_token_source(state, HOST_BEARER));
    let mut req = Request::post(path)
        .header("content-type", "application/json")
        .body(Body::from(BODY))
        .unwrap();
    let peer: SocketAddr = "127.0.0.1:40000".parse().unwrap();
    req.extensions_mut().insert(ConnectInfo(peer));
    app.oneshot(req).await.unwrap().status()
}

async fn authorizations(upstream: &MockServer) -> Vec<String> {
    upstream
        .received_requests()
        .await
        .unwrap()
        .iter()
        .filter_map(|r| r.headers.get("authorization"))
        .map(|v| v.to_str().unwrap().to_owned())
        .collect()
}

#[tokio::test]
async fn default_config_does_not_sign_bedrock_requests() {
    let up = upstream().await;
    for verb in ["invoke", "invoke-with-response-stream", "converse"] {
        post(
            default_config(&up),
            &format!("/model/anthropic.claude-3-haiku-20240307-v1:0/{verb}"),
        )
        .await;
    }
    let auth = authorizations(&up).await;
    assert!(
        auth.iter().all(|a| !a.starts_with("AWS4-HMAC-SHA256")),
        "host AWS credentials signed a request by default: {auth:?}"
    );
}

#[tokio::test]
async fn default_config_never_attaches_host_adc_bearer() {
    let up = upstream().await;
    for verb in ["rawPredict", "streamRawPredict"] {
        post(default_config(&up), &format!("{VERTEX_PREFIX}:{verb}")).await;
    }
    assert_eq!(up.received_requests().await.unwrap().len(), 2);
    let auth = authorizations(&up).await;
    assert!(
        auth.iter().all(|a| !a.contains(HOST_BEARER)),
        "host ADC bearer sent by default: {auth:?}"
    );
}

#[tokio::test]
async fn vertex_bearer_is_not_sent_to_a_non_google_upstream() {
    let up = upstream().await;
    let mut config = default_config(&up);
    config.enable_vertex_native = true;
    let status = post(config, &format!("{VERTEX_PREFIX}:rawPredict")).await;
    assert_eq!(status, StatusCode::BAD_GATEWAY);
    assert!(up.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn vertex_opt_in_still_signs_when_non_google_upstream_is_allowed() {
    let up = upstream().await;
    let mut config = default_config(&up);
    config.enable_vertex_native = true;
    config.vertex_allow_non_google_upstream = true;
    let status = post(config, &format!("{VERTEX_PREFIX}:rawPredict")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        authorizations(&up).await,
        vec![format!("Bearer {HOST_BEARER}")]
    );
}

#[test]
fn configured_adc_scope_reaches_token_source() {
    let mut config = Config::for_test("http://127.0.0.1:9".parse().unwrap());
    config.vertex_adc_scope = "https://www.googleapis.com/auth/test-narrow-scope".into();
    let state = AppState::new(config).expect("app state");
    // `GcpAdcTokenSource`'s Debug prints its scope; no GCP call is made.
    let debug = format!("{:?}", state.vertex_token_source);
    assert!(debug.contains("test-narrow-scope"), "{debug}");
}
