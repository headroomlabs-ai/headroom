//! End-to-end proxy tests backed by the Rust provider simulators.
//!
//! These tests run Headroom and the simulator in-process. The client only
//! talks to Headroom; Headroom must reach every provider surface through the
//! simulator, never through live provider credentials.

mod common;

use std::net::SocketAddr;

use aws_credential_types::Credentials;
use common::{install_static_token_source, start_proxy_with_state};
use headroom_simulators::config::{ConfiguredResponse, SimulatorConfig, StubRule};
use headroom_simulators::{build_app as build_simulator_app, Simulator};
use serde_json::{json, Value};
use tokio::sync::oneshot;
use url::Url;

const BEDROCK_MODEL: &str = "anthropic.claude-3-haiku-20240307-v1:0";
const VERTEX_PROJECT: &str = "headroom-simulator-e2e";
const VERTEX_LOCATION: &str = "us-central1";
const VERTEX_MODEL: &str = "claude-3-5-sonnet@20240620";
const TEST_BEARER: &str = "ya29.simulator-e2e-token";

struct SimulatorHandle {
    addr: SocketAddr,
    shutdown: Option<oneshot::Sender<()>>,
    task: tokio::task::JoinHandle<()>,
}

impl SimulatorHandle {
    fn url(&self) -> String {
        format!("http://{}", self.addr)
    }

    async fn shutdown(mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        let _ = self.task.await;
    }
}

async fn start_simulator() -> SimulatorHandle {
    start_simulator_with_config(SimulatorConfig::default()).await
}

async fn start_simulator_with_config(config: SimulatorConfig) -> SimulatorHandle {
    let app = build_simulator_app(Simulator::new(config)).into_make_service();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind simulator");
    let addr = listener.local_addr().expect("simulator addr");
    let (tx, rx) = oneshot::channel::<()>();
    let task = tokio::spawn(async move {
        let _ = axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = rx.await;
            })
            .await;
    });
    SimulatorHandle {
        addr,
        shutdown: Some(tx),
        task,
    }
}

fn test_credentials() -> Credentials {
    Credentials::new(
        "SIMULATOR_ACCESS_KEY_ID",
        "simulator-secret-key-not-real",
        None,
        None,
        "simulator-e2e",
    )
}

async fn start_simulator_proxy(simulator_url: &str) -> common::ProxyHandle {
    let bedrock_endpoint: Url = simulator_url.parse().expect("simulator url");
    start_proxy_with_state(
        simulator_url,
        |c| {
            c.compression = false;
            c.bedrock_endpoint = Some(bedrock_endpoint);
            c.enable_bedrock_native = true;
            c.enable_responses_streaming = true;
            c.enable_conversations_passthrough = true;
        },
        |s| {
            install_static_token_source(s.with_bedrock_credentials(test_credentials()), TEST_BEARER)
        },
    )
    .await
}

async fn start_simulator_proxy_without_bedrock_credentials(
    simulator_url: &str,
) -> common::ProxyHandle {
    let bedrock_endpoint: Url = simulator_url.parse().expect("simulator url");
    start_proxy_with_state(
        simulator_url,
        |c| {
            c.compression = false;
            c.bedrock_endpoint = Some(bedrock_endpoint);
            c.enable_bedrock_native = true;
            c.enable_responses_streaming = true;
            c.enable_conversations_passthrough = true;
        },
        |s| install_static_token_source(s, TEST_BEARER),
    )
    .await
}

fn assert_simulator_header(resp: &reqwest::Response) {
    assert_eq!(
        resp.headers()
            .get("x-headroom-simulator")
            .and_then(|v| v.to_str().ok()),
        Some("true"),
        "response must have come from the simulator"
    );
}

fn assert_not_simulator_response(resp: &reqwest::Response) {
    assert!(
        resp.headers().get("x-headroom-simulator").is_none(),
        "proxy-owned preflight errors must not be simulator responses"
    );
}

fn json_error_stub(
    name: &str,
    path: &str,
    body_contains: &str,
    status: u16,
    error_type: &str,
) -> StubRule {
    StubRule {
        name: name.to_string(),
        method: Some("POST".to_string()),
        path: path.to_string(),
        body_contains: Some(body_contains.to_string()),
        body_json_pointer: None,
        response: ConfiguredResponse {
            status,
            headers: [("x-simulator-error-path".to_string(), name.to_string())].into(),
            json: Some(json!({
                "error": {
                    "type": error_type,
                    "message": format!("simulated {name}")
                }
            })),
            body: None,
            sse: vec![],
        },
    }
}

async fn json_post(
    client: &reqwest::Client,
    url: impl Into<String>,
    body: Value,
) -> reqwest::Response {
    client
        .post(url.into())
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .expect("request succeeds")
}

#[tokio::test]
async fn proxy_health_confirms_simulator_upstream() {
    let simulator = start_simulator().await;
    let proxy = start_simulator_proxy(&simulator.url()).await;
    let client = reqwest::Client::new();

    let own: Value = client
        .get(format!("{}/healthz", proxy.url()))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(own["ok"], json!(true));

    let upstream: Value = client
        .get(format!("{}/healthz/upstream", proxy.url()))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(upstream["ok"], json!(true));

    proxy.shutdown().await;
    simulator.shutdown().await;
}

#[tokio::test]
async fn anthropic_messages_json_and_stream_use_simulator() {
    let simulator = start_simulator().await;
    let proxy = start_simulator_proxy(&simulator.url()).await;
    let client = reqwest::Client::new();

    let resp = json_post(
        &client,
        format!("{}/v1/messages", proxy.url()),
        json!({"model":"claude-3-5-sonnet","max_tokens":32,"messages":[{"role":"user","content":"hi"}]}),
    )
    .await;
    assert_eq!(resp.status(), 200);
    assert_simulator_header(&resp);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["type"], json!("message"));
    assert_eq!(
        body["content"][0]["text"],
        json!("simulated anthropic response")
    );

    let stream = json_post(
        &client,
        format!("{}/v1/messages", proxy.url()),
        json!({"model":"claude-3-5-sonnet","max_tokens":32,"stream":true,"messages":[{"role":"user","content":"stream"}]}),
    )
    .await;
    assert_eq!(stream.status(), 200);
    assert_simulator_header(&stream);
    let text = stream.text().await.unwrap();
    assert!(text.contains("event: message_start"));
    assert!(text.contains("simulated anthropic stream"));

    proxy.shutdown().await;
    simulator.shutdown().await;
}

#[tokio::test]
async fn openai_chat_responses_and_conversations_use_simulator() {
    let simulator = start_simulator().await;
    let proxy = start_simulator_proxy(&simulator.url()).await;
    let client = reqwest::Client::new();

    let chat = json_post(
        &client,
        format!("{}/v1/chat/completions", proxy.url()),
        json!({"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}),
    )
    .await;
    assert_eq!(chat.status(), 200);
    assert_simulator_header(&chat);
    let chat_body: Value = chat.json().await.unwrap();
    assert_eq!(chat_body["object"], json!("chat.completion"));

    let chat_stream = json_post(
        &client,
        format!("{}/v1/chat/completions", proxy.url()),
        json!({"model":"gpt-4o","stream":true,"messages":[{"role":"user","content":"hi"}]}),
    )
    .await;
    assert_eq!(chat_stream.status(), 200);
    assert_simulator_header(&chat_stream);
    assert!(chat_stream.text().await.unwrap().contains("[DONE]"));

    let responses = json_post(
        &client,
        format!("{}/v1/responses", proxy.url()),
        json!({"model":"gpt-5","input":"hi"}),
    )
    .await;
    assert_eq!(responses.status(), 200);
    assert_simulator_header(&responses);
    let responses_body: Value = responses.json().await.unwrap();
    assert_eq!(responses_body["object"], json!("response"));

    let responses_stream = json_post(
        &client,
        format!("{}/v1/responses", proxy.url()),
        json!({"model":"gpt-5","input":"hi","stream":true}),
    )
    .await;
    assert_eq!(responses_stream.status(), 200);
    assert_simulator_header(&responses_stream);
    let responses_stream_text = responses_stream.text().await.unwrap();
    assert!(responses_stream_text.contains("event: response.created"));
    assert!(responses_stream_text.contains("event: response.completed"));

    let conversation = json_post(
        &client,
        format!("{}/v1/conversations", proxy.url()),
        json!({"metadata":{"suite":"simulator-e2e"}}),
    )
    .await;
    assert_eq!(conversation.status(), 200);
    assert_simulator_header(&conversation);
    let conversation_body: Value = conversation.json().await.unwrap();
    assert_eq!(conversation_body["id"], json!("conv_sim_0001"));

    let item = json_post(
        &client,
        format!("{}/v1/conversations/conv_sim_0001/items", proxy.url()),
        json!({"items":[{"type":"message","role":"user","content":"persist me"}]}),
    )
    .await;
    assert_eq!(item.status(), 200);
    assert_simulator_header(&item);
    let item_body: Value = item.json().await.unwrap();
    assert_eq!(item_body["object"], json!("conversation.item"));

    let listed: Value = client
        .get(format!(
            "{}/v1/conversations/conv_sim_0001/items",
            proxy.url()
        ))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(listed["object"], json!("list"));

    let delete = client
        .delete(format!(
            "{}/v1/conversations/conv_sim_0001/items/item_sim_0001",
            proxy.url()
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(delete.status(), 200);
    assert_simulator_header(&delete);
    let delete_body: Value = delete.json().await.unwrap();
    assert_eq!(delete_body["deleted"], json!(true));

    proxy.shutdown().await;
    simulator.shutdown().await;
}

#[tokio::test]
async fn bedrock_invoke_converse_and_streaming_use_simulator() {
    let simulator = start_simulator().await;
    let proxy = start_simulator_proxy(&simulator.url()).await;
    let client = reqwest::Client::new();
    let body = json!({
        "anthropic_version":"bedrock-2023-05-31",
        "max_tokens":32,
        "messages":[{"role":"user","content":"hi"}]
    });

    for action in ["invoke", "converse"] {
        let resp = json_post(
            &client,
            format!("{}/model/{BEDROCK_MODEL}/{action}", proxy.url()),
            body.clone(),
        )
        .await;
        assert_eq!(resp.status(), 200, "{action}");
        assert_simulator_header(&resp);
        let json: Value = resp.json().await.unwrap();
        assert_eq!(json["role"], json!("assistant"));
        assert_eq!(
            json["content"][0]["text"],
            json!("simulated bedrock anthropic response")
        );
    }

    let sse = json_post(
        &client,
        format!(
            "{}/model/{BEDROCK_MODEL}/invoke-with-response-stream",
            proxy.url()
        ),
        body.clone(),
    )
    .await;
    assert_eq!(sse.status(), 200);
    assert_simulator_header(&sse);
    let sse_text = sse.text().await.unwrap();
    assert!(sse_text.contains("data: "));
    assert!(sse_text.contains("message_start"));

    let binary = client
        .post(format!(
            "{}/model/{BEDROCK_MODEL}/converse-stream",
            proxy.url()
        ))
        .header("content-type", "application/json")
        .header("accept", "application/vnd.amazon.eventstream")
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(binary.status(), 200);
    assert_simulator_header(&binary);
    let bytes = binary.bytes().await.unwrap();
    assert!(bytes.len() > 16);
    let total_len = u32::from_be_bytes(bytes[0..4].try_into().unwrap()) as usize;
    assert_eq!(total_len, bytes.len());

    proxy.shutdown().await;
    simulator.shutdown().await;
}

#[tokio::test]
async fn vertex_raw_and_stream_predict_use_simulator() {
    let simulator = start_simulator().await;
    let proxy = start_simulator_proxy(&simulator.url()).await;
    let client = reqwest::Client::new();

    let raw_url = format!(
        "{}/v1beta1/projects/{VERTEX_PROJECT}/locations/{VERTEX_LOCATION}/publishers/anthropic/models/{VERTEX_MODEL}:rawPredict",
        proxy.url()
    );
    let stream_url = format!(
        "{}/v1beta1/projects/{VERTEX_PROJECT}/locations/{VERTEX_LOCATION}/publishers/anthropic/models/{VERTEX_MODEL}:streamRawPredict",
        proxy.url()
    );

    let raw = json_post(
        &client,
        raw_url,
        json!({"anthropic_version":"vertex-2023-10-16","max_tokens":32,"messages":[{"role":"user","content":"hi"}]}),
    )
    .await;
    assert_eq!(raw.status(), 200);
    assert_simulator_header(&raw);
    let raw_body: Value = raw.json().await.unwrap();
    assert_eq!(
        raw_body["model"],
        json!("headroom-simulator-vertex-anthropic")
    );

    let stream = json_post(
        &client,
        stream_url,
        json!({"anthropic_version":"vertex-2023-10-16","stream":true,"max_tokens":32,"messages":[{"role":"user","content":"hi"}]}),
    )
    .await;
    assert_eq!(stream.status(), 200);
    assert_simulator_header(&stream);
    let text = stream.text().await.unwrap();
    assert!(text.contains("event: message_start"));
    assert!(text.contains("simulated anthropic stream"));

    proxy.shutdown().await;
    simulator.shutdown().await;
}

#[tokio::test]
async fn simulator_provider_errors_flow_through_headroom() {
    let vertex_path = format!(
        "/v1beta1/projects/{VERTEX_PROJECT}/locations/{VERTEX_LOCATION}/publishers/anthropic/models/{VERTEX_MODEL}:rawPredict"
    );
    let simulator = start_simulator_with_config(SimulatorConfig {
        stubs: vec![
            json_error_stub(
                "openai-rate-limit",
                "/v1/chat/completions",
                "rate-limit-me",
                429,
                "rate_limit_error",
            ),
            json_error_stub(
                "anthropic-overloaded",
                "/v1/messages",
                "server-error-me",
                529,
                "overloaded_error",
            ),
            json_error_stub(
                "bedrock-upstream-fault",
                &format!("/model/{BEDROCK_MODEL}/invoke"),
                "bedrock-fail",
                502,
                "bedrock_upstream_error",
            ),
            json_error_stub(
                "vertex-upstream-fault",
                &vertex_path,
                "vertex-fail",
                503,
                "vertex_upstream_error",
            ),
        ],
    })
    .await;
    let proxy = start_simulator_proxy(&simulator.url()).await;
    let client = reqwest::Client::new();

    let cases = [
        (
            format!("{}/v1/chat/completions", proxy.url()),
            json!({"model":"gpt-4o","messages":[{"role":"user","content":"rate-limit-me"}]}),
            429,
            "rate_limit_error",
        ),
        (
            format!("{}/v1/messages", proxy.url()),
            json!({"model":"claude-3-5-sonnet","max_tokens":32,"messages":[{"role":"user","content":"server-error-me"}]}),
            529,
            "overloaded_error",
        ),
        (
            format!("{}/model/{BEDROCK_MODEL}/invoke", proxy.url()),
            json!({"anthropic_version":"bedrock-2023-05-31","max_tokens":32,"messages":[{"role":"user","content":"bedrock-fail"}]}),
            502,
            "bedrock_upstream_error",
        ),
        (
            format!("{}{}", proxy.url(), vertex_path),
            json!({"anthropic_version":"vertex-2023-10-16","max_tokens":32,"messages":[{"role":"user","content":"vertex-fail"}]}),
            503,
            "vertex_upstream_error",
        ),
    ];

    for (url, body, expected_status, expected_type) in cases {
        let resp = json_post(&client, url, body).await;
        assert_eq!(resp.status().as_u16(), expected_status);
        assert_simulator_header(&resp);
        assert!(
            resp.headers().get("x-simulator-error-path").is_some(),
            "configured simulator error path should survive Headroom response filtering"
        );
        let error_body: Value = resp.json().await.unwrap();
        assert_eq!(error_body["error"]["type"], json!(expected_type));
    }

    proxy.shutdown().await;
    simulator.shutdown().await;
}

#[tokio::test]
async fn headroom_preflight_errors_stop_before_simulator_fallback() {
    let simulator = start_simulator().await;
    let proxy = start_simulator_proxy_without_bedrock_credentials(&simulator.url()).await;
    let client = reqwest::Client::new();

    let bedrock = json_post(
        &client,
        format!("{}/model/{BEDROCK_MODEL}/invoke", proxy.url()),
        json!({
            "anthropic_version":"bedrock-2023-05-31",
            "max_tokens":32,
            "messages":[{"role":"user","content":"must not be forwarded unsigned"}]
        }),
    )
    .await;
    assert_eq!(bedrock.status(), 500);
    assert_not_simulator_response(&bedrock);
    let bedrock_body: Value = bedrock.json().await.unwrap();
    assert_eq!(
        bedrock_body["error"]["type"],
        json!("bedrock_credentials_missing")
    );

    let vertex = json_post(
        &client,
        format!(
            "{}/v1beta1/projects/{VERTEX_PROJECT}/locations/{VERTEX_LOCATION}/publishers/anthropic/models/{VERTEX_MODEL}:rawPredict",
            proxy.url()
        ),
        json!({"model":"must-not-be-in-vertex-envelope","messages":[]}),
    )
    .await;
    assert_eq!(vertex.status(), 400);
    assert_not_simulator_response(&vertex);
    let vertex_body = vertex.text().await.unwrap();
    assert_eq!(vertex_body, "vertex envelope invalid");

    proxy.shutdown().await;
    simulator.shutdown().await;
}
