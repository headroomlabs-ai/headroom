use std::net::SocketAddr;

use headroom_simulators::config::{
    ConfiguredResponse, JsonPointerMatch, SimulatorConfig, StubRule,
};
use headroom_simulators::{build_app, Simulator};
use serde_json::{json, Value};
use tokio::sync::oneshot;

struct TestServer {
    addr: SocketAddr,
    shutdown: Option<oneshot::Sender<()>>,
    task: tokio::task::JoinHandle<()>,
}

impl TestServer {
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

async fn start(config: SimulatorConfig) -> TestServer {
    let app = build_app(Simulator::new(config)).into_make_service();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    let (tx, rx) = oneshot::channel::<()>();
    let task = tokio::spawn(async move {
        let _ = axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = rx.await;
            })
            .await;
    });
    TestServer {
        addr,
        shutdown: Some(tx),
        task,
    }
}

#[tokio::test]
async fn openai_chat_default_is_provider_shaped() {
    let server = start(SimulatorConfig::default()).await;
    let response: Value = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", server.url()))
        .json(&json!({"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(response["object"], "chat.completion");
    assert_eq!(response["choices"][0]["message"]["role"], "assistant");
    server.shutdown().await;
}

#[tokio::test]
async fn configured_stub_overrides_default_response() {
    let server = start(SimulatorConfig {
        stubs: vec![StubRule {
            name: "configured chat".to_string(),
            method: Some("POST".to_string()),
            path: "/v1/chat/completions".to_string(),
            body_contains: None,
            body_json_pointer: Some(JsonPointerMatch {
                pointer: "/messages/0/content".to_string(),
                equals: json!("configured"),
            }),
            response: ConfiguredResponse {
                status: 209,
                headers: [("x-test-stub".to_string(), "yes".to_string())].into(),
                json: Some(json!({"stubbed": true})),
                body: None,
                sse: vec![],
            },
        }],
    })
    .await;
    let response = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", server.url()))
        .json(&json!({"messages":[{"content":"configured"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status().as_u16(), 209);
    assert_eq!(response.headers()["x-test-stub"], "yes");
    assert_eq!(
        response.json::<Value>().await.unwrap(),
        json!({"stubbed": true})
    );
    server.shutdown().await;
}

#[tokio::test]
async fn responses_stream_returns_named_sse_events() {
    let server = start(SimulatorConfig::default()).await;
    let body = reqwest::Client::new()
        .post(format!("{}/v1/responses", server.url()))
        .json(&json!({"model":"gpt-5","input":"hi","stream":true}))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(body.contains("event: response.created"));
    assert!(body.contains("event: response.completed"));
    server.shutdown().await;
}

#[tokio::test]
async fn bedrock_stream_can_emit_binary_eventstream() {
    let server = start(SimulatorConfig::default()).await;
    let bytes = reqwest::Client::new()
        .post(format!(
            "{}/model/anthropic.claude-3-haiku/invoke-with-response-stream",
            server.url()
        ))
        .header("accept", "application/vnd.amazon.eventstream")
        .json(&json!({"messages":[{"role":"user","content":"hi"}]}))
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    assert!(bytes.len() > 16);
    let total_len = u32::from_be_bytes(bytes[0..4].try_into().unwrap()) as usize;
    assert_eq!(total_len, bytes.len());
    server.shutdown().await;
}

#[tokio::test]
async fn vertex_raw_predict_default_is_anthropic_shaped() {
    let server = start(SimulatorConfig::default()).await;
    let response: Value = reqwest::Client::new()
        .post(format!(
            "{}/v1beta1/projects/p/locations/us/publishers/anthropic/models/claude:rawPredict",
            server.url()
        ))
        .json(&json!({"anthropic_version":"vertex-2023-10-16","messages":[]}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(response["type"], "message");
    assert_eq!(response["role"], "assistant");
    server.shutdown().await;
}
