//! Credentials that are missing when the proxy starts and appear later.
//!
//! The startup sequence lives in `main.rs`, so this test runs the real
//! binary. Credentials come from a local ECS container endpoint
//! (`AWS_CONTAINER_CREDENTIALS_FULL_URI`) that answers 503 until it is
//! switched on; the Bedrock upstream is wiremock. Before the fix the
//! failed startup fetch dropped the provider, so the request sent after
//! the endpoint recovered still got `bedrock_credentials_missing`.

use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

const TEST_MODEL: &str = "anthropic.claude-3-haiku-20240307-v1:0";

/// ECS credentials endpoint: 503 until `ready` is set.
struct LateCredentials {
    ready: Arc<AtomicBool>,
}

impl Respond for LateCredentials {
    fn respond(&self, _request: &Request) -> ResponseTemplate {
        if !self.ready.load(Ordering::SeqCst) {
            return ResponseTemplate::new(503);
        }
        ResponseTemplate::new(200).set_body_json(json!({
            "AccessKeyId": "AKIDSTARTUPTEST",
            "SecretAccessKey": "secret",
            "Token": "session-token",
            "Expiration": "2099-01-01T00:00:00Z",
        }))
    }
}

/// Kills the proxy and removes its empty AWS file when the test ends,
/// pass or fail.
struct ProxyProcess(Child, std::path::PathBuf);

impl Drop for ProxyProcess {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
        let _ = std::fs::remove_file(&self.1);
    }
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

#[tokio::test]
async fn credentials_that_appear_after_startup_are_used() {
    let ready = Arc::new(AtomicBool::new(false));
    let creds = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/creds"))
        .respond_with(LateCredentials {
            ready: ready.clone(),
        })
        .mount(&creds)
        .await;
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_string(r#"{"id":"msg_x","content":[]}"#),
        )
        .mount(&upstream)
        .await;

    let empty = std::env::temp_dir().join(format!("hr-empty-aws-{}", std::process::id()));
    std::fs::write(&empty, "").unwrap();
    let port = free_port();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_headroom-proxy"));
    for (key, _) in std::env::vars() {
        if key.starts_with("AWS_") {
            cmd.env_remove(key);
        }
    }
    cmd.env("AWS_CONFIG_FILE", &empty)
        .env("AWS_SHARED_CREDENTIALS_FILE", &empty)
        .env("AWS_EC2_METADATA_DISABLED", "true")
        .env(
            "AWS_CONTAINER_CREDENTIALS_FULL_URI",
            format!("{}/creds", creds.uri()),
        )
        .args([
            "--listen",
            &format!("127.0.0.1:{port}"),
            "--upstream",
            "http://127.0.0.1:1",
            "--enable-bedrock-native",
            "true",
            "--bedrock-endpoint",
            &upstream.uri(),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let _proxy = ProxyProcess(cmd.spawn().expect("spawn headroom-proxy"), empty.clone());

    let client = reqwest::Client::new();
    let base = format!("http://127.0.0.1:{port}");
    let deadline = Instant::now() + Duration::from_secs(60);
    while client.get(format!("{base}/healthz")).send().await.is_err() {
        assert!(Instant::now() < deadline, "proxy did not start");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    let body = serde_json::to_vec(&json!({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}]
    }))
    .unwrap();
    let invoke = || {
        client
            .post(format!("{base}/model/{TEST_MODEL}/invoke"))
            .header("content-type", "application/json")
            .body(body.clone())
            .send()
    };

    let before = invoke().await.unwrap();
    assert_eq!(before.status(), 500, "no credentials yet");

    ready.store(true, Ordering::SeqCst);
    let after = invoke().await.unwrap();
    let status = after.status();
    let text = after.text().await.unwrap();
    assert_eq!(status, 200, "credentials served now: {text}");

    let signed = upstream.received_requests().await.unwrap();
    assert_eq!(signed.len(), 1);
    let auth = signed[0]
        .headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default();
    assert!(
        auth.contains("Credential=AKIDSTARTUPTEST/"),
        "signed with the late credentials: {auth}"
    );
}
