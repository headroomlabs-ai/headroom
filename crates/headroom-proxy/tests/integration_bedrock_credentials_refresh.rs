//! Temporary AWS credentials are refreshed between Bedrock requests.
//!
//! A test provider hands out `AKID_<n>` credentials that expire one
//! second after issue (inside the refresh-ahead window), so each
//! request must ask the provider again and be signed with the new key.
//! No real AWS: the upstream is wiremock.

mod common;

use aws_credential_types::provider::{future, ProvideCredentials};
use aws_credential_types::Credentials;
use common::start_proxy_with_state;
use headroom_proxy::bedrock::BedrockCredentials;
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use url::Url;
use wiremock::matchers::method;
use wiremock::{Mock, MockServer, ResponseTemplate};

const TEST_MODEL: &str = "anthropic.claude-3-haiku-20240307-v1:0";

#[derive(Debug)]
struct ShortLivedProvider {
    calls: Arc<AtomicUsize>,
}

impl ProvideCredentials for ShortLivedProvider {
    fn provide_credentials<'a>(&'a self) -> future::ProvideCredentials<'a>
    where
        Self: 'a,
    {
        let n = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        future::ProvideCredentials::ready(Ok(Credentials::new(
            format!("AKID_{n}"),
            "secret",
            Some("session-token".into()),
            Some(SystemTime::now() + Duration::from_secs(1)),
            "short-lived-test",
        )))
    }
}

async fn signed_key_ids(action: &str) -> (Vec<String>, usize) {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_string(r#"{"id":"msg_x","content":[]}"#),
        )
        .mount(&upstream)
        .await;
    let calls = Arc::new(AtomicUsize::new(0));
    let provider = ShortLivedProvider {
        calls: calls.clone(),
    };
    let endpoint: Url = upstream.uri().parse().unwrap();
    let proxy = start_proxy_with_state(
        &upstream.uri(),
        |c| {
            c.bedrock_endpoint = Some(endpoint);
            c.compression_mode = headroom_proxy::config::CompressionMode::Off;
        },
        |s| s.with_bedrock_credentials_source(BedrockCredentials::new(provider)),
    )
    .await;

    let body = serde_json::to_vec(&json!({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}]
    }))
    .unwrap();
    for _ in 0..2 {
        let resp = reqwest::Client::new()
            .post(format!("{}/model/{TEST_MODEL}/{action}", proxy.url()))
            .header("content-type", "application/json")
            .body(body.clone())
            .send()
            .await
            .unwrap();
        let _ = resp.bytes().await;
    }
    proxy.shutdown().await;

    let keys = upstream
        .received_requests()
        .await
        .unwrap()
        .iter()
        .map(|r| {
            let auth = r.headers.get("authorization").unwrap().to_str().unwrap();
            let start = auth.find("Credential=").unwrap() + "Credential=".len();
            auth[start..].split('/').next().unwrap().to_string()
        })
        .collect();
    (keys, calls.load(Ordering::SeqCst))
}

#[tokio::test]
async fn invoke_signs_with_refreshed_credentials_after_expiry() {
    let (keys, calls) = signed_key_ids("invoke").await;
    assert_eq!(keys, vec!["AKID_1", "AKID_2"]);
    assert_eq!(calls, 2);
}

#[tokio::test]
async fn invoke_streaming_signs_with_refreshed_credentials_after_expiry() {
    let (keys, calls) = signed_key_ids("invoke-with-response-stream").await;
    assert_eq!(keys, vec!["AKID_1", "AKID_2"]);
    assert_eq!(calls, 2);
}
