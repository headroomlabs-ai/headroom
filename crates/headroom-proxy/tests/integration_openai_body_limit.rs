//! `/v1/chat/completions` and `/v1/responses` buffer the request body with
//! axum's `Bytes` extractor. Without a `DefaultBodyLimit` layer axum caps
//! that at 2 MiB, so `--max-body-bytes` (default 100 MB) never applied.
//! These tests pin that the configured cap governs both routes.

mod common;

use common::start_proxy_with;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const ROUTES: [&str; 2] = ["/v1/chat/completions", "/v1/responses"];

fn body_of(len: usize) -> String {
    format!(
        r#"{{"model":"gpt-4o","messages":[{{"role":"user","content":"{}"}}]}}"#,
        "A".repeat(len)
    )
}

async fn upstream_flag(route: &str) -> (MockServer, Arc<AtomicBool>) {
    let upstream = MockServer::start().await;
    let hit = Arc::new(AtomicBool::new(false));
    let hit2 = hit.clone();
    Mock::given(method("POST"))
        .and(path(route))
        .respond_with(move |_: &wiremock::Request| {
            hit2.store(true, Ordering::SeqCst);
            ResponseTemplate::new(200).set_body_string(r#"{"ok":true}"#)
        })
        .mount(&upstream)
        .await;
    (upstream, hit)
}

#[tokio::test]
async fn three_mib_body_reaches_upstream_with_default_cap() {
    for route in ROUTES {
        let (upstream, hit) = upstream_flag(route).await;
        let proxy = start_proxy_with(&upstream.uri(), |c| c.compression = false).await;
        let resp = reqwest::Client::new()
            .post(format!("{}{}", proxy.url(), route))
            .header("content-type", "application/json")
            .body(body_of(3 * 1024 * 1024))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 200, "{route}: 3 MiB body rejected");
        assert!(hit.load(Ordering::SeqCst), "{route}: upstream never saw it");
        proxy.shutdown().await;
    }
}

#[tokio::test]
async fn body_above_configured_cap_still_413() {
    for route in ROUTES {
        let (upstream, hit) = upstream_flag(route).await;
        let proxy = start_proxy_with(&upstream.uri(), |c| {
            c.compression = false;
            c.max_body_bytes = 1024;
        })
        .await;
        let resp = reqwest::Client::new()
            .post(format!("{}{}", proxy.url(), route))
            .header("content-type", "application/json")
            .body(body_of(4096))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 413, "{route}");
        assert!(!hit.load(Ordering::SeqCst), "{route}: upstream saw it");
        proxy.shutdown().await;
    }
}
