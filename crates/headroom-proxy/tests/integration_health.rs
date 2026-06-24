//! Health endpoints: own /healthz always 200; /healthz/upstream reflects upstream.

mod common;

use common::start_proxy;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn healthz_ok_when_upstream_down() {
    let proxy = start_proxy("http://127.0.0.1:1").await; // unroutable port
    let resp = reqwest::get(format!("{}/healthz", proxy.url()))
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    proxy.shutdown().await;
}

#[tokio::test]
async fn healthz_upstream_503_when_upstream_down() {
    let proxy = start_proxy("http://127.0.0.1:1").await;
    let resp = reqwest::get(format!("{}/healthz/upstream", proxy.url()))
        .await
        .unwrap();
    assert_eq!(resp.status(), 503);
    proxy.shutdown().await;
}

#[tokio::test]
async fn healthz_upstream_200_when_upstream_healthy() {
    let upstream = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/healthz"))
        .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
        .mount(&upstream)
        .await;
    let proxy = start_proxy(&upstream.uri()).await;
    let resp = reqwest::get(format!("{}/healthz/upstream", proxy.url()))
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    proxy.shutdown().await;
}
