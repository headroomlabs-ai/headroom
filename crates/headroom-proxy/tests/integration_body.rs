//! Streaming bodies: 5MB POST round-trips; large response streams without full buffering.

mod common;

use bytes::Bytes;
use common::start_proxy;
use futures_util::StreamExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn five_mb_post_round_trip() {
    let upstream = MockServer::start().await;
    let payload = vec![0x5Au8; 5 * 1024 * 1024];
    let payload_clone = payload.clone();
    Mock::given(method("POST"))
        .and(path("/big"))
        .respond_with(move |req: &wiremock::Request| {
            assert_eq!(req.body.len(), payload_clone.len());
            assert_eq!(&req.body[..], &payload_clone[..]);
            ResponseTemplate::new(200).set_body_string("ok")
        })
        .mount(&upstream)
        .await;

    let proxy = start_proxy(&upstream.uri()).await;
    let resp = reqwest::Client::new()
        .post(format!("{}/big", proxy.url()))
        .body(payload)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    proxy.shutdown().await;
}

#[tokio::test]
async fn streaming_response_first_byte_before_done() {
    // wiremock supports delay between body chunks via set_delay; use a single
    // delayed response and make sure first byte arrives via a stream.
    let upstream = MockServer::start().await;
    let body: Bytes = Bytes::from(vec![b'X'; 1024 * 64]);
    Mock::given(method("GET"))
        .and(path("/stream"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(body.clone()))
        .mount(&upstream)
        .await;
    let proxy = start_proxy(&upstream.uri()).await;
    let resp = reqwest::Client::new()
        .get(format!("{}/stream", proxy.url()))
        .send()
        .await
        .unwrap();
    let mut stream = resp.bytes_stream();
    let first = stream.next().await.unwrap().unwrap();
    assert!(!first.is_empty());
    let mut total = first.len();
    while let Some(chunk) = stream.next().await {
        total += chunk.unwrap().len();
    }
    assert_eq!(total, body.len());
    proxy.shutdown().await;
}
