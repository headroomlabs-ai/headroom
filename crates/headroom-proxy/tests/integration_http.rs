//! HTTP method round-trip + status passthrough.

mod common;

use common::start_proxy;
use wiremock::matchers::{any, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn all_methods_round_trip_with_body() {
    let upstream = MockServer::start().await;

    for m in ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"] {
        Mock::given(method(m))
            .and(path("/echo"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("x-from-upstream", "yes")
                    .set_body_string(format!("ok-{m}")),
            )
            .mount(&upstream)
            .await;
    }
    Mock::given(method("HEAD"))
        .and(path("/echo"))
        .respond_with(ResponseTemplate::new(200).insert_header("x-from-upstream", "yes"))
        .mount(&upstream)
        .await;

    let proxy = start_proxy(&upstream.uri()).await;
    let client = reqwest::Client::new();

    for m in ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"] {
        let url = format!("{}/echo", proxy.url());
        let resp = client
            .request(reqwest::Method::from_bytes(m.as_bytes()).unwrap(), &url)
            .body(format!("payload-{m}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200, "method {m}");
        assert_eq!(resp.headers().get("x-from-upstream").unwrap(), "yes");
        let body = resp.text().await.unwrap();
        assert_eq!(body, format!("ok-{m}"));
    }

    // HEAD has no body but should round-trip headers + status.
    let head_resp = client
        .head(format!("{}/echo", proxy.url()))
        .send()
        .await
        .unwrap();
    assert_eq!(head_resp.status(), 200);
    assert_eq!(head_resp.headers().get("x-from-upstream").unwrap(), "yes");

    proxy.shutdown().await;
}

#[tokio::test]
async fn upstream_error_codes_passthrough() {
    let upstream = MockServer::start().await;
    for status in [404u16, 500, 502] {
        Mock::given(method("GET"))
            .and(path(format!("/code/{status}")))
            .respond_with(ResponseTemplate::new(status).set_body_string(format!("err-{status}")))
            .mount(&upstream)
            .await;
    }
    let proxy = start_proxy(&upstream.uri()).await;
    let client = reqwest::Client::new();
    for status in [404u16, 500, 502] {
        let resp = client
            .get(format!("{}/code/{status}", proxy.url()))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), status);
        assert_eq!(resp.text().await.unwrap(), format!("err-{status}"));
    }
    proxy.shutdown().await;
}

#[tokio::test]
async fn query_string_preserved() {
    let upstream = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/q"))
        .and(query_param("a", "1"))
        .and(query_param("b", "two"))
        .respond_with(ResponseTemplate::new(200).set_body_string("matched"))
        .mount(&upstream)
        .await;
    Mock::given(any())
        .respond_with(ResponseTemplate::new(418))
        .mount(&upstream)
        .await;

    let proxy = start_proxy(&upstream.uri()).await;
    let resp = reqwest::get(format!("{}/q?a=1&b=two", proxy.url()))
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(resp.text().await.unwrap(), "matched");
    proxy.shutdown().await;
}

#[tokio::test]
async fn one_mb_post_streams_through() {
    let upstream = MockServer::start().await;
    let payload = vec![0xABu8; 1024 * 1024];
    let payload_clone = payload.clone();
    Mock::given(method("POST"))
        .and(path("/upload"))
        .respond_with(move |req: &wiremock::Request| {
            assert_eq!(
                req.body.len(),
                payload_clone.len(),
                "upstream got full body"
            );
            assert_eq!(&req.body[..], &payload_clone[..]);
            ResponseTemplate::new(200).set_body_string("uploaded")
        })
        .mount(&upstream)
        .await;
    let proxy = start_proxy(&upstream.uri()).await;
    let resp = reqwest::Client::new()
        .post(format!("{}/upload", proxy.url()))
        .body(payload)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(resp.text().await.unwrap(), "uploaded");
    proxy.shutdown().await;
}
