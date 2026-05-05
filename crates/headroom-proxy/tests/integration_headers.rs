//! Header passthrough + hop-by-hop filtering + X-Forwarded-* injection +
//! internal `x-headroom-*` strip (PR-A5, fixes P5-49).

mod common;

use common::{start_proxy, start_proxy_with};
use headroom_proxy::config::StripInternalHeaders;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn custom_headers_pass_through_both_ways() {
    let upstream = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/h"))
        .respond_with(move |req: &wiremock::Request| {
            assert_eq!(req.headers.get("authorization").unwrap(), "Bearer foo");
            assert_eq!(req.headers.get("x-custom").unwrap(), "bar");
            // Hop-by-hop must be stripped from the upstream-side request.
            assert!(req.headers.get("transfer-encoding").is_none());
            // X-Forwarded-* should be injected.
            let xff = req
                .headers
                .get("x-forwarded-for")
                .unwrap()
                .to_str()
                .unwrap();
            assert!(xff.contains("127.0.0.1"));
            assert!(req.headers.get("x-forwarded-proto").is_some());
            assert!(req.headers.get("x-forwarded-host").is_some());
            ResponseTemplate::new(200)
                .insert_header("x-server-side", "ack")
                .insert_header("x-multi", "v1")
                .append_header("x-multi", "v2")
                // Hop-by-hop on response side must be stripped by the proxy.
                .insert_header("connection", "close")
                .set_body_string("done")
        })
        .mount(&upstream)
        .await;

    let proxy = start_proxy(&upstream.uri()).await;
    let resp = reqwest::Client::new()
        .get(format!("{}/h", proxy.url()))
        .header("authorization", "Bearer foo")
        .header("x-custom", "bar")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(resp.headers().get("x-server-side").unwrap(), "ack");
    assert!(
        resp.headers().get("connection").is_none(),
        "hop-by-hop must be stripped"
    );
    let multi: Vec<_> = resp
        .headers()
        .get_all("x-multi")
        .iter()
        .map(|v| v.to_str().unwrap().to_string())
        .collect();
    assert_eq!(multi, vec!["v1".to_string(), "v2".to_string()]);
    proxy.shutdown().await;
}

#[tokio::test]
async fn x_headroom_request_headers_stripped() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(move |req: &wiremock::Request| {
            // PR-A5: internal x-headroom-* must NOT reach upstream.
            assert!(
                req.headers.get("x-headroom-bypass").is_none(),
                "x-headroom-bypass leaked upstream"
            );
            assert!(
                req.headers.get("x-headroom-mode").is_none(),
                "x-headroom-mode leaked upstream"
            );
            assert!(
                req.headers.get("x-headroom-user-id").is_none(),
                "x-headroom-user-id leaked upstream"
            );
            // Legitimate headers must still arrive.
            assert_eq!(req.headers.get("authorization").unwrap(), "Bearer sk-x");
            assert_eq!(req.headers.get("anthropic-version").unwrap(), "2023-06-01");
            ResponseTemplate::new(200).set_body_string("{}")
        })
        .mount(&upstream)
        .await;

    let proxy = start_proxy(&upstream.uri()).await;
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        .header("authorization", "Bearer sk-x")
        .header("anthropic-version", "2023-06-01")
        .header("x-headroom-bypass", "true")
        .header("x-headroom-mode", "passthrough")
        .header("x-headroom-user-id", "alice")
        .body("{}")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    proxy.shutdown().await;
}

#[tokio::test]
async fn x_headroom_case_insensitive_stripped() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(move |req: &wiremock::Request| {
            // Mixed-case variants — all should be stripped.
            for hdr in [
                "x-headroom-foo",
                "x-headroom-bar",
                "x-headroom-baz",
                "X-Headroom-Foo",
                "X-HEADROOM-BAR",
            ] {
                assert!(
                    req.headers.get(hdr).is_none(),
                    "internal header {hdr} leaked upstream"
                );
            }
            ResponseTemplate::new(200).set_body_string("{}")
        })
        .mount(&upstream)
        .await;

    let proxy = start_proxy(&upstream.uri()).await;
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        .header("X-Headroom-Foo", "1")
        .header("x-Headroom-Bar", "2")
        .header("X-HEADROOM-BAZ", "3")
        .body("{}")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    proxy.shutdown().await;
}

#[tokio::test]
async fn legitimate_headers_passthrough_with_strip_enabled() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/echo"))
        .respond_with(move |req: &wiremock::Request| {
            // Non-internal x-* headers must NOT be stripped.
            assert_eq!(req.headers.get("x-api-key").unwrap(), "k1");
            assert_eq!(req.headers.get("x-trace-id").unwrap(), "trace-1");
            assert_eq!(req.headers.get("authorization").unwrap(), "Bearer x");
            assert_eq!(req.headers.get("anthropic-version").unwrap(), "2023-06-01");
            // Strip happened — internal flag absent.
            assert!(req.headers.get("x-headroom-bypass").is_none());
            ResponseTemplate::new(200)
        })
        .mount(&upstream)
        .await;

    let proxy = start_proxy(&upstream.uri()).await;
    let resp = reqwest::Client::new()
        .post(format!("{}/echo", proxy.url()))
        .header("x-api-key", "k1")
        .header("x-trace-id", "trace-1")
        .header("authorization", "Bearer x")
        .header("anthropic-version", "2023-06-01")
        .header("x-headroom-bypass", "true")
        .body("{}")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    proxy.shutdown().await;
}

#[tokio::test]
async fn disabled_mode_passes_internal_headers_through() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(move |req: &wiremock::Request| {
            // Operator opt-in: internal header IS forwarded.
            assert_eq!(req.headers.get("x-headroom-bypass").unwrap(), "true");
            assert_eq!(req.headers.get("x-headroom-mode").unwrap(), "passthrough");
            ResponseTemplate::new(200).set_body_string("{}")
        })
        .mount(&upstream)
        .await;

    let proxy = start_proxy_with(&upstream.uri(), |cfg| {
        cfg.strip_internal_headers = StripInternalHeaders::Disabled;
    })
    .await;
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        .header("x-headroom-bypass", "true")
        .header("x-headroom-mode", "passthrough")
        .body("{}")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    proxy.shutdown().await;
}

#[tokio::test]
async fn xff_appends_existing_value() {
    let upstream = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/xff"))
        .respond_with(move |req: &wiremock::Request| {
            let xff = req
                .headers
                .get("x-forwarded-for")
                .unwrap()
                .to_str()
                .unwrap();
            // existing 1.2.3.4 must be preserved + appended.
            assert!(
                xff.starts_with("1.2.3.4"),
                "expected appended xff, got: {xff}"
            );
            assert!(xff.contains("127.0.0.1"));
            ResponseTemplate::new(200)
        })
        .mount(&upstream)
        .await;
    let proxy = start_proxy(&upstream.uri()).await;
    let resp = reqwest::Client::new()
        .get(format!("{}/xff", proxy.url()))
        .header("x-forwarded-for", "1.2.3.4")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    proxy.shutdown().await;
}
