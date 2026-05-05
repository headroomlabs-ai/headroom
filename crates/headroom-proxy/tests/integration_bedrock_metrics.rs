//! Integration tests for the Phase D PR-D3 Prometheus instrumentation.
//!
//! Coverage:
//!
//! 1. `metrics_increment_per_invoke` — fire 3 invoke calls; assert
//!    `bedrock_invoke_count_total` registers 3 increments tagged
//!    with the right `model` + `region` + `auth_mode=oauth`.
//! 2. `metrics_observe_latency` — fire one invoke; assert
//!    `bedrock_invoke_latency_seconds` observed exactly one sample.
//! 3. `eventstream_metrics_per_message_type` — drive D2's streaming
//!    path with a captured Bedrock binary stream that yields N
//!    `chunk` messages and assert the counter registers
//!    `event_type=chunk` with N. (The Anthropic-on-Bedrock vocabulary
//!    in D2's translator only accepts `:event-type=chunk`; metadata
//!    frames are not produced by Bedrock for the Anthropic shape.
//!    We assert the chunk path; a future PR-H2 may add metadata
//!    frame support and extend this test.)
//! 4. `metrics_endpoint_serves_scrape` — GET `/metrics` and assert
//!    the three Bedrock metric families appear in the text-format
//!    output.
//!
//! All tests use wiremock as the upstream — no live AWS dependency.

mod common;

use aws_credential_types::Credentials;
use bytes::{Bytes, BytesMut};
use common::start_proxy_with_state;
use headroom_proxy::bedrock::MessageBuilder;
use serde_json::json;
use url::Url;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// Each test in this file owns a UNIQUE (model, region) tuple so the
// global Prometheus registry — shared across all parallel tests in
// the same binary — gives each test isolated label rows. Without
// this isolation, parallel-running tests would cross-contaminate the
// counters they read back. Bumping a counter never tears down its
// row, so absolute counts are not assertable after-the-fact;
// per-tuple isolation gives each test a fresh row to assert deltas
// against.
const TEST_MODEL_INVOKE_COUNT: &str = "anthropic.claude-3-haiku-test-invoke-count-v1:0";
const TEST_MODEL_LATENCY: &str = "anthropic.claude-3-haiku-test-latency-v1:0";
const TEST_MODEL_EVENTSTREAM: &str = "anthropic.claude-3-haiku-test-eventstream-v1:0";
const TEST_MODEL_SCRAPE: &str = "anthropic.claude-3-haiku-test-scrape-v1:0";
const TEST_REGION_INVOKE_COUNT: &str = "us-test-invoke-count-1";
const TEST_REGION_LATENCY: &str = "us-test-latency-1";
const TEST_REGION_EVENTSTREAM: &str = "us-test-eventstream-1";
const TEST_REGION_SCRAPE: &str = "us-test-scrape-1";

fn test_credentials() -> Credentials {
    Credentials::new(
        "AKIAEXAMPLEAKIDFORTEST",
        "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        None,
        None,
        "test",
    )
}

async fn bedrock_proxy_with_region(
    upstream: &MockServer,
    region: &str,
    customize: impl FnOnce(&mut headroom_proxy::Config),
) -> common::ProxyHandle {
    let endpoint: Url = upstream.uri().parse().unwrap();
    let region = region.to_string();
    start_proxy_with_state(
        &upstream.uri(),
        |c| {
            c.bedrock_endpoint = Some(endpoint);
            c.bedrock_region = region;
            customize(c);
        },
        |s| s.with_bedrock_credentials(test_credentials()),
    )
    .await
}

async fn mount_simple_invoke_for(upstream: &MockServer, model: &str) {
    Mock::given(method("POST"))
        .and(path(format!("/model/{model}/invoke")))
        .respond_with(ResponseTemplate::new(200).set_body_string(r#"{"id":"msg_x","content":[]}"#))
        .mount(upstream)
        .await;
}

/// Fetch the proxy's `/metrics` text-format scrape.
async fn scrape_metrics(proxy_url: &str) -> String {
    let resp = reqwest::Client::new()
        .get(format!("{proxy_url}/metrics"))
        .send()
        .await
        .expect("metrics scrape");
    assert_eq!(resp.status(), 200, "metrics endpoint must return 200");
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    assert!(
        ct.starts_with("text/plain"),
        "metrics content-type must be text/plain (Prometheus text format); got {ct}"
    );
    resp.text().await.unwrap()
}

/// Count the number of Prometheus text lines that contain the
/// metric name + every label key/value pair in `label_pairs`. The
/// label-set rendering uses lexical ordering of label names so we
/// MUST NOT compare exact substrings — instead, every pair must
/// appear in the same line, in any order.
fn count_lines_with_labels(
    scrape: &str,
    metric: &str,
    label_pairs: &[(&str, &str)],
) -> Option<u64> {
    for line in scrape.lines() {
        if !line.starts_with(metric) {
            continue;
        }
        if !label_pairs
            .iter()
            .all(|(k, v)| line.contains(&format!("{k}=\"{v}\"")))
        {
            continue;
        }
        // Counter / gauge: " <value>" tail. We split on the last
        // whitespace, parse as u64.
        if let Some(value_str) = line.rsplit_once(' ').map(|(_, v)| v.trim()) {
            if let Ok(value) = value_str.parse::<u64>() {
                return Some(value);
            }
            if let Ok(f) = value_str.parse::<f64>() {
                return Some(f as u64);
            }
        }
    }
    None
}

/// Test 1: `bedrock_invoke_count_total` increments per request
/// with the right model / region / auth_mode labels.
#[tokio::test]
async fn metrics_increment_per_invoke() {
    let upstream = MockServer::start().await;
    mount_simple_invoke_for(&upstream, TEST_MODEL_INVOKE_COUNT).await;
    let proxy = bedrock_proxy_with_region(&upstream, TEST_REGION_INVOKE_COUNT, |c| {
        c.compression_mode = headroom_proxy::config::CompressionMode::Off;
    })
    .await;

    // Per-tuple-isolated counter — start at 0 (no other test
    // touches this label set), so absolute count == invocations.
    let payload = json!({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 8,
        "messages": [{"role":"user","content":"hi"}]
    });
    let body = serde_json::to_vec(&payload).unwrap();
    for _ in 0..3 {
        let resp = reqwest::Client::new()
            .post(format!(
                "{}/model/{TEST_MODEL_INVOKE_COUNT}/invoke",
                proxy.url()
            ))
            .header("content-type", "application/json")
            .body(body.clone())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
    }

    let after = scrape_metrics(&proxy.url()).await;
    let after_count = count_lines_with_labels(
        &after,
        "bedrock_invoke_count_total",
        &[
            ("model", TEST_MODEL_INVOKE_COUNT),
            ("region", TEST_REGION_INVOKE_COUNT),
            ("auth_mode", "oauth"),
        ],
    )
    .expect("counter row must appear after first request");
    assert_eq!(
        after_count, 3,
        "expected exactly 3 increments on isolated labels; got {after_count}"
    );

    proxy.shutdown().await;
}

/// Test 2: `bedrock_invoke_latency_seconds` records exactly one
/// sample for one request.
#[tokio::test]
async fn metrics_observe_latency() {
    let upstream = MockServer::start().await;
    mount_simple_invoke_for(&upstream, TEST_MODEL_LATENCY).await;
    let proxy = bedrock_proxy_with_region(&upstream, TEST_REGION_LATENCY, |c| {
        c.compression_mode = headroom_proxy::config::CompressionMode::Off;
    })
    .await;

    let payload = json!({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 8,
        "messages": [{"role":"user","content":"hi"}]
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/model/{TEST_MODEL_LATENCY}/invoke", proxy.url()))
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let after = scrape_metrics(&proxy.url()).await;
    let after_count = count_lines_with_labels(
        &after,
        "bedrock_invoke_latency_seconds_count",
        &[
            ("model", TEST_MODEL_LATENCY),
            ("region", TEST_REGION_LATENCY),
        ],
    )
    .expect("histogram count row must appear after first request");
    assert_eq!(
        after_count, 1,
        "expected exactly 1 latency observation on isolated labels; got {after_count}"
    );

    // Sum line for the same labels must appear and be > 0.
    let sum_line = after
        .lines()
        .find(|l| {
            l.starts_with("bedrock_invoke_latency_seconds_sum")
                && l.contains(&format!("model=\"{TEST_MODEL_LATENCY}\""))
        })
        .expect("histogram sum line must appear for our labels");
    let sum_value: f64 = sum_line
        .rsplit_once(' ')
        .map(|(_, v)| v.trim())
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    assert!(
        sum_value > 0.0,
        "histogram sum must reflect a real observation > 0s; saw {sum_value}"
    );

    proxy.shutdown().await;
}

/// Synthesise N chunk EventStream messages.
fn synthesize_chunks(n: usize) -> Bytes {
    let mut buf = BytesMut::new();
    for i in 0..n {
        let payload = serde_json::to_string(&json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": format!("t{i}")}
        }))
        .unwrap();
        let bytes = MessageBuilder::new()
            .header_string(":event-type", "chunk")
            .header_string(":content-type", "application/json")
            .header_string(":message-type", "event")
            .payload(Bytes::from(payload))
            .build();
        buf.extend_from_slice(&bytes);
    }
    buf.freeze()
}

/// Test 3: per-EventStream-message metrics increment with the
/// correct `event_type` label.
#[tokio::test]
async fn eventstream_metrics_per_message_type() {
    let upstream = MockServer::start().await;
    let chunks = synthesize_chunks(5);
    Mock::given(method("POST"))
        .and(path(format!(
            "/model/{TEST_MODEL_EVENTSTREAM}/invoke-with-response-stream"
        )))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/vnd.amazon.eventstream")
                .set_body_bytes(chunks.to_vec()),
        )
        .mount(&upstream)
        .await;

    let proxy = bedrock_proxy_with_region(&upstream, TEST_REGION_EVENTSTREAM, |c| {
        c.compression_mode = headroom_proxy::config::CompressionMode::Off;
    })
    .await;

    // Default Accept → SSE translation, which is the path that
    // parses messages and increments the counter (passthrough mode
    // forwards bytes verbatim and therefore can't categorize event
    // types — the spec defers that to a future H2 PR).
    let payload = json!({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 8,
        "messages": [{"role":"user","content":"hi"}]
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!(
            "{}/model/{TEST_MODEL_EVENTSTREAM}/invoke-with-response-stream",
            proxy.url()
        ))
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    // Drain the response body so the translator runs to completion.
    let _ = resp.bytes().await.unwrap();

    let after = scrape_metrics(&proxy.url()).await;
    let after_count = count_lines_with_labels(
        &after,
        "bedrock_eventstream_message_count_total",
        &[
            ("model", TEST_MODEL_EVENTSTREAM),
            ("region", TEST_REGION_EVENTSTREAM),
            ("event_type", "chunk"),
        ],
    )
    .expect("eventstream chunk counter row must appear after first stream");
    assert_eq!(
        after_count, 5,
        "expected 5 chunk increments on isolated labels; got {after_count}"
    );

    proxy.shutdown().await;
}

/// Test 4: `/metrics` endpoint serves a valid Prometheus text-format
/// scrape that includes the three Bedrock metric families. Every
/// metric family must be touched at least once for the
/// `prometheus` crate to render its HELP/TYPE lines (`gather()`
/// skips empty vectors), so this test explicitly drives both the
/// invoke and the streaming routes — each populates a different
/// family, and the latency histogram comes for free with the
/// invoke route.
#[tokio::test]
async fn metrics_endpoint_serves_scrape() {
    let upstream = MockServer::start().await;
    mount_simple_invoke_for(&upstream, TEST_MODEL_SCRAPE).await;
    // Mount the streaming endpoint too, so the third metric family
    // (eventstream message counter) gets at least one increment
    // and its HELP/TYPE lines render in the scrape.
    let chunks = synthesize_chunks(1);
    Mock::given(method("POST"))
        .and(path(format!(
            "/model/{TEST_MODEL_SCRAPE}/invoke-with-response-stream"
        )))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/vnd.amazon.eventstream")
                .set_body_bytes(chunks.to_vec()),
        )
        .mount(&upstream)
        .await;
    let proxy = bedrock_proxy_with_region(&upstream, TEST_REGION_SCRAPE, |c| {
        c.compression_mode = headroom_proxy::config::CompressionMode::Off;
    })
    .await;

    // Fire one invoke (populates invoke_count + invoke_latency)
    // and one streaming invoke (populates eventstream_count).
    let payload = json!({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 8,
        "messages": [{"role":"user","content":"hi"}]
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/model/{TEST_MODEL_SCRAPE}/invoke", proxy.url()))
        .header("content-type", "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let stream_resp = reqwest::Client::new()
        .post(format!(
            "{}/model/{TEST_MODEL_SCRAPE}/invoke-with-response-stream",
            proxy.url()
        ))
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(stream_resp.status(), 200);
    let _ = stream_resp.bytes().await.unwrap();

    let scrape = scrape_metrics(&proxy.url()).await;
    // HELP + TYPE lines are advertised even before any increment;
    // after the increment the labelled rows also appear.
    assert!(
        scrape.contains("# HELP bedrock_invoke_count_total"),
        "scrape missing bedrock_invoke_count_total HELP: {scrape}"
    );
    assert!(
        scrape.contains("# TYPE bedrock_invoke_count_total counter"),
        "scrape missing bedrock_invoke_count_total TYPE: {scrape}"
    );
    assert!(
        scrape.contains("# HELP bedrock_invoke_latency_seconds"),
        "scrape missing bedrock_invoke_latency_seconds HELP"
    );
    assert!(
        scrape.contains("# TYPE bedrock_invoke_latency_seconds histogram"),
        "scrape missing bedrock_invoke_latency_seconds TYPE"
    );
    assert!(
        scrape.contains("# HELP bedrock_eventstream_message_count_total"),
        "scrape missing bedrock_eventstream_message_count_total HELP"
    );
    assert!(
        scrape.contains("# TYPE bedrock_eventstream_message_count_total counter"),
        "scrape missing bedrock_eventstream_message_count_total TYPE"
    );

    proxy.shutdown().await;
}
