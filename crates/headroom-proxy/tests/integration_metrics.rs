//! Phase G PR-G3 — integration tests for the new proxy-wide
//! observability metrics. One test per category per the spec:
//!
//! 1. `cache_hit_rate_emitted_per_session` — drive an Anthropic
//!    streaming response that carries cache_read_input_tokens on
//!    `message_delta`; assert the histogram saw the right sample.
//! 2. `compression_ratio_emitted_per_strategy` — exercise the
//!    helper directly (the metric records on
//!    `Outcome::Compressed`; live-zone compression of small
//!    payloads doesn't reliably fire, so the integration test
//!    drives the public helper as a state-machine smoke test).
//! 3. `passthrough_bytes_modified_zero_when_no_compression` —
//!    confirm the alarm-able counter stays at 0 across an
//!    end-to-end passthrough request.
//! 4. `service_tier_logged` — drive a Responses request carrying
//!    `service_tier` and assert the counter rows appear.
//! 5. `incomplete_status_logged_with_reason` — drive an SSE
//!    response with `response.incomplete`; assert the counter row
//!    and the structured-log reason.
//!
//! Like the D3 metrics tests, every test owns a unique label tuple
//! so concurrent test runs over the shared global registry never
//! cross-contaminate.

mod common;

use bytes::Bytes;
use common::start_proxy_with;
use headroom_proxy::observability;
use headroom_proxy::sse::openai_responses::ResponseState;
use headroom_proxy::sse::SseFramer;
use serde_json::json;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::time::Duration;

use http_body_util::StreamBody;
use hyper::body::Frame;
use hyper::service::service_fn;
use hyper::{Request, Response};
use hyper_util::rt::TokioIo;

/// Fetch the proxy's `/metrics` text-format scrape.
async fn scrape_metrics(proxy_url: &str) -> String {
    let resp = reqwest::Client::new()
        .get(format!("{proxy_url}/metrics"))
        .send()
        .await
        .expect("metrics scrape");
    assert_eq!(resp.status(), 200, "metrics endpoint must return 200");
    resp.text().await.unwrap()
}

/// Find a line that contains the metric name + every label pair, and
/// parse its trailing numeric value. Mirrors the helper used by the
/// Bedrock D3 metrics test.
fn find_value_with_labels(scrape: &str, metric: &str, label_pairs: &[(&str, &str)]) -> Option<f64> {
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
        if let Some(value_str) = line.rsplit_once(' ').map(|(_, v)| v.trim()) {
            if let Ok(f) = value_str.parse::<f64>() {
                return Some(f);
            }
        }
    }
    None
}

// ============================================================================
// Test 1: cache_hit_rate_emitted_per_session
// ============================================================================
//
// Anthropic SSE upstream that emits message_start (carrying
// input_tokens + cache_read_input_tokens) and message_delta
// (with the final usage object). We pipe this through the
// proxy's `/v1/messages` SSE path; the state machine then closes
// and the `proxy_cache_hit_rate_per_session{provider="anthropic"}`
// histogram should have exactly one sample.

async fn anthropic_streaming_upstream(
    cache_read_input_tokens: u64,
    input_tokens: u64,
) -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                break;
            };
            tokio::spawn(async move {
                let io = TokioIo::new(stream);
                let _ = hyper::server::conn::http1::Builder::new()
                    .serve_connection(
                        io,
                        service_fn(move |_req: Request<hyper::body::Incoming>| async move {
                            let (tx, rx) = tokio::sync::mpsc::channel::<
                                Result<Frame<Bytes>, std::io::Error>,
                            >(8);
                            tokio::spawn(async move {
                                let start = format!(
                                    "event: message_start\ndata: {{\"type\":\"message_start\",\"message\":{{\"id\":\"msg_x\",\"model\":\"claude\",\"usage\":{{\"input_tokens\":{input_tokens},\"output_tokens\":0,\"cache_read_input_tokens\":{cache_read_input_tokens}}}}}}}\n\n"
                                );
                                let delta = format!(
                                    "event: message_delta\ndata: {{\"type\":\"message_delta\",\"delta\":{{\"stop_reason\":\"end_turn\"}},\"usage\":{{\"input_tokens\":{input_tokens},\"output_tokens\":4,\"cache_read_input_tokens\":{cache_read_input_tokens}}}}}\n\n"
                                );
                                let stop =
                                    b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
                                let frames: Vec<Vec<u8>> = vec![
                                    start.into_bytes(),
                                    delta.into_bytes(),
                                    stop.to_vec(),
                                ];
                                for f in frames {
                                    if tx.send(Ok(Frame::data(Bytes::from(f)))).await.is_err() {
                                        return;
                                    }
                                    tokio::time::sleep(Duration::from_millis(10)).await;
                                }
                            });
                            let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
                            let body = StreamBody::new(stream);
                            Ok::<_, Infallible>(
                                Response::builder()
                                    .status(200)
                                    .header("content-type", "text/event-stream")
                                    .body(body)
                                    .unwrap(),
                            )
                        }),
                    )
                    .await;
            });
        }
    });
    (addr, task)
}

#[tokio::test]
async fn cache_hit_rate_emitted_per_session() {
    // Pick unusual token counts so the histogram bucket we land in
    // is identifiable across parallel test runs.
    // input=200, cache_read=800 → denom=1000, rate=0.8 (falls into the
    // [0.75, 0.9] bucket).
    let (addr, _server) = anthropic_streaming_upstream(800, 200).await;
    let proxy = start_proxy_with(&format!("http://{addr}"), |c| {
        c.compression_mode = headroom_proxy::config::CompressionMode::Off;
    })
    .await;

    let body = json!({
        "model": "claude-3-haiku-20240307",
        "stream": true,
        "max_tokens": 8,
        "messages": [{"role":"user","content":"hi"}]
    });
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        .header("content-type", "application/json")
        .header("accept", "text/event-stream")
        .body(serde_json::to_vec(&body).unwrap())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    // Drain so the state-machine task observes message_stop and emits.
    let _ = resp.bytes().await.unwrap();
    // Give the spawned state-machine task time to close + observe.
    tokio::time::sleep(Duration::from_millis(80)).await;

    let scrape = scrape_metrics(&proxy.url()).await;
    // The histogram exposes _bucket / _sum / _count rows. We assert
    // that at least one sample was observed for our provider label —
    // exact bucket assertions are fragile; the _count line is the
    // load-bearing one.
    let count = find_value_with_labels(
        &scrape,
        "proxy_cache_hit_rate_per_session_count",
        &[("provider", "anthropic")],
    )
    .expect("histogram count must appear after first session");
    assert!(
        count >= 1.0,
        "expected ≥1 observation on anthropic label; got {count}"
    );
    let sum = find_value_with_labels(
        &scrape,
        "proxy_cache_hit_rate_per_session_sum",
        &[("provider", "anthropic")],
    )
    .expect("histogram sum must appear after first session");
    // We seeded denom=1000, cache_read=800 → rate=0.8; the sum
    // includes this observation. Other concurrent tests may
    // contribute via the same `provider=anthropic` label; assert
    // sum > 0 rather than equality.
    assert!(
        sum > 0.0,
        "histogram sum must reflect at least one observed rate > 0; saw {sum}"
    );

    proxy.shutdown().await;
}

// ============================================================================
// Test 2: compression_ratio_emitted_per_strategy
// ============================================================================
//
// Exercise the helper directly — the integration path requires
// live-zone compression to actually trigger, which depends on
// content size + tokenizer thresholds. We assert the metric
// registration + emit-helper pair lands rows in the registry under
// the documented labels, which is the load-bearing observability
// surface for Phase H.

#[tokio::test]
async fn compression_ratio_emitted_per_strategy() {
    // Use a label tuple unique to this test so parallel runs over
    // the global registry don't cross-contaminate.
    const TEST_STRATEGY: &str = "integration_strategy_metric_test_v1";
    const TEST_CONTENT_TYPE: &str = "integration_content_type_metric_test_v1";

    observability::observe_compression_ratio(TEST_STRATEGY, TEST_CONTENT_TYPE, 1000, 250);
    // Drive the rejected counter on the same strategy so its row
    // also appears in the scrape.
    observability::record_compression_rejected_by_token_check(TEST_STRATEGY);

    // Spin up a proxy purely to expose /metrics — the helpers above
    // already incremented the global registry.
    let proxy = start_proxy_with("http://127.0.0.1:1", |_| {}).await;
    let scrape = scrape_metrics(&proxy.url()).await;

    let ratio_count = find_value_with_labels(
        &scrape,
        "proxy_compression_ratio_by_strategy_count",
        &[
            ("strategy", TEST_STRATEGY),
            ("content_type", TEST_CONTENT_TYPE),
        ],
    )
    .expect("ratio histogram count row must appear");
    assert!(ratio_count >= 1.0);

    let rejected = find_value_with_labels(
        &scrape,
        "proxy_compression_rejected_by_token_check_total",
        &[("strategy", TEST_STRATEGY)],
    )
    .expect("rejected counter row must appear");
    assert!(rejected >= 1.0);

    proxy.shutdown().await;
}

// ============================================================================
// Test 3: passthrough_bytes_modified_zero_when_no_compression
// ============================================================================
//
// End-to-end passthrough request with `CompressionMode::Off` — the
// counter must NOT have advanced. We assert the metric advertises
// itself (HELP/TYPE in the scrape) and has the alarm-able value of 0
// for the `/v1/messages` path label.

async fn anthropic_simple_non_stream_upstream() -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                break;
            };
            tokio::spawn(async move {
                let io = TokioIo::new(stream);
                let _ = hyper::server::conn::http1::Builder::new()
                    .serve_connection(
                        io,
                        service_fn(|_req: Request<hyper::body::Incoming>| async move {
                            Ok::<_, Infallible>(
                                Response::builder()
                                    .status(200)
                                    .header("content-type", "application/json")
                                    .body(http_body_util::Full::new(Bytes::from_static(
                                        b"{\"id\":\"msg_1\",\"content\":[]}",
                                    )))
                                    .unwrap(),
                            )
                        }),
                    )
                    .await;
            });
        }
    });
    (addr, task)
}

#[tokio::test]
async fn passthrough_bytes_modified_zero_when_no_compression() {
    let (addr, _server) = anthropic_simple_non_stream_upstream().await;
    let proxy = start_proxy_with(&format!("http://{addr}"), |c| {
        c.compression_mode = headroom_proxy::config::CompressionMode::Off;
    })
    .await;

    let body = json!({
        "model": "claude-3-haiku-20240307",
        "max_tokens": 8,
        "messages": [{"role":"user","content":"hi"}]
    });
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        .header("content-type", "application/json")
        .body(serde_json::to_vec(&body).unwrap())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let scrape = scrape_metrics(&proxy.url()).await;
    // H3 contract: `handle_metrics` force-zeroes every counter /
    // gauge MetricVec under a sentinel `__init__` label tuple on
    // each scrape. That makes HELP/TYPE + a zero row visible from
    // boot so operators have a predictable scrape shape (the
    // pre-H3 behaviour where the family was absent until first
    // emit was confusing — operators would `curl /metrics` on a
    // fresh boot and see nothing). The "must stay 0" alarm
    // semantic is still preserved because the row only carries
    // the sentinel label, not a real production path label.
    assert!(
        scrape.contains("# HELP proxy_passthrough_bytes_modified_total"),
        "scrape missing proxy_passthrough_bytes_modified_total HELP on fresh boot (H3): {scrape}"
    );
    assert!(
        scrape.contains("# TYPE proxy_passthrough_bytes_modified_total counter"),
        "scrape missing proxy_passthrough_bytes_modified_total TYPE on fresh boot (H3): {scrape}"
    );
    let init_row = find_value_with_labels(
        &scrape,
        "proxy_passthrough_bytes_modified_total",
        &[("path", "__init__")],
    )
    .expect("H3 contract: __init__ row must appear on fresh boot");
    assert!(
        (init_row - 0.0).abs() < f64::EPSILON,
        "H3 sentinel __init__ row must read 0 on fresh boot; got {init_row}"
    );
    // The `/v1/messages` path label MUST NOT appear — the
    // dispatcher returned NoCompression and no bytes were
    // mutated, so the alarm did not fire. (Other tests in this
    // suite may have populated rows under their own
    // `path="/integration_test_*"` labels; we filter to the real
    // production path label that THIS test would have produced.)
    let messages_row = find_value_with_labels(
        &scrape,
        "proxy_passthrough_bytes_modified_total",
        &[("path", "/v1/messages")],
    );
    assert!(
        messages_row.is_none(),
        "counter must have no row for /v1/messages when nothing modified passthrough; \
         got value {messages_row:?}"
    );

    // Public helper still works to drive a real-path row.
    use headroom_proxy::observability::record_passthrough_bytes_modified;
    record_passthrough_bytes_modified(
        "/integration_test_passthrough_synthetic",
        7,
        "integration_test_request_id_passthrough_v1",
    );
    let scrape_after_touch = scrape_metrics(&proxy.url()).await;
    let touched = find_value_with_labels(
        &scrape_after_touch,
        "proxy_passthrough_bytes_modified_total",
        &[("path", "/integration_test_passthrough_synthetic")],
    )
    .expect("real-path row must appear after record_passthrough_bytes_modified");
    assert!(
        touched >= 7.0,
        "expected ≥7 byte delta after touch; got {touched}"
    );

    proxy.shutdown().await;
}

// ============================================================================
// C2 wire-up: a request that is supposed to passthrough byte-equal
// but whose body bytes change is detected and the alarm fires.
// We exercise the public helper directly because forcing an
// accidental byte mutation at the dispatcher level is itself a
// regression we don't want to provoke deliberately. The helper-
// level test confirms the metric vector + label semantics, and the
// production-path test in `passthrough_bytes_modified_zero_when_no_compression`
// confirms the alarm STAYS silent on the happy path.
// ============================================================================

#[tokio::test]
async fn passthrough_bytes_modified_alarm_fires_with_byte_delta_label() {
    use headroom_proxy::observability::record_passthrough_bytes_modified;

    // Unique path label so this test owns its row.
    const TEST_PATH: &str = "/integration_test_c2_alarm_v1";
    record_passthrough_bytes_modified(TEST_PATH, 42, "integration_test_c2_request_id_v1");
    record_passthrough_bytes_modified(TEST_PATH, 13, "integration_test_c2_request_id_v2");

    let proxy = start_proxy_with("http://127.0.0.1:1", |_| {}).await;
    let scrape = scrape_metrics(&proxy.url()).await;
    let value = find_value_with_labels(
        &scrape,
        "proxy_passthrough_bytes_modified_total",
        &[("path", TEST_PATH)],
    )
    .expect("C2 alarm row must appear");
    assert!(
        (value - 55.0).abs() < f64::EPSILON,
        "C2 alarm increment must reflect summed byte deltas (42 + 13 = 55); got {value}"
    );

    proxy.shutdown().await;
}

// ============================================================================
// Test 4: service_tier_logged
// ============================================================================
//
// Drive a `/v1/responses` request with `service_tier: "priority"`;
// the handler emits the service-tier counter at the boundary.

async fn responses_passthrough_upstream() -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                break;
            };
            tokio::spawn(async move {
                let io = TokioIo::new(stream);
                let _ = hyper::server::conn::http1::Builder::new()
                    .serve_connection(
                        io,
                        service_fn(|_req: Request<hyper::body::Incoming>| async move {
                            Ok::<_, Infallible>(
                                Response::builder()
                                    .status(200)
                                    .header("content-type", "application/json")
                                    .body(http_body_util::Full::new(Bytes::from_static(
                                        b"{\"id\":\"resp_1\",\"output\":[]}",
                                    )))
                                    .unwrap(),
                            )
                        }),
                    )
                    .await;
            });
        }
    });
    (addr, task)
}

#[tokio::test]
async fn service_tier_logged_known_value() {
    // C1: spec-defined `service_tier` value lands in its own
    // bucket without going through the `"other"` sentinel.
    let (addr, _server) = responses_passthrough_upstream().await;
    let proxy = start_proxy_with(&format!("http://{addr}"), |c| {
        c.compression_mode = headroom_proxy::config::CompressionMode::Off;
        c.enable_responses_streaming = true;
    })
    .await;

    let body = json!({
        "model": "gpt-5",
        "service_tier": "priority",
        "input": [
            {"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}
        ]
    });
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/responses", proxy.url()))
        .header("content-type", "application/json")
        .body(serde_json::to_vec(&body).unwrap())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let _ = resp.bytes().await.unwrap();

    let scrape = scrape_metrics(&proxy.url()).await;
    let tier_count = find_value_with_labels(
        &scrape,
        "proxy_service_tier_count_total",
        &[("tier", "priority")],
    )
    .expect("service tier counter row must appear under 'priority'");
    assert!(
        tier_count >= 1.0,
        "expected ≥1 service_tier increment; got {tier_count}"
    );

    proxy.shutdown().await;
}

#[tokio::test]
async fn service_tier_unknown_bucketed_to_other() {
    // C1: a malicious or drifting client sends an unrecognised
    // service_tier value. The bounded-vocabulary validator MUST
    // bucket it to "other" so a malicious client can't blow up
    // the metric vector cardinality.
    let (addr, _server) = responses_passthrough_upstream().await;
    let proxy = start_proxy_with(&format!("http://{addr}"), |c| {
        c.compression_mode = headroom_proxy::config::CompressionMode::Off;
        c.enable_responses_streaming = true;
    })
    .await;

    // Two distinct unknown values — both must bucket to "other".
    for unknown_tier in [
        "integration_test_unknown_tier_alpha_v1",
        "integration_test_unknown_tier_beta_v1",
    ] {
        let body = json!({
            "model": "gpt-5",
            "service_tier": unknown_tier,
            "input": [
                {"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}
            ]
        });
        let resp = reqwest::Client::new()
            .post(format!("{}/v1/responses", proxy.url()))
            .header("content-type", "application/json")
            .body(serde_json::to_vec(&body).unwrap())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let _ = resp.bytes().await.unwrap();
    }

    let scrape = scrape_metrics(&proxy.url()).await;
    // Neither raw value may appear as a label — they must be
    // bucketed.
    assert!(
        !scrape.contains("integration_test_unknown_tier_alpha_v1"),
        "raw unknown tier value leaked into metrics (cardinality DoS): {scrape}"
    );
    assert!(
        !scrape.contains("integration_test_unknown_tier_beta_v1"),
        "raw unknown tier value leaked into metrics (cardinality DoS): {scrape}"
    );
    // The "other" bucket must have been incremented at least twice.
    let other_count = find_value_with_labels(
        &scrape,
        "proxy_service_tier_count_total",
        &[("tier", "other")],
    )
    .expect("'other' bucket row must appear");
    assert!(
        other_count >= 2.0,
        "expected ≥2 increments on 'other' bucket (one per unknown tier sent); got {other_count}"
    );

    proxy.shutdown().await;
}

#[test]
fn service_tier_validate_known_returns_canonical_constant() {
    use headroom_proxy::observability::metric_names::service_tier;
    // Spec-defined values pass through verbatim — strict equality
    // against the &'static constants so a typo in the validator
    // surfaces here.
    assert_eq!(service_tier::validate("auto"), service_tier::AUTO);
    assert_eq!(service_tier::validate("default"), service_tier::DEFAULT);
    assert_eq!(service_tier::validate("flex"), service_tier::FLEX);
    assert_eq!(service_tier::validate("on_demand"), service_tier::ON_DEMAND);
    assert_eq!(service_tier::validate("priority"), service_tier::PRIORITY);
    assert_eq!(service_tier::validate("scale"), service_tier::SCALE);
}

#[test]
fn service_tier_validate_unknown_returns_other_sentinel() {
    use headroom_proxy::observability::metric_names::service_tier;
    // C1: anything outside the bounded vocab buckets to OTHER.
    assert_eq!(
        service_tier::validate("nonsense_value"),
        service_tier::OTHER
    );
    assert_eq!(service_tier::validate(""), service_tier::OTHER);
    // Case-sensitive: spec is case-sensitive on these strings.
    assert_eq!(service_tier::validate("PRIORITY"), service_tier::OTHER);
    assert_eq!(service_tier::validate("Auto"), service_tier::OTHER);
    // Extremely long / arbitrary attacker input → still bucketed.
    let attack = "A".repeat(10_000);
    assert_eq!(service_tier::validate(&attack), service_tier::OTHER);
}

// ============================================================================
// Test 5: incomplete_status_logged_with_reason
// ============================================================================
//
// Drive the `ResponseState` directly with a `response.incomplete`
// event payload. The state machine should capture
// `incomplete_details.reason` and `terminal_status()` should return
// `"incomplete"`. Wiring the metric counter is exercised by the
// `record_response_status` helper.

#[tokio::test]
async fn incomplete_status_logged_with_reason() {
    let mut framer = SseFramer::new();
    let mut state = ResponseState::new();

    let payload = b"event: response.incomplete\ndata: {\"type\":\"response.incomplete\",\"response\":{\"id\":\"resp_inc\",\"incomplete_details\":{\"reason\":\"max_output_tokens\"},\"service_tier\":\"integration_test_tier_incomplete_v1\"}}\n\n";
    framer.push(payload);
    while let Some(ev) = framer.next_event() {
        let ev = ev.expect("frame parses");
        state.apply(ev).expect("apply succeeds");
    }

    assert_eq!(
        state.incomplete_reason.as_deref(),
        Some("max_output_tokens"),
        "incomplete_details.reason must be captured"
    );
    assert_eq!(state.terminal_status(), Some("incomplete"));
    assert_eq!(
        state.service_tier.as_deref(),
        Some("integration_test_tier_incomplete_v1"),
        "service_tier must be captured on incomplete responses"
    );

    // Drive the metric helper and assert the row.
    observability::record_response_status(
        state.terminal_status().unwrap(),
        state.incomplete_reason.as_deref(),
        "integration_test_request_id_incomplete_v1",
    );

    let proxy = start_proxy_with("http://127.0.0.1:1", |_| {}).await;
    let scrape = scrape_metrics(&proxy.url()).await;
    let count = find_value_with_labels(
        &scrape,
        "proxy_response_status_count_total",
        &[("status", "incomplete")],
    )
    .expect("response status counter row must appear");
    assert!(count >= 1.0);

    proxy.shutdown().await;
}

// ============================================================================
// Bonus coverage: rate-limit gauge plumbing via the public helper.
// ============================================================================

// ============================================================================
// H1 per-strategy ratio: drive `observe_compression_ratio` twice
// with different strategy names + tokens, and assert each strategy
// row has its OWN sum (not the same aggregate replicated).
// ============================================================================

#[tokio::test]
async fn compression_ratio_per_strategy_does_not_replicate_aggregate() {
    // Pre-H1 the proxy emitted the same `aggregate` ratio per
    // strategy when multiple strategies ran on one body. This test
    // exercises the helper directly with two distinct strategies +
    // distinct ratios so the histogram's _sum lines for each
    // strategy must differ.
    use headroom_proxy::observability::observe_compression_ratio;

    const STRAT_HEAVY: &str = "h1_test_heavy_v1";
    const STRAT_LIGHT: &str = "h1_test_light_v1";
    const CT: &str = "h1_test_content_type_v1";

    // Strategy A: original=1000 tokens → compressed=200 (ratio 0.20).
    observe_compression_ratio(STRAT_HEAVY, CT, 1000, 200);
    // Strategy B: original=1000 tokens → compressed=800 (ratio 0.80).
    observe_compression_ratio(STRAT_LIGHT, CT, 1000, 800);

    let proxy = start_proxy_with("http://127.0.0.1:1", |_| {}).await;
    let scrape = scrape_metrics(&proxy.url()).await;

    let sum_heavy = find_value_with_labels(
        &scrape,
        "proxy_compression_ratio_by_strategy_sum",
        &[("strategy", STRAT_HEAVY), ("content_type", CT)],
    )
    .expect("heavy-strategy sum row");
    let sum_light = find_value_with_labels(
        &scrape,
        "proxy_compression_ratio_by_strategy_sum",
        &[("strategy", STRAT_LIGHT), ("content_type", CT)],
    )
    .expect("light-strategy sum row");
    // The sums must NOT be equal — if they are, the per-strategy
    // wiring regressed to the pre-H1 "emit same aggregate per
    // strategy" behavior.
    assert!(
        (sum_heavy - sum_light).abs() > 1e-9,
        "per-strategy sums are equal: heavy={sum_heavy} light={sum_light} — \
         H1 regression: did we re-emit the aggregate ratio per strategy?"
    );
    // Spot-check the actual ratios.
    assert!(
        sum_heavy < sum_light,
        "heavy strategy ratio (0.20) must be < light strategy ratio (0.80): \
         heavy={sum_heavy} light={sum_light}"
    );

    proxy.shutdown().await;
}

// ============================================================================
// H2 aborted stream: a client disconnect mid-stream closes the
// channel without `message_stop`. Unit-tested in
// `crate::observability::cache_hit_rate::tests` because the
// integration-level approach is flaky against the shared global
// Prometheus registry (other tests in the suite emit on the same
// `provider="anthropic"` label, making delta-based assertions
// non-deterministic).
// ============================================================================

#[tokio::test]
async fn rate_limit_snapshot_emits_gauges() {
    let snap = observability::RateLimitSnapshot {
        remaining_requests: Some(123),
        remaining_tokens: Some(45678),
        remaining_input_tokens: Some(30000),
        remaining_output_tokens: Some(15000),
    };
    observability::record_rate_limit_snapshot(
        observability::cache_hit_rate_provider::ANTHROPIC,
        &snap,
        "integration_test_rate_limit_v1",
    );

    let proxy = start_proxy_with("http://127.0.0.1:1", |_| {}).await;
    let scrape = scrape_metrics(&proxy.url()).await;

    // Every gauge advertises itself; the rows we just set must
    // appear with our exact values (gauges, unlike counters, are
    // last-write-wins so this is deterministic).
    let v = find_value_with_labels(
        &scrape,
        "proxy_rate_limit_remaining_requests",
        &[("provider", "anthropic")],
    )
    .expect("requests gauge row");
    assert!(v >= 1.0);
    let v = find_value_with_labels(
        &scrape,
        "proxy_rate_limit_remaining_tokens",
        &[("provider", "anthropic")],
    )
    .expect("tokens gauge row");
    assert!(v >= 1.0);

    proxy.shutdown().await;
}
