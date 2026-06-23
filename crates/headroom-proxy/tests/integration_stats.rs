//! `/stats` endpoint: the Rust-native savings JSON is served by default and
//! exposes the backend-agnostic dashboard contract.

mod common;

use common::{get_stats, post_messages, start_proxy, start_proxy_with};
use wiremock::matchers::{method, path as path_matcher};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Recording is gated on `Config::should_record()` — the compression master
/// switch AND a non-`off` mode. Tests that assert `/stats` recording enable both.
fn enable_recording(c: &mut headroom_proxy::config::Config) {
    c.compression = true;
    c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
}

#[tokio::test]
async fn stats_endpoint_serves_savings_json_by_default() {
    let upstream = MockServer::start().await;
    let proxy = start_proxy(&upstream.uri()).await;

    let resp = reqwest::Client::new()
        .get(format!("{}/stats", proxy.url()))
        .send()
        .await
        .expect("GET /stats");
    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.expect("stats json");

    // Core contract the dashboard consumes is present and well-formed on a
    // fresh proxy (no traffic yet → zeros, not missing keys).
    assert_eq!(body["requests"]["total"], 0);
    assert_eq!(body["requests"]["failed"], 0);
    // Coverage field the dashboard reads to contextualize savings_percent.
    assert_eq!(body["requests"]["compressed"], 0);
    assert_eq!(body["tokens"]["saved"], 0);
    assert_eq!(body["tokens"]["savings_percent"], 0.0);
    assert!(body["requests"]["by_provider"].is_object());
    assert!(body["cost"]["per_model"].is_object());
    assert!(body["persistent_savings"]["lifetime"].is_object());
    assert!(body["display_session"].is_object());

    proxy.shutdown().await;
}

#[tokio::test]
async fn stats_records_llm_request_attributed_by_provider() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path_matcher("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_string("{}"))
        .mount(&upstream)
        .await;

    // Recording is gated on the compression master switch; enable it.
    let proxy = start_proxy_with(&upstream.uri(), enable_recording).await;

    let resp = post_messages(&proxy).await;
    assert_eq!(resp.status(), 200);

    let stats = get_stats(&proxy).await;

    // The request was recorded and attributed to the Anthropic backend.
    assert_eq!(stats["requests"]["total"], 1);
    assert_eq!(stats["requests"]["by_provider"]["anthropic"], 1);

    proxy.shutdown().await;
}

#[tokio::test]
async fn stats_attributes_openai_chat_completions() {
    // The Copilot lane reaches the proxy as OpenAI Chat Completions; it must
    // land in the unified store attributed to `openai`, not lumped with others.
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path_matcher("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_string("{}"))
        .mount(&upstream)
        .await;
    let proxy = start_proxy_with(&upstream.uri(), enable_recording).await;

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", proxy.url()))
        .header("content-type", "application/json")
        .body(
            serde_json::json!({
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 10
            })
            .to_string(),
        )
        .send()
        .await
        .expect("POST /v1/chat/completions");
    assert_eq!(resp.status(), 200);

    let stats = get_stats(&proxy).await;
    assert_eq!(stats["requests"]["by_provider"]["openai"], 1);

    proxy.shutdown().await;
}

#[tokio::test]
async fn dashboard_endpoint_serves_embedded_html() {
    let upstream = MockServer::start().await;
    let proxy = start_proxy(&upstream.uri()).await;

    let resp = reqwest::Client::new()
        .get(format!("{}/dashboard", proxy.url()))
        .send()
        .await
        .expect("GET /dashboard");
    assert_eq!(resp.status(), 200);
    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert!(content_type.contains("text/html"), "got {content_type}");

    let body = resp.text().await.expect("dashboard html");
    // The embedded page is the real dashboard and polls the stats endpoint.
    assert!(body.contains("/stats"), "dashboard should reference /stats");

    proxy.shutdown().await;
}

#[tokio::test]
async fn stats_folds_in_supplemental_python_blocks() {
    let upstream = MockServer::start().await; // Rust-native LLM upstream
    let python = MockServer::start().await; // transitional Python proxy

    // Rust lane: a real /v1/messages request the Rust proxy records itself.
    Mock::given(method("POST"))
        .and(path_matcher("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_string("{}"))
        .mount(&upstream)
        .await;
    // Python lane: a transitional /stats with a Python-only block plus a
    // `requests.total` that must NOT clobber the Rust-native count.
    Mock::given(method("GET"))
        .and(path_matcher("/stats"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "copilot_quota": {"latest": {"used": 7}},
            "requests": {"total": 999}
        })))
        .mount(&python)
        .await;

    let url = format!("{}/stats", python.uri());
    let proxy = start_proxy_with(&upstream.uri(), move |c| {
        enable_recording(c);
        c.upstream_stats_url = Some(url.clone());
    })
    .await;

    // Drive one Rust-native request so the unified store holds real Rust data
    // alongside the folded-in Python block.
    let resp = post_messages(&proxy).await;
    assert_eq!(resp.status(), 200);

    let stats = get_stats(&proxy).await;

    // Harmony: the Python-only block surfaces AND the Rust-native count is the
    // real one (1), not the Python proxy's 999 — both backends coexist.
    assert_eq!(stats["copilot_quota"]["latest"]["used"], 7);
    assert_eq!(stats["requests"]["total"], 1);

    proxy.shutdown().await;
}

#[tokio::test]
async fn stats_fail_open_when_python_upstream_unreachable() {
    let upstream = MockServer::start().await;
    // Port 1 is never listening → fetch fails; /stats must still succeed.
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.upstream_stats_url = Some("http://127.0.0.1:1/stats".to_string());
    })
    .await;

    let resp = reqwest::Client::new()
        .get(format!("{}/stats", proxy.url()))
        .send()
        .await
        .expect("GET /stats");
    assert_eq!(resp.status(), 200);
    let stats: serde_json::Value = resp.json().await.expect("stats json");
    assert!(stats.get("copilot_quota").is_none());
    assert_eq!(stats["requests"]["total"], 0);

    proxy.shutdown().await;
}

#[tokio::test]
async fn stats_persist_across_restart() {
    let dir = std::env::temp_dir().join(format!("hr-persist-{}", uuid::Uuid::new_v4()));
    let path = dir.join("savings.json");

    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path_matcher("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_string("{}"))
        .mount(&upstream)
        .await;

    // First instance: persists to `path`, records one request, shuts down.
    {
        let p = path.clone();
        let proxy = start_proxy_with(&upstream.uri(), move |c| {
            enable_recording(c);
            c.savings_path = Some(p);
        })
        .await;
        let resp = post_messages(&proxy).await;
        assert_eq!(resp.status(), 200);
        proxy.shutdown().await;
    }
    assert!(path.exists(), "savings file should have been written");

    // Second instance: same path, no traffic — `/stats` reflects the persisted
    // lifetime from the first instance.
    {
        let p = path.clone();
        let proxy = start_proxy_with(&upstream.uri(), move |c| c.savings_path = Some(p)).await;
        let stats = get_stats(&proxy).await;
        assert_eq!(stats["requests"]["total"], 1);
        assert_eq!(stats["persistent_savings"]["lifetime"]["requests"], 1);
        proxy.shutdown().await;
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn health_endpoint_serves_dashboard_shape() {
    // The embedded dashboard polls `/health` and reads `{ status, version }`
    // (its status pill checks `status === "healthy"`). Served by the proxy
    // itself so the dashboard's poll cycle works without an upstream.
    let upstream = MockServer::start().await;
    let proxy = start_proxy(&upstream.uri()).await;

    let resp = reqwest::Client::new()
        .get(format!("{}/health", proxy.url()))
        .send()
        .await
        .expect("GET /health");
    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.expect("health json");
    assert_eq!(body["status"], "healthy");
    assert_eq!(body["service"], "headroom-proxy");
    assert!(body["version"].as_str().is_some_and(|v| !v.is_empty()));

    proxy.shutdown().await;
}

#[tokio::test]
async fn transformations_feed_serves_empty_well_formed_feed() {
    // The Rust proxy keeps no per-request transformation detail, so the feed is
    // empty but well-formed — the dashboard renders "no data" instead of
    // erroring on a 404/502.
    let upstream = MockServer::start().await;
    let proxy = start_proxy(&upstream.uri()).await;

    let resp = reqwest::Client::new()
        .get(format!("{}/transformations/feed?limit=50", proxy.url()))
        .send()
        .await
        .expect("GET /transformations/feed");
    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.expect("feed json");
    assert_eq!(body["transformations"].as_array().unwrap().len(), 0);
    assert_eq!(body["log_full_messages"], false);

    proxy.shutdown().await;
}

#[tokio::test]
async fn stats_history_endpoint_serves_history_contract() {
    // `/stats-history` exposes the durable history the dashboard's Historical
    // view consumes: lifetime totals, the raw checkpoint list, and a daily
    // rollup series. The per-day rollup logic itself is unit-tested in
    // `build_history_json`; here we cover the route + contract shape and that
    // lifetime reflects recorded traffic. (A no-savings request increments
    // lifetime but adds no checkpoint — history checkpoints are gated on
    // `saved > 0`, and this branch's compressors are no-ops.)
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path_matcher("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_string("{}"))
        .mount(&upstream)
        .await;
    let proxy = start_proxy_with(&upstream.uri(), enable_recording).await;

    // Fresh proxy: well-formed and empty.
    let empty: serde_json::Value = reqwest::Client::new()
        .get(format!("{}/stats-history", proxy.url()))
        .send()
        .await
        .expect("GET /stats-history")
        .json()
        .await
        .expect("history json");
    assert_eq!(empty["lifetime"]["requests"], 0);
    assert_eq!(empty["history"].as_array().unwrap().len(), 0);
    assert_eq!(empty["series"]["daily"].as_array().unwrap().len(), 0);

    // Record one request; lifetime should advance and the contract stays valid.
    post_messages(&proxy).await;

    let hist: serde_json::Value = reqwest::Client::new()
        .get(format!("{}/stats-history", proxy.url()))
        .send()
        .await
        .expect("GET /stats-history")
        .json()
        .await
        .expect("history json");
    assert_eq!(hist["lifetime"]["requests"], 1);
    assert!(hist["history"].is_array());
    assert!(hist["series"]["daily"].is_array());

    proxy.shutdown().await;
}

#[tokio::test]
async fn stats_counts_failed_when_upstream_returns_5xx() {
    // A non-2xx upstream must count toward requests.failed — the request is
    // recorded after the upstream status is known, not optimistically before.
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path_matcher("/v1/messages"))
        .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
        .mount(&upstream)
        .await;
    let proxy = start_proxy_with(&upstream.uri(), enable_recording).await;

    let resp = post_messages(&proxy).await;
    assert_eq!(resp.status(), 500);

    let stats = get_stats(&proxy).await;
    assert_eq!(stats["requests"]["total"], 1);
    assert_eq!(
        stats["requests"]["failed"], 1,
        "5xx upstream must count as failed"
    );

    proxy.shutdown().await;
}

#[tokio::test]
async fn stats_does_not_count_failed_on_2xx() {
    // Sanity counterpart: a 2xx upstream records total but not failed.
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path_matcher("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_string("{}"))
        .mount(&upstream)
        .await;
    let proxy = start_proxy_with(&upstream.uri(), enable_recording).await;

    post_messages(&proxy).await;

    let stats = get_stats(&proxy).await;
    assert_eq!(stats["requests"]["total"], 1);
    assert_eq!(stats["requests"]["failed"], 0);

    proxy.shutdown().await;
}

#[tokio::test]
async fn stats_exposes_recent_requests_feed() {
    // The dashboard's Recent Requests panel reads `stats.recent_requests`. After
    // one request the feed has a well-formed, keyed row.
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path_matcher("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_string("{}"))
        .mount(&upstream)
        .await;
    let proxy = start_proxy_with(&upstream.uri(), enable_recording).await;

    // Fresh proxy: present and empty.
    let empty = get_stats(&proxy).await;
    assert!(empty["recent_requests"].is_array());
    assert_eq!(empty["recent_requests"].as_array().unwrap().len(), 0);

    post_messages(&proxy).await;

    let stats = get_stats(&proxy).await;
    let feed = stats["recent_requests"].as_array().unwrap();
    assert_eq!(feed.len(), 1);
    let row = &feed[0];
    assert!(row["request_id"].as_str().is_some_and(|s| !s.is_empty()));
    assert_eq!(row["model"], "claude-haiku-4-5");
    assert!(row["savings_percent"].is_number());
    assert!(row["total_latency_ms"].is_number());

    proxy.shutdown().await;
}

#[tokio::test]
async fn background_flusher_persists_off_the_request_path() {
    // End-to-end proof of the off-hot-path design: `record` writes nothing, and
    // the background flusher (spawned with a short interval here) does the disk
    // I/O on the blocking pool. Guards against reintroducing synchronous
    // request-path persistence.
    use headroom_proxy::observability::stats::{
        load_state, RequestOutcome, SavingsStore, StoreConfig,
    };
    use std::sync::Arc;

    let dir = std::env::temp_dir().join(format!("hr-bgflush-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("s.json");

    let store = Arc::new(SavingsStore::with_path(&path, StoreConfig::default()));
    let _flusher =
        headroom_proxy::spawn_savings_flusher(store.clone(), std::time::Duration::from_millis(40));

    // record() only marks dirty — the file does not exist yet.
    store.record(
        &RequestOutcome {
            provider: "anthropic".to_string(),
            model: "claude-haiku-4-5".to_string(),
            tokens_before: 100,
            tokens_after: 50,
            ..Default::default()
        },
        std::time::SystemTime::now(),
    );
    assert!(!path.exists(), "record must not write on the request path");

    // The background flusher persists it within a couple of intervals.
    for _ in 0..50 {
        if path.exists() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    assert_eq!(load_state(&path).lifetime.requests, 1);

    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn stats_records_connect_error_as_failed() {
    // A transport error (dead upstream) on the forward_http lane must be recorded
    // as a failed request — consistent with the Bedrock/Vertex handlers — not
    // dropped silently, so OpenAI/Anthropic outages stay visible in /stats.
    let proxy = start_proxy_with("http://127.0.0.1:1", enable_recording).await;

    let resp = post_messages(&proxy).await;
    assert!(resp.status().is_server_error(), "got {}", resp.status());

    let stats = get_stats(&proxy).await;
    assert_eq!(stats["requests"]["total"], 1);
    assert_eq!(stats["requests"]["failed"], 1);

    proxy.shutdown().await;
}

#[tokio::test]
async fn stats_rejects_oversized_supplemental_response() {
    // The supplemental `/stats` fetch is bounded: a misbehaving upstream
    // returning a huge body is dropped (fail-open), not read into memory whole.
    let upstream = MockServer::start().await;
    let python = MockServer::start().await;
    let huge = serde_json::json!({ "copilot_quota": { "x": "z".repeat(2 * 1024 * 1024) } });
    Mock::given(method("GET"))
        .and(path_matcher("/stats"))
        .respond_with(ResponseTemplate::new(200).set_body_json(huge))
        .mount(&python)
        .await;
    let url = format!("{}/stats", python.uri());
    let proxy = start_proxy_with(&upstream.uri(), move |c| {
        c.upstream_stats_url = Some(url.clone());
    })
    .await;

    let stats = get_stats(&proxy).await;
    // Oversized supplemental is dropped — no copilot_quota folded in, /stats still 200.
    assert!(stats.get("copilot_quota").is_none());
    assert_eq!(stats["requests"]["total"], 0);

    proxy.shutdown().await;
}

#[tokio::test]
async fn stats_not_recorded_when_mode_off() {
    // Decision (b): compression on but mode == off forwards byte-equal with zero
    // savings, so it is not a savings event — the request is still forwarded, but
    // `/stats` stays empty (consistent with the master switch being off).
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path_matcher("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_string("{}"))
        .mount(&upstream)
        .await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::Off;
    })
    .await;

    let resp = post_messages(&proxy).await;
    assert_eq!(resp.status(), 200);

    let stats = get_stats(&proxy).await;
    assert_eq!(stats["requests"]["total"], 0);

    proxy.shutdown().await;
}
