//! Integration tests for the native savings stats surface:
//! `/stats`, `/stats/timeseries`, `/stats/events`, `/dashboard`,
//! and the per-lane recording hooks (deferred SSE finalization,
//! non-streaming JSON usage capture, failed-request accounting,
//! Bedrock camelCase usage).

mod common;

use std::time::Duration;

use common::{
    install_static_token_source, start_proxy_with, start_proxy_with_state, test_credentials,
};
use headroom_proxy::bedrock::MessageBuilder;
use serde_json::{json, Value};
use url::Url;
use wiremock::matchers::{method, path, path_regex};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// The capture tasks finalize asynchronously after the client sees
/// the response — poll /stats until the predicate holds.
async fn wait_for_stats<F>(base: &str, pred: F) -> Value
where
    F: Fn(&Value) -> bool,
{
    let client = reqwest::Client::new();
    for _ in 0..100 {
        let v: Value = client
            .get(format!("{base}/stats"))
            .send()
            .await
            .expect("GET /stats")
            .json()
            .await
            .expect("stats json");
        if pred(&v) {
            return v;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("stats predicate not satisfied within 5s");
}

/// POST a JSON body the way every recording test does.
async fn post_json(url: String, body: &Value) -> reqwest::Response {
    reqwest::Client::new()
        .post(url)
        .header("content-type", "application/json")
        .json(body)
        .send()
        .await
        .expect("POST")
}

/// One binary EventStream frame with the standard event headers.
fn event_frame(event_type: &str, payload: &Value) -> bytes::Bytes {
    MessageBuilder::new()
        .header_string(":message-type", "event")
        .header_string(":event-type", event_type)
        .header_string(":content-type", "application/json")
        .payload(payload.to_string().into())
        .build()
}

#[tokio::test]
async fn stats_endpoints_serve_locally_and_validate_input() {
    let upstream = MockServer::start().await;
    let proxy = start_proxy_with(&upstream.uri(), |_| {}).await;
    let client = reqwest::Client::new();

    let stats: Value = client
        .get(format!("{}/stats", proxy.url()))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(stats["proxy"], "headroom-rust");
    assert_eq!(stats["requests"]["total"], 0);
    // headway's unified-stats reader contract.
    assert!(stats["summary"]["api_requests"].is_u64());
    assert!(stats["tokens"]["proxy_compression_saved"].is_u64());
    assert!(stats["requests"]["cached"].is_u64());

    let ts: Value = client
        .get(format!("{}/stats/timeseries?bucket=day", proxy.url()))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(ts["bucket"], "day");
    let bad = client
        .get(format!("{}/stats/timeseries?bucket=fortnight", proxy.url()))
        .send()
        .await
        .unwrap();
    assert_eq!(bad.status(), 400);

    let events: Value = client
        .get(format!("{}/stats/events", proxy.url()))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(events["events"].as_array().unwrap().is_empty());

    let dash = client
        .get(format!("{}/dashboard", proxy.url()))
        .send()
        .await
        .unwrap();
    assert_eq!(dash.status(), 200);
    let ct = dash
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap();
    assert!(ct.starts_with("text/html"), "content-type: {ct}");
    let body = dash.text().await.unwrap();
    assert!(body.contains("Headroom Proxy — Savings"));
    // Self-contained page: no network resource references of any
    // kind. The only URLs allowed are XML namespaces and data: URIs.
    for needle in [
        "src=\"http",
        "src=\"//",
        "href=\"http",
        "href=\"//",
        "@import",
        "fonts.googleapis",
        "cdn.",
    ] {
        assert!(
            !body.contains(needle),
            "dashboard must be self-contained; found {needle:?}"
        );
    }
}

#[tokio::test]
async fn stats_disabled_falls_through_to_upstream() {
    let upstream = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/stats"))
        .respond_with(ResponseTemplate::new(200).set_body_string("python-proxy-stats"))
        .mount(&upstream)
        .await;
    let proxy = start_proxy_with(&upstream.uri(), |c| c.stats = false).await;

    let body = reqwest::Client::new()
        .get(format!("{}/stats", proxy.url()))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert_eq!(
        body, "python-proxy-stats",
        "with --stats=false the path must tunnel upstream (pre-feature behaviour)"
    );
}

#[tokio::test]
async fn anthropic_non_streaming_response_records_usage() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_1",
            "type": "message",
            "model": "claude-sonnet-4-5-20250929",
            "content": [{"type": "text", "text": "hi"}],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 25,
                "cache_read_input_tokens": 40,
                "cache_creation_input_tokens": 10
            }
        })))
        .mount(&upstream)
        .await;
    // compression=true engages the buffered arm (mode stays Off —
    // recording must work in pure-passthrough deployments too).
    let proxy = start_proxy_with(&upstream.uri(), |c| c.compression = true).await;

    let resp = post_json(
        format!("{}/v1/messages", proxy.url()),
        &json!({
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hello"}]
        }),
    )
    .await;
    assert_eq!(resp.status(), 200);
    let _ = resp.bytes().await.unwrap(); // drain so the capture tee closes

    let stats = wait_for_stats(&proxy.url(), |v| v["requests"]["total"] == 1).await;
    assert_eq!(stats["tokens"]["input"], 100);
    assert_eq!(stats["tokens"]["output"], 25);
    assert_eq!(stats["tokens"]["cache_read"], 40);
    assert_eq!(stats["tokens"]["cache_write"], 10);
    assert_eq!(stats["requests"]["failed"], 0);
    // 40 cache-read tokens ⇒ this request counts as a cache hit.
    assert_eq!(stats["requests"]["cached"], 1);
    assert_eq!(stats["session_by_provider"]["anthropic"]["requests"], 1);
    // 40 cache reads on sonnet pricing → non-zero cache savings.
    assert!(stats["session"]["cache_savings_usd"].as_f64().unwrap() > 0.0);
    assert!(stats["session"]["input_cost_usd"].as_f64().unwrap() > 0.0);
    // Model attribution landed.
    assert_eq!(
        stats["lifetime_by_model"]["claude-sonnet-4-5-20250929"]["requests"],
        1
    );
    // Timeseries picked it up.
    let ts: Value = reqwest::Client::new()
        .get(format!("{}/stats/timeseries?bucket=hour", proxy.url()))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(ts["points"][0]["requests"], 1);
}

/// Recording must not depend on the compression master switch —
/// it is OFF by default, and spend/cache observability is the whole
/// point of the feature on a passthrough deployment. The model is
/// learned from the response there (the request body is never
/// buffered), and compression savings are legitimately 0.
#[tokio::test]
async fn records_with_compression_off_using_model_from_response() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_1",
            "type": "message",
            "model": "claude-sonnet-4-5-20250929",
            "content": [{"type": "text", "text": "hi"}],
            "usage": {"input_tokens": 200, "output_tokens": 10,
                      "cache_read_input_tokens": 80}
        })))
        .mount(&upstream)
        .await;
    // Config::for_test defaults compression to false — same as the
    // shipped binary's default.
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        assert!(!c.compression, "guard: this test covers compression OFF");
    })
    .await;

    let resp = post_json(
        format!("{}/v1/messages", proxy.url()),
        &json!({
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hello"}]
        }),
    )
    .await;
    assert_eq!(resp.status(), 200);
    let _ = resp.bytes().await.unwrap();

    let stats = wait_for_stats(&proxy.url(), |v| v["requests"]["total"] == 1).await;
    assert_eq!(stats["tokens"]["input"], 200);
    assert_eq!(stats["tokens"]["cache_read"], 80);
    assert_eq!(stats["tokens"]["saved"], 0, "no compression ran → 0 saved");
    // Model came from the response envelope, so spend is priced.
    assert_eq!(
        stats["lifetime_by_model"]["claude-sonnet-4-5-20250929"]["requests"], 1,
        "model must be learned from the response when the request isn't buffered"
    );
    assert!(stats["session"]["input_cost_usd"].as_f64().unwrap() > 0.0);
    assert!(stats["session"]["cache_savings_usd"].as_f64().unwrap() > 0.0);
}

#[tokio::test]
async fn anthropic_sse_stream_records_deferred_usage() {
    let upstream = MockServer::start().await;
    let sse_body = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"usage\":",
        "{\"input_tokens\":50,\"output_tokens\":1,\"cache_read_input_tokens\":12}}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":",
        "{\"type\":\"text_delta\",\"text\":\"hey\"}}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},",
        "\"usage\":{\"output_tokens\":30}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    );
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(sse_body.as_bytes().to_vec(), "text/event-stream"),
        )
        .mount(&upstream)
        .await;
    let proxy = start_proxy_with(&upstream.uri(), |c| c.compression = true).await;

    let resp = post_json(
        format!("{}/v1/messages", proxy.url()),
        &json!({
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 32,
            "stream": true,
            "messages": [{"role": "user", "content": "hello"}]
        }),
    )
    .await;
    assert_eq!(resp.status(), 200);
    let _ = resp.bytes().await.unwrap();

    // Deferred recording: counts come from the stream's usage
    // frames, observed only after the stream closed.
    let stats = wait_for_stats(&proxy.url(), |v| v["requests"]["total"] == 1).await;
    assert_eq!(stats["tokens"]["input"], 50);
    assert_eq!(
        stats["tokens"]["output"], 30,
        "output from message_delta, not 0"
    );
    assert_eq!(stats["tokens"]["cache_read"], 12);
}

#[tokio::test]
async fn failed_upstream_counts_but_accrues_no_savings() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(529).set_body_json(json!({
            "type": "error",
            "error": {"type": "overloaded_error", "message": "overloaded"}
        })))
        .mount(&upstream)
        .await;
    let proxy = start_proxy_with(&upstream.uri(), |c| c.compression = true).await;

    let resp = post_json(
        format!("{}/v1/messages", proxy.url()),
        &json!({
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hello"}]
        }),
    )
    .await;
    assert_eq!(resp.status(), 529);
    let _ = resp.bytes().await.unwrap();

    let stats = wait_for_stats(&proxy.url(), |v| v["requests"]["total"] == 1).await;
    assert_eq!(stats["requests"]["failed"], 1);
    assert_eq!(stats["tokens"]["saved"], 0);
    assert_eq!(stats["session"]["savings_usd"], 0.0);
    assert_eq!(stats["session"]["input_cost_usd"], 0.0);
    // Visible in the feed, flagged.
    let events: Value = reqwest::Client::new()
        .get(format!("{}/stats/events?limit=5", proxy.url()))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(events["events"][0]["failed"], true);
    assert_eq!(events["events"][0]["tokens_saved"], 0);
}

#[tokio::test]
async fn openai_chat_response_normalises_cached_tokens() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"},
                         "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 90,
                "completion_tokens": 12,
                "prompt_tokens_details": {"cached_tokens": 60}
            }
        })))
        .mount(&upstream)
        .await;
    let proxy = start_proxy_with(&upstream.uri(), |c| c.compression = true).await;

    let resp = post_json(
        format!("{}/v1/chat/completions", proxy.url()),
        &json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}]
        }),
    )
    .await;
    assert_eq!(resp.status(), 200);
    let _ = resp.bytes().await.unwrap();

    let stats = wait_for_stats(&proxy.url(), |v| v["requests"]["total"] == 1).await;
    // prompt_tokens INCLUDES cached — the ledger stores the split.
    assert_eq!(stats["tokens"]["input"], 30);
    assert_eq!(stats["tokens"]["cache_read"], 60);
    assert_eq!(stats["tokens"]["output"], 12);
    // Provider label matches the Prometheus vocabulary
    // (`metric_names`): the chat lane is `openai_chat`, not `openai`.
    assert_eq!(stats["session_by_provider"]["openai_chat"]["requests"], 1);
}

#[tokio::test]
async fn bedrock_converse_sync_records_camelcase_usage() {
    let upstream = MockServer::start().await;
    let model_id = "anthropic.claude-3-5-haiku-20241022-v1:0";
    Mock::given(method("POST"))
        .and(path(format!("/model/{model_id}/converse")))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "output": {"message": {"role": "assistant",
                "content": [{"text": "hello"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 9, "outputTokens": 4,
                      "cacheReadInputTokens": 2, "cacheWriteInputTokens": 1}
        })))
        .mount(&upstream)
        .await;

    let endpoint: Url = upstream.uri().parse().unwrap();
    let proxy = start_proxy_with_state(
        &upstream.uri(),
        |c| c.bedrock_endpoint = Some(endpoint),
        |s| s.with_bedrock_credentials(test_credentials()),
    )
    .await;

    let resp = post_json(
        format!("{}/model/{model_id}/converse", proxy.url()),
        &json!({
            "messages": [{"role": "user", "content": [{"text": "hello"}]}]
        }),
    )
    .await;
    assert_eq!(resp.status(), 200);
    let _ = resp.bytes().await.unwrap();

    let stats = wait_for_stats(&proxy.url(), |v| v["requests"]["total"] == 1).await;
    assert_eq!(stats["session_by_provider"]["bedrock"]["requests"], 1);
    assert_eq!(stats["tokens"]["input"], 9);
    assert_eq!(stats["tokens"]["output"], 4);
    assert_eq!(stats["tokens"]["cache_read"], 2);
    assert_eq!(stats["tokens"]["cache_write"], 1);
    // Bedrock model ids price via the vendored table (cache reads
    // discounted vs list input price).
    assert!(stats["session"]["cache_savings_usd"].as_f64().unwrap() > 0.0);
    assert_eq!(stats["lifetime_by_model"][model_id]["requests"], 1);
}

#[tokio::test]
async fn bedrock_invoke_sync_records_snake_case_usage() {
    let upstream = MockServer::start().await;
    let model_id = "anthropic.claude-3-5-haiku-20241022-v1:0";
    Mock::given(method("POST"))
        .and(path(format!("/model/{model_id}/invoke")))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_1", "type": "message", "role": "assistant",
            "model": "claude-3-5-haiku-20241022",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 12, "output_tokens": 4}
        })))
        .mount(&upstream)
        .await;
    let endpoint: Url = upstream.uri().parse().unwrap();
    let proxy = start_proxy_with_state(
        &upstream.uri(),
        |c| c.bedrock_endpoint = Some(endpoint),
        |s| s.with_bedrock_credentials(test_credentials()),
    )
    .await;

    let resp = post_json(
        format!("{}/model/{model_id}/invoke", proxy.url()),
        &json!({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hello"}]
        }),
    )
    .await;
    assert_eq!(resp.status(), 200);
    let _ = resp.bytes().await.unwrap();

    let stats = wait_for_stats(&proxy.url(), |v| v["requests"]["total"] == 1).await;
    assert_eq!(stats["tokens"]["input"], 12);
    assert_eq!(stats["tokens"]["output"], 4);
    assert_eq!(stats["session_by_provider"]["bedrock"]["requests"], 1);
    // The request-side model id (with region/revision) wins over the
    // response echo's bare id.
    assert_eq!(stats["lifetime_by_model"][model_id]["requests"], 1);
}

/// The passthrough lane nothing else parses: binary EventStream
/// (`Accept: application/vnd.amazon.eventstream`), with InvokeModel's
/// base64-wrapped Anthropic events. Usage must come out of the tee's
/// telemetry parser while the client receives byte-equal frames.
#[tokio::test]
async fn bedrock_eventstream_passthrough_records_usage() {
    use base64::Engine as _;
    let b64 = |v: &Value| base64::engine::general_purpose::STANDARD.encode(v.to_string());
    let frame = |inner: &Value| event_frame("chunk", &json!({"bytes": b64(inner)}));
    let mut body_bytes: Vec<u8> = Vec::new();
    for inner in [
        json!({"type": "message_start",
               "message": {"id": "msg_1", "usage": {"input_tokens": 21, "output_tokens": 1,
                            "cache_read_input_tokens": 7}}}),
        json!({"type": "content_block_delta", "index": 0,
               "delta": {"type": "text_delta", "text": "ok"}}),
        json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"},
               "usage": {"output_tokens": 9}}),
        json!({"type": "message_stop"}),
    ] {
        body_bytes.extend_from_slice(&frame(&inner));
    }

    let upstream = MockServer::start().await;
    let model_id = "anthropic.claude-3-5-haiku-20241022-v1:0";
    Mock::given(method("POST"))
        .and(path(format!(
            "/model/{model_id}/invoke-with-response-stream"
        )))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/vnd.amazon.eventstream")
                // Cloned: the assertion below compares the client's
                // bytes against these exact frames.
                .set_body_raw(body_bytes.clone(), "application/vnd.amazon.eventstream"),
        )
        .mount(&upstream)
        .await;
    let endpoint: Url = upstream.uri().parse().unwrap();
    let proxy = start_proxy_with_state(
        &upstream.uri(),
        |c| c.bedrock_endpoint = Some(endpoint),
        |s| s.with_bedrock_credentials(test_credentials()),
    )
    .await;

    let resp = reqwest::Client::new()
        .post(format!(
            "{}/model/{model_id}/invoke-with-response-stream",
            proxy.url()
        ))
        .header("content-type", "application/json")
        // EventStream passthrough mode — the client wants raw frames.
        .header("accept", "application/vnd.amazon.eventstream")
        .json(&json!({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let echoed = resp.bytes().await.unwrap();
    // The whole claim of this lane is that telemetry rides alongside
    // the bytes without touching them. Assert that literally: the
    // client's stream must be byte-identical to what the upstream
    // sent, tee or no tee. `!is_empty()` would pass a mangled body.
    assert_eq!(
        echoed.as_ref(),
        body_bytes.as_slice(),
        "EventStream passthrough must be byte-identical through the tee"
    );

    let stats = wait_for_stats(&proxy.url(), |v| v["requests"]["total"] == 1).await;
    assert_eq!(
        stats["tokens"]["input"], 21,
        "from base64-wrapped message_start"
    );
    assert_eq!(
        stats["tokens"]["output"], 9,
        "monotone max from message_delta"
    );
    assert_eq!(stats["tokens"]["cache_read"], 7);
    assert_eq!(stats["requests"]["cached"], 1);
}

/// Converse-stream in SSE-translation mode: usage arrives ONLY in the
/// camelCase `metadata` frame, which carries no Anthropic `type` and
/// is dropped by the Anthropic state machine — the fallback scan must
/// pick it up.
#[tokio::test]
async fn bedrock_converse_stream_sse_records_camel_metadata_usage() {
    let mut body_bytes: Vec<u8> = Vec::new();
    for (et, payload) in [
        ("messageStart", json!({"role": "assistant"})),
        (
            "contentBlockDelta",
            json!({"contentBlockIndex": 0, "delta": {"text": "ok"}}),
        ),
        ("messageStop", json!({"stopReason": "end_turn"})),
        (
            "metadata",
            json!({"usage": {"inputTokens": 15, "outputTokens": 6,
                              "cacheWriteInputTokens": 11},
                   "metrics": {"latencyMs": 250}}),
        ),
    ] {
        body_bytes.extend_from_slice(&event_frame(et, &payload));
    }

    let upstream = MockServer::start().await;
    let model_id = "anthropic.claude-3-5-haiku-20241022-v1:0";
    Mock::given(method("POST"))
        .and(path(format!("/model/{model_id}/converse-stream")))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/vnd.amazon.eventstream")
                .set_body_raw(body_bytes, "application/vnd.amazon.eventstream"),
        )
        .mount(&upstream)
        .await;
    let endpoint: Url = upstream.uri().parse().unwrap();
    let proxy = start_proxy_with_state(
        &upstream.uri(),
        |c| c.bedrock_endpoint = Some(endpoint),
        |s| s.with_bedrock_credentials(test_credentials()),
    )
    .await;

    let resp = reqwest::Client::new()
        .post(format!("{}/model/{model_id}/converse-stream", proxy.url()))
        .header("content-type", "application/json")
        .header("accept", "text/event-stream") // SSE translation mode
        .json(&json!({
            "messages": [{"role": "user", "content": [{"text": "hello"}]}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let _ = resp.bytes().await.unwrap();

    let stats = wait_for_stats(&proxy.url(), |v| v["requests"]["total"] == 1).await;
    assert_eq!(
        stats["tokens"]["input"], 15,
        "camelCase metadata usage captured"
    );
    assert_eq!(stats["tokens"]["output"], 6);
    assert_eq!(stats["tokens"]["cache_write"], 11);
}

#[tokio::test]
async fn vertex_raw_predict_records_usage() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path_regex(
            r"^/v1beta1/projects/[^/]+/locations/[^/]+/publishers/anthropic/models/.+:rawPredict$",
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_1", "type": "message", "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 33, "output_tokens": 8,
                       "cache_read_input_tokens": 5}
        })))
        .mount(&upstream)
        .await;
    let proxy = start_proxy_with_state(
        &upstream.uri(),
        |_| {},
        |s| install_static_token_source(s, "ya29.stats-test-bearer"),
    )
    .await;

    let resp = reqwest::Client::new()
        .post(format!(
            "{}/v1beta1/projects/p1/locations/us-central1/publishers/anthropic/models/claude-sonnet-4-5@20250929:rawPredict",
            proxy.url()
        ))
        .header("content-type", "application/json")
        .json(&json!({
            "anthropic_version": "vertex-2023-10-16",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let _ = resp.bytes().await.unwrap();

    let stats = wait_for_stats(&proxy.url(), |v| v["requests"]["total"] == 1).await;
    assert_eq!(stats["session_by_provider"]["vertex"]["requests"], 1);
    assert_eq!(stats["tokens"]["input"], 33);
    assert_eq!(stats["tokens"]["output"], 8);
    assert_eq!(stats["tokens"]["cache_read"], 5);
}

/// A proxy-side rejection (here: no AWS credentials configured) is an
/// outcome too — it must show in /stats as a failure, not vanish. An
/// operator-side credentials outage reading as "zero traffic" would
/// hide exactly the incident this dashboard exists to surface.
#[tokio::test]
async fn bedrock_proxy_side_rejection_records_failure() {
    let upstream = MockServer::start().await;
    let model_id = "anthropic.claude-3-5-haiku-20241022-v1:0";
    let endpoint: Url = upstream.uri().parse().unwrap();
    // No .with_bedrock_credentials(...) → handler must 5xx before
    // any upstream attempt.
    let proxy = start_proxy_with(&upstream.uri(), |c| c.bedrock_endpoint = Some(endpoint)).await;

    let resp = post_json(
        format!("{}/model/{model_id}/invoke", proxy.url()),
        &json!({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hello"}]
        }),
    )
    .await;
    assert_eq!(resp.status(), 500);

    let stats = wait_for_stats(&proxy.url(), |v| v["requests"]["total"] == 1).await;
    assert_eq!(stats["requests"]["failed"], 1);
    assert_eq!(stats["tokens"]["input"], 0);
    assert_eq!(stats["session"]["savings_usd"], 0.0);
    assert_eq!(stats["session_by_provider"]["bedrock"]["failed"], 1);
}

/// Persistence through the real config surface: proxy A (started with
/// a stats path) records and flushes; proxy B on the same path serves
/// the lifetime numbers proxy A earned.
#[tokio::test]
async fn stats_path_survives_across_proxy_instances() {
    let dir = std::env::temp_dir().join(format!("stats-restart-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let stats_path = dir.join("native_stats.json");

    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_1", "type": "message",
            "model": "claude-sonnet-4-5-20250929",
            "content": [{"type": "text", "text": "hi"}],
            "usage": {"input_tokens": 60, "output_tokens": 6}
        })))
        .mount(&upstream)
        .await;

    // Proxy A: record one request, then flush via the ledger handle
    // (the background flusher + shutdown flush live in main.rs, which
    // this in-process harness deliberately doesn't run).
    let ledger_slot: std::sync::Arc<
        std::sync::Mutex<Option<std::sync::Arc<headroom_proxy::observability::ledger::Ledger>>>,
    > = Default::default();
    let slot = ledger_slot.clone();
    let sp = stats_path.clone();
    let proxy_a = start_proxy_with_state(
        &upstream.uri(),
        move |c| c.stats_path = Some(sp),
        move |s| {
            *slot.lock().unwrap() = Some(s.stats.clone());
            s
        },
    )
    .await;
    let resp = post_json(
        format!("{}/v1/messages", proxy_a.url()),
        &json!({
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hello"}]
        }),
    )
    .await;
    assert_eq!(resp.status(), 200);
    let _ = resp.bytes().await.unwrap();
    wait_for_stats(&proxy_a.url(), |v| v["requests"]["total"] == 1).await;
    let ledger = ledger_slot.lock().unwrap().take().unwrap();
    ledger.flush().await;
    proxy_a.shutdown().await;
    assert!(stats_path.exists(), "flush persisted the ledger");

    // Proxy B on the same path: lifetime carries over, session fresh.
    let sp = stats_path.clone();
    let proxy_b = start_proxy_with(&upstream.uri(), move |c| c.stats_path = Some(sp)).await;
    let stats: Value = reqwest::Client::new()
        .get(format!("{}/stats", proxy_b.url()))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        stats["lifetime"]["requests"], 1,
        "lifetime survived restart"
    );
    assert_eq!(stats["lifetime"]["input_tokens"], 60);
    assert_eq!(stats["session"]["requests"], 0, "session resets per boot");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A 200 OK SSE stream that fails IN-BAND (`{"type":"error"}` mid-
/// stream — e.g. Anthropic overloading during generation) is not a
/// billable success. The state machine lands in `Errored`; the
/// ledger must count the request as failed and accrue nothing.
#[tokio::test]
async fn in_stream_error_event_records_failure() {
    let sse_body = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"usage\":",
        "{\"input_tokens\":50,\"output_tokens\":1}}}\n\n",
        "event: error\n",
        "data: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",",
        "\"message\":\"Overloaded\"}}\n\n",
    );
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(sse_body.as_bytes().to_vec(), "text/event-stream"),
        )
        .mount(&upstream)
        .await;
    let proxy = start_proxy_with(&upstream.uri(), |_| {}).await;

    let resp = post_json(
        format!("{}/v1/messages", proxy.url()),
        &json!({
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 32,
            "stream": true,
            "messages": [{"role": "user", "content": "hello"}]
        }),
    )
    .await;
    assert_eq!(resp.status(), 200, "the HTTP status was a success");
    let _ = resp.bytes().await.unwrap();

    let stats = wait_for_stats(&proxy.url(), |v| v["requests"]["total"] == 1).await;
    assert_eq!(
        stats["requests"]["failed"], 1,
        "in-band stream error must count as a failed request"
    );
    assert_eq!(
        stats["tokens"]["input"], 0,
        "failed requests accrue nothing"
    );
    assert_eq!(stats["session"]["input_cost_usd"], 0.0);
}

/// A hostile (or buggy) upstream reporting absurd token counts must
/// not poison the lifetime aggregates: the ledger clamps every
/// per-event token field to its 50M ceiling before pricing.
#[tokio::test]
async fn hostile_usage_counts_are_clamped_before_recording() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_1",
            "type": "message",
            "model": "claude-sonnet-4-5-20250929",
            "content": [{"type": "text", "text": "hi"}],
            "usage": {
                "input_tokens": u64::MAX,
                "output_tokens": 9_000_000_000_000u64,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0
            }
        })))
        .mount(&upstream)
        .await;
    let proxy = start_proxy_with(&upstream.uri(), |_| {}).await;

    let resp = post_json(
        format!("{}/v1/messages", proxy.url()),
        &json!({
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hello"}]
        }),
    )
    .await;
    assert_eq!(resp.status(), 200);
    let _ = resp.bytes().await.unwrap();

    let stats = wait_for_stats(&proxy.url(), |v| v["requests"]["total"] == 1).await;
    const CEILING: u64 = 50_000_000;
    assert_eq!(stats["tokens"]["input"], CEILING, "input clamped");
    assert_eq!(stats["tokens"]["output"], CEILING, "output clamped");
    let cost = stats["session"]["input_cost_usd"].as_f64().unwrap();
    assert!(
        cost.is_finite() && cost < 1_000.0,
        "clamped pricing stays sane: {cost}"
    );
}
