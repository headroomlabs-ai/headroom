//! POST `/model/{model_id}/invoke` handler — Phase D PR-D1.
//!
//! # Pipeline
//!
//! 1. Extract `{model_id}` from the path. The Bedrock convention is
//!    `anthropic.claude-3-haiku-20240307-v1:0` — dot-separated
//!    `<vendor>.<model>-<date>-<rev>`.
//! 2. If the vendor is `anthropic`, parse the body as a Bedrock
//!    envelope (`{"anthropic_version": "...", ...rest}`) and run the
//!    live-zone Anthropic compression dispatcher over the body bytes.
//!    The dispatcher is the SAME one `/v1/messages` uses; Bedrock's
//!    body shape is just Anthropic-without-the-`model`-field.
//! 3. Re-emit the (possibly compressed) body with `anthropic_version`
//!    preserved as the first key.
//! 4. Build the upstream URL (`https://bedrock-runtime.{region}.amazonaws.com/model/{model}/invoke`
//!    or operator override).
//! 5. Sign the (post-compression) body bytes with AWS SigV4. Sign
//!    over `host`, `x-amz-date`, `x-amz-content-sha256` plus any
//!    extra headers (`content-type`, `accept`).
//! 6. Forward to Bedrock; stream the response back to the client.
//!
//! # Failure modes
//!
//! - **Missing credentials**: log
//!   `event=bedrock_credentials_missing` at WARN, return `500` with
//!   a JSON error body. NEVER forwards an unsigned request.
//! - **Envelope parse failure**: log `event=bedrock_envelope_parse_error`,
//!   pass the bytes through unchanged. Bedrock will reject anyway,
//!   but the failure is the customer's, not ours — we just route.
//!   This matches the Anthropic compression path's
//!   `Outcome::Passthrough` behaviour.
//! - **Non-anthropic model**: skip compression, but still sign +
//!   forward. Other vendors (Amazon Titan, Cohere, AI21, Meta) have
//!   different body shapes that the proxy doesn't yet understand;
//!   we pass them through opaquely.
//! - **SigV4 signing failure**: log
//!   `event=bedrock_sigv4_failed`, return `500`. NEVER forwards
//!   unsigned.

use std::net::SocketAddr;
use std::time::{Instant, SystemTime};

use axum::body::Body;
use axum::extract::{ConnectInfo, Extension, Path, State};
use axum::http::{HeaderMap, Method, StatusCode, Uri};
use axum::response::Response;
use bytes::Bytes;
use http::HeaderName;
use url::Url;

use crate::bedrock::envelope::BedrockEnvelope;
use crate::bedrock::sigv4::{sign_request, SigningInputs};
use crate::compression::{
    compress_anthropic_request, Outcome as AnthropicOutcome, PassthroughReason,
};
use crate::headers::filter_response_headers;
use crate::observability::{observe_bedrock_invoke_latency, record_bedrock_invoke};
use crate::proxy::AppState;
// Phase F PR-F1 + PR-D3: the bedrock auth-mode layer
// (`classify_and_attach_auth_mode`) populates `request.extensions()`
// with `AuthMode` BEFORE this handler runs. We extract it via
// `Extension<AuthMode>` so the middleware-supplied value is the
// single source of truth — handler does NOT re-classify; that
// would risk drift from the middleware's resolution + WARN log.
use headroom_core::auth_mode::AuthMode;

use crate::bedrock::vendor::is_anthropic_model_id;

/// RAII guard that observes the `bedrock_invoke_latency_seconds`
/// histogram on drop. Created at handler entry; observed when the
/// guard goes out of scope no matter how the handler exits. Owning
/// `String` rather than `&str` for the labels avoids capture-order
/// dramas with the borrow checker on early-return paths.
struct LatencyGuard {
    model: String,
    region: String,
    start: Instant,
}

impl LatencyGuard {
    fn start(model: &str, region: &str) -> Self {
        Self {
            model: model.to_string(),
            region: region.to_string(),
            start: Instant::now(),
        }
    }
}

impl Drop for LatencyGuard {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed().as_secs_f64();
        observe_bedrock_invoke_latency(&self.model, &self.region, elapsed);
    }
}

/// AWS Bedrock Runtime DNS template. The `{}` placeholder is the
/// region. Only used when `Config::bedrock_endpoint` is `None`.
const BEDROCK_RUNTIME_HOST_TEMPLATE: &str = "bedrock-runtime.{region}.amazonaws.com";

/// Bedrock non-streaming action path segments. `invoke` is the legacy
/// InvokeModel surface; `converse` is the unified Converse surface.
/// Both mount the same handler (see `proxy.rs`), so the action is
/// resolved from the inbound path — otherwise `/converse` requests
/// would be forwarded to the upstream `/invoke` endpoint.
const INVOKE_ACTION: &str = "invoke";
const CONVERSE_ACTION: &str = "converse";

/// Resolve the Bedrock action from the inbound request path. Mirrors
/// `invoke_streaming::extract_streaming_action` for the non-streaming
/// surfaces (`/invoke`, `/converse`).
fn extract_invoke_action(path: &str) -> Option<&'static str> {
    if path.ends_with(&format!("/{INVOKE_ACTION}")) {
        Some(INVOKE_ACTION)
    } else if path.ends_with(&format!("/{CONVERSE_ACTION}")) {
        Some(CONVERSE_ACTION)
    } else {
        None
    }
}

/// Axum POST handler for `/model/{model_id}/invoke`.
///
/// Buffers the body so the live-zone compressor + SigV4 signer can
/// inspect it. Both are required to be applied to the SAME byte slice
/// — the signer hashes whatever the forwarder will actually send.
#[allow(clippy::too_many_arguments)] // axum extractors demand one argument per role
pub async fn handle_invoke(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    Extension(auth_mode): Extension<AuthMode>,
    Path(model_id): Path<String>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let _ = client_addr; // accepted for ConnectInfo extractor; not used directly today
    let request_id = headers
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    // PR-D3: latency stopwatch starts at handler entry (after
    // routing + middleware). The histogram observes wall-clock
    // time, so it captures upstream RTT + sign + compress as a
    // single number — operators can split contributions via the
    // `bedrock_*` structured-log timing fields if a slow path
    // shows up. Wrapped in a `LatencyGuard` so EVERY return path
    // (success, sign-failure, upstream-error, response-build error)
    // observes the histogram. RAII keeps the call site to one
    // line and rules out future regressions where someone adds a
    // new error path and forgets to instrument.
    let region = state.config.bedrock_region.clone();
    let _latency_guard = LatencyGuard::start(&model_id, &region);
    // Wall-clock for the savings-store recent-request latency (the guard's clock
    // is private and observes a Prometheus histogram on drop).
    let rec_start = Instant::now();

    // PR-D3: count every invoke at handler entry (one per request,
    // before any error path can early-return). Pairs with the
    // structured log emitted below so operators can join the
    // counter with the trace by `request_id`.
    record_bedrock_invoke(&model_id, &region, auth_mode);

    tracing::info!(
        event = "bedrock_invoke_received",
        request_id = %request_id,
        method = %method,
        model_id = %model_id,
        region = %region,
        auth_mode = auth_mode.as_str(),
        body_bytes = body.len(),
        "bedrock invoke route received request"
    );

    let is_anthropic = is_anthropic_model_id(&model_id);
    let (outbound_body, rec_tokens_before, rec_tokens_after): (Bytes, u64, u64) = if is_anthropic {
        run_anthropic_compression(&body, &state, auth_mode, &request_id)
    } else {
        tracing::info!(
            event = "bedrock_compression_skipped",
            request_id = %request_id,
            model_id = %model_id,
            reason = "non_anthropic_vendor",
            "bedrock invoke: skipping live-zone compression for non-anthropic vendor"
        );
        (body.clone(), 0, 0)
    };

    // Phase H: record every Bedrock request into the savings store so `/stats`
    // and the dashboard cover this backend too — unified with the Anthropic /
    // OpenAI lanes that flow through `forward_http`. The request-side outcome is
    // built here (compression token counts are known now) but recorded only once
    // the upstream result is known, so `requests.failed` reflects connect errors
    // / non-2xx upstreams instead of counting every attempt as a success.
    //
    // Recording boundary (same as forward_http): only requests that *attempt* the
    // upstream are recorded. Pre-upstream rejections below — missing credentials,
    // SigV4 failure, an invalid endpoint or action path — return before the
    // record site and are intentionally NOT counted; they are config/client
    // errors, not upstream interactions.
    //
    // Recording is gated on `should_record()` (compression on AND mode != off),
    // the same predicate every lane uses — a request that can't actually compress
    // is not a savings event. Note this is narrower than the interception gate:
    // `forward_http`'s `should_intercept` is the plain `config.compression` flag
    // (it still buffers/forwards a mode=off request, just doesn't record it).
    let rec_outcome = state.config.should_record().then(|| {
        crate::observability::stats::RequestOutcome::priced(
            "bedrock",
            model_id.clone(),
            rec_tokens_before,
            rec_tokens_after,
            request_id.clone(),
            &state.price_book,
        )
    });

    // Resolve the Bedrock action from the inbound path so `/converse`
    // forwards to the upstream Converse endpoint instead of `/invoke`.
    // Both paths mount this handler (see `proxy.rs`); the streaming
    // sibling resolves its action the same way.
    let action = match extract_invoke_action(uri.path()) {
        Some(a) => a,
        None => {
            tracing::error!(
                event = "bedrock_invoke_action_invalid",
                request_id = %request_id,
                path = %uri.path(),
                "bedrock invoke: unrecognized action path"
            );
            return super::error_response(
                StatusCode::BAD_REQUEST,
                "bedrock_invoke_action_invalid",
                "Unsupported Bedrock action path",
            );
        }
    };

    // Build the upstream URL based on configured endpoint or
    // region-derived default.
    let upstream_url = match build_bedrock_upstream(&state, &model_id, &uri, action) {
        Ok(u) => u,
        Err(msg) => {
            tracing::error!(
                event = "bedrock_endpoint_invalid",
                request_id = %request_id,
                error = %msg,
                "bedrock invoke: failed to construct upstream URL"
            );
            return super::error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "bedrock_endpoint_invalid",
                &msg,
            );
        }
    };

    // Resolve credentials. No silent fallback: missing creds → 5xx.
    let creds = match state.bedrock_credentials.as_ref() {
        Some(c) => c.clone(),
        None => {
            tracing::warn!(
                event = "bedrock_credentials_missing",
                request_id = %request_id,
                model_id = %model_id,
                "bedrock invoke: refusing to forward without AWS credentials"
            );
            return super::error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "bedrock_credentials_missing",
                "AWS credentials not configured; refusing to forward unsigned",
            );
        }
    };

    // Build the headers we sign + forward. Start from the inbound
    // headers, drop the ones the upstream client manages, then sign.
    let extra_signed: Vec<(String, String)> =
        super::collect_signed_headers(&headers, &upstream_url);
    let extra_signed_refs: Vec<(&str, &str)> = extra_signed
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();

    let sign_inputs = SigningInputs {
        method: method.as_str(),
        url: &upstream_url,
        region: &state.config.bedrock_region,
        credentials: creds.as_ref(),
        body: &outbound_body,
        extra_signed_headers: &extra_signed_refs,
        time: SystemTime::now(),
    };
    let signed = match sign_request(&sign_inputs) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!(
                event = "bedrock_sigv4_failed",
                request_id = %request_id,
                model_id = %model_id,
                error = %e,
                "bedrock invoke: SigV4 signing failed; refusing to forward"
            );
            return super::error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "bedrock_sigv4_failed",
                &e.to_string(),
            );
        }
    };

    // Compose the outgoing header map. Start with the headers we'll
    // forward (filter out hop-by-hop / Host / Content-Length;
    // reqwest sets those itself), then layer the SigV4 outputs on
    // top — they replace any pre-existing copies of the same name.
    let mut outbound_headers = HeaderMap::new();
    for (name, value) in extra_signed.iter() {
        if let (Ok(n), Ok(v)) = (
            HeaderName::from_bytes(name.as_bytes()),
            http::HeaderValue::from_str(value),
        ) {
            outbound_headers.insert(n, v);
        }
    }
    for (name, value) in signed.entries.iter() {
        if let (Ok(n), Ok(v)) = (
            HeaderName::from_bytes(name.as_bytes()),
            http::HeaderValue::from_str(value),
        ) {
            outbound_headers.insert(n, v);
        }
    }

    // Forward. We surface upstream errors as 502; the byte path
    // streams the response back to the client.
    let reqwest_method = match reqwest::Method::from_bytes(method.as_str().as_bytes()) {
        Ok(m) => m,
        Err(e) => {
            tracing::error!(
                event = "bedrock_invalid_method",
                request_id = %request_id,
                error = %e,
                "bedrock invoke: invalid HTTP method"
            );
            return super::error_response(
                StatusCode::BAD_REQUEST,
                "bedrock_invalid_method",
                &e.to_string(),
            );
        }
    };

    let upstream_resp = state
        .client
        .request(reqwest_method, upstream_url.clone())
        .headers(outbound_headers)
        .body(outbound_body.clone())
        .send()
        .await;

    let upstream_resp = match upstream_resp {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(
                event = "bedrock_upstream_error",
                request_id = %request_id,
                error = %e,
                "bedrock invoke: upstream request failed"
            );
            // Connect/timeout failure — record as a failed request.
            if let Some(o) = rec_outcome {
                state.savings.record_finalized(o, true, rec_start);
            }
            let status = if e.is_timeout() {
                StatusCode::GATEWAY_TIMEOUT
            } else {
                StatusCode::BAD_GATEWAY
            };
            return super::error_response(status, "bedrock_upstream_error", &e.to_string());
        }
    };

    let status =
        StatusCode::from_u16(upstream_resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let resp_headers = filter_response_headers(upstream_resp.headers());

    // Buffer the response body to extract token usage before recording.
    // InvokeModel responses are complete JSON objects (not streams), so
    // buffering doesn't hurt TTFB. The streaming counterpart
    // (`invoke-with-response-stream`) is handled separately.
    let resp_bytes = match upstream_resp.bytes().await {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!(
                event = "bedrock_invoke_body_read_failed",
                request_id = %request_id,
                error = %e,
                "bedrock invoke: failed to read upstream response body"
            );
            if let Some(o) = rec_outcome {
                state.savings.record_finalized(o, true, rec_start);
            }
            return super::error_response(
                StatusCode::BAD_GATEWAY,
                "upstream_body_read_failed",
                &e.to_string(),
            );
        }
    };

    // Extract usage from the Bedrock/Anthropic response body and record it.
    if let Some(mut o) = rec_outcome {
        if status.is_success() {
            apply_bedrock_response_usage(&resp_bytes, &mut o);
        }
        state
            .savings
            .record_finalized(o, !status.is_success(), rec_start);
    }

    tracing::info!(
        event = "bedrock_invoke_forwarded",
        request_id = %request_id,
        model_id = %model_id,
        upstream_status = status.as_u16(),
        upstream_url = %upstream_url,
        "bedrock invoke: response forwarded"
    );

    let body_out = Body::from(resp_bytes);

    let mut builder = Response::builder().status(status);
    if let Some(h) = builder.headers_mut() {
        h.extend(resp_headers);
        if let Ok(v) = http::HeaderValue::from_str(&request_id) {
            h.insert(HeaderName::from_static("x-request-id"), v);
        }
    }
    builder.body(body_out).unwrap_or_else(|e| {
        tracing::error!(
            event = "bedrock_response_build_failed",
            request_id = %request_id,
            error = %e,
            "bedrock invoke: failed to build response"
        );
        Response::builder()
            .status(StatusCode::INTERNAL_SERVER_ERROR)
            .body(Body::from("internal handler error"))
            .expect("static response")
    })
}

/// Run the live-zone Anthropic compressor over a Bedrock-shape body.
///
/// The compressor only inspects `messages` — it doesn't care that the
/// Bedrock body has `anthropic_version` instead of `model`. The
/// `Outcome::Compressed` body bytes still preserve key order via
/// `serde_json`'s `preserve_order` feature, so the caller's
/// re-emission step (`ensure_anthropic_version_first`) almost always
/// no-ops. We still call it as a defence-in-depth assertion that
/// the byte order is correct before signing.
/// Run the live-zone compressor and return the outbound body plus the
/// pre/post-compression input token counts (`(0, 0)` when nothing compressed),
/// so the caller can attribute Bedrock savings into the stats store.
fn run_anthropic_compression(
    body: &Bytes,
    state: &AppState,
    _auth_mode: AuthMode,
    request_id: &str,
) -> (Bytes, u64, u64) {
    // Detect envelope shape. A parseable InvokeModel envelope takes the
    // re-emit path below (anthropic_version pinned first); a non-envelope
    // body (e.g. a Converse-shaped payload) still runs through the
    // compressor but skips envelope re-emit. The body is NOT guaranteed
    // unchanged on parse failure — we log which path we took.
    let parsed_envelope = BedrockEnvelope::parse(body).is_ok();
    if parsed_envelope {
        tracing::info!(
            event = "bedrock_envelope_parsed",
            request_id = %request_id,
            body_bytes = body.len(),
            "bedrock invoke: envelope validated; dispatching to live-zone compressor"
        );
    } else {
        tracing::info!(
            event = "bedrock_envelope_parse_skipped",
            request_id = %request_id,
            "bedrock invoke: envelope parse skipped; attempting generic anthropic compression"
        );
    }

    // PR-E3: Bedrock uses IAM-signed AWS SigV4 downstream. Inbound
    // requests to the proxy may or may not carry their own auth, but
    // Bedrock itself is a subscription/IAM channel — never PAYG —
    // so we hard-code `RequestAuthMode::OAuth` to skip E3
    // cache_control auto-placement. This keeps the Bedrock byte
    // contract stable; live-zone compression continues to run.
    let outcome = compress_anthropic_request(
        body,
        state.config.compression_mode,
        state.config.cache_control_auto_frozen,
        headroom_core::auth_mode::AuthMode::OAuth,
        request_id,
    );
    let (tokens_before, tokens_after) = match &outcome {
        AnthropicOutcome::Compressed {
            tokens_before,
            tokens_after,
            ..
        } => (*tokens_before as u64, *tokens_after as u64),
        _ => (0, 0),
    };
    let out_body = match outcome {
        AnthropicOutcome::NoCompression => body.clone(),
        AnthropicOutcome::Passthrough { reason } => {
            tracing::info!(
                event = "bedrock_compression_passthrough",
                request_id = %request_id,
                reason = ?reason,
                "bedrock invoke: live-zone dispatcher fell through to passthrough"
            );
            // The compressor's passthrough variants all leave bytes
            // unchanged. Forward the original.
            let _ = (PassthroughReason::ModeOff, PassthroughReason::NoMessages); // pin types
            body.clone()
        }
        AnthropicOutcome::Compressed { body: new_body, .. } => {
            if parsed_envelope {
                // Defence-in-depth: re-emit so anthropic_version is the
                // first key. With preserve_order this is a no-op on the
                // happy path.
                match BedrockEnvelope::ensure_anthropic_version_first(&new_body) {
                    Ok(b) => b,
                    Err(e) => {
                        tracing::error!(
                            event = "bedrock_envelope_reemit_failed",
                            request_id = %request_id,
                            error = %e,
                            "bedrock invoke: failed to re-emit envelope; falling back to original body"
                        );
                        body.clone()
                    }
                }
            } else {
                new_body
            }
        }
    };
    (out_body, tokens_before, tokens_after)
}

/// Build the upstream URL for the Bedrock route. Honours the
/// operator-supplied `bedrock_endpoint` first, falling back to the
/// region-derived default. The path/query portion is taken from the
/// original URI verbatim — Bedrock's path schema (`/model/{id}/{action}`)
/// is identical to the proxy's external path.
fn build_bedrock_upstream(
    state: &AppState,
    model_id: &str,
    uri: &Uri,
    action: &str,
) -> Result<Url, String> {
    let base = match state.config.bedrock_endpoint.as_ref() {
        Some(u) => u.clone(),
        None => {
            let host =
                BEDROCK_RUNTIME_HOST_TEMPLATE.replace("{region}", &state.config.bedrock_region);
            Url::parse(&format!("https://{host}/"))
                .map_err(|e| format!("bedrock derived base URL parse error: {e}"))?
        }
    };
    // Compose the path. We trust the captured `model_id` (Axum
    // already URL-decoded it) and append `/{action}`.
    let path = format!(
        "/model/{model_id}/{action}",
        model_id = model_id,
        action = action,
    );
    let mut joined = base;
    joined.set_path(&path);
    if let Some(q) = uri.query() {
        joined.set_query(Some(q));
    }
    Ok(joined)
}

/// Build the list of headers to sign + forward. Drops hop-by-hop,
/// Populate `outcome` with token counts from a Bedrock response body.
/// Silently no-ops on parse failures or absent fields.
///
/// Handles two Bedrock response shapes:
/// - InvokeModel (Anthropic): `input_tokens`, `output_tokens`, `cache_read_input_tokens`
/// - Converse: `inputTokens`, `outputTokens`, `cacheReadInputTokens`
///
/// When compression didn't run (`tokens_before == 0`), the billed input count
/// fills both slots so savings_percent stays 0 instead of dividing by zero.
fn apply_bedrock_response_usage(
    resp_bytes: &[u8],
    outcome: &mut crate::observability::stats::RequestOutcome,
) {
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(resp_bytes) else {
        return;
    };
    let Some(usage) = v.get("usage") else { return };
    let get = |snake: &str, camel: &str| {
        usage
            .get(snake)
            .or_else(|| usage.get(camel))
            .and_then(|t| t.as_u64())
            .unwrap_or(0)
    };
    let input_tokens = get("input_tokens", "inputTokens");
    if outcome.tokens_before == 0 {
        outcome.tokens_before = input_tokens;
        outcome.tokens_after = input_tokens;
    }
    outcome.output_tokens = get("output_tokens", "outputTokens");
    outcome.cache_read_tokens = get("cache_read_input_tokens", "cacheReadInputTokens");
    outcome.cache_write_tokens = get("cache_creation_input_tokens", "cacheWriteInputTokens");
}

#[cfg(test)]
mod tests {
    use super::*;

    // Vendor/model-id classification is tested in `bedrock::vendor`.

    #[test]
    fn extract_invoke_action_supports_both_bedrock_paths() {
        assert_eq!(
            extract_invoke_action("/model/anthropic.claude-3-haiku-20240307-v1:0/invoke"),
            Some(INVOKE_ACTION)
        );
        assert_eq!(
            extract_invoke_action("/model/anthropic.claude-3-haiku-20240307-v1:0/converse"),
            Some(CONVERSE_ACTION)
        );
        // Streaming actions are handled by `invoke_streaming`, not here.
        assert_eq!(
            extract_invoke_action(
                "/model/anthropic.claude-3-haiku-20240307-v1:0/invoke-with-response-stream"
            ),
            None
        );
        assert_eq!(extract_invoke_action("/model/foo/unknown"), None);
    }

    #[test]
    fn build_upstream_routes_converse_to_converse_endpoint() {
        use crate::config::Config;
        let mut config = Config::for_test(Url::parse("http://up:8080").unwrap());
        config.bedrock_region = "us-west-2".to_string();
        let state = AppState {
            config: std::sync::Arc::new(config),
            client: reqwest::Client::new(),
            bedrock_credentials: None,
            drift_state: crate::cache_stabilization::drift_detector::DriftState::new(8),
            vertex_token_source: std::sync::Arc::new(crate::vertex::StaticTokenSource::new(
                "test".to_string(),
            )),
            savings: std::sync::Arc::new(crate::observability::stats::SavingsStore::in_memory()),
            price_book: std::sync::Arc::new(crate::observability::pricing::PriceBook::empty()),
        };
        let uri: Uri = "/model/anthropic.claude-3-haiku-20240307-v1:0/converse"
            .parse()
            .unwrap();
        let action = extract_invoke_action(uri.path()).unwrap();
        let url = build_bedrock_upstream(
            &state,
            "anthropic.claude-3-haiku-20240307-v1:0",
            &uri,
            action,
        )
        .unwrap();
        assert_eq!(
            url.as_str(),
            "https://bedrock-runtime.us-west-2.amazonaws.com/model/anthropic.claude-3-haiku-20240307-v1:0/converse"
        );
    }

    #[test]
    fn build_upstream_uses_region_default() {
        use crate::config::Config;
        let mut config = Config::for_test(Url::parse("http://up:8080").unwrap());
        config.bedrock_region = "us-west-2".to_string();
        let state = AppState {
            config: std::sync::Arc::new(config),
            client: reqwest::Client::new(),
            bedrock_credentials: None,
            // PR-E6: small capacity is fine — the Bedrock URL builder
            // unit test never observes drift, but `AppState` requires
            // the field to be populated.
            drift_state: crate::cache_stabilization::drift_detector::DriftState::new(8),
            // PR-D4: unit tests for the Bedrock URL builder don't
            // touch the Vertex route, but `AppState` is one struct
            // — supply a dummy token source so the test compiles.
            vertex_token_source: std::sync::Arc::new(crate::vertex::StaticTokenSource::new(
                "test".to_string(),
            )),
            savings: std::sync::Arc::new(crate::observability::stats::SavingsStore::in_memory()),
            price_book: std::sync::Arc::new(crate::observability::pricing::PriceBook::empty()),
        };
        let uri: Uri = "/model/anthropic.claude-3-haiku-20240307-v1:0/invoke"
            .parse()
            .unwrap();
        let url = build_bedrock_upstream(
            &state,
            "anthropic.claude-3-haiku-20240307-v1:0",
            &uri,
            "invoke",
        )
        .unwrap();
        assert_eq!(
            url.as_str(),
            "https://bedrock-runtime.us-west-2.amazonaws.com/model/anthropic.claude-3-haiku-20240307-v1:0/invoke"
        );
    }

    #[test]
    fn build_upstream_honors_explicit_endpoint() {
        use crate::config::Config;
        let mut config = Config::for_test(Url::parse("http://up:8080").unwrap());
        config.bedrock_endpoint = Some(Url::parse("http://127.0.0.1:9999").unwrap());
        let state = AppState {
            config: std::sync::Arc::new(config),
            client: reqwest::Client::new(),
            bedrock_credentials: None,
            // PR-E6: see above — drift detector is unused by this
            // test; we just satisfy the struct shape.
            drift_state: crate::cache_stabilization::drift_detector::DriftState::new(8),
            // PR-D4: unit tests for the Bedrock URL builder don't
            // touch the Vertex route, but `AppState` is one struct
            // — supply a dummy token source so the test compiles.
            vertex_token_source: std::sync::Arc::new(crate::vertex::StaticTokenSource::new(
                "test".to_string(),
            )),
            savings: std::sync::Arc::new(crate::observability::stats::SavingsStore::in_memory()),
            price_book: std::sync::Arc::new(crate::observability::pricing::PriceBook::empty()),
        };
        let uri: Uri = "/model/anthropic.claude-3-haiku-20240307-v1:0/invoke"
            .parse()
            .unwrap();
        let url = build_bedrock_upstream(
            &state,
            "anthropic.claude-3-haiku-20240307-v1:0",
            &uri,
            "invoke",
        )
        .unwrap();
        assert_eq!(
            url.as_str(),
            "http://127.0.0.1:9999/model/anthropic.claude-3-haiku-20240307-v1:0/invoke"
        );
    }

    #[test]
    fn collect_signed_headers_strips_client_managed() {
        let mut headers = HeaderMap::new();
        headers.insert("content-type", "application/json".parse().unwrap());
        headers.insert("host", "proxy.example".parse().unwrap());
        headers.insert("authorization", "Bearer x".parse().unwrap());
        headers.insert("x-headroom-mode", "live".parse().unwrap());
        headers.insert("accept", "application/json".parse().unwrap());
        let upstream =
            Url::parse("https://bedrock-runtime.us-east-1.amazonaws.com/model/x/invoke").unwrap();
        let out = super::super::collect_signed_headers(&headers, &upstream);
        let names: Vec<&str> = out.iter().map(|(k, _)| k.as_str()).collect();
        assert!(names.contains(&"content-type"));
        assert!(names.contains(&"accept"));
        assert!(names.contains(&"host"));
        assert!(!names.contains(&"authorization"));
        assert!(!names.contains(&"x-headroom-mode"));
        // host must be the upstream host, not the proxy host.
        let host = out
            .iter()
            .find(|(k, _)| k == "host")
            .map(|(_, v)| v.as_str())
            .unwrap();
        assert_eq!(host, "bedrock-runtime.us-east-1.amazonaws.com");
    }

    #[test]
    fn apply_bedrock_response_usage_extracts_all_fields() {
        use crate::observability::pricing::PriceBook;
        use crate::observability::stats::RequestOutcome;
        let mut o = RequestOutcome::priced(
            "bedrock",
            "anthropic.claude-haiku",
            0,
            0,
            "r",
            &PriceBook::empty(),
        );
        let resp = br#"{"id":"msg_01","type":"message","content":[],"usage":{"input_tokens":215,"output_tokens":100,"cache_read_input_tokens":50,"cache_creation_input_tokens":0}}"#;
        apply_bedrock_response_usage(resp, &mut o);
        assert_eq!(o.tokens_before, 215);
        assert_eq!(o.tokens_after, 215); // no compression → before == after
        assert_eq!(o.output_tokens, 100);
        assert_eq!(o.cache_read_tokens, 50);
    }

    #[test]
    fn apply_bedrock_response_usage_preserves_compression_tokens() {
        use crate::observability::pricing::PriceBook;
        use crate::observability::stats::RequestOutcome;
        // When compression ran, tokens_before > 0 — must not be overwritten.
        let mut o = RequestOutcome::priced(
            "bedrock",
            "anthropic.claude-haiku",
            1000,
            600,
            "r",
            &PriceBook::empty(),
        );
        let resp =
            br#"{"usage":{"input_tokens":600,"output_tokens":50,"cache_read_input_tokens":0}}"#;
        apply_bedrock_response_usage(resp, &mut o);
        assert_eq!(o.tokens_before, 1000); // preserved
        assert_eq!(o.tokens_after, 600); // preserved
        assert_eq!(o.output_tokens, 50);
    }

    #[test]
    fn apply_bedrock_response_usage_handles_converse_camelcase() {
        use crate::observability::pricing::PriceBook;
        use crate::observability::stats::RequestOutcome;
        let mut o = RequestOutcome::priced(
            "bedrock",
            "anthropic.claude-haiku",
            0,
            0,
            "r",
            &PriceBook::empty(),
        );
        // Converse API uses camelCase keys
        let resp = br#"{"output":{"message":{"role":"assistant","content":[{"text":"OK"}]}},"stopReason":"end_turn","usage":{"inputTokens":15,"outputTokens":4,"totalTokens":19,"cacheReadInputTokens":0,"cacheWriteInputTokens":0}}"#;
        apply_bedrock_response_usage(resp, &mut o);
        assert_eq!(o.tokens_before, 15);
        assert_eq!(o.tokens_after, 15);
        assert_eq!(o.output_tokens, 4);
        assert_eq!(o.cache_read_tokens, 0);
    }

    #[test]
    fn apply_bedrock_response_usage_noop_on_non_json() {
        use crate::observability::pricing::PriceBook;
        use crate::observability::stats::RequestOutcome;
        let mut o = RequestOutcome::priced(
            "bedrock",
            "anthropic.claude-haiku",
            0,
            0,
            "r",
            &PriceBook::empty(),
        );
        apply_bedrock_response_usage(b"not json", &mut o);
        assert_eq!(o.tokens_before, 0);
        assert_eq!(o.output_tokens, 0);
    }
}
