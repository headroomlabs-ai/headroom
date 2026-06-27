//! Core reverse-proxy router and HTTP forwarding handler.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use axum::body::{to_bytes, Body};
use axum::extract::{ConnectInfo, DefaultBodyLimit, State, WebSocketUpgrade};
use axum::http::{HeaderMap, HeaderName, Request, Response, StatusCode, Uri};
use axum::response::IntoResponse;
use axum::routing::{any, get, post};
use axum::Router;
#[cfg(test)]
use bytes::Bytes;
use futures_util::{StreamExt as _, TryStreamExt};
#[cfg(test)]
use http_body_util::BodyExt;

use crate::cache_stabilization;
use crate::cache_stabilization::drift_detector::{
    compute_structural_hash, derive_session_key, observe_drift, ApiKind, DriftState,
};
use crate::compression;
use crate::config::Config;
use crate::error::ProxyError;
use crate::headers::{build_forward_request_headers, filter_response_headers};
use crate::health::{healthz, healthz_upstream};
use crate::websocket::ws_handler;
// Phase F PR-F1: imported as `classify_auth_mode` to make the call
// site self-documenting. `AuthMode` is re-exported under the same
// path for downstream handlers that read the value back out of
// `req.extensions()` (Phase F PR-F2/F3/F4).
use headroom_core::auth_mode::{classify as classify_auth_mode, AuthMode};
use headroom_core::compression_policy::CompressionPolicy;

/// Shared state passed to every handler.
///
/// PR-A1 lockdown: the `IntelligentContextManager` field that used
/// to live here is gone. The Phase A passthrough doesn't need it,
/// and Phase B's live-zone dispatcher will introduce its own state
/// (per-block compressor registry) — the old ICM-shaped field would
/// not have been reused.
///
/// PR-D4 adds `vertex_token_source`: an `Arc<dyn TokenSource>` used
/// by the Vertex `:rawPredict` / `:streamRawPredict` handlers to
/// resolve a GCP ADC bearer token. Production wires
/// [`crate::vertex::adc::GcpAdcTokenSource`] (lazy ADC chain
/// resolution + cached tokens with refresh-ahead-of-expiry); tests
/// inject [`crate::vertex::adc::StaticTokenSource`] so they never
/// hit real GCP.
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub client: reqwest::Client,
    /// PR-D1: AWS credentials resolved at startup via the
    /// `aws-config` default chain. `None` when the proxy boots
    /// without AWS creds available (operator running locally
    /// against a non-Bedrock upstream); the Bedrock invoke handler
    /// returns 5xx with a structured `event=bedrock_credentials_missing`
    /// log so failures are LOUD — no silent fallback to unsigned
    /// requests.
    pub bedrock_credentials: Option<Arc<aws_credential_types::Credentials>>,
    /// PR-E6: per-session structural-hash LRU for the cache-bust
    /// drift detector. Bounded to 1000 sessions in production. The
    /// detector is read-only — observing it never mutates the
    /// request body — so this can be cloned freely into every handler
    /// path that buffers the body.
    pub drift_state: DriftState,
    /// PR-D4: GCP ADC bearer-token source for Vertex routes. Default:
    /// [`crate::vertex::adc::GcpAdcTokenSource`] constructed lazily;
    /// the actual ADC chain is only resolved when the first Vertex
    /// route hits `bearer()`. Tests override via
    /// [`AppState::with_token_source`].
    pub vertex_token_source: Arc<dyn crate::vertex::TokenSource>,
}

/// PR-E6: maximum number of sessions tracked by the drift detector
/// LRU. Picked so that a noisy test fleet of 1000 distinct API keys
/// stays in cache for at least one full turn before the oldest
/// evicts. Operators with larger fleets can bump this; the memory
/// cost per entry is ~150 bytes (key string + 96-byte StructuralHash
/// + LRU overhead).
const DRIFT_DETECTOR_CAPACITY: usize = 1000;

impl AppState {
    pub fn new(config: Config) -> Result<Self, ProxyError> {
        let mut builder = reqwest::Client::builder()
            .connect_timeout(config.upstream_connect_timeout)
            .timeout(config.upstream_timeout)
            // Don't auto-follow redirects: pass them through verbatim.
            .redirect(reqwest::redirect::Policy::none())
            // Pool needs to be allowed to be idle for long-lived streams.
            .pool_idle_timeout(std::time::Duration::from_secs(90))
            // Both HTTP/1.1 and HTTP/2 negotiated via ALPN.
            ;
        if let Some(proxy_url) = &config.outbound_proxy {
            builder = builder
                .proxy(reqwest::Proxy::all(proxy_url).map_err(ProxyError::Upstream)?);
        }
        let client = builder.build().map_err(ProxyError::Upstream)?;

        // PR-D4: lazy ADC token source. Provider resolution is
        // deferred to first `bearer()` call so proxy startup stays
        // cheap when no Vertex route is exercised.
        let vertex_token_source: Arc<dyn crate::vertex::TokenSource> =
            Arc::new(crate::vertex::adc::GcpAdcTokenSource::new());

        Ok(Self {
            config: Arc::new(config),
            client,
            bedrock_credentials: None,
            drift_state: DriftState::new(DRIFT_DETECTOR_CAPACITY),
            vertex_token_source,
        })
    }

    /// PR-D1: attach AWS credentials resolved out-of-band (via
    /// `aws-config`'s default chain at startup). Returns the
    /// modified state; intended to be chained off `AppState::new`.
    /// Tests that don't exercise the Bedrock route can leave
    /// credentials unset (the catch-all paths never read them).
    pub fn with_bedrock_credentials(mut self, creds: aws_credential_types::Credentials) -> Self {
        self.bedrock_credentials = Some(Arc::new(creds));
        self
    }

    /// Test helper: build an `AppState` with an explicit token source.
    /// Lets the integration tests substitute a `StaticTokenSource` so
    /// the test suite never hits real GCP.
    pub fn with_token_source(
        config: Config,
        token_source: Arc<dyn crate::vertex::TokenSource>,
    ) -> Result<Self, ProxyError> {
        let mut s = Self::new(config)?;
        s.vertex_token_source = token_source;
        Ok(s)
    }
}

/// Build the axum app. `/healthz` and `/healthz/upstream` are intercepted;
/// everything else hits the catch-all forwarder. WebSocket upgrades are
/// handled inside the catch-all handler when an `Upgrade: websocket` header
/// is present.
pub fn build_app(state: AppState) -> Router {
    let mut router = Router::new()
        .route("/healthz", get(healthz))
        .route("/healthz/upstream", get(healthz_upstream))
        // PR-D3: Prometheus scrape endpoint. Renders the global
        // registry in text format. The handler is stateless — no
        // `AppState` needed — and idempotent across concurrent
        // scrapes (`prometheus`'s registry uses internal locking).
        // Mounted unconditionally because it has no dependencies on
        // any feature flag; an operator who doesn't want it scraped
        // simply firewalls the path.
        .route("/metrics", get(crate::observability::handle_metrics))
        // PR-C2: explicit POST route for /v1/chat/completions. The
        // handler buffers the body and re-injects it into
        // `forward_http`, which runs the OpenAI live-zone gate
        // alongside the existing Anthropic dispatcher. Non-POST
        // methods (and other paths) still fall through to
        // `catch_all` so the proxy stays a transparent reverse
        // proxy for everything else.
        .route(
            "/v1/chat/completions",
            post(crate::handlers::chat_completions::handle_chat_completions),
        )
        // PR-C3: explicit POST route for /v1/responses. Same forward
        // pattern as /v1/chat/completions — the handler buffers the
        // body, then `forward_http`'s gate dispatches to the
        // Responses live-zone walker via `compress_openai_responses_request`.
        .route(
            "/v1/responses",
            post(crate::handlers::responses::handle_responses),
        )
        // PR-D4: native Vertex publisher path. The Vertex AI Anthropic
        // publisher endpoints look like
        // `POST /v1beta1/projects/{p}/locations/{l}/publishers/anthropic/models/{m}:rawPredict`
        // (and `:streamRawPredict`). The trailing `:<verb>` is awkward
        // in axum's `:param` syntax, so we capture the entire trailing
        // segment as `:model_action` and split on the last `:` inside
        // the dispatcher. Both verbs share the same axum route shape
        // — matchit can't distinguish two patterns that overlap on the
        // literal parameter. The verb dispatch lives in
        // [`crate::vertex::handle_vertex_predict_dispatch`].
        .route(
            "/v1beta1/projects/:project/locations/:location/publishers/anthropic/models/:model_action",
            post(crate::vertex::handle_vertex_predict_dispatch),
        );

    // PR-D1: native AWS Bedrock InvokeModel route. Mounts only when
    // `enable_bedrock_native` is on (default). The handler runs the
    // live-zone compressor over Anthropic-shape bodies, signs with
    // SigV4, and forwards to the configured Bedrock endpoint. The
    // `/converse` route mounts the same handler — the wire shape is
    // identical for `anthropic.claude-*` model IDs (Bedrock just
    // accepts both legacy `invoke` and modern `converse` paths).
    if state.config.enable_bedrock_native {
        // PR-D3: Bedrock-scoped auth-mode middleware. Build a
        // sub-router with ONLY the Bedrock routes, attach the
        // auth-mode layer (so it fires before the handler runs and
        // is scoped to these routes alone — `/v1/messages`,
        // `/healthz`, etc. do NOT run through this middleware), and
        // merge it into the parent router. The merge composes
        // routes without changing their layer stacks; the parent's
        // `with_state` (applied at the end) hands `AppState` to the
        // Bedrock handlers identically.
        let bedrock_router: Router<AppState> = Router::new()
            .route(
                "/model/:model_id/invoke",
                post(crate::bedrock::invoke::handle_invoke),
            )
            .route(
                "/model/:model_id/converse",
                post(crate::bedrock::invoke::handle_invoke),
            )
            // PR-D2/PR-D5: streaming counterparts. Bedrock's protocol is
            // binary EventStream; the handler parses incrementally,
            // optionally translates each chunk to an SSE frame, and
            // tees translated frames into AnthropicStreamState for
            // telemetry. `invoke-with-response-stream` and
            // `converse-stream` share the same wire framing and
            // processing pipeline, so both route to the same handler.
            // See `bedrock::invoke_streaming`.
            .route(
                "/model/:model_id/invoke-with-response-stream",
                post(crate::bedrock::invoke_streaming::handle_invoke_streaming),
            )
            .route(
                "/model/:model_id/converse-stream",
                post(crate::bedrock::invoke_streaming::handle_invoke_streaming),
            )
            .route_layer(axum::middleware::from_fn(
                crate::bedrock::classify_and_attach_auth_mode,
            ))
            // Match the explicit body-size cap used by the other proxy handlers.
            // The `Bytes` extractor axum uses for Bedrock would otherwise cap
            // at axum's built-in 2 MiB default, rejecting valid large payloads.
            .layer(DefaultBodyLimit::max(state.config.max_body_bytes as usize));
        router = router.merge(bedrock_router);
        if !state.config.bedrock_validate_eventstream_crc {
            tracing::warn!(
                event = "bedrock_eventstream_crc_validation_disabled",
                "Bedrock EventStream CRC validation is DISABLED — \
                 only safe for debugging; production must keep \
                 --bedrock-validate-eventstream-crc=true"
            );
        }
    } else {
        tracing::warn!(
            event = "bedrock_native_disabled",
            "Bedrock native InvokeModel route disabled by \
             --enable-bedrock-native=false; Bedrock requests will fall \
             through to the catch-all (no SigV4 re-signing — fails closed)"
        );
    }

    // PR-C4: Conversations API (passthrough-with-instrumentation).
    // The flag is read once at app-build time so router shape
    // matches the configured policy. When disabled, requests still
    // reach upstream via `catch_all`'s streaming forwarder, but the
    // per-route handlers (and their structured-log breadcrumbs) are
    // NOT mounted — operators flip the toggle to silence logs, not
    // to break the surface. The catch-all preserves byte equivalence.
    if state.config.enable_conversations_passthrough {
        router = router
            .route(
                "/v1/conversations",
                post(crate::handlers::conversations::handle_conversations_create),
            )
            .route(
                "/v1/conversations/:conversation_id",
                get(crate::handlers::conversations::handle_conversations_get)
                    .post(crate::handlers::conversations::handle_conversations_update)
                    .delete(crate::handlers::conversations::handle_conversations_delete),
            )
            .route(
                "/v1/conversations/:conversation_id/items",
                post(crate::handlers::conversations::handle_conversations_items_create)
                    .get(crate::handlers::conversations::handle_conversations_items_list),
            )
            .route(
                "/v1/conversations/:conversation_id/items/:item_id",
                get(crate::handlers::conversations::handle_conversations_item_get)
                    .delete(crate::handlers::conversations::handle_conversations_item_delete),
            );
    } else {
        // Mirror the WARN we use elsewhere when a default-on guard
        // is flipped off. Logged at app-build time, not per-request.
        tracing::warn!(
            event = "conversations_passthrough_disabled",
            "Conversations API per-route handlers disabled by \
             --enable-conversations-passthrough=false; requests will \
             still reach upstream via the catch-all (no per-route logs)"
        );
    }

    router.fallback(any(catch_all)).with_state(state)
}

/// Catch-all handler. If the request is a WebSocket upgrade, hand off to the
/// ws module; otherwise forward as plain HTTP.
async fn catch_all(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    ws: Option<WebSocketUpgrade>,
    req: Request<Body>,
) -> Response<Body> {
    if is_websocket_upgrade(req.headers()) {
        if let Some(ws) = ws {
            return ws_handler(ws, state, client_addr, req).await;
        }
        // Header says websocket but axum didn't extract it (likely missing
        // Sec-WebSocket-Key) — fall through to HTTP forwarding which will
        // surface the upstream error.
    }
    forward_http(state, client_addr, req)
        .await
        .unwrap_or_else(|e| e.into_response())
}

/// True if `Content-Type` is `application/json` (with any optional
/// parameters like `; charset=utf-8`). Compression only inspects JSON
/// bodies — multipart uploads, form-encoded posts, and binary
/// payloads stream through untouched.
fn is_application_json(headers: &HeaderMap) -> bool {
    headers
        .get(http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|s| {
            // Take the media-type portion before any ';'. Trim and
            // compare case-insensitively per RFC 7231 §3.1.1.1.
            let media_type = s.split(';').next().unwrap_or("").trim();
            media_type.eq_ignore_ascii_case("application/json")
        })
        .unwrap_or(false)
}

fn is_websocket_upgrade(headers: &HeaderMap) -> bool {
    let upgrade = headers
        .get(http::header::UPGRADE)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.eq_ignore_ascii_case("websocket"))
        .unwrap_or(false);
    let connection = headers
        .get(http::header::CONNECTION)
        .and_then(|v| v.to_str().ok())
        .map(|s| {
            s.split(',')
                .any(|t| t.trim().eq_ignore_ascii_case("upgrade"))
        })
        .unwrap_or(false);
    upgrade && connection
}

/// Build the upstream URL by joining the configured base with the incoming
/// path-and-query. Preserves '?' and the query string verbatim.
pub(crate) fn build_upstream_url(base: &url::Url, uri: &Uri) -> Result<url::Url, ProxyError> {
    Ok(join_upstream_path(base, uri.path(), uri.query()))
}

/// Shared path-join helper used by HTTP and WebSocket handlers.
/// Appends `path` to `base`, preserving any base path prefix, then sets `query`.
pub(crate) fn join_upstream_path(base: &url::Url, path: &str, query: Option<&str>) -> url::Url {
    let mut joined = base.clone();
    // Strip trailing slash from base path so "http://x:1/api" + "/v1/foo"
    // yields "http://x:1/api/v1/foo" rather than "http://x:1/v1/foo".
    let base_path = joined.path().trim_end_matches('/').to_string();
    let combined = if path.is_empty() || path == "/" {
        if base_path.is_empty() {
            "/".to_string()
        } else {
            base_path
        }
    } else if base_path.is_empty() {
        path.to_string()
    } else {
        format!("{base_path}{path}")
    };
    joined.set_path(&combined);
    joined.set_query(query);
    joined
}

/// Forward an HTTP request to the upstream and stream the response back.
pub(crate) async fn forward_http(
    state: AppState,
    client_addr: SocketAddr,
    mut req: Request<Body>,
) -> Result<Response<Body>, ProxyError> {
    let start = Instant::now();
    let request_id = ensure_request_id(req.headers());
    let method = req.method().clone();
    let uri = req.uri().clone();
    let path_for_log = uri.path().to_string();
    let body_bytes_hint = req
        .headers()
        .get(http::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok());

    // Phase F PR-F1: classify auth mode at request entry. The result
    // is stored in request extensions so downstream handlers (cache
    // gates, header injection, lossy-compressor gates) read it
    // without re-classifying. Pure function, <10us per call —
    // doing it once here is cheaper than threading the result.
    let auth_mode = classify_auth_mode(req.headers());
    req.extensions_mut().insert(auth_mode);

    // Phase F PR-F2.1, c2/6: derive the per-mode CompressionPolicy at
    // request entry and stash alongside auth_mode. Storing the policy
    // (not just auth_mode) in extensions lets downstream stages read
    // the gate they need directly — no per-stage `for_mode` call.
    //
    // c3/6: when `auth_mode_policy_enforcement` is `Disabled` (default
    // until c6/6), force the policy to PAYG regardless of classifier
    // output. This means c4/6 + c5/6 only ship behaviour change when
    // an operator opts in via the env var, so the PR sequence is
    // safely landed in main without flipping the live wire on default
    // users until the final commit.
    let policy = if state.config.auth_mode_policy_enforcement.is_enabled() {
        CompressionPolicy::for_mode(auth_mode)
    } else {
        CompressionPolicy::for_mode(AuthMode::Payg)
    };
    req.extensions_mut().insert(policy);

    // Per PR-A1: structured entry log. The `auth_mode` field is now
    // populated with the real classification result (Phase F PR-F1
    // replaces the prior `auth_mode_placeholder = "unknown"`). Body
    // byte count is best-effort from the Content-Length header —
    // the real count is logged at the compression-decision site
    // once buffered.
    tracing::debug!(
        event = "auth_mode_classified",
        request_id = %request_id,
        auth_mode = auth_mode.as_str(),
        method = %method,
        path = %path_for_log,
        content_length_bytes = ?body_bytes_hint,
        "request received"
    );

    // F2.1 c2/6: emit the policy that the request will run under so
    // F2.2 has bake-time data to tune from. One log per request,
    // structured fields so it joins on auth_mode + request_id.
    // c3/6 adds `enforcement` so the dashboard can split "policy
    // resolved as PAYG because mode is PAYG" from "policy resolved as
    // PAYG because the enforcement flag is off."
    //
    // F2.2 c2/3: extend the structured fields with the three new
    // tuning fields so the bake dashboard has per-mode observability
    // for the F2.2-followup tune. ``volatile_token_threshold`` /
    // ``max_lossy_ratio`` are plumbed-but-unconsumed today, so the
    // log lines are the only signal that the values are flowing
    // correctly through the proxy → handlers → transforms path.
    tracing::debug!(
        event = "policy_selected",
        request_id = %request_id,
        auth_mode = auth_mode.as_str(),
        enforcement = state.config.auth_mode_policy_enforcement.as_str(),
        live_zone_only = policy.live_zone_only,
        cache_aligner_enabled = policy.cache_aligner_enabled,
        volatile_token_threshold = policy.volatile_token_threshold,
        max_lossy_ratio = policy.max_lossy_ratio,
        toin_read_only = policy.toin_read_only,
        "compression policy resolved"
    );

    let upstream_url = build_upstream_url(&state.config.upstream, &uri)?;

    // Forwarded-Host: prefer client's Host. Forwarded-Proto: assume http for
    // now (we don't terminate TLS in this binary; if a TLS terminator is in
    // front, it should rewrite this — which we'd handle by not overwriting
    // an existing one in a future change).
    let forwarded_host = req
        .headers()
        .get(http::header::HOST)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    // Build the outgoing headers off the incoming ones, then optionally drop
    // Host (rewrite_host=true => let reqwest set its own Host for the upstream).
    // PR-A5 (P5-49): strip internal `x-headroom-*` from upstream-bound
    // requests when `Config::strip_internal_headers == Enabled` (default).
    let strip_internal = state.config.strip_internal_headers.is_enabled();
    let pre_strip_internal_count = req
        .headers()
        .iter()
        .filter(|(name, _)| crate::headers::is_internal_header(name))
        .count();
    let mut outgoing_headers = build_forward_request_headers(
        req.headers(),
        client_addr.ip(),
        "http",
        forwarded_host.as_deref(),
        &request_id,
        strip_internal,
    );
    if strip_internal && pre_strip_internal_count > 0 {
        tracing::info!(
            event = "outbound_headers",
            forwarder = "rust_proxy",
            stripped_count = pre_strip_internal_count,
            request_id = %request_id,
            "stripped internal x-headroom-* headers from upstream-bound request"
        );
    } else if !strip_internal && pre_strip_internal_count > 0 {
        tracing::warn!(
            event = "outbound_headers",
            forwarder = "rust_proxy",
            mode = "disabled",
            internal_count = pre_strip_internal_count,
            request_id = %request_id,
            "HEADROOM_PROXY_STRIP_INTERNAL_HEADERS=disabled; \
             internal x-headroom-* headers forwarded to upstream"
        );
    }
    if !state.config.rewrite_host {
        if let Some(h) = req.headers().get(http::header::HOST) {
            outgoing_headers.insert(http::header::HOST, h.clone());
        }
    }

    // ─── COMPRESSION GATE ──────────────────────────────────────────────
    //
    // PR-A1 lockdown (per `REALIGNMENT/03-phase-A-lockdown.md`): the
    // `/v1/messages` path no longer mutates the body. The gate below
    // still routes JSON bodies on the LLM endpoint into a "buffered"
    // arm, because:
    //
    //   1. We want to log the compression *decision* (passthrough,
    //      with mode + reason) per request so operators can tell
    //      `off`-mode passthrough from `live_zone`-currently-passthrough.
    //   2. Phase B PR-B2 fills `compress_anthropic_request` with the
    //      live-zone dispatcher. Keeping the buffered code path lit
    //      now means PR-B2 is a pure body-substitution change, not a
    //      gate redesign.
    //   3. The buffered branch issues a `debug_assert!` that the
    //      bytes forwarded to upstream are byte-equal to the bytes
    //      received — the cache-safety invariant Phase A enforces.
    //
    // Gate criteria (ALL true → buffered passthrough; otherwise stream):
    //
    //   - `state.config.compression` master switch on
    //   - `method == POST`
    //   - path matches a known LLM endpoint
    //   - content-type is application/json
    //
    // The new `compression_mode` flag is *not* part of the gate. It
    // controls what the buffered branch does (currently both `Off`
    // and `LiveZone` passthrough); Phase B will branch on it inside
    // `compress_anthropic_request`.
    let should_intercept = state.config.compression
        && method == axum::http::Method::POST
        && compression::is_compressible_path(uri.path())
        && is_application_json(req.headers());

    // PR-E6: capture a header snapshot BEFORE the body is consumed so
    // the drift detector can derive a per-session key from
    // `Authorization`/`x-api-key`/`User-Agent`. `req` will be moved
    // into either `to_bytes(req.into_body())` (buffered branch) or
    // `req.into_body().into_data_stream()` (streaming branch); both
    // discard the headers along with the body. Snapshot here keeps
    // both branches clean.
    let headers_snapshot = if should_intercept {
        Some(req.headers().clone())
    } else {
        None
    };

    let reqwest_method = reqwest::Method::from_bytes(method.as_str().as_bytes())
        .map_err(|e| ProxyError::InvalidHeader(e.to_string()))?;

    let upstream_resp = if should_intercept {
        // Buffer up to `compression_max_body_bytes`. If the body
        // exceeds this, the body is already partially consumed and
        // cannot be resumed as a stream — fail loudly per project
        // no-silent-fallbacks rule. Operators tune
        // `--compression-max-body-bytes` upward if they hit this.
        //
        // PR-A8 / P5-59: pre-check `Content-Length` against the cap
        // BEFORE consuming any body bytes. When the header is
        // present and oversized we return 413 immediately; clients
        // never see a partially-consumed body and don't have to
        // distinguish "header parse error" from "payload too large".
        // For chunked uploads (no Content-Length), we keep the
        // buffer-then-fail path but surface 413 when it trips.
        let max = state.config.compression_max_body_bytes as usize;
        if let Some(len) = body_bytes_hint {
            if len as usize > max {
                tracing::warn!(
                    request_id = %request_id,
                    path = %path_for_log,
                    limit_bytes = max,
                    content_length = len,
                    "compression: Content-Length exceeds buffer limit; \
                     returning 413 without consuming body"
                );
                return Err(ProxyError::PayloadTooLarge(format!(
                    "request Content-Length {len} exceeds compression \
                     buffer limit ({max} bytes)"
                )));
            }
        }
        let buffered = match to_bytes(req.into_body(), max).await {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(
                    request_id = %request_id,
                    path = %path_for_log,
                    limit_bytes = max,
                    error = %e,
                    "compression: body exceeds buffer limit; failing loudly (cannot \
                     resume streaming once the body has been partially consumed)"
                );
                return Err(ProxyError::PayloadTooLarge(format!(
                    "request body exceeds compression buffer limit ({max} bytes): {e}"
                )));
            }
        };

        // PR-C2: dispatch on the endpoint classification so each
        // provider hits its own live-zone walker. PR-B2/B3/B4 wired
        // the Anthropic dispatcher; PR-C2 adds the OpenAI Chat
        // Completions sibling. The classification was already
        // computed by `is_compressible_path` above; we re-classify
        // here so a single-source `match` decides which dispatcher
        // runs and what skip rules apply.
        //
        // Skip rules (per spec PR-C2):
        // - OpenAI Chat: `n > 1` skips compression entirely (multiple
        //   completions imply non-determinism scenarios). `tool_choice`
        //   and `stream_options` are NOT skip conditions — they
        //   round-trip byte-equal as a side effect of byte-range surgery.
        // - Anthropic: no extra skip rules at this layer.
        let endpoint = compression::classify_compressible_path(uri.path())
            .expect("is_compressible_path guarded above");

        // PR-E5 + PR-E6: cache-stabilization observability hooks.
        // Both run READ-ONLY against the buffered body and emit
        // structured logs only — passthrough invariant from Phase A
        // is preserved. Parsing happens once and is shared. Cheap
        // parse failure (malformed JSON) silently skips both
        // detectors; the dispatcher below logs its own parse-error
        // decision. The hooks run regardless of whether the
        // dispatcher returns `NoCompression`, `Compressed`, or
        // `Passthrough`.
        //
        // Bedrock and other shape-mismatched paths skip the drift
        // detector specifically; their wire shape is different
        // enough that a canonical-bytes hash would compare apples
        // to oranges. The volatile detector handles its own
        // shape-dispatch via `ApiKind::from_endpoint`.
        if let Ok(parsed) = serde_json::from_slice::<serde_json::Value>(&buffered) {
            // PR-E5: volatile-content detector. Emits one WARN per
            // finding (capped at 10) for content that busts cache
            // (timestamps, UUIDs, ID-named fields).
            let volatile_kind =
                cache_stabilization::volatile_detector::ApiKind::from_endpoint(endpoint);
            let findings = cache_stabilization::volatile_detector::detect_volatile_content(
                &parsed,
                volatile_kind,
            );
            if !findings.is_empty() {
                cache_stabilization::volatile_detector::emit_volatile_warnings(
                    &findings,
                    &request_id,
                );
            }

            // PR-E6: cache-bust drift detector. SHA-256 fingerprints
            // the cache hot zone (system / tools / first 3 messages);
            // a mismatch between consecutive turns of the same session
            // emits a `cache_drift_observed` event so operators see
            // invisible cache busts.
            let drift_kind = match endpoint {
                compression::CompressibleEndpoint::AnthropicMessages => Some(ApiKind::Anthropic),
                compression::CompressibleEndpoint::OpenAiChatCompletions => {
                    Some(ApiKind::OpenAiChat)
                }
                compression::CompressibleEndpoint::OpenAiResponses => {
                    Some(ApiKind::OpenAiResponses)
                }
            };
            if let (Some(kind), Some(headers)) = (drift_kind, headers_snapshot.as_ref()) {
                let session_key = derive_session_key(headers, &client_addr);
                let hash = compute_structural_hash(&parsed, kind);
                observe_drift(&state.drift_state, &session_key, hash);
            }
        }
        let outcome = match endpoint {
            compression::CompressibleEndpoint::AnthropicMessages => {
                // PR-E3: thread the F1-classified auth_mode into the
                // dispatcher so cache_control auto-placement gates on
                // PAYG only. Pulled from request extensions where it
                // was stashed at request entry (line ~325 above).
                compression::compress_anthropic_request(
                    &buffered,
                    state.config.compression_mode,
                    state.config.cache_control_auto_frozen,
                    auth_mode,
                    &request_id,
                )
            }
            compression::CompressibleEndpoint::OpenAiChatCompletions => {
                let skip = compression::should_skip_compression(&buffered);
                if skip.is_skip() {
                    tracing::info!(
                        event = "compression_decision",
                        request_id = %request_id,
                        path = "/v1/chat/completions",
                        method = "POST",
                        compression_mode = state.config.compression_mode.as_str(),
                        decision = "passthrough",
                        reason = skip.as_log_str(),
                        body_bytes = buffered.len(),
                        "openai chat compression skipped pre-dispatch"
                    );
                    compression::Outcome::NoCompression
                } else {
                    compression::compress_openai_chat_request(
                        &buffered,
                        state.config.compression_mode,
                        auth_mode,
                        &request_id,
                    )
                }
            }
            // PR-C3: OpenAI Responses (`/v1/responses`). The Responses
            // dispatcher walks an explicitly-typed `input` array and
            // only rewrites the latest of each compressible `*_output`
            // kind plus the latest `message` text. Cache hot zone is
            // every other item type (passthrough verbatim).
            compression::CompressibleEndpoint::OpenAiResponses => {
                compression::compress_openai_responses_request(
                    &buffered,
                    state.config.compression_mode,
                    auth_mode,
                    &request_id,
                )
            }
        };

        // C2 fix: snapshot the original buffered byte-length AND the
        // dispatcher's "is this a passthrough arm?" decision BEFORE
        // `outcome` is consumed by the match below. The
        // passthrough-bytes-modified alarm fires when a path that
        // promised byte-equal passthrough produces a different
        // length downstream.
        let original_buffered_len = buffered.len();
        let outcome_is_passthrough_class = matches!(
            outcome,
            compression::Outcome::NoCompression | compression::Outcome::Passthrough { .. }
        );
        let body_to_send = match outcome {
            compression::Outcome::NoCompression => {
                // PR-B2: forward the *original* buffered bytes. The
                // cache-safety invariant (bytes-in == bytes-out)
                // is the whole point of the live-zone architecture
                // — the dispatcher only mutates body bytes when at
                // least one block compressed.
                buffered
            }
            // PR-B3+ produces `Compressed` from the live-zone
            // dispatcher when at least one per-type compressor
            // mutates a block. Already wired here so the next phase
            // is a pure addition.
            compression::Outcome::Compressed {
                body,
                tokens_before,
                tokens_after,
                strategies_applied,
                markers_inserted,
                per_strategy_tokens,
            } => {
                tracing::info!(
                    request_id = %request_id,
                    path = %path_for_log,
                    tokens_before = tokens_before,
                    tokens_after = tokens_after,
                    tokens_freed = tokens_before.saturating_sub(tokens_after),
                    strategies = ?strategies_applied,
                    markers = markers_inserted.len(),
                    "compression applied"
                );
                // Phase G PR-G3 + H1: emit one
                // `proxy_compression_ratio_by_strategy` sample per
                // strategy with the *strategy's own* before/after
                // token counts. The pre-H1 code emitted the same
                // aggregate ratio for every strategy in
                // `strategies_applied`, so Phase H per-strategy
                // dashboards read garbage when multiple strategies
                // ran on one body. We now plumb per-strategy tokens
                // from the manifest at the wrapper site
                // (`live_zone_anthropic`, `live_zone_openai`,
                // `live_zone_responses`).
                //
                // Fallback: when `per_strategy_tokens` is empty —
                // i.e. the Outcome came from a Phase E
                // normalization pass that doesn't track per-strategy
                // tokens — we emit one aggregate-labelled sample so
                // dashboards still see *that* a compression ran. We
                // log loudly so this is visible.
                if !per_strategy_tokens.is_empty() {
                    for entry in &per_strategy_tokens {
                        crate::observability::observe_compression_ratio(
                            entry.strategy,
                            "aggregate",
                            entry.original_tokens,
                            entry.compressed_tokens,
                        );
                    }
                } else if tokens_before > 0 && tokens_after < tokens_before {
                    tracing::debug!(
                        event = "compression_ratio_emit_aggregate_only",
                        request_id = %request_id,
                        path = %path_for_log,
                        strategies = ?strategies_applied,
                        reason = "no_per_strategy_tokens",
                        "emitting one aggregate-labelled compression_ratio sample because \
                         the dispatcher did not surface per-strategy token counts \
                         (Phase E normalization paths)"
                    );
                    crate::observability::observe_compression_ratio(
                        "aggregate",
                        "aggregate",
                        tokens_before,
                        tokens_after,
                    );
                }
                body
            }
            compression::Outcome::Passthrough { reason } => {
                tracing::warn!(
                    request_id = %request_id,
                    path = %path_for_log,
                    reason = ?reason,
                    "compression: passthrough on parse/serialize"
                );
                buffered
            }
        };

        // C2 fix: cache-safety alarm. When the dispatcher returned
        // `NoCompression` or `Passthrough`, the post-dispatcher body
        // MUST be byte-length-equal to the original buffered body.
        // Any delta is an accidental cache-poisoning regression and
        // the alarm metric `proxy_passthrough_bytes_modified_total{path}`
        // fires with the byte delta as its increment. We check BEFORE
        // the PR-E4 prompt_cache_key injector runs because that
        // injector is a legitimate, intentional byte mutation gated
        // on PAYG; it must not trip the alarm.
        if outcome_is_passthrough_class && body_to_send.len() != original_buffered_len {
            let delta = body_to_send.len().abs_diff(original_buffered_len) as u64;
            crate::observability::record_passthrough_bytes_modified(
                &path_for_log,
                delta,
                &request_id,
            );
        }

        // PR-E4: OpenAI `prompt_cache_key` auto-injection.
        //
        // Universal safety contract: only mutate when the caller
        // is on `AuthMode::Payg`. OAuth/Subscription bytes flow
        // through byte-equal — those clients cannot afford
        // synthesised cache keys (OAuth scopes pin to
        // `(account, model, session)` and subscription clients
        // are programmatically fingerprinted by the upstream).
        //
        // The injector also self-skips when the customer has
        // already set a non-empty `prompt_cache_key`. Every skip
        // path emits a structured `e4_skipped` event so cache-hit
        // dashboards can attribute miss rates to gating reasons
        // rather than guessing.
        let body_to_send = match endpoint {
            compression::CompressibleEndpoint::OpenAiChatCompletions
            | compression::CompressibleEndpoint::OpenAiResponses => {
                let shape = match endpoint {
                    compression::CompressibleEndpoint::OpenAiResponses => {
                        cache_stabilization::openai_cache_key::OpenAiShape::Responses
                    }
                    _ => cache_stabilization::openai_cache_key::OpenAiShape::ChatCompletions,
                };
                maybe_inject_openai_prompt_cache_key(
                    body_to_send,
                    shape,
                    auth_mode,
                    &request_id,
                    &path_for_log,
                )
            }
            compression::CompressibleEndpoint::AnthropicMessages => body_to_send,
        };

        // Forward the (Phase A: identical) buffered bytes. reqwest
        // sets its own Content-Length from the body bytes — the
        // existing `build_forward_request_headers` already strips
        // the client-supplied Content-Length for us.
        state
            .client
            .request(reqwest_method, upstream_url.clone())
            .headers(outgoing_headers)
            .body(body_to_send)
            .send()
            .await?
    } else {
        // Pure streaming path — the original passthrough behaviour.
        let body_stream =
            TryStreamExt::map_err(req.into_body().into_data_stream(), std::io::Error::other);
        let reqwest_body = reqwest::Body::wrap_stream(body_stream);
        state
            .client
            .request(reqwest_method, upstream_url.clone())
            .headers(outgoing_headers)
            .body(reqwest_body)
            .send()
            .await?
    };

    let upstream_status = upstream_resp.status();
    let status = StatusCode::from_u16(upstream_status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);

    // PR-A8 / P5-57: capture the upstream request id BEFORE we move
    // `upstream_resp.headers()` into the response filter. Anthropic
    // emits `request-id` (lowercase, no `x-`); OpenAI emits
    // `x-request-id`. We forward both to the client unchanged in
    // `resp_headers` and additionally surface a side-channel
    // `headroom-request-id` header so callers can correlate proxy
    // logs without conflating with the proxy's own `x-request-id`.
    let upstream_request_id_anthropic = upstream_resp
        .headers()
        .get("request-id")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let upstream_request_id_openai = upstream_resp
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    // Prefer the provider-specific id whichever was set. Both
    // present is unusual but legal; prefer Anthropic since it's the
    // path-shape we lockdown with cache invariants.
    let upstream_request_id = upstream_request_id_anthropic
        .clone()
        .or_else(|| upstream_request_id_openai.clone());

    // PR-C1: detect SSE responses so the state machine can run in
    // parallel with the byte-passthrough. We classify ONCE here and
    // pick the response provider arm based on the request path —
    // bytes flow to the client unchanged; the state machine sinks
    // bytes into a `tokio::sync::mpsc` and runs in a spawned task
    // that can never block the byte path.
    //
    // PR-C4: the OpenAI Responses arm is gated by
    // `enable_responses_streaming`. When that flag is false the
    // tee is short-circuited to `None` so the framer + state
    // machine don't spin up and bytes flow opaquely. Other
    // providers' state machines are unaffected.
    let is_sse = is_sse_response(upstream_resp.headers());
    let sse_kind = if is_sse {
        let kind = SseStreamKind::for_request_path(&path_for_log);
        if matches!(kind, SseStreamKind::OpenAiResponses)
            && !state.config.enable_responses_streaming
        {
            tracing::info!(
                request_id = %request_id,
                path = %path_for_log,
                event = "responses_streaming_state_machine_skipped",
                reason = "enable_responses_streaming=false",
                "PR-C4 streaming pipeline disabled; SSE bytes pass through without telemetry"
            );
            SseStreamKind::None
        } else {
            kind
        }
    } else {
        SseStreamKind::None
    };

    let resp_headers = filter_response_headers(upstream_resp.headers());

    // Phase G PR-G3: extract upstream rate-limit headers from this
    // response and record them as gauges. The `provider` label is
    // chosen by which of the upstream `request-id` shapes we saw
    // (Anthropic vs OpenAI). When neither shape was detected we
    // skip emission rather than guessing — per realignment build-
    // constraint "no silent fallbacks".
    let rate_limit_snapshot =
        crate::observability::extract_rate_limit_snapshot(upstream_resp.headers());
    let rate_limit_provider: Option<&'static str> = if upstream_request_id_anthropic.is_some() {
        Some(crate::observability::cache_hit_rate_provider::ANTHROPIC)
    } else if upstream_request_id_openai.is_some() {
        // We can't distinguish chat vs responses purely from the
        // request-id header; the `path_for_log` is more specific.
        Some(if path_for_log.contains("/v1/responses") {
            crate::observability::cache_hit_rate_provider::OPENAI_RESPONSES
        } else {
            crate::observability::cache_hit_rate_provider::OPENAI_CHAT
        })
    } else {
        None
    };
    if let Some(provider) = rate_limit_provider {
        crate::observability::record_rate_limit_snapshot(
            provider,
            &rate_limit_snapshot,
            &request_id,
        );
    } else if rate_limit_snapshot.remaining_requests.is_some()
        || rate_limit_snapshot.remaining_tokens.is_some()
        || rate_limit_snapshot.remaining_input_tokens.is_some()
        || rate_limit_snapshot.remaining_output_tokens.is_some()
    {
        // Headers present but provider unattributable. Log loud so
        // operators see the wire-format drift; do not emit unlabelled
        // metrics.
        tracing::debug!(
            event = "rate_limit_snapshot_unattributable",
            request_id = %request_id,
            path = %path_for_log,
            "rate-limit headers present but provider couldn't be inferred; skipping gauge emit"
        );
    }

    // Stream response body back without buffering. Wrap errors so mid-stream
    // upstream failures are logged rather than silently truncating the client.
    //
    // PR-C1: when this is an SSE response, tee each chunk into a
    // bounded mpsc so the spawned state-machine task can update
    // telemetry without ever holding up the client. The mpsc is
    // bounded; if the parser falls behind, `try_send` fails and we
    // log + drop — the byte path is not affected. This is the
    // explicit "never block on parser readiness" contract.
    let rid = request_id.clone();
    let parser_tx = if !matches!(sse_kind, SseStreamKind::None) {
        let (tx, rx) = tokio::sync::mpsc::channel::<bytes::Bytes>(SSE_PARSER_QUEUE_DEPTH);
        let rid_for_parser = request_id.clone();
        tokio::spawn(run_sse_state_machine(sse_kind, rx, rid_for_parser));
        Some(tx)
    } else {
        None
    };
    let resp_stream = upstream_resp.bytes_stream().map(move |r| match r {
        Ok(b) => {
            if let Some(tx) = &parser_tx {
                // Best-effort tee. Bounded channel; the state
                // machine never blocks the client byte path.
                if let Err(e) = tx.try_send(b.clone()) {
                    tracing::debug!(
                        request_id = %rid,
                        error = %e,
                        "sse parser queue full or closed; skipping telemetry chunk"
                    );
                }
            }
            Ok(b)
        }
        Err(e) => {
            tracing::warn!(request_id = %rid, error = %e, "upstream stream error mid-response");
            Err(e)
        }
    });
    let body = Body::from_stream(resp_stream);

    let mut response = Response::builder().status(status);
    {
        let h = response.headers_mut().expect("builder has headers");
        h.extend(resp_headers);
        // Echo X-Request-Id back to the client.
        if let Ok(v) = http::HeaderValue::from_str(&request_id) {
            h.insert(HeaderName::from_static("x-request-id"), v);
        }
        // PR-A8 / P5-57: surface the upstream id in a distinct
        // header so it's never conflated with the proxy's own.
        if let Some(uid) = upstream_request_id.as_deref() {
            if let Ok(v) = http::HeaderValue::from_str(uid) {
                h.insert(HeaderName::from_static("headroom-upstream-request-id"), v);
            }
        }
    }
    let response = response
        .body(body)
        .map_err(|e| ProxyError::InvalidHeader(e.to_string()))?;

    tracing::info!(
        request_id = %request_id,
        upstream_request_id = upstream_request_id.as_deref().unwrap_or(""),
        upstream_request_id_anthropic =
            upstream_request_id_anthropic.as_deref().unwrap_or(""),
        upstream_request_id_openai =
            upstream_request_id_openai.as_deref().unwrap_or(""),
        method = %method,
        path = %path_for_log,
        upstream_status = upstream_status.as_u16(),
        latency_ms = start.elapsed().as_millis() as u64,
        protocol = "http",
        "forwarded"
    );

    Ok(response)
}

/// Bound on the in-flight queue between the byte-passthrough and the
/// SSE state-machine task. Picked so that under steady-state streaming
/// load (~5 events/100ms typical) the parser is never blocked on
/// queue space, yet a stalled parser can't grow memory unboundedly.
/// Tunable via `proxy.toml` if a deployment finds this insufficient.
const SSE_PARSER_QUEUE_DEPTH: usize = 256;

/// Which provider's state machine should run on this stream. Picked
/// from the *request* path because the response content-type
/// (`text/event-stream`) is identical across providers.
#[derive(Debug, Clone, Copy)]
enum SseStreamKind {
    None,
    Anthropic,
    OpenAiChat,
    OpenAiResponses,
}

impl SseStreamKind {
    fn for_request_path(path: &str) -> Self {
        match path {
            "/v1/messages" => Self::Anthropic,
            "/v1/chat/completions" => Self::OpenAiChat,
            "/v1/responses" => Self::OpenAiResponses,
            // No telemetry parser registered for this endpoint.
            // We still pass bytes through unchanged.
            _ => Self::None,
        }
    }
}

/// True if the upstream response is an SSE stream. Compares
/// `content-type` against `text/event-stream` (with optional
/// parameters). RFC 7231 §3.1.1.1: media types compare
/// case-insensitive on the type/subtype tokens.
fn is_sse_response(headers: &http::HeaderMap) -> bool {
    headers
        .get(http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|s| {
            let media_type = s.split(';').next().unwrap_or("").trim();
            media_type.eq_ignore_ascii_case("text/event-stream")
        })
        .unwrap_or(false)
}

/// PR-E4: OpenAI `prompt_cache_key` auto-injection helper.
///
/// Gates on [`AuthMode::Payg`] and the in-body
/// `prompt_cache_key` skip rule, parses the body once, mutates if
/// appropriate, and re-serialises. Returns the original `body` on
/// any non-applicable path — every error / skip leaves the bytes
/// untouched (Phase A passthrough invariant).
///
/// Logs `e4_skipped` for each skip reason and `e4_applied` with
/// only the first [`KEY_PREFIX_LOG_LEN`] hex chars of the key
/// (never the full key, which is identifying material).
///
/// [`KEY_PREFIX_LOG_LEN`]: cache_stabilization::openai_cache_key::KEY_PREFIX_LOG_LEN
fn maybe_inject_openai_prompt_cache_key(
    body: bytes::Bytes,
    shape: cache_stabilization::openai_cache_key::OpenAiShape,
    auth_mode: AuthMode,
    request_id: &str,
    path: &str,
) -> bytes::Bytes {
    use cache_stabilization::openai_cache_key::{
        inject_prompt_cache_key, InjectOutcome, SkipReason,
    };

    // Auth-mode gate: only PAYG bodies are eligible. OAuth /
    // Subscription requests pass through byte-equal — synthesised
    // cache keys would look like cache-evasion to the upstream
    // and could void OAuth scopes pinned to `(account, model,
    // session)`.
    if !matches!(auth_mode, AuthMode::Payg) {
        tracing::info!(
            event = "e4_skipped",
            request_id = %request_id,
            path = %path,
            reason = "auth_mode",
            auth_mode = auth_mode.as_str(),
            "PR-E4: skipped prompt_cache_key injection (non-PAYG auth mode)"
        );
        return body;
    }

    // Parse for the inject step. Failure here is silent — the
    // dispatcher above already logged the parse outcome on its
    // own decision path; we don't want to double-log. The body
    // round-trips unchanged.
    let mut parsed: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => {
            return body;
        }
    };

    match inject_prompt_cache_key(&mut parsed, shape) {
        InjectOutcome::Applied { key_prefix } => {
            // Re-serialise. If serialization fails (would be very
            // unusual — we just successfully parsed), fall back
            // to the original bytes. No-silent-fallback rule: log
            // it loudly so a regression can't hide.
            match serde_json::to_vec(&parsed) {
                Ok(buf) => {
                    tracing::info!(
                        event = "e4_applied",
                        request_id = %request_id,
                        path = %path,
                        key_prefix = %key_prefix,
                        body_bytes_in = body.len(),
                        body_bytes_out = buf.len(),
                        "PR-E4: injected prompt_cache_key"
                    );
                    bytes::Bytes::from(buf)
                }
                Err(e) => {
                    tracing::error!(
                        event = "e4_serialize_error",
                        request_id = %request_id,
                        path = %path,
                        error = %e,
                        "PR-E4: re-serialize after injection failed; forwarding original bytes"
                    );
                    body
                }
            }
        }
        InjectOutcome::Skipped { reason } => {
            // Log only the customer-visible KeyPresent skip; the
            // NotAnObject skip is structurally impossible past
            // the dispatcher gate but is surfaced separately for
            // operators chasing pathological inputs.
            match reason {
                SkipReason::KeyPresent => {
                    tracing::info!(
                        event = "e4_skipped",
                        request_id = %request_id,
                        path = %path,
                        reason = SkipReason::KeyPresent.as_str(),
                        "PR-E4: skipped prompt_cache_key injection (customer-set value preserved)"
                    );
                }
                SkipReason::NotAnObject => {
                    tracing::warn!(
                        event = "e4_skipped",
                        request_id = %request_id,
                        path = %path,
                        reason = SkipReason::NotAnObject.as_str(),
                        "PR-E4: body is not a JSON object; passthrough"
                    );
                }
            }
            body
        }
    }
}

/// Drive the per-provider state machine over a stream of byte chunks.
/// Lives in its own task; the byte path never waits on it.
async fn run_sse_state_machine(
    kind: SseStreamKind,
    mut rx: tokio::sync::mpsc::Receiver<bytes::Bytes>,
    request_id: String,
) {
    use crate::sse::framing::SseFramer;

    let mut framer = SseFramer::new();
    // The state machines are different types; rather than introducing
    // a trait object dance, run each variant in its own arm. The dead
    // branches compile out cleanly and the hot path stays monomorphic.
    match kind {
        SseStreamKind::Anthropic => {
            let mut state = crate::sse::anthropic::AnthropicStreamState::new();
            while let Some(chunk) = rx.recv().await {
                framer.push(&chunk);
                while let Some(ev_result) = framer.next_event() {
                    match ev_result {
                        Ok(ev) => {
                            if let Err(e) = state.apply(ev) {
                                tracing::warn!(
                                    request_id = %request_id,
                                    error = %e,
                                    "sse anthropic state-machine apply error"
                                );
                            }
                        }
                        Err(e) => {
                            tracing::warn!(
                                request_id = %request_id,
                                error = %e,
                                "sse framer error"
                            );
                        }
                    }
                }
            }
            // Phase G PR-G3 + H2: emit per-session cache-hit-rate
            // ONLY when the stream completed cleanly with
            // `message_stop`. The gate is encapsulated by the
            // pure function `compute_anthropic_session_hit_rate`
            // so the H2 contract has a unit-testable surface.
            match crate::observability::cache_hit_rate::compute_anthropic_session_hit_rate(&state) {
                Some(rate) => {
                    crate::observability::observe_cache_hit_rate(
                        crate::observability::cache_hit_rate_provider::ANTHROPIC,
                        &request_id,
                        rate,
                    );
                }
                None => {
                    tracing::debug!(
                        event = "cache_hit_rate_skipped",
                        request_id = %request_id,
                        provider = "anthropic",
                        status = ?state.status,
                        input_tokens = state.usage.input_tokens,
                        cache_read_input_tokens = state.usage.cache_read_input_tokens,
                        cache_creation_input_tokens = state.usage.cache_creation_input_tokens,
                        "skipping proxy_cache_hit_rate_per_session: H2 gate or zero denominator"
                    );
                }
            }
            tracing::info!(
                request_id = %request_id,
                provider = "anthropic",
                input_tokens = state.usage.input_tokens,
                output_tokens = state.usage.output_tokens,
                cache_creation_input_tokens = state.usage.cache_creation_input_tokens,
                cache_read_input_tokens = state.usage.cache_read_input_tokens,
                stop_reason = state.stop_reason.as_deref().unwrap_or(""),
                blocks = state.blocks.len(),
                "sse stream closed"
            );
        }
        SseStreamKind::OpenAiChat => {
            let mut state = crate::sse::openai_chat::ChunkState::new();
            while let Some(chunk) = rx.recv().await {
                framer.push(&chunk);
                while let Some(ev_result) = framer.next_event() {
                    match ev_result {
                        Ok(ev) => {
                            if let Err(e) = state.apply(ev) {
                                tracing::warn!(
                                    request_id = %request_id,
                                    error = %e,
                                    "sse openai_chat state-machine apply error"
                                );
                            }
                        }
                        Err(e) => {
                            tracing::warn!(
                                request_id = %request_id,
                                error = %e,
                                "sse framer error"
                            );
                        }
                    }
                }
            }
            // Phase G PR-G3: emit cache-hit-rate from the final usage
            // chunk. OpenAI only emits this when
            // `stream_options.include_usage = true`; absence is a
            // signal, not a fallback condition — `usage = None` →
            // skip. The H2 gate is implicit here: the final usage
            // chunk only arrives when the stream completed (it's
            // OpenAI's terminal-status equivalent).
            if let Some(usage) = &state.usage {
                let input_tokens = usage
                    .get("prompt_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let cached_tokens = usage
                    .get("prompt_tokens_details")
                    .and_then(|d| d.get("cached_tokens"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                // M1: `cached_tokens > input_tokens` is a wire-
                // format pathology — log + skip instead of silently
                // clamping (saturating_sub would yield 0 → fake 1.0
                // hit-rate sample).
                if cached_tokens > input_tokens {
                    tracing::warn!(
                        event = "cache_hit_rate_skipped",
                        request_id = %request_id,
                        provider = "openai_chat",
                        reason = "cached_gt_input",
                        input_tokens = input_tokens,
                        cached_tokens = cached_tokens,
                        "skipping proxy_cache_hit_rate_per_session: cached_tokens > prompt_tokens \
                         (wire-format pathology; clamping would synthesise a bad sample)"
                    );
                } else {
                    // OpenAI's `prompt_tokens` already INCLUDES cached
                    // tokens (per Chat Completions API docs), so the
                    // denominator is `prompt_tokens`, not the sum. The
                    // numerator is `cached_tokens`; `input_tokens` arg
                    // to `compute_cache_hit_rate` carries the
                    // *non-cached* portion (denom-only), so we
                    // synthesise that here.
                    let non_cached = input_tokens - cached_tokens;
                    match crate::observability::compute_cache_hit_rate(non_cached, cached_tokens, 0)
                    {
                        Some(rate) => {
                            crate::observability::observe_cache_hit_rate(
                                crate::observability::cache_hit_rate_provider::OPENAI_CHAT,
                                &request_id,
                                rate,
                            );
                        }
                        None => {
                            tracing::debug!(
                                event = "cache_hit_rate_skipped",
                                request_id = %request_id,
                                provider = "openai_chat",
                                reason = "zero_denominator",
                                "skipping proxy_cache_hit_rate_per_session: no input tokens"
                            );
                        }
                    }
                }
            } else {
                tracing::debug!(
                    event = "cache_hit_rate_skipped",
                    request_id = %request_id,
                    provider = "openai_chat",
                    reason = "no_usage_chunk",
                    "skipping proxy_cache_hit_rate_per_session: stream_options.include_usage=false"
                );
            }
            tracing::info!(
                request_id = %request_id,
                provider = "openai_chat",
                choices = state.choices.len(),
                has_usage = state.usage.is_some(),
                "sse stream closed"
            );
        }
        SseStreamKind::OpenAiResponses => {
            let mut state = crate::sse::openai_responses::ResponseState::new();
            while let Some(chunk) = rx.recv().await {
                framer.push(&chunk);
                while let Some(ev_result) = framer.next_event() {
                    match ev_result {
                        Ok(ev) => {
                            if let Err(e) = state.apply(ev) {
                                tracing::warn!(
                                    request_id = %request_id,
                                    error = %e,
                                    "sse openai_responses state-machine apply error"
                                );
                            }
                        }
                        Err(e) => {
                            tracing::warn!(
                                request_id = %request_id,
                                error = %e,
                                "sse framer error"
                            );
                        }
                    }
                }
            }
            // Phase G PR-G3 + H2: cache hit rate + service_tier +
            // response status emit ONLY when the stream reached a
            // terminal status (`response.completed/failed/incomplete`).
            // Mid-stream client disconnects close the channel without
            // a terminal — `terminal_status().is_none()` then guards
            // emit so we don't observe garbage samples.
            //
            // The Responses API uses `input_tokens` /
            // `cached_input_tokens` shape (Responses-specific —
            // distinct from Chat Completions' `prompt_tokens`).
            let stream_completed = state.terminal_status().is_some();
            if stream_completed {
                if let Some(usage) = &state.usage {
                    let input_tokens = usage
                        .get("input_tokens")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    let cached_tokens = usage
                        .get("input_tokens_details")
                        .and_then(|d| d.get("cached_tokens"))
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    // M1: a cached count greater than input is a
                    // wire-format pathology — usage shouldn't have
                    // `cached > input` for OpenAI Responses. Per
                    // "no silent fallbacks", log + skip the emit
                    // instead of silently clamping.
                    if cached_tokens > input_tokens {
                        tracing::warn!(
                            event = "cache_hit_rate_skipped",
                            request_id = %request_id,
                            provider = "openai_responses",
                            reason = "cached_gt_input",
                            input_tokens = input_tokens,
                            cached_tokens = cached_tokens,
                            "skipping proxy_cache_hit_rate_per_session: cached_tokens > input_tokens \
                             (wire-format pathology; clamping would synthesise a bad sample)"
                        );
                    } else {
                        // Like Chat, `input_tokens` already INCLUDES cached
                        // tokens, so split for the helper.
                        let non_cached = input_tokens - cached_tokens;
                        match crate::observability::compute_cache_hit_rate(
                            non_cached,
                            cached_tokens,
                            0,
                        ) {
                            Some(rate) => {
                                crate::observability::observe_cache_hit_rate(
                                    crate::observability::cache_hit_rate_provider::OPENAI_RESPONSES,
                                    &request_id,
                                    rate,
                                );
                            }
                            None => {
                                tracing::debug!(
                                    event = "cache_hit_rate_skipped",
                                    request_id = %request_id,
                                    provider = "openai_responses",
                                    reason = "zero_denominator",
                                    "skipping proxy_cache_hit_rate_per_session: no input tokens"
                                );
                            }
                        }
                    }
                }
            } else {
                tracing::debug!(
                    event = "cache_hit_rate_skipped",
                    request_id = %request_id,
                    provider = "openai_responses",
                    reason = "stream_did_not_complete",
                    "skipping proxy_cache_hit_rate_per_session: no terminal status seen"
                );
            }
            // Service tier + status are sourced from
            // `state.last_response_envelope` populated by the
            // ResponseState on `response.completed/failed/incomplete`.
            //
            // C1 fix: the tier value comes from the upstream response
            // body; even though the upstream is more trustworthy than
            // a client-side header, an unrecognised value would still
            // grow the metric vector unboundedly. We bucket through
            // the same validator the request-side handler uses.
            if let Some(tier) = state.service_tier.as_deref() {
                let bucketed = crate::observability::metric_names::service_tier::validate(tier);
                crate::observability::record_service_tier(bucketed, &request_id);
            }
            if let Some(status) = state.terminal_status() {
                crate::observability::record_response_status(
                    status,
                    state.incomplete_reason.as_deref(),
                    &request_id,
                );
            }
            tracing::info!(
                request_id = %request_id,
                provider = "openai_responses",
                items = state.items.len(),
                has_usage = state.usage.is_some(),
                service_tier = state.service_tier.as_deref().unwrap_or(""),
                terminal_status = state.terminal_status().unwrap_or(""),
                incomplete_reason = state.incomplete_reason.as_deref().unwrap_or(""),
                "sse stream closed"
            );
        }
        SseStreamKind::None => {}
    }
}

fn ensure_request_id(headers: &HeaderMap) -> String {
    headers
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string())
}

/// Test-only helper: drain a body to bytes (uses BodyExt).
#[cfg(test)]
pub async fn body_to_bytes(body: Body) -> Result<Bytes, axum::Error> {
    use axum::Error;
    body.collect()
        .await
        .map(|c| c.to_bytes())
        .map_err(Error::new)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_build_basic() {
        let base: url::Url = "http://up:8080".parse().unwrap();
        let uri: Uri = "/v1/messages?stream=true".parse().unwrap();
        let out = build_upstream_url(&base, &uri).unwrap();
        assert_eq!(out.as_str(), "http://up:8080/v1/messages?stream=true");
    }

    #[test]
    fn url_build_with_base_path() {
        let base: url::Url = "http://up:8080/api".parse().unwrap();
        let uri: Uri = "/v1/messages".parse().unwrap();
        let out = build_upstream_url(&base, &uri).unwrap();
        assert_eq!(out.as_str(), "http://up:8080/api/v1/messages");
    }

    #[test]
    fn url_build_root() {
        let base: url::Url = "http://up:8080/".parse().unwrap();
        let uri: Uri = "/".parse().unwrap();
        let out = build_upstream_url(&base, &uri).unwrap();
        assert_eq!(out.as_str(), "http://up:8080/");
    }
}
