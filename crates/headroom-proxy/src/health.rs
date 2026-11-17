//! Health endpoints. These are intercepted by Rust and never forwarded.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

use crate::proxy::AppState;

/// Own health: 200 if the proxy process is up.
pub async fn healthz() -> impl IntoResponse {
    Json(json!({ "ok": true, "service": "headroom-proxy" }))
}

/// Effective rollout state of this running Rust proxy process.
pub async fn rollout_status(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(state.config.rollout.to_value())
}

/// Upstream health: GETs upstream `/healthz`. Returns 200 when reachable +
/// 2xx, 503 otherwise. The endpoint name is reserved by the proxy and is
/// not forwarded; operators must not name a real upstream route this.
pub async fn healthz_upstream(State(state): State<AppState>) -> Response {
    // Use an absolute path so upstream URLs with non-trailing-slash paths
    // (e.g. http://localhost:8788/api) resolve to /healthz, not replace the
    // last segment. Url::join("healthz") would strip "api" per RFC 3986.
    let mut url = state.config.upstream.clone();
    url.set_path("/healthz");
    url.set_query(None);
    match state.client.get(url).send().await {
        Ok(resp) if resp.status().is_success() => {
            (StatusCode::OK, Json(json!({"ok": true}))).into_response()
        }
        Ok(resp) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"ok": false, "upstream_status": resp.status().as_u16()})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"ok": false, "error": e.to_string()})),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Config;

    #[tokio::test]
    async fn rollout_status_exposes_running_snapshot() {
        let state = AppState::new(Config::for_test("http://127.0.0.1:9".parse().unwrap())).unwrap();
        let expected = state.config.rollout.snapshot_digest();
        let Json(payload) = rollout_status(State(state)).await;
        assert_eq!(payload["snapshot_digest"], expected);
        assert_eq!(payload["qualification_eligible"], true);
    }
}
