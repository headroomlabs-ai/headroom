//! HTTP surface for the native savings ledger — Phase H:
//! `GET /stats`, `GET /stats/timeseries`, `GET /stats/events`, and
//! the embedded `GET /dashboard` page.
//!
//! All handlers are read-only snapshots over
//! [`super::ledger::Ledger`] — no handler can mutate accounting
//! state. Mounted by [`crate::proxy::build_app`] when
//! `Config::stats` is explicitly enabled; when off (the default), the paths fall through
//! to the catch-all forwarder like any other route.
//!
//! # Exposure: aggregate tier is open, per-request tier is local-only
//!
//! There is no auth here, and `--listen` defaults to `0.0.0.0:8787`,
//! so the payloads are split by sensitivity the same way the Python
//! proxy splits its own `/stats`:
//!
//! - **Aggregates** (spend, tokens, per-model/provider rollups,
//!   history) are served to anyone, like the Prometheus `/metrics`
//!   endpoint next door — same data class, same stance.
//! - **Per-request rows** (`recent_requests`, all of
//!   `/stats/events`) and the ledger's filesystem path are served
//!   only to local callers ([`caller_is_local`]). A request id plus a
//!   model plus a timestamp is a different tier from a counter, and
//!   `/metrics` never exposes it — so "same as /metrics" would be the
//!   wrong argument for handing it to the open internet.
//!
//! `--stats=false` removes the routes entirely. Making the local-only
//! tier reachable from a trusted reverse proxy is tracked upstream in
//! headroomlabs-ai/headroom#1959, which is defining that policy for
//! the Python dashboard first; this surface should adopt whatever
//! lands there rather than invent a second scheme.

use std::net::{IpAddr, SocketAddr};

use axum::extract::{ConnectInfo, Query, State};
use axum::http::header::HOST;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use serde_json::Value;

use crate::proxy::AppState;

/// True when the caller is genuinely local — gates the per-request
/// tier of `/stats` and the whole of `/stats/events`.
///
/// BOTH conditions are required, mirroring the Python proxy's
/// `loopback_guard`:
///
/// 1. the TCP peer is a loopback address, and
/// 2. the `Host` header names loopback.
///
/// (2) is not redundant with (1) — it is the DNS-rebinding defence. A
/// hostile page can rebind its own domain to `127.0.0.1`, so the
/// victim's browser connects to this proxy *from* loopback and (1)
/// passes, while `Host` still reads the attacker's name and their JS
/// reads the response. A real local tool always sends a loopback
/// `Host`; a rebound request does not.
pub(crate) fn caller_is_local(peer: SocketAddr, headers: &HeaderMap) -> bool {
    peer.ip().is_loopback()
        && headers
            .get(HOST)
            .and_then(|h| h.to_str().ok())
            .is_some_and(host_is_loopback)
}

/// `127.0.0.1`, `127.0.0.1:8787`, `[::1]:8787`, `localhost:8787`, …
fn host_is_loopback(host: &str) -> bool {
    let host = host.trim();
    // A bare IP with no port must be recognised before any port
    // split: `::1` is all colons.
    if let Ok(ip) = host.parse::<IpAddr>() {
        return ip.is_loopback();
    }
    let name = match host.strip_prefix('[') {
        Some(rest) => match rest.split_once(']') {
            Some((inner, _)) => inner,
            None => return false,
        },
        None => host.rsplit_once(':').map_or(host, |(h, _)| h),
    };
    name.eq_ignore_ascii_case("localhost")
        || name.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback())
}

/// The dashboard page, embedded at compile time so the binary is
/// self-contained (no template dir to deploy, no CDN dependency).
const DASHBOARD_HTML: &str = include_str!("dashboard.html");

pub async fn handle_stats(
    State(state): State<AppState>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
) -> Json<Value> {
    Json(state.stats.stats_payload(caller_is_local(peer, &headers)))
}

#[derive(Deserialize)]
pub struct TimeseriesParams {
    /// hour | day | week | month (default: day)
    bucket: Option<String>,
}

pub async fn handle_timeseries(
    State(state): State<AppState>,
    Query(params): Query<TimeseriesParams>,
) -> Response {
    let bucket = params.bucket.as_deref().unwrap_or("day");
    match state.stats.timeseries_payload(bucket) {
        Some(payload) => Json(payload).into_response(),
        None => (
            StatusCode::BAD_REQUEST,
            // The Anthropic-style envelope the Bedrock lanes emit
            // (`bedrock::invoke::error_response`). Note this is NOT
            // universal: `ProxyError` renders plain text, and axum's
            // `Query` extractor rejects a malformed `?bucket=` with
            // its own plain-text 400 before this handler runs — so a
            // client can't assume JSON for every 4xx from this path.
            Json(serde_json::json!({
                "error": {
                    "type": "unknown_bucket",
                    "message": "bucket must be one of: hour, day, week, month",
                }
            })),
        )
            .into_response(),
    }
}

#[derive(Deserialize)]
pub struct EventsParams {
    limit: Option<usize>,
}

/// This endpoint is per-request metadata end to end — there is no
/// aggregate tier to hand a network caller, so the whole thing is
/// gated. 404 rather than 403, matching the Python guard's stance:
/// keep the surface invisible to scanners rather than merely refuse
/// them.
pub async fn handle_events(
    State(state): State<AppState>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    Query(params): Query<EventsParams>,
) -> Response {
    if !caller_is_local(peer, &headers) {
        return (StatusCode::NOT_FOUND, "not found").into_response();
    }
    Json(state.stats.events_payload(params.limit.unwrap_or(50))).into_response()
}

/// `Html` already sets `text/html; charset=utf-8`.
pub async fn handle_dashboard() -> Html<&'static str> {
    Html(DASHBOARD_HTML)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers_with_host(host: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(HOST, host.parse().unwrap());
        h
    }

    fn peer(addr: &str) -> SocketAddr {
        addr.parse().unwrap()
    }

    #[test]
    fn local_caller_needs_loopback_peer_and_loopback_host() {
        for host in [
            "127.0.0.1:8787",
            "127.0.0.1",
            "localhost:8787",
            "LOCALHOST",
            "[::1]:8787",
            "::1",
            "127.5.6.7:8787", // the whole 127/8 block is loopback
        ] {
            assert!(
                caller_is_local(peer("127.0.0.1:5555"), &headers_with_host(host)),
                "{host} should read as local"
            );
        }
    }

    /// The DNS-rebinding case, and the entire reason the `Host` check
    /// exists: the victim's browser really is on loopback, so the peer
    /// check alone would wave this through while attacker JS reads the
    /// per-request feed.
    #[test]
    fn loopback_peer_with_foreign_host_is_not_local() {
        for host in ["attacker.com", "attacker.com:8787", "evil.localhost.dev"] {
            assert!(
                !caller_is_local(peer("127.0.0.1:5555"), &headers_with_host(host)),
                "{host} must not read as local — DNS rebinding"
            );
        }
    }

    #[test]
    fn remote_peer_is_never_local_even_claiming_a_loopback_host() {
        assert!(!caller_is_local(
            peer("10.1.2.3:5555"),
            &headers_with_host("127.0.0.1:8787")
        ));
    }

    #[test]
    fn missing_host_header_fails_closed() {
        assert!(!caller_is_local(peer("127.0.0.1:5555"), &HeaderMap::new()));
    }
}
