//! Inbound `HEADROOM_PROXY_TOKEN` gate.
//!
//! The Rust front connects to the Python proxy from `127.0.0.1`, so
//! Python's own token gate (which exempts loopback peers) cannot see a
//! network caller behind it. These tests pin that the Rust proxy checks
//! the token itself, before any upstream contact, using the same
//! credential rules as the Python gate. The app is driven with
//! `oneshot` and an injected `ConnectInfo` so a non-loopback peer can be
//! simulated; a real loopback-only test listener cannot produce one.

use std::net::SocketAddr;

use axum::body::Body;
use axum::extract::ConnectInfo;
use axum::http::{Request, StatusCode};
use headroom_proxy::inbound_auth::ProxyToken;
use headroom_proxy::{build_app, AppState, Config};
use tower::util::ServiceExt;
use wiremock::matchers::any;
use wiremock::{Mock, MockServer, ResponseTemplate};

const TOKEN: &str = "s3cret-token";
// 203.0.113.0/24 is TEST-NET-3 (RFC 5737): never loopback.
const REMOTE: &str = "203.0.113.7:40000";
const LOCAL: &str = "127.0.0.1:40000";

async fn upstream() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(any())
        .respond_with(ResponseTemplate::new(200).set_body_string("upstream"))
        .mount(&server)
        .await;
    server
}

async fn send(
    upstream: &MockServer,
    token: Option<&str>,
    peer: Option<&str>,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
) -> StatusCode {
    let mut config = Config::for_test(upstream.uri().parse().unwrap());
    config.proxy_token = token.and_then(ProxyToken::new);
    let app = build_app(AppState::new(config).unwrap());
    let mut builder = Request::builder().method(method).uri(path);
    for (k, v) in headers {
        builder = builder.header(*k, *v);
    }
    let mut req = builder.body(Body::empty()).unwrap();
    if let Some(peer) = peer {
        let addr: SocketAddr = peer.parse().unwrap();
        req.extensions_mut().insert(ConnectInfo(addr));
    }
    app.oneshot(req).await.unwrap().status()
}

#[tokio::test]
async fn remote_peer_without_token_gets_401_and_upstream_is_not_contacted() {
    let up = upstream().await;
    let status = send(
        &up,
        Some(TOKEN),
        Some(REMOTE),
        "GET",
        "/admin/upstream",
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert!(up.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn remote_peer_with_wrong_token_gets_401() {
    let up = upstream().await;
    for header in [
        ("x-headroom-proxy-token", "wrong"),
        ("authorization", "Bearer wrong"),
    ] {
        let status = send(
            &up,
            Some(TOKEN),
            Some(REMOTE),
            "POST",
            "/v1/messages",
            &[header],
        )
        .await;
        assert_eq!(status, StatusCode::UNAUTHORIZED, "{header:?}");
    }
    assert!(up.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn remote_peer_with_token_header_is_forwarded_without_it() {
    let up = upstream().await;
    let status = send(
        &up,
        Some(TOKEN),
        Some(REMOTE),
        "GET",
        "/v1/models",
        &[("x-headroom-proxy-token", TOKEN)],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let seen = up.received_requests().await.unwrap();
    assert_eq!(seen.len(), 1);
    assert!(seen[0].headers.get("x-headroom-proxy-token").is_none());
}

#[tokio::test]
async fn remote_peer_with_bearer_token_is_forwarded_without_it() {
    let up = upstream().await;
    let bearer = format!("bearer {TOKEN}");
    let status = send(
        &up,
        Some(TOKEN),
        Some(REMOTE),
        "GET",
        "/v1/models",
        &[("authorization", &bearer)],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let seen = up.received_requests().await.unwrap();
    assert_eq!(seen.len(), 1);
    // The proxy credential is not the provider's business.
    assert!(seen[0].headers.get("authorization").is_none());
}

#[tokio::test]
async fn token_header_wins_over_a_provider_bearer() {
    // Same rule as Python's read_proxy_token: when the explicit header is
    // present, Authorization belongs to the provider and is forwarded.
    let up = upstream().await;
    let status = send(
        &up,
        Some(TOKEN),
        Some(REMOTE),
        "GET",
        "/v1/models",
        &[
            ("x-headroom-proxy-token", TOKEN),
            ("authorization", "Bearer provider-key"),
        ],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let seen = up.received_requests().await.unwrap();
    assert_eq!(
        seen[0].headers.get("authorization").unwrap(),
        "Bearer provider-key"
    );
}

#[tokio::test]
async fn loopback_peer_without_token_is_still_forwarded() {
    let up = upstream().await;
    let status = send(&up, Some(TOKEN), Some(LOCAL), "GET", "/v1/models", &[]).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(up.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn ipv4_mapped_loopback_peer_counts_as_loopback() {
    let up = upstream().await;
    let status = send(
        &up,
        Some(TOKEN),
        Some("[::ffff:127.0.0.1]:40000"),
        "GET",
        "/v1/models",
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn missing_peer_address_fails_closed() {
    let up = upstream().await;
    let status = send(&up, Some(TOKEN), None, "GET", "/v1/models", &[]).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert!(up.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn health_probes_stay_reachable_without_token() {
    let up = upstream().await;
    assert_eq!(
        send(&up, Some(TOKEN), Some(REMOTE), "GET", "/healthz", &[]).await,
        StatusCode::OK
    );
    assert_eq!(
        send(&up, Some(TOKEN), Some(REMOTE), "GET", "/readyz", &[]).await,
        StatusCode::OK
    );
    // Only GET is exempt, as in the Python gate: anything else on a probe
    // path would fall through to the relay.
    assert_eq!(
        send(&up, Some(TOKEN), Some(REMOTE), "POST", "/readyz", &[]).await,
        StatusCode::UNAUTHORIZED
    );
}

#[tokio::test]
async fn no_token_configured_keeps_todays_passthrough() {
    let up = upstream().await;
    let status = send(&up, None, Some(REMOTE), "GET", "/v1/models", &[]).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn hello_probe_is_answered_locally_without_token() {
    let up = upstream().await;
    for method in ["GET", "HEAD"] {
        let status = send(&up, Some(TOKEN), Some(REMOTE), method, "/api/hello", &[]).await;
        assert_eq!(status, StatusCode::OK, "{method}");
    }
    assert!(up.received_requests().await.unwrap().is_empty());
}

/// `Host` the upstream received for one request.
async fn upstream_host(
    token: Option<&str>,
    peer: &str,
    path: &str,
    headers: &[(&str, &str)],
) -> String {
    let up = upstream().await;
    let status = send(&up, token, Some(peer), "GET", path, headers).await;
    assert_eq!(status, StatusCode::OK);
    let seen = up.received_requests().await.unwrap();
    seen[0]
        .headers
        .get("host")
        .unwrap()
        .to_str()
        .unwrap()
        .to_owned()
}

fn is_loopback_name(host: &str) -> bool {
    host.starts_with("127.") || host.starts_with("localhost") || host.starts_with("[::1]")
}

#[tokio::test]
async fn local_caller_keeps_the_rewritten_upstream_host() {
    let host = upstream_host(
        None,
        LOCAL,
        "/admin/upstream",
        &[("host", "127.0.0.1:8787")],
    )
    .await;
    assert!(is_loopback_name(&host), "{host}");
}

#[tokio::test]
async fn loopback_peer_with_a_foreign_host_is_not_relayed_as_local() {
    // A browser reaching the port through a name that resolves to
    // 127.0.0.1: loopback peer, foreign Host.
    let host = upstream_host(
        None,
        LOCAL,
        "/admin/upstream",
        &[("host", "evil.example:8787")],
    )
    .await;
    assert_eq!(host, "headroom-remote.invalid");
}

#[tokio::test]
async fn remote_token_holder_is_not_relayed_as_local() {
    // `[::1%lo]` is a scoped IPv6 literal the Python proxy accepts as
    // loopback; it must not be passed through.
    for client_host in ["proxy.example:8787", "localhost:8787", "[::1%lo]:8787"] {
        let host = upstream_host(
            Some(TOKEN),
            REMOTE,
            "/admin/upstream",
            &[("x-headroom-proxy-token", TOKEN), ("host", client_host)],
        )
        .await;
        assert_eq!(host, "headroom-remote.invalid", "{client_host}");
    }
}
