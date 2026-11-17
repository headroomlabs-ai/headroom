//! `/metrics` loopback-gate integration coverage.
//!
//! `--metrics-require-loopback` (`Config::metrics_require_loopback`)
//! restricts the Prometheus scrape endpoint to loopback peers as
//! defense-in-depth for non-loopback binds. The shared harness serves
//! the app over a real `127.0.0.1` listener with
//! `ConnectInfo<SocketAddr>`, so a reqwest client here is itself a
//! loopback peer: with the gate ON the scrape must STILL succeed (200),
//! proving the middleware is wired correctly and never breaks the
//! legitimate local-scrape path. The non-loopback rejection (403) path
//! can't be driven from a loopback-only test client; it is enforced by
//! the middleware's `peer.ip().is_loopback()` check in
//! `crate::proxy::require_metrics_loopback`.

mod common;

use common::start_proxy_with;

/// Gate ON: a loopback client is still allowed through, so the scrape
/// returns 200 with the usual Prometheus descriptor lines.
#[tokio::test]
async fn metrics_gate_allows_loopback_scrape() {
    // Upstream is never contacted — `/metrics` is served locally.
    let proxy = start_proxy_with("http://127.0.0.1:1", |cfg| {
        cfg.metrics_require_loopback = true;
    })
    .await;

    let resp = reqwest::Client::new()
        .get(format!("{}/metrics", proxy.url()))
        .send()
        .await
        .expect("metrics scrape");

    assert_eq!(
        resp.status(),
        200,
        "loopback client must still pass the metrics loopback gate"
    );
    let body = resp.text().await.unwrap();
    assert!(
        body.contains("# HELP") || body.contains("# TYPE"),
        "scrape body should carry Prometheus metric descriptors"
    );
}

/// Gate OFF (default): unchanged behaviour — the scrape is served.
#[tokio::test]
async fn metrics_default_open_serves_scrape() {
    let proxy = start_proxy_with("http://127.0.0.1:1", |_| {}).await;

    let resp = reqwest::Client::new()
        .get(format!("{}/metrics", proxy.url()))
        .send()
        .await
        .expect("metrics scrape");

    assert_eq!(resp.status(), 200);
}
