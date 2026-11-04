//! `wss://` relay: an `https://` upstream means the WebSocket pump has to
//! complete a TLS handshake. This binary links two rustls crypto providers
//! (ring via reqwest, aws-lc-rs via the AWS SDK) and rustls panics on
//! `ClientConfig::builder()` when no process-level default is installed —
//! which used to kill every `wss://` relay from inside the pump task, on any
//! client's request. The other ws tests all use plaintext `ws://`; this file
//! is the only one that drives the TLS path end to end.
//!
//! The upstream certificate is generated per test process (rcgen) and
//! handed to the proxy through `HEADROOM_CA_BUNDLE`, the same knob an
//! operator uses for a corporate TLS-inspection root.

mod common;

use std::net::SocketAddr;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use common::start_proxy;
use futures_util::{SinkExt, StreamExt};
use rustls_pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use tokio::sync::oneshot;
use tokio_tungstenite::tungstenite::Message;

struct TestCert {
    cert: CertificateDer<'static>,
    cert_pem: String,
    key_pkcs8: Vec<u8>,
}

fn self_signed_for_loopback() -> TestCert {
    let ck =
        rcgen::generate_simple_self_signed(vec!["localhost".to_string(), "127.0.0.1".to_string()])
            .expect("self-signed test certificate");
    TestCert {
        cert: ck.cert.der().clone(),
        cert_pem: ck.cert.pem(),
        key_pkcs8: ck.signing_key.serialize_der(),
    }
}

/// The one certificate the proxy trusts in this process. The proxy reads
/// `HEADROOM_CA_BUNDLE` once (at `AppState::new` for HTTP, lazily for the
/// first `wss://` connect), so the variable is set here, before any proxy
/// starts, and never changed again.
fn trusted_cert() -> &'static TestCert {
    static TRUSTED: OnceLock<TestCert> = OnceLock::new();
    TRUSTED.get_or_init(|| {
        let cert = self_signed_for_loopback();
        let dir = std::env::temp_dir().join(format!("headroom-ws-tls-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let bundle = dir.join("upstream-ca.pem");
        std::fs::write(&bundle, &cert.cert_pem).expect("write bundle");
        std::env::set_var("HEADROOM_CA_BUNDLE", &bundle);
        cert
    })
}

/// A TLS WebSocket echo server on 127.0.0.1, presenting `cert`.
async fn tls_echo_upstream(cert: &TestCert) -> (SocketAddr, oneshot::Sender<()>) {
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let config = rustls::ServerConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .expect("ring supports the default TLS versions")
        .with_no_client_auth()
        .with_single_cert(
            vec![cert.cert.clone()],
            PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(cert.key_pkcs8.clone())),
        )
        .expect("server config");
    let acceptor = tokio_rustls::TlsAcceptor::from(Arc::new(config));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (stop_tx, mut stop_rx) = oneshot::channel();
    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = &mut stop_rx => break,
                accepted = listener.accept() => {
                    let Ok((stream, _)) = accepted else { continue };
                    let acceptor = acceptor.clone();
                    tokio::spawn(async move {
                        let Ok(tls) = acceptor.accept(stream).await else { return };
                        let Ok(ws) = tokio_tungstenite::accept_async(tls).await else { return };
                        let (mut sink, mut src) = ws.split();
                        while let Some(Ok(msg)) = src.next().await {
                            match msg {
                                Message::Close(cf) => {
                                    let _ = sink.send(Message::Close(cf)).await;
                                    break;
                                }
                                m => {
                                    if sink.send(m).await.is_err() {
                                        break;
                                    }
                                }
                            }
                        }
                    });
                }
            }
        }
    });
    (addr, stop_tx)
}

#[tokio::test]
async fn wss_upstream_relays_frames_over_tls() {
    let (upstream, _stop) = tls_echo_upstream(trusted_cert()).await;
    let proxy = start_proxy(&format!("https://127.0.0.1:{}", upstream.port())).await;

    let (mut ws, _) = tokio_tungstenite::connect_async(format!("{}/v1/realtime", proxy.ws_url()))
        .await
        .expect("client handshake with the proxy");

    ws.send(Message::Text("over-tls".into())).await.unwrap();
    let echoed = tokio::time::timeout(Duration::from_secs(10), ws.next())
        .await
        .expect("echo within 10s: the wss:// pump must not die on connect")
        .expect("stream still open")
        .expect("frame, not a transport error");
    match echoed {
        Message::Text(t) => assert_eq!(t.as_str(), "over-tls"),
        other => panic!("expected the echoed text frame, got {other:?}"),
    }

    ws.send(Message::Close(None)).await.ok();
    proxy.shutdown().await;
}

#[tokio::test]
async fn wss_upstream_with_untrusted_cert_fails_closed_and_proxy_survives() {
    // Make sure the trusted bundle is installed first, so this test's
    // untrusted certificate is genuinely untrusted rather than "no bundle".
    let _ = trusted_cert();
    let untrusted = self_signed_for_loopback();
    let (upstream, _stop) = tls_echo_upstream(&untrusted).await;
    let proxy = start_proxy(&format!("https://127.0.0.1:{}", upstream.port())).await;

    let (mut ws, _) = tokio_tungstenite::connect_async(format!("{}/v1/realtime", proxy.ws_url()))
        .await
        .expect("client handshake with the proxy");
    ws.send(Message::Text("should-not-arrive".into()))
        .await
        .ok();

    // The upstream handshake must fail verification and the relay must end
    // cleanly: no echo, no hang, no panic.
    let outcome = tokio::time::timeout(Duration::from_secs(10), ws.next())
        .await
        .expect("relay must end within 10s when the upstream cert is untrusted");
    match outcome {
        Some(Ok(Message::Text(t))) => panic!("frame relayed to an untrusted upstream: {t}"),
        Some(Ok(Message::Binary(_))) => panic!("frame relayed to an untrusted upstream"),
        _ => {}
    }

    // The proxy is still serving: the failure stayed inside the one relay.
    let health = reqwest::get(format!("{}/healthz", proxy.url()))
        .await
        .unwrap();
    assert_eq!(health.status(), 200);
    proxy.shutdown().await;
}
