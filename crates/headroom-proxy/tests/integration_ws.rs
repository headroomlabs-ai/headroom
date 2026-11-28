//! WebSocket proxy: bidirectional pump + close propagation.

mod common;

use std::net::SocketAddr;
use std::time::Duration;

use common::start_proxy;
use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::protocol::CloseFrame;
use tokio_tungstenite::tungstenite::Message;

/// Spawns an upstream WS echo server. Handshake uses tungstenite over a raw TCP listener.
async fn echo_upstream() -> (SocketAddr, tokio::sync::oneshot::Sender<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (stop_tx, mut stop_rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = &mut stop_rx => break,
                accepted = listener.accept() => {
                    let Ok((stream, _)) = accepted else { continue };
                    tokio::spawn(async move {
                        let Ok(ws) = tokio_tungstenite::accept_async(stream).await else { return };
                        let (mut sink, mut src) = ws.split();
                        while let Some(Ok(msg)) = src.next().await {
                            match msg {
                                Message::Close(cf) => {
                                    let _ = sink.send(Message::Close(cf)).await;
                                    break;
                                }
                                m => {
                                    if sink.send(m).await.is_err() { break; }
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
async fn ws_text_and_binary_round_trip() {
    let (upstream_addr, _stop) = echo_upstream().await;
    let proxy = start_proxy(&format!("http://{upstream_addr}")).await;

    let url = format!("{}/ws", proxy.ws_url());
    let (mut ws, _) = tokio_tungstenite::connect_async(&url).await.unwrap();

    for i in 0..5 {
        let m = format!("hello-{i}");
        ws.send(Message::Text(m.clone().into())).await.unwrap();
        let echoed = ws.next().await.unwrap().unwrap();
        match echoed {
            Message::Text(t) => assert_eq!(t.as_str(), m),
            other => panic!("expected text, got {other:?}"),
        }
    }
    for i in 0..5u8 {
        let m: Vec<u8> = (0..32u8).map(|b| b ^ i).collect();
        ws.send(Message::Binary(m.clone().into())).await.unwrap();
        let echoed = ws.next().await.unwrap().unwrap();
        match echoed {
            Message::Binary(b) => assert_eq!(b.to_vec(), m),
            other => panic!("expected binary, got {other:?}"),
        }
    }
    ws.send(Message::Close(None)).await.unwrap();
    proxy.shutdown().await;
}

#[tokio::test]
async fn ws_client_close_propagates() {
    let (upstream_addr, _stop) = echo_upstream().await;
    let proxy = start_proxy(&format!("http://{upstream_addr}")).await;

    let (mut ws, _) = tokio_tungstenite::connect_async(format!("{}/ws", proxy.ws_url()))
        .await
        .unwrap();

    ws.send(Message::Close(Some(CloseFrame {
        code: tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode::Normal,
        reason: "bye".into(),
    })))
    .await
    .unwrap();

    // Server-side echo will reflect the close; we should see a Close back.
    let got = tokio::time::timeout(Duration::from_secs(3), ws.next()).await;
    assert!(got.is_ok(), "expected close echo within 3s");
    proxy.shutdown().await;
}

/// Upstream that accepts one WebSocket handshake and reports its headers.
// tungstenite's handshake callback returns its own (large) error response type.
#[allow(clippy::result_large_err)]
async fn header_capturing_upstream() -> (SocketAddr, tokio::sync::oneshot::Receiver<http::HeaderMap>)
{
    use tokio_tungstenite::tungstenite::handshake::server::{Request, Response};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut tx = Some(tx);
        let callback = |req: &Request, resp: Response| {
            if let Some(tx) = tx.take() {
                let _ = tx.send(req.headers().clone());
            }
            Ok(resp)
        };
        if let Ok(mut ws) = tokio_tungstenite::accept_hdr_async(stream, callback).await {
            let _ = ws.close(None).await;
        }
    });
    (addr, rx)
}

/// Headers a `--python-upstream` upstream saw for one WebSocket handshake
/// through a front with a token, from a client sending `extra` headers.
async fn relayed_ws_handshake(extra: &[(&str, &str)]) -> http::HeaderMap {
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;
    let (upstream_addr, seen) = header_capturing_upstream().await;
    let proxy = common::start_proxy_with(&format!("http://{upstream_addr}"), |c| {
        c.proxy_token = headroom_proxy::inbound_auth::ProxyToken::new("ws-token");
        c.python_upstream = true;
    })
    .await;
    let mut req = format!("{}/v1/live", proxy.ws_url())
        .into_client_request()
        .unwrap();
    for (k, v) in extra {
        req.headers_mut().insert(
            http::HeaderName::from_bytes(k.as_bytes()).unwrap(),
            http::HeaderValue::from_str(v).unwrap(),
        );
    }
    let _ = tokio_tungstenite::connect_async(req).await;
    let headers = tokio::time::timeout(Duration::from_secs(5), seen)
        .await
        .unwrap()
        .unwrap();
    proxy.shutdown().await;
    headers
}

#[tokio::test]
async fn ws_relay_for_a_remote_caller_carries_the_marker_and_proof() {
    // A loopback peer naming the front by a public Host with the token is a
    // remote caller relayed by a same-host reverse proxy.
    let seen = relayed_ws_handshake(&[
        ("host", "proxy.example:8787"),
        ("x-headroom-proxy-token", "ws-token"),
        ("x-headroom-front-remote", "0"),
    ])
    .await;
    let expected = headroom_proxy::inbound_auth::ProxyToken::new("ws-token")
        .unwrap()
        .relay_proof();
    assert_eq!(seen.get("x-headroom-front-remote").unwrap(), "1");
    assert_eq!(
        seen.get("x-headroom-front-proof")
            .unwrap()
            .to_str()
            .unwrap(),
        expected
    );
    assert!(seen.get("x-headroom-proxy-token").is_none());
}

#[tokio::test]
async fn ws_relay_for_a_local_caller_carries_no_marker() {
    let seen = relayed_ws_handshake(&[("x-headroom-front-remote", "1")]).await;
    assert!(seen.get("x-headroom-front-remote").is_none());
    assert!(seen.get("x-headroom-front-proof").is_none());
}
