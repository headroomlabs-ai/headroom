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
        ws.send(Message::Text(m.clone())).await.unwrap();
        let echoed = ws.next().await.unwrap().unwrap();
        match echoed {
            Message::Text(t) => assert_eq!(t.as_str(), m),
            other => panic!("expected text, got {other:?}"),
        }
    }
    for i in 0..5u8 {
        let m: Vec<u8> = (0..32u8).map(|b| b ^ i).collect();
        ws.send(Message::Binary(m.clone())).await.unwrap();
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
