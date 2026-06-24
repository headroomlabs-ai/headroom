//! SSE chunk fidelity: events stream through with timing preserved.

mod common;

use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use common::start_proxy;
use futures_util::StreamExt;
use http_body_util::StreamBody;
use hyper::body::Frame;
use hyper::service::service_fn;
use hyper::{Request, Response};
use hyper_util::rt::TokioIo;
use tokio::sync::Notify;

async fn sse_upstream(on_disconnect: Arc<Notify>) -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                break;
            };
            let on_disconnect = on_disconnect.clone();
            tokio::spawn(async move {
                let io = TokioIo::new(stream);
                let _ = hyper::server::conn::http1::Builder::new()
                    .serve_connection(
                        io,
                        service_fn(move |_req: Request<hyper::body::Incoming>| {
                            let on_disconnect = on_disconnect.clone();
                            async move {
                                let (tx, rx) = tokio::sync::mpsc::channel::<
                                    Result<Frame<Bytes>, std::io::Error>,
                                >(4);
                                tokio::spawn(async move {
                                    for i in 0..10u32 {
                                        let payload = format!("data: event-{i}\n\n");
                                        if tx
                                            .send(Ok(Frame::data(Bytes::from(payload))))
                                            .await
                                            .is_err()
                                        {
                                            // Client disconnected — notify the test.
                                            on_disconnect.notify_one();
                                            return;
                                        }
                                        tokio::time::sleep(Duration::from_millis(50)).await;
                                    }
                                });
                                let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
                                let body = StreamBody::new(stream);
                                Ok::<_, Infallible>(
                                    Response::builder()
                                        .status(200)
                                        .header("content-type", "text/event-stream")
                                        .header("cache-control", "no-cache")
                                        .body(body)
                                        .unwrap(),
                                )
                            }
                        }),
                    )
                    .await;
            });
        }
    });
    (addr, task)
}

#[tokio::test]
async fn sse_chunks_arrive_with_preserved_timing() {
    let on_disconnect = Arc::new(Notify::new());
    let (addr, _server) = sse_upstream(on_disconnect.clone()).await;
    let proxy = start_proxy(&format!("http://{addr}")).await;

    let resp = reqwest::Client::new()
        .get(format!("{}/sse", proxy.url()))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.headers().get("content-type").unwrap(),
        "text/event-stream"
    );
    let mut stream = resp.bytes_stream();

    let mut events = Vec::new();
    let mut last = Instant::now();
    let mut max_gap = Duration::ZERO;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        let now = Instant::now();
        let gap = now.duration_since(last);
        last = now;
        // SSE chunks aren't always 1:1 with events on slow boxes — accumulate
        // and split per `\n\n`.
        let s = String::from_utf8_lossy(&chunk).to_string();
        events.push(s);
        if events.len() > 1 && gap > max_gap {
            max_gap = gap;
        }
    }
    let combined = events.join("");
    let parsed: Vec<&str> = combined.split("\n\n").filter(|s| !s.is_empty()).collect();
    assert_eq!(parsed.len(), 10, "got events: {parsed:?}");
    for (i, ev) in parsed.iter().enumerate() {
        assert_eq!(ev.trim(), format!("data: event-{i}"));
    }
    // Loose CI bound — chunks should not be buffered until end. Each event was
    // ~50ms apart, so the longest inter-chunk gap should be well under 500ms.
    assert!(
        max_gap < Duration::from_millis(500),
        "max chunk gap {max_gap:?} suggests buffering"
    );

    proxy.shutdown().await;
}

#[tokio::test]
async fn client_disconnect_propagates_to_upstream() {
    let on_disconnect = Arc::new(Notify::new());
    let (addr, _server) = sse_upstream(on_disconnect.clone()).await;
    let proxy = start_proxy(&format!("http://{addr}")).await;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{}/sse", proxy.url()))
        .send()
        .await
        .unwrap();
    let mut stream = resp.bytes_stream();
    // Read the first chunk, then drop the stream to disconnect.
    let _ = stream.next().await;
    drop(stream);

    // Upstream should observe disconnect within 1s.
    let observed = tokio::time::timeout(Duration::from_secs(2), on_disconnect.notified())
        .await
        .is_ok();
    assert!(observed, "upstream did not see client disconnect within 2s");
    proxy.shutdown().await;
}
