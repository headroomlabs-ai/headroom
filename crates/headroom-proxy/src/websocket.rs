//! WebSocket reverse-proxy handler.
//!
//! Accepts a client upgrade via axum, opens a tungstenite connection to the
//! upstream (rewriting scheme http->ws / https->wss), and bidirectionally
//! pumps messages until either side closes.

use std::net::SocketAddr;

use axum::body::Body;
use axum::extract::ws::{CloseFrame, Message as AxMsg, WebSocket, WebSocketUpgrade};
use axum::http::{HeaderName, HeaderValue, Request, Response, StatusCode};
use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::protocol::CloseFrame as TgCloseFrame;
use tokio_tungstenite::tungstenite::Message as TgMsg;

use crate::headers::build_forward_request_headers;
use crate::proxy::{join_upstream_path, AppState};

/// Entry point invoked from the catch-all when an upgrade is detected.
pub async fn ws_handler(
    ws: WebSocketUpgrade,
    state: AppState,
    client_addr: SocketAddr,
    req: Request<Body>,
) -> Response<Body> {
    let request_id = req
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    // Build the upstream WS URL.
    let upstream_url = match build_upstream_ws_url(&state.config.upstream, req.uri()) {
        Ok(u) => u,
        Err(e) => {
            tracing::warn!(error = %e, "failed to build upstream ws url");
            return (StatusCode::BAD_GATEWAY, e).into_response_body();
        }
    };

    // Build forwarded headers (drop hop-by-hop EXCEPT Upgrade/Connection — but
    // tungstenite generates its own; we forward only the user-meaningful ones
    // such as Authorization, Sec-WebSocket-Protocol, etc.).
    let forwarded_host = req
        .headers()
        .get(http::header::HOST)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    // PR-A5: same strip policy as the HTTP path — operators flip both
    // simultaneously via the single `Config::strip_internal_headers` knob.
    let strip_internal_ws = state.config.strip_internal_headers.is_enabled();
    let forward_headers = build_forward_request_headers(
        req.headers(),
        client_addr.ip(),
        "http",
        forwarded_host.as_deref(),
        &request_id,
        strip_internal_ws,
    );
    // Sec-WebSocket-Protocol must be propagated for subprotocol negotiation.
    let subprotocols: Option<String> = req
        .headers()
        .get("sec-websocket-protocol")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    let path = req.uri().path().to_string();
    ws.on_upgrade(move |client_ws| async move {
        if let Err(e) = run_ws_pump(
            client_ws,
            upstream_url,
            forward_headers,
            subprotocols,
            request_id,
            path,
        )
        .await
        {
            tracing::warn!(error = %e, "websocket pump ended with error");
        }
    })
}

trait IntoResponseBody {
    fn into_response_body(self) -> Response<Body>;
}
impl IntoResponseBody for (StatusCode, String) {
    fn into_response_body(self) -> Response<Body> {
        Response::builder()
            .status(self.0)
            .body(Body::from(self.1))
            .unwrap()
    }
}

fn build_upstream_ws_url(base: &url::Url, req_uri: &http::Uri) -> Result<url::Url, String> {
    let mut joined = base.clone();
    let new_scheme = match joined.scheme() {
        "http" => "ws",
        "https" => "wss",
        "ws" | "wss" => "ws", // already WS; set_scheme is a no-op but keeps it uniform
        other => return Err(format!("unsupported upstream scheme: {other}")),
    };
    joined
        .set_scheme(new_scheme)
        .map_err(|()| "failed to set ws scheme".to_string())?;
    Ok(join_upstream_path(&joined, req_uri.path(), req_uri.query()))
}

async fn run_ws_pump(
    client_ws: WebSocket,
    upstream_url: url::Url,
    forward_headers: http::HeaderMap,
    subprotocols: Option<String>,
    request_id: String,
    path: String,
) -> Result<(), String> {
    // Build the upstream handshake request manually so we can inject headers.
    let mut req = upstream_url
        .as_str()
        .into_client_request()
        .map_err(|e| format!("ws into_client_request: {e}"))?;
    {
        let h = req.headers_mut();
        // Tungstenite will set Host, Upgrade, Connection, Sec-WebSocket-Key,
        // Sec-WebSocket-Version itself. We add user-meaningful pass-throughs.
        for (name, value) in forward_headers.iter() {
            // Skip headers tungstenite manages.
            let n = name.as_str().to_ascii_lowercase();
            if matches!(
                n.as_str(),
                "host"
                    | "upgrade"
                    | "connection"
                    | "sec-websocket-key"
                    | "sec-websocket-version"
                    | "sec-websocket-extensions"
                    | "content-length"
            ) {
                continue;
            }
            h.append(name, value.clone());
        }
        if let Some(sp) = subprotocols {
            if let Ok(v) = HeaderValue::from_str(&sp) {
                h.insert(HeaderName::from_static("sec-websocket-protocol"), v);
            }
        }
    }

    let (upstream_ws, _resp) = tokio_tungstenite::connect_async(req)
        .await
        .map_err(|e| format!("upstream ws connect: {e}"))?;

    let (mut upstream_sink, mut upstream_stream) = upstream_ws.split();
    let (mut client_sink, mut client_stream) = client_ws.split();

    tracing::info!(
        request_id = %request_id,
        path = %path,
        protocol = "ws",
        upstream = %upstream_url,
        "ws session opened"
    );

    // Each direction runs in its own task. We use a cancel token so that when
    // either side closes/errors, the other is aborted immediately rather than
    // blocking forever on next().await (the half-close hang bug).
    let cancel = tokio_util::sync::CancellationToken::new();
    let cancel_c2u = cancel.clone();
    let cancel_u2c = cancel.clone();

    // Pump client -> upstream.
    let c2u = tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = cancel_c2u.cancelled() => break,
                msg = client_stream.next() => {
                    let Some(msg) = msg else { break };
                    let m = match msg { Ok(m) => m, Err(_) => break };
                    let tg = match ax_to_tg(m) { Some(tg) => tg, None => continue };
                    let close = matches!(tg, TgMsg::Close(_));
                    if upstream_sink.send(tg).await.is_err() { break; }
                    if close { break; }
                }
            }
        }
        let _ = upstream_sink.close().await;
        cancel_c2u.cancel();
    });

    // Pump upstream -> client.
    let u2c = tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = cancel_u2c.cancelled() => break,
                msg = upstream_stream.next() => {
                    let Some(msg) = msg else { break };
                    let m = match msg { Ok(m) => m, Err(_) => break };
                    let ax = match tg_to_ax(m) { Some(ax) => ax, None => continue };
                    let close = matches!(ax, AxMsg::Close(_));
                    if client_sink.send(ax).await.is_err() { break; }
                    if close { break; }
                }
            }
        }
        let _ = client_sink.close().await;
        cancel_u2c.cancel();
    });

    let _ = tokio::join!(c2u, u2c);
    tracing::info!(request_id = %request_id, path = %path, protocol = "ws", "ws session closed");
    Ok(())
}

fn ax_to_tg(m: AxMsg) -> Option<TgMsg> {
    Some(match m {
        AxMsg::Text(t) => TgMsg::Text(t.to_string()),
        AxMsg::Binary(b) => TgMsg::Binary(b.to_vec()),
        AxMsg::Ping(p) => TgMsg::Ping(p.to_vec()),
        AxMsg::Pong(p) => TgMsg::Pong(p.to_vec()),
        AxMsg::Close(Some(cf)) => TgMsg::Close(Some(TgCloseFrame {
            code: tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode::from(cf.code),
            reason: cf.reason.to_string().into(),
        })),
        AxMsg::Close(None) => TgMsg::Close(None),
    })
}

fn tg_to_ax(m: TgMsg) -> Option<AxMsg> {
    Some(match m {
        TgMsg::Text(t) => AxMsg::Text(t.as_str().to_string()),
        TgMsg::Binary(b) => AxMsg::Binary(b.to_vec()),
        TgMsg::Ping(p) => AxMsg::Ping(p.to_vec()),
        TgMsg::Pong(p) => AxMsg::Pong(p.to_vec()),
        TgMsg::Close(Some(cf)) => AxMsg::Close(Some(CloseFrame {
            code: cf.code.into(),
            reason: cf.reason.to_string().into(),
        })),
        TgMsg::Close(None) => AxMsg::Close(None),
        TgMsg::Frame(_) => return None,
    })
}
