//! POST `/v1/chat/completions` handler — Phase C PR-C2.
//!
//! # Why an explicit handler?
//!
//! Most paths flow through `forward_http` via the catch-all fallback;
//! the path gate in `forward_http` runs the per-provider live-zone
//! dispatcher (added to the gate by PR-C2). Spec PR-C2 still mandates
//! an explicit route handler for `/v1/chat/completions` so future
//! Phase-D wiring (Bedrock OpenAI-shape, Vertex), Phase-E auth-mode
//! gating, and per-endpoint rate-limit shaping have an obvious
//! attachment point.
//!
//! # What this handler does
//!
//! 1. Pre-buffers the request body (Bytes) so we can inspect
//!    `n`, `stream`, `messages`, `tool_choice`, `stream_options`
//!    before forwarding.
//! 2. Reconstructs a `Request<Body>` from the buffered bytes plus
//!    the original method, URI, and headers.
//! 3. Hands off to [`crate::proxy::forward_http`] — the same single
//!    forwarder that the catch-all uses. The compression gate inside
//!    `forward_http` re-classifies the path and runs
//!    [`crate::compression::compress_openai_chat_request`].
//!
//! Re-using `forward_http` keeps the SSE state-machine wiring
//! (PR-C1), header-stripping (PR-A5), `x-headroom-*` policy, and
//! request-id plumbing single-source. The alternative — duplicating
//! the forwarder body inside this handler — would diverge over time.
//!
//! # Skip / passthrough behaviours surfaced here
//!
//! - **`n > 1`** — multiple completions imply non-determinism.
//!   `compression::should_skip_compression` (called from the gate
//!   inside `forward_http`) returns `NGreaterThanOne(n)` and the
//!   gate skips dispatch entirely. The handler does not need to
//!   touch the body.
//! - **`stream: true`** — handled by the existing SSE state-machine
//!   tee in `forward_http` (PR-C1's `ChunkState`).
//! - **`tool_choice` change** — never read, never mutated.
//!   `tools[]` definitions live in the cache hot zone and the
//!   live-zone dispatcher only walks `messages[*].content`.
//! - **`stream_options.include_usage`** — same. Round-trips byte-equal
//!   as a side effect of byte-range surgery in the dispatcher.

use axum::body::Body;
use axum::extract::{ConnectInfo, State};
use axum::http::{HeaderMap, Method, Request, Uri};
use axum::response::Response;
use bytes::Bytes;
use std::net::SocketAddr;

use crate::proxy::{forward_http, AppState};

/// Axum POST handler for `/v1/chat/completions`. Buffers the body,
/// stitches a fresh `Request<Body>` together, and forwards via
/// [`forward_http`]. Compression dispatch + SSE telemetry is handled
/// inside `forward_http`'s shared gate (PR-C1 + PR-C2).
pub async fn handle_chat_completions(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    // Reconstruct the Request<Body> shape forward_http expects.
    // Cloning the headers into a fresh builder keeps the original
    // method/uri/version intact. `axum::body::Body::from(Bytes)` is
    // a single-shot stream, which is exactly what the buffered
    // compression branch wants.
    let mut builder = Request::builder().method(method).uri(uri);
    if let Some(hs) = builder.headers_mut() {
        *hs = headers;
    }
    let req = match builder.body(Body::from(body)) {
        Ok(r) => r,
        Err(e) => {
            // Building the request out of pieces we already have
            // shouldn't fail; if it does it's an internal bug. Don't
            // silently swallow — log loudly and 500.
            tracing::error!(
                event = "handler_error",
                handler = "chat_completions",
                error = %e,
                "failed to reconstruct request from buffered body"
            );
            return Response::builder()
                .status(http::StatusCode::INTERNAL_SERVER_ERROR)
                .body(Body::from("internal handler error"))
                .expect("static response");
        }
    };

    forward_http(state, client_addr, req)
        .await
        .unwrap_or_else(|e| {
            use axum::response::IntoResponse;
            e.into_response()
        })
}
