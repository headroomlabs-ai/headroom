//! Conversations API (`/v1/conversations*`) — Phase C PR-C4.
//!
//! # Why explicit handlers?
//!
//! OpenAI's Conversations API is the stateful "thread" surface
//! sitting alongside the Responses API. The client creates a
//! conversation, attaches items (messages, tool calls, tool outputs)
//! to it, and references the conversation ID on subsequent
//! `/v1/responses` requests. The wire shape is:
//!
//!   POST   /v1/conversations                      — create
//!   GET    /v1/conversations/{id}                 — read
//!   POST   /v1/conversations/{id}                 — update (e.g. metadata)
//!   DELETE /v1/conversations/{id}                 — delete
//!   POST   /v1/conversations/{id}/items           — append item(s)
//!   GET    /v1/conversations/{id}/items           — list items
//!   GET    /v1/conversations/{id}/items/{item_id} — read one item
//!   DELETE /v1/conversations/{id}/items/{item_id} — delete one item
//!
//! For PR-C4 every handler is **passthrough-with-instrumentation**:
//! we forward upstream byte-equal and emit a structured-log event
//! (`event = "conversations_passthrough_pr_c4"`) carrying the
//! request_id, method, path-shape, and (for path-templated routes)
//! the extracted IDs. Compression for stored conversation items is
//! C5+/B-phase territory — explicitly out of scope here.
//!
//! # Why explicit routes (not a regex / catch-all)?
//!
//! Per the realignment build constraints we forbid regex routing.
//! Each handler binds to an exact axum path matcher
//! (`/v1/conversations/:id/items/:item_id`, etc.). Path params are
//! extracted via `axum::extract::Path` so they round-trip into
//! structured logs without string-splitting.
//!
//! # Streaming bodies
//!
//! Conversation `items` payloads can be multi-MB (long histories).
//! These handlers do NOT buffer the body — they accept
//! `Request<Body>` directly and hand off to
//! [`crate::proxy::forward_http`], which streams the body to upstream
//! via `reqwest::Body::wrap_stream`. The compression gate inside
//! `forward_http` does not match `/v1/conversations*`
//! (see [`crate::compression::is_compressible_path`]) so no buffering
//! ever happens.
//!
//! # Structured-log shape
//!
//! Every handler emits exactly one `event = "conversations_passthrough_pr_c4"`
//! info-level log per request, BEFORE forwarding (so a stalled
//! upstream is still observable). On forward error we surface the
//! upstream error verbatim — no swallowing, per project
//! no-silent-fallbacks rule.

use axum::body::Body;
use axum::extract::{ConnectInfo, Path, State};
use axum::http::Request;
use axum::response::Response;
use std::net::SocketAddr;

use crate::proxy::{forward_http, AppState};

/// Common forwarding tail shared by every conversations handler.
/// Logs the breadcrumb, then defers to `forward_http`. Kept inline
/// (rather than #[axum::debug_handler]-decorated wrappers) so the
/// per-route handlers stay one obvious function each.
async fn forward_conversations(
    state: AppState,
    client_addr: SocketAddr,
    req: Request<Body>,
    route: &'static str,
    conversation_id: Option<&str>,
    item_id: Option<&str>,
) -> Response {
    let method = req.method().clone();
    let path = req.uri().path().to_string();

    // PR-C4: structured-log breadcrumb. We log BEFORE forwarding so a
    // stalled / failed upstream call still leaves a trace pointing
    // at this code path.
    tracing::info!(
        event = "conversations_passthrough_pr_c4",
        method = %method,
        path = %path,
        route = route,
        conversation_id = conversation_id.unwrap_or(""),
        item_id = item_id.unwrap_or(""),
        passthrough_only = true,
        compression_in_scope = false,
        "conversations request: passthrough with instrumentation (compression deferred to C5+)"
    );

    forward_http(state, client_addr, req)
        .await
        .unwrap_or_else(|e| {
            use axum::response::IntoResponse;
            // No silent fallback: surface the upstream error verbatim.
            // The structured `tracing::warn!` emitted by
            // `ProxyError::into_response` carries the original cause.
            e.into_response()
        })
}

/// `POST /v1/conversations` — create a new conversation.
pub async fn handle_conversations_create(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response {
    forward_conversations(state, client_addr, req, "conversations.create", None, None).await
}

/// `GET /v1/conversations/{conversation_id}` — read a conversation.
pub async fn handle_conversations_get(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    Path(conversation_id): Path<String>,
    req: Request<Body>,
) -> Response {
    forward_conversations(
        state,
        client_addr,
        req,
        "conversations.get",
        Some(&conversation_id),
        None,
    )
    .await
}

/// `POST /v1/conversations/{conversation_id}` — update conversation
/// metadata (e.g. tags). Same shape as create.
pub async fn handle_conversations_update(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    Path(conversation_id): Path<String>,
    req: Request<Body>,
) -> Response {
    forward_conversations(
        state,
        client_addr,
        req,
        "conversations.update",
        Some(&conversation_id),
        None,
    )
    .await
}

/// `DELETE /v1/conversations/{conversation_id}` — delete a conversation.
pub async fn handle_conversations_delete(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    Path(conversation_id): Path<String>,
    req: Request<Body>,
) -> Response {
    forward_conversations(
        state,
        client_addr,
        req,
        "conversations.delete",
        Some(&conversation_id),
        None,
    )
    .await
}

/// `POST /v1/conversations/{conversation_id}/items` — append items.
/// Body is streamed to upstream — never buffered (histories can be
/// multi-MB).
pub async fn handle_conversations_items_create(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    Path(conversation_id): Path<String>,
    req: Request<Body>,
) -> Response {
    forward_conversations(
        state,
        client_addr,
        req,
        "conversations.items.create",
        Some(&conversation_id),
        None,
    )
    .await
}

/// `GET /v1/conversations/{conversation_id}/items` — list items.
pub async fn handle_conversations_items_list(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    Path(conversation_id): Path<String>,
    req: Request<Body>,
) -> Response {
    forward_conversations(
        state,
        client_addr,
        req,
        "conversations.items.list",
        Some(&conversation_id),
        None,
    )
    .await
}

/// `GET /v1/conversations/{conversation_id}/items/{item_id}` —
/// read one item.
pub async fn handle_conversations_item_get(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    Path((conversation_id, item_id)): Path<(String, String)>,
    req: Request<Body>,
) -> Response {
    forward_conversations(
        state,
        client_addr,
        req,
        "conversations.items.get",
        Some(&conversation_id),
        Some(&item_id),
    )
    .await
}

/// `DELETE /v1/conversations/{conversation_id}/items/{item_id}` —
/// delete one item.
pub async fn handle_conversations_item_delete(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    Path((conversation_id, item_id)): Path<(String, String)>,
    req: Request<Body>,
) -> Response {
    forward_conversations(
        state,
        client_addr,
        req,
        "conversations.items.delete",
        Some(&conversation_id),
        Some(&item_id),
    )
    .await
}
