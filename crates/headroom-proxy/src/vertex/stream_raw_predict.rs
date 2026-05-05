//! Vertex `:streamRawPredict` handler module.
//!
//! Path:
//! ```text
//! POST /v1beta1/projects/{project}/locations/{location}/publishers/anthropic/models/{model}:streamRawPredict
//! ```
//!
//! Vertex's streaming verb returns the same `text/event-stream`
//! payload Anthropic Messages emits (Vertex does not use the binary
//! EventStream Bedrock does). The existing PR-C1
//! [`crate::sse::anthropic::AnthropicStreamState`] state machine
//! works unchanged — bytes are teed into a bounded mpsc + spawned
//! parser, identical to the `/v1/messages` SSE pipeline in
//! [`crate::proxy::forward_http`].
//!
//! # Why this module is thin
//!
//! Both `:rawPredict` and `:streamRawPredict` share the same axum
//! route shape (matchit can't distinguish two patterns where the verb
//! is part of a captured parameter). The shared dispatcher in
//! [`crate::vertex::handle_vertex_predict_dispatch`] splits the verb
//! and flips a single `attach_sse_tee` flag. This module exists so
//! the file structure mirrors the spec (PR-D4 calls for distinct
//! `raw_predict.rs` and `stream_raw_predict.rs` files); the
//! streaming-specific behaviour is the SSE tee in
//! [`crate::vertex::raw_predict::forward_vertex_request`] when
//! `attach_sse_tee == true`.
//!
//! # Streaming behaviour invariants
//!
//! - **Request body**: same envelope detection + live-zone Anthropic
//!   compression as `:rawPredict`. The `stream: true` flag (when
//!   present) is preserved byte-equal — the live-zone dispatcher
//!   never rewrites top-level fields.
//! - **Response body**: SSE bytes flow back to the client unchanged.
//!   The state-machine tee runs in a spawned task and can never
//!   block the byte path (bounded `try_send` channel).
//! - **Telemetry**: per-stream summary log emitted at stream close
//!   (`event = "vertex_sse_stream_closed"`) with token counts and
//!   block count.
//!
//! See [`crate::vertex::raw_predict`] for the actual forwarding
//! implementation.

// Re-export the shared dispatcher so callers that want to address the
// streaming verb explicitly have a name in this module. The
// dispatcher is the same single-axum-route entry point for both
// verbs — see module-level rationale.
pub use crate::vertex::handle_vertex_predict_dispatch as handle_stream_raw_predict;
