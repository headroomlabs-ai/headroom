//! Native Vertex AI publisher path — Phase D PR-D4.
//!
//! # What this module owns
//!
//! Vertex AI exposes Anthropic models via two POST verbs on a single
//! template:
//!
//! ```text
//! POST /v1beta1/projects/{project}/locations/{location}/publishers/anthropic/models/{model}:rawPredict
//! POST /v1beta1/projects/{project}/locations/{location}/publishers/anthropic/models/{model}:streamRawPredict
//! ```
//!
//! Crucially, the **request body** is the Anthropic Messages envelope
//! with two surface differences from the Anthropic API:
//!
//! 1. The body carries an `anthropic_version` field (e.g.
//!    `"vertex-2023-10-16"`) and **no** `model` field — the model id
//!    travels in the URL path instead.
//! 2. Auth is GCP ADC (Application Default Credentials) → a short-lived
//!    bearer token in `Authorization: Bearer <jwt>`. There is no
//!    Anthropic API key in scope.
//!
//! Everything else — `messages`, `system`, `tools`, `tool_choice`,
//! `cache_control`, `thinking`, `metadata.user_id`, `stop_sequences`,
//! `stream: true` for the streaming verb — round-trips byte-equal
//! with the live-zone compression pass running on top.
//!
//! # Why a native module (vs LiteLLM)
//!
//! The Python LiteLLM converter at `headroom/backends/litellm.py:486-628`
//! lossy-converts Anthropic ↔ OpenAI shapes for Vertex, dropping
//! `thinking`, `redacted_thinking`, `document`, `search_result`,
//! `image`, `server_tool_use`, `mcp_tool_use` block kinds. The bug
//! tracker entry P4-38 (mirrored from P4-37 for Bedrock) flags this;
//! PR-D4 retires it on the Rust side.
//!
//! The native module:
//!
//! - Buffers the request body up to `compression_max_body_bytes`.
//! - Parses the envelope (no `model` field check, `anthropic_version`
//!   present) to confirm we're on the Vertex publisher shape.
//! - Routes to the **same** live-zone Anthropic dispatcher as
//!   `/v1/messages` — the body shape is identical apart from
//!   `anthropic_version` vs `model`. We feed the dispatcher a body
//!   with a synthetic `model` set from the path's model id (so block
//!   metadata lookups work) and re-emit the body without the
//!   synthetic field on the way out.
//! - Resolves the ADC bearer token via [`adc::TokenSource`], cached
//!   with a refresh window so back-to-back requests don't pay the
//!   round-trip.
//! - Forwards to the configured Vertex endpoint
//!   (`https://{region}-aiplatform.googleapis.com/...`) with the
//!   bearer attached.
//!
//! # Module layout
//!
//! - [`adc`] — `TokenSource` trait, plus `GcpAdcTokenSource`
//!   (production) and `StaticTokenSource` (tests). Caching and
//!   refresh-ahead-of-expiry live here.
//! - [`raw_predict`] — POST handler for the non-streaming verb.
//! - [`stream_raw_predict`] — POST handler for the streaming verb.
//!   Vertex uses **SSE** for streaming (unlike Bedrock's binary
//!   EventStream — much simpler), so the existing PR-C1
//!   [`crate::sse::anthropic::AnthropicStreamState`] state machine
//!   drives telemetry directly.
//! - [`envelope`] — minimal envelope parser; same shape as the future
//!   PR-D1 Bedrock envelope module by design (the two PRs are running
//!   in parallel and will reconcile at merge).
//!
//! # Routing
//!
//! Vertex's path uses a **colon-suffix verb** (`:rawPredict`) that is
//! awkward in axum's parameter syntax (where `:name` reserves `:` as
//! the parameter sigil). We register the routes with a single
//! parameter capturing the entire trailing segment and split on the
//! last `:` inside the handler — no regex, just `str::rsplit_once`.
//! See [`split_model_action`].

pub mod adc;
pub mod envelope;
pub mod raw_predict;
pub mod stream_raw_predict;

pub use adc::{StaticTokenSource, TokenSource, TokenSourceError};
pub use envelope::{ParsedEnvelope, VertexEnvelopeError};

use axum::body::Body;
use axum::extract::{ConnectInfo, Path, State};
use axum::http::{HeaderMap, Method, StatusCode, Uri};
use axum::response::Response;
use std::net::SocketAddr;

use crate::proxy::AppState;

/// Single axum handler mounted at the
/// `/v1beta1/projects/{project}/locations/{location}/publishers/anthropic/models/{model_action}`
/// path. The trailing `model_action` segment carries `<model>:<verb>`
/// (the verb is the colon-suffix Vertex appends). We split on the
/// last `:` and dispatch to either [`raw_predict::handle_raw_predict`]
/// (logically) or [`stream_raw_predict::handle_stream_raw_predict`].
///
/// The two sub-handlers share most of their logic — see
/// [`raw_predict::forward_vertex_request`] — so the dispatch is a
/// single argument flip (`attach_sse_tee`). Routing the two verbs
/// through one axum handler keeps the path matcher unambiguous: matchit
/// can't distinguish two patterns that share the literal `model_action`
/// parameter shape.
pub async fn handle_vertex_predict_dispatch(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    Path((project, location, model_action)): Path<(String, String, String)>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Body,
) -> Response {
    let request_id = headers
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    let (model_id, verb_str) = match split_model_action(&model_action) {
        Some(parts) => parts,
        None => {
            tracing::warn!(
                event = "vertex_path_parse_failed",
                request_id = %request_id,
                path = %uri.path(),
                segment = %model_action,
                "vertex path final segment missing `:verb` separator"
            );
            return Response::builder()
                .status(StatusCode::NOT_FOUND)
                .body(Body::from("vertex path: bad model_action"))
                .expect("static");
        }
    };

    let verb = match VertexVerb::parse(verb_str) {
        Some(v) => v,
        None => {
            tracing::warn!(
                event = "vertex_unknown_verb",
                request_id = %request_id,
                verb = %verb_str,
                "vertex path verb not recognized; only rawPredict / streamRawPredict are supported"
            );
            return Response::builder()
                .status(StatusCode::NOT_FOUND)
                .body(Body::from("vertex: unknown verb"))
                .expect("static");
        }
    };

    let attach_sse_tee = matches!(verb, VertexVerb::StreamRawPredict);
    if attach_sse_tee {
        tracing::info!(
            event = "vertex_streaming_pipeline_active",
            request_id = %request_id,
            method = %method,
            path = %uri.path(),
            framer = "byte_level_sse",
            state_machine = "anthropic",
            "vertex streaming pipeline engaged: SSE framer + AnthropicStreamState telemetry tee"
        );
    }

    raw_predict::forward_vertex_request(
        state,
        client_addr,
        request_id,
        method,
        uri,
        headers,
        body,
        raw_predict::VertexCallContext {
            project,
            location,
            model_id: model_id.to_string(),
            verb,
        },
        attach_sse_tee,
    )
    .await
}

/// Split the trailing `:model_action` path segment into
/// `(model_id, verb)`.
///
/// Vertex's path looks like
/// `.../models/claude-3-5-sonnet@20240620:rawPredict`. The model id may
/// itself contain `@` and other special characters, but Vertex never
/// puts a literal `:` inside a model id — the `:` is the verb
/// separator. We match on the **last** `:` for safety.
///
/// Returns `None` when the segment carries no colon (unknown shape;
/// the handler logs and 404s).
pub fn split_model_action(segment: &str) -> Option<(&str, &str)> {
    segment.rsplit_once(':')
}

/// Recognized Vertex publisher verbs. Future verbs (e.g. `:countTokens`)
/// would extend this enum. The handler maps unknown verbs to
/// `event = "vertex_unknown_verb"` warn logs and 404s — never a silent
/// fallback to a "default" verb.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VertexVerb {
    /// Non-streaming Anthropic Messages call.
    RawPredict,
    /// SSE-streaming Anthropic Messages call.
    StreamRawPredict,
}

impl VertexVerb {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "rawPredict" => Some(Self::RawPredict),
            "streamRawPredict" => Some(Self::StreamRawPredict),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::RawPredict => "rawPredict",
            Self::StreamRawPredict => "streamRawPredict",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_model_action_basic() {
        assert_eq!(
            split_model_action("claude-3-5-sonnet@20240620:rawPredict"),
            Some(("claude-3-5-sonnet@20240620", "rawPredict"))
        );
        assert_eq!(
            split_model_action("claude-3-haiku@20240307:streamRawPredict"),
            Some(("claude-3-haiku@20240307", "streamRawPredict"))
        );
    }

    #[test]
    fn split_model_action_no_colon_returns_none() {
        assert_eq!(split_model_action("claude-3-5-sonnet"), None);
    }

    #[test]
    fn split_model_action_uses_last_colon() {
        // Defensive: even if the model id were to contain a colon
        // (Vertex doesn't emit such ids today), the verb is whatever
        // follows the LAST colon.
        assert_eq!(
            split_model_action("weird:model:rawPredict"),
            Some(("weird:model", "rawPredict"))
        );
    }

    #[test]
    fn vertex_verb_parse() {
        assert_eq!(
            VertexVerb::parse("rawPredict"),
            Some(VertexVerb::RawPredict)
        );
        assert_eq!(
            VertexVerb::parse("streamRawPredict"),
            Some(VertexVerb::StreamRawPredict)
        );
        assert_eq!(VertexVerb::parse("predict"), None);
        assert_eq!(VertexVerb::parse(""), None);
    }
}
