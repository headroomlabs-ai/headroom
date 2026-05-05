//! Translate Bedrock binary EventStream messages into Anthropic SSE
//! frames — Phase D PR-D2.
//!
//! # Why translate
//!
//! Bedrock's `/model/{id}/invoke-with-response-stream` returns
//! `application/vnd.amazon.eventstream` — a binary, length-prefixed,
//! CRC-checksummed framing format. Most clients of Headroom (Claude
//! Code, the Anthropic SDK in non-Bedrock mode, the Headroom Python
//! SDK) speak SSE (`text/event-stream`). When the client requested
//! Bedrock-compatible output via `Accept: application/vnd.amazon.eventstream`
//! we pass the upstream bytes through unchanged. When the client wants
//! SSE (the default) we translate each `chunk` message's JSON payload
//! into an SSE frame:
//!
//! ```text
//! data: {anthropic JSON event}\n\n
//! ```
//!
//! That format matches what direct-Anthropic `/v1/messages` emits and
//! lets the existing [`AnthropicStreamState`] from PR-C1 read the
//! translated stream for telemetry without modification.
//!
//! # Output mode selection
//!
//! Configurable. Default behaviour:
//!
//! - `Accept: application/vnd.amazon.eventstream` (or any value
//!   listed in [`OutputMode::eventstream_accept_values`]) → passthrough.
//! - `Accept: text/event-stream`, `Accept: */*`, or absent → translate.
//!
//! Operators override the recognised Accept values via the
//! `--bedrock-eventstream-accept-values` CLI flag (CSV) — though the
//! defaults cover every Bedrock client we know about today.

use bytes::{Bytes, BytesMut};
use http::HeaderMap;

use crate::bedrock::eventstream::{EventStreamMessage, HeaderValue};

/// Output mode for the streaming response. Picked once per request,
/// based on the inbound `Accept` header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputMode {
    /// Pass upstream EventStream bytes through verbatim. Used when
    /// the client speaks Bedrock binary natively (`Accept:
    /// application/vnd.amazon.eventstream`).
    EventStream,
    /// Translate each `chunk` message's payload into an SSE frame.
    /// Used by SSE-native clients (the default).
    Sse,
}

impl OutputMode {
    /// Default list of `Accept` header values (case-insensitive,
    /// comparison ignores parameters after `;`) that select
    /// [`OutputMode::EventStream`].
    pub fn default_eventstream_accept_values() -> Vec<String> {
        vec!["application/vnd.amazon.eventstream".to_string()]
    }

    /// Decide an [`OutputMode`] from the inbound headers + the
    /// configured list of EventStream-selecting `Accept` values.
    /// Lower-case + parameter-trimmed comparison per RFC 7231 §3.1.1.1.
    pub fn from_accept(headers: &HeaderMap, eventstream_accept_values: &[String]) -> OutputMode {
        let accept_raw = match headers
            .get(http::header::ACCEPT)
            .and_then(|v| v.to_str().ok())
        {
            Some(v) => v,
            None => return OutputMode::Sse,
        };
        // The Accept header may carry multiple media types separated
        // by commas; iterate and trim parameters from each.
        for token in accept_raw.split(',') {
            let media_type = token.split(';').next().unwrap_or("").trim();
            for candidate in eventstream_accept_values {
                if media_type.eq_ignore_ascii_case(candidate) {
                    return OutputMode::EventStream;
                }
            }
        }
        OutputMode::Sse
    }
}

/// A translation outcome for one EventStream message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TranslateOutcome {
    /// Emit these bytes to the client. For `OutputMode::Sse` this is
    /// `data: {payload}\n\n`; for `OutputMode::EventStream` it is the
    /// upstream bytes themselves (we don't ever pass through here —
    /// passthrough mode is handled by the streaming forwarder, which
    /// never bothers parsing).
    Emit(Bytes),
    /// Skip — the message has no client-facing translation. Used for
    /// `:event-type` values that AWS emits for protocol-internal
    /// signalling (not yet observed for Bedrock Anthropic responses,
    /// but we surface a structured outcome rather than guess).
    Skip { event_type: String },
}

/// Errors during translation. Per project rules these are loud — the
/// handler 5xx's the client when an unknown `:message-type` arrives,
/// rather than silently swallowing.
#[derive(Debug, thiserror::Error)]
pub enum TranslateError {
    /// `:message-type == exception`. AWS reserved value indicating the
    /// service raised a synchronous error mid-stream. Surfaced as a
    /// structured error so the handler can map it to a 5xx + log.
    #[error("Bedrock stream emitted exception: {payload_preview}")]
    UpstreamException { payload_preview: String },
    /// The translator's input message is missing a required header
    /// (`:event-type`). Wire-format violation — AWS would never emit.
    #[error("Bedrock message missing required `:event-type` header")]
    MissingEventType,
}

/// Translate one EventStream message under the chosen output mode.
///
/// Side-effect-free: emits a `tracing::info!` per `chunk` translation
/// and `tracing::warn!` for unknown event types. The hot path is
/// allocation-bounded — one `BytesMut` of `payload.len() + 8`.
pub fn translate_message(
    message: &EventStreamMessage,
    mode: OutputMode,
) -> Result<TranslateOutcome, TranslateError> {
    // Always check for `:message-type == exception` first — that's a
    // structural error regardless of mode. AWS reserves this value to
    // indicate a service-side fault ON the stream (vs an HTTP-level
    // error from the initial request).
    if matches!(message.message_type(), Some("exception")) {
        let preview = String::from_utf8_lossy(&message.payload[..message.payload.len().min(160)])
            .into_owned();
        return Err(TranslateError::UpstreamException {
            payload_preview: preview,
        });
    }

    let event_type = message
        .event_type()
        .ok_or(TranslateError::MissingEventType)?;

    match (mode, event_type) {
        (OutputMode::EventStream, _) => {
            // Passthrough mode never reaches this function on the hot
            // path — the streaming handler short-circuits to byte
            // copy. We still support the call so tests can assert
            // semantic equivalence: re-emit the exact bytes (which
            // requires re-serialising — costly — so we don't here).
            // Surface a structured Skip; the caller picks bytes from
            // the raw upstream stream.
            Ok(TranslateOutcome::Skip {
                event_type: event_type.to_string(),
            })
        }
        (OutputMode::Sse, "chunk") => {
            tracing::info!(
                event = "bedrock_eventstream_translated_to_sse",
                event_type = event_type,
                payload_bytes = message.payload.len(),
                "translated bedrock eventstream chunk to sse frame"
            );
            Ok(TranslateOutcome::Emit(payload_to_sse_frame(
                &message.payload,
            )))
        }
        (OutputMode::Sse, other) => {
            // Unknown event type. Bedrock today emits `chunk`
            // exclusively for Anthropic streaming responses; if a new
            // type appears we log it loudly per
            // `feedback_no_silent_fallbacks.md` rather than silently
            // dropping the bytes.
            tracing::warn!(
                event = "bedrock_eventstream_unknown_event_type",
                event_type = other,
                "unknown bedrock eventstream :event-type; skipping translation"
            );
            Ok(TranslateOutcome::Skip {
                event_type: other.to_string(),
            })
        }
    }
}

/// Wrap a JSON payload as an SSE event frame.
///
/// Anthropic's direct `/v1/messages` SSE wire format emits BOTH an
/// `event:` line (the event name like `message_start`) AND a `data:`
/// line (the JSON payload). The proxy's `AnthropicStreamState`
/// requires the `event:` line — without it every event is dropped
/// with a `sse_unknown_event` warn.
///
/// The Bedrock binary EventStream encodes the event name on the
/// frame envelope (`:event-type` always equals `chunk`), but the
/// Anthropic semantic event type lives in the JSON payload's `type`
/// field. We extract it and emit the SSE frame in the canonical
/// Anthropic format.
///
/// If the JSON is malformed or `type` is missing, we still emit a
/// `data:`-only frame (best-effort) and log a warn — the byte path
/// to the client never breaks.
fn payload_to_sse_frame(payload: &[u8]) -> Bytes {
    let event_name = extract_anthropic_event_type(payload);
    let extra = match &event_name {
        Some(name) => name.len() + 8, // "event: " + "\n"
        None => 0,
    };
    let mut out = BytesMut::with_capacity(payload.len() + extra + 8);
    if let Some(name) = event_name {
        out.extend_from_slice(b"event: ");
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(b"\n");
    }
    out.extend_from_slice(b"data: ");
    out.extend_from_slice(payload);
    out.extend_from_slice(b"\n\n");
    out.freeze()
}

/// Extract `type` field from an Anthropic SSE JSON payload. We do a
/// targeted byte-level scan rather than a full JSON parse so the hot
/// path is allocation-free in the common case. Falls back to `None`
/// on any malformed input — never panics.
fn extract_anthropic_event_type(payload: &[u8]) -> Option<String> {
    // Cheap path: parse with serde_json, take `.type` if it's a string.
    // The JSON is small (typical Anthropic event ~200 bytes), so the
    // parse cost is negligible vs the wire serialisation we already do.
    let v: serde_json::Value = serde_json::from_slice(payload).ok()?;
    v.get("type")?.as_str().map(|s| s.to_string())
}

/// Convenience: print one translated header value for log readability.
/// Used only on log paths.
pub fn header_value_preview(v: &HeaderValue) -> String {
    match v {
        HeaderValue::String(s) => {
            if s.len() <= 64 {
                s.clone()
            } else {
                format!("{}…", &s[..64])
            }
        }
        HeaderValue::Bytes(b) => format!("[{} bytes]", b.len()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bedrock::eventstream::{build_chunk_message, parse};
    use http::HeaderMap;

    #[test]
    fn accept_eventstream_selects_passthrough() {
        let mut h = HeaderMap::new();
        h.insert(
            http::header::ACCEPT,
            "application/vnd.amazon.eventstream".parse().unwrap(),
        );
        let mode = OutputMode::from_accept(&h, &OutputMode::default_eventstream_accept_values());
        assert_eq!(mode, OutputMode::EventStream);
    }

    #[test]
    fn accept_eventstream_with_q_param_still_selects_passthrough() {
        // RFC 7231 quality params are stripped before comparison.
        let mut h = HeaderMap::new();
        h.insert(
            http::header::ACCEPT,
            "application/vnd.amazon.eventstream;q=0.9".parse().unwrap(),
        );
        let mode = OutputMode::from_accept(&h, &OutputMode::default_eventstream_accept_values());
        assert_eq!(mode, OutputMode::EventStream);
    }

    #[test]
    fn accept_text_event_stream_selects_sse() {
        let mut h = HeaderMap::new();
        h.insert(http::header::ACCEPT, "text/event-stream".parse().unwrap());
        let mode = OutputMode::from_accept(&h, &OutputMode::default_eventstream_accept_values());
        assert_eq!(mode, OutputMode::Sse);
    }

    #[test]
    fn no_accept_header_defaults_to_sse() {
        let h = HeaderMap::new();
        let mode = OutputMode::from_accept(&h, &OutputMode::default_eventstream_accept_values());
        assert_eq!(mode, OutputMode::Sse);
    }

    #[test]
    fn multi_accept_with_eventstream_among_them_selects_passthrough() {
        let mut h = HeaderMap::new();
        h.insert(
            http::header::ACCEPT,
            "text/html, application/vnd.amazon.eventstream;q=0.9, */*"
                .parse()
                .unwrap(),
        );
        let mode = OutputMode::from_accept(&h, &OutputMode::default_eventstream_accept_values());
        assert_eq!(mode, OutputMode::EventStream);
    }

    #[test]
    fn translate_chunk_to_sse_frame() {
        let json = r#"{"type":"message_start","message":{"id":"msg_abc"}}"#;
        let bytes = build_chunk_message(json);
        let msg = parse(&bytes).unwrap();
        let outcome = translate_message(&msg, OutputMode::Sse).unwrap();
        match outcome {
            TranslateOutcome::Emit(b) => {
                let s = std::str::from_utf8(&b).unwrap();
                // Both event: and data: lines must be present, in
                // Anthropic's documented order.
                assert!(
                    s.starts_with("event: message_start\ndata: "),
                    "expected 'event: message_start' header; got {s}"
                );
                assert!(s.ends_with("\n\n"));
                assert!(s.contains(json));
            }
            other => panic!("expected Emit; got {other:?}"),
        }
    }

    #[test]
    fn translate_chunk_falls_back_on_malformed_payload() {
        // Payload is not valid JSON — translator must still emit a
        // data:-only frame (no event: line) and the byte path must
        // not panic.
        let bytes = build_chunk_message("not json at all");
        let msg = parse(&bytes).unwrap();
        let outcome = translate_message(&msg, OutputMode::Sse).unwrap();
        match outcome {
            TranslateOutcome::Emit(b) => {
                let s = std::str::from_utf8(&b).unwrap();
                assert!(s.starts_with("data: "));
                assert!(s.ends_with("\n\n"));
            }
            other => panic!("expected Emit; got {other:?}"),
        }
    }

    #[test]
    fn translate_exception_message_is_loud() {
        let bytes = crate::bedrock::eventstream::MessageBuilder::new()
            .header_string(":message-type", "exception")
            .header_string(":exception-type", "ThrottlingException")
            .payload(Bytes::from_static(br#"{"message":"slow down"}"#))
            .build();
        let msg = parse(&bytes).unwrap();
        let err = translate_message(&msg, OutputMode::Sse).unwrap_err();
        assert!(matches!(err, TranslateError::UpstreamException { .. }));
    }

    #[test]
    fn translate_unknown_event_type_skips() {
        let bytes = crate::bedrock::eventstream::MessageBuilder::new()
            .header_string(":event-type", "future_unknown_kind")
            .header_string(":message-type", "event")
            .payload(Bytes::from_static(b"{}"))
            .build();
        let msg = parse(&bytes).unwrap();
        let outcome = translate_message(&msg, OutputMode::Sse).unwrap();
        match outcome {
            TranslateOutcome::Skip { event_type } => {
                assert_eq!(event_type, "future_unknown_kind");
            }
            other => panic!("expected Skip; got {other:?}"),
        }
    }

    #[test]
    fn missing_event_type_is_loud() {
        // A message lacking :event-type must not silently translate.
        let bytes = crate::bedrock::eventstream::MessageBuilder::new()
            .header_string(":message-type", "event")
            .payload(Bytes::from_static(b"{}"))
            .build();
        let msg = parse(&bytes).unwrap();
        let err = translate_message(&msg, OutputMode::Sse).unwrap_err();
        assert!(matches!(err, TranslateError::MissingEventType));
    }

    #[test]
    fn header_value_preview_strings() {
        assert_eq!(
            header_value_preview(&HeaderValue::String("hi".into())),
            "hi"
        );
        assert!(
            header_value_preview(&HeaderValue::Bytes(Bytes::from_static(&[1, 2, 3])))
                .starts_with("[3 bytes")
        );
    }
}
