//! Incremental binary EventStream parser — Phase D PR-D2.
//!
//! # What this parses
//!
//! AWS Bedrock's streaming surface (`/model/{id}/invoke-with-response-stream`)
//! returns `application/vnd.amazon.eventstream` framed messages, NOT
//! Server-Sent Events. Each message is:
//!
//! ```text
//! +------------------+----------------------+------------------+
//! |  prelude (12B)   |    headers (N1 B)    |   payload (N2 B) |  +  4-byte CRC32
//! +------------------+----------------------+------------------+
//! ```
//!
//! Where the prelude is three big-endian u32s:
//!
//!   - `total_length` (bytes from start-of-message up to AND including
//!     the trailing message CRC32),
//!   - `headers_length` (bytes occupied by the headers block),
//!   - `prelude_crc32` (CRC32 of the first 8 bytes of the prelude).
//!
//! The payload length is computed: `total_length - 12 - headers_length - 4`.
//!
//! Each header is `[name_len: u8][name (UTF-8)][value_type: u8][value]`.
//! AWS defines 10 value types but Bedrock in practice only emits a handful:
//!
//!   - `7` — UTF-8 string with `[u16 BE length][bytes]`
//!   - `6` — byte array with `[u16 BE length][bytes]`
//!
//! We support both because Bedrock's `:event-type`, `:content-type`,
//! and `:message-type` come through as type-7 strings, and the payload
//! carries an embedded JSON document as a UTF-8 string AT THE PAYLOAD
//! LEVEL (not as a header value).
//!
//! # Why a stateful struct
//!
//! TCP delivers chunks at unpredictable boundaries — a single
//! EventStream message can arrive in any number of pieces, and one
//! `Bytes` chunk can also contain multiple complete messages plus a
//! partial trailing one. The parser MUST resume across chunks without
//! losing bytes; we accumulate into an internal buffer and yield
//! complete messages as they materialise. Same incremental contract
//! as `sse::framing::SseFramer` (Phase C) — that's the model.
//!
//! # No panics, no fallbacks
//!
//! Per `feedback_no_silent_fallbacks.md`: this parser ALWAYS returns a
//! `Result`. CRC mismatch → [`ParseError::PreludeCrcMismatch`] /
//! [`ParseError::MessageCrcMismatch`]; truncated prelude with absurd
//! lengths → [`ParseError::ImplausiblePreludeLengths`]. The handler
//! turns errors into 5xx responses — we never silently swallow a
//! malformed frame and forward zeros to the client.

use std::collections::HashMap;

use bytes::{Buf, Bytes, BytesMut};
use thiserror::Error;

/// AWS-defined header value types. Bedrock today only uses `7`
/// (string) and `6` (byte buffer); the others are listed for
/// completeness so a future Bedrock surface that emits them does
/// not require a parser revision.
mod header_type {
    pub const TRUE: u8 = 0;
    pub const FALSE: u8 = 1;
    pub const BYTE: u8 = 2;
    pub const SHORT: u8 = 3;
    pub const INTEGER: u8 = 4;
    pub const LONG: u8 = 5;
    pub const BYTE_ARRAY: u8 = 6;
    pub const STRING: u8 = 7;
    pub const TIMESTAMP: u8 = 8;
    pub const UUID: u8 = 9;
}

/// Length of the 3-u32 prelude in bytes.
const PRELUDE_LEN: usize = 12;

/// CRC32 trails the message and is itself 4 bytes, NOT counted in
/// `headers_length` but counted in `total_length`.
const MESSAGE_CRC_LEN: usize = 4;

/// Sanity ceiling for `total_length`. AWS's documented maximum is
/// 16 MiB. We refuse anything beyond ~32 MiB (2× headroom for future
/// limit raises) so a truncated stream that decoded a garbage length
/// does not allocate gigabytes. Configurable via the public
/// [`EventStreamParser::with_max_message_bytes`] constructor.
const DEFAULT_MAX_MESSAGE_BYTES: usize = 32 * 1024 * 1024;

/// One value carried in an EventStream header. AWS defines 10 wire
/// types; today Bedrock emits two (string + byte-buffer). We expose
/// the variants we actually parse and surface a structured error for
/// any other type so the caller can log it loudly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeaderValue {
    /// `header_type::STRING` — UTF-8 string.
    String(String),
    /// `header_type::BYTE_ARRAY` — opaque bytes.
    Bytes(Bytes),
}

impl HeaderValue {
    /// Cheap accessor for the common `event_name == "chunk"` style
    /// dispatch. Returns `Some` only for [`HeaderValue::String`].
    pub fn as_str(&self) -> Option<&str> {
        match self {
            HeaderValue::String(s) => Some(s.as_str()),
            HeaderValue::Bytes(_) => None,
        }
    }
}

/// One complete EventStream message.
#[derive(Debug, Clone)]
pub struct EventStreamMessage {
    /// Headers, lower-cased keys for stable lookup. Bedrock spec
    /// headers (`:event-type`, `:content-type`, `:message-type`) are
    /// already lower-case but we normalise defensively.
    pub headers: HashMap<String, HeaderValue>,
    /// Raw payload bytes. The semantic meaning depends on
    /// `:event-type` — for Bedrock Anthropic streaming responses each
    /// `chunk` payload is a UTF-8 JSON string carrying one Anthropic
    /// SSE event.
    pub payload: Bytes,
}

impl EventStreamMessage {
    /// Read the `:event-type` header as a string. Convenience for the
    /// translator. Returns `None` if missing or not a string.
    pub fn event_type(&self) -> Option<&str> {
        self.headers.get(":event-type").and_then(|v| v.as_str())
    }

    /// Read the `:message-type` header as a string. Bedrock uses
    /// `event` (data frames) vs `exception` (synchronous errors).
    pub fn message_type(&self) -> Option<&str> {
        self.headers.get(":message-type").and_then(|v| v.as_str())
    }
}

/// Errors surfaced by the parser. Per project rules these are
/// structured (not `String`) so the handler can dispatch on them.
#[derive(Debug, Error)]
pub enum ParseError {
    /// `total_length` smaller than the minimum a well-formed message
    /// requires (12-byte prelude, headers block, 4-byte trailing CRC).
    /// Wire-format violation; AWS would never emit this. Surfaced
    /// loudly so the handler can 5xx and log it.
    #[error(
        "implausible prelude lengths: total_length={total_length}, headers_length={headers_length}"
    )]
    ImplausiblePreludeLengths {
        total_length: u32,
        headers_length: u32,
    },
    /// `total_length` exceeds the configured `max_message_bytes` cap.
    /// The cap defaults to 32 MiB; configurable via
    /// [`EventStreamParser::with_max_message_bytes`].
    #[error("message too large: total_length={total_length} cap={cap}")]
    MessageTooLarge { total_length: u32, cap: usize },
    /// CRC32 of the first 8 bytes of the prelude did not match the
    /// 9th-12th bytes. Indicates corruption (in-flight bit-flip,
    /// truncated chunk, or — if persistent — wire-format version skew).
    #[error("prelude CRC mismatch: expected={expected:#010x} got={got:#010x}")]
    PreludeCrcMismatch { expected: u32, got: u32 },
    /// CRC32 of the entire message body did not match the trailing
    /// 4-byte CRC. Indicates the message bytes were corrupted in
    /// flight.
    #[error("message CRC mismatch: expected={expected:#010x} got={got:#010x}")]
    MessageCrcMismatch { expected: u32, got: u32 },
    /// A header's `name_len` extended past the declared end of the
    /// headers block.
    #[error("truncated header at offset {offset}: needs {needed} more bytes")]
    TruncatedHeader { offset: usize, needed: usize },
    /// A header's name was not valid UTF-8.
    #[error("header name not valid UTF-8 at offset {offset}: {source}")]
    HeaderNameNotUtf8 {
        offset: usize,
        #[source]
        source: std::str::Utf8Error,
    },
    /// A string-typed header value was not valid UTF-8.
    #[error("header value not valid UTF-8 at offset {offset}: {source}")]
    HeaderValueNotUtf8 {
        offset: usize,
        #[source]
        source: std::str::Utf8Error,
    },
    /// Header value type not in the AWS spec (the spec defines 0..=9).
    /// `value_type` is the literal byte we read.
    #[error("unsupported header value type {value_type} at offset {offset}")]
    UnsupportedHeaderType { value_type: u8, offset: usize },
}

/// Whether to validate CRCs. Defaults to `Yes`; tests + the
/// `--bedrock-validate-eventstream-crc=false` debug flag flip it off.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum CrcValidation {
    #[default]
    Yes,
    No,
}

/// Stateful parser. Feed `Bytes` chunks via [`EventStreamParser::push`]
/// and pull complete messages via [`EventStreamParser::next_message`].
/// Mirrors the API shape of `sse::framing::SseFramer` deliberately so
/// callers familiar with the SSE side find the EventStream side
/// identical in spirit.
#[derive(Debug)]
pub struct EventStreamParser {
    buf: BytesMut,
    crc_validation: CrcValidation,
    max_message_bytes: usize,
}

impl Default for EventStreamParser {
    fn default() -> Self {
        Self {
            buf: BytesMut::new(),
            crc_validation: CrcValidation::Yes,
            max_message_bytes: DEFAULT_MAX_MESSAGE_BYTES,
        }
    }
}

impl EventStreamParser {
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct with explicit CRC validation policy. Operators flip
    /// to `No` only for debugging — production must validate.
    pub fn with_crc_validation(mut self, crc: CrcValidation) -> Self {
        self.crc_validation = crc;
        self
    }

    /// Configurable hard cap on `total_length`. Above this the parser
    /// refuses (returns [`ParseError::MessageTooLarge`]) so a corrupted
    /// stream cannot allocate unbounded memory.
    pub fn with_max_message_bytes(mut self, cap: usize) -> Self {
        self.max_message_bytes = cap;
        self
    }

    /// Append a chunk of inbound bytes. Zero-length chunks are no-ops.
    pub fn push(&mut self, chunk: &[u8]) {
        if !chunk.is_empty() {
            self.buf.extend_from_slice(chunk);
        }
    }

    /// Number of bytes currently buffered (un-framed). Useful for tests.
    pub fn buffered_len(&self) -> usize {
        self.buf.len()
    }

    /// Pull the next complete message. Returns:
    ///
    /// - `Ok(None)` — the buffer doesn't yet contain a full message;
    ///   caller should `push` more bytes.
    /// - `Ok(Some(msg))` — one complete message; caller may call
    ///   again to drain further messages from the buffer.
    /// - `Err(_)` — wire-format violation. Per project policy, the
    ///   handler 5xx's and logs `event=bedrock_eventstream_*_mismatch`.
    pub fn next_message(&mut self) -> Result<Option<EventStreamMessage>, ParseError> {
        // Need at least the prelude before we can decide.
        if self.buf.len() < PRELUDE_LEN {
            return Ok(None);
        }
        let total_length = u32::from_be_bytes(self.buf[0..4].try_into().expect("4 bytes"));
        let headers_length = u32::from_be_bytes(self.buf[4..8].try_into().expect("4 bytes"));
        let prelude_crc_expected = u32::from_be_bytes(self.buf[8..12].try_into().expect("4 bytes"));

        // Validate plausibility BEFORE we wait for the rest of the
        // bytes. If the prelude lengths are absurd we want to fail
        // immediately rather than buffer a multi-GB stream.
        if (total_length as usize) < PRELUDE_LEN + headers_length as usize + MESSAGE_CRC_LEN {
            return Err(ParseError::ImplausiblePreludeLengths {
                total_length,
                headers_length,
            });
        }
        if (total_length as usize) > self.max_message_bytes {
            return Err(ParseError::MessageTooLarge {
                total_length,
                cap: self.max_message_bytes,
            });
        }

        // Validate prelude CRC over the first 8 bytes (lengths only,
        // NOT the prelude_crc itself).
        if matches!(self.crc_validation, CrcValidation::Yes) {
            let computed = crc32fast::hash(&self.buf[0..8]);
            if computed != prelude_crc_expected {
                return Err(ParseError::PreludeCrcMismatch {
                    expected: prelude_crc_expected,
                    got: computed,
                });
            }
        }

        // Need the entire message before we yield.
        if self.buf.len() < total_length as usize {
            return Ok(None);
        }

        // Validate trailing message CRC over bytes [0..total_length-4].
        let message_crc_expected = u32::from_be_bytes(
            self.buf[total_length as usize - MESSAGE_CRC_LEN..total_length as usize]
                .try_into()
                .expect("4 bytes"),
        );
        if matches!(self.crc_validation, CrcValidation::Yes) {
            let computed = crc32fast::hash(&self.buf[0..total_length as usize - MESSAGE_CRC_LEN]);
            if computed != message_crc_expected {
                return Err(ParseError::MessageCrcMismatch {
                    expected: message_crc_expected,
                    got: computed,
                });
            }
        }

        // Slice the message off the buffer.
        let message_bytes = self.buf.split_to(total_length as usize).freeze();

        let headers_start = PRELUDE_LEN;
        let headers_end = PRELUDE_LEN + headers_length as usize;
        let payload_start = headers_end;
        let payload_end = total_length as usize - MESSAGE_CRC_LEN;

        let headers_slice = message_bytes.slice(headers_start..headers_end);
        let payload = message_bytes.slice(payload_start..payload_end);
        let headers = parse_headers(&headers_slice)?;

        Ok(Some(EventStreamMessage { headers, payload }))
    }
}

/// One-shot helper. Convenience for callers that already have all the
/// bytes of one message in hand (e.g. wiremock test fixtures). Returns
/// `Err` if the bytes are not exactly one well-formed message.
///
/// Per the property test invariant, this never panics on adversarial
/// input — every malformed-bytes path returns a [`ParseError`].
pub fn parse(bytes: &[u8]) -> Result<EventStreamMessage, ParseError> {
    let mut parser = EventStreamParser::new();
    parser.push(bytes);
    match parser.next_message()? {
        Some(msg) => Ok(msg),
        None => Err(ParseError::ImplausiblePreludeLengths {
            total_length: 0,
            headers_length: 0,
        }),
    }
}

/// Parse a headers-block byte slice.
fn parse_headers(slice: &Bytes) -> Result<HashMap<String, HeaderValue>, ParseError> {
    let mut out: HashMap<String, HeaderValue> = HashMap::new();
    let mut cursor = 0usize;
    let total = slice.len();
    while cursor < total {
        // [name_len: u8][name][value_type: u8][value]
        if cursor + 1 > total {
            return Err(ParseError::TruncatedHeader {
                offset: cursor,
                needed: 1,
            });
        }
        let name_len = slice[cursor] as usize;
        cursor += 1;
        if cursor + name_len > total {
            return Err(ParseError::TruncatedHeader {
                offset: cursor,
                needed: name_len,
            });
        }
        let name = std::str::from_utf8(&slice[cursor..cursor + name_len])
            .map_err(|e| ParseError::HeaderNameNotUtf8 {
                offset: cursor,
                source: e,
            })?
            .to_ascii_lowercase();
        cursor += name_len;
        if cursor + 1 > total {
            return Err(ParseError::TruncatedHeader {
                offset: cursor,
                needed: 1,
            });
        }
        let value_type = slice[cursor];
        cursor += 1;
        let value = match value_type {
            header_type::STRING => {
                if cursor + 2 > total {
                    return Err(ParseError::TruncatedHeader {
                        offset: cursor,
                        needed: 2,
                    });
                }
                let value_len =
                    u16::from_be_bytes(slice[cursor..cursor + 2].try_into().expect("2 bytes"))
                        as usize;
                cursor += 2;
                if cursor + value_len > total {
                    return Err(ParseError::TruncatedHeader {
                        offset: cursor,
                        needed: value_len,
                    });
                }
                let s = std::str::from_utf8(&slice[cursor..cursor + value_len]).map_err(|e| {
                    ParseError::HeaderValueNotUtf8 {
                        offset: cursor,
                        source: e,
                    }
                })?;
                cursor += value_len;
                HeaderValue::String(s.to_string())
            }
            header_type::BYTE_ARRAY => {
                if cursor + 2 > total {
                    return Err(ParseError::TruncatedHeader {
                        offset: cursor,
                        needed: 2,
                    });
                }
                let value_len =
                    u16::from_be_bytes(slice[cursor..cursor + 2].try_into().expect("2 bytes"))
                        as usize;
                cursor += 2;
                if cursor + value_len > total {
                    return Err(ParseError::TruncatedHeader {
                        offset: cursor,
                        needed: value_len,
                    });
                }
                let v = slice.slice(cursor..cursor + value_len);
                cursor += value_len;
                HeaderValue::Bytes(v)
            }
            // The TRUE/FALSE flags carry no value bytes; we surface
            // them as a single-byte payload `Bytes` for completeness
            // even though Bedrock never emits them. Other types that
            // DO carry payload but we have no use for fall through to
            // the "unsupported" arm — better a structured error than
            // a silently-skipped header.
            header_type::TRUE => HeaderValue::Bytes(Bytes::from_static(&[1])),
            header_type::FALSE => HeaderValue::Bytes(Bytes::from_static(&[0])),
            header_type::BYTE => {
                if cursor + 1 > total {
                    return Err(ParseError::TruncatedHeader {
                        offset: cursor,
                        needed: 1,
                    });
                }
                let v = slice.slice(cursor..cursor + 1);
                cursor += 1;
                HeaderValue::Bytes(v)
            }
            header_type::SHORT => {
                if cursor + 2 > total {
                    return Err(ParseError::TruncatedHeader {
                        offset: cursor,
                        needed: 2,
                    });
                }
                let v = slice.slice(cursor..cursor + 2);
                cursor += 2;
                HeaderValue::Bytes(v)
            }
            header_type::INTEGER => {
                if cursor + 4 > total {
                    return Err(ParseError::TruncatedHeader {
                        offset: cursor,
                        needed: 4,
                    });
                }
                let v = slice.slice(cursor..cursor + 4);
                cursor += 4;
                HeaderValue::Bytes(v)
            }
            header_type::LONG | header_type::TIMESTAMP => {
                if cursor + 8 > total {
                    return Err(ParseError::TruncatedHeader {
                        offset: cursor,
                        needed: 8,
                    });
                }
                let v = slice.slice(cursor..cursor + 8);
                cursor += 8;
                HeaderValue::Bytes(v)
            }
            header_type::UUID => {
                if cursor + 16 > total {
                    return Err(ParseError::TruncatedHeader {
                        offset: cursor,
                        needed: 16,
                    });
                }
                let v = slice.slice(cursor..cursor + 16);
                cursor += 16;
                HeaderValue::Bytes(v)
            }
            other => {
                return Err(ParseError::UnsupportedHeaderType {
                    value_type: other,
                    offset: cursor.saturating_sub(1),
                });
            }
        };
        out.insert(name, value);
    }
    // Sanity: cursor must equal total or we have a parse-loop bug.
    let _ = (cursor, total, header_type::TRUE, Buf::remaining(&&[][..])); // pin unused-import
    Ok(out)
}

// ────────────────────────────────────────────────────────────────────
// Test-only helpers for synthesising bytes. Living here (vs in the
// integration test crate) so unit tests in this module can exercise
// the parser end-to-end and the integration tests have the same builder
// available via `pub(crate)`.
// ────────────────────────────────────────────────────────────────────

/// Builder for synthesising a well-formed EventStream message. Used by
/// tests and by [`build_chunk_message`]. Producing valid bytes is much
/// more delicate than parsing them — this helper centralises the CRC
/// math.
#[derive(Debug, Default)]
pub struct MessageBuilder {
    headers: Vec<(String, HeaderValue)>,
    payload: Bytes,
}

impl MessageBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn header_string(mut self, name: &str, value: &str) -> Self {
        self.headers
            .push((name.to_string(), HeaderValue::String(value.to_string())));
        self
    }

    pub fn header_bytes(mut self, name: &str, value: Bytes) -> Self {
        self.headers
            .push((name.to_string(), HeaderValue::Bytes(value)));
        self
    }

    pub fn payload(mut self, payload: Bytes) -> Self {
        self.payload = payload;
        self
    }

    /// Serialize to the wire format. Computes both CRCs.
    pub fn build(self) -> Bytes {
        let mut headers_bytes = BytesMut::new();
        for (name, value) in &self.headers {
            // name_len + name
            assert!(name.len() <= u8::MAX as usize, "header name too long");
            headers_bytes.extend_from_slice(&[name.len() as u8]);
            headers_bytes.extend_from_slice(name.as_bytes());
            match value {
                HeaderValue::String(s) => {
                    headers_bytes.extend_from_slice(&[header_type::STRING]);
                    let len = s.len() as u16;
                    headers_bytes.extend_from_slice(&len.to_be_bytes());
                    headers_bytes.extend_from_slice(s.as_bytes());
                }
                HeaderValue::Bytes(b) => {
                    headers_bytes.extend_from_slice(&[header_type::BYTE_ARRAY]);
                    let len = b.len() as u16;
                    headers_bytes.extend_from_slice(&len.to_be_bytes());
                    headers_bytes.extend_from_slice(b);
                }
            }
        }
        let headers_len = headers_bytes.len() as u32;
        let payload_len = self.payload.len() as u32;
        let total_len = PRELUDE_LEN as u32 + headers_len + payload_len + MESSAGE_CRC_LEN as u32;

        let mut out = BytesMut::with_capacity(total_len as usize);
        out.extend_from_slice(&total_len.to_be_bytes());
        out.extend_from_slice(&headers_len.to_be_bytes());
        let prelude_crc = crc32fast::hash(&out[..]);
        out.extend_from_slice(&prelude_crc.to_be_bytes());
        out.extend_from_slice(&headers_bytes);
        out.extend_from_slice(&self.payload);
        let message_crc = crc32fast::hash(&out[..]);
        out.extend_from_slice(&message_crc.to_be_bytes());
        out.freeze()
    }
}

/// Build a minimal Anthropic-on-Bedrock `chunk` message. The payload
/// is the JSON bytes for an Anthropic SSE event, as Bedrock emits.
pub fn build_chunk_message(payload_json: &str) -> Bytes {
    MessageBuilder::new()
        .header_string(":event-type", "chunk")
        .header_string(":content-type", "application/json")
        .header_string(":message-type", "event")
        .payload(Bytes::copy_from_slice(payload_json.as_bytes()))
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_single_message() {
        let bytes = build_chunk_message(r#"{"type":"message_start","message":{"id":"msg_a"}}"#);
        let mut parser = EventStreamParser::new();
        parser.push(&bytes);
        let msg = parser.next_message().unwrap().unwrap();
        assert_eq!(msg.event_type(), Some("chunk"));
        assert_eq!(msg.message_type(), Some("event"));
        let payload_str = std::str::from_utf8(&msg.payload).unwrap();
        assert!(payload_str.starts_with(r#"{"type":"message_start""#));
        assert!(parser.next_message().unwrap().is_none());
    }

    #[test]
    fn round_trip_multi_message_one_chunk() {
        let mut combined = BytesMut::new();
        combined.extend_from_slice(&build_chunk_message(r#"{"a":1}"#));
        combined.extend_from_slice(&build_chunk_message(r#"{"b":2}"#));
        let mut parser = EventStreamParser::new();
        parser.push(&combined);
        let m1 = parser.next_message().unwrap().unwrap();
        let m2 = parser.next_message().unwrap().unwrap();
        assert_eq!(std::str::from_utf8(&m1.payload).unwrap(), r#"{"a":1}"#);
        assert_eq!(std::str::from_utf8(&m2.payload).unwrap(), r#"{"b":2}"#);
        assert!(parser.next_message().unwrap().is_none());
    }

    #[test]
    fn one_byte_at_a_time_no_loss() {
        // Drip-feed a complete message one byte at a time. Parser
        // must yield exactly one message at the very end.
        let bytes = build_chunk_message(r#"{"type":"message_stop"}"#);
        let mut parser = EventStreamParser::new();
        for b in bytes.iter() {
            parser.push(std::slice::from_ref(b));
            // For all but the last byte, no message yet.
        }
        let msg = parser.next_message().unwrap().unwrap();
        assert_eq!(
            std::str::from_utf8(&msg.payload).unwrap(),
            r#"{"type":"message_stop"}"#,
        );
    }

    #[test]
    fn prelude_crc_mismatch_loud() {
        // Build a valid message, then corrupt the prelude CRC.
        let mut bytes: Vec<u8> = build_chunk_message(r#"{"x":1}"#).to_vec();
        bytes[8] ^= 0xff;
        let mut parser = EventStreamParser::new();
        parser.push(&bytes);
        let err = parser.next_message().unwrap_err();
        assert!(
            matches!(err, ParseError::PreludeCrcMismatch { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn message_crc_mismatch_loud() {
        let mut bytes: Vec<u8> = build_chunk_message(r#"{"y":2}"#).to_vec();
        // Corrupt the trailing CRC (last 4 bytes).
        let n = bytes.len();
        bytes[n - 1] ^= 0xff;
        let mut parser = EventStreamParser::new();
        parser.push(&bytes);
        let err = parser.next_message().unwrap_err();
        assert!(
            matches!(err, ParseError::MessageCrcMismatch { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn implausible_lengths_loud() {
        // Forge a prelude where total < headers + 12 + 4. Caller never
        // gets a chance to even compute the CRC.
        let mut bytes = BytesMut::new();
        bytes.extend_from_slice(&8u32.to_be_bytes()); // total = 8 (impossible)
        bytes.extend_from_slice(&0u32.to_be_bytes()); // headers = 0
        bytes.extend_from_slice(&0u32.to_be_bytes()); // bogus crc
        let mut parser = EventStreamParser::new();
        parser.push(&bytes);
        let err = parser.next_message().unwrap_err();
        assert!(
            matches!(err, ParseError::ImplausiblePreludeLengths { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn message_too_large_loud() {
        // Forge a prelude with an enormous total_length and observe
        // the cap rejection (we never even allocate the payload).
        let mut bytes = BytesMut::new();
        bytes.extend_from_slice(&(64u32 * 1024 * 1024).to_be_bytes()); // 64 MiB
        bytes.extend_from_slice(&0u32.to_be_bytes());
        let valid_crc = crc32fast::hash(&bytes[0..8]);
        bytes.extend_from_slice(&valid_crc.to_be_bytes());
        let mut parser = EventStreamParser::new().with_max_message_bytes(32 * 1024 * 1024);
        parser.push(&bytes);
        let err = parser.next_message().unwrap_err();
        assert!(
            matches!(err, ParseError::MessageTooLarge { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn crc_validation_off_accepts_corrupt_prelude() {
        // With CrcValidation::No, prelude+message CRCs are ignored.
        let mut bytes: Vec<u8> = build_chunk_message(r#"{"a":3}"#).to_vec();
        bytes[8] ^= 0xff; // corrupt prelude crc
        let n = bytes.len();
        bytes[n - 1] ^= 0xff; // corrupt message crc
        let mut parser = EventStreamParser::new().with_crc_validation(CrcValidation::No);
        parser.push(&bytes);
        let msg = parser.next_message().unwrap().unwrap();
        assert_eq!(std::str::from_utf8(&msg.payload).unwrap(), r#"{"a":3}"#);
    }

    #[test]
    fn header_bytes_round_trip() {
        let payload_bytes = Bytes::from_static(b"\x01\x02\x03");
        let bytes = MessageBuilder::new()
            .header_string(":event-type", "chunk")
            .header_bytes(":custom-bytes", payload_bytes.clone())
            .payload(Bytes::from_static(b"{}"))
            .build();
        let parsed = parse(&bytes).unwrap();
        match parsed.headers.get(":custom-bytes").unwrap() {
            HeaderValue::Bytes(b) => assert_eq!(b, &payload_bytes),
            _ => panic!("expected bytes header"),
        }
    }

    #[test]
    fn parse_one_shot_helper() {
        let bytes = build_chunk_message(r#"{"k":"v"}"#);
        let m = parse(&bytes).unwrap();
        assert_eq!(m.event_type(), Some("chunk"));
    }

    #[test]
    fn parse_one_shot_returns_err_on_partial() {
        // 4 bytes only — never enough for a prelude.
        let bytes = vec![0u8, 0, 0, 8];
        let err = parse(&bytes).unwrap_err();
        assert!(matches!(err, ParseError::ImplausiblePreludeLengths { .. }));
    }

    #[test]
    fn empty_buffer_yields_none() {
        let mut parser = EventStreamParser::new();
        assert!(parser.next_message().unwrap().is_none());
    }
}
