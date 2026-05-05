//! Byte-level Server-Sent Events (SSE) framing.
//!
//! # Why byte-level
//!
//! The Python proxy decoded each TCP chunk to UTF-8 with
//! `errors="ignore"`, which silently lost bytes whenever a multi-byte
//! codepoint (any emoji, any non-ASCII character) straddled a chunk
//! boundary. Production telemetry logged 1946 parse failures over 9
//! days from this single quirk — see
//! `~/Desktop/HEADROOM_PROXY_LOG_FINDINGS_2026_05_03.md` (P1-15).
//!
//! The Rust framer accumulates raw bytes into a `BytesMut` buffer and
//! finds event terminators (`\n\n`) in **bytes**. UTF-8 is decoded
//! exactly once, per complete event, so split-codepoint chunks rejoin
//! correctly. Per project invariants we never call `from_utf8_lossy`
//! on partial buffers — the buffer holds bytes until a full event is
//! framed, then a single `String::from_utf8` decodes the whole thing.
//!
//! # Wire format
//!
//! Per WHATWG SSE spec + provider extensions:
//! - Lines end with `\n` (CR is tolerated and stripped, but providers
//!   in practice send LF only).
//! - An event terminates on a blank line (i.e. consecutive `\n\n`).
//! - Lines starting with `:` are comments. Anthropic and OpenAI use
//!   `: ping` keepalives; we drop them silently (they convey no data).
//! - The literal `data: [DONE]` is a stream-end sentinel used by
//!   OpenAI Chat/Responses (Anthropic uses `event: message_stop`
//!   instead). Some providers append a trailing `\n\n` after `[DONE]`;
//!   we tolerate any number of trailing bytes.
//! - Multiple `data:` lines in one event concatenate with `\n` per
//!   the WHATWG spec. (Anthropic does not currently use this; OpenAI
//!   Responses occasionally does.)
//!
//! # Zero-copy where possible
//!
//! The framer is built on `bytes::Bytes` which is reference-counted;
//! `extract_event_payload` slices the underlying buffer rather than
//! copying. UTF-8 decoding is the one unavoidable allocation per
//! event (we need a `String`/`&str` for downstream parsing).

use bytes::{Bytes, BytesMut};

/// One framed SSE event ready for state-machine consumption.
///
/// `event_name` is the value of the `event:` field (e.g.
/// `message_start`, `content_block_delta`). It is `None` when the
/// event has no `event:` line — the OpenAI Chat Completions stream
/// does this for every chunk (only the `data:` field is present, and
/// the event type is encoded inside the JSON payload).
///
/// `data` is the concatenated value of all `data:` fields in the
/// event, joined by `\n` per the SSE spec. `data` is **not** UTF-8
/// validated here — the framer only guarantees that `data` is the
/// exact byte slice between the field prefix and the event
/// terminator. State-machine code that needs UTF-8 calls
/// `event.data_str()` which returns `Result<&str, Utf8Error>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    pub event_name: Option<String>,
    pub data: Bytes,
}

impl SseEvent {
    /// UTF-8 decode the data payload. Returns `Err` on invalid UTF-8;
    /// the framer never panics, so malformed bytes surface here as a
    /// recoverable error the state machine can log and skip.
    pub fn data_str(&self) -> Result<&str, std::str::Utf8Error> {
        std::str::from_utf8(&self.data)
    }

    /// True iff the data field is the literal `[DONE]` sentinel used
    /// by OpenAI Chat Completions and OpenAI Responses to mark end of
    /// stream. Anthropic does not use this — it terminates with an
    /// `event: message_stop`.
    pub fn is_done_sentinel(&self) -> bool {
        // Compare bytes directly so we never touch UTF-8 validation
        // on the hot path. `[DONE]` is pure ASCII, so any byte-equal
        // match is a genuine sentinel.
        self.data.as_ref() == b"[DONE]"
    }
}

/// Stateful byte-level SSE framer.
///
/// Feed `Bytes` chunks via [`SseFramer::push`]; pull complete framed
/// events via [`SseFramer::next_event`]. The framer never blocks and
/// never allocates on the input chunks (it appends to an internal
/// `BytesMut` and then slices that). On `take_remaining`, callers can
/// recover unparsed trailing bytes (used by tests; production code
/// just drops the framer at end-of-stream).
#[derive(Debug, Default)]
pub struct SseFramer {
    /// Accumulator for inbound bytes that have not yet been framed
    /// into a complete event. Always holds at most one in-flight
    /// event's worth of data plus any partial trailing bytes.
    buf: BytesMut,
    /// Once we see `[DONE]` we still tolerate further events (some
    /// providers append a final empty `\n\n`), but the state machine
    /// can use this to gate "no more useful events expected".
    done_seen: bool,
}

impl SseFramer {
    pub fn new() -> Self {
        Self::default()
    }

    /// True after the framer has yielded an event whose `data` is the
    /// `[DONE]` sentinel. Once this is set, additional bytes pushed
    /// in are still framed (so trailing whitespace doesn't error),
    /// but state-machine code may stop attending to events.
    pub fn done_seen(&self) -> bool {
        self.done_seen
    }

    /// Append a chunk of inbound bytes to the framer's buffer.
    ///
    /// `chunk` may straddle event boundaries, line boundaries, or
    /// multi-byte UTF-8 codepoints. The framer makes no assumptions
    /// beyond "these bytes appear, in order, after the bytes we've
    /// already seen on this stream". Zero-length chunks are a no-op.
    pub fn push(&mut self, chunk: &[u8]) {
        if chunk.is_empty() {
            return;
        }
        self.buf.extend_from_slice(chunk);
    }

    /// Drain the next complete event from the buffer, if one is
    /// available. Returns `None` if the buffer doesn't yet contain a
    /// blank-line terminator (`\n\n`). Returns `Some(Ok(event))` for
    /// each complete event. Returns `Some(Err(...))` only on
    /// genuinely malformed event content (e.g. `data:` line with
    /// invalid UTF-8 in `event:` name — `event:` must be ASCII per
    /// spec). Comments (`: ping`) and empty events are silently
    /// skipped: the framer keeps consuming bytes until either a real
    /// event surfaces or the buffer is exhausted.
    pub fn next_event(&mut self) -> Option<Result<SseEvent, FramingError>> {
        loop {
            let term_len: usize;
            let block_end = match find_double_newline(&self.buf) {
                Some((end, len)) => {
                    term_len = len;
                    end
                }
                None => return None,
            };
            // The block is bytes [0, block_end); skip past the
            // terminator (1 or 2 bytes — see find_double_newline).
            // BytesMut::split_to is O(1): it advances the head
            // pointer without copying, so we keep zero-copy framing.
            let block = self.buf.split_to(block_end).freeze();
            let _term = self.buf.split_to(term_len);

            match parse_event_block(block) {
                Ok(Some(event)) => {
                    if event.is_done_sentinel() {
                        self.done_seen = true;
                    }
                    return Some(Ok(event));
                }
                // No data lines (comment-only / empty event /
                // pure ping). Skip and try the next block.
                Ok(None) => continue,
                Err(e) => return Some(Err(e)),
            }
        }
    }

    /// Number of bytes currently buffered (un-framed). Useful for
    /// tests asserting that no bytes were lost across chunk boundaries.
    pub fn buffered_len(&self) -> usize {
        self.buf.len()
    }

    /// Drain any remaining un-framed bytes. Used by tests to assert
    /// "no bytes left over"; production code drops the framer.
    pub fn take_remaining(&mut self) -> Bytes {
        std::mem::take(&mut self.buf).freeze()
    }
}

/// Errors the framer surfaces. Per project rules we do **not**
/// silently swallow these — callers must `tracing::warn!` and either
/// drop the event or close the stream. The framer itself never
/// panics; the property test in `tests/sse_framing.rs` enforces this.
#[derive(Debug, thiserror::Error)]
pub enum FramingError {
    /// `event:` field present but not valid UTF-8. The SSE spec
    /// requires the field name and value to be ASCII; binary in
    /// `event:` is a wire-format violation, not data we can recover
    /// from. Real providers never emit this.
    #[error("event field is not valid UTF-8: {0}")]
    EventNameNotUtf8(std::str::Utf8Error),
}

/// Find the byte offset of the first `\n\n` (or `\r\n\r\n`) terminator
/// in `buf`. Returns `(offset, terminator_len)` so the caller can
/// split off the event block and skip the terminator. `\r\n\r\n` is
/// tolerated for completeness; in practice all production providers
/// emit pure `\n\n`.
fn find_double_newline(buf: &[u8]) -> Option<(usize, usize)> {
    // Manual byte-level search. Memchr would be faster on long
    // buffers, but a typical SSE event is < 4KB and we'd be searching
    // a tight window, so a simple loop wins on cache locality.
    let mut i = 0;
    while i + 1 < buf.len() {
        if buf[i] == b'\n' && buf[i + 1] == b'\n' {
            return Some((i, 2));
        }
        // \r\n\r\n
        if i + 3 < buf.len()
            && buf[i] == b'\r'
            && buf[i + 1] == b'\n'
            && buf[i + 2] == b'\r'
            && buf[i + 3] == b'\n'
        {
            return Some((i, 4));
        }
        i += 1;
    }
    None
}

/// Parse a single SSE event block (the bytes between two `\n\n`
/// terminators). Returns `Ok(None)` when the block contained no
/// `data:` lines (comment-only, empty, or pure ping).
fn parse_event_block(block: Bytes) -> Result<Option<SseEvent>, FramingError> {
    let mut event_name: Option<String> = None;
    let mut data_parts: Vec<Bytes> = Vec::new();

    let mut start = 0usize;
    while start < block.len() {
        // Find end of line.
        let line_end = block[start..]
            .iter()
            .position(|&b| b == b'\n')
            .map(|p| start + p)
            .unwrap_or(block.len());

        // Strip a single trailing \r if present (CRLF tolerance).
        let mut line_stop = line_end;
        if line_stop > start && block[line_stop.saturating_sub(1)] == b'\r' {
            line_stop -= 1;
        }
        let line = &block[start..line_stop];
        start = line_end.saturating_add(1);

        if line.is_empty() {
            continue;
        }
        if line[0] == b':' {
            // Comment line. Includes `: ping` keepalives and any
            // provider-specific debug comments. Drop silently.
            continue;
        }

        // Split field:value on the FIRST colon. Per SSE spec, a
        // single space following the colon is stripped from the value.
        let (field, value_with_space) = match line.iter().position(|&b| b == b':') {
            Some(p) => (&line[..p], &line[p + 1..]),
            // No colon → entire line is the field name with empty value.
            None => (line, &line[line.len()..]),
        };
        let value = if value_with_space.first() == Some(&b' ') {
            &value_with_space[1..]
        } else {
            value_with_space
        };

        match field {
            b"event" => {
                let name = std::str::from_utf8(value).map_err(FramingError::EventNameNotUtf8)?;
                event_name = Some(name.to_string());
            }
            b"data" => {
                // We slice `block` (a `Bytes`) so the resulting
                // payload shares the underlying allocation. No copy.
                let abs_start = value.as_ptr() as usize - block.as_ptr() as usize;
                let abs_end = abs_start + value.len();
                data_parts.push(block.slice(abs_start..abs_end));
            }
            // `id:` and `retry:` are valid SSE fields but neither
            // Anthropic nor OpenAI uses them on streamed responses.
            // Track-and-ignore: future providers can be added.
            _ => continue,
        }
    }

    if data_parts.is_empty() {
        return Ok(None);
    }

    // Per WHATWG SSE: multiple data: lines are joined with '\n'.
    // The vast majority of provider events have exactly one data: line.
    let data = if data_parts.len() == 1 {
        data_parts.into_iter().next().unwrap()
    } else {
        let total: usize =
            data_parts.iter().map(|b| b.len()).sum::<usize>() + (data_parts.len() - 1);
        let mut out = BytesMut::with_capacity(total);
        for (i, part) in data_parts.iter().enumerate() {
            if i > 0 {
                out.extend_from_slice(b"\n");
            }
            out.extend_from_slice(part);
        }
        out.freeze()
    };

    Ok(Some(SseEvent { event_name, data }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_buffer_yields_nothing() {
        let mut f = SseFramer::new();
        assert!(f.next_event().is_none());
    }

    #[test]
    fn comment_skipped_no_event_yielded() {
        let mut f = SseFramer::new();
        f.push(b": ping\n\n");
        assert!(f.next_event().is_none());
    }

    #[test]
    fn data_line_only_yields_event_with_no_event_name() {
        let mut f = SseFramer::new();
        f.push(b"data: hello\n\n");
        let ev = f.next_event().unwrap().unwrap();
        assert_eq!(ev.event_name, None);
        assert_eq!(ev.data.as_ref(), b"hello");
    }

    #[test]
    fn event_name_and_data() {
        let mut f = SseFramer::new();
        f.push(b"event: message_start\ndata: {\"x\":1}\n\n");
        let ev = f.next_event().unwrap().unwrap();
        assert_eq!(ev.event_name.as_deref(), Some("message_start"));
        assert_eq!(ev.data.as_ref(), b"{\"x\":1}");
    }

    #[test]
    fn multiple_data_lines_joined_with_newline() {
        let mut f = SseFramer::new();
        f.push(b"data: a\ndata: b\n\n");
        let ev = f.next_event().unwrap().unwrap();
        assert_eq!(ev.data.as_ref(), b"a\nb");
    }

    #[test]
    fn done_sentinel_detected() {
        let mut f = SseFramer::new();
        f.push(b"data: [DONE]\n\n");
        let ev = f.next_event().unwrap().unwrap();
        assert!(ev.is_done_sentinel());
        assert!(f.done_seen());
    }
}
