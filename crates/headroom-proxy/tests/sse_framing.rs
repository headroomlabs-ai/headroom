//! Integration tests for the byte-level SSE framer.
//!
//! These tests pin down the wire-format invariants the framer must
//! preserve regardless of how the underlying TCP stream chunks the
//! bytes:
//!
//!   - Multi-byte UTF-8 codepoints split across chunks must rejoin
//!     intact (P1-15: the Python proxy lost bytes here).
//!   - A single `\n` is NOT an event terminator; only `\n\n` is.
//!   - `: ping` keepalive comments yield no event (silently skipped).
//!   - `data: [DONE]` is detected via the `is_done_sentinel()` API.
//!   - Trailing bytes after `[DONE]` (some providers send `\n\n` or
//!     additional pings) are tolerated without error.

use bytes::Bytes;
use headroom_proxy::sse::{SseEvent, SseFramer};

/// A 4-byte UTF-8 emoji (U+1F600 GRINNING FACE) is a deterministic
/// torture-test for split-codepoint chunking. Bytes: F0 9F 98 80.
const EMOJI: &[u8] = "\u{1F600}".as_bytes();

#[test]
fn utf8_split_emoji_across_chunks_preserved() {
    // Build the event "data: <emoji>\n\n" then split it at the
    // BYTE BOUNDARY in the middle of the emoji's 4-byte sequence.
    // Concretely we split between the second and third byte of the
    // codepoint, the worst case for naive per-chunk decoders.
    let mut full = Vec::from(b"data: ".as_slice());
    full.extend_from_slice(EMOJI);
    full.extend_from_slice(b"\n\n");

    // Find the emoji's first byte index (always 6 — `data: ` is six
    // ASCII bytes — but compute it so a future doc-edit can't drift).
    let emoji_start = 6usize;
    let split_at = emoji_start + 2; // mid-codepoint split.
    assert!(split_at < full.len() - 2);

    let mut framer = SseFramer::new();
    framer.push(&full[..split_at]);
    // No complete event yet (no \n\n in the first chunk).
    assert!(framer.next_event().is_none());
    framer.push(&full[split_at..]);

    let ev = framer.next_event().expect("event must surface").unwrap();
    assert_eq!(ev.event_name, None);
    // The bytes in `data` must equal the original emoji bytes EXACTLY.
    assert_eq!(ev.data.as_ref(), EMOJI, "no bytes lost across split");
    // And the UTF-8 decode must round-trip.
    assert_eq!(ev.data_str().unwrap(), "\u{1F600}");
    // No more events; buffer empty.
    assert!(framer.next_event().is_none());
    assert_eq!(framer.buffered_len(), 0);
}

#[test]
fn single_newline_does_not_emit_event() {
    let mut framer = SseFramer::new();
    framer.push(b"data: hello\n");
    // Only one newline — event not yet terminated. The framer must
    // hold the bytes pending the second newline.
    assert!(framer.next_event().is_none());
    // The full payload is still buffered (no bytes silently consumed).
    assert!(framer.buffered_len() > 0);
}

#[test]
fn double_newline_emits_event() {
    let mut framer = SseFramer::new();
    framer.push(b"data: hello\n\n");
    let ev = framer.next_event().expect("event must surface").unwrap();
    assert_eq!(ev.event_name, None);
    assert_eq!(ev.data.as_ref(), b"hello");
    // Buffer fully drained.
    assert_eq!(framer.buffered_len(), 0);
}

#[test]
fn ping_keepalive_skipped() {
    let mut framer = SseFramer::new();
    // Both forms of keepalive a real provider might send:
    //   `: ping\n\n` — SSE-spec comment line.
    //   `event: ping\ndata: {}\n\n` — Anthropic's explicit ping event
    //     (handled at the state-machine layer, not the framer).
    framer.push(b": ping\n\n");
    // Comment-only block: framer skips silently.
    assert!(framer.next_event().is_none());

    // After a real event arrives, the framer surfaces it.
    framer.push(b"data: real\n\n");
    let ev = framer.next_event().expect("event must surface").unwrap();
    assert_eq!(ev.data.as_ref(), b"real");
}

#[test]
fn done_sentinel_detected() {
    let mut framer = SseFramer::new();
    framer.push(b"data: [DONE]\n\n");
    let ev = framer.next_event().expect("event must surface").unwrap();
    assert!(ev.is_done_sentinel(), "[DONE] must be detected via the API");
    assert!(framer.done_seen(), "framer flag must record the sentinel");
}

#[test]
fn trailing_data_after_done_tolerated() {
    let mut framer = SseFramer::new();
    framer.push(b"data: [DONE]\n\n");
    let ev = framer.next_event().unwrap().unwrap();
    assert!(ev.is_done_sentinel());
    // Some providers append a final empty event or comment after
    // [DONE]. None of these may cause the framer to error.
    framer.push(b": closing\n\n");
    framer.push(b"\n\n"); // empty event
    framer.push(b"data: trailing\n\n");
    // The empty / comment events yield None; the data event surfaces.
    let next = framer.next_event().expect("trailing data event").unwrap();
    assert_eq!(next.data.as_ref(), b"trailing");
    // done_seen remains true even after subsequent events.
    assert!(framer.done_seen());
}

#[test]
fn multiple_events_one_chunk() {
    // A single TCP read may deliver several complete SSE events.
    let mut framer = SseFramer::new();
    framer.push(b"event: a\ndata: 1\n\nevent: b\ndata: 2\n\n");
    let e1 = framer.next_event().unwrap().unwrap();
    assert_eq!(e1.event_name.as_deref(), Some("a"));
    assert_eq!(e1.data.as_ref(), b"1");
    let e2 = framer.next_event().unwrap().unwrap();
    assert_eq!(e2.event_name.as_deref(), Some("b"));
    assert_eq!(e2.data.as_ref(), b"2");
    assert!(framer.next_event().is_none());
}

#[test]
fn chunk_boundary_inside_event_name() {
    // Chunk boundary in the middle of `event: messa|ge_start`.
    let mut framer = SseFramer::new();
    framer.push(b"event: messa");
    assert!(framer.next_event().is_none());
    framer.push(b"ge_start\ndata: x\n\n");
    let ev = framer.next_event().unwrap().unwrap();
    assert_eq!(ev.event_name.as_deref(), Some("message_start"));
    assert_eq!(ev.data.as_ref(), b"x");
}

// ───────────────────────────── property test ─────────────────────────
//
// The framer must NEVER panic on arbitrary byte input. TCP can hand us
// anything — partial codepoints, NUL bytes, fuzz noise. Keep this in the
// same order of magnitude as the other Rust parser fuzz tests so
// `cargo test --workspace` remains practical in CI.

use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 4_096,
        // We want the fuzzer to shrink any panic it finds instead of
        // giving up early.
        max_shrink_iters: 1024,
        ..ProptestConfig::default()
    })]
    #[test]
    fn sse_parser_no_panic_on_arbitrary_bytes(
        bytes in proptest::collection::vec(any::<u8>(), 0..2048),
    ) {
        let mut framer = SseFramer::new();
        framer.push(&bytes);
        // Drain until next_event returns None or yields an Err. Either
        // outcome is acceptable; what's NOT acceptable is a panic.
        loop {
            match framer.next_event() {
                None => break,
                Some(Ok(_)) => continue,
                Some(Err(_)) => continue,
            }
        }
        // Sanity: take_remaining never panics either.
        let rest: Bytes = framer.take_remaining();
        // Touch the bytes so the optimizer doesn't elide the call.
        prop_assert!(rest.len() <= bytes.len() + 1);
    }

    /// Same as above but feeds the bytes one byte at a time, simulating
    /// a degenerate slow TCP. The framer's chunk-independence invariant
    /// requires this to behave identically (no panic).
    #[test]
    fn sse_parser_no_panic_one_byte_at_a_time(
        bytes in proptest::collection::vec(any::<u8>(), 0..512),
    ) {
        let mut framer = SseFramer::new();
        for b in &bytes {
            framer.push(std::slice::from_ref(b));
            while let Some(r) = framer.next_event() {
                let _ = r; // tolerate Ok or Err.
            }
        }
        let _ = framer.take_remaining();
    }
}

/// Sanity: `SseEvent` derives the trait we documented (Clone/Eq).
/// This catches accidental future regressions where someone removes
/// a derive that internal callers depend on.
#[test]
fn sse_event_traits_present() {
    let e = SseEvent {
        event_name: Some("x".into()),
        data: Bytes::from_static(b"y"),
    };
    let cloned = e.clone();
    assert_eq!(e, cloned);
}
