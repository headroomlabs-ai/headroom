//! Server-Sent Events (SSE) parsing for streaming LLM responses.
//!
//! This module replaces the Python proxy's bug-prone `errors="ignore"`
//! UTF-8-decode-per-chunk SSE parsing with a byte-level framer plus
//! three provider-specific state machines.
//!
//! # Layering
//!
//! ```text
//! [TCP bytes] ─▶ SseFramer (framing.rs)
//!                   │  yields SseEvent { event_name, data: Bytes }
//!                   ▼
//!         ┌─────────┴──────────┐
//!  AnthropicStreamState    ChunkState        ResponseState
//!  (anthropic.rs)          (openai_chat.rs)  (openai_responses.rs)
//! ```
//!
//! The framer is provider-agnostic. Each state machine consumes
//! `SseEvent`s and updates structured per-stream state used by
//! telemetry. None of these mutate the bytes flowing back to the
//! client — see `proxy.rs` for the byte-passthrough + parallel
//! state-machine wiring.
//!
//! # Bugs this retires
//!
//! - **P1-15** UTF-8 split across TCP reads (framer decodes per
//!   complete event, not per chunk).
//! - **P1-8** missing `thinking_delta` arm.
//! - **P1-9** missing `signature_delta` arm.
//! - **P1-14** missing `citations_delta` arm.
//! - **P1-17** OpenAI Responses items keyed by position not id.
//! - **P4-48** OpenAI tool_call `id` overwritten by None on
//!   subsequent chunks.

pub mod anthropic;
pub mod framing;
pub mod openai_chat;
pub mod openai_responses;

pub use framing::{FramingError, SseEvent, SseFramer};
