//! headroom-xray library surface.
//!
//! Phase 1 exposes just enough of the binary's internals for integration tests:
//! Node detection, the footer pipeline, transcript parsing. The actual CLI lives
//! in `main.rs`.

pub mod codeburn;
pub mod footer;
pub mod node;
pub mod tokenize;
pub mod transcripts;
