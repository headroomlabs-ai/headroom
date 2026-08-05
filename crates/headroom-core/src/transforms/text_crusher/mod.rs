//! TextCrusher — fast deterministic extractive prose compressor (Phase 2, #1171).
//!
//! The request-path-safe alternative to ModernBERT (kompress) for large plain
//! text: heuristic sentence scoring (recency + reused BM25 relevance +
//! salience) with near-duplicate suppression, in one O(n) pass.

mod config;
mod crusher;

pub use config::TextCrusherConfig;
pub use crusher::{TextCrusher, TextCrusherResult};
