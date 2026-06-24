//! Offload transforms — drop bytes from the wire, stash original via CCR.
//!
//! Every transform here implements [`super::traits::OffloadTransform`].
//! The output is a SUBSET of the input plus a retrieval marker; the
//! original payload sits in a [`crate::ccr::CcrStore`] keyed by the
//! returned `cache_key`. The LLM retrieves any dropped piece by
//! issuing a tool call against the runtime layer, which queries the
//! same store.
//!
//! # Per-domain bloat
//!
//! Each offload carries a cheap, structural [`estimate_bloat`] method
//! the orchestrator runs in parallel with the reformat phase. The
//! estimate is the gating signal: if it falls below a configurable
//! threshold AND reformat shrunk enough on its own, the orchestrator
//! skips this offload entirely (no parse, no store write, no marker).
//!
//! [`estimate_bloat`]: super::traits::OffloadTransform::estimate_bloat

pub mod diff_noise;
pub mod diff_offload;
pub mod json_offload;
pub mod log_offload;
pub mod search_offload;

pub use diff_noise::DiffNoise;
pub use diff_offload::DiffOffload;
pub use json_offload::JsonOffload;
pub use log_offload::LogOffload;
// `SearchOffload` is intentionally NOT re-exported here. The
// orchestrator-default registration omits it; keep the type accessible
// via the explicit module path for opt-in callers, but discourage new
// adoption (see `search_offload.rs` head docs for rationale).
pub use search_offload::SearchOffload;
