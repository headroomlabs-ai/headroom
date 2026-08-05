//! Compression pipeline — formal orchestrator for reformat + bloat-gated CCR offload.
//!
//! # The architecture in one paragraph
//!
//! With CCR (Compress-Cache-Retrieve), no transform here destroys
//! information. Bytes drop from the wire, but the original payload is
//! stashed in a [`crate::ccr::CcrStore`] keyed by a hash. The LLM
//! retrieves any dropped piece via a tool call. So we don't have
//! "lossless" vs "lossy" — we have two distinct *mechanisms*:
//!
//! * [`ReformatTransform`] — pack denser without dropping anything.
//!   Output bytes are semantically equivalent to input bytes
//!   (`JsonMinifier` removes whitespace; future entries: log RLE,
//!   schema extraction, comment stripping).
//! * [`OffloadTransform`] — drop bytes from the wire, stash the
//!   original via CCR, emit a retrieval marker. Required to expose a
//!   cheap, **domain-specific** [`estimate_bloat`] method so the
//!   orchestrator can decide whether the offload is worth the
//!   retrieval round trip.
//!
//! [`CompressionPipeline`] dispatches both kinds by content type. It
//! runs the reformat phase serially while running per-offload bloat
//! estimators in parallel via `rayon::join` — so large inputs don't
//! pay a sequential cost for the gating decision.
//!
//! # Why parallel + domain-specific bloat
//!
//! Different content shapes have different "is this bloaty?" signals.
//! A generic byte-redundancy heuristic (zlib over a sample) misses
//! domain semantics: a log full of unique-but-irrelevant lines doesn't
//! compress with zlib but should still trigger CCR. Each
//! [`OffloadTransform`] carries its own structural estimator —
//! [`crate::transforms::pipeline::offloads::LogOffload`] looks at line
//! repetition + priority dilution; `DiffOffload` looks at the
//! context-to-change ratio; `SearchOffload` looks at how matches
//! cluster across files.
//!
//! Estimators MUST be cheap (under O(n) on input length, no
//! allocations beyond the structural read). They run in parallel with
//! the reformat phase via `rayon::par_iter` — so a 100-offload pipeline
//! over a 1MB log doesn't pay 100× the scan cost.
//!
//! # No regex
//!
//! Per project convention. JsonMinifier is `serde_json` round-trip;
//! offload bloat estimators are byte-prefix checks and
//! `signals::LineImportanceDetector` lookups (which use aho-corasick +
//! ASCII word boundary).
//!
//! # Coverage today vs deferred
//!
//! Reformats:
//! - [`reformats::JsonMinifier`] — JSON whitespace stripping.
//! - [`reformats::LogTemplate`] — Drain-style template miner for
//!   build/log output. Lossless — emits `[Template Tn: ...] (Nx)` +
//!   variant table, every original line reconstructible.
//!
//! Offloads:
//! - [`offloads::JsonOffload`] — wraps `SmartCrusher` for JSON arrays
//!   of dicts. Estimator counts row separators; apply delegates the
//!   heavy work to SmartCrusher (schema dedup, row sampling,
//!   anchor-aware selection) and adds a wrapper-level CCR marker
//!   that resolves in the orchestrator's store.
//! - [`offloads::LogOffload`] — wraps the existing `LogCompressor`,
//!   gates on per-line bloat heuristic.
//! - [`offloads::DiffOffload`] — wraps the existing `DiffCompressor`,
//!   gates on context-to-change ratio. Stores under the cache_key the
//!   wrapped compressor mints (closes a leak in the parity-bound port).
//! - [`offloads::DiffNoise`] — drops lockfile + whitespace-only hunks
//!   via CCR. Runs alongside `DiffOffload`; both are useful for
//!   different shapes of diff bloat.
//! - `SearchOffload` exists at `offloads::search_offload::SearchOffload`
//!   but is NOT in the default re-exports — modern agents use scoped
//!   `rg`/`grep`, the marginal value didn't justify default registration.
//!
//!
//! [`estimate_bloat`]: traits::OffloadTransform::estimate_bloat

pub mod config;
pub mod offloads;
pub mod orchestrator;
pub mod reformats;
pub mod traits;

pub use config::{
    BloatConfigs, ConfigError, DiffBloatConfig, DiffNoiseConfig, JsonOffloadConfig, LogBloatConfig,
    LogTemplateConfig, OffloadConfigs, OrchestratorConfig, PipelineConfig, ProseFieldConfig,
    ReformatConfigs, SearchBloatConfig,
};
// `SearchOffload` is intentionally NOT in the top-level re-export
// (deprecated from default pipeline; reach via the explicit module
// path if you want to opt in). See `offloads::search_offload` head
// docs for rationale.
pub use offloads::{DiffNoise, DiffOffload, JsonOffload, LogOffload, ProseFieldOffload};
pub use orchestrator::{CompressionPipeline, CompressionPipelineBuilder, PipelineResult};
pub use reformats::{JsonMinifier, LogTemplate};
pub use traits::{
    CompressionContext, OffloadOutput, OffloadTransform, ReformatOutput, ReformatTransform,
    TransformError,
};
