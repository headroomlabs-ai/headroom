//! Compression transforms — Rust ports of `headroom.transforms.*`.
//!
//! # Guiding principle: information preservation > aggressive compression
//!
//! When in doubt, prefer keeping bytes. The fixtures lock the Python
//! algorithm's exact behavior, so this crate cannot drop information that
//! Python keeps. But the inverse is also true — we MUST drop everything
//! Python drops, even when it feels lossy. Stage 3a's faithful port is
//! parity-bound. A follow-up stage (token-budget-aware compression) is
//! where we earn the right to keep more.
//!
//! Observability is the escape hatch: every transform returns a sidecar
//! `Stats` struct with the granular metrics Python doesn't emit (e.g. which
//! files were dropped, how many context lines were trimmed, per-file hunk
//! drop counts). These flow through `tracing` spans for OTel scraping in
//! prod and are returned alongside the parity-equal output for tests.

pub mod adaptive_sizer;
pub mod anchor_selector;
pub mod content_detector;
pub mod detection;
pub mod diff_compressor;
pub mod live_zone;
pub mod log_compressor;
pub mod magika_detector;
pub mod pipeline;
pub mod recommendations;
pub mod safety;
pub mod search_compressor;
pub mod smart_crusher;
pub mod tag_protector;
pub mod unidiff_detector;

pub use content_detector::{
    detect_content_type, is_json_array_of_dicts, ContentType, DetectionResult,
};
pub use detection::detect;
pub use diff_compressor::{
    DiffCompressionResult, DiffCompressor, DiffCompressorConfig, DiffCompressorStats,
};
pub use live_zone::{
    compress_anthropic_live_zone, compress_openai_chat_live_zone,
    compress_openai_responses_live_zone, AuthMode, BlockAction, BlockOutcome, CompressionManifest,
    ExclusionReason, LiveZoneError, LiveZoneOutcome,
};
pub use log_compressor::{
    LogCompressionResult, LogCompressor, LogCompressorConfig, LogCompressorStats, LogFormat,
    LogLevel, LogLine,
};
pub use magika_detector::{magika_detect, map_magika_label, MagikaDetectorError};
pub use pipeline::{
    CompressionContext, CompressionPipeline, CompressionPipelineBuilder, DiffNoise, DiffOffload,
    JsonMinifier, JsonOffload, LogOffload, LogTemplate, OffloadOutput, OffloadTransform,
    PipelineConfig, PipelineResult, ReformatOutput, ReformatTransform, TransformError,
};
pub use recommendations::{Recommendation, RecommendationStore, RECOMMENDATIONS_PATH_ENV_VAR};
pub use safety::{tool_pair_indices, ToolPair};
pub use search_compressor::{
    FileMatches, SearchCompressionResult, SearchCompressor, SearchCompressorConfig,
    SearchCompressorStats, SearchMatch,
};
pub use tag_protector::{is_known_html_tag, protect_tags, restore_tags, ProtectStats};
pub use unidiff_detector::{detect_diff, is_diff};
