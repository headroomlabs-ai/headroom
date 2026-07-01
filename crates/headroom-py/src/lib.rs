//! PyO3 bindings for headroom-core. Exposed to Python as `headroom._core`.
//!
//! # Stage 3b — diff_compressor bridge
//!
//! The `DiffCompressor` family is exported here so the Python
//! `ContentRouter` can route to the Rust implementation in-process via
//! PyO3 instead of running the Python port. Backend selection happens in
//! `headroom.transforms._rust_diff_compressor.RustBackedDiffCompressor`,
//! which mirrors the Python `DiffCompressor` API one-for-one (so callers
//! don't notice the swap).
//!
//! Why in-process: ContentRouter compresses on the proxy's hot path. Any
//! IPC / subprocess / RPC bridge would dominate the cost we're trying to
//! save. PyO3 calls cost ~microseconds; staying in-process is ~free.

use std::collections::BTreeMap;

use headroom_core::signals::{
    ImportanceCategory, ImportanceContext, KeywordDetector, KeywordRegistry, LineImportanceDetector,
};
use headroom_core::transforms::smart_crusher::compaction::{
    ClassifyConfig, CompactConfig, DocumentCompactor,
};
use headroom_core::transforms::smart_crusher::{
    CrushResult as RustCrushResult, SmartCrusher as RustSmartCrusher,
    SmartCrusherConfig as RustSmartCrusherConfig,
};
use headroom_core::transforms::tag_protector::{
    is_known_html_tag as rust_is_known_html_tag, known_html_tag_names as rust_known_html_tag_names,
    protect_tags as rust_protect_tags, restore_tags as rust_restore_tags,
};
use headroom_core::transforms::{
    compress_openai_responses_live_zone as rust_compress_openai_responses_live_zone,
    detect as rust_detect_chain, is_json_array_of_dicts as rust_is_json_array_of_dicts,
    summarize_openai_responses_no_change_reason as rust_summarize_openai_responses_no_change_reason,
    AuthMode as RustLiveZoneAuthMode, ContentType as RustContentType,
    DetectionResult as RustDetectionResult, DiffCompressionResult, DiffCompressor,
    DiffCompressorConfig, DiffCompressorStats, LiveZoneOutcome,
    LogCompressionResult as RustLogResult, LogCompressor as RustLogCompressor,
    LogCompressorConfig as RustLogConfig, LogCompressorStats as RustLogStats,
    LogFormat as RustLogFormat, LogLevel as RustLogLevel,
    SearchCompressionResult as RustSearchResult, SearchCompressor as RustSearchCompressor,
    SearchCompressorConfig as RustSearchConfig, SearchCompressorStats as RustSearchStats,
};
use headroom_core::transforms::{
    TextCrusher as RustTextCrusher, TextCrusherConfig as RustTextCrusherConfig,
    TextCrusherResult as RustTextCrusherResult,
};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

/// Identity stub used by the Python smoke test to verify linkage.
#[pyfunction]
fn hello() -> &'static str {
    headroom_core::hello()
}

fn type_name(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

/// Build the dict returned by `SmartCrusher.crush_array_json`. Kept
/// outside `#[pymethods]` so we can `unwrap()` `set_item` (it cannot
/// fail when keys are static str literals and values are owned String /
/// Option<String> / Option<&'static str>) without tripping the
/// `clippy::useless_conversion` false positive that fires inside the
/// pyo3 0.22 method-attribute macro.
fn build_crush_array_dict<'py>(
    py: Python<'py>,
    kept_json: String,
    ccr_hash: Option<String>,
    dropped_summary: String,
    strategy_info: String,
    compacted: Option<String>,
    compaction_kind: Option<&'static str>,
) -> Bound<'py, PyDict> {
    let dict = PyDict::new(py);
    dict.set_item("items", kept_json).unwrap();
    dict.set_item("ccr_hash", ccr_hash).unwrap();
    dict.set_item("dropped_summary", dropped_summary).unwrap();
    dict.set_item("strategy_info", strategy_info).unwrap();
    dict.set_item("compacted", compacted).unwrap();
    dict.set_item("compaction_kind", compaction_kind).unwrap();
    dict
}

// ─── DiffCompressorConfig ──────────────────────────────────────────────────

/// Mirror of `headroom.transforms.diff_compressor.DiffCompressorConfig`.
/// Defaults match Python; constructor accepts every field as a kwarg with
/// the same name and type as the Python dataclass for drop-in
/// compatibility.
#[pyclass(
    name = "DiffCompressorConfig",
    module = "headroom._core",
    from_py_object
)]
#[derive(Clone)]
struct PyDiffCompressorConfig {
    inner: DiffCompressorConfig,
}

#[pymethods]
impl PyDiffCompressorConfig {
    #[new]
    #[pyo3(signature = (
        max_context_lines = 2,
        max_hunks_per_file = 10,
        max_files = 20,
        always_keep_additions = true,
        always_keep_deletions = true,
        enable_ccr = true,
        min_lines_for_ccr = 50,
        min_compression_ratio_for_ccr = 0.8,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        max_context_lines: usize,
        max_hunks_per_file: usize,
        max_files: usize,
        always_keep_additions: bool,
        always_keep_deletions: bool,
        enable_ccr: bool,
        min_lines_for_ccr: usize,
        min_compression_ratio_for_ccr: f64,
    ) -> Self {
        Self {
            inner: DiffCompressorConfig {
                max_context_lines,
                max_hunks_per_file,
                max_files,
                always_keep_additions,
                always_keep_deletions,
                enable_ccr,
                min_lines_for_ccr,
                min_compression_ratio_for_ccr,
            },
        }
    }

    // Read-only field accessors mirroring the Python dataclass surface.
    #[getter]
    fn max_context_lines(&self) -> usize {
        self.inner.max_context_lines
    }
    #[getter]
    fn max_hunks_per_file(&self) -> usize {
        self.inner.max_hunks_per_file
    }
    #[getter]
    fn max_files(&self) -> usize {
        self.inner.max_files
    }
    #[getter]
    fn always_keep_additions(&self) -> bool {
        self.inner.always_keep_additions
    }
    #[getter]
    fn always_keep_deletions(&self) -> bool {
        self.inner.always_keep_deletions
    }
    #[getter]
    fn enable_ccr(&self) -> bool {
        self.inner.enable_ccr
    }
    #[getter]
    fn min_lines_for_ccr(&self) -> usize {
        self.inner.min_lines_for_ccr
    }
    #[getter]
    fn min_compression_ratio_for_ccr(&self) -> f64 {
        self.inner.min_compression_ratio_for_ccr
    }

    fn __repr__(&self) -> String {
        format!(
            "DiffCompressorConfig(max_context_lines={}, max_hunks_per_file={}, max_files={}, \
             always_keep_additions={}, always_keep_deletions={}, enable_ccr={}, \
             min_lines_for_ccr={}, min_compression_ratio_for_ccr={})",
            self.inner.max_context_lines,
            self.inner.max_hunks_per_file,
            self.inner.max_files,
            self.inner.always_keep_additions,
            self.inner.always_keep_deletions,
            self.inner.enable_ccr,
            self.inner.min_lines_for_ccr,
            self.inner.min_compression_ratio_for_ccr,
        )
    }
}

// ─── DiffCompressionResult ─────────────────────────────────────────────────

/// Mirror of `headroom.transforms.diff_compressor.DiffCompressionResult`.
/// Read-only on the Python side: ContentRouter consumes fields, doesn't
/// mutate. `compression_ratio` and `tokens_saved_estimate` are exposed as
/// methods (not `@property`) — Python callers reach them via `.method()`.
/// The Python adapter wraps and re-exposes them as properties for full
/// dataclass compatibility.
#[pyclass(name = "DiffCompressionResult", module = "headroom._core")]
struct PyDiffCompressionResult {
    inner: DiffCompressionResult,
}

#[pymethods]
impl PyDiffCompressionResult {
    #[getter]
    fn compressed(&self) -> &str {
        &self.inner.compressed
    }
    #[getter]
    fn original_line_count(&self) -> usize {
        self.inner.original_line_count
    }
    #[getter]
    fn compressed_line_count(&self) -> usize {
        self.inner.compressed_line_count
    }
    #[getter]
    fn files_affected(&self) -> usize {
        self.inner.files_affected
    }
    #[getter]
    fn additions(&self) -> usize {
        self.inner.additions
    }
    #[getter]
    fn deletions(&self) -> usize {
        self.inner.deletions
    }
    #[getter]
    fn hunks_kept(&self) -> usize {
        self.inner.hunks_kept
    }
    #[getter]
    fn hunks_removed(&self) -> usize {
        self.inner.hunks_removed
    }
    #[getter]
    fn cache_key(&self) -> Option<String> {
        self.inner.cache_key.clone()
    }

    /// Mirror of Python `@property compression_ratio`. Returns
    /// `compressed_line_count / original_line_count` (1.0 if input was
    /// empty).
    fn compression_ratio(&self) -> f64 {
        if self.inner.original_line_count == 0 {
            1.0
        } else {
            self.inner.compressed_line_count as f64 / self.inner.original_line_count as f64
        }
    }

    /// Mirror of Python `@property tokens_saved_estimate`. Same `chars *
    /// 40 / 4` heuristic; bytes-equivalent numeric result.
    fn tokens_saved_estimate(&self) -> usize {
        let saved = self
            .inner
            .original_line_count
            .saturating_sub(self.inner.compressed_line_count);
        (saved * 40) / 4
    }

    fn __repr__(&self) -> String {
        format!(
            "DiffCompressionResult(compressed=<{} chars>, original_line_count={}, \
             compressed_line_count={}, files_affected={}, additions={}, deletions={}, \
             hunks_kept={}, hunks_removed={}, cache_key={:?})",
            self.inner.compressed.len(),
            self.inner.original_line_count,
            self.inner.compressed_line_count,
            self.inner.files_affected,
            self.inner.additions,
            self.inner.deletions,
            self.inner.hunks_kept,
            self.inner.hunks_removed,
            self.inner.cache_key,
        )
    }
}

// ─── DiffCompressorStats ───────────────────────────────────────────────────

/// Mirror of Rust `DiffCompressorStats` — sidecar observability not
/// present in the Python dataclass. Returned only from `compress_with_stats`,
/// which the Python adapter exposes as a method on the wrapper. `Vec`s are
/// returned as Python lists; the `BTreeMap` becomes a `dict`.
#[pyclass(name = "DiffCompressorStats", module = "headroom._core")]
struct PyDiffCompressorStats {
    inner: DiffCompressorStats,
}

#[pymethods]
impl PyDiffCompressorStats {
    #[getter]
    fn input_lines(&self) -> usize {
        self.inner.input_lines
    }
    #[getter]
    fn output_lines(&self) -> usize {
        self.inner.output_lines
    }
    #[getter]
    fn compression_ratio(&self) -> f64 {
        self.inner.compression_ratio
    }
    #[getter]
    fn files_total(&self) -> usize {
        self.inner.files_total
    }
    #[getter]
    fn files_kept(&self) -> usize {
        self.inner.files_kept
    }
    #[getter]
    fn files_dropped(&self) -> Vec<String> {
        self.inner.files_dropped.clone()
    }
    #[getter]
    fn hunks_total(&self) -> usize {
        self.inner.hunks_total
    }
    #[getter]
    fn hunks_kept(&self) -> usize {
        self.inner.hunks_kept
    }
    #[getter]
    fn hunks_dropped(&self) -> usize {
        self.inner.hunks_dropped
    }
    #[getter]
    fn hunks_dropped_per_file(&self) -> BTreeMap<String, usize> {
        self.inner.hunks_dropped_per_file.clone()
    }
    #[getter]
    fn context_lines_input(&self) -> usize {
        self.inner.context_lines_input
    }
    #[getter]
    fn context_lines_kept(&self) -> usize {
        self.inner.context_lines_kept
    }
    #[getter]
    fn context_lines_trimmed(&self) -> usize {
        self.inner.context_lines_trimmed
    }
    #[getter]
    fn largest_hunk_kept_lines(&self) -> usize {
        self.inner.largest_hunk_kept_lines
    }
    #[getter]
    fn largest_hunk_dropped_lines(&self) -> usize {
        self.inner.largest_hunk_dropped_lines
    }
    #[getter]
    fn parse_warnings(&self) -> Vec<String> {
        self.inner.parse_warnings.clone()
    }
    #[getter]
    fn processing_duration_us(&self) -> u64 {
        self.inner.processing_duration_us
    }
    #[getter]
    fn cache_key_emitted(&self) -> bool {
        self.inner.cache_key_emitted
    }
    #[getter]
    fn ccr_skipped_reason(&self) -> Option<String> {
        self.inner.ccr_skipped_reason.clone()
    }
    #[getter]
    fn file_mode_normalizations(&self) -> Vec<(String, String)> {
        self.inner.file_mode_normalizations.clone()
    }
    #[getter]
    fn binary_files_simplified(&self) -> Vec<String> {
        self.inner.binary_files_simplified.clone()
    }
}

// ─── DiffCompressor ────────────────────────────────────────────────────────

/// Mirror of `headroom.transforms.diff_compressor.DiffCompressor`. The
/// Python adapter wraps this in `RustBackedDiffCompressor` so
/// `ContentRouter` can swap backends transparently.
#[pyclass(name = "DiffCompressor", module = "headroom._core")]
struct PyDiffCompressor {
    inner: DiffCompressor,
}

#[pymethods]
impl PyDiffCompressor {
    /// `__init__(config: DiffCompressorConfig | None = None)` — matches the
    /// Python constructor signature one-for-one.
    #[new]
    #[pyo3(signature = (config = None))]
    fn new(config: Option<&PyDiffCompressorConfig>) -> Self {
        let cfg = config.map(|c| c.inner.clone()).unwrap_or_default();
        Self {
            inner: DiffCompressor::new(cfg),
        }
    }

    /// `compress(content: str, context: str = "") -> DiffCompressionResult`.
    /// Argument order and keyword names match the Python implementation.
    ///
    /// Releases the GIL across the Rust compress call so concurrent
    /// Python threads (uvicorn workers, asyncio tasks) can keep
    /// running while we hash + parse + filter the diff. The
    /// `&str` inputs are copied to owned `String`s first because
    /// PyO3 ties their lifetime to the GIL hold.
    #[pyo3(signature = (content, context = ""))]
    fn compress(&self, py: Python<'_>, content: &str, context: &str) -> PyDiffCompressionResult {
        let content = content.to_string();
        let context = context.to_string();
        let inner = py.detach(|| self.inner.compress(&content, &context));
        PyDiffCompressionResult { inner }
    }

    /// `compress_with_stats(content, context="") -> (result, stats)`.
    /// Sidecar API not present in Python — exposes the Rust observability
    /// struct alongside the parity-equal result. Returned as a 2-tuple to
    /// keep the call site Pythonic.
    #[pyo3(signature = (content, context = ""))]
    fn compress_with_stats(
        &self,
        py: Python<'_>,
        content: &str,
        context: &str,
    ) -> (PyDiffCompressionResult, PyDiffCompressorStats) {
        let content = content.to_string();
        let context = context.to_string();
        let (result, stats) = py.detach(|| self.inner.compress_with_stats(&content, &context));
        (
            PyDiffCompressionResult { inner: result },
            PyDiffCompressorStats { inner: stats },
        )
    }
}

// ─── SmartCrusherConfig ────────────────────────────────────────────────────

/// Mirror of `headroom.transforms.smart_crusher.SmartCrusherConfig`.
/// Defaults match Python's dataclass byte-for-byte. The constructor
/// accepts every field as a kwarg with the same name and type so the
/// Python shim can pass `SmartCrusherConfig(**asdict(py_cfg))`.
#[pyclass(name = "SmartCrusherConfig", module = "headroom._core", from_py_object)]
#[derive(Clone)]
struct PySmartCrusherConfig {
    inner: RustSmartCrusherConfig,
}

#[pymethods]
impl PySmartCrusherConfig {
    #[new]
    #[pyo3(signature = (
        enabled = true,
        min_items_to_analyze = 5,
        min_tokens_to_crush = 200,
        variance_threshold = 2.0,
        uniqueness_threshold = 0.1,
        similarity_threshold = 0.8,
        max_items_after_crush = 15,
        preserve_change_points = true,
        factor_out_constants = false,
        include_summaries = false,
        use_feedback_hints = true,
        toin_confidence_threshold = 0.5,
        dedup_identical_items = true,
        first_fraction = 0.3,
        last_fraction = 0.15,
        relevance_threshold = 0.3,
        lossless_min_savings_ratio = 0.15,
        enable_ccr_marker = true,
        lossless_only = false,
        compaction_core_field_fraction = 0.8,
        compaction_heterogeneous_core_ratio = 0.6,
        compaction_max_flatten_inner_keys = 6,
        compaction_min_buckets = 2,
        compaction_max_buckets = 8,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        enabled: bool,
        min_items_to_analyze: usize,
        min_tokens_to_crush: usize,
        variance_threshold: f64,
        uniqueness_threshold: f64,
        similarity_threshold: f64,
        max_items_after_crush: usize,
        preserve_change_points: bool,
        factor_out_constants: bool,
        include_summaries: bool,
        use_feedback_hints: bool,
        toin_confidence_threshold: f64,
        dedup_identical_items: bool,
        first_fraction: f64,
        last_fraction: f64,
        relevance_threshold: f64,
        lossless_min_savings_ratio: f64,
        enable_ccr_marker: bool,
        lossless_only: bool,
        compaction_core_field_fraction: f64,
        compaction_heterogeneous_core_ratio: f64,
        compaction_max_flatten_inner_keys: usize,
        compaction_min_buckets: usize,
        compaction_max_buckets: usize,
    ) -> Self {
        Self {
            inner: RustSmartCrusherConfig {
                enabled,
                min_items_to_analyze,
                min_tokens_to_crush,
                variance_threshold,
                uniqueness_threshold,
                similarity_threshold,
                max_items_after_crush,
                preserve_change_points,
                factor_out_constants,
                include_summaries,
                use_feedback_hints,
                toin_confidence_threshold,
                dedup_identical_items,
                first_fraction,
                last_fraction,
                relevance_threshold,
                lossless_min_savings_ratio,
                enable_ccr_marker,
                lossless_only,
                compaction_core_field_fraction,
                compaction_heterogeneous_core_ratio,
                compaction_max_flatten_inner_keys,
                compaction_min_buckets,
                compaction_max_buckets,
            },
        }
    }

    #[getter]
    fn enabled(&self) -> bool {
        self.inner.enabled
    }
    #[getter]
    fn min_items_to_analyze(&self) -> usize {
        self.inner.min_items_to_analyze
    }
    #[getter]
    fn min_tokens_to_crush(&self) -> usize {
        self.inner.min_tokens_to_crush
    }
    #[getter]
    fn variance_threshold(&self) -> f64 {
        self.inner.variance_threshold
    }
    #[getter]
    fn uniqueness_threshold(&self) -> f64 {
        self.inner.uniqueness_threshold
    }
    #[getter]
    fn similarity_threshold(&self) -> f64 {
        self.inner.similarity_threshold
    }
    #[getter]
    fn max_items_after_crush(&self) -> usize {
        self.inner.max_items_after_crush
    }
    #[getter]
    fn preserve_change_points(&self) -> bool {
        self.inner.preserve_change_points
    }
    #[getter]
    fn factor_out_constants(&self) -> bool {
        self.inner.factor_out_constants
    }
    #[getter]
    fn include_summaries(&self) -> bool {
        self.inner.include_summaries
    }
    #[getter]
    fn use_feedback_hints(&self) -> bool {
        self.inner.use_feedback_hints
    }
    #[getter]
    fn toin_confidence_threshold(&self) -> f64 {
        self.inner.toin_confidence_threshold
    }
    #[getter]
    fn dedup_identical_items(&self) -> bool {
        self.inner.dedup_identical_items
    }
    #[getter]
    fn first_fraction(&self) -> f64 {
        self.inner.first_fraction
    }
    #[getter]
    fn last_fraction(&self) -> f64 {
        self.inner.last_fraction
    }
    #[getter]
    fn relevance_threshold(&self) -> f64 {
        self.inner.relevance_threshold
    }
    #[getter]
    fn enable_ccr_marker(&self) -> bool {
        self.inner.enable_ccr_marker
    }
    #[getter]
    fn lossless_only(&self) -> bool {
        self.inner.lossless_only
    }
    #[getter]
    fn lossless_min_savings_ratio(&self) -> f64 {
        self.inner.lossless_min_savings_ratio
    }
    #[getter]
    fn compaction_core_field_fraction(&self) -> f64 {
        self.inner.compaction_core_field_fraction
    }
    #[getter]
    fn compaction_heterogeneous_core_ratio(&self) -> f64 {
        self.inner.compaction_heterogeneous_core_ratio
    }
    #[getter]
    fn compaction_max_flatten_inner_keys(&self) -> usize {
        self.inner.compaction_max_flatten_inner_keys
    }
    #[getter]
    fn compaction_min_buckets(&self) -> usize {
        self.inner.compaction_min_buckets
    }
    #[getter]
    fn compaction_max_buckets(&self) -> usize {
        self.inner.compaction_max_buckets
    }

    fn __repr__(&self) -> String {
        format!(
            "SmartCrusherConfig(enabled={}, min_items_to_analyze={}, \
             min_tokens_to_crush={}, max_items_after_crush={}, \
             relevance_threshold={})",
            self.inner.enabled,
            self.inner.min_items_to_analyze,
            self.inner.min_tokens_to_crush,
            self.inner.max_items_after_crush,
            self.inner.relevance_threshold,
        )
    }
}

// ─── CrushResult ───────────────────────────────────────────────────────────

/// Mirror of `headroom.transforms.smart_crusher.CrushResult`. Read-only;
/// the Python shim builds its own dataclass instance from these
/// attributes so callers that destructure with `asdict()` keep working.
#[pyclass(name = "CrushResult", module = "headroom._core")]
struct PyCrushResult {
    inner: RustCrushResult,
}

#[pymethods]
impl PyCrushResult {
    #[getter]
    fn compressed(&self) -> &str {
        &self.inner.compressed
    }
    #[getter]
    fn original(&self) -> &str {
        &self.inner.original
    }
    #[getter]
    fn was_modified(&self) -> bool {
        self.inner.was_modified
    }
    #[getter]
    fn strategy(&self) -> &str {
        &self.inner.strategy
    }

    fn __repr__(&self) -> String {
        format!(
            "CrushResult(compressed=<{} chars>, was_modified={}, strategy={:?})",
            self.inner.compressed.len(),
            self.inner.was_modified,
            self.inner.strategy,
        )
    }
}

// ─── SmartCrusher ──────────────────────────────────────────────────────────

/// Mirror of `headroom.transforms.smart_crusher.SmartCrusher`.
///
/// Constructor accepts only `config` — Python's `relevance_config`,
/// `scorer`, and `ccr_config` parameters are handled in the Python
/// shim (Stage 3c.1 keeps the optional subsystems disabled in Rust;
/// the shim drops those args to preserve call-site compatibility).
#[pyclass(name = "SmartCrusher", module = "headroom._core")]
struct PySmartCrusher {
    inner: RustSmartCrusher,
}

#[pymethods]
impl PySmartCrusher {
    #[new]
    #[pyo3(signature = (config = None))]
    fn new(config: Option<&PySmartCrusherConfig>) -> Self {
        let cfg = config.map(|c| c.inner.clone()).unwrap_or_default();
        Self {
            inner: RustSmartCrusher::new(cfg),
        }
    }

    /// Construct WITHOUT the lossless-first compaction stage. The
    /// public `crush()` API runs the lossy path directly (still with
    /// CCR-Dropped retrieval markers populated when rows are dropped).
    /// Used by the legacy parity fixture harness — those fixtures
    /// were recorded against the pre-PR4 lossy-only behavior.
    #[staticmethod]
    #[pyo3(signature = (config = None))]
    fn without_compaction(config: Option<&PySmartCrusherConfig>) -> Self {
        let cfg = config.map(|c| c.inner.clone()).unwrap_or_default();
        Self {
            inner: RustSmartCrusher::without_compaction(cfg),
        }
    }

    /// Construct with the lossless-first compaction stage's formatter
    /// chosen by name: `"csv-schema"` (the `new()` default), `"json"`,
    /// or `"markdown-kv"`. Raises `ValueError` on unknown names so a
    /// misconfigured knob is visible instead of silently falling back.
    #[staticmethod]
    #[pyo3(signature = (config = None, format_name = "csv-schema"))]
    fn with_compaction_format(
        config: Option<&PySmartCrusherConfig>,
        format_name: &str,
    ) -> PyResult<Self> {
        let cfg = config.map(|c| c.inner.clone()).unwrap_or_default();
        match RustSmartCrusher::with_compaction_format(cfg, format_name) {
            Some(inner) => Ok(Self { inner }),
            None => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown compaction format {format_name:?}; expected one of: {}",
                headroom_core::transforms::smart_crusher::compaction::CompactionStage::SUPPORTED_FORMAT_NAMES.join(", ")
            ))),
        }
    }

    /// `crush(content, query="", bias=1.0) -> CrushResult`. Argument
    /// order and keyword names mirror the Python implementation.
    ///
    /// Releases the GIL across the Rust crush call. Concurrent Python
    /// threads in the proxy keep running during the JSON parse +
    /// recursive process_value + per-array compression work. `&str`
    /// inputs are copied to owned `String`s up-front since PyO3 ties
    /// their lifetime to the GIL hold.
    #[pyo3(signature = (content, query = "", bias = 1.0))]
    fn crush(&self, py: Python<'_>, content: &str, query: &str, bias: f64) -> PyCrushResult {
        let content = content.to_string();
        let query = query.to_string();
        let inner = py.detach(|| self.inner.crush(&content, &query, bias));
        PyCrushResult { inner }
    }

    /// `smart_crush_content(content, query="", bias=1.0) -> (str, bool, str)`.
    /// Mirrors Python's `_smart_crush_content` — used by
    /// `smart_crush_tool_output` convenience function and direct
    /// callers that want the tuple form. Releases the GIL across the
    /// compute (same rationale as `crush`).
    #[pyo3(signature = (content, query = "", bias = 1.0))]
    fn smart_crush_content(
        &self,
        py: Python<'_>,
        content: &str,
        query: &str,
        bias: f64,
    ) -> (String, bool, String) {
        let content = content.to_string();
        let query = query.to_string();
        py.detach(|| self.inner.smart_crush_content(&content, &query, bias))
    }

    /// Crush a JSON array directly and return the structured result.
    ///
    /// Input is a JSON string holding an array (`[item, item, ...]`).
    /// Returns a dict with:
    /// - `items`: JSON array string of the kept rows after compression
    /// - `ccr_hash`: 12-char hash if rows were dropped, else `None`
    /// - `dropped_summary`: `<<ccr:HASH N_rows_offloaded>>` marker
    ///   text, empty if nothing dropped
    /// - `strategy_info`: debug string describing what ran (e.g.
    ///   `"smart_sample"`, `"lossless:table"`, `"none:adaptive_at_limit"`)
    /// - `compacted`: rendered bytes when the lossless path won, else `None`
    /// - `compaction_kind`: `"table" | "buckets" | "ccr" | None`
    ///
    /// This surfaces `CrushArrayResult` to Python so tests and the proxy
    /// runtime can reach the CCR hash directly (rather than parsing it
    /// out of the prompt marker).
    #[pyo3(signature = (items_json, query = "", bias = 1.0))]
    fn crush_array_json<'py>(
        &self,
        py: Python<'py>,
        items_json: &str,
        query: &str,
        bias: f64,
    ) -> Bound<'py, PyDict> {
        // GIL-release pattern: own the inputs, do all heavy compute
        // (JSON parse, crush, re-serialize) without the GIL, then
        // re-acquire to build the PyDict from the owned outputs.
        let items_json = items_json.to_string();
        let query = query.to_string();
        let (kept_json, ccr_hash, dropped_summary, strategy_info, compacted, compaction_kind) = py
            .detach(|| {
                let parsed: serde_json::Value = serde_json::from_str(&items_json)
                    .unwrap_or_else(|e| panic!("items_json must be JSON: {e}"));
                let items = match parsed {
                    serde_json::Value::Array(a) => a,
                    other => panic!("items_json must be a JSON array, got {}", type_name(&other)),
                };
                let result = self.inner.crush_array(&items, &query, bias);
                let kept_json = serde_json::to_string(&serde_json::Value::Array(result.items))
                    .expect("serialize kept items");
                (
                    kept_json,
                    result.ccr_hash,
                    result.dropped_summary,
                    result.strategy_info,
                    result.compacted,
                    result.compaction_kind,
                )
            });
        build_crush_array_dict(
            py,
            kept_json,
            ccr_hash,
            dropped_summary,
            strategy_info,
            compacted,
            compaction_kind,
        )
    }

    /// Run the document-level walker on `doc_json` (JSON string) and
    /// return the compacted document as JSON.
    ///
    /// The walker recursively descends through objects, arrays, and
    /// strings; tabular sub-arrays become rendered CSV+schema strings,
    /// long opaque blobs become `<<ccr:HASH,KIND,SIZE>>` markers (with
    /// originals stashed in this crusher's CCR store, so `ccr_get`
    /// resolves them).
    ///
    /// Distinct from `crush_array_json`: this is the lossless walker
    /// pass without per-array lossy crushing — useful when the caller
    /// wants document-shape compaction (forms, configs, mixed records)
    /// rather than statistical row drop.
    fn compact_document_json(&self, py: Python<'_>, doc_json: &str) -> String {
        // Heavy: JSON parse + recursive walker + tabular compaction +
        // re-serialize. None of it touches Python; release the GIL.
        let doc_json = doc_json.to_string();
        py.detach(|| {
            let parsed: serde_json::Value = serde_json::from_str(&doc_json)
                .unwrap_or_else(|e| panic!("doc_json must be JSON: {e}"));
            // Honor the crusher's CCR gate on the document-walker path: the
            // DocumentCompactor defaults to emit_opaque_markers=true, so without
            // this it would replace long string cells with <<ccr:>> markers even
            // when the caller disabled CCR.
            let mut dc = DocumentCompactor::new().with_config(CompactConfig {
                classify: ClassifyConfig {
                    emit_opaque_markers: self.inner.config.opaque_markers_enabled(),
                    ..ClassifyConfig::default()
                },
                ..CompactConfig::default()
            });
            if let Some(store) = self.inner.ccr_store() {
                dc = dc.with_ccr_store(store.clone());
            }
            let out = dc.compact(parsed);
            serde_json::to_string(&out).expect("serialize compacted document")
        })
    }

    /// Look up an original payload by CCR hash.
    ///
    /// When the lossy path drops rows, it stashes the **full original**
    /// array into the in-memory CCR store keyed by the 12-char hash
    /// embedded in the prompt's `<<ccr:HASH ...>>` marker. The runtime
    /// (proxy server / MCP retrieval tool) calls this to serve the
    /// dropped rows back to the LLM on demand.
    ///
    /// Returns the canonical-JSON serialization of the original
    /// `[item, item, ...]` array, or `None` if the hash is unknown,
    /// expired, or the crusher was constructed without a CCR store.
    fn ccr_get(&self, hash: &str) -> Option<String> {
        self.inner.ccr_store().and_then(|s| s.get(hash))
    }

    /// Number of entries currently held by the CCR store. `0` if no
    /// store is configured. Informational; use it from tests and
    /// telemetry, not from the retrieval hot path.
    fn ccr_len(&self) -> usize {
        self.inner.ccr_store().map(|s| s.len()).unwrap_or(0)
    }
}

// ─── ContentDetector ───────────────────────────────────────────────────────

/// Mirror of `headroom.transforms.content_detector.DetectionResult`.
///
/// Field names + types match the Python dataclass exactly so the existing
/// Python `ContentRouter` (which `import`s `DetectionResult` directly) can
/// continue to read `.content_type`, `.confidence`, and `.metadata` without
/// modification.
///
/// `content_type` is exposed as the lowercase string tag (e.g.
/// `"json_array"`). The Python wrapper translates it back into the
/// `ContentType` enum so the call-site looks identical.
#[pyclass(name = "DetectionResult", module = "headroom._core", from_py_object)]
#[derive(Clone)]
struct PyDetectionResult {
    inner: RustDetectionResult,
}

#[pymethods]
impl PyDetectionResult {
    #[getter]
    fn content_type(&self) -> &'static str {
        self.inner.content_type.as_str()
    }

    #[getter]
    fn confidence(&self) -> f64 {
        self.inner.confidence
    }

    /// Per-type metadata bag (e.g. `{"language": "python", "pattern_matches": 5}`
    /// for code, `{"item_count": 3, "is_dict_array": true}` for JSON arrays).
    /// Returned as a fresh `dict` so callers can mutate without affecting
    /// the underlying Rust value.
    #[getter]
    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (k, v) in &self.inner.metadata {
            // Convert each JSON value into the closest Python primitive.
            // Detection metadata is always a flat dict of scalars (ints,
            // bools, strings) so we don't need to recurse.
            match v {
                serde_json::Value::Bool(b) => dict.set_item(k, b)?,
                serde_json::Value::Number(n) => {
                    if let Some(i) = n.as_u64() {
                        dict.set_item(k, i)?
                    } else if let Some(i) = n.as_i64() {
                        dict.set_item(k, i)?
                    } else if let Some(f) = n.as_f64() {
                        dict.set_item(k, f)?
                    } else {
                        dict.set_item(k, py.None())?
                    }
                }
                serde_json::Value::String(s) => dict.set_item(k, s)?,
                serde_json::Value::Null => dict.set_item(k, py.None())?,
                // Detection never emits arrays / objects in metadata
                // today; if it ever does, fall through to JSON-string for
                // visibility rather than silently dropping.
                other => dict.set_item(k, other.to_string())?,
            };
        }
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!(
            "DetectionResult(content_type={:?}, confidence={}, metadata=<{} keys>)",
            self.inner.content_type.as_str(),
            self.inner.confidence,
            self.inner.metadata.len()
        )
    }
}

/// Detect the type of `content`. Returns a `DetectionResult` with the
/// same field surface as Python's dataclass.
///
/// Stage-3d (PR5) wired this through the magika→unidiff→PlainText
/// detection chain — the regex `content_detector` is no longer on
/// the production path. The chain returns a `ContentType` only;
/// we synthesize the legacy `DetectionResult` shape here with
/// `confidence = 1.0` (the chain doesn't surface a probabilistic
/// score) and an empty metadata bag (no production caller reads
/// metadata from this binding today — see audit notes in
/// `headroom/transforms/content_router.py`).
///
/// Releases the GIL while detecting — magika inference and unidiff
/// parsing can be substantial on large bodies, and freeing the GIL
/// lets other Python threads make progress in the meantime.
#[pyfunction]
fn detect_content_type(py: Python<'_>, content: &str) -> PyDetectionResult {
    let owned = content.to_string();
    let content_type = py.detach(move || rust_detect_chain(&owned));
    PyDetectionResult {
        inner: RustDetectionResult {
            content_type,
            confidence: 1.0,
            metadata: serde_json::Map::new(),
        },
    }
}

/// Quick check: is `content` a JSON array of dictionaries (the format
/// `SmartCrusher` natively handles)?
#[pyfunction]
fn is_json_array_of_dicts(py: Python<'_>, content: &str) -> bool {
    let owned = content.to_string();
    py.detach(move || rust_is_json_array_of_dicts(&owned))
}

// Suppress unused-import warning when ContentType isn't referenced
// directly — `as_str()` is the public surface.
const _: fn() = || {
    let _ = RustContentType::PlainText;
};

// ─── signals: line-importance detector bridge ────────────────────────────
//
// One process-wide [`KeywordDetector`] is shared via `OnceLock` because
// the underlying aho-corasick automaton is stateless and cheap to clone
// nothing on call. The Python shim re-exports the keyword tables and a
// pair of thin functions; that's enough surface for the legacy
// `error_detection` callers without dragging the trait into Python.

use std::sync::OnceLock;

fn shared_keyword_detector() -> &'static KeywordDetector {
    static DETECTOR: OnceLock<KeywordDetector> = OnceLock::new();
    DETECTOR.get_or_init(KeywordDetector::new)
}

/// Returns `Some(ctx)` for known names and `None` otherwise — caller
/// converts to PyValueError. Avoids the pyo3-0.22 + clippy
/// `useless_conversion` false positive that fires when `?` propagates a
/// `PyResult<_>` through another `PyResult<_>`.
fn ctx_from_str(name: &str) -> Option<ImportanceContext> {
    match name {
        "text" => Some(ImportanceContext::Text),
        "search" => Some(ImportanceContext::Search),
        "diff" => Some(ImportanceContext::Diff),
        "log" => Some(ImportanceContext::Log),
        _ => None,
    }
}

fn category_to_str(cat: ImportanceCategory) -> &'static str {
    match cat {
        ImportanceCategory::Error => "error",
        ImportanceCategory::Warning => "warning",
        ImportanceCategory::Importance => "importance",
        ImportanceCategory::Security => "security",
        ImportanceCategory::Markdown => "markdown",
    }
}

/// Score a line against the default Headroom keyword detector.
///
/// Returns `Some((category | None, priority, confidence))` for known
/// contexts (`text|search|diff|log`) and `None` for an unknown context
/// — the Python shim translates `None` into `ValueError` for the
/// caller. Returning `Option` instead of `PyResult` dodges the
/// pyo3-0.22 + clippy `useless_conversion` false positive that the
/// `#[pyfunction]` macro triggers when its inner result-shape carries
/// `PyErr`. The bridge layer is the right place for this conversion;
/// keeping the Rust signature panic-free and `PyResult`-free is worth
/// a one-line shim on the Python side.
#[pyfunction]
#[pyo3(signature = (line, context = "text"))]
fn score_line(line: &str, context: &str) -> Option<(Option<&'static str>, f32, f32)> {
    let ctx = ctx_from_str(context)?;
    let signal = shared_keyword_detector().score(line, ctx);
    Some((
        signal.category.map(category_to_str),
        signal.priority,
        signal.confidence,
    ))
}

/// Lax substring check: does `text` contain any error indicator? Mirrors
/// Python `error_detection.content_has_error_indicators`.
#[pyfunction]
fn content_has_error_indicators(text: &str) -> bool {
    shared_keyword_detector().contains_error_indicator(text)
}

/// Snapshot of the default keyword sets, exposed as a dict so the Python
/// shim can recompile the legacy `re.Pattern` objects without
/// re-declaring keyword data on the Python side. Uses `.unwrap()` on
/// `set_item` because keys are static str literals and values are
/// `Vec<&'static str>`, which can't fail — and avoids the pyo3-0.22
/// `useless_conversion` clippy false positive.
#[pyfunction]
fn keyword_registry_snapshot(py: Python<'_>) -> Py<PyDict> {
    let registry = KeywordRegistry::default_set();
    let dict = PyDict::new(py);
    for (key, words) in registry.as_map() {
        dict.set_item(key, words).unwrap();
    }
    dict.unbind()
}

// ─── search_compressor bridge (Phase 3e.2) ────────────────────────────
//
// Mirrors `headroom.transforms.search_compressor.SearchCompressor` so the
// Python shim can swap in via PyO3. The Rust implementation consumes the
// `signals::LineImportanceDetector` trait for priority scoring (instead of
// the regex registry the Python original used) and fixes the Windows-path
// + dashes-in-filename parser bugs.
//
// CCR persistence is exposed via a callback hook because the proxy's
// `CompressionStore` already lives Python-side. The Rust crate holds no
// long-lived store reference; instead the caller passes the dict back
// through the result and the Python shim writes it to the existing
// store. This avoids dragging a second CCR backend into Rust before the
// Phase 3g pipeline formalization owns CCR end-to-end.

#[pyclass(
    name = "SearchCompressorConfig",
    module = "headroom._core",
    from_py_object
)]
#[derive(Clone)]
struct PySearchCompressorConfig {
    inner: RustSearchConfig,
}

#[pymethods]
impl PySearchCompressorConfig {
    #[new]
    #[pyo3(signature = (
        max_matches_per_file = 5,
        always_keep_first = true,
        always_keep_last = true,
        max_total_matches = 30,
        max_files = 15,
        context_keywords = vec![],
        boost_errors = true,
        enable_ccr = true,
        min_matches_for_ccr = 10,
        min_compression_ratio_for_ccr = 0.8,
        group_by_file = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        max_matches_per_file: usize,
        always_keep_first: bool,
        always_keep_last: bool,
        max_total_matches: usize,
        max_files: usize,
        context_keywords: Vec<String>,
        boost_errors: bool,
        enable_ccr: bool,
        min_matches_for_ccr: usize,
        min_compression_ratio_for_ccr: f64,
        group_by_file: bool,
    ) -> Self {
        Self {
            inner: RustSearchConfig {
                max_matches_per_file,
                always_keep_first,
                always_keep_last,
                max_total_matches,
                max_files,
                context_keywords,
                boost_errors,
                enable_ccr,
                min_matches_for_ccr,
                min_compression_ratio_for_ccr,
                group_by_file,
            },
        }
    }
}

#[pyclass(name = "SearchCompressionResult", module = "headroom._core")]
struct PySearchCompressionResult {
    inner: RustSearchResult,
    stats: RustSearchStats,
}

#[pymethods]
impl PySearchCompressionResult {
    #[getter]
    fn compressed(&self) -> &str {
        &self.inner.compressed
    }
    #[getter]
    fn original(&self) -> &str {
        &self.inner.original
    }
    #[getter]
    fn original_match_count(&self) -> usize {
        self.inner.original_match_count
    }
    #[getter]
    fn compressed_match_count(&self) -> usize {
        self.inner.compressed_match_count
    }
    #[getter]
    fn files_affected(&self) -> usize {
        self.inner.files_affected
    }
    #[getter]
    fn compression_ratio(&self) -> f64 {
        self.inner.compression_ratio
    }
    #[getter]
    fn cache_key(&self) -> Option<&str> {
        self.inner.cache_key.as_deref()
    }
    #[getter]
    fn summaries<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        let dict = PyDict::new(py);
        for (k, v) in &self.inner.summaries {
            dict.set_item(k, v).unwrap();
        }
        dict
    }
    /// Sidecar stats — same shape every Rust transform uses for OTel.
    #[getter]
    fn lines_unparsed(&self) -> usize {
        self.stats.lines_unparsed
    }
    #[getter]
    fn files_dropped(&self) -> usize {
        self.stats.files_dropped
    }
    #[getter]
    fn ccr_emitted(&self) -> bool {
        self.stats.ccr_emitted
    }
    #[getter]
    fn ccr_skip_reason(&self) -> Option<&str> {
        self.stats.ccr_skip_reason
    }
}

#[pyclass(name = "SearchCompressor", module = "headroom._core")]
struct PySearchCompressor {
    inner: RustSearchCompressor,
}

#[pymethods]
impl PySearchCompressor {
    #[new]
    #[pyo3(signature = (config = None))]
    fn new(config: Option<PySearchCompressorConfig>) -> Self {
        let cfg = config.map(|c| c.inner).unwrap_or_default();
        Self {
            inner: RustSearchCompressor::new(cfg),
        }
    }

    /// Compress `content`. CCR persistence is the caller's responsibility
    /// — the Rust side never writes to the store. If the result needs a
    /// CCR marker, `cache_key` will be populated and the Python shim
    /// writes the original to the existing `CompressionStore`. This
    /// matches Python's existing CCR plumbing and avoids dragging a
    /// second backend into the Rust crate.
    ///
    /// (Internally the Rust CcrStore trait is used for unit tests; the
    /// PyO3 surface stays Python-CCR-friendly.)
    #[pyo3(signature = (content, context = "", bias = 1.0))]
    fn compress(
        &self,
        py: Python<'_>,
        content: &str,
        context: &str,
        bias: f64,
    ) -> PySearchCompressionResult {
        // Synthesize a tiny in-memory store so the Rust path can
        // populate `cache_key`; the Python side reads `cache_key` and
        // writes the original to its own `CompressionStore` if it
        // wants persistence beyond the request lifecycle.
        let owned = content.to_string();
        let owned_ctx = context.to_string();
        let (result, stats) = py.detach(move || {
            let store = headroom_core::ccr::InMemoryCcrStore::new();
            let (r, s) = self
                .inner
                .compress_with_store(&owned, &owned_ctx, bias, Some(&store));
            (r, s)
        });
        PySearchCompressionResult {
            inner: result,
            stats,
        }
    }
}

/// Parse one grep/ripgrep line into `(file, line_number, content)`. Used
/// by the Python shim's `_parse_search_results` so the bug-fixed parser
/// runs even when callers use the legacy internal helpers (which exist
/// only for backwards-compat with the existing test surface).
#[pyfunction]
fn parse_search_lines(content: &str) -> Vec<(String, u64, String)> {
    let compressor = RustSearchCompressor::new(RustSearchConfig::default());
    let mut stats = RustSearchStats::default();
    let parsed = compressor.parse_search_results(content, &mut stats);
    let mut out = Vec::new();
    for fm in parsed.values() {
        for m in &fm.matches {
            out.push((m.file.clone(), m.line_number, m.content.clone()));
        }
    }
    out
}

// ─── log_compressor bridge (Phase 3e.5) ───────────────────────────────
//
// Mirrors `headroom.transforms.log_compressor.LogCompressor`. Same CCR
// pattern as search_compressor: Rust emits a `cache_key`, Python shim
// writes the original to the production `CompressionStore`.

#[pyclass(
    name = "LogCompressorConfig",
    module = "headroom._core",
    from_py_object
)]
#[derive(Clone)]
struct PyLogCompressorConfig {
    inner: RustLogConfig,
}

#[pymethods]
impl PyLogCompressorConfig {
    #[new]
    #[pyo3(signature = (
        max_errors = 10,
        error_context_lines = 3,
        keep_first_error = true,
        keep_last_error = true,
        max_stack_traces = 3,
        stack_trace_max_lines = 20,
        max_warnings = 5,
        dedupe_warnings = true,
        keep_summary_lines = true,
        max_total_lines = 100,
        enable_ccr = true,
        min_lines_for_ccr = 50,
        min_compression_ratio_for_ccr = 0.5,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        max_errors: usize,
        error_context_lines: usize,
        keep_first_error: bool,
        keep_last_error: bool,
        max_stack_traces: usize,
        stack_trace_max_lines: usize,
        max_warnings: usize,
        dedupe_warnings: bool,
        keep_summary_lines: bool,
        max_total_lines: usize,
        enable_ccr: bool,
        min_lines_for_ccr: usize,
        min_compression_ratio_for_ccr: f64,
    ) -> Self {
        Self {
            inner: RustLogConfig {
                max_errors,
                error_context_lines,
                keep_first_error,
                keep_last_error,
                max_stack_traces,
                stack_trace_max_lines,
                max_warnings,
                dedupe_warnings,
                keep_summary_lines,
                max_total_lines,
                enable_ccr,
                min_lines_for_ccr,
                min_compression_ratio_for_ccr,
            },
        }
    }
}

#[pyclass(name = "LogCompressionResult", module = "headroom._core")]
struct PyLogCompressionResult {
    inner: RustLogResult,
    stats: RustLogStats,
}

#[pymethods]
impl PyLogCompressionResult {
    #[getter]
    fn compressed(&self) -> &str {
        &self.inner.compressed
    }
    #[getter]
    fn original(&self) -> &str {
        &self.inner.original
    }
    #[getter]
    fn original_line_count(&self) -> usize {
        self.inner.original_line_count
    }
    #[getter]
    fn compressed_line_count(&self) -> usize {
        self.inner.compressed_line_count
    }
    #[getter]
    fn format_detected(&self) -> &'static str {
        self.inner.format_detected.as_str()
    }
    #[getter]
    fn compression_ratio(&self) -> f64 {
        self.inner.compression_ratio
    }
    #[getter]
    fn cache_key(&self) -> Option<&str> {
        self.inner.cache_key.as_deref()
    }
    #[getter]
    fn stats<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        let dict = PyDict::new(py);
        for (k, v) in &self.inner.stats {
            dict.set_item(k, v).unwrap();
        }
        dict
    }
    // Sidecar diagnostics
    #[getter]
    fn stack_traces_seen(&self) -> usize {
        self.stats.stack_traces_seen
    }
    #[getter]
    fn stack_traces_kept(&self) -> usize {
        self.stats.stack_traces_kept
    }
    #[getter]
    fn warnings_dropped_by_dedupe(&self) -> usize {
        self.stats.warnings_dropped_by_dedupe
    }
    #[getter]
    fn ccr_emitted(&self) -> bool {
        self.stats.ccr_emitted
    }
    #[getter]
    fn ccr_skip_reason(&self) -> Option<&str> {
        self.stats.ccr_skip_reason
    }
}

#[pyclass(name = "LogCompressor", module = "headroom._core")]
struct PyLogCompressor {
    inner: RustLogCompressor,
}

#[pymethods]
impl PyLogCompressor {
    #[new]
    #[pyo3(signature = (config = None))]
    fn new(config: Option<PyLogCompressorConfig>) -> Self {
        let cfg = config.map(|c| c.inner).unwrap_or_default();
        Self {
            inner: RustLogCompressor::new(cfg),
        }
    }

    /// Compress `content`. Same CCR pattern as search_compressor: Rust
    /// emits the `cache_key`; the Python shim is responsible for
    /// writing the original to the production `CompressionStore`.
    #[pyo3(signature = (content, bias = 1.0))]
    fn compress(&self, py: Python<'_>, content: &str, bias: f64) -> PyLogCompressionResult {
        let owned = content.to_string();
        let (result, stats) = py.detach(move || {
            let store = headroom_core::ccr::InMemoryCcrStore::new();
            let (r, s) = self.inner.compress_with_store(&owned, bias, Some(&store));
            (r, s)
        });
        PyLogCompressionResult {
            inner: result,
            stats,
        }
    }
}

/// Helper for the Python shim's `_detect_format`.
#[pyfunction]
fn detect_log_format(lines: Vec<String>) -> &'static str {
    let compressor = RustLogCompressor::new(RustLogConfig::default());
    let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
    compressor.detect_format(&refs).as_str()
}

/// Suppress unused-import warnings for the LogLevel/LogFormat imports
/// kept for future expansion (the Python shim consumes them via
/// detect_log_format and the result format_detected getter).
const _: fn() = || {
    let _ = RustLogFormat::Generic;
    let _ = RustLogLevel::Unknown;
};

// ─── tag_protector bridge (Phase 3e.4) ───────────────────────────────────
//
// Mirrors `headroom.transforms.tag_protector.{protect_tags,restore_tags,
// is_html_tag,KNOWN_HTML_TAGS}`. The Rust walker is single-pass and
// fixes five real bugs the Python original carried (see crate-level
// docs in `tag_protector.rs`). The GIL is released during the walk
// because the algorithm holds no Python references.

/// Replace custom workflow tags in `text` with opaque placeholders so
/// downstream ML compressors can't accidentally drop them.
///
/// Returns `(cleaned_text, blocks)` where `blocks` is a list of
/// `(placeholder, original)` tuples for `restore_tags`.
#[pyfunction]
#[pyo3(signature = (text, compress_tagged_content = false))]
fn protect_tags(
    py: Python<'_>,
    text: &str,
    compress_tagged_content: bool,
) -> (String, Vec<(String, String)>) {
    let owned = text.to_string();
    py.detach(move || {
        let (cleaned, blocks, _stats) = rust_protect_tags(&owned, compress_tagged_content);
        (cleaned, blocks)
    })
}

/// Splice protected blocks back into `text`. Missing placeholders fall
/// back to appending the original block (lossy-compression incident).
#[pyfunction]
fn restore_tags(py: Python<'_>, text: &str, blocks: Vec<(String, String)>) -> String {
    let owned = text.to_string();
    py.detach(move || rust_restore_tags(&owned, &blocks))
}

/// Case-insensitive HTML5 tag check. The Python shim uses this to
/// preserve the legacy private `_is_html_tag` import surface for tests.
#[pyfunction]
fn is_html_tag(name: &str) -> bool {
    rust_is_known_html_tag(name)
}

/// Return the canonical HTML5 tag name list. The Python shim
/// reconstructs `KNOWN_HTML_TAGS` from this so callers that import the
/// frozenset (the existing test does) continue to work without
/// re-declaring the set in two languages.
#[pyfunction]
fn known_html_tag_names() -> Vec<&'static str> {
    rust_known_html_tag_names().to_vec()
}

// ─── Module init ───────────────────────────────────────────────────────────

/// Apply OpenAI `/v1/responses` live-zone compression to a request body.
///
/// Hot-fix entry point added 2026-05-06: re-enables `/v1/responses`
/// compression on the Python proxy after PR-C5 retired the Python
/// pipeline. PR-C5's "Rust handles it" claim assumed the standalone
/// `crates/headroom-proxy` binary would sit in front of Python; that
/// binary is not deployed by the CLI (`headroom proxy`,
/// `headroom wrap codex`). This binding lets the Python proxy call
/// the live-zone dispatcher inline so Codex `/v1/responses` traffic
/// is compressed end-to-end.
///
/// # Arguments
/// * `body` — raw request body bytes (post memory-injection).
/// * `auth_mode` — one of `"payg"`, `"oauth"`, `"subscription"`,
///   `"unknown"`. Currently unused by the dispatcher (the policy
///   gating is upstream); accepted for forward-compat.
/// * `model` — model name from the request body. Empty string defaults
///   to `headroom_core::transforms::live_zone::DEFAULT_MODEL`.
///
/// # Returns
/// `(body, modified)`:
/// * Modified: `(new_body_bytes, True)` — caller forwards the new bytes.
/// * Unchanged / passthrough: `(input_bytes, False)` — caller forwards
///   the original.
///
/// # Failure mode
/// Never raises. The dispatcher's `LiveZoneError` outcomes (body not
/// JSON, no `messages`/`input` array) are passthrough conditions, not
/// failures — matching the Rust proxy's
/// `compress_openai_responses_request` contract.
#[pyfunction]
#[pyo3(signature = (body, auth_mode = "payg", model = ""))]
fn compress_openai_responses_live_zone(
    py: Python<'_>,
    body: &[u8],
    auth_mode: &str,
    model: &str,
) -> (Py<PyBytes>, bool, u64, Vec<String>, Option<String>) {
    let mode = match auth_mode.to_ascii_lowercase().as_str() {
        "payg" => RustLiveZoneAuthMode::Payg,
        "oauth" => RustLiveZoneAuthMode::OAuth,
        "subscription" => RustLiveZoneAuthMode::Subscription,
        _ => RustLiveZoneAuthMode::Unknown,
    };
    let model_str = if model.is_empty() {
        headroom_core::transforms::live_zone::DEFAULT_MODEL
    } else {
        model
    };

    match rust_compress_openai_responses_live_zone(body, mode, model_str) {
        Ok(LiveZoneOutcome::NoChange { manifest }) => {
            let saved = manifest.tokens_saved() as u64;
            let transforms: Vec<String> = manifest
                .transforms_applied()
                .into_iter()
                .map(String::from)
                .collect();
            let reason = rust_summarize_openai_responses_no_change_reason(&manifest).to_string();
            (
                PyBytes::new(py, body).unbind(),
                false,
                saved,
                transforms,
                Some(reason),
            )
        }
        Ok(LiveZoneOutcome::Modified { new_body, manifest }) => {
            // `RawValue::get` returns the underlying serialized JSON
            // as `&str`; bytes are valid UTF-8 by construction.
            let bytes = new_body.get().as_bytes();
            let saved = manifest.tokens_saved() as u64;
            let transforms: Vec<String> = manifest
                .transforms_applied()
                .into_iter()
                .map(String::from)
                .collect();
            (
                PyBytes::new(py, bytes).unbind(),
                true,
                saved,
                transforms,
                None,
            )
        }
        Err(_) => {
            // BodyNotJson / NoMessagesArray are non-fatal: nothing to
            // compress, fall through to passthrough byte-for-byte.
            (
                PyBytes::new(py, body).unbind(),
                false,
                0,
                Vec::new(),
                Some("dispatch_error".to_string()),
            )
        }
    }
}

// --- TextCrusher (Phase 2, #1171): fast extractive prose compressor ---

#[pyclass(name = "TextCrusherConfig", module = "headroom._core", from_py_object)]
#[derive(Clone)]
struct PyTextCrusherConfig {
    inner: RustTextCrusherConfig,
}

#[pymethods]
impl PyTextCrusherConfig {
    #[new]
    #[pyo3(signature = (
        target_ratio = 0.5,
        w_recency = 1.0,
        w_relevance = 2.0,
        w_salience = 1.5,
        min_segment_chars = 12,
        near_dup_threshold = 0.85,
        min_segments_for_crush = 6,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        target_ratio: f64,
        w_recency: f64,
        w_relevance: f64,
        w_salience: f64,
        min_segment_chars: usize,
        near_dup_threshold: f64,
        min_segments_for_crush: usize,
    ) -> Self {
        Self {
            inner: RustTextCrusherConfig {
                target_ratio,
                w_recency,
                w_relevance,
                w_salience,
                min_segment_chars,
                near_dup_threshold,
                min_segments_for_crush,
            },
        }
    }

    #[getter]
    fn target_ratio(&self) -> f64 {
        self.inner.target_ratio
    }
    #[getter]
    fn near_dup_threshold(&self) -> f64 {
        self.inner.near_dup_threshold
    }
    #[getter]
    fn min_segments_for_crush(&self) -> usize {
        self.inner.min_segments_for_crush
    }
    #[getter]
    fn w_recency(&self) -> f64 {
        self.inner.w_recency
    }
    #[getter]
    fn w_relevance(&self) -> f64 {
        self.inner.w_relevance
    }
    #[getter]
    fn w_salience(&self) -> f64 {
        self.inner.w_salience
    }
    #[getter]
    fn min_segment_chars(&self) -> usize {
        self.inner.min_segment_chars
    }
}

#[pyclass(name = "TextCrusherResult", module = "headroom._core")]
struct PyTextCrusherResult {
    inner: RustTextCrusherResult,
}

#[pymethods]
impl PyTextCrusherResult {
    #[getter]
    fn compressed(&self) -> String {
        self.inner.compressed.clone()
    }
    #[getter]
    fn original_tokens(&self) -> usize {
        self.inner.original_tokens
    }
    #[getter]
    fn compressed_tokens(&self) -> usize {
        self.inner.compressed_tokens
    }
    #[getter]
    fn compression_ratio(&self) -> f64 {
        self.inner.compression_ratio
    }
    #[getter]
    fn kept_segments(&self) -> usize {
        self.inner.kept_segments
    }
    #[getter]
    fn total_segments(&self) -> usize {
        self.inner.total_segments
    }
}

#[pyclass(name = "TextCrusher", module = "headroom._core")]
struct PyTextCrusher {
    inner: RustTextCrusher,
}

#[pymethods]
impl PyTextCrusher {
    #[new]
    #[pyo3(signature = (config = None))]
    fn new(config: Option<&PyTextCrusherConfig>) -> Self {
        let cfg = config.map(|c| c.inner.clone()).unwrap_or_default();
        Self {
            inner: RustTextCrusher::new(cfg),
        }
    }

    /// `compress(content, context="", target_ratio=None) -> TextCrusherResult`.
    /// Releases the GIL across the Rust compress call.
    #[pyo3(signature = (content, context = "", target_ratio = None))]
    fn compress(
        &self,
        py: Python<'_>,
        content: &str,
        context: &str,
        target_ratio: Option<f64>,
    ) -> PyTextCrusherResult {
        let content = content.to_string();
        let context = context.to_string();
        let inner = py.detach(|| self.inner.compress(&content, &context, target_ratio));
        PyTextCrusherResult { inner }
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Bridge Rust diagnostics into Python's `logging`. headroom-core emits
    // `tracing` events (e.g. magika init timeout warnings), but a cdylib has
    // no tracing subscriber, so they were silently dropped. The workspace
    // `tracing` dep now enables the `log` compat feature — events become
    // `log` records when no subscriber is active — and pyo3-log forwards
    // those to Python loggers (named like
    // `headroom_core.transforms.magika_detector`), which is what lands in
    // the proxy's log file. `try_init` because the global logger may
    // legitimately already be set (re-import, embedders).
    let _ = pyo3_log::try_init();

    m.add_function(wrap_pyfunction!(hello, m)?)?;
    m.add_class::<PyDiffCompressorConfig>()?;
    m.add_class::<PyDiffCompressionResult>()?;
    m.add_class::<PyDiffCompressorStats>()?;
    m.add_class::<PyDiffCompressor>()?;
    m.add_class::<PySearchCompressorConfig>()?;
    m.add_class::<PySearchCompressionResult>()?;
    m.add_class::<PySearchCompressor>()?;
    m.add_function(wrap_pyfunction!(parse_search_lines, m)?)?;
    m.add_class::<PySmartCrusherConfig>()?;
    m.add_class::<PyCrushResult>()?;
    m.add_class::<PySmartCrusher>()?;
    m.add_class::<PyTextCrusherConfig>()?;
    m.add_class::<PyTextCrusherResult>()?;
    m.add_class::<PyTextCrusher>()?;
    m.add_class::<PyDetectionResult>()?;
    m.add_class::<PyLogCompressorConfig>()?;
    m.add_class::<PyLogCompressionResult>()?;
    m.add_class::<PyLogCompressor>()?;
    m.add_function(wrap_pyfunction!(detect_log_format, m)?)?;
    m.add_function(wrap_pyfunction!(protect_tags, m)?)?;
    m.add_function(wrap_pyfunction!(restore_tags, m)?)?;
    m.add_function(wrap_pyfunction!(is_html_tag, m)?)?;
    m.add_function(wrap_pyfunction!(known_html_tag_names, m)?)?;
    m.add_function(wrap_pyfunction!(detect_content_type, m)?)?;
    m.add_function(wrap_pyfunction!(is_json_array_of_dicts, m)?)?;
    m.add_function(wrap_pyfunction!(score_line, m)?)?;
    m.add_function(wrap_pyfunction!(content_has_error_indicators, m)?)?;
    m.add_function(wrap_pyfunction!(keyword_registry_snapshot, m)?)?;
    m.add_function(wrap_pyfunction!(compress_openai_responses_live_zone, m)?)?;
    Ok(())
}
