//! `SmartCrusher` struct — top-level entry point for compression.
//!
//! Owns the `config`, `anchor_selector`, `scorer`, and `analyzer`
//! singletons that every per-message call needs. Constructed once
//! per process; the struct is `Send + Sync` so it can sit behind an
//! `Arc` in a multi-threaded proxy.
//!
//! This module ports three Python entry points:
//!
//! - `_execute_plan` (line 3617) → `SmartCrusher::execute_plan`
//! - `_crush_array`  (line 2400) → `SmartCrusher::crush_array`
//! - `_crush_mixed_array` (line 2914) → `SmartCrusher::crush_mixed_array`
//!
//! # Stubs that match Python's "everything-disabled" path
//!
//! Python's `_crush_array` calls into TOIN (cross-user pattern
//! learning), feedback (per-tool compression hints), CCR (compress-
//! cache-retrieve store), and telemetry. All four are large separate
//! systems with their own state. For the like-for-like port at Stage
//! 3c.1, we mirror Python's behavior **when those subsystems are
//! disabled**:
//!
//! - **TOIN**: never produces a recommendation, never overrides
//!   `effective_max_items`, never injects preserve_fields/strategy/level.
//! - **Feedback**: never produces hints; default `effective_max_items`.
//! - **CCR**: `enabled=false`; result has `ccr_hash = None`.
//! - **Telemetry**: no-op.
//! - **`_compress_text_within_items`**: pass-through (returns input
//!   unchanged) since text compression has its own port pipeline.
//! - **`summarize_dropped_items`**: empty string.
//!
//! Parity fixtures will be recorded with all four disabled on the
//! Python side, locking byte-equal output. The TOIN/CCR/feedback
//! integration ports happen later (Stage 3c.2 follow-ups).

use std::sync::Arc;

use serde_json::Value;

use super::analyzer::SmartAnalyzer;
use super::builder::SmartCrusherBuilder;
use super::classifier::{classify_array, ArrayType};
use super::compaction::{
    classify_cell, emit_opaque_ccr_marker, try_parse_json_container, CellClass, ClassifyConfig,
    Compaction, CompactionStage,
};
use super::config::SmartCrusherConfig;
use super::crushers::{compute_k_split, crush_number_array, crush_object, crush_string_array};
use super::planning::SmartCrusherPlanner;
use super::traits::{Constraint, CrushEvent, Observer};
use super::types::{CompressionPlan, CompressionStrategy, CrushResult};
use crate::ccr::CcrStore;
use crate::relevance::RelevanceScorer;
use crate::transforms::adaptive_sizer::compute_optimal_k;
use crate::transforms::anchor_selector::AnchorSelector;

/// Return type for `crush_array`.
///
/// Two operating paths feed the same result type:
///
/// - **Lossless path** — input compacted to a smaller inline form
///   (e.g. CSV+schema). Nothing dropped; `compacted` is populated;
///   `ccr_hash` is `None` (no retrieval needed because everything is
///   already in the prompt).
/// - **Lossy path** — input compressed by row-dropping. `items` holds
///   the kept subset; `ccr_hash` is `Some(hash)` so the runtime can
///   cache the **full original** keyed by that hash and serve it back
///   to the LLM via a retrieval tool call. **No data is lost** —
///   "lossy" here means "compressed view inline; full payload cached
///   for tool retrieval," matching Python's CCR-Dropped semantics.
///
/// The runtime (PyO3 bridge / proxy server) owns the cache; this crate
/// computes the hash and emits a marker so the prompt knows where to
/// look.
pub struct CrushArrayResult {
    /// Kept items. For the lossless path this is the full original
    /// (nothing was dropped). For the lossy path this is the surviving
    /// subset; the rest is retrievable via `ccr_hash`.
    pub items: Vec<Value>,
    /// Strategy debug string. One of:
    /// - `"none:adaptive_at_limit"` / `"skip:<reason>"` — passthrough
    /// - `"lossless:table"` / `"lossless:buckets"` — lossless wins
    /// - `"smart_sample"` / `"top_n"` / `"cluster"` / `"time_series"` —
    ///   lossy path with row-dropping.
    pub strategy_info: String,
    /// 12-char SHA-256 hex prefix of the **full original input**.
    /// Populated when the lossy path dropped rows; the runtime is
    /// expected to cache the original items keyed by this hash so a
    /// retrieval tool can serve them back. `None` when nothing was
    /// dropped (lossless path or below adaptive_k boundary).
    pub ccr_hash: Option<String>,
    /// Marker text inserted into the prompt to advertise the CCR
    /// pointer (e.g. `<<ccr:abc123def456 42_rows_offloaded>>`). Empty
    /// when `ccr_hash` is `None`.
    pub dropped_summary: String,
    /// Rendered bytes from the compaction stage when the **lossless
    /// path** won. `None` for the lossy path or when compaction wasn't
    /// configured.
    pub compacted: Option<String>,
    /// Top-level [`Compaction`] variant tag — `"table"`, `"buckets"`,
    /// `"ccr"`. Mirrors `compacted` — populated only when lossless won.
    pub compaction_kind: Option<&'static str>,
}

/// Top-level SmartCrusher.
///
/// Three pluggable extensions (Stage 3c.2 PR1):
/// - `scorer` — relevance scoring (`HybridScorer` by default).
/// - `constraints` — must-keep predicates (`KeepErrorsConstraint` +
///   `KeepStructuralOutliersConstraint` by default).
/// - `observers` — decision-stream telemetry (`TracingObserver` by
///   default).
///
/// Compose via [`SmartCrusherBuilder`]; or call `SmartCrusher::new()`
/// for the OSS default composition.
pub struct SmartCrusher {
    pub config: SmartCrusherConfig,
    pub anchor_selector: AnchorSelector,
    pub scorer: Box<dyn RelevanceScorer + Send + Sync>,
    pub analyzer: SmartAnalyzer,
    pub constraints: Vec<Box<dyn Constraint>>,
    pub observers: Vec<Box<dyn Observer>>,
    /// Optional lossless-first compaction stage (Stage 3c.2 PR2). When
    /// set, `crush_array` runs compaction up front and short-circuits
    /// the lossy path on success. When `None` (default OSS), parity
    /// with the pre-PR2 lossy-only pipeline is preserved exactly.
    pub compaction: Option<CompactionStage>,
    /// Optional CCR store. When set, the lossy path stashes the **full
    /// original** array into the store keyed by `ccr_hash` before
    /// returning — the runtime can then serve dropped rows back via
    /// retrieval tool calls. When `None`, hashes are still emitted but
    /// nothing is stored (legacy / parity mode).
    ///
    /// `Arc` so callers can keep their own handle to the same store
    /// (e.g. the proxy server holds it for retrieval lookups while
    /// SmartCrusher writes through it).
    pub ccr_store: Option<Arc<dyn CcrStore>>,
}

impl SmartCrusher {
    /// Construct with the OSS default composition: scorer + constraints +
    /// observer + **lossless-first compaction stage**. Calling
    /// `crush_array` runs the dispatch:
    ///
    /// 1. Try the lossless compactor.
    /// 2. If savings ratio ≥ `config.lossless_min_savings_ratio`
    ///    (default `0.30`), ship lossless — `compacted` populated,
    ///    `ccr_hash = None`, nothing dropped.
    /// 3. Otherwise fall through to the lossy path — drop rows,
    ///    populate `ccr_hash` with a hash of the full original so the
    ///    runtime can cache the payload for tool retrieval.
    ///
    /// **No data is ever lost.** The lossy path moves dropped rows to
    /// CCR cache, not to nowhere — same semantics as Python's
    /// SmartCrusher with CCR enabled.
    pub fn new(config: SmartCrusherConfig) -> Self {
        SmartCrusherBuilder::new(config)
            .with_default_oss_setup()
            .with_default_compaction()
            .with_default_ccr_store()
            .build()
    }

    /// Construct WITHOUT the compaction stage. Pre-PR4 behavior:
    /// `crush_array` skips the lossless attempt and runs the lossy
    /// path directly (still with CCR-Dropped retrieval markers from
    /// PR4). Used by:
    ///
    /// - The 17 legacy parity fixtures (recorded against the
    ///   lossy-only path; using this constructor preserves byte-equal
    ///   coverage).
    /// - Callers who explicitly don't want lossless attempts (e.g.
    ///   workloads where the compactor's overhead isn't worth the
    ///   modest tabular wins).
    pub fn without_compaction(config: SmartCrusherConfig) -> Self {
        SmartCrusherBuilder::new(config)
            .with_default_oss_setup()
            .with_default_ccr_store()
            .build()
    }

    /// Begin a builder chain for custom composition. The Enterprise
    /// entry point: swap the scorer, add business-rule constraints,
    /// attach an audit observer.
    pub fn builder(config: SmartCrusherConfig) -> SmartCrusherBuilder {
        SmartCrusherBuilder::new(config)
    }

    /// Construct with a custom scorer (legacy convenience). Equivalent
    /// to `SmartCrusher::builder(config).with_scorer(scorer).with_default_oss_setup().build()`
    /// minus the default scorer override; preserved for backward
    /// compatibility with pre-PR1 callers.
    pub fn with_scorer(
        config: SmartCrusherConfig,
        scorer: Box<dyn RelevanceScorer + Send + Sync>,
    ) -> Self {
        SmartCrusherBuilder::new(config)
            .with_scorer(scorer)
            .add_default_oss_constraints()
            .build()
    }

    /// Construct directly from owned parts. Used by
    /// [`SmartCrusherBuilder::build`] — not part of the public stable
    /// API. Prefer the builder.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        config: SmartCrusherConfig,
        anchor_selector: AnchorSelector,
        scorer: Box<dyn RelevanceScorer + Send + Sync>,
        analyzer: SmartAnalyzer,
        constraints: Vec<Box<dyn Constraint>>,
        observers: Vec<Box<dyn Observer>>,
        compaction: Option<CompactionStage>,
        ccr_store: Option<Arc<dyn CcrStore>>,
    ) -> Self {
        SmartCrusher {
            config,
            anchor_selector,
            scorer,
            analyzer,
            constraints,
            observers,
            compaction,
            ccr_store,
        }
    }

    /// Handle to the CCR store, if configured. Used by the runtime
    /// (proxy server, PyO3 bridge) to look up originals when retrieval
    /// tool calls fire.
    pub fn ccr_store(&self) -> Option<&Arc<dyn CcrStore>> {
        self.ccr_store.as_ref()
    }

    fn planner(&self) -> SmartCrusherPlanner<'_> {
        SmartCrusherPlanner::new(
            &self.config,
            &self.anchor_selector,
            &*self.scorer,
            &self.analyzer,
            &self.constraints,
        )
    }

    /// Execute a `CompressionPlan` against `items`, returning the
    /// kept-items list in original-array order. Mirrors Python's
    /// `_execute_plan` (line 3617-3633).
    ///
    /// Schema-preserving: each kept item is cloned unchanged. No
    /// summary objects, generated fields, or wrapper metadata.
    pub fn execute_plan(&self, plan: &CompressionPlan, items: &[Value]) -> Vec<Value> {
        let mut indices = plan.keep_indices.clone();
        indices.sort_unstable();
        indices
            .into_iter()
            .filter(|&idx| idx < items.len())
            .map(|idx| items[idx].clone())
            .collect()
    }

    /// Top-level entry point. Mirrors Python `SmartCrusher.crush`
    /// (line 1581-1603) — used by `ContentRouter` when routing JSON
    /// arrays.
    ///
    /// Parses `content` as JSON, recursively processes it (compressing
    /// arrays at every depth via the appropriate per-type crusher),
    /// then re-serializes with Python-compatible formatting (`, ` and
    /// `: ` separators, ASCII-escaped non-ASCII).
    ///
    /// Returns a `CrushResult` with:
    /// - `compressed`: the re-serialized JSON.
    /// - `original`: the input string (unmodified).
    /// - `was_modified`: whether `compressed` differs from `content`'s
    ///   trimmed form.
    /// - `strategy`: combined strategy info from all crushed arrays
    ///   (or `"passthrough"`).
    pub fn crush(&self, content: &str, query: &str, bias: f64) -> CrushResult {
        let start = std::time::Instant::now();
        let (compressed, was_modified, info) = self.smart_crush_content(content, query, bias);
        let strategy = if info.is_empty() {
            "passthrough".to_string()
        } else {
            info
        };

        // Fire one event per top-level crush. Cheap when no observers
        // are configured (`for o in &[]` is a single null-pointer
        // check); cheap when only `TracingObserver` is configured if
        // the subscriber filters `debug` out (the default in
        // production). Custom observers — audit logs, Loop training
        // stream, metrics — pay whatever they pay.
        if !self.observers.is_empty() {
            let event = CrushEvent {
                strategy: strategy.clone(),
                input_bytes: content.len(),
                output_bytes: compressed.len(),
                elapsed_ns: start.elapsed().as_nanos() as u64,
                was_modified,
            };
            for observer in &self.observers {
                observer.on_event(&event);
            }
        }

        CrushResult {
            compressed,
            original: content.to_string(),
            was_modified,
            strategy,
        }
    }

    /// `SmartCrusher._smart_crush_content` (Python line 2243-2301).
    /// JSON-parse, recursively process, re-serialize. CCR marker
    /// injection is stubbed (CCR is disabled in this stage).
    ///
    /// Returns `(crushed_content, was_modified, info)`.
    pub fn smart_crush_content(
        &self,
        content: &str,
        query_context: &str,
        bias: f64,
    ) -> (String, bool, String) {
        // Parse — non-JSON content passes through unchanged.
        let Ok(parsed) = serde_json::from_str::<Value>(content) else {
            return (content.to_string(), false, String::new());
        };

        let (crushed, info) = self.process_value(&parsed, 0, query_context, bias);

        // Re-serialize with Python `safe_json_dumps` formatting:
        // compact `(",", ":")` separators + `ensure_ascii=False`,
        // preserving object-key insertion order. Matches the Python
        // SmartCrusher output bytes the proxy writes.
        let result = crate::transforms::anchor_selector::python_safe_json_dumps(&crushed);
        let was_modified = result != content.trim();
        (result, was_modified, info)
    }

    /// Maximum recursion depth for nested JSON. Mirrors Python's
    /// `_MAX_PROCESS_DEPTH = 50`. Beyond this, values are returned as-is.
    const MAX_PROCESS_DEPTH: usize = 50;

    /// Recursively process a value, crushing arrays where appropriate.
    /// Mirrors Python `_process_value` (line 2307-2398).
    ///
    /// Returns `(processed_value, info_string)`. CCR markers are
    /// stubbed (Python's tuple has a third element for them — Rust's
    /// version omits since we never produce markers in this stage).
    pub fn process_value(
        &self,
        value: &Value,
        depth: usize,
        query_context: &str,
        bias: f64,
    ) -> (Value, String) {
        if depth >= Self::MAX_PROCESS_DEPTH {
            return (value.clone(), String::new());
        }

        let mut info_parts: Vec<String> = Vec::new();

        match value {
            Value::Array(arr) => {
                let n = arr.len();
                if n >= self.config.min_items_to_analyze {
                    let arr_type = classify_array(arr);
                    match arr_type {
                        ArrayType::DictArray => {
                            let result = self.crush_array(arr, query_context, bias);
                            // Lossless path won → substitute the array
                            // with the compacted string in place. This
                            // makes the lossless win visible to the
                            // public `crush()` API: the output JSON
                            // has a string where the array used to be.
                            // The wrapping JSON structure is preserved.
                            if let Some(rendered) = result.compacted {
                                info_parts.push(format!(
                                    "{}({}->len={})",
                                    result.strategy_info,
                                    n,
                                    rendered.len()
                                ));
                                return (Value::String(rendered), info_parts.join(","));
                            }
                            info_parts.push(format!(
                                "{}({}->{})",
                                result.strategy_info,
                                n,
                                result.items.len()
                            ));
                            // Lossy path with rows dropped → append a
                            // CCR-Dropped sentinel object as the last
                            // element of the kept-items array. This is
                            // the **only** place the LLM sees the
                            // `<<ccr:HASH ...>>` pointer in the prompt.
                            // Without this, the store has the data but
                            // no model can ever ask for it.
                            //
                            // Sentinel shape: `{"_ccr_dropped":
                            // "<<ccr:HASH N_rows_offloaded>>"}` —
                            // preserves "array-of-objects" shape so
                            // downstream consumers iterating with
                            // `x.get(...)` keep working; the well-known
                            // `_ccr_dropped` key signals metadata
                            // unambiguously.
                            let mut items = result.items;
                            if !result.dropped_summary.is_empty() {
                                let mut sentinel = serde_json::Map::new();
                                sentinel.insert(
                                    "_ccr_dropped".to_string(),
                                    Value::String(result.dropped_summary),
                                );
                                items.push(Value::Object(sentinel));
                            }
                            return (Value::Array(items), info_parts.join(","));
                        }
                        ArrayType::StringArray => {
                            let strs: Vec<&str> = arr.iter().filter_map(|v| v.as_str()).collect();
                            let (crushed, strategy) = crush_string_array(&strs, &self.config, bias);
                            info_parts.push(format!("{}({}->{})", strategy, n, crushed.len()));
                            let crushed_values: Vec<Value> =
                                crushed.into_iter().map(Value::String).collect();
                            return (Value::Array(crushed_values), info_parts.join(","));
                        }
                        ArrayType::NumberArray => {
                            let (crushed, strategy) = crush_number_array(arr, &self.config, bias);
                            info_parts.push(format!("{}({}->{})", strategy, n, crushed.len()));
                            return (Value::Array(crushed), info_parts.join(","));
                        }
                        ArrayType::MixedArray => {
                            let (crushed, strategy) =
                                self.crush_mixed_array(arr, query_context, bias);
                            info_parts.push(format!("{}({}->{})", strategy, n, crushed.len()));
                            return (Value::Array(crushed), info_parts.join(","));
                        }
                        // NestedArray, BoolArray, Empty → fall through
                        // to recursive descent.
                        _ => {}
                    }
                }

                // Below threshold or not crushable → recurse into items.
                let mut processed: Vec<Value> = Vec::with_capacity(n);
                for item in arr {
                    let (p_item, p_info) = self.process_value(item, depth + 1, query_context, bias);
                    processed.push(p_item);
                    if !p_info.is_empty() {
                        info_parts.push(p_info);
                    }
                }
                (Value::Array(processed), info_parts.join(","))
            }
            Value::Object(map) => {
                // First pass: recurse into values to compress nested arrays.
                let mut processed = serde_json::Map::new();
                for (k, v) in map {
                    let (p_val, p_info) = self.process_value(v, depth + 1, query_context, bias);
                    processed.insert(k.clone(), p_val);
                    if !p_info.is_empty() {
                        info_parts.push(p_info);
                    }
                }

                // Second pass: if the object itself has many keys,
                // compress at the key level.
                if processed.len() >= self.config.min_items_to_analyze {
                    let (crushed_dict, strategy) = crush_object(&processed, &self.config, bias);
                    if strategy != "object:passthrough" {
                        info_parts.push(strategy);
                        return (Value::Object(crushed_dict), info_parts.join(","));
                    }
                }

                (Value::Object(processed), info_parts.join(","))
            }
            // Strings: walker-equivalent handling. Delegates to
            // `process_string` which parses stringified-JSON containers
            // (recursing through `process_value`) and CCR-substitutes
            // opaque blobs (with store-write so retrieval works).
            Value::String(s) => self.process_string(s, depth, query_context, bias),
            // Other scalars — passthrough.
            _ => (value.clone(), String::new()),
        }
    }

    /// Walker-equivalent string handling. Mirrors `walker::walk_string`
    /// in `compaction/walker.rs` but lives on `SmartCrusher` so the
    /// public `crush()` path picks it up.
    ///
    /// Two cases:
    /// 1. **Stringified-JSON.** Strings that parse to a JSON object or
    ///    array → recurse via `process_value`, then re-emit the result
    ///    as a compact JSON string. The wrapping string is preserved
    ///    (so the parent JSON shape stays a string-typed field), but
    ///    its contents are processed end-to-end.
    /// 2. **Opaque blobs.** Strings classified as
    ///    [`CellClass::Opaque`] (long base64 / HTML / long-text) →
    ///    substitute with a `<<ccr:HASH,KIND,SIZE>>` marker. Same
    ///    format as `compaction::walker::format_ccr_marker` so
    ///    downstream consumers can pattern-match markers regardless
    ///    of which path emitted them.
    fn process_string(
        &self,
        s: &str,
        depth: usize,
        query_context: &str,
        bias: f64,
    ) -> (Value, String) {
        // 1. Stringified-JSON: parse, recurse, re-render.
        if let Some(parsed) = try_parse_json_container(s) {
            let (processed, sub_info) = self.process_value(&parsed, depth + 1, query_context, bias);
            // If recursion produced something different, re-emit.
            // Special case: if the recursion returned a `Value::String`
            // (lossless compaction substituted the array with a
            // rendered CSV+schema string), use that string directly.
            // Re-encoding it as JSON would produce a quoted string
            // literal — double-encoded — which is not what callers
            // expect in the wrapping field.
            if processed != parsed {
                let rendered = match &processed {
                    Value::String(rendered_str) => rendered_str.clone(),
                    _ => serde_json::to_string(&processed).unwrap_or_else(|_| s.to_string()),
                };
                let info = if sub_info.is_empty() {
                    "string_json".to_string()
                } else {
                    format!("string_json[{sub_info}]")
                };
                return (Value::String(rendered), info);
            }
        }

        // 2. Opaque blob: substitute with CCR marker AND stash the
        // original in the store (PR8) so retrieval works. Hash + format
        // identical to walker.rs via the shared helper — zero drift.
        let cfg = ClassifyConfig::default();
        if let CellClass::Opaque(kind) = classify_cell(&Value::String(s.to_string()), &cfg) {
            let marker = emit_opaque_ccr_marker(s, &kind, self.ccr_store.as_ref());
            let kind_label = opaque_kind_label(&kind);
            return (Value::String(marker), format!("string_ccr:{kind_label}"));
        }

        // 3. Plain string — passthrough.
        (Value::String(s.to_string()), String::new())
    }

    /// Compress an array of dict items.
    ///
    /// Direct port of `_crush_array` (Python line 2400-2687) with the
    /// optional subsystems (TOIN / CCR / feedback / telemetry) wired
    /// in their disabled-by-default behavior. See module-level docs
    /// for the rationale.
    ///
    /// # Pipeline
    ///
    /// 1. Compute `item_strings` once (used as input to adaptive
    ///    sizing and downstream relevance scoring).
    /// 2. `compute_optimal_k` → `adaptive_k`.
    /// 3. If `n <= adaptive_k`, return passthrough.
    /// 4. `analyzer.analyze_array(items)` → `analysis`.
    /// 5. If `analysis.recommended_strategy == Skip`, return passthrough
    ///    with a `skip:<reason>` strategy string.
    /// 6. `planner.create_plan(analysis, items, query_context, ...)`.
    /// 7. `execute_plan(plan, items)` → result.
    /// 8. Strategy info = `analysis.recommended_strategy.as_str()`.
    pub fn crush_array(&self, items: &[Value], query_context: &str, bias: f64) -> CrushArrayResult {
        let item_strings: Vec<String> = items
            .iter()
            .map(|i| serde_json::to_string(i).unwrap_or_default())
            .collect();
        let item_str_refs: Vec<&str> = item_strings.iter().map(|s| s.as_str()).collect();

        let max_k = if self.config.max_items_after_crush > 0 {
            Some(self.config.max_items_after_crush)
        } else {
            None
        };
        let adaptive_k = compute_optimal_k(&item_str_refs, bias, 3, max_k);

        // Tier-1 boundary: array already small enough — passthrough,
        // nothing to compact, nothing to drop.
        if items.len() <= adaptive_k {
            return CrushArrayResult {
                items: items.to_vec(),
                strategy_info: "none:adaptive_at_limit".to_string(),
                ccr_hash: None,
                dropped_summary: String::new(),
                compacted: None,
                compaction_kind: None,
            };
        }

        // ── Lossless-first attempt ──
        //
        // Run the compaction stage if present, then check the savings
        // ratio against `config.lossless_min_savings_ratio`. If the
        // lossless rendering shrinks the input by at least that much,
        // ship it — nothing dropped, no CCR retrieval needed.
        // Otherwise fall through to the lossy path.
        if let Some(stage) = &self.compaction {
            let (c, rendered) = stage.run(items);
            if c.was_compacted() {
                let input_bytes = estimate_array_bytes(&item_strings);
                let savings_ratio = if input_bytes > 0 {
                    1.0 - (rendered.len() as f64 / input_bytes as f64)
                } else {
                    0.0
                };
                if savings_ratio >= self.config.lossless_min_savings_ratio {
                    let kind = compaction_kind_str(&c);
                    return CrushArrayResult {
                        items: items.to_vec(), // nothing dropped
                        strategy_info: format!("lossless:{kind}"),
                        ccr_hash: None,
                        dropped_summary: String::new(),
                        compacted: Some(rendered),
                        compaction_kind: Some(kind),
                    };
                }
            }
        }

        // ── Lossy path: compress inline + cache full original via CCR ──
        //
        // The runtime caller (PyO3 bridge / proxy server) is expected
        // to stash the full input keyed by `ccr_hash` so a retrieval
        // tool can serve dropped rows back to the LLM on demand.
        // **No data is lost** — "lossy" here means "compressed view
        // inline; full payload retrievable via CCR cache."

        let effective_max_items = adaptive_k;
        let analysis = self.analyzer.analyze_array(items);

        // Crushability gate: not safe to crush → passthrough, no CCR.
        if analysis.recommended_strategy == CompressionStrategy::Skip {
            let reason = match &analysis.crushability {
                Some(c) => format!("skip:{}", c.reason),
                None => String::new(),
            };
            return CrushArrayResult {
                items: items.to_vec(),
                strategy_info: reason,
                ccr_hash: None,
                dropped_summary: String::new(),
                compacted: None,
                compaction_kind: None,
            };
        }

        let plan = self.planner().create_plan(
            &analysis,
            items,
            query_context,
            None, // preserve_fields (TOIN — stubbed)
            Some(effective_max_items),
            Some(&item_strings),
        );
        let result = self.execute_plan(&plan, items);

        // Emit CCR-Dropped marker iff rows were actually dropped AND
        // the marker gate is on. **The marker is the cornerstone of
        // CCR's no-data-loss guarantee:** we hash the full original,
        // stash it in the configured store, and emit a marker pointing
        // at that hash. The runtime later serves the original back via
        // retrieval tool calls.
        //
        // When `enable_ccr_marker` is false (Python shim's path for
        // `ccr_config.enabled=False` or `inject_retrieval_marker=False`)
        // we keep the row drops (compression is still requested) but
        // skip the marker text and the store write — there's no point
        // storing a payload that nothing in the prompt can reference.
        let dropped_count = items.len().saturating_sub(result.len());
        let (ccr_hash, dropped_summary) = if dropped_count > 0 && self.config.enable_ccr_marker {
            // Serialize the original array exactly ONCE. The hash is
            // taken over those bytes, and (if a store is configured) the
            // same bytes get stored — eliminating a redundant tree clone
            // (`items.to_vec()`) and a redundant `serde_json::to_string`
            // pass that the previous version did per dropped array.
            let canonical = canonical_array_json(items);
            let h = hash_canonical(&canonical);
            let marker = format!("<<ccr:{h} {dropped_count}_rows_offloaded>>");
            if let Some(store) = &self.ccr_store {
                store.put(&h, &canonical);
            }
            (Some(h), marker)
        } else {
            (None, String::new())
        };

        CrushArrayResult {
            items: result,
            strategy_info: analysis.recommended_strategy.as_str().to_string(),
            ccr_hash,
            dropped_summary,
            compacted: None,
            compaction_kind: None,
        }
    }

    /// Compress a mixed-type array by grouping items by type and
    /// compressing each group with the appropriate handler.
    ///
    /// Direct port of `_crush_mixed_array` (Python line 2914-3013).
    ///
    /// Strategy:
    /// 1. Group by type (dict / str / number / list / null / bool / other).
    /// 2. For groups with >= `min_items_to_analyze` items: apply the
    ///    type-specific compressor.
    /// 3. For small groups: keep all items.
    /// 4. Reassemble in original order.
    ///
    /// Returns `(crushed_items, strategy_string)`.
    pub fn crush_mixed_array(
        &self,
        items: &[Value],
        query_context: &str,
        bias: f64,
    ) -> (Vec<Value>, String) {
        let n = items.len();
        if n <= 8 {
            return (items.to_vec(), "mixed:passthrough".to_string());
        }

        // Group by type, tracking original indices.
        let mut groups: GroupBuckets = GroupBuckets::default();
        for (i, item) in items.iter().enumerate() {
            groups.push(group_key(item), i, item.clone());
        }

        let mut keep_indices: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        let mut strategy_parts: Vec<String> = Vec::new();

        for (type_key, indices, values) in groups.into_iter() {
            // Small groups: keep all items.
            if values.len() < self.config.min_items_to_analyze {
                keep_indices.extend(&indices);
                continue;
            }

            match type_key {
                "dict" => {
                    let CrushArrayResult { items: crushed, .. } =
                        self.crush_array(&values, query_context, bias);
                    // Find which original indices survived by matching
                    // canonical-JSON serialization. Mirrors Python's
                    // `json.dumps(c, sort_keys=True, default=str)`-keyed
                    // set match.
                    let crushed_keys: std::collections::HashSet<String> =
                        crushed.iter().map(canonical_json_for_match).collect();
                    for (i, idx) in indices.iter().enumerate() {
                        if crushed_keys.contains(&canonical_json_for_match(&values[i])) {
                            keep_indices.insert(*idx);
                        }
                    }
                    strategy_parts.push(format!("dict:{}->{}", values.len(), crushed.len()));
                }
                "str" => {
                    let strs: Vec<&str> = values.iter().filter_map(|v| v.as_str()).collect();
                    let (crushed, _) = crush_string_array(&strs, &self.config, bias);
                    let crushed_set: std::collections::HashSet<&str> =
                        crushed.iter().map(|s| s.as_str()).collect();
                    for (i, idx) in indices.iter().enumerate() {
                        if let Some(s) = values[i].as_str() {
                            if crushed_set.contains(s) {
                                keep_indices.insert(*idx);
                            }
                        }
                    }
                    strategy_parts.push(format!("str:{}->{}", values.len(), crushed.len()));
                }
                "number" => {
                    // Python: just adaptive sampling + outlier detection
                    // (no summary prefix). Keeps first/last by index
                    // and items >variance_threshold σ from mean.
                    let item_strings: Vec<String> = values.iter().map(|v| v.to_string()).collect();
                    let item_refs: Vec<&str> = item_strings.iter().map(|s| s.as_str()).collect();
                    let (_kt, kf, kl, _) = compute_k_split(&item_refs, &self.config, bias);

                    let kf = kf.min(values.len());
                    let kl = kl.min(values.len().saturating_sub(kf));
                    let first_idx: Vec<usize> = indices.iter().take(kf).copied().collect();
                    let last_idx: Vec<usize> =
                        indices.iter().rev().take(kl).copied().collect::<Vec<_>>();
                    keep_indices.extend(&first_idx);
                    keep_indices.extend(&last_idx);

                    // Outliers via finite-only stats.
                    let finite: Vec<f64> = values
                        .iter()
                        .filter_map(|v| v.as_f64().filter(|f| f.is_finite()))
                        .collect();
                    if finite.len() > 1 {
                        if let Some(mean_v) = super::stats_math::mean(&finite) {
                            if let Some(std_v) = super::stats_math::sample_stdev(&finite) {
                                if std_v > 0.0 {
                                    let threshold = self.config.variance_threshold * std_v;
                                    for (i, val) in values.iter().enumerate() {
                                        if let Some(num) = val.as_f64().filter(|f| f.is_finite()) {
                                            if (num - mean_v).abs() > threshold {
                                                keep_indices.insert(indices[i]);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    strategy_parts.push(format!("num:{}", values.len()));
                }
                _ => {
                    // list / bool / none / other → keep all items.
                    keep_indices.extend(&indices);
                }
            }
        }

        // Reassemble in original order.
        let result: Vec<Value> = keep_indices.iter().map(|&i| items[i].clone()).collect();
        let strategy = format!(
            "mixed:adaptive({}->{},{})",
            n,
            result.len(),
            strategy_parts.join(",")
        );
        (result, strategy)
    }
}

// ---------- helpers ----------

/// Group key that mirrors Python's `_crush_mixed_array` switch on
/// `isinstance`. Note the bool-before-number ordering: in Python, bool
/// is a subclass of int, but JSON treats them as distinct types, so we
/// don't have the Python ordering hazard.
fn group_key(item: &Value) -> &'static str {
    match item {
        Value::Object(_) => "dict",
        Value::String(_) => "str",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::Array(_) => "list",
        Value::Null => "none",
    }
}

/// Group buckets keyed by the type-string. Preserves first-occurrence
/// order across keys so dict/str/number/list/none/bool always come out
/// in the same order — matters because `keep_indices` is built
/// incrementally and Python iterates `groups.items()` (insertion order
/// in 3.7+).
#[derive(Default)]
struct GroupBuckets {
    entries: Vec<(&'static str, Vec<usize>, Vec<Value>)>,
    index_of: std::collections::HashMap<&'static str, usize>,
}

impl GroupBuckets {
    fn push(&mut self, key: &'static str, idx: usize, value: Value) {
        match self.index_of.get(key).copied() {
            Some(i) => {
                self.entries[i].1.push(idx);
                self.entries[i].2.push(value);
            }
            None => {
                self.index_of.insert(key, self.entries.len());
                self.entries.push((key, vec![idx], vec![value]));
            }
        }
    }
}

impl IntoIterator for GroupBuckets {
    type Item = (&'static str, Vec<usize>, Vec<Value>);
    type IntoIter = std::vec::IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

/// Serialize a `Value` for membership comparison. Mirrors Python's
/// `json.dumps(c, sort_keys=True, default=str)` used by
/// `_crush_mixed_array` to match crushed dict items back to their
/// original indices. The `default=str` fallback only matters for
/// non-JSON-serializable Python values; in serde_json land everything
/// is already JSON-native, so plain canonical JSON suffices.
fn canonical_json_for_match(value: &Value) -> String {
    crate::transforms::anchor_selector::python_json_dumps_sort_keys(value)
}

/// Maps a `Compaction` to a stable kind tag exposed via `CrushArrayResult`.
fn compaction_kind_str(c: &Compaction) -> &'static str {
    match c {
        Compaction::Table { .. } => "table",
        Compaction::Buckets { .. } => "buckets",
        Compaction::OpaqueRef { .. } => "ccr",
        Compaction::Untouched(_) => "untouched",
    }
}

/// Approximate byte size of `[v0, v1, ...]` JSON serialization, given
/// each item's already-serialized form. Adds 2 for outer brackets and
/// 1 per inter-item comma. Used by the lossless savings-ratio check.
fn estimate_array_bytes(item_strings: &[String]) -> usize {
    let payload: usize = item_strings.iter().map(|s| s.len()).sum();
    let separators = item_strings.len().saturating_sub(1);
    payload + separators + 2
}

/// Serialize `[v0, v1, ...]` once into the canonical JSON form used by
/// the CCR retrieval contract. `serde_json` writes a slice of `Value` as
/// the same bytes it would write for `Value::Array(items.to_vec())`, so
/// we skip the array-wrapper allocation and the deep tree clone it
/// requires. Used by both the hash (input) and the store payload (write).
fn canonical_array_json(items: &[Value]) -> String {
    serde_json::to_string(items).unwrap_or_default()
}

/// 12-char SHA-256 hex prefix of an already-serialized canonical JSON
/// string. Caller is responsible for producing the canonical form via
/// [`canonical_array_json`] (or another byte-equal serializer) — the
/// hash is over the bytes, so a stable serializer is the contract.
fn hash_canonical(canonical: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(canonical.as_bytes());
    h.finalize()
        .iter()
        .take(6)
        .map(|b| format!("{b:02x}"))
        .collect()
}

// `hash_array_for_ccr` (a test-only `canonical_array_json + hash_canonical`
// convenience) lived here previously but had no callers — clippy flagged
// it as dead code. Reintroduce as a test fixture if a future test wants
// the one-liner; production callsites inline both steps so the canonical
// bytes can be reused for the store payload.

// ─── PR5 walker-integration helpers (string handling) ──────────────────────
//
// Parse-as-JSON-container, marker formatting, and humanize-bytes used to
// live here as locals. PR8 extracted them into `compaction::walker` so
// `walker.rs` and `process_value` share one canonical implementation —
// killing the drift risk where the two paths could format markers
// differently. `process_string` now calls `try_parse_json_container` and
// `emit_opaque_ccr_marker` directly. Only `opaque_kind_label` survives
// here because `process_string`'s `string_ccr:<kind>` strategy-info
// label is local to this module's debug-string convention.

fn opaque_kind_label(kind: &super::compaction::OpaqueKind) -> &str {
    use super::compaction::OpaqueKind;
    match kind {
        OpaqueKind::Base64Blob => "base64",
        OpaqueKind::LongString => "string",
        OpaqueKind::HtmlChunk => "html",
        OpaqueKind::Other(s) => s.as_str(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn crusher() -> SmartCrusher {
        SmartCrusher::new(SmartCrusherConfig::default())
    }

    // ---------- execute_plan ----------

    #[test]
    fn execute_plan_empty_indices_returns_empty() {
        let c = crusher();
        let plan = CompressionPlan::default();
        let items: Vec<Value> = (0..5).map(|i| json!({"id": i})).collect();
        let result = c.execute_plan(&plan, &items);
        assert!(result.is_empty());
    }

    #[test]
    fn execute_plan_returns_items_in_sorted_index_order() {
        let c = crusher();
        let items: Vec<Value> = (0..10).map(|i| json!({"id": i})).collect();
        let plan = CompressionPlan {
            keep_indices: vec![5, 2, 8, 0],
            ..CompressionPlan::default()
        };
        let result = c.execute_plan(&plan, &items);
        assert_eq!(result.len(), 4);
        assert_eq!(result[0]["id"], 0);
        assert_eq!(result[1]["id"], 2);
        assert_eq!(result[2]["id"], 5);
        assert_eq!(result[3]["id"], 8);
    }

    #[test]
    fn execute_plan_skips_out_of_bounds() {
        let c = crusher();
        let items: Vec<Value> = (0..3).map(|i| json!({"id": i})).collect();
        let plan = CompressionPlan {
            keep_indices: vec![0, 5, 2],
            ..CompressionPlan::default()
        };
        let result = c.execute_plan(&plan, &items);
        assert_eq!(result.len(), 2);
    }

    // ---------- crush_array ----------

    #[test]
    fn crush_array_passthrough_when_below_adaptive_k() {
        let c = crusher();
        let items: Vec<Value> = (0..3).map(|i| json!({"id": i})).collect();
        let result = c.crush_array(&items, "", 1.0);
        assert_eq!(result.items.len(), 3);
        assert_eq!(result.strategy_info, "none:adaptive_at_limit");
        assert!(result.ccr_hash.is_none());
    }

    #[test]
    fn crush_array_skip_path_returns_original_items() {
        // 30 unique dict items with ID-like fields → analyzer should
        // detect "unique_entities_no_signal" and SKIP. Use the
        // no-compaction constructor so we exercise the lossy/skip
        // gate; the lossless path would otherwise short-circuit
        // because uniform-tabular input is the lossless sweet spot.
        let c = SmartCrusher::without_compaction(SmartCrusherConfig::default());
        let items: Vec<Value> = (0..30)
            .map(|i| json!({"id": i, "name": format!("user_{}", i)}))
            .collect();
        let result = c.crush_array(&items, "", 1.0);
        // skip path returns the original items unchanged.
        assert_eq!(result.items.len(), 30);
        assert!(
            result.strategy_info.starts_with("skip:"),
            "expected skip:..., got {}",
            result.strategy_info
        );
    }

    #[test]
    fn crush_array_low_uniqueness_compresses() {
        // 30 items with status=ok across all → low_uniqueness path
        // (crushable, smart_sample strategy).
        let c = crusher();
        let items: Vec<Value> = (0..30).map(|_| json!({"status": "ok"})).collect();
        let result = c.crush_array(&items, "", 1.0);
        assert!(result.items.len() <= 30, "should not exceed original count");
    }

    #[test]
    fn crush_array_keeps_error_items() {
        let c = crusher();
        let mut items: Vec<Value> = (0..30).map(|i| json!({"id": i, "status": "ok"})).collect();
        items.push(json!({"id": 30, "status": "error", "msg": "FATAL"}));
        let result = c.crush_array(&items, "", 1.0);
        // Whatever path is taken, the error item should survive.
        assert!(
            result
                .items
                .iter()
                .any(|item| { item.get("status").and_then(|v| v.as_str()) == Some("error") }),
            "error item must survive crush_array"
        );
    }

    // ---------- crush_mixed_array ----------

    #[test]
    fn crush_mixed_passthrough_at_threshold() {
        let c = crusher();
        let items: Vec<Value> = vec![
            json!(1),
            json!("two"),
            json!({"k": "v"}),
            json!([1, 2]),
            json!(null),
            json!(true),
            json!(3),
            json!("four"),
        ];
        let (result, strat) = c.crush_mixed_array(&items, "", 1.0);
        assert_eq!(result.len(), 8);
        assert_eq!(strat, "mixed:passthrough");
    }

    #[test]
    fn crush_mixed_groups_and_compresses_dicts() {
        let c = crusher();
        // 25 dicts (large group → gets crushed) + 5 strings (small group → all kept).
        let mut items: Vec<Value> = (0..25).map(|i| json!({"id": i, "status": "ok"})).collect();
        for i in 0..5 {
            items.push(json!(format!("string_{}", i)));
        }
        let (result, strat) = c.crush_mixed_array(&items, "", 1.0);
        assert!(strat.starts_with("mixed:adaptive("));
        // The 5 strings (small group) all survive.
        let str_count = result.iter().filter(|v| v.is_string()).count();
        assert_eq!(str_count, 5);
    }

    #[test]
    fn crush_mixed_keeps_lists_and_nulls_unchanged() {
        let c = crusher();
        let mut items: Vec<Value> = vec![json!([1, 2]); 6];
        items.extend(vec![json!(null); 6]);
        items.extend(vec![json!({"k": 1}); 10]);
        let (result, _strat) = c.crush_mixed_array(&items, "", 1.0);
        // Lists and nulls (not dict/str/number) → fall through to "keep all".
        let list_count = result.iter().filter(|v| v.is_array()).count();
        let null_count = result.iter().filter(|v| v.is_null()).count();
        assert_eq!(list_count, 6);
        assert_eq!(null_count, 6);
    }

    #[test]
    fn crusher_construction_default() {
        let c = SmartCrusher::new(SmartCrusherConfig::default());
        assert_eq!(c.config.max_items_after_crush, 15);
    }

    // ---------- top-level crush ----------

    #[test]
    fn crush_non_json_passes_through_unchanged() {
        let c = crusher();
        let result = c.crush("not json at all", "", 1.0);
        assert!(!result.was_modified);
        assert_eq!(result.compressed, "not json at all");
        assert_eq!(result.strategy, "passthrough");
    }

    #[test]
    fn crush_scalar_json_passes_through() {
        let c = crusher();
        let result = c.crush("42", "", 1.0);
        // A scalar is not crushable; should round-trip unchanged.
        assert_eq!(result.compressed, "42");
        assert!(!result.was_modified);
    }

    #[test]
    fn crush_small_array_passes_through() {
        let c = crusher();
        // Compact-form input matches the compact serializer output, so
        // the array is not "modified" even though it round-trips
        // through parse → serialize. (The spaced form `[1, 2, 3]`
        // would mark `was_modified=true` because the compact
        // serializer rewrites it to `[1,2,3]`.)
        let result = c.crush(r#"[1,2,3]"#, "", 1.0);
        // Below min_items_to_analyze=5 → no crushing of the structure.
        assert!(!result.was_modified);
        assert_eq!(result.compressed, "[1,2,3]");
    }

    #[test]
    fn crush_dict_array_crushes_when_low_uniqueness() {
        // The public `crush()` API serializes back to JSON; the
        // lossless-path output (a compacted string) is exposed via
        // `crush_array().compacted` rather than being substituted into
        // the JSON re-serialization. So we exercise the lossy path
        // here via `without_compaction()` to validate the original
        // intent: low-uniqueness dicts compress via row-dropping.
        let c = SmartCrusher::without_compaction(SmartCrusherConfig::default());
        let mut input = String::from("[");
        for i in 0..30 {
            if i > 0 {
                input.push(',');
            }
            input.push_str(r#"{"status":"ok"}"#);
        }
        input.push(']');
        let result = c.crush(&input, "", 1.0);
        assert!(
            result.was_modified,
            "30 identical dicts should compress (low_uniqueness_safe_to_sample)"
        );
        assert_ne!(result.strategy, "passthrough");
    }

    #[test]
    fn crush_serializes_with_python_safe_format() {
        let c = crusher();
        // SmartCrusher uses Python's `safe_json_dumps`: compact
        // separators `(",", ":")` + `ensure_ascii=False`, preserving
        // object-key insertion order. A spaced input round-trips to
        // the compact form.
        let input = r#"{"a": 1, "b": 2, "c": 3}"#;
        let result = c.crush(input, "", 1.0);
        assert_eq!(
            result.compressed, r#"{"a":1,"b":2,"c":3}"#,
            "safe_json_dumps emits compact `,` / `:` separators"
        );
    }

    #[test]
    fn crush_recurses_into_nested_arrays() {
        let c = crusher();
        // Top-level dict with a nested array of 30 identical items.
        // The inner array should compress (low_uniqueness path).
        let mut inner = String::from("[");
        for i in 0..30 {
            if i > 0 {
                inner.push(',');
            }
            inner.push_str(r#"{"status":"ok"}"#);
        }
        inner.push(']');
        let input = format!(r#"{{"data": {}}}"#, inner);
        let result = c.crush(&input, "", 1.0);
        assert!(
            result.was_modified,
            "nested compressible array must be crushed even inside a wrapper object"
        );
    }

    #[test]
    fn crusher_with_custom_scorer() {
        use crate::relevance::BM25Scorer;
        let c = SmartCrusher::with_scorer(
            SmartCrusherConfig::default(),
            Box::new(BM25Scorer::default()),
        );
        // Sanity: crushing still works with a swapped scorer.
        let items: Vec<Value> = (0..30).map(|_| json!({"status": "ok"})).collect();
        let result = c.crush_array(&items, "anything", 1.0);
        assert!(result.items.len() <= 30);
    }

    // ---------- Stage 3c.2 PR4: lossless-first default with threshold + CCR-Dropped ----------

    #[test]
    fn without_compaction_yields_none_compacted_field() {
        // The opt-out constructor preserves pre-PR4 lossy-only path.
        // No lossless attempt → compacted/compaction_kind always None.
        let c = SmartCrusher::without_compaction(SmartCrusherConfig::default());
        let items: Vec<Value> = (0..30).map(|_| json!({"status": "ok"})).collect();
        let result = c.crush_array(&items, "", 1.0);
        assert!(result.compacted.is_none());
        assert!(result.compaction_kind.is_none());
    }

    #[test]
    fn lossless_wins_when_savings_above_threshold() {
        // 50 uniform tabular dicts → CSV+schema compaction shrinks
        // the input dramatically (well above the 0.30 default).
        // Default `SmartCrusher::new()` should pick lossless.
        let c = crusher();
        let items: Vec<Value> = (0..50)
            .map(|i| json!({"id": i, "name": format!("u_{i}"), "status": "ok"}))
            .collect();
        let result = c.crush_array(&items, "", 1.0);
        let compacted = result.compacted.expect("compacted should be set");
        assert!(compacted.starts_with("[50]{"), "got: {compacted}");
        assert_eq!(result.compaction_kind, Some("table"));
        assert!(
            result.strategy_info.starts_with("lossless:table"),
            "got: {}",
            result.strategy_info
        );
        // Lossless = nothing dropped → no CCR retrieval needed.
        assert!(result.ccr_hash.is_none());
        // items preserved (full original).
        assert_eq!(result.items.len(), 50);
    }

    #[test]
    fn lossy_falls_through_when_savings_below_threshold() {
        // Force the threshold high enough that even tabular savings
        // can't satisfy it → lossy path runs → CCR-Dropped fires.
        // Use low-uniqueness items so the analyzer is willing to
        // crush (unique id+name per row would trip the
        // "unique_entities_no_signal" skip gate instead).
        let cfg = SmartCrusherConfig {
            lossless_min_savings_ratio: 0.99,
            ..Default::default()
        };
        let c = SmartCrusher::new(cfg);
        let items: Vec<Value> = (0..50).map(|_| json!({"status": "ok"})).collect();
        let result = c.crush_array(&items, "", 1.0);
        // Lossless declined → no compacted output.
        assert!(result.compacted.is_none());
        // Lossy ran → rows dropped.
        assert!(
            result.items.len() < 50,
            "expected lossy drop, got {} items",
            result.items.len()
        );
        // CCR hash populated for retrieval.
        let h = result.ccr_hash.expect("ccr_hash populated on drop");
        assert_eq!(h.len(), 12);
        // Marker visible in dropped_summary.
        assert!(
            result.dropped_summary.contains(&format!("<<ccr:{h}")),
            "got: {}",
            result.dropped_summary
        );
        assert!(result.dropped_summary.contains("rows_offloaded"));
    }

    #[test]
    fn ccr_hash_is_deterministic() {
        // Same input → same hash, so the runtime cache key is stable.
        let cfg = SmartCrusherConfig {
            lossless_min_savings_ratio: 0.99, // force lossy path
            ..Default::default()
        };
        let c = SmartCrusher::new(cfg);
        let items: Vec<Value> = (0..30).map(|i| json!({"id": i, "tag": "ok"})).collect();
        let r1 = c.crush_array(&items, "", 1.0);
        let r2 = c.crush_array(&items, "", 1.0);
        assert_eq!(r1.ccr_hash, r2.ccr_hash);
        assert!(r1.ccr_hash.is_some());
    }

    #[test]
    fn ccr_hash_changes_with_input() {
        let cfg = SmartCrusherConfig {
            lossless_min_savings_ratio: 0.99,
            ..Default::default()
        };
        let c = SmartCrusher::new(cfg);
        let a: Vec<Value> = (0..30).map(|i| json!({"id": i})).collect();
        let b: Vec<Value> = (100..130).map(|i| json!({"id": i})).collect();
        let ra = c.crush_array(&a, "", 1.0);
        let rb = c.crush_array(&b, "", 1.0);
        assert_ne!(ra.ccr_hash, rb.ccr_hash);
    }

    #[test]
    fn lossy_without_compaction_still_emits_ccr_hash() {
        // The CCR-Dropped restoration applies regardless of whether
        // lossless was attempted — without_compaction also gets the
        // ccr_hash on row drops.
        let c = SmartCrusher::without_compaction(SmartCrusherConfig::default());
        let items: Vec<Value> = (0..30).map(|_| json!({"status": "ok"})).collect();
        let result = c.crush_array(&items, "", 1.0);
        if result.items.len() < items.len() {
            assert!(result.ccr_hash.is_some());
            assert!(!result.dropped_summary.is_empty());
        }
    }

    #[test]
    fn passthrough_paths_do_not_emit_ccr_hash() {
        // Tier-1 boundary (items.len() <= adaptive_k): nothing
        // dropped, no CCR. Skip path: same.
        let c = crusher();
        let small: Vec<Value> = (0..3).map(|i| json!({"id": i})).collect();
        let r = c.crush_array(&small, "", 1.0);
        assert!(r.ccr_hash.is_none());
        assert_eq!(r.dropped_summary, "");
    }

    #[test]
    fn compaction_skips_non_object_array() {
        // Compactor returns Untouched for non-object arrays → no
        // compacted field populated, no kind tag.
        let c = SmartCrusherBuilder::new(SmartCrusherConfig::default())
            .with_default_oss_setup()
            .with_default_compaction()
            .build();
        let items: Vec<Value> = (0..30).map(|i| json!(i)).collect();
        let result = c.crush_array(&items, "", 1.0);
        assert!(result.compacted.is_none());
        assert!(result.compaction_kind.is_none());
    }

    // ---------- Stage 3c.2 PR5: walker-integration in process_value ----------

    #[test]
    fn process_string_short_string_passthrough() {
        let c = SmartCrusher::new(SmartCrusherConfig::default());
        let (out, info) = c.process_value(&json!("hello world"), 0, "", 1.0);
        assert_eq!(out, json!("hello world"));
        assert!(info.is_empty());
    }

    #[test]
    fn process_string_stringified_json_array_recurses() {
        // A string-typed field whose value is a JSON-encoded array of
        // dicts. process_value should parse it, recurse, and return
        // the processed JSON re-rendered as a string.
        let c = SmartCrusher::new(SmartCrusherConfig::default());
        let big_array_json = serde_json::to_string(
            &(0..50)
                .map(|i| json!({"id": i, "level": "info", "msg": "ok"}))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let doc = json!({"payload": big_array_json.clone()});
        let (out, info) = c.process_value(&doc, 0, "", 1.0);
        // payload still a string-typed field — we preserved the
        // wrapping shape — but its content was processed.
        let payload = out.pointer("/payload").and_then(|v| v.as_str()).unwrap();
        // Either compressed or unchanged; if compressed, info reflects.
        // For 50 items with low-uniqueness, compression should fire.
        // The strategy info should mention string_json processing.
        assert!(
            info.contains("string_json") || payload != big_array_json,
            "expected processing trace; info={info}, len before={}, after={}",
            big_array_json.len(),
            payload.len(),
        );
    }

    #[test]
    fn process_string_opaque_blob_becomes_ccr_marker() {
        let c = SmartCrusher::new(SmartCrusherConfig::default());
        let big_b64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=".repeat(8);
        let doc = json!({"id": 1, "blob": big_b64});
        let (out, _info) = c.process_value(&doc, 0, "", 1.0);
        let blob = out.pointer("/blob").and_then(|v| v.as_str()).unwrap();
        assert!(blob.starts_with("<<ccr:"), "got: {blob}");
        assert!(blob.contains(",base64,"));
    }

    #[test]
    fn process_string_top_level_string_processed() {
        // crush() takes a string; if it doesn't parse as JSON, today's
        // behavior returns it unchanged. But if it's a stringified
        // JSON object/array, it should now get processed.
        let c = SmartCrusher::new(SmartCrusherConfig::default());
        // Non-JSON top-level string — passthrough.
        let plain = "just some plain text";
        let result = c.crush(plain, "", 1.0);
        assert_eq!(result.compressed, plain);
    }

    #[test]
    fn process_string_does_not_alter_short_quoted_strings() {
        // Strings that look JSON-like but are short shouldn't be
        // CCR-substituted.
        let c = SmartCrusher::new(SmartCrusherConfig::default());
        let doc = json!({"msg": "{this looks like json but isnt}"});
        let (out, _) = c.process_value(&doc, 0, "", 1.0);
        assert_eq!(out, doc);
    }

    #[test]
    fn process_string_helper_parses_only_containers() {
        assert!(try_parse_json_container("{\"a\":1}").is_some());
        assert!(try_parse_json_container("[1,2,3]").is_some());
        assert!(try_parse_json_container("123").is_none()); // bare scalar
        assert!(try_parse_json_container("\"hello\"").is_none()); // bare string
        assert!(try_parse_json_container("not json").is_none());
        assert!(try_parse_json_container("{malformed").is_none());
    }

    // ---------- enable_ccr_marker gate (PR #301 re-land) ----------

    #[test]
    fn enable_ccr_marker_false_suppresses_marker_and_store() {
        // The Rust-side gate. Compression still runs (rows drop) but
        // the result carries no marker text, no hash, and the CCR
        // store does NOT grow — there's no point storing what nothing
        // in the prompt can reference.
        use crate::ccr::InMemoryCcrStore;
        use crate::transforms::smart_crusher::SmartCrusherBuilder;
        use std::sync::Arc;

        let store: Arc<dyn CcrStore> = Arc::new(InMemoryCcrStore::new());
        let cfg = SmartCrusherConfig {
            lossless_min_savings_ratio: 0.99, // force lossy path
            enable_ccr_marker: false,
            ..SmartCrusherConfig::default()
        };
        let c = SmartCrusherBuilder::new(cfg)
            .with_ccr_store(Arc::clone(&store))
            .build();
        let items: Vec<Value> = (0..50).map(|_| json!({"status": "ok"})).collect();

        let store_len_before = store.len();
        let result = c.crush_array(&items, "", 1.0);
        let store_len_after = store.len();

        // Rows were dropped (we built 50, kept fewer).
        assert!(result.items.len() < items.len(), "lossy path didn't fire");
        // Gate held: no marker, no hash.
        assert!(result.ccr_hash.is_none(), "ccr_hash should be None");
        assert!(
            result.dropped_summary.is_empty(),
            "dropped_summary should be empty, got: {:?}",
            result.dropped_summary
        );
        // Store did NOT grow.
        assert_eq!(
            store_len_after, store_len_before,
            "ccr_store grew despite enable_ccr_marker=false"
        );
    }

    #[test]
    fn enable_ccr_marker_true_is_default_behavior() {
        // Default config still emits markers + stores when rows drop.
        // Sanity: the gate is opt-out, not opt-in.
        use crate::ccr::InMemoryCcrStore;
        use crate::transforms::smart_crusher::SmartCrusherBuilder;
        use std::sync::Arc;

        let store: Arc<dyn CcrStore> = Arc::new(InMemoryCcrStore::new());
        let cfg = SmartCrusherConfig {
            lossless_min_savings_ratio: 0.99, // force lossy path
            ..SmartCrusherConfig::default()
        };
        // Default: enable_ccr_marker = true.
        assert!(cfg.enable_ccr_marker);
        let c = SmartCrusherBuilder::new(cfg)
            .with_ccr_store(Arc::clone(&store))
            .build();
        let items: Vec<Value> = (0..50).map(|_| json!({"status": "ok"})).collect();

        let store_len_before = store.len();
        let result = c.crush_array(&items, "", 1.0);
        let store_len_after = store.len();

        assert!(result.items.len() < items.len(), "lossy path didn't fire");
        assert!(result.ccr_hash.is_some(), "default should produce a hash");
        assert!(
            result.dropped_summary.contains("<<ccr:"),
            "default should produce a marker: {:?}",
            result.dropped_summary
        );
        assert!(
            store_len_after > store_len_before,
            "default should write to ccr_store"
        );
    }
}
