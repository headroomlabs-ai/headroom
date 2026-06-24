//! Smart statistical tool output compression — Rust port of
//! `headroom/transforms/smart_crusher.py`.
//!
//! # Stage 3c.1: like-for-like parity port
//!
//! This module is a literal Rust port of the Python `SmartCrusher`
//! implementation. The goal of Stage 3c.1 is **byte-equal output parity** for
//! every fixture in `tests/parity/fixtures/smart_crusher/`. Architectural
//! improvements (lossless-first, unified saliency score, structured CCR
//! markers, token budget) are deferred to Stage 3c.2 and tracked in
//! `~/Desktop/SmartCrusher-Architecture-Improvements.md`.
//!
//! # Bugs fixed in BOTH Python and Rust during 3c.1
//!
//! Four defects in the Python source (`headroom/transforms/smart_crusher.py`)
//! were caught during port review. They're fixed in both languages
//! simultaneously so the parity fixtures continue to byte-match:
//!
//! - **k-split overshoot** (line 2722): `_compute_k_split` keeps 2 items when
//!   `k_total = 1` because `max(1, round(k_total * fraction))` floors both
//!   first and last to 1. Violates `max_items_after_crush`.
//! - **sequential-pattern false positive** (line 444): `_detect_sequential_pattern`
//!   does `int("001")` and silently loses zero padding. Padded string IDs
//!   misclassified as sequential numeric IDs.
//! - **rare-status detection short-circuit** (line 674): `_detect_rare_status_values`
//!   exits early at >10 distinct values. Datasets with 50+ error codes lose
//!   rare-error preservation.
//! - **percentile off-by-one** (line 2844): For `len < 8`, integer-division
//!   percentile indices are off by one. Cosmetic — only affects strategy
//!   debug strings.
//!
//! Each fix has a fixture entry in the parity harness and a corresponding
//! test in `tests/test_transforms/test_smart_crusher_bugs.py`.

mod analyzer;
mod anchors;
mod builder;
mod classifier;
pub mod compaction;
mod config;
mod constraints;
mod crusher;
mod crushers;
mod error_keywords;
mod field_detect;
mod hashing;
mod observer;
mod orchestration;
mod outliers;
mod planning;
mod statistics;
mod stats_math;
mod traits;
mod types;

pub use analyzer::SmartAnalyzer;
pub use anchors::{extract_query_anchors, item_matches_anchors};
pub use builder::SmartCrusherBuilder;
pub use classifier::{classify_array, ArrayType};
pub use config::SmartCrusherConfig;
pub use constraints::{
    default_oss_constraints, KeepErrorsConstraint, KeepStructuralOutliersConstraint,
};
pub use crusher::{CrushArrayResult, SmartCrusher};
pub use crushers::{compute_k_split, crush_number_array, crush_object, crush_string_array};
pub use error_keywords::ERROR_KEYWORDS;
pub use field_detect::{detect_id_field_statistically, detect_score_field_statistically};
pub use hashing::hash_field_name;
pub use observer::TracingObserver;
pub use orchestration::{deduplicate_indices_by_content, fill_remaining_slots, prioritize_indices};
pub use outliers::{
    detect_error_items_for_preservation, detect_rare_status_values, detect_structural_outliers,
};
pub use planning::{item_has_preserve_field_match, map_to_anchor_pattern, SmartCrusherPlanner};
pub use statistics::{calculate_string_entropy, detect_sequential_pattern, is_uuid_format};
pub use stats_math::{format_g, mean, median, sample_stdev, sample_variance};
pub use traits::{Constraint, CrushEvent, Observer, Scorer};
pub use types::{
    ArrayAnalysis, CompressionPlan, CompressionStrategy, CrushResult, CrushabilityAnalysis,
    FieldStats,
};
