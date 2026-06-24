//! Detection-trait module — cross-cutting classifiers used by transforms.
//!
//! # Why a top-level module
//!
//! Transforms in [`crate::transforms`] mutate data; signals in this module
//! *classify* it. The same classifier feeds many transforms (e.g. line
//! importance scoring is consumed by `text_compressor`, `search_compressor`,
//! `diff_compressor`, and `log_compressor`), so the layering belongs at the
//! crate root, not nested under any one transform.
//!
//! # The shape we follow
//!
//! Detection in Headroom matures along a known curve:
//!
//! 1. **Pattern fallback** — keyword/regex scanning. Cheap, brittle, the
//!    starting point for every detector. This is what
//!    [`keyword_detector::KeywordDetector`] gives us today.
//! 2. **Structured parser** — when the input has grammar (diffs, JSON,
//!    code), parse it. Already done for `unidiff` (content type) and
//!    `tree-sitter` (language).
//! 3. **ML model** — for fuzzy categories (line importance, anchor cells,
//!    HTML extraction), a small classifier trained on labeled traffic
//!    outperforms keywords. The canonical extension path here is a
//!    classification head on the existing `bge-small-en-v1.5` embedder
//!    (already loaded for `relevance`); see `signals/README.md`.
//!
//! All three live behind the same per-granularity trait. Tiering is
//! *composition* via [`tiered::Tiered`] — never inheritance. A future ML
//! detector slots in as a new tier without touching the keyword detector
//! or any caller.
//!
//! # Per-granularity, not per-domain
//!
//! Different inputs warrant different trait signatures:
//!
//! - [`line_importance::LineImportanceDetector`] — single line → priority
//! - (future) `ContentTypeDetector` — whole blob → category
//! - (future) `ItemImportanceDetector<I>` — `&[I]` → ranking
//!
//! Cramming everything into one `Detector<Any>` would force callers to
//! match on input shape at every site. Three traits keep each callsite
//! type-checked.
//!
//! # No silent fallbacks
//!
//! Per project conventions, every concrete impl in this module is real.
//! No `NoOpDetector`, no stub-ML impl that returns zeros, no
//! "fallback" classifier that quietly degrades. If a tier is registered,
//! it does the work; if no tier confidently matches, the signal carries
//! that fact in its `confidence` field rather than being silently
//! coerced to a positive answer.

pub mod keyword_detector;
pub mod line_importance;
pub mod tiered;

pub use keyword_detector::{KeywordDetector, KeywordRegistry};
pub use line_importance::{
    ImportanceCategory, ImportanceContext, ImportanceSignal, LineImportanceDetector,
};
pub use tiered::Tiered;
