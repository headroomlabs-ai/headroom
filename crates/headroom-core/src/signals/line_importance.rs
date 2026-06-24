//! Line-level importance detection trait.
//!
//! Compressors call this when deciding which lines to drop under a token
//! budget. The signal carries category, priority, and confidence — never
//! a bare bool — so future tiers can short-circuit on high confidence
//! and lower-priority callers can fall through.

/// Where the line came from. Determines which pattern set fires (e.g.
/// markdown headers count as priority signals in prose, but not in diff
/// hunks).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImportanceContext {
    /// Free-form prose (text_compressor) — markdown structure matters.
    Text,
    /// grep/ripgrep output (search_compressor) — error/warn keywords win.
    Search,
    /// git diff (diff_compressor) — error + security + importance keywords.
    Diff,
    /// Log output (log_compressor) — error/warn keywords + level prefixes.
    Log,
}

/// Why a line earned its priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImportanceCategory {
    Error,
    Warning,
    Importance,
    Security,
    /// Markdown structure — headers, bold, blockquotes. Only meaningful
    /// in `ImportanceContext::Text`.
    Markdown,
}

/// Output of a single detector for a single line.
///
/// `priority` is what compressors rank by; `confidence` is what the
/// [`super::tiered::Tiered`] combinator uses to decide whether to keep
/// asking the next tier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImportanceSignal {
    /// The category the detector matched on, if any.
    pub category: Option<ImportanceCategory>,
    /// 0.0 = drop first, 1.0 = keep at all costs.
    pub priority: f32,
    /// 0.0 = no information, 1.0 = the detector is sure.
    pub confidence: f32,
}

impl ImportanceSignal {
    /// "I have no opinion on this line." Returned when nothing matched.
    pub const fn neutral() -> Self {
        Self {
            category: None,
            priority: 0.0,
            confidence: 0.0,
        }
    }

    /// A fired detection with explicit category and priority.
    pub const fn matched(category: ImportanceCategory, priority: f32, confidence: f32) -> Self {
        Self {
            category: Some(category),
            priority,
            confidence,
        }
    }

    /// True when the detector saw something it recognized.
    pub fn is_match(&self) -> bool {
        self.category.is_some()
    }
}

/// Single-line importance classifier.
///
/// Implementations are expected to be cheap (keyword automaton, lexical
/// features) or amortizable (embedding+classifier head with batched
/// inference). They MUST be `Send + Sync` because compressors share
/// detector instances across tokio worker threads.
pub trait LineImportanceDetector: Send + Sync {
    /// Score a single line in the given context.
    fn score(&self, line: &str, ctx: ImportanceContext) -> ImportanceSignal;
}
