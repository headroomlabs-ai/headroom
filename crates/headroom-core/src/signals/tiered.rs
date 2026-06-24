//! Composition combinator for layered detectors.
//!
//! `Tiered<dyn Trait>` chains an ordered list of detectors. The first
//! tier whose signal exceeds [`ESCALATE_THRESHOLD`] confidence wins;
//! lower-confidence tiers are skipped past. If no tier exceeds the
//! threshold, the highest-confidence signal seen is returned (so the
//! caller still gets the best guess, with the confidence score
//! reflecting how unsure the stack is).
//!
//! Tiering is *composition*, not inheritance. `KeywordDetector` knows
//! nothing about a future ML detector; the ML detector knows nothing
//! about the keyword detector. They both implement the trait and the
//! `Tiered` wrapper orders them.

use super::line_importance::{ImportanceContext, ImportanceSignal, LineImportanceDetector};

/// Confidence at which `Tiered` accepts a tier's signal without
/// consulting later tiers. KeywordDetector emits 0.7, so it wins by
/// default; an ML tier with calibrated confidence ≥ 0.8 (high-precision
/// region) would short-circuit the keyword tier.
pub const ESCALATE_THRESHOLD: f32 = 0.7;

pub struct Tiered<T: ?Sized> {
    tiers: Vec<Box<T>>,
}

impl<T: ?Sized> Tiered<T> {
    pub fn new() -> Self {
        Self { tiers: Vec::new() }
    }

    /// Push a tier onto the stack. Order matters: most-precise first.
    pub fn with(mut self, tier: Box<T>) -> Self {
        self.tiers.push(tier);
        self
    }

    pub fn len(&self) -> usize {
        self.tiers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tiers.is_empty()
    }
}

impl<T: ?Sized> Default for Tiered<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl LineImportanceDetector for Tiered<dyn LineImportanceDetector> {
    fn score(&self, line: &str, ctx: ImportanceContext) -> ImportanceSignal {
        let mut best = ImportanceSignal::neutral();
        for tier in &self.tiers {
            let signal = tier.score(line, ctx);
            if signal.confidence >= ESCALATE_THRESHOLD {
                return signal;
            }
            if signal.confidence > best.confidence {
                best = signal;
            }
        }
        best
    }
}

impl Tiered<dyn LineImportanceDetector> {
    /// Convenience: take an owned detector, box it, coerce it to the
    /// trait object. Keeps callsites free of `as Box<dyn …>` clutter.
    pub fn with_detector<D: LineImportanceDetector + 'static>(self, detector: D) -> Self {
        self.with(Box::new(detector) as Box<dyn LineImportanceDetector>)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signals::keyword_detector::KeywordDetector;
    use crate::signals::line_importance::ImportanceCategory;

    /// Synthetic high-confidence detector for testing short-circuit
    /// behavior. Always asserts a specific signal so we can prove
    /// `Tiered` consults it before the keyword tier.
    struct AlwaysFiresHigh;
    impl LineImportanceDetector for AlwaysFiresHigh {
        fn score(&self, _line: &str, _ctx: ImportanceContext) -> ImportanceSignal {
            ImportanceSignal::matched(ImportanceCategory::Security, 0.99, 0.95)
        }
    }

    /// Synthetic low-confidence detector. Confidence 0.5 is below the
    /// escalate threshold so `Tiered` MUST fall through to the next
    /// tier.
    struct AlwaysFiresLow;
    impl LineImportanceDetector for AlwaysFiresLow {
        fn score(&self, _line: &str, _ctx: ImportanceContext) -> ImportanceSignal {
            ImportanceSignal::matched(ImportanceCategory::Importance, 0.4, 0.5)
        }
    }

    #[test]
    fn high_confidence_tier_short_circuits() {
        let tiered: Tiered<dyn LineImportanceDetector> = Tiered::new()
            .with_detector(AlwaysFiresHigh)
            .with_detector(KeywordDetector::new());
        let s = tiered.score("ERROR: connection refused", ImportanceContext::Diff);
        // AlwaysFiresHigh asserts Security; if the keyword detector ran
        // it would have asserted Error.
        assert_eq!(s.category, Some(ImportanceCategory::Security));
    }

    #[test]
    fn low_confidence_tier_falls_through_to_keyword() {
        let tiered: Tiered<dyn LineImportanceDetector> = Tiered::new()
            .with_detector(AlwaysFiresLow)
            .with_detector(KeywordDetector::new());
        let s = tiered.score("ERROR: connection refused", ImportanceContext::Diff);
        assert_eq!(s.category, Some(ImportanceCategory::Error));
    }

    #[test]
    fn no_tier_matches_returns_best_seen() {
        let tiered: Tiered<dyn LineImportanceDetector> = Tiered::new()
            .with_detector(AlwaysFiresLow)
            .with_detector(KeywordDetector::new());
        let s = tiered.score("the quick brown fox", ImportanceContext::Text);
        // Keyword detector returns neutral (confidence 0.0); AlwaysFiresLow
        // returned Importance @ 0.5 so that wins as best-seen.
        assert_eq!(s.category, Some(ImportanceCategory::Importance));
        assert_eq!(s.confidence, 0.5);
    }

    #[test]
    fn empty_stack_returns_neutral() {
        let tiered: Tiered<dyn LineImportanceDetector> = Tiered::new();
        let s = tiered.score("anything", ImportanceContext::Text);
        assert!(!s.is_match());
    }
}
