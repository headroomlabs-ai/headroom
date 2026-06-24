//! `SmartCrusherBuilder` — explicit composition of the three traits.
//!
//! `SmartCrusher::new(config)` returns the OSS default composition
//! (HybridScorer + KeepErrorsConstraint + KeepStructuralOutliersConstraint
//! + TracingObserver) — drop-in compatible with pre-PR1 callers.
//!
//! Builder is for callers who want to customize the composition:
//!
//! ```ignore
//! use headroom_core::transforms::smart_crusher::{
//!     SmartCrusher, SmartCrusherConfig, SmartCrusherBuilder,
//! };
//! // Enterprise: swap the scorer, add a business-rule constraint,
//! // attach an audit observer.
//! let crusher = SmartCrusherBuilder::new(SmartCrusherConfig::default())
//!     .with_scorer(Box::new(my_loop_scorer))
//!     .add_default_oss_constraints()        // KeepErrors + KeepStructuralOutliers
//!     .add_constraint(Box::new(my_business_rule))
//!     .add_observer(Box::new(my_audit_observer))
//!     .build();
//! ```
//!
//! # Defaults vs explicit
//!
//! `SmartCrusherBuilder::new()` starts EMPTY — no scorer, no
//! constraints, no observers. You get exactly what you ask for. Use
//! [`with_default_oss_setup`](SmartCrusherBuilder::with_default_oss_setup)
//! to start from the OSS default and customize from there. This is
//! the "no silent fallback" rule applied to composition: the builder
//! makes your intent explicit; the `new()` factory shorthand for the
//! OSS preset.

use std::sync::Arc;

use crate::ccr::{CcrStore, InMemoryCcrStore};
use crate::relevance::{HybridScorer, RelevanceScorer};
use crate::transforms::anchor_selector::{AnchorConfig, AnchorSelector};

use super::analyzer::SmartAnalyzer;
use super::compaction::CompactionStage;
use super::config::SmartCrusherConfig;
use super::constraints::default_oss_constraints;
use super::crusher::SmartCrusher;
use super::observer::TracingObserver;
use super::traits::{Constraint, Observer};

/// Builder for `SmartCrusher`. See module docs.
pub struct SmartCrusherBuilder {
    config: SmartCrusherConfig,
    anchor_config: Option<AnchorConfig>,
    scorer: Option<Box<dyn RelevanceScorer + Send + Sync>>,
    constraints: Vec<Box<dyn Constraint>>,
    observers: Vec<Box<dyn Observer>>,
    compaction: Option<CompactionStage>,
    ccr_store: Option<Arc<dyn CcrStore>>,
}

impl SmartCrusherBuilder {
    /// Empty builder — no scorer, no constraints, no observers, no
    /// compaction stage.
    pub fn new(config: SmartCrusherConfig) -> Self {
        SmartCrusherBuilder {
            config,
            anchor_config: None,
            scorer: None,
            constraints: Vec::new(),
            observers: Vec::new(),
            compaction: None,
            ccr_store: None,
        }
    }

    /// Override the default `AnchorConfig` (rare — most callers leave
    /// this as the default).
    pub fn anchor_config(mut self, cfg: AnchorConfig) -> Self {
        self.anchor_config = Some(cfg);
        self
    }

    /// Set the relevance scorer. The Enterprise plug-in point — pass
    /// a `LoopScorer`, custom `HybridScorer { adaptive: false, alpha: 0.5 }`,
    /// or any other `RelevanceScorer` impl.
    pub fn with_scorer(mut self, scorer: Box<dyn RelevanceScorer + Send + Sync>) -> Self {
        self.scorer = Some(scorer);
        self
    }

    /// Append a constraint. Constraints stack — the must-keep set is
    /// the union of every constraint's output. Order does not affect
    /// correctness but is preserved in observer event strategy strings
    /// for determinism.
    pub fn add_constraint(mut self, c: Box<dyn Constraint>) -> Self {
        self.constraints.push(c);
        self
    }

    /// Append the OSS default constraint stack (`KeepErrorsConstraint`
    /// plus `KeepStructuralOutliersConstraint`) to the current builder.
    /// Composes naturally with `add_constraint`:
    ///
    /// ```ignore
    /// SmartCrusherBuilder::new(cfg)
    ///     .add_default_oss_constraints()
    ///     .add_constraint(Box::new(MyBusinessRule))
    /// ```
    pub fn add_default_oss_constraints(mut self) -> Self {
        self.constraints.extend(default_oss_constraints());
        self
    }

    /// Append an observer. Observers stack — every event fires every
    /// observer in registration order.
    pub fn add_observer(mut self, o: Box<dyn Observer>) -> Self {
        self.observers.push(o);
        self
    }

    /// Apply the OSS default setup: `HybridScorer`,
    /// default-OSS-constraints, `TracingObserver`. Equivalent to
    /// `SmartCrusher::new(config)` if no further customization is
    /// applied. Use this when starting from the OSS preset and
    /// adding a few enterprise components.
    pub fn with_default_oss_setup(self) -> Self {
        self.with_scorer(Box::<HybridScorer>::default())
            .add_default_oss_constraints()
            .add_observer(Box::new(TracingObserver))
    }

    /// Plug in a compaction stage. When set, `crush_array` runs the
    /// stage before the lossy pipeline; if it produces a non-`Untouched`
    /// compaction the rendered bytes are returned via
    /// [`CrushArrayResult::compacted`]. The lossy result still fills
    /// `items` so callers can choose either output.
    ///
    /// [`CrushArrayResult::compacted`]: super::crusher::CrushArrayResult::compacted
    pub fn with_compaction(mut self, stage: CompactionStage) -> Self {
        self.compaction = Some(stage);
        self
    }

    /// Convenience: enable the OSS compaction preset (CSV+schema
    /// formatter, default `CompactConfig`). Equivalent to
    /// `with_compaction(CompactionStage::default_csv_schema())`.
    pub fn with_default_compaction(self) -> Self {
        self.with_compaction(CompactionStage::default_csv_schema())
    }

    /// Plug in a CCR store. The lossy `crush_array` path stashes each
    /// dropped array's full original here keyed by its hash, so the
    /// runtime can serve retrieval tool calls with no data loss.
    pub fn with_ccr_store(mut self, store: Arc<dyn CcrStore>) -> Self {
        self.ccr_store = Some(store);
        self
    }

    /// Convenience: install the default in-memory CCR store
    /// (1000 entries, 5-minute TTL — matches Python).
    pub fn with_default_ccr_store(self) -> Self {
        self.with_ccr_store(Arc::new(InMemoryCcrStore::new()))
    }

    /// Construct the `SmartCrusher`. If `with_scorer` was not called,
    /// falls back to `HybridScorer::default()` so a builder with no
    /// other customization still produces a working crusher.
    pub fn build(self) -> SmartCrusher {
        let analyzer = SmartAnalyzer::new(self.config.clone());
        let anchor_selector = AnchorSelector::new(self.anchor_config.unwrap_or_default());
        let scorer = self
            .scorer
            .unwrap_or_else(|| Box::<HybridScorer>::default());
        SmartCrusher::from_parts(
            self.config,
            anchor_selector,
            scorer,
            analyzer,
            self.constraints,
            self.observers,
            self.compaction,
            self.ccr_store,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transforms::smart_crusher::traits::{Constraint, CrushEvent, Observer};
    use serde_json::Value;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    struct MarkerConstraint {
        name: &'static str,
    }
    impl Constraint for MarkerConstraint {
        fn name(&self) -> &str {
            self.name
        }
        fn must_keep(&self, _: &[Value], _: Option<&[String]>) -> Vec<usize> {
            Vec::new()
        }
    }

    struct MarkerObserver {
        count: Arc<AtomicUsize>,
    }
    impl Observer for MarkerObserver {
        fn on_event(&self, _: &CrushEvent) {
            self.count.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn empty_builder_builds_with_default_scorer() {
        let crusher = SmartCrusherBuilder::new(SmartCrusherConfig::default()).build();
        assert!(crusher.constraints.is_empty());
        assert!(crusher.observers.is_empty());
    }

    #[test]
    fn add_default_oss_constraints_appends_two() {
        let crusher = SmartCrusherBuilder::new(SmartCrusherConfig::default())
            .add_default_oss_constraints()
            .build();
        assert_eq!(crusher.constraints.len(), 2);
        let names: Vec<&str> = crusher.constraints.iter().map(|c| c.name()).collect();
        assert_eq!(names, vec!["keep_errors", "keep_structural_outliers"]);
    }

    #[test]
    fn add_constraint_preserves_order() {
        let crusher = SmartCrusherBuilder::new(SmartCrusherConfig::default())
            .add_constraint(Box::new(MarkerConstraint { name: "first" }))
            .add_constraint(Box::new(MarkerConstraint { name: "second" }))
            .add_constraint(Box::new(MarkerConstraint { name: "third" }))
            .build();
        let names: Vec<&str> = crusher.constraints.iter().map(|c| c.name()).collect();
        assert_eq!(names, vec!["first", "second", "third"]);
    }

    #[test]
    fn with_default_oss_setup_yields_two_constraints_one_observer() {
        let crusher = SmartCrusherBuilder::new(SmartCrusherConfig::default())
            .with_default_oss_setup()
            .build();
        assert_eq!(crusher.constraints.len(), 2);
        assert_eq!(crusher.observers.len(), 1);
    }

    #[test]
    fn builder_observer_fires_on_crush() {
        // Wire a counting observer, run a crush, expect exactly one
        // event. Pins the observer integration end-to-end.
        let counter = Arc::new(AtomicUsize::new(0));
        let crusher = SmartCrusherBuilder::new(SmartCrusherConfig::default())
            .add_observer(Box::new(MarkerObserver {
                count: counter.clone(),
            }))
            .build();
        let _ = crusher.crush(r#"[1, 2, 3]"#, "", 1.0);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }
}
