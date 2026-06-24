//! OSS default `Constraint` implementations.
//!
//! Two constraints ship in the default OSS composition:
//!
//! - [`KeepErrorsConstraint`] — items containing error/failure
//!   keywords (`error`, `exception`, `failed`, `fatal`, etc.). Wraps
//!   the existing [`detect_error_items_for_preservation`] function.
//! - [`KeepStructuralOutliersConstraint`] — items with rare fields or
//!   rare categorical values. Wraps the existing
//!   [`detect_structural_outliers`] function.
//!
//! Both are byte-equivalent to the pre-PR1 hardcoded behavior — the
//! detection logic is unchanged; the constraints are thin trait
//! adapters so that custom Enterprise constraints can be stacked
//! alongside or in place of these defaults.
//!
//! # The default factory
//!
//! [`default_oss_constraints`] returns the OSS default stack as a
//! `Vec<Box<dyn Constraint>>`. `SmartCrusher::new(config)` uses this;
//! `SmartCrusherBuilder::new(config)` does NOT (the builder gives you
//! exactly what you ask for, no surprises). To get OSS defaults plus
//! your own constraints, use `with_default_constraints()` on the
//! builder.

use serde_json::Value;

use super::outliers::{detect_error_items_for_preservation, detect_structural_outliers};
use super::traits::Constraint;

// ── KeepErrorsConstraint ──────────────────────────────────────────────────

/// OSS default: keep items that contain error keywords.
///
/// "Error keyword" is matched case-insensitively against the JSON
/// serialization of each item against the list in [`super::error_keywords::ERROR_KEYWORDS`]
/// (`error`, `exception`, `failed`, `fatal`, `critical`, `crash`, `panic`,
/// `abort`, `timeout`, `denied`, `rejected`).
#[derive(Debug, Default, Clone, Copy)]
pub struct KeepErrorsConstraint;

impl Constraint for KeepErrorsConstraint {
    fn name(&self) -> &str {
        "keep_errors"
    }

    fn must_keep(&self, items: &[Value], item_strings: Option<&[String]>) -> Vec<usize> {
        detect_error_items_for_preservation(items, item_strings)
    }
}

// ── KeepStructuralOutliersConstraint ─────────────────────────────────────

/// OSS default: keep items that are *structurally* unusual within the
/// array. Two flavors of unusual:
///
/// - **Rare fields**: items that have a key present in fewer than the
///   uniqueness threshold of items.
/// - **Rare values for common fields**: items whose value for a
///   high-cardinality field appears infrequently (the "rare-status"
///   path that fires on enums like `level: "ERROR"` among many `INFO`).
///
/// Implementation is unchanged from pre-PR1; this is a thin wrapper.
#[derive(Debug, Default, Clone, Copy)]
pub struct KeepStructuralOutliersConstraint;

impl Constraint for KeepStructuralOutliersConstraint {
    fn name(&self) -> &str {
        "keep_structural_outliers"
    }

    fn must_keep(&self, items: &[Value], _item_strings: Option<&[String]>) -> Vec<usize> {
        detect_structural_outliers(items)
    }
}

// ── Default OSS stack ─────────────────────────────────────────────────────

/// Returns the default OSS constraint stack used by
/// `SmartCrusher::new(config)`.
///
/// Stack contents (in order):
/// 1. [`KeepErrorsConstraint`]
/// 2. [`KeepStructuralOutliersConstraint`]
///
/// The order does not affect output (the must-keep set is a union),
/// but is fixed for determinism in observer-emitted strategy strings
/// and audit logs.
pub fn default_oss_constraints() -> Vec<Box<dyn Constraint>> {
    vec![
        Box::new(KeepErrorsConstraint),
        Box::new(KeepStructuralOutliersConstraint),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn keep_errors_constraint_finds_error_items() {
        // 9 normal items + 1 with "ERROR" keyword.
        let mut items: Vec<Value> = (0..9).map(|i| json!({"id": i, "status": "ok"})).collect();
        items.push(json!({"id": 9, "status": "ERROR", "msg": "FATAL: boom"}));
        let kept = KeepErrorsConstraint.must_keep(&items, None);
        // Exact index 9 must be in the result.
        assert!(kept.contains(&9), "error item must be flagged for keep");
    }

    #[test]
    fn keep_errors_constraint_uses_item_strings_when_provided() {
        // Pre-computed strings parity with the on-the-fly path: same
        // content, same indices returned.
        let items: Vec<Value> = vec![json!({"a": 1}), json!({"a": "exception"})];
        let strings: Vec<String> = items
            .iter()
            .map(|v| serde_json::to_string(v).unwrap())
            .collect();
        let with_cache = KeepErrorsConstraint.must_keep(&items, Some(&strings));
        let without_cache = KeepErrorsConstraint.must_keep(&items, None);
        assert_eq!(with_cache, without_cache);
        assert!(with_cache.contains(&1));
    }

    #[test]
    fn keep_structural_outliers_constraint_returns_indices() {
        // Build an array where one item has a unique extra field —
        // it should be flagged as a rare-field outlier.
        let mut items: Vec<Value> = (0..20)
            .map(|i| json!({"id": i, "kind": "common"}))
            .collect();
        items.push(json!({"id": 20, "kind": "common", "rare_extra_field": "x"}));
        let kept = KeepStructuralOutliersConstraint.must_keep(&items, None);
        assert!(
            kept.contains(&20),
            "item with rare field should be a structural outlier"
        );
    }

    #[test]
    fn default_oss_constraints_returns_two() {
        let cs = default_oss_constraints();
        assert_eq!(cs.len(), 2);
        let names: Vec<&str> = cs.iter().map(|c| c.name()).collect();
        assert_eq!(names, vec!["keep_errors", "keep_structural_outliers"]);
    }

    #[test]
    fn constraints_handle_empty_array() {
        // No panics, no allocations beyond an empty Vec — the array
        // path will bypass us when items is empty, but constraints
        // must still be safe to call.
        assert!(KeepErrorsConstraint.must_keep(&[], None).is_empty());
        assert!(KeepStructuralOutliersConstraint
            .must_keep(&[], None)
            .is_empty());
    }
}
