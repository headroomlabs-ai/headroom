//! Compaction IR — recursive tree representation for lossless / row-lossy
//! compaction of JSON arrays.
//!
//! The IR is the boundary between [`TabularCompactor`] (which produces it)
//! and [`Formatter`] implementations (which consume it). Renderer-agnostic.
//!
//! # Recursive structure
//!
//! A `Compaction::Table` has rows of [`CellValue`]s, and a `CellValue` may
//! itself hold a nested `Compaction`. This enables multi-level compression:
//! an array whose rows hold stringified-JSON gets recursively compacted
//! into a sub-table; an opaque blob gets CCR-substituted; a heterogeneous
//! array gets bucketed by discriminator.
//!
//! [`TabularCompactor`]: super::compactor::TabularCompactor
//! [`Formatter`]: super::formatter::Formatter

use serde_json::Value;

/// What kind of opaque payload was substituted by CCR.
///
/// Carried for telemetry and so formatters can render a one-line hint
/// next to the CCR pointer (e.g. `<<ccr:abc123 base64,2.1KB>>`) without
/// re-parsing the original bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpaqueKind {
    /// Looks base64-encoded — long, restricted alphabet.
    Base64Blob,
    /// Long opaque string the classifier couldn't otherwise place.
    LongString,
    /// HTML/XML chunk (detected by `<` density).
    HtmlChunk,
    /// Detected format the classifier knows about by name (e.g. "diff",
    /// "code"). Routing of these into the right transform is deferred
    /// to a later PR; for now they're treated as `LongString`.
    Other(String),
}

/// One column's metadata in a tabular compaction.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldSpec {
    /// Column name. May be dotted for flattened nested fields,
    /// e.g. `"meta.region"`.
    pub name: String,
    /// Inferred type tag. One of: `"int"`, `"float"`, `"string"`,
    /// `"bool"`, `"null"`, `"json"` (cells render as JSON literals —
    /// last-resort), `"ccr"` (cells are CCR pointers).
    pub type_tag: String,
    /// True if at least one row had this field absent or `null`.
    pub nullable: bool,
}

/// Column set for a homogeneous table.
#[derive(Debug, Clone, PartialEq)]
pub struct Schema {
    pub fields: Vec<FieldSpec>,
}

impl Schema {
    pub fn field_names(&self) -> Vec<&str> {
        self.fields.iter().map(|f| f.name.as_str()).collect()
    }
}

/// One cell in a row. Most cells are scalar; nested/opaque/recursive
/// cells branch the tree.
#[derive(Debug, Clone)]
pub enum CellValue {
    /// Scalar JSON value (number, string, bool, null). Formatter renders
    /// directly per its conventions.
    Scalar(Value),
    /// Recursive sub-compaction. Created for inner arrays, parsed
    /// stringified-JSON, or nested-mixed objects. Formatter recurses.
    Nested(Box<Compaction>),
    /// CCR pointer substituting an opaque/large payload. The original
    /// bytes live in the CCR store keyed by `ccr_hash`.
    OpaqueRef {
        ccr_hash: String,
        byte_size: usize,
        kind: OpaqueKind,
    },
    /// Field is absent in this row. Distinct from `Scalar(Value::Null)`
    /// — `Missing` means the original object had no such key, while
    /// `Scalar(Value::Null)` means the key existed and was null.
    Missing,
}

/// A row of a tabular compaction. Order and length match the parent
/// table's [`Schema::fields`].
#[derive(Debug, Clone)]
pub struct Row(pub Vec<CellValue>);

impl Row {
    pub fn new(cells: Vec<CellValue>) -> Self {
        Self(cells)
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// One bucket of a heterogeneous array, partitioned by a discriminator
/// field's value (e.g. all rows where `type == "user"`).
#[derive(Debug, Clone)]
pub struct Bucket {
    /// The discriminator value that defines this bucket.
    pub key: Value,
    pub schema: Schema,
    pub rows: Vec<Row>,
}

/// Top-level compaction result. Tree-shaped via `Nested` cells.
///
/// [`Compaction::Table`] is the common case. [`Compaction::Buckets`]
/// only fires for heterogeneous arrays where a discriminator field
/// cleanly partitions rows. [`Compaction::Untouched`] is the
/// fall-through when the compactor declines to operate (e.g. mixed
/// scalars, or fewer than 2 rows).
#[derive(Debug, Clone)]
pub enum Compaction {
    /// Homogeneous tabular form: N rows × C columns.
    Table {
        schema: Schema,
        rows: Vec<Row>,
        /// Row count BEFORE any row-dropping under budget pressure.
        /// `original_count - rows.len()` = rows we had to drop.
        original_count: usize,
    },
    /// Heterogeneous array bucketed by discriminator field.
    Buckets {
        discriminator: String,
        buckets: Vec<Bucket>,
        /// Total rows across all buckets BEFORE row-dropping.
        original_count: usize,
    },
    /// Single CCR pointer — top-level opaque content. Rare; usually
    /// CCR refs live inside table cells, not at the top.
    OpaqueRef {
        ccr_hash: String,
        byte_size: usize,
        kind: OpaqueKind,
    },
    /// Compactor declined to compact; pass-through original value.
    /// The crusher will fall back to the existing lossy path.
    Untouched(Value),
}

impl Compaction {
    /// Total kept rows in this compaction (sum across buckets if
    /// applicable). 0 for `OpaqueRef` and `Untouched`.
    pub fn kept_row_count(&self) -> usize {
        match self {
            Compaction::Table { rows, .. } => rows.len(),
            Compaction::Buckets { buckets, .. } => buckets.iter().map(|b| b.rows.len()).sum(),
            Compaction::OpaqueRef { .. } | Compaction::Untouched(_) => 0,
        }
    }

    /// Original (pre-drop) row count. 0 for `OpaqueRef` and `Untouched`.
    pub fn original_row_count(&self) -> usize {
        match self {
            Compaction::Table { original_count, .. } => *original_count,
            Compaction::Buckets { original_count, .. } => *original_count,
            Compaction::OpaqueRef { .. } | Compaction::Untouched(_) => 0,
        }
    }

    pub fn was_compacted(&self) -> bool {
        matches!(
            self,
            Compaction::Table { .. } | Compaction::Buckets { .. } | Compaction::OpaqueRef { .. }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn schema_field_names_returns_in_order() {
        let s = Schema {
            fields: vec![
                FieldSpec {
                    name: "id".into(),
                    type_tag: "int".into(),
                    nullable: false,
                },
                FieldSpec {
                    name: "name".into(),
                    type_tag: "string".into(),
                    nullable: false,
                },
            ],
        };
        assert_eq!(s.field_names(), vec!["id", "name"]);
    }

    #[test]
    fn untouched_is_not_compacted() {
        let c = Compaction::Untouched(json!([1, 2, 3]));
        assert!(!c.was_compacted());
        assert_eq!(c.kept_row_count(), 0);
        assert_eq!(c.original_row_count(), 0);
    }

    #[test]
    fn table_row_counts() {
        let c = Compaction::Table {
            schema: Schema { fields: vec![] },
            rows: vec![Row::new(vec![]), Row::new(vec![])],
            original_count: 5,
        };
        assert!(c.was_compacted());
        assert_eq!(c.kept_row_count(), 2);
        assert_eq!(c.original_row_count(), 5);
    }

    #[test]
    fn buckets_aggregate_row_counts() {
        let c = Compaction::Buckets {
            discriminator: "type".into(),
            buckets: vec![
                Bucket {
                    key: json!("user"),
                    schema: Schema { fields: vec![] },
                    rows: vec![Row::new(vec![]), Row::new(vec![])],
                },
                Bucket {
                    key: json!("order"),
                    schema: Schema { fields: vec![] },
                    rows: vec![Row::new(vec![])],
                },
            ],
            original_count: 10,
        };
        assert_eq!(c.kept_row_count(), 3);
        assert_eq!(c.original_row_count(), 10);
    }

    #[test]
    fn cell_missing_distinct_from_scalar_null() {
        let m = CellValue::Missing;
        let n = CellValue::Scalar(Value::Null);
        // Smoke test: just confirm both variants exist and Debug differs.
        assert_ne!(format!("{m:?}"), format!("{n:?}"));
    }
}
