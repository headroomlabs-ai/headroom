//! Formatter trait + the built-in implementations.
//!
//! [`Formatter`] walks a [`Compaction`] tree and renders bytes. It's the
//! pluggable seam where users (or Enterprise plugins) choose how the
//! compacted output looks.
//!
//! # Built-ins
//!
//! - [`JsonFormatter`] — single-line / pretty JSON. Easy to parse,
//!   wider model familiarity, larger byte size. Default for the
//!   debugging path.
//! - [`CsvSchemaFormatter`] — `[N]{cols}` row-count-and-shape
//!   declaration + typed column header + CSV-escaped rows. Steals
//!   TOON's most useful idea (the `[N]{cols}` declaration) without
//!   adopting TOON's bespoke escaping rules — every model has seen
//!   millions of CSV examples in training.
//! - [`MarkdownKvFormatter`] — the same `[N]{cols}` declaration +
//!   one Markdown list item per row with `key: value` lines.
//!   Token-heavier than CSV (field names repeat per row) but
//!   format-comprehension benchmarks favor KV for read-back accuracy.
//!
//! # Nested cells
//!
//! The formatters handle [`CellValue::Nested`] by recursively
//! formatting the sub-compaction and embedding the result. The CSV
//! formatter wraps nested output in CSV-quoted form; the JSON
//! formatter embeds it as a structured JSON object.
//!
//! # Opaque cells
//!
//! [`CellValue::OpaqueRef`] renders as a structured marker the model
//! can recognize: `<<ccr:HASH,KIND,SIZE>>`. This format is fixed across
//! all built-in formatters so downstream consumers can pattern-match
//! markers regardless of which formatter produced them.

use serde_json::{json, Value};

use super::ir::{CellValue, Compaction, OpaqueKind, Row, Schema};

/// Format a `Compaction` tree into bytes.
pub trait Formatter: Send + Sync {
    /// Stable name for telemetry (e.g. `"json"`, `"csv-schema"`).
    fn name(&self) -> &str;

    /// Render the compaction. Implementations should be deterministic
    /// for stable test parity.
    fn format(&self, c: &Compaction) -> String;

    /// Cheap byte-size estimate. Default impl renders and measures.
    /// Override for cases where rendering is expensive.
    fn estimate_bytes(&self, c: &Compaction) -> usize {
        self.format(c).len()
    }
}

// ─────────────────────────── JSON formatter ───────────────────────────

/// Renders a `Compaction` as structured JSON. Single-line by default
/// for token-tight output; set `pretty = true` for human inspection.
#[derive(Debug, Clone, Default)]
pub struct JsonFormatter {
    pub pretty: bool,
}

impl JsonFormatter {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn pretty(mut self) -> Self {
        self.pretty = true;
        self
    }
}

impl Formatter for JsonFormatter {
    fn name(&self) -> &str {
        "json"
    }

    fn format(&self, c: &Compaction) -> String {
        let v = compaction_to_json(c);
        if self.pretty {
            serde_json::to_string_pretty(&v).unwrap_or_default()
        } else {
            serde_json::to_string(&v).unwrap_or_default()
        }
    }
}

fn compaction_to_json(c: &Compaction) -> Value {
    match c {
        Compaction::Table {
            schema,
            rows,
            original_count,
        } => json!({
            "_compaction": "table",
            "_schema": schema_to_json(schema),
            "_kept": rows.len(),
            "_total": original_count,
            "_rows": rows.iter().map(row_to_json).collect::<Vec<_>>(),
        }),
        Compaction::Buckets {
            discriminator,
            buckets,
            original_count,
        } => json!({
            "_compaction": "buckets",
            "_discriminator": discriminator,
            "_total": original_count,
            "_buckets": buckets
                .iter()
                .map(|b| json!({
                    "_key": b.key.clone(),
                    "_schema": schema_to_json(&b.schema),
                    "_rows": b.rows.iter().map(row_to_json).collect::<Vec<_>>(),
                }))
                .collect::<Vec<_>>(),
        }),
        Compaction::OpaqueRef {
            ccr_hash,
            byte_size,
            kind,
        } => json!({
            "_compaction": "ccr",
            "_hash": ccr_hash,
            "_size": byte_size,
            "_kind": opaque_kind_str(kind),
        }),
        Compaction::Untouched(v) => v.clone(),
    }
}

fn schema_to_json(s: &Schema) -> Value {
    Value::Array(
        s.fields
            .iter()
            .map(|f| {
                let mut obj = serde_json::Map::new();
                obj.insert("name".into(), Value::String(f.name.clone()));
                obj.insert("type".into(), Value::String(f.type_tag.clone()));
                if f.nullable {
                    obj.insert("nullable".into(), Value::Bool(true));
                }
                Value::Object(obj)
            })
            .collect(),
    )
}

fn row_to_json(row: &Row) -> Value {
    Value::Array(row.0.iter().map(cell_to_json).collect())
}

fn cell_to_json(c: &CellValue) -> Value {
    match c {
        CellValue::Scalar(v) => v.clone(),
        CellValue::Missing => Value::Null,
        CellValue::Nested(sub) => compaction_to_json(sub),
        CellValue::OpaqueRef {
            ccr_hash,
            byte_size,
            kind,
        } => json!({
            "_ccr": ccr_hash,
            "_size": byte_size,
            "_kind": opaque_kind_str(kind),
        }),
    }
}

fn opaque_kind_str(k: &OpaqueKind) -> String {
    match k {
        OpaqueKind::Base64Blob => "base64".into(),
        OpaqueKind::LongString => "string".into(),
        OpaqueKind::HtmlChunk => "html".into(),
        OpaqueKind::Other(s) => s.clone(),
    }
}

// ─────────────────────────── CSV+schema formatter ───────────────────────────

/// Renders a `Compaction` as `[N]{col1:type1,col2:type2}` declaration +
/// CSV-escaped rows. Nested cells render as JSON inline; opaque cells
/// render as `<<ccr:...>>` markers.
#[derive(Debug, Clone, Default)]
pub struct CsvSchemaFormatter {
    /// If true, emit a `__total:N` line when rows were dropped under
    /// budget. Costs a few bytes; useful for downstream telemetry.
    pub include_drop_summary: bool,
}

impl CsvSchemaFormatter {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_drop_summary(mut self) -> Self {
        self.include_drop_summary = true;
        self
    }
}

impl Formatter for CsvSchemaFormatter {
    fn name(&self) -> &str {
        "csv-schema"
    }

    fn format(&self, c: &Compaction) -> String {
        let mut out = String::new();
        write_compaction(&mut out, c, self);
        out
    }
}

fn write_compaction(out: &mut String, c: &Compaction, fmt: &CsvSchemaFormatter) {
    match c {
        Compaction::Table {
            schema,
            rows,
            original_count,
        } => {
            write_table(out, schema, rows, *original_count, fmt);
        }
        Compaction::Buckets {
            discriminator,
            buckets,
            original_count,
        } => {
            out.push_str("__buckets:");
            out.push_str(discriminator);
            if fmt.include_drop_summary {
                let kept: usize = buckets.iter().map(|b| b.rows.len()).sum();
                if kept < *original_count {
                    out.push_str(&format!(" __dropped:{}", original_count - kept));
                }
            }
            out.push('\n');
            for b in buckets {
                out.push_str(&format!("__key:{}\n", json_scalar_to_csv(&b.key)));
                write_table(out, &b.schema, &b.rows, b.rows.len(), fmt);
            }
        }
        Compaction::OpaqueRef {
            ccr_hash,
            byte_size,
            kind,
        } => {
            out.push_str(&format_ccr_marker(ccr_hash, *byte_size, kind));
        }
        Compaction::Untouched(v) => {
            out.push_str(&serde_json::to_string(v).unwrap_or_default());
        }
    }
}

fn write_table(
    out: &mut String,
    schema: &Schema,
    rows: &[Row],
    original_count: usize,
    fmt: &CsvSchemaFormatter,
) {
    // Declaration line: [N]{col:type,col:type,...}
    out.push('[');
    out.push_str(&rows.len().to_string());
    out.push_str("]{");
    let col_decl: Vec<String> = schema
        .fields
        .iter()
        .map(|f| {
            if f.nullable {
                format!("{}:{}?", f.name, f.type_tag)
            } else {
                format!("{}:{}", f.name, f.type_tag)
            }
        })
        .collect();
    out.push_str(&col_decl.join(","));
    out.push('}');
    if fmt.include_drop_summary && rows.len() < original_count {
        out.push_str(&format!(" __dropped:{}", original_count - rows.len()));
    }
    out.push('\n');

    // Rows.
    for row in rows {
        let cells: Vec<String> = row.0.iter().map(format_cell).collect();
        out.push_str(&cells.join(","));
        out.push('\n');
    }
}

/// Bare token for an absent key, distinct from null's empty cell. Lets the
/// decoder tell `missing key` (`\N`) from `null` (empty) from `""` (`""`).
/// A literal string equal to this token is CSV-quoted so it never collides.
const MISSING_SENTINEL: &str = "\\N";

fn format_cell(c: &CellValue) -> String {
    match c {
        // Absent key → sentinel; null → empty cell. Previously both rendered
        // empty, collapsing the missing/null/empty-string distinction (lossy).
        CellValue::Missing => MISSING_SENTINEL.to_string(),
        CellValue::Scalar(v) => json_scalar_to_csv(v),
        CellValue::Nested(sub) => {
            // Render nested as compact JSON; CSV-quote because it
            // contains commas and structural chars.
            let nested_fmt = JsonFormatter::new();
            csv_quote(&nested_fmt.format(sub))
        }
        CellValue::OpaqueRef {
            ccr_hash,
            byte_size,
            kind,
        } => format_ccr_marker(ccr_hash, *byte_size, kind),
    }
}

fn format_ccr_marker(hash: &str, byte_size: usize, kind: &OpaqueKind) -> String {
    let kind_str = match kind {
        OpaqueKind::Base64Blob => "base64",
        OpaqueKind::LongString => "string",
        OpaqueKind::HtmlChunk => "html",
        OpaqueKind::Other(s) => s.as_str(),
    };
    format!(
        "<<ccr:{},{},{}>>",
        hash,
        kind_str,
        humanize_bytes(byte_size)
    )
}

fn humanize_bytes(n: usize) -> String {
    if n < 1024 {
        return format!("{n}B");
    }
    let kb = n as f64 / 1024.0;
    if kb < 1024.0 {
        return format!("{kb:.1}KB");
    }
    let mb = kb / 1024.0;
    format!("{mb:.1}MB")
}

fn json_scalar_to_csv(v: &Value) -> String {
    match v {
        Value::Null => String::new(),
        Value::Bool(b) => if *b { "true" } else { "false" }.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => {
            if needs_csv_quote(s) {
                csv_quote(s)
            } else {
                s.clone()
            }
        }
        // Object/array fall back to JSON-quoted (rare — usually
        // already promoted to Nested by the compactor).
        _ => csv_quote(&serde_json::to_string(v).unwrap_or_default()),
    }
}

fn needs_csv_quote(s: &str) -> bool {
    // Quote the empty string so it renders `""` — distinct from a bare empty
    // cell (null). Quote a literal `\N` so it can't be read back as the
    // missing-key sentinel. Both are required for lossless round-tripping.
    s.is_empty()
        || s == MISSING_SENTINEL
        || s.contains(',')
        || s.contains('"')
        || s.contains('\n')
        || s.contains('\r')
}

fn csv_quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        if c == '"' {
            out.push('"');
            out.push('"');
        } else {
            out.push(c);
        }
    }
    out.push('"');
    out
}

// ─────────────────────────── Markdown-KV formatter ───────────────────────────

/// Renders a `Compaction` as a `[N]{cols}` declaration followed by one
/// Markdown list item per row, each cell on its own `key: value` line.
///
/// Token-heavier than [`CsvSchemaFormatter`] (field names repeat per
/// row), but format-comprehension benchmarks show models retrieve
/// values from Markdown-KV substantially more reliably than from CSV.
/// Offered as an opt-in trade of tokens for read accuracy.
///
/// Rendering rules:
/// - Missing cells are omitted entirely (no `key:` line) — sparse rows
///   cost nothing, unlike CSV's positional empty cells.
/// - Strings that would be ambiguous on a line (contain newlines,
///   leading/trailing whitespace, or are empty) render JSON-quoted;
///   everything else renders raw.
/// - Nested cells render as compact inline JSON, matching
///   [`CsvSchemaFormatter`].
/// - Opaque cells keep the fixed `<<ccr:HASH,KIND,SIZE>>` marker
///   contract shared by all formatters.
#[derive(Debug, Clone, Default)]
pub struct MarkdownKvFormatter {
    /// If true, emit a `__dropped:N` note on the declaration line when
    /// rows were dropped under budget. Mirrors
    /// [`CsvSchemaFormatter::include_drop_summary`].
    pub include_drop_summary: bool,
}

impl MarkdownKvFormatter {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_drop_summary(mut self) -> Self {
        self.include_drop_summary = true;
        self
    }
}

impl Formatter for MarkdownKvFormatter {
    fn name(&self) -> &str {
        "markdown-kv"
    }

    fn format(&self, c: &Compaction) -> String {
        let mut out = String::new();
        write_compaction_kv(&mut out, c, self);
        out
    }
}

fn write_compaction_kv(out: &mut String, c: &Compaction, fmt: &MarkdownKvFormatter) {
    match c {
        Compaction::Table {
            schema,
            rows,
            original_count,
        } => {
            write_kv_table(out, schema, rows, *original_count, fmt);
        }
        Compaction::Buckets {
            discriminator,
            buckets,
            original_count,
        } => {
            out.push_str("__buckets:");
            out.push_str(discriminator);
            if fmt.include_drop_summary {
                let kept: usize = buckets.iter().map(|b| b.rows.len()).sum();
                if kept < *original_count {
                    out.push_str(&format!(" __dropped:{}", original_count - kept));
                }
            }
            out.push('\n');
            for b in buckets {
                out.push_str(&format!("__key:{}\n", kv_scalar(&b.key)));
                write_kv_table(out, &b.schema, &b.rows, b.rows.len(), fmt);
            }
        }
        Compaction::OpaqueRef {
            ccr_hash,
            byte_size,
            kind,
        } => {
            out.push_str(&format_ccr_marker(ccr_hash, *byte_size, kind));
        }
        Compaction::Untouched(v) => {
            out.push_str(&serde_json::to_string(v).unwrap_or_default());
        }
    }
}

fn write_kv_table(
    out: &mut String,
    schema: &Schema,
    rows: &[Row],
    original_count: usize,
    fmt: &MarkdownKvFormatter,
) {
    // Same declaration line as the CSV formatter: keeps row count and
    // typed shape up front where the model (and telemetry) expect it.
    // Unlike CSV (pre-existing exposure, kept byte-identical), KV quotes
    // pathological field names here so the declaration parses the same
    // way as the row lines below.
    out.push('[');
    out.push_str(&rows.len().to_string());
    out.push_str("]{");
    let col_decl: Vec<String> = schema
        .fields
        .iter()
        .map(|f| {
            let name = kv_field_name(&f.name);
            if f.nullable {
                format!("{}:{}?", name, f.type_tag)
            } else {
                format!("{}:{}", name, f.type_tag)
            }
        })
        .collect();
    out.push_str(&col_decl.join(","));
    out.push('}');
    if fmt.include_drop_summary && rows.len() < original_count {
        out.push_str(&format!(" __dropped:{}", original_count - rows.len()));
    }
    out.push('\n');

    for row in rows {
        // Compactor invariant: one cell per schema field. zip() would
        // silently drop extras — fail loudly in debug builds instead.
        debug_assert_eq!(row.0.len(), schema.fields.len());
        let mut wrote_first = false;
        for (field, cell) in schema.fields.iter().zip(row.0.iter()) {
            let rendered = match cell {
                CellValue::Missing => continue,
                CellValue::Scalar(v) => kv_scalar(v),
                CellValue::Nested(sub) => JsonFormatter::new().format(sub),
                CellValue::OpaqueRef {
                    ccr_hash,
                    byte_size,
                    kind,
                } => format_ccr_marker(ccr_hash, *byte_size, kind),
            };
            out.push_str(if wrote_first { "  " } else { "- " });
            out.push_str(&kv_field_name(&field.name));
            out.push_str(": ");
            out.push_str(&rendered);
            out.push('\n');
            wrote_first = true;
        }
        // All-missing row: keep a bare list item so the rendered row
        // count still matches the declaration.
        if !wrote_first {
            out.push_str("-\n");
        }
    }
}

fn kv_scalar(v: &Value) -> String {
    match v {
        Value::Null => "null".to_string(),
        Value::Bool(b) => if *b { "true" } else { "false" }.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => {
            if needs_kv_quote(s) {
                serde_json::to_string(s).unwrap_or_default()
            } else {
                s.clone()
            }
        }
        // Object/array fall back to compact JSON (rare — usually
        // already promoted to Nested by the compactor).
        _ => serde_json::to_string(v).unwrap_or_default(),
    }
}

fn needs_kv_quote(s: &str) -> bool {
    s.is_empty()
        || s.contains('\n')
        || s.contains('\r')
        || s.starts_with(char::is_whitespace)
        || s.ends_with(char::is_whitespace)
}

/// Field names are normally bare identifiers, but nothing upstream
/// enforces that. Quote the pathological ones the same way as values:
/// an embedded newline would inject fake row lines, and `": "` inside
/// a key would split the line at the wrong colon on read-back.
fn kv_field_name(name: &str) -> String {
    if needs_kv_quote(name) || name.contains(": ") {
        serde_json::to_string(name).unwrap_or_default()
    } else {
        name.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transforms::smart_crusher::compaction::compactor::compact;
    use crate::transforms::smart_crusher::compaction::compactor::CompactConfig;
    use serde_json::json;

    fn cfg() -> CompactConfig {
        CompactConfig::default()
    }

    // ── JsonFormatter ──

    #[test]
    fn json_formatter_renders_table() {
        let items = vec![
            json!({"id": 1, "name": "alice"}),
            json!({"id": 2, "name": "bob"}),
        ];
        let c = compact(&items, &cfg());
        let out = JsonFormatter::new().format(&c);
        assert!(out.contains("\"_compaction\":\"table\""), "got: {out}");
        assert!(out.contains("\"_kept\":2"));
        assert!(out.contains("alice"));
    }

    #[test]
    fn json_formatter_renders_untouched_verbatim() {
        let c = Compaction::Untouched(json!({"a": 1, "b": [2, 3]}));
        let out = JsonFormatter::new().format(&c);
        assert_eq!(out, r#"{"a":1,"b":[2,3]}"#);
    }

    #[test]
    fn json_formatter_renders_opaque_ref_marker() {
        let mut row = Row::new(vec![CellValue::OpaqueRef {
            ccr_hash: "abc123def456".into(),
            byte_size: 2048,
            kind: OpaqueKind::Base64Blob,
        }]);
        let c = Compaction::Table {
            schema: Schema {
                fields: vec![super::super::ir::FieldSpec {
                    name: "blob".into(),
                    type_tag: "ccr".into(),
                    nullable: false,
                }],
            },
            rows: vec![std::mem::replace(&mut row, Row::new(vec![]))],
            original_count: 1,
        };
        let out = JsonFormatter::new().format(&c);
        assert!(out.contains("\"_ccr\":\"abc123def456\""));
        assert!(out.contains("base64"));
    }

    // ── CsvSchemaFormatter ──

    #[test]
    fn csv_formatter_pure_tabular() {
        let items = vec![
            json!({"id": 1, "name": "alice", "status": "ok"}),
            json!({"id": 2, "name": "bob", "status": "ok"}),
            json!({"id": 3, "name": "carol", "status": "fail"}),
        ];
        let c = compact(&items, &cfg());
        let out = CsvSchemaFormatter::new().format(&c);
        let lines: Vec<&str> = out.trim_end().lines().collect();
        // First line: declaration with [3]{...}
        assert!(lines[0].starts_with("[3]{"), "got line[0]: {}", lines[0]);
        assert!(lines[0].contains("id:int"));
        assert!(lines[0].contains("name:string"));
        assert!(lines[0].contains("status:string"));
        assert_eq!(lines.len(), 4);
    }

    #[test]
    fn csv_formatter_quotes_strings_with_commas() {
        let items = vec![
            json!({"id": 1, "name": "alice, the great"}),
            json!({"id": 2, "name": "bob"}),
        ];
        let c = compact(&items, &cfg());
        let out = CsvSchemaFormatter::new().format(&c);
        assert!(out.contains(r#""alice, the great""#));
    }

    #[test]
    fn csv_formatter_escapes_internal_quotes() {
        let items = vec![
            json!({"id": 1, "msg": "she said \"hi\""}),
            json!({"id": 2, "msg": "ok"}),
        ];
        let c = compact(&items, &cfg());
        let out = CsvSchemaFormatter::new().format(&c);
        assert!(out.contains(r#""she said ""hi""""#));
    }

    #[test]
    fn csv_formatter_renders_buckets() {
        let items = vec![
            json!({"type": "user", "id": 1, "name": "alice"}),
            json!({"type": "user", "id": 2, "name": "bob"}),
            json!({"type": "order", "id": 99, "total": 50}),
            json!({"type": "order", "id": 100, "total": 75}),
        ];
        let c = compact(&items, &cfg());
        let out = CsvSchemaFormatter::new().format(&c);
        assert!(out.starts_with("__buckets:type"));
        assert!(out.contains("__key:order"));
        assert!(out.contains("__key:user"));
    }

    #[test]
    fn csv_formatter_emits_ccr_marker() {
        let big = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=".repeat(8);
        let items = vec![
            json!({"id": 1, "blob": big.clone()}),
            json!({"id": 2, "blob": big.clone()}),
        ];
        let c = compact(&items, &cfg());
        let out = CsvSchemaFormatter::new().format(&c);
        assert!(out.contains("<<ccr:"), "got: {out}");
        assert!(out.contains(",base64,"));
    }

    #[test]
    fn csv_formatter_nested_cell_inline_json() {
        let items = vec![
            json!({"event": "batch", "payload": r#"[{"x":1},{"x":2},{"x":3}]"#}),
            json!({"event": "batch", "payload": r#"[{"x":4},{"x":5}]"#}),
        ];
        let c = compact(&items, &cfg());
        let out = CsvSchemaFormatter::new().format(&c);
        // Nested compaction is JSON-rendered then CSV-quoted, so a
        // `_compaction":"table"` substring should appear inside quotes.
        assert!(out.contains("_compaction"), "got: {out}");
    }

    #[test]
    fn csv_formatter_drop_summary_opt_in() {
        let mut rows = vec![Row::new(vec![CellValue::Scalar(json!(1))])];
        rows.push(Row::new(vec![CellValue::Scalar(json!(2))]));
        let c = Compaction::Table {
            schema: Schema {
                fields: vec![super::super::ir::FieldSpec {
                    name: "x".into(),
                    type_tag: "int".into(),
                    nullable: false,
                }],
            },
            rows,
            original_count: 5, // 3 dropped
        };
        let with_summary = CsvSchemaFormatter::new().with_drop_summary().format(&c);
        assert!(with_summary.contains("__dropped:3"));
        let without = CsvSchemaFormatter::new().format(&c);
        assert!(!without.contains("__dropped"));
    }

    #[test]
    fn estimate_matches_format_len() {
        let items = vec![json!({"a": 1}), json!({"a": 2})];
        let c = compact(&items, &cfg());
        let f = CsvSchemaFormatter::new();
        assert_eq!(f.estimate_bytes(&c), f.format(&c).len());
    }

    // ── Cross-formatter property: same input → smaller CSV than JSON ──
    // This is the headline value prop. If it doesn't hold for "obviously
    // tabular" input, the formatter is broken or the fixture is wrong.

    #[test]
    fn csv_smaller_than_json_for_tabular() {
        let items: Vec<Value> = (0..50)
            .map(|i| {
                json!({
                    "id": i,
                    "name": format!("user_{i}"),
                    "email": format!("user_{i}@example.com"),
                    "status": if i % 3 == 0 { "ok" } else { "pending" },
                })
            })
            .collect();
        let c = compact(&items, &cfg());
        let json_out = JsonFormatter::new().format(&c);
        let csv_out = CsvSchemaFormatter::new().format(&c);
        // CSV should beat the structured-JSON formatter (both
        // deduplicate the schema, so the win comes from removing
        // structural punctuation only — modest, but real).
        assert!(
            csv_out.len() < json_out.len(),
            "csv {} bytes vs json {} bytes",
            csv_out.len(),
            json_out.len()
        );
    }

    #[test]
    fn csv_substantially_smaller_than_raw_json() {
        // The headline value prop: CSV+schema beats naïve JSON
        // serialization of the same array (where every row repeats
        // every field name) by a wide margin.
        let items: Vec<Value> = (0..50)
            .map(|i| {
                json!({
                    "id": i,
                    "name": format!("user_{i}"),
                    "email": format!("user_{i}@example.com"),
                    "status": if i % 3 == 0 { "ok" } else { "pending" },
                })
            })
            .collect();
        let c = compact(&items, &cfg());
        let csv_out = CsvSchemaFormatter::new().format(&c);
        let raw_json = serde_json::to_string(&Value::Array(items.clone())).unwrap();
        assert!(
            csv_out.len() * 10 < raw_json.len() * 7,
            "csv {} bytes vs raw json {} bytes — expected >30% reduction",
            csv_out.len(),
            raw_json.len()
        );
    }

    // ── MarkdownKvFormatter ──

    #[test]
    fn markdown_kv_renders_table() {
        let items = vec![
            json!({"id": 1, "name": "alice"}),
            json!({"id": 2, "name": "bob"}),
        ];
        let c = compact(&items, &cfg());
        let out = MarkdownKvFormatter::new().format(&c);
        let lines: Vec<&str> = out.trim_end().lines().collect();
        assert!(lines[0].starts_with("[2]{"), "got line[0]: {}", lines[0]);
        assert!(lines[0].contains("id:int"));
        assert!(out.contains("- id: 1\n  name: alice\n"), "got: {out}");
        assert!(out.contains("- id: 2\n  name: bob\n"), "got: {out}");
    }

    #[test]
    fn markdown_kv_omits_missing_cells() {
        let items = vec![json!({"id": 1, "note": "has note"}), json!({"id": 2})];
        let c = compact(&items, &cfg());
        let out = MarkdownKvFormatter::new().format(&c);
        assert!(out.contains("note: has note"), "got: {out}");
        // Row 2 has no `note:` line at all.
        let row2 = out.split("- id: 2").nth(1).expect("row 2 present");
        assert!(!row2.contains("note:"), "got row2 tail: {row2}");
    }

    #[test]
    fn markdown_kv_quotes_ambiguous_strings() {
        let items = vec![
            json!({"id": 1, "msg": "line one\nline two"}),
            json!({"id": 2, "msg": "plain"}),
        ];
        let c = compact(&items, &cfg());
        let out = MarkdownKvFormatter::new().format(&c);
        assert!(
            out.contains(r#"msg: "line one\nline two""#),
            "multiline must be JSON-quoted, got: {out}"
        );
        assert!(out.contains("msg: plain\n"), "got: {out}");
    }

    #[test]
    fn markdown_kv_quotes_pathological_field_names() {
        // A newline in a key would inject fake row lines; ": " in a key
        // would split read-back at the wrong colon. Both get JSON-quoted
        // in the declaration and in every row line.
        let items = vec![
            json!({"bad\nkey": 1, "note: extra": "x"}),
            json!({"bad\nkey": 2, "note: extra": "y"}),
        ];
        let c = compact(&items, &cfg());
        let out = MarkdownKvFormatter::new().format(&c);
        assert!(!out.contains("bad\nkey"), "raw newline key leaked: {out}");
        assert!(out.contains(r#""bad\nkey""#), "got: {out}");
        assert!(out.contains(r#""note: extra": x"#), "got: {out}");
        let decl = out.lines().next().unwrap();
        assert!(decl.contains(r#""bad\nkey":int"#), "got decl: {decl}");
    }

    #[test]
    fn markdown_kv_plain_strings_unquoted() {
        let items = vec![
            json!({"id": 1, "name": "alice, the \"great\""}),
            json!({"id": 2, "name": "bob"}),
        ];
        let c = compact(&items, &cfg());
        let out = MarkdownKvFormatter::new().format(&c);
        // Commas and quotes are fine on a KV line — no CSV-style quoting.
        assert!(out.contains(r#"name: alice, the "great""#), "got: {out}");
    }

    #[test]
    fn markdown_kv_emits_ccr_marker() {
        let big = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=".repeat(8);
        let items = vec![
            json!({"id": 1, "blob": big.clone()}),
            json!({"id": 2, "blob": big.clone()}),
        ];
        let c = compact(&items, &cfg());
        let out = MarkdownKvFormatter::new().format(&c);
        assert!(out.contains("<<ccr:"), "got: {out}");
        assert!(out.contains(",base64,"));
    }

    #[test]
    fn markdown_kv_renders_buckets() {
        let items = vec![
            json!({"type": "user", "id": 1, "name": "alice"}),
            json!({"type": "user", "id": 2, "name": "bob"}),
            json!({"type": "order", "id": 99, "total": 50}),
            json!({"type": "order", "id": 100, "total": 75}),
        ];
        let c = compact(&items, &cfg());
        let out = MarkdownKvFormatter::new().format(&c);
        assert!(out.starts_with("__buckets:type"), "got: {out}");
        assert!(out.contains("__key:order"));
        assert!(out.contains("__key:user"));
        assert!(out.contains("- id:"), "got: {out}");
    }

    #[test]
    fn markdown_kv_drop_summary_opt_in() {
        let rows = vec![
            Row::new(vec![CellValue::Scalar(json!(1))]),
            Row::new(vec![CellValue::Scalar(json!(2))]),
        ];
        let c = Compaction::Table {
            schema: Schema {
                fields: vec![super::super::ir::FieldSpec {
                    name: "x".into(),
                    type_tag: "int".into(),
                    nullable: false,
                }],
            },
            rows,
            original_count: 5, // 3 dropped
        };
        let with_summary = MarkdownKvFormatter::new().with_drop_summary().format(&c);
        assert!(with_summary.contains("__dropped:3"));
        let without = MarkdownKvFormatter::new().format(&c);
        assert!(!without.contains("__dropped"));
    }

    #[test]
    fn markdown_kv_nested_cell_inline_json() {
        let items = vec![
            json!({"event": "batch", "payload": r#"[{"x":1},{"x":2},{"x":3}]"#}),
            json!({"event": "batch", "payload": r#"[{"x":4},{"x":5}]"#}),
        ];
        let c = compact(&items, &cfg());
        let out = MarkdownKvFormatter::new().format(&c);
        assert!(out.contains("_compaction"), "got: {out}");
    }

    #[test]
    fn markdown_kv_estimate_matches_format_len() {
        let items = vec![json!({"a": 1}), json!({"a": 2})];
        let c = compact(&items, &cfg());
        let f = MarkdownKvFormatter::new();
        assert_eq!(f.estimate_bytes(&c), f.format(&c).len());
    }

    #[test]
    fn markdown_kv_smaller_than_raw_json() {
        // KV repeats field names per row, so it loses to CSV on bytes —
        // but it should still beat naïve JSON (quotes + braces + commas).
        let items: Vec<Value> = (0..50)
            .map(|i| {
                json!({
                    "id": i,
                    "name": format!("user_{i}"),
                    "email": format!("user_{i}@example.com"),
                    "status": if i % 3 == 0 { "ok" } else { "pending" },
                })
            })
            .collect();
        let c = compact(&items, &cfg());
        let kv_out = MarkdownKvFormatter::new().format(&c);
        let raw_json = serde_json::to_string(&Value::Array(items.clone())).unwrap();
        assert!(
            kv_out.len() < raw_json.len(),
            "kv {} bytes vs raw json {} bytes",
            kv_out.len(),
            raw_json.len()
        );
    }
}
