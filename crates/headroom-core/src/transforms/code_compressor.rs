//! CodeCompressor — Rust port of `headroom.transforms.code_compressor`.
//!
//! AST-preserving compression for source code. Unlike the ML Kompress
//! engine or the deterministic structural compressors (Log/Search/Diff),
//! this parses code into a tree-sitter AST and selectively compresses
//! function bodies while preserving imports, signatures, type definitions,
//! decorators, and top-level code. The output is guaranteed to be
//! syntactically valid (re-parsed; on ERROR/MISSING it returns the
//! original).
//!
//! # Grammar-version parity (the make-or-break invariant)
//!
//! The Python reference parses with `tree_sitter_language_pack`'s bundled
//! grammars; this port uses the per-language `tree-sitter-<lang>` crates.
//! Byte-parity requires node-for-node identical ASTs (same node `kind`
//! strings, same tree shape, same `start_point`/`end_point` rows). The
//! Cargo pins (`tree-sitter-python = "=0.25.0"`, …) match the exact PyPI
//! wheel versions the fixtures were recorded against; a canary over 9
//! samples × 8 languages confirmed 100% identical node-type + line-span
//! trees. See `crates/headroom-core/Cargo.toml` for the full pin table.
//!
//! # Parity scope
//!
//! All slicing that the Python reference does by *byte offset into a `str`*
//! (`code[node.start_byte:node.end_byte]`) is reproduced here as correct
//! UTF-8 byte slicing (`&code[start..end]`). For ASCII inputs these are
//! identical; for non-ASCII identifiers/strings the Python path is latently
//! buggy (slices a `str` by byte index) and the two diverge. Parity
//! fixtures are therefore ASCII; non-ASCII is out of parity scope, exactly
//! as the line-based body slicing (which both sides do by row, and which is
//! always correct) sidesteps the issue for the main compression path.
//!
//! # CCR
//!
//! Like the other engines, the Rust port returns the compressed string
//! only. The Python inline `# [N tokens compressed... hash=]` marker is
//! intentionally not reproduced; live-zone CCR uses the `<<ccr:>>`
//! convention via the dispatcher. Parity fixtures are recorded with
//! `enable_ccr=False` so the output is deterministic and store-independent.

use std::collections::{BTreeSet, HashMap};

use tree_sitter::{Language, Node, Parser, Tree};

// ─── Enums ──────────────────────────────────────────────────────────────

/// Supported programming languages. `value()` matches the Python
/// `CodeLanguage` enum `.value` strings (used in the serialized result).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CodeLanguage {
    Python,
    Javascript,
    Typescript,
    Go,
    Rust,
    Java,
    C,
    Cpp,
    Unknown,
}

impl CodeLanguage {
    pub fn value(self) -> &'static str {
        match self {
            CodeLanguage::Python => "python",
            CodeLanguage::Javascript => "javascript",
            CodeLanguage::Typescript => "typescript",
            CodeLanguage::Go => "go",
            CodeLanguage::Rust => "rust",
            CodeLanguage::Java => "java",
            CodeLanguage::C => "c",
            CodeLanguage::Cpp => "cpp",
            CodeLanguage::Unknown => "unknown",
        }
    }

    /// Parse from a lowercase language name. `None` for unrecognized
    /// (Python raises `ValueError`; callers that need that semantics check
    /// for `None`).
    pub fn from_name(s: &str) -> Option<Self> {
        Some(match s {
            "python" => CodeLanguage::Python,
            "javascript" => CodeLanguage::Javascript,
            "typescript" => CodeLanguage::Typescript,
            "go" => CodeLanguage::Go,
            "rust" => CodeLanguage::Rust,
            "java" => CodeLanguage::Java,
            "c" => CodeLanguage::C,
            "cpp" => CodeLanguage::Cpp,
            "unknown" => CodeLanguage::Unknown,
            _ => return None,
        })
    }

    /// The tree-sitter grammar for this language, or `None` for `Unknown`.
    fn grammar(self) -> Option<Language> {
        Some(match self {
            CodeLanguage::Python => tree_sitter_python::LANGUAGE.into(),
            CodeLanguage::Javascript => tree_sitter_javascript::LANGUAGE.into(),
            CodeLanguage::Typescript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            CodeLanguage::Go => tree_sitter_go::LANGUAGE.into(),
            CodeLanguage::Rust => tree_sitter_rust::LANGUAGE.into(),
            CodeLanguage::Java => tree_sitter_java::LANGUAGE.into(),
            CodeLanguage::C => tree_sitter_c::LANGUAGE.into(),
            CodeLanguage::Cpp => tree_sitter_cpp::LANGUAGE.into(),
            CodeLanguage::Unknown => return None,
        })
    }
}

/// How to handle Python docstrings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocstringMode {
    Full,
    FirstLine,
    Remove,
    /// Alias for `Remove` (deprecated).
    None,
}

impl DocstringMode {
    pub fn value(self) -> &'static str {
        match self {
            DocstringMode::Full => "full",
            DocstringMode::FirstLine => "first_line",
            DocstringMode::Remove => "remove",
            DocstringMode::None => "none",
        }
    }

    pub fn from_value(s: &str) -> Option<Self> {
        Some(match s {
            "full" => DocstringMode::Full,
            "first_line" => DocstringMode::FirstLine,
            "remove" => DocstringMode::Remove,
            "none" => DocstringMode::None,
            _ => return None,
        })
    }
}

// ─── Language config ────────────────────────────────────────────────────

/// Data-driven AST node-type tables + syntactic conventions for one
/// language. Mirrors the Python `LangConfig` frozen dataclass.
struct LangConfig {
    import_nodes: &'static [&'static str],
    function_nodes: &'static [&'static str],
    class_nodes: &'static [&'static str],
    type_nodes: &'static [&'static str],
    body_node_types: &'static [&'static str],
    decorator_node: Option<&'static str>,
    comment_prefix: &'static str,
    uses_colon_after_signature: bool,
    package_node: Option<&'static str>,
}

impl LangConfig {
    fn is_import(&self, k: &str) -> bool {
        self.import_nodes.contains(&k)
    }
    fn is_function(&self, k: &str) -> bool {
        self.function_nodes.contains(&k)
    }
    fn is_class(&self, k: &str) -> bool {
        self.class_nodes.contains(&k)
    }
    fn is_type(&self, k: &str) -> bool {
        self.type_nodes.contains(&k)
    }
    fn is_body(&self, k: &str) -> bool {
        self.body_node_types.contains(&k)
    }
}

/// Returns the `LangConfig` for a language, or `None` for `Unknown`
/// (mirrors `_LANG_CONFIGS.get(language)`).
fn lang_config(language: CodeLanguage) -> Option<LangConfig> {
    Some(match language {
        CodeLanguage::Python => LangConfig {
            import_nodes: &["import_statement", "import_from_statement"],
            function_nodes: &["function_definition"],
            class_nodes: &["class_definition"],
            type_nodes: &["type_alias_statement"],
            body_node_types: &["block"],
            decorator_node: Some("decorated_definition"),
            comment_prefix: "#",
            uses_colon_after_signature: true,
            package_node: None,
        },
        CodeLanguage::Javascript => LangConfig {
            import_nodes: &["import_statement", "import_declaration"],
            function_nodes: &["function_declaration", "method_definition"],
            class_nodes: &["class_declaration"],
            type_nodes: &[],
            body_node_types: &["statement_block"],
            decorator_node: None,
            comment_prefix: "//",
            uses_colon_after_signature: false,
            package_node: None,
        },
        CodeLanguage::Typescript => LangConfig {
            import_nodes: &["import_statement", "import_declaration"],
            function_nodes: &["function_declaration", "method_definition"],
            class_nodes: &["class_declaration"],
            type_nodes: &["interface_declaration", "type_alias_declaration"],
            body_node_types: &["statement_block"],
            decorator_node: None,
            comment_prefix: "//",
            uses_colon_after_signature: false,
            package_node: None,
        },
        CodeLanguage::Go => LangConfig {
            import_nodes: &["import_declaration"],
            function_nodes: &["function_declaration", "method_declaration"],
            class_nodes: &[],
            type_nodes: &["type_declaration"],
            body_node_types: &["block"],
            decorator_node: None,
            comment_prefix: "//",
            uses_colon_after_signature: false,
            package_node: Some("package_clause"),
        },
        CodeLanguage::Rust => LangConfig {
            import_nodes: &["use_declaration"],
            function_nodes: &["function_item"],
            class_nodes: &["impl_item"],
            type_nodes: &["struct_item", "enum_item", "type_item", "trait_item"],
            body_node_types: &["block"],
            decorator_node: None,
            comment_prefix: "//",
            uses_colon_after_signature: false,
            package_node: None,
        },
        CodeLanguage::Java => LangConfig {
            import_nodes: &["import_declaration"],
            function_nodes: &["method_declaration", "constructor_declaration"],
            class_nodes: &["class_declaration", "interface_declaration"],
            type_nodes: &["enum_declaration"],
            body_node_types: &["block"],
            decorator_node: None,
            comment_prefix: "//",
            uses_colon_after_signature: false,
            package_node: Some("package_declaration"),
        },
        CodeLanguage::C => LangConfig {
            import_nodes: &["preproc_include"],
            function_nodes: &["function_definition"],
            class_nodes: &[],
            type_nodes: &["struct_specifier", "enum_specifier", "type_definition"],
            body_node_types: &["compound_statement"],
            decorator_node: None,
            comment_prefix: "//",
            uses_colon_after_signature: false,
            package_node: None,
        },
        CodeLanguage::Cpp => LangConfig {
            import_nodes: &["preproc_include"],
            function_nodes: &["function_definition"],
            class_nodes: &["class_specifier"],
            type_nodes: &["struct_specifier", "enum_specifier", "type_definition"],
            body_node_types: &["compound_statement"],
            decorator_node: None,
            comment_prefix: "//",
            uses_colon_after_signature: false,
            package_node: None,
        },
        CodeLanguage::Unknown => return None,
    })
}

// ─── Config ─────────────────────────────────────────────────────────────

/// Configuration for code-aware compression. Field defaults match the
/// Python `CodeCompressorConfig` dataclass. The `preserve_*` and
/// `compress_comments` flags are vestigial in the reference (the extractor
/// always preserves structure regardless), kept here for fidelity.
#[derive(Debug, Clone)]
pub struct CodeCompressorConfig {
    pub preserve_imports: bool,
    pub preserve_signatures: bool,
    pub preserve_type_annotations: bool,
    pub preserve_decorators: bool,
    pub docstring_mode: DocstringMode,
    pub target_compression_rate: f64,
    pub max_body_lines: i64,
    pub compress_comments: bool,
    pub min_tokens_for_compression: i64,
    pub language_hint: Option<String>,
    pub fallback_to_kompress: bool,
    pub semantic_analysis: bool,
    pub enable_ccr: bool,
    pub ccr_ttl: i64,
}

impl Default for CodeCompressorConfig {
    fn default() -> Self {
        Self {
            preserve_imports: true,
            preserve_signatures: true,
            preserve_type_annotations: true,
            preserve_decorators: true,
            docstring_mode: DocstringMode::FirstLine,
            target_compression_rate: 0.2,
            max_body_lines: 5,
            compress_comments: true,
            min_tokens_for_compression: 100,
            language_hint: None,
            fallback_to_kompress: true,
            semantic_analysis: true,
            enable_ccr: true,
            ccr_ttl: 300,
        }
    }
}

// ─── Result ─────────────────────────────────────────────────────────────

/// Result of code-aware compression. Field set + serialization mirror the
/// Python `CodeCompressionResult` dataclass (via `asdict`); the `@property`
/// derivatives (`tokens_saved`, `savings_percentage`, `summary`) are not
/// serialized and live as methods here.
#[derive(Debug, Clone, PartialEq)]
pub struct CodeCompressionResult {
    pub compressed: String,
    pub original: String,
    pub original_tokens: i64,
    pub compressed_tokens: i64,
    pub compression_ratio: f64,
    pub language: CodeLanguage,
    pub language_confidence: f64,
    pub preserved_imports: i64,
    pub preserved_signatures: i64,
    pub compressed_bodies: i64,
    pub syntax_valid: bool,
    pub cache_key: Option<String>,
    /// Short-name → score (round-3), insertion order is irrelevant to the
    /// serialized object (compared as a map).
    pub symbol_scores: Vec<(String, f64)>,
}

impl CodeCompressionResult {
    pub fn tokens_saved(&self) -> i64 {
        (self.original_tokens - self.compressed_tokens).max(0)
    }

    pub fn savings_percentage(&self) -> f64 {
        if self.original_tokens == 0 {
            0.0
        } else {
            (self.tokens_saved() as f64 / self.original_tokens as f64) * 100.0
        }
    }
}

// ─── Extracted structure ────────────────────────────────────────────────

/// `(compressed, structure, symbol_scores)` — the result of an AST pass.
type AstCompression = (String, CodeStructure, Vec<(String, f64)>);

#[derive(Default)]
struct CodeStructure {
    imports: Vec<String>,
    type_definitions: Vec<String>,
    class_definitions: Vec<String>,
    function_signatures: Vec<String>,
    /// (signature, body, line) — never populated by the current paths, so
    /// `compressed_bodies` is always 0 (mirrors the reference).
    function_bodies: Vec<(String, String, i64)>,
    top_level_code: Vec<String>,
    other: Vec<String>,
}

// ─── Symbol analysis ────────────────────────────────────────────────────

#[derive(Default)]
struct SymbolAnalysis {
    /// qname → normalized score (round-3), insertion order preserved.
    scores: Vec<(String, f64)>,
    /// qname → set of short names it calls, insertion order preserved
    /// (matters for `make_omitted_comment` first-match selection).
    calls: Vec<(String, BTreeSet<String>)>,
    /// qname → short name.
    bare_names: HashMap<String, String>,
    /// qname → body line count.
    body_line_counts: HashMap<String, i64>,
}

impl SymbolAnalysis {
    fn calls_of(&self, qname: &str) -> Option<&BTreeSet<String>> {
        self.calls.iter().find(|(k, _)| k == qname).map(|(_, v)| v)
    }
}

// ─── Module helpers (stateless) ─────────────────────────────────────────

/// `code[node.start_byte:node.end_byte]` — correct UTF-8 byte slice.
fn node_text<'a>(node: Node, code: &'a str) -> &'a str {
    &code[node.start_byte()..node.end_byte()]
}

/// First child whose kind is a name token; returns its (real) text. Mirrors
/// `_get_definition_name`.
fn get_definition_name(node: Node, code: &str) -> Option<String> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        let k = child.kind();
        if k == "identifier" || k == "name" || k == "type_identifier" || k == "property_identifier"
        {
            return Some(node_text(child, code).to_string());
        }
    }
    None
}

fn is_public_symbol(name: &str, language: CodeLanguage) -> bool {
    if name.is_empty() {
        return false;
    }
    if language == CodeLanguage::Go {
        return name.chars().next().is_some_and(|c| c.is_uppercase());
    }
    !name.starts_with('_')
}

/// Look up the allocated body-line limit for a function. `max_body_lines`
/// always acts as a hard cap. Mirrors `_get_body_limit`.
fn get_body_limit(func_name: Option<&str>, body_limits: &HashMap<String, i64>, max_body_lines: i64) -> i64 {
    if let Some(name) = func_name {
        if !body_limits.is_empty() {
            if let Some(&v) = body_limits.get(name) {
                return v.min(max_body_lines);
            }
        }
    }
    max_body_lines
}

/// Detect the indentation used in a list of code lines. Mirrors `_detect_indent`.
fn detect_indent(lines: &[&str]) -> String {
    for line in lines {
        if !line.trim().is_empty() {
            return leading_ws(line).to_string();
        }
    }
    "    ".to_string()
}

/// Leading-whitespace prefix of a line (`line[:len-len(lstrip)]`).
fn leading_ws(line: &str) -> &str {
    &line[..line.len() - line.trim_start().len()]
}

/// Build the omitted-body comment with call info. Mirrors `_make_omitted_comment`.
fn make_omitted_comment(
    func_name: Option<&str>,
    omitted_count: i64,
    indent: &str,
    comment_prefix: &str,
    analysis: &SymbolAnalysis,
) -> String {
    let mut calls_info = String::new();
    if let Some(func_name) = func_name {
        // Candidate keys: the bare name first, then every qname ending in
        // `.func_name` (insertion order). Pick the first present in `calls`.
        let suffix = format!(".{func_name}");
        let mut candidates: Vec<&str> = vec![func_name];
        for (k, _) in &analysis.calls {
            if k.ends_with(&suffix) {
                candidates.push(k.as_str());
            }
        }
        for key in candidates {
            if let Some(called) = analysis.calls_of(key) {
                if !called.is_empty() {
                    // BTreeSet iterates sorted == Python sorted(called).
                    let sorted_calls: Vec<&str> = called.iter().take(5).map(|s| s.as_str()).collect();
                    calls_info = format!("; calls: {}", sorted_calls.join(", "));
                    if called.len() > 5 {
                        calls_info.push_str(&format!(" +{} more", called.len() - 5));
                    }
                }
                break;
            }
        }
    }
    format!("{indent}{comment_prefix} [{omitted_count} lines omitted{calls_info}]")
}

/// Count ERROR + MISSING nodes (recursive). Mirrors `_count_error_nodes`.
fn count_error_nodes(node: Node) -> i64 {
    let mut count = 0;
    if node.kind() == "ERROR" || node.is_missing() {
        count += 1;
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        count += count_error_nodes(child);
    }
    count
}

/// True if the tree contains an ERROR or MISSING node. Mirrors `_has_syntax_issues`.
fn has_syntax_issues(node: Node) -> bool {
    if node.kind() == "ERROR" || node.is_missing() {
        return true;
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if has_syntax_issues(child) {
            return true;
        }
    }
    false
}

/// CPython `round(x)` (ndigits=None): nearest int, ties to even.
fn py_round_int(x: f64) -> i64 {
    let r = x.round(); // half away from zero == C round()
    if (x - r).abs() == 0.5 {
        // Halfway case: round to even.
        (2.0 * (x / 2.0).round()) as i64
    } else {
        r as i64
    }
}

/// CPython `round(x, 3)`: correctly-rounded to 3 decimals, ties to even.
/// Rust's `{:.3}` formatter is correctly rounded half-to-even, so format +
/// re-parse yields the same f64 CPython's dtoa-based rounding produces.
fn py_round3(x: f64) -> f64 {
    format!("{x:.3}").parse::<f64>().unwrap()
}

/// `len(text) // 4` over Unicode code points, min 1. Mirrors
/// `_estimate_tokens` with `tokenizer=None`.
fn estimate_tokens(text: &str) -> i64 {
    ((text.chars().count() / 4).max(1)) as i64
}

/// Parse `code` with the grammar for `language`. `None` for `Unknown` or on
/// parse failure.
fn parse_code(code: &str, language: CodeLanguage) -> Option<Tree> {
    let grammar = language.grammar()?;
    let mut parser = Parser::new();
    parser.set_language(&grammar).ok()?;
    parser.parse(code.as_bytes(), None)
}

/// First `n` Unicode code points of `s` (Python `s[:n]` for a `str`).
fn char_prefix(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

// ─── Language detection ─────────────────────────────────────────────────

mod prefilter {
    use std::sync::OnceLock;

    use regex::Regex;

    use super::CodeLanguage;

    /// (language, patterns) in the Python `_LANGUAGE_PREFILTER` insertion
    /// order — that order is the stable tie-break for detection.
    pub fn patterns() -> &'static [(CodeLanguage, Vec<Regex>)] {
        static P: OnceLock<Vec<(CodeLanguage, Vec<Regex>)>> = OnceLock::new();
        P.get_or_init(|| {
            let c = |s: &str| Regex::new(s).unwrap();
            vec![
                (
                    CodeLanguage::Python,
                    vec![
                        c(r"(?m)^\s*(def|class|import|from|async def)\s+\w+"),
                        c(r"(?m)^\s*@\w+"),
                        c(r#"(?m)^\s*""""#),
                        c(r"(?m)^\s*if __name__\s*=="),
                    ],
                ),
                (
                    CodeLanguage::Javascript,
                    vec![
                        c(r"(?m)^\s*(function|const|let|var|class|export)\s+\w+"),
                        c(r"(?m)^\s*async\s+(function|=>)"),
                        c(r"(?m)^\s*module\.exports"),
                        c(r#"(?m)^\s*(import|export)\s+.*\s+from\s+['"]"#),
                    ],
                ),
                (
                    CodeLanguage::Typescript,
                    vec![
                        c(r"(?m)^\s*(interface|type|enum|namespace)\s+\w+"),
                        c(r"(?m):\s*(string|number|boolean|any|void|Promise)\b"),
                    ],
                ),
                (
                    CodeLanguage::Go,
                    vec![
                        c(r"(?m)^\s*(func|type|package|import)\s+"),
                        c(r"(?m)^\s*func\s+\([^)]+\)\s+\w+"),
                        c(r"(?m)\bstruct\s*\{"),
                    ],
                ),
                (
                    CodeLanguage::Rust,
                    vec![
                        c(r"(?m)^\s*(fn|struct|enum|impl|mod|use|pub)\s+"),
                        c(r"(?m)^\s*#\["),
                    ],
                ),
                (
                    CodeLanguage::Java,
                    vec![
                        c(r"(?m)^\s*(public|private|protected)\s+(class|interface|enum)"),
                        c(r"(?m)^\s*package\s+[\w.]+;"),
                    ],
                ),
                (
                    CodeLanguage::C,
                    vec![
                        c(r#"(?m)^\s*#include\s*[<"]"#),
                        c(r"(?m)^\s*(int|void|char|float|double)\s+\w+\s*\("),
                        c(r"(?m)^\s*typedef\s+"),
                    ],
                ),
                (
                    CodeLanguage::Cpp,
                    vec![
                        c(r#"(?m)^\s*#include\s*[<"]"#),
                        c(r"(?m)\bnamespace\s+\w+"),
                        c(r"(?m)::\w+"),
                    ],
                ),
            ]
        })
    }
}

/// Detect the language of `code`. Mirrors `detect_language`: regex prefilter
/// → tree-sitter fewest-errors → regex-only fallback.
pub fn detect_language(code: &str) -> (CodeLanguage, f64) {
    if code.trim().is_empty() {
        return (CodeLanguage::Unknown, 0.0);
    }

    let sample = char_prefix(code, 5000);

    // Phase 1: prefilter scores, in fixed enum order.
    let mut candidates: Vec<(CodeLanguage, i64)> = Vec::new();
    for (lang, pats) in prefilter::patterns() {
        let mut score = 0i64;
        for pat in pats {
            score += pat.find_iter(&sample).count() as i64;
        }
        if score > 0 {
            candidates.push((*lang, score));
        }
    }

    if candidates.is_empty() {
        return (CodeLanguage::Unknown, 0.0);
    }

    // Disambiguation: TS superset of JS; C++ superset of C.
    let get = |cs: &[(CodeLanguage, i64)], l: CodeLanguage| {
        cs.iter().find(|(x, _)| *x == l).map(|(_, s)| *s)
    };
    if let (Some(ts), Some(_js)) = (
        get(&candidates, CodeLanguage::Typescript),
        get(&candidates, CodeLanguage::Javascript),
    ) {
        if ts >= 2 {
            if let Some(e) = candidates.iter_mut().find(|(x, _)| *x == CodeLanguage::Javascript) {
                e.1 = 0;
            }
        }
    }
    if let (Some(cpp), Some(_c)) = (
        get(&candidates, CodeLanguage::Cpp),
        get(&candidates, CodeLanguage::C),
    ) {
        if cpp >= 2 {
            if let Some(e) = candidates.iter_mut().find(|(x, _)| *x == CodeLanguage::C) {
                e.1 = 0;
            }
        }
    }

    // Phase 2: tree-sitter, fewest errors then most top-level children.
    // (tree-sitter is always available in the Rust port.)
    let code_bytes_src = char_prefix(code, 10000);
    let mut best_lang = CodeLanguage::Unknown;
    let mut min_errors = i64::MAX;
    let mut best_node_count: i64 = 0;

    // Sort candidates by score desc, stable (preserves enum order on ties).
    let mut sorted_candidates = candidates.clone();
    sorted_candidates.sort_by_key(|&(_, s)| std::cmp::Reverse(s));

    for (lang, _score) in &sorted_candidates {
        if *lang == CodeLanguage::Unknown || get(&candidates, *lang) == Some(0) {
            continue;
        }
        let Some(tree) = parse_code(&code_bytes_src, *lang) else {
            continue;
        };
        let root = tree.root_node();
        let error_count = count_error_nodes(root);
        let node_count = root.child_count() as i64;
        if error_count < min_errors || (error_count == min_errors && node_count > best_node_count) {
            min_errors = error_count;
            best_lang = *lang;
            best_node_count = node_count;
        }
    }

    if best_lang != CodeLanguage::Unknown {
        let total_lines = (code.trim().split('\n').count() as i64).max(1);
        let error_ratio = min_errors as f64 / total_lines as f64;
        // Python: max(0.3, min(1.0, 1.0 - error_ratio)) == clamp(0.3, 1.0).
        let confidence = (1.0 - error_ratio).clamp(0.3, 1.0);
        return (best_lang, confidence);
    }

    // Phase 3: regex-only fallback (first max in insertion order).
    let mut best = candidates[0];
    for &cand in &candidates[1..] {
        if cand.1 > best.1 {
            best = cand;
        }
    }
    if best.1 == 0 {
        return (CodeLanguage::Unknown, 0.0);
    }
    let confidence = (0.3 + best.1 as f64 * 0.1).min(1.0);
    (best.0, confidence)
}

// ─── Compressor ─────────────────────────────────────────────────────────

/// AST-preserving code compressor. Construct with a config and call
/// [`CodeAwareCompressor::compress`].
#[derive(Debug, Clone)]
pub struct CodeAwareCompressor {
    pub config: CodeCompressorConfig,
}

/// Shared per-compression context (source + parsed tables), to keep the
/// traversal methods' signatures small.
struct Ctx<'a> {
    code: &'a str,
    code_lines: Vec<&'a str>,
    language: CodeLanguage,
    lang: &'a LangConfig,
    body_limits: &'a HashMap<String, i64>,
    analysis: &'a SymbolAnalysis,
    config: &'a CodeCompressorConfig,
}

impl CodeAwareCompressor {
    pub fn new(config: CodeCompressorConfig) -> Self {
        Self { config }
    }

    /// Compress with all defaults (language auto-detect, no context). This
    /// is the path the parity recorder exercises (`compress(code)`).
    pub fn compress(&self, code: &str) -> CodeCompressionResult {
        self.compress_with(code, None, "")
    }

    /// Full compression entry point. `language` overrides detection;
    /// `context` boosts symbol importance for matching names. Mirrors
    /// `CodeAwareCompressor.compress` (with `tokenizer=None`).
    pub fn compress_with(
        &self,
        code: &str,
        language: Option<&str>,
        context: &str,
    ) -> CodeCompressionResult {
        if code.trim().is_empty() {
            return passthrough_result(code, 0, CodeLanguage::Unknown, 0.0);
        }

        let original_tokens = estimate_tokens(code);

        if original_tokens < self.config.min_tokens_for_compression {
            return passthrough_result(code, original_tokens, CodeLanguage::Unknown, 0.0);
        }

        // Detect or use specified language.
        let (detected_lang, confidence) = if let Some(lang) = language {
            match CodeLanguage::from_name(&lang.to_lowercase()) {
                Some(l) => (l, 1.0),
                None => (CodeLanguage::Unknown, 1.0),
            }
        } else if let Some(hint) = &self.config.language_hint {
            match CodeLanguage::from_name(&hint.to_lowercase()) {
                Some(l) => (l, 1.0),
                None => (CodeLanguage::Unknown, 1.0),
            }
        } else {
            detect_language(code)
        };

        // Unknown language → fallback (Kompress) or passthrough.
        if detected_lang == CodeLanguage::Unknown {
            // Kompress fallback is delegated to the live-zone dispatcher in
            // the Rust port (the engine doesn't own the model); fixtures are
            // recorded with fallback_to_kompress=False, so this is the
            // passthrough branch.
            return CodeCompressionResult {
                language: CodeLanguage::Unknown,
                language_confidence: 0.0,
                ..passthrough_result(code, original_tokens, CodeLanguage::Unknown, 0.0)
            };
        }

        // Parse + compress (tree-sitter always available here).
        let Some((compressed, structure, symbol_scores)) =
            self.compress_with_ast(code, detected_lang, context)
        else {
            // AST exception path → fallback/passthrough.
            return passthrough_result(code, original_tokens, detected_lang, confidence);
        };

        let compressed_tokens = estimate_tokens(&compressed);

        // Verify syntax validity (ERROR + MISSING).
        let syntax_valid = self.verify_syntax(&compressed, detected_lang);
        if !syntax_valid {
            return passthrough_result(code, original_tokens, detected_lang, confidence);
        }

        let ratio = compressed_tokens as f64 / original_tokens.max(1) as f64;

        // Guard against over-aggressive compression (data loss).
        if ratio < 0.05 {
            return passthrough_result(code, original_tokens, detected_lang, confidence);
        }

        // CCR offload (enable_ccr && ratio < 0.8) is owned by the dispatcher
        // in the Rust port; fixtures record with enable_ccr=False so
        // `cache_key` stays None and no marker is appended.

        CodeCompressionResult {
            compressed,
            original: code.to_string(),
            original_tokens,
            compressed_tokens,
            compression_ratio: ratio,
            language: detected_lang,
            language_confidence: confidence,
            preserved_imports: structure.imports.len() as i64,
            preserved_signatures: structure.function_signatures.len() as i64,
            compressed_bodies: structure.function_bodies.len() as i64,
            syntax_valid,
            cache_key: None,
            symbol_scores,
        }
    }

    /// Parse + analyze + extract + assemble. Returns
    /// `(compressed, structure, symbol_scores)`. `None` mirrors the Python
    /// `except Exception` fallback (here only reachable on a parse miss).
    fn compress_with_ast(
        &self,
        code: &str,
        language: CodeLanguage,
        context: &str,
    ) -> Option<AstCompression> {
        let tree = parse_code(code, language)?;
        let root = tree.root_node();

        let analysis = self.analyze_symbol_importance(root, code, language, context);
        let body_limits = self.allocate_body_budget(&analysis, code);

        let lang = lang_config(language);
        let (structure, symbol_scores) = if let Some(lang) = lang {
            let code_lines: Vec<&str> = code.split('\n').collect();
            let ctx = Ctx {
                code,
                code_lines,
                language,
                lang: &lang,
                body_limits: &body_limits,
                analysis: &analysis,
                config: &self.config,
            };
            let structure = ctx.extract_structure(root);
            // Expose scores under short names (max per short name).
            let mut symbol_scores: Vec<(String, f64)> = Vec::new();
            for (qname, score) in &analysis.scores {
                let short = analysis
                    .bare_names
                    .get(qname)
                    .cloned()
                    .unwrap_or_else(|| qname.clone());
                if let Some(existing) = symbol_scores.iter_mut().find(|(k, _)| *k == short) {
                    if *score > existing.1 {
                        existing.1 = *score;
                    }
                } else {
                    symbol_scores.push((short, *score));
                }
            }
            (structure, symbol_scores)
        } else {
            (extract_generic_structure(code), Vec::new())
        };

        let compressed = assemble_compressed(&structure);
        Some((compressed, structure, symbol_scores))
    }

    /// Verify that `code` re-parses without ERROR/MISSING. Mirrors `_verify_syntax`.
    fn verify_syntax(&self, code: &str, language: CodeLanguage) -> bool {
        match parse_code(code, language) {
            Some(tree) => !has_syntax_issues(tree.root_node()),
            None => false,
        }
    }
}

/// Build a passthrough result (compressed == original). Used for every
/// short-circuit / fallback branch.
fn passthrough_result(
    code: &str,
    original_tokens: i64,
    language: CodeLanguage,
    confidence: f64,
) -> CodeCompressionResult {
    CodeCompressionResult {
        compressed: code.to_string(),
        original: code.to_string(),
        original_tokens,
        compressed_tokens: original_tokens,
        compression_ratio: 1.0,
        language,
        language_confidence: confidence,
        preserved_imports: 0,
        preserved_signatures: 0,
        compressed_bodies: 0,
        syntax_valid: true,
        cache_key: None,
        symbol_scores: Vec::new(),
    }
}

fn extract_generic_structure(code: &str) -> CodeStructure {
    CodeStructure {
        other: code.split('\n').map(|s| s.to_string()).collect(),
        ..Default::default()
    }
}

/// Assemble compressed code from structure. Mirrors `_assemble_compressed`.
fn assemble_compressed(structure: &CodeStructure) -> String {
    let mut parts: Vec<String> = Vec::new();
    let push_section = |parts: &mut Vec<String>, section: &[String]| {
        if !section.is_empty() {
            parts.extend(section.iter().cloned());
            parts.push(String::new());
        }
    };
    push_section(&mut parts, &structure.imports);
    push_section(&mut parts, &structure.type_definitions);
    push_section(&mut parts, &structure.class_definitions);
    push_section(&mut parts, &structure.function_signatures);
    push_section(&mut parts, &structure.top_level_code);
    if !structure.other.is_empty() {
        parts.extend(structure.other.iter().cloned());
    }
    // Remove trailing blank lines.
    while let Some(last) = parts.last() {
        if last.trim().is_empty() {
            parts.pop();
        } else {
            break;
        }
    }
    parts.join("\n")
}

// ─── Symbol importance + body budget (impl CodeAwareCompressor) ─────────

/// Insertion-ordered put: update value if key present (keeps position),
/// else append. Mirrors Python dict assignment semantics.
fn ordered_put<V>(v: &mut Vec<(String, V)>, key: String, val: V) {
    if let Some(e) = v.iter_mut().find(|(k, _)| *k == key) {
        e.1 = val;
    } else {
        v.push((key, val));
    }
}

impl CodeAwareCompressor {
    /// Distribution-based symbol importance. Mirrors `_analyze_symbol_importance`.
    fn analyze_symbol_importance(
        &self,
        root: Node,
        code: &str,
        language: CodeLanguage,
        context: &str,
    ) -> SymbolAnalysis {
        if !self.config.semantic_analysis {
            return SymbolAnalysis::default();
        }
        let Some(lang) = lang_config(language) else {
            return SymbolAnalysis::default();
        };

        let is_def = |k: &str| lang.is_function(k) || lang.is_class(k);

        // Pass 1: collect definitions with qualified names (DFS, ordered).
        let mut definitions: Vec<(String, Node)> = Vec::new();
        let mut bare_names: HashMap<String, String> = HashMap::new();
        collect_definitions(
            root,
            "",
            code,
            &is_def,
            lang.decorator_node,
            &mut definitions,
            &mut bare_names,
        );
        if definitions.is_empty() {
            return SymbolAnalysis::default();
        }

        // Pass 2: collect all identifiers (short name → count).
        let mut all_identifiers: HashMap<String, i64> = HashMap::new();
        collect_identifiers(root, code, &mut all_identifiers);

        // Pass 3: call relationships + body sizes.
        let defined_short_names: BTreeSet<String> = bare_names.values().cloned().collect();
        let mut function_calls: Vec<(String, BTreeSet<String>)> = Vec::new();
        let mut body_line_counts: HashMap<String, i64> = HashMap::new();
        for (qname, node) in &definitions {
            let func_short = bare_names.get(qname).cloned().unwrap_or_default();
            let mut calls: BTreeSet<String> = BTreeSet::new();
            collect_calls(*node, code, &defined_short_names, &func_short, &mut calls);
            function_calls.push((qname.clone(), calls));
            let text = node_text(*node, code);
            let line_count = text.split('\n').count() as i64;
            body_line_counts.insert(qname.clone(), (line_count - 2).max(1));
        }

        // Reference counts: subtract definition occurrences.
        let mut short_name_def_count: HashMap<String, i64> = HashMap::new();
        for short in bare_names.values() {
            *short_name_def_count.entry(short.clone()).or_insert(0) += 1;
        }
        let mut ref_counts: HashMap<String, i64> = HashMap::new();
        for (qname, _) in &definitions {
            let short = &bare_names[qname];
            let count = *all_identifiers.get(short).unwrap_or(&0);
            let def_count = *short_name_def_count.get(short).unwrap_or(&1);
            ref_counts.insert(qname.clone(), (count - def_count).max(0));
        }

        // Context words (empty when context is "").
        let context_lower = context.to_lowercase();
        let context_words: BTreeSet<String> = if context.is_empty() {
            BTreeSet::new()
        } else {
            static SPLIT: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
            let re = SPLIT.get_or_init(|| regex::Regex::new(r#"[\s,;:.()\[\]{}"']+"#).unwrap());
            re.split(&context_lower)
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .collect()
        };

        // Raw importance signals per symbol.
        let mut raw_signals: Vec<(String, f64)> = Vec::new();
        for (qname, _) in &definitions {
            let short = bare_names[qname].clone();
            let refs = *ref_counts.get(qname).unwrap_or(&0);
            let fan_out = function_calls
                .iter()
                .find(|(k, _)| k == qname)
                .map(|(_, s)| s.len())
                .unwrap_or(0) as f64;
            let is_public = is_public_symbol(&short, language);

            let mut raw = refs as f64;
            raw += if is_public { 1.0 } else { 0.0 };
            raw += fan_out * 0.5;

            // Language conventions are mutually exclusive, so collapsing the
            // nested guards preserves the reference's branch behavior.
            if language == CodeLanguage::Python
                && short.starts_with("__")
                && short.ends_with("__")
            {
                raw += 2.0;
            } else if language == CodeLanguage::Go
                && short.chars().next().is_some_and(|c| c.is_uppercase())
            {
                raw += 1.0;
            }

            if !context_words.is_empty() {
                let name_lower = short.to_lowercase();
                if context_words.contains(&name_lower)
                    || (name_lower.chars().count() > 3 && context_lower.contains(&name_lower))
                {
                    raw += 3.0;
                }
            }
            raw_signals.push((qname.clone(), raw));
        }

        // Min-max normalization to [0, 1], round-3.
        let values: Vec<f64> = raw_signals.iter().map(|(_, v)| *v).collect();
        let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range_val = max_val - min_val;

        let mut scores: Vec<(String, f64)> = Vec::new();
        if range_val > 0.0 {
            for (name, v) in &raw_signals {
                scores.push((name.clone(), py_round3((v - min_val) / range_val)));
            }
        } else {
            for (name, _) in &raw_signals {
                scores.push((name.clone(), 0.5));
            }
        }

        SymbolAnalysis {
            scores,
            calls: function_calls,
            bare_names,
            body_line_counts,
        }
    }

    /// Allocate per-symbol body-line budgets. Mirrors `_allocate_body_budget`.
    fn allocate_body_budget(&self, analysis: &SymbolAnalysis, code: &str) -> HashMap<String, i64> {
        if analysis.scores.is_empty() || analysis.body_line_counts.is_empty() {
            return HashMap::new();
        }
        let target_rate = self.config.target_compression_rate;
        let total_lines = code.trim().split('\n').count() as i64;
        let total_body_lines: i64 = analysis.body_line_counts.values().sum();
        let fixed_lines = (total_lines - total_body_lines).max(0);
        let target_total = total_lines as f64 * target_rate;
        let body_budget = (target_total - fixed_lines as f64).max(0.0);

        if total_body_lines == 0 {
            return HashMap::new();
        }

        let score_floor = 0.05;
        // weights keyed by qname, in scores order.
        let mut weights: Vec<(String, f64)> = Vec::new();
        for (name, score) in &analysis.scores {
            let s = score.max(score_floor);
            let size = *analysis.body_line_counts.get(name).unwrap_or(&0);
            weights.push((name.clone(), s * size as f64));
        }
        let total_weight: f64 = weights.iter().map(|(_, w)| *w).sum();

        let mut limits: HashMap<String, i64> = HashMap::new();
        if total_weight == 0.0 {
            let per_func = (body_budget / (analysis.scores.len().max(1) as f64)).trunc().max(0.0) as i64;
            for (name, _) in &analysis.scores {
                let size = *analysis.body_line_counts.get(name).unwrap_or(&0);
                limits.insert(name.clone(), per_func.min(size));
            }
            return limits;
        }

        for (qname, _) in &analysis.scores {
            let weight = weights.iter().find(|(k, _)| k == qname).map(|(_, w)| *w).unwrap_or(0.0);
            let allocation = body_budget * weight / total_weight;
            let max_lines = *analysis.body_line_counts.get(qname).unwrap_or(&0);
            let limit = py_round_int(allocation).min(max_lines);
            limits.insert(qname.clone(), limit);
            let short = analysis.bare_names.get(qname).cloned().unwrap_or_else(|| qname.clone());
            match limits.get(&short) {
                Some(&existing) if limit <= existing => {}
                _ => {
                    limits.insert(short, limit);
                }
            }
        }
        limits
    }
}

/// DFS collect of qualified definition names → node. Mirrors the nested
/// `collect_definitions` closure.
fn collect_definitions<'t>(
    node: Node<'t>,
    parent_name: &str,
    code: &str,
    is_def: &dyn Fn(&str) -> bool,
    decorator_node: Option<&str>,
    definitions: &mut Vec<(String, Node<'t>)>,
    bare_names: &mut HashMap<String, String>,
) {
    let nt = node.kind();
    if is_def(nt) {
        if let Some(short) = get_definition_name(node, code) {
            let qualified = if parent_name.is_empty() {
                short.clone()
            } else {
                format!("{parent_name}.{short}")
            };
            ordered_put(definitions, qualified.clone(), node);
            bare_names.insert(qualified.clone(), short);
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                collect_definitions(child, &qualified, code, is_def, decorator_node, definitions, bare_names);
            }
            return;
        }
    }
    if let Some(dn) = decorator_node {
        if nt == dn {
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                if is_def(child.kind()) {
                    if let Some(short) = get_definition_name(child, code) {
                        let qualified = if parent_name.is_empty() {
                            short.clone()
                        } else {
                            format!("{parent_name}.{short}")
                        };
                        ordered_put(definitions, qualified.clone(), child);
                        bare_names.insert(qualified.clone(), short);
                        let mut gc = child.walk();
                        for grandchild in child.children(&mut gc) {
                            collect_definitions(grandchild, &qualified, code, is_def, decorator_node, definitions, bare_names);
                        }
                        return;
                    }
                }
            }
        }
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_definitions(child, parent_name, code, is_def, decorator_node, definitions, bare_names);
    }
}

/// DFS count of identifier-like nodes by (real) text. Mirrors `collect_identifiers`.
fn collect_identifiers(node: Node, code: &str, out: &mut HashMap<String, i64>) {
    let k = node.kind();
    if k == "identifier" || k == "property_identifier" || k == "type_identifier" {
        let name = node_text(node, code).to_string();
        *out.entry(name).or_insert(0) += 1;
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_identifiers(child, code, out);
    }
}

/// DFS collect of calls within a function. Mirrors `collect_calls_in_function`.
fn collect_calls(
    node: Node,
    code: &str,
    defined_short_names: &BTreeSet<String>,
    func_short: &str,
    calls: &mut BTreeSet<String>,
) {
    let k = node.kind();
    if k == "identifier" || k == "property_identifier" {
        let name = node_text(node, code);
        if defined_short_names.contains(name) && name != func_short {
            calls.insert(name.to_string());
        }
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_calls(child, code, defined_short_names, func_short, calls);
    }
}

// ─── Structure extraction (impl Ctx) ────────────────────────────────────

impl<'a> Ctx<'a> {
    fn node_text(&self, node: Node) -> &'a str {
        &self.code[node.start_byte()..node.end_byte()]
    }

    /// Lines `code_lines[start..=end]` joined with `\n`.
    fn lines_joined(&self, start: usize, end_inclusive: usize) -> String {
        self.code_lines[start..=end_inclusive].join("\n")
    }

    /// Extract structure from the AST. Mirrors `_extract_structure`.
    fn extract_structure(&self, root: Node) -> CodeStructure {
        let mut structure = CodeStructure::default();
        let mut captured: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
        self.visit(root, &mut structure, &mut captured);

        // Top-level children not captured → top_level_code.
        let mut cursor = root.walk();
        for child in root.children(&mut cursor) {
            let range = (child.start_byte(), child.end_byte());
            if !captured.contains(&range) {
                let text = self.node_text(child).trim();
                if !text.is_empty() {
                    structure.top_level_code.push(text.to_string());
                }
            }
        }
        structure
    }

    fn visit(
        &self,
        node: Node,
        structure: &mut CodeStructure,
        captured: &mut std::collections::HashSet<(usize, usize)>,
    ) {
        let nt = node.kind();
        let range = (node.start_byte(), node.end_byte());

        // Package declarations (Go, Java).
        if self.lang.package_node == Some(nt) {
            structure.imports.insert(0, self.node_text(node).to_string());
            captured.insert(range);
            return;
        }
        // Import statements.
        if self.lang.is_import(nt) {
            structure.imports.push(self.node_text(node).to_string());
            captured.insert(range);
            return;
        }
        // Export statements (JS/TS).
        if nt == "export_statement" {
            let text = self.node_text(node).to_string();
            let mut has_func_or_class = false;
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                if self.lang.is_function(child.kind()) || self.lang.is_class(child.kind()) {
                    has_func_or_class = true;
                    let compressed = self.compress_function_ast(child);
                    let export_prefix = &self.code[node.start_byte()..child.start_byte()];
                    let export_suffix = &self.code[child.end_byte()..node.end_byte()];
                    structure
                        .function_signatures
                        .push(format!("{export_prefix}{compressed}{export_suffix}"));
                    break;
                }
            }
            if !has_func_or_class {
                structure.imports.push(text);
            }
            captured.insert(range);
            return;
        }
        // Decorated definitions (Python).
        if self.lang.decorator_node == Some(nt) {
            let mut decorator_text: Vec<String> = Vec::new();
            let mut definition_compressed: Option<String> = None;
            let mut has_class_child = false;
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                let ck = child.kind();
                if ck == "decorator" {
                    decorator_text.push(self.node_text(child).to_string());
                } else if self.lang.is_function(ck) {
                    definition_compressed = Some(self.compress_function_ast(child));
                } else if self.lang.is_class(ck) {
                    definition_compressed = Some(self.compress_class_ast(child));
                }
                if self.lang.is_class(ck) {
                    has_class_child = true;
                }
            }
            match definition_compressed {
                Some(def) if !decorator_text.is_empty() => {
                    let full_def = format!("{}\n{}", decorator_text.join("\n"), def);
                    if has_class_child {
                        structure.class_definitions.push(full_def);
                    } else {
                        structure.function_signatures.push(full_def);
                    }
                }
                Some(def) => structure.function_signatures.push(def),
                None => {}
            }
            captured.insert(range);
            return;
        }
        // Function/method definitions.
        if self.lang.is_function(nt) {
            let compressed = self.compress_function_ast(node);
            structure.function_signatures.push(compressed);
            captured.insert(range);
            return;
        }
        // Class definitions.
        if self.lang.is_class(nt) {
            let compressed = self.compress_class_ast(node);
            structure.class_definitions.push(compressed);
            captured.insert(range);
            return;
        }
        // Type definitions.
        if self.lang.is_type(nt) {
            structure.type_definitions.push(self.node_text(node).to_string());
            captured.insert(range);
            return;
        }
        // Recurse.
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.visit(child, structure, captured);
        }
    }

    /// Compress a function/method body. Mirrors `_compress_function_ast`.
    fn compress_function_ast(&self, node: Node) -> String {
        let start_row = node.start_position().row;
        let end_row = node.end_position().row;
        let node_lines: Vec<&str> = self.code_lines[start_row..=end_row].to_vec();
        let node_text = node_lines.join("\n");

        let func_name = get_definition_name(node, self.code);
        let body_limit = get_body_limit(
            func_name.as_deref(),
            self.body_limits,
            self.config.max_body_lines,
        );

        if node_lines.len() as i64 <= body_limit + 2 {
            return node_text;
        }

        // Find the body node.
        let mut body_node: Option<Node> = None;
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if self.lang.is_body(child.kind()) {
                body_node = Some(child);
                break;
            }
        }
        let Some(body_node) = body_node else {
            return node_text;
        };

        let node_start_line = start_row;
        let body_start_line = body_node.start_position().row;
        let body_end_line = body_node.end_position().row;
        let sig_end = body_start_line - node_start_line; // exclusive
        let body_end_rel = body_end_line - node_start_line + 1; // inclusive

        let signature_lines: Vec<&str>;
        let mut body_lines: Vec<&str>;
        let after_lines: Vec<&str>;
        let brace_in_signature: bool;

        if sig_end == 0 && !self.lang.uses_colon_after_signature {
            let first_line = node_lines[0];
            signature_lines = vec![first_line.trim_end()];
            body_lines = node_lines[1..body_end_rel].to_vec();
            after_lines = node_lines[body_end_rel..].to_vec();
            brace_in_signature = true;
        } else {
            signature_lines = node_lines[..sig_end].to_vec();
            body_lines = node_lines[sig_end..body_end_rel].to_vec();
            after_lines = node_lines[body_end_rel..].to_vec();
            brace_in_signature = false;
        }

        // Brace detection for non-colon languages.
        let mut opening_brace_line: Option<&str> = None;
        let mut closing_brace_line: Option<&str> = None;
        if !self.lang.uses_colon_after_signature {
            if brace_in_signature {
                // opening brace already in signature line.
            } else if body_lines.first().is_some_and(|l| l.trim_start().starts_with('{')) {
                opening_brace_line = Some(body_lines[0]);
                body_lines = body_lines[1..].to_vec();
            }
            if body_lines.last().is_some_and(|l| l.trim_end().ends_with('}')) {
                closing_brace_line = Some(body_lines[body_lines.len() - 1]);
                body_lines = body_lines[..body_lines.len() - 1].to_vec();
            }
        }

        // Python docstring handling via AST.
        let mut docstring_text = String::new();
        let mut ds_skip_lines: usize = 0;
        if self.language == CodeLanguage::Python && body_node.child_count() > 0 {
            let first_child = body_node.child(0).unwrap();
            // tree-sitter Python represents a docstring either as a bare
            // `string` node in the block or as an `expression_statement`
            // wrapping a `string`. Both map to the same docstring node.
            let mut ds_node: Option<Node> = None;
            if first_child.kind() == "string"
                || (first_child.kind() == "expression_statement"
                    && first_child.child_count() > 0
                    && first_child.child(0).unwrap().kind() == "string")
            {
                ds_node = Some(first_child);
            }
            if let Some(ds_node) = ds_node {
                let ds_lines_count = ds_node.end_position().row - ds_node.start_position().row + 1;
                let ds_start_rel = ds_node.start_position().row - body_node.start_position().row;

                match self.config.docstring_mode {
                    DocstringMode::Full => {
                        let endi = (ds_start_rel + ds_lines_count).min(body_lines.len());
                        if ds_start_rel < body_lines.len() {
                            docstring_text = body_lines[ds_start_rel..endi].join("\n");
                        }
                    }
                    DocstringMode::FirstLine => {
                        if ds_lines_count == 1 {
                            if let Some(l) = body_lines.get(ds_start_rel) {
                                docstring_text = (*l).to_string();
                            }
                        } else if let Some(first_ds_line) = body_lines.get(ds_start_rel).copied() {
                            docstring_text =
                                first_line_docstring(first_ds_line, &body_lines, ds_start_rel);
                        }
                    }
                    DocstringMode::Remove | DocstringMode::None => {}
                }
                ds_skip_lines = ds_start_rel + ds_lines_count;
            }
        }

        // Statement-based body truncation.
        let indent = if !body_lines.is_empty() {
            detect_indent(&body_lines)
        } else {
            "    ".to_string()
        };

        let mut ds_end_row: i64 = -1;
        if ds_skip_lines > 0 && body_node.child_count() > 0 {
            ds_end_row = (body_node.start_position().row + ds_skip_lines) as i64 - 1;
        }

        const SKIP_TYPES: &[&str] =
            &["{", "}", ";", ",", "comment", "line_comment", "block_comment"];

        let mut body_stmts: Vec<(usize, usize)> = Vec::new();
        let mut bcursor = body_node.walk();
        for child in body_node.children(&mut bcursor) {
            if (child.start_position().row as i64) <= ds_end_row {
                continue;
            }
            if SKIP_TYPES.contains(&child.kind()) {
                continue;
            }
            if !child.is_named() {
                continue;
            }
            body_stmts.push((child.start_position().row, child.end_position().row));
        }

        let total_body_lines_count: i64 =
            body_stmts.iter().map(|(s, e)| (*e - *s + 1) as i64).sum();

        let mut kept_lines: Vec<&str> = Vec::new();
        let mut kept_line_count: i64 = 0;
        for (s_row, e_row) in &body_stmts {
            let stmt_lines: Vec<&str> = self.code_lines[*s_row..=*e_row].to_vec();
            let stmt_line_count = stmt_lines.len() as i64;
            // `!kept_lines.is_empty()` == Python's `stmts_kept > 0` guard:
            // always keep at least the first statement.
            if kept_line_count + stmt_line_count > body_limit && !kept_lines.is_empty() {
                break;
            }
            kept_lines.extend(stmt_lines);
            kept_line_count += stmt_line_count;
        }

        let omitted_lines = total_body_lines_count - kept_line_count;

        // Assemble.
        let mut result_parts: Vec<String> = Vec::new();
        if !signature_lines.is_empty() {
            result_parts.extend(signature_lines.iter().map(|s| s.to_string()));
        } else {
            let sig_text = self.code[node.start_byte()..body_node.start_byte()].trim_end();
            result_parts.push(sig_text.to_string());
        }
        if let Some(ob) = opening_brace_line {
            result_parts.push(ob.to_string());
        }
        if !docstring_text.is_empty()
            && self.config.docstring_mode != DocstringMode::None
            && self.config.docstring_mode != DocstringMode::Remove
        {
            result_parts.push(docstring_text);
        }
        if !kept_lines.is_empty() {
            result_parts.extend(kept_lines.iter().map(|s| s.to_string()));
        }
        if omitted_lines > 0 {
            result_parts.push(make_omitted_comment(
                func_name.as_deref(),
                omitted_lines,
                &indent,
                self.lang.comment_prefix,
                self.analysis,
            ));
            if self.lang.uses_colon_after_signature {
                result_parts.push(format!("{indent}pass"));
            }
        }
        if let Some(cb) = closing_brace_line {
            result_parts.push(cb.to_string());
        } else if !after_lines.is_empty() {
            result_parts.extend(after_lines.iter().map(|s| s.to_string()));
        }

        result_parts.join("\n")
    }

    /// Compress a class by compressing each method individually. Mirrors
    /// `_compress_class_ast`.
    fn compress_class_ast(&self, node: Node) -> String {
        let start_row = node.start_position().row;
        let end_row = node.end_position().row;
        let node_lines: Vec<&str> = self.code_lines[start_row..=end_row].to_vec();
        let node_text = node_lines.join("\n");

        let mut body_node: Option<Node> = None;
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if self.lang.is_body(child.kind()) {
                body_node = Some(child);
                break;
            }
        }
        let Some(body_node) = body_node else {
            return node_text;
        };

        let node_start_line = start_row;
        let body_start_line = body_node.start_position().row;
        let sig_end = body_start_line - node_start_line;
        let header_lines: Vec<&str> = if sig_end > 0 {
            node_lines[..sig_end].to_vec()
        } else {
            vec![node_lines[0]]
        };

        let mut body_parts: Vec<String> = Vec::new();
        let mut bcursor = body_node.walk();
        for child in body_node.children(&mut bcursor) {
            let ck = child.kind();
            let child_start = child.start_position().row;
            let child_end = child.end_position().row;
            let child_text = self.lines_joined(child_start, child_end);

            if self.lang.is_function(ck) {
                body_parts.push(self.compress_function_ast(child));
            } else if self.lang.decorator_node == Some(ck) {
                let mut decorator_lines: Vec<String> = Vec::new();
                let mut method_compressed: Option<String> = None;
                let mut dc = child.walk();
                for deco_child in child.children(&mut dc) {
                    if deco_child.kind() == "decorator" {
                        decorator_lines.push(self.node_text(deco_child).to_string());
                    } else if self.lang.is_function(deco_child.kind()) {
                        method_compressed = Some(self.compress_function_ast(deco_child));
                    }
                }
                match method_compressed {
                    Some(m) if !decorator_lines.is_empty() => {
                        body_parts.push(format!("{}\n{}", decorator_lines.join("\n"), m));
                    }
                    Some(m) => body_parts.push(m),
                    None => body_parts.push(child_text),
                }
            } else if self.lang.is_class(ck) {
                body_parts.push(self.compress_class_ast(child));
            } else if !child_text.trim().is_empty() {
                body_parts.push(child_text);
            }
        }

        let mut result_parts: Vec<String> = header_lines.iter().map(|s| s.to_string()).collect();
        result_parts.extend(body_parts);

        let body_end_line = body_node.end_position().row;
        let body_end_rel = body_end_line - node_start_line + 1;
        let after_lines: Vec<&str> = node_lines[body_end_rel..].to_vec();
        if !after_lines.is_empty() {
            result_parts.extend(after_lines.iter().map(|s| s.to_string()));
        } else if !self.lang.uses_colon_after_signature {
            let last_body_line = node_lines.last().copied().unwrap_or("");
            if last_body_line.trim() == "}" {
                result_parts.push(last_body_line.to_string());
            }
        }

        result_parts.join("\n")
    }
}

/// FIRST_LINE multi-line docstring reconstruction. Mirrors the inner block
/// of `_compress_function_ast` (DocstringMode.FIRST_LINE, multi-line).
fn first_line_docstring(first_ds_line: &str, body_lines: &[&str], ds_start_rel: usize) -> String {
    let ds_indent = leading_ws(first_ds_line);
    let stripped = first_ds_line.trim();

    const OPENERS: &[&str] = &["r\"\"\"", "r'''", "\"\"\"", "'''"];
    let mut quote = "\"\"\"";
    let mut content_start = 0usize;
    for opener in OPENERS {
        if stripped.starts_with(opener) {
            quote = &opener[opener.len() - 3..];
            content_start = opener.len();
            break;
        }
    }

    let mut first_content = stripped[content_start..].trim().to_string();
    for q in ["\"\"\"", "'''"] {
        if first_content.ends_with(q) {
            first_content = first_content[..first_content.len() - q.len()].trim().to_string();
        }
    }

    if !first_content.is_empty() {
        let prefix_part = &stripped[..content_start];
        format!("{ds_indent}{prefix_part}{first_content}{quote}")
    } else if ds_start_rel + 1 < body_lines.len() {
        let mut second_line = body_lines[ds_start_rel + 1].trim().to_string();
        for q in ["\"\"\"", "'''"] {
            if second_line.ends_with(q) {
                second_line = second_line[..second_line.len() - q.len()].trim().to_string();
            }
        }
        if !second_line.is_empty() {
            format!("{ds_indent}{quote}{second_line}{quote}")
        } else {
            first_ds_line.to_string()
        }
    } else {
        first_ds_line.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn py_round_int_is_half_to_even() {
        // Reference: CPython round(x) — ties to even.
        assert_eq!(py_round_int(0.5), 0);
        assert_eq!(py_round_int(1.5), 2);
        assert_eq!(py_round_int(2.5), 2);
        assert_eq!(py_round_int(3.5), 4);
        assert_eq!(py_round_int(-2.5), -2);
        assert_eq!(py_round_int(2.4), 2);
        assert_eq!(py_round_int(2.6), 3);
        assert_eq!(py_round_int(0.0), 0);
        assert_eq!(py_round_int(4.5), 4);
    }

    #[test]
    fn py_round3_matches_cpython() {
        // Reference: CPython round(x, 3) — correctly rounded, ties to even.
        assert_eq!(py_round3(1.0 / 3.0), 0.333);
        assert_eq!(py_round3(2.0 / 3.0), 0.667);
        assert_eq!(py_round3(0.0625), 0.062); // half-even (not 0.063)
        assert_eq!(py_round3(0.1235), 0.123);
        assert_eq!(py_round3(0.5005), 0.5);
        assert_eq!(py_round3(0.12345), 0.123);
        assert_eq!(py_round3(1.0), 1.0);
        assert_eq!(py_round3(0.524822695035461), 0.525);
    }

    #[test]
    fn estimate_tokens_uses_chars_div_4_min_1() {
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens("abcdefgh"), 2);
        assert_eq!(estimate_tokens("a"), 1); // max(1, 0)
    }

    #[test]
    fn detect_language_basic() {
        let (lang, conf) = detect_language("import os\n\ndef f(x):\n    return x + 1\n");
        assert_eq!(lang, CodeLanguage::Python);
        assert!(conf >= 0.3 && conf <= 1.0);

        let (lang, _) = detect_language("package main\n\nfunc main() {}\n");
        assert_eq!(lang, CodeLanguage::Go);

        let (lang, conf) = detect_language("");
        assert_eq!(lang, CodeLanguage::Unknown);
        assert_eq!(conf, 0.0);

        let (lang, _) = detect_language("just plain english prose with no code here at all");
        assert_eq!(lang, CodeLanguage::Unknown);
    }

    #[test]
    fn empty_and_short_passthrough() {
        let c = CodeAwareCompressor::new(CodeCompressorConfig::default());
        let r = c.compress("");
        assert_eq!(r.compressed, "");
        assert_eq!(r.original_tokens, 0);
        assert_eq!(r.language, CodeLanguage::Unknown);

        let r = c.compress("def f(): pass");
        assert_eq!(r.compressed, "def f(): pass"); // < min_tokens → passthrough
        assert_eq!(r.compression_ratio, 1.0);
        assert_eq!(r.language, CodeLanguage::Unknown);
    }
}
