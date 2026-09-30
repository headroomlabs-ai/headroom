//! Byte-parity integration test for the CodeCompressor Rust port.
//!
//! Runs the production [`CodeAwareCompressor`] against the fixtures recorded
//! from the Python reference (`tests/parity/fixtures/code_aware_compressor/`)
//! and asserts the serialized result matches field-for-field.
//!
//! Unlike the Kompress test, this needs no model/network: the per-language
//! tree-sitter grammars are compiled into the crate. Grammar-version parity
//! is guaranteed by the exact Cargo pins matching the Python wheels the
//! fixtures were recorded against (see `Cargo.toml`).

use std::fs;
use std::path::{Path, PathBuf};

use headroom_core::transforms::code_compressor::{
    CodeAwareCompressor, CodeCompressorConfig, DocstringMode,
};
use serde_json::Value;

fn config_from_fixture(config: &Value) -> CodeCompressorConfig {
    let d = CodeCompressorConfig::default();
    CodeCompressorConfig {
        preserve_imports: config
            .get("preserve_imports")
            .and_then(|v| v.as_bool())
            .unwrap_or(d.preserve_imports),
        preserve_signatures: config
            .get("preserve_signatures")
            .and_then(|v| v.as_bool())
            .unwrap_or(d.preserve_signatures),
        preserve_type_annotations: config
            .get("preserve_type_annotations")
            .and_then(|v| v.as_bool())
            .unwrap_or(d.preserve_type_annotations),
        preserve_decorators: config
            .get("preserve_decorators")
            .and_then(|v| v.as_bool())
            .unwrap_or(d.preserve_decorators),
        docstring_mode: config
            .get("docstring_mode")
            .and_then(|v| v.as_str())
            .and_then(DocstringMode::from_value)
            .unwrap_or(d.docstring_mode),
        target_compression_rate: config
            .get("target_compression_rate")
            .and_then(|v| v.as_f64())
            .unwrap_or(d.target_compression_rate),
        max_body_lines: config
            .get("max_body_lines")
            .and_then(|v| v.as_i64())
            .unwrap_or(d.max_body_lines),
        compress_comments: config
            .get("compress_comments")
            .and_then(|v| v.as_bool())
            .unwrap_or(d.compress_comments),
        min_tokens_for_compression: config
            .get("min_tokens_for_compression")
            .and_then(|v| v.as_i64())
            .unwrap_or(d.min_tokens_for_compression),
        language_hint: config
            .get("language_hint")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        fallback_to_kompress: config
            .get("fallback_to_kompress")
            .and_then(|v| v.as_bool())
            .unwrap_or(d.fallback_to_kompress),
        semantic_analysis: config
            .get("semantic_analysis")
            .and_then(|v| v.as_bool())
            .unwrap_or(d.semantic_analysis),
        enable_ccr: config
            .get("enable_ccr")
            .and_then(|v| v.as_bool())
            .unwrap_or(d.enable_ccr),
        ccr_ttl: config
            .get("ccr_ttl")
            .and_then(|v| v.as_i64())
            .unwrap_or(d.ccr_ttl),
    }
}

#[test]
fn code_compressor_matches_python_fixtures_byte_for_byte() {
    let fixtures_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/parity/fixtures/code_aware_compressor");
    assert!(
        fixtures_dir.exists(),
        "fixtures dir {} missing — run scripts/record_code_compressor_fixtures.py",
        fixtures_dir.display()
    );

    let mut paths: Vec<PathBuf> = fs::read_dir(&fixtures_dir)
        .expect("read fixtures dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "json").unwrap_or(false))
        .collect();
    paths.sort();

    let mut checked = 0usize;
    let mut nontrivial = 0usize;
    for path in &paths {
        let fx: Value = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let content = fx["input"].as_str().expect("fixture.input string");
        let expected = &fx["output"];

        let cfg = config_from_fixture(&fx["config"]);
        let result = CodeAwareCompressor::new(cfg).compress(content);

        let mut symbol_scores = serde_json::Map::new();
        for (k, v) in &result.symbol_scores {
            symbol_scores.insert(k.clone(), serde_json::json!(v));
        }
        let actual = serde_json::json!({
            "cache_key": result.cache_key,
            "compressed": result.compressed,
            "compressed_bodies": result.compressed_bodies,
            "compressed_tokens": result.compressed_tokens,
            "compression_ratio": result.compression_ratio,
            "language": result.language.value(),
            "language_confidence": result.language_confidence,
            "original": result.original,
            "original_tokens": result.original_tokens,
            "preserved_imports": result.preserved_imports,
            "preserved_signatures": result.preserved_signatures,
            "symbol_scores": serde_json::Value::Object(symbol_scores),
            "syntax_valid": result.syntax_valid,
        });
        // Normalize through serde (f64 round-trip), matching the harness.
        let actual: Value = serde_json::from_str(&serde_json::to_string(&actual).unwrap()).unwrap();

        assert_eq!(
            &actual, expected,
            "[{name}] code compressor output diverged from Python reference"
        );
        if result.compressed != content {
            nontrivial += 1;
        }
        checked += 1;
    }

    assert!(checked >= 20, "expected >= 20 fixtures, got {checked}");
    assert!(
        nontrivial >= 10,
        "expected >= 10 non-trivial compressions, got {nontrivial} (are the fixtures all passthroughs?)"
    );
    eprintln!("code_compressor parity: {checked} fixtures matched ({nontrivial} non-trivial)");
}
