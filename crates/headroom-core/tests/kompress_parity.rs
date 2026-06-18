//! Byte-parity integration test for the Kompress Rust port.
//!
//! Runs the production [`Kompress`] engine against the trace fixtures
//! recorded from the Python reference (`tests/parity/fixtures/kompress/`)
//! and asserts the compressed output matches byte-for-byte.
//!
//! Model-gated: if the ModernBERT tokenizer + kompress-v2-base ONNX
//! artifact are not present in the local HuggingFace cache (e.g. CI with
//! no network / no preloaded model), the test SKIPS rather than fails —
//! mirroring the parity harness's "stub → Skipped" tolerance. Run it
//! locally after `python scripts/record_kompress_trace.py` to get the
//! real assertion.

use std::fs;
use std::path::{Path, PathBuf};

use headroom_core::transforms::kompress::{Kompress, KompressConfig};
use serde_json::Value;

fn hf_cache_file(repo_dir: &str, rel: &[&str]) -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let snapshots = Path::new(&home)
        .join(".cache/huggingface/hub")
        .join(repo_dir)
        .join("snapshots");
    for snap in fs::read_dir(snapshots).ok()?.filter_map(|e| e.ok()) {
        let mut cand = snap.path();
        for part in rel {
            cand = cand.join(part);
        }
        if cand.exists() {
            return Some(cand);
        }
    }
    None
}

#[test]
fn kompress_matches_python_fixtures_byte_for_byte() {
    let tok = hf_cache_file(
        "models--answerdotai--ModernBERT-base",
        &["tokenizer.json"],
    );
    let onnx = hf_cache_file(
        "models--chopratejas--kompress-v2-base",
        &["onnx", "kompress-int8-wo.onnx"],
    );
    let (tok, onnx) = match (tok, onnx) {
        (Some(t), Some(o)) => (t, o),
        _ => {
            eprintln!(
                "SKIP: kompress model/tokenizer not in HF cache; \
                 run `python scripts/record_kompress_trace.py` first"
            );
            return;
        }
    };

    let fixtures_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/parity/fixtures/kompress");
    if !fixtures_dir.exists() {
        eprintln!("SKIP: fixtures dir {} missing", fixtures_dir.display());
        return;
    }

    let kompress = Kompress::from_files(&tok, &onnx, KompressConfig::default())
        .expect("load kompress from local files");

    let mut checked = 0usize;
    let mut paths: Vec<PathBuf> = fs::read_dir(&fixtures_dir)
        .expect("read fixtures dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().map(|x| x == "json").unwrap_or(false)
                && p.file_name().map(|n| n != "_manifest.json").unwrap_or(false)
        })
        .collect();
    paths.sort();

    for path in paths {
        let fx: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        // Standard parity fixture: {transform, input, config, output}.
        let content = fx["input"].as_str().expect("fixture.input string");
        let out = &fx["output"];
        let exp_compressed = out["compressed"].as_str().expect("output.compressed");
        let exp_ratio = out["compression_ratio"]
            .as_f64()
            .expect("output.compression_ratio");

        let result = kompress.compress(content);

        assert_eq!(
            result.compressed, exp_compressed,
            "[{name}] compressed output diverged from Python reference"
        );
        assert!(
            (result.compression_ratio - exp_ratio).abs() < 1e-6,
            "[{name}] ratio {} != python {}",
            result.compression_ratio,
            exp_ratio
        );
        checked += 1;
    }

    assert!(checked > 0, "no kompress fixtures were checked");
    eprintln!("kompress parity: {checked} fixtures matched byte-for-byte");
}

#[test]
fn short_input_passes_through() {
    // Pure-logic check — no model needed. Fewer than MIN_WORDS words must
    // pass through unchanged regardless of model availability... but the
    // engine needs a model to construct. Guard on cache like the main test.
    let tok = hf_cache_file("models--answerdotai--ModernBERT-base", &["tokenizer.json"]);
    let onnx = hf_cache_file(
        "models--chopratejas--kompress-v2-base",
        &["onnx", "kompress-int8-wo.onnx"],
    );
    let (Some(tok), Some(onnx)) = (tok, onnx) else {
        eprintln!("SKIP: model not cached");
        return;
    };
    let kompress = Kompress::from_files(&tok, &onnx, KompressConfig::default()).unwrap();

    let short = "only a few words here";
    let r = kompress.compress(short);
    assert!(r.is_passthrough());
    assert_eq!(r.compressed, short);
    assert_eq!(r.compression_ratio, 1.0);
}
