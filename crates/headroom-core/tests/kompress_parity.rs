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
//!
//! Runtime-gated too: with the model cached but no way to run it (a CPU
//! without AVX2, or no ONNX Runtime library: `ORT_DYLIB_PATH` unset and no
//! pip `onnxruntime` under `VIRTUAL_ENV`, `CONDA_PREFIX` or `./.venv`), it
//! also SKIPS, the way `magika_idle_threads` skips without an ONNX Runtime.
//! The check is `kompress::onnx_runtime_ready`, the constructors' own
//! CPU-then-loader preconditions, so it never loads the runtime on a CPU
//! that can't run it.

use std::fs;
use std::path::{Path, PathBuf};

use headroom_core::transforms::kompress::{self, Kompress, KompressConfig};
use serde_json::Value;

/// The tokenizer + int8 ONNX artifact production would load: the pinned
/// snapshots in Python's hub cache. The fixtures were recorded against them.
fn cached_artifacts() -> Option<(PathBuf, PathBuf)> {
    kompress::cached_artifacts(&KompressConfig::default(), "onnx/kompress-int8-wo.onnx")
}

/// [`cached_artifacts`], or `None` (with the reason on stderr) when the
/// model isn't cached or this machine can't run ONNX Runtime.
fn loadable_artifacts() -> Option<(PathBuf, PathBuf)> {
    let Some(paths) = cached_artifacts() else {
        eprintln!(
            "SKIP: kompress model/tokenizer not in HF cache; \
             run `python scripts/record_kompress_trace.py` first"
        );
        return None;
    };
    if let Err(reason) = kompress::onnx_runtime_ready() {
        eprintln!("SKIP: can't run ONNX Runtime for kompress here: {reason}");
        return None;
    }
    Some(paths)
}

#[test]
fn kompress_matches_python_fixtures_byte_for_byte() {
    let Some((tok, onnx)) = loadable_artifacts() else {
        return;
    };

    let fixtures_dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/fixtures/kompress");
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
                && p.file_name()
                    .map(|n| n != "_manifest.json")
                    .unwrap_or(false)
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
    // engine needs a model to construct. Guard like the main test.
    let Some((tok, onnx)) = loadable_artifacts() else {
        return;
    };
    let kompress = Kompress::from_files(&tok, &onnx, KompressConfig::default()).unwrap();

    let short = "only a few words here";
    let r = kompress.compress(short);
    assert!(r.is_passthrough());
    assert_eq!(r.compressed, short);
    assert_eq!(r.compression_ratio, 1.0);
}
