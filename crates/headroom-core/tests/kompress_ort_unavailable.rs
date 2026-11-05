//! Kompress must return an error, not hang, when ONNX Runtime can't load.
//!
//! The crate loads ONNX Runtime dynamically (`ort-load-dynamic`). When the
//! dylib can't be loaded, `ort` 2.0.0-rc.12 deadlocks in its API-setup error
//! path (recursive `OnceLock` init) instead of returning an error, so every
//! session must go through `dynamic_ort_loader_ready` first, as magika and
//! fastembed do. Kompress used to call `Session::builder()` directly: with
//! the model in the HF cache and no loadable ONNX Runtime, the
//! `kompress_parity` test hung forever.
//!
//! This is its own test binary because the loader result is process-wide.

#![cfg(feature = "ml")]

use std::io::Write;
use std::sync::mpsc;
use std::time::Duration;

use headroom_core::transforms::kompress::{Kompress, KompressConfig, KompressError};

/// Smallest tokenizer `tokenizers` accepts, so loading gets past the
/// tokenizer and reaches the ONNX Runtime session.
const TOKENIZER_JSON: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null,
  "model": {"type": "WordLevel", "vocab": {"[UNK]": 0}, "unk_token": "[UNK]"}
}"#;

#[test]
fn missing_onnx_runtime_is_an_error_not_a_hang() {
    let dir = tempfile::tempdir().expect("tempdir");
    let tokenizer = dir.path().join("tokenizer.json");
    std::fs::write(&tokenizer, TOKENIZER_JSON).expect("write tokenizer");
    let onnx = dir.path().join("model.onnx");
    std::fs::write(&onnx, b"never read: ONNX Runtime fails to load first").expect("write onnx");
    std::env::set_var(
        "ORT_DYLIB_PATH",
        dir.path().join("libonnxruntime-does-not-exist.dylib"),
    );

    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(Kompress::from_files(
            &tokenizer,
            &onnx,
            KompressConfig::default(),
        ));
    });

    match rx.recv_timeout(Duration::from_secs(60)) {
        Ok(Err(err @ KompressError::Onnx { .. })) => {
            // On an x86 CPU without AVX2 the CPU gate answers first, before
            // any library is loaded; either reason proves it returned.
            let message = err.to_string();
            assert!(
                message.contains("ORT_DYLIB_PATH") || message.contains("cpu_unsupported"),
                "error should name ORT_DYLIB_PATH (or the CPU gate): {message}"
            );
        }
        Ok(Err(other)) => panic!("expected KompressError::Onnx, got: {other}"),
        Ok(Ok(_)) => panic!("Kompress loaded without an ONNX Runtime library"),
        Err(_) => {
            // The stuck thread also wedges normal process exit (ort's
            // teardown waits on the same lock), so abort instead of
            // returning a failure the harness could never report. Write to
            // stderr directly: libtest's capture would swallow `eprintln!`.
            let _ = std::io::stderr()
                .write_all(b"Kompress::from_files deadlocked loading a missing ONNX Runtime\n");
            std::process::abort();
        }
    }
}
