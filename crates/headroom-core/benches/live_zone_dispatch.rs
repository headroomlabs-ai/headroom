//! Latency benchmark for the PR-B4 live-zone dispatch arms.
//!
//! Measures `compress_anthropic_live_zone` end to end — body parse,
//! live-zone walk, content-type detection, byte-threshold gate, and the
//! per-arm compressor — over representative payload shapes:
//!
//! - `source_code_2kb` / `source_code_20kb`: generated Python routed to
//!   the CodeAwareCompressor arm (the tree-sitter parse dominates).
//! - `plain_text_8kb`: prose routed to the Kompress arm. What this case
//!   actually measures depends on the machine's Hugging Face cache:
//!   with the Kompress model cache-resident it benchmarks a real ONNX
//!   inference; on a cold cache `Kompress::from_cache` resolves to a
//!   deterministic NoOp and the number is detection + gate cost only.
//!   Both are legitimate measurements — record which state applied
//!   alongside any published number.
//! - `below_threshold_prose`: a payload under every byte threshold —
//!   the cost of the gate itself, i.e. the overhead every small block
//!   pays whether or not compression ever fires.
//!
//! Deliberately NOT wired into CI — benchmarks in shared CI runners
//! generate noise, not signal. Run manually:
//!
//! ```text
//! cargo bench -p headroom-core --bench live_zone_dispatch
//! ```
//!
//! # Windows: `ORT_DYLIB_PATH`
//!
//! On Windows with a warm model cache, ONNX Runtime's shared-library
//! resolution can deadlock inside `ort` init unless `ORT_DYLIB_PATH`
//! points at the onnxruntime library (see
//! `docs/content/docs/troubleshooting.mdx`, "Windows ML DLL" entry). A
//! hung benchmark is strictly worse than a refused one, so on Windows
//! this harness exits early with instructions when the variable is
//! missing AND the Kompress model cache is warm — the only state where
//! the hang is reachable. Cache-cold it proceeds (no ONNX session is
//! ever created; the plain_text case then measures the deterministic
//! no-op path, a valid mode in its own right). Other platforms resolve
//! the library normally and are not gated.

use std::hint::black_box;

use criterion::{criterion_group, Criterion, Throughput};
use headroom_core::transforms::live_zone::DEFAULT_MODEL;
use headroom_core::transforms::{compress_anthropic_live_zone, AuthMode};
use serde_json::json;

/// Build the standard single-`tool_result` Anthropic body around `text`
/// — the same shape the dispatch integration tests use (duplicated, not
/// shared: benches and integration tests are independent targets).
fn body_with_tool_result(text: &str) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "system": "you are a helpful assistant",
        "messages": [{
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_dispatch_bench",
                "content": text,
            }],
        }],
    }))
    .expect("bench body serializes")
}

/// Syntactically valid Python that detects as `SourceCode`, sized to at
/// least `target_bytes`. Same generator style as the dispatch tests'
/// `python_module_source`, parameterized by byte target instead of
/// function count so the two size cases are explicit at the call site.
fn python_source(target_bytes: usize) -> String {
    let mut code = String::from(
        "\"\"\"Example data-processing module used by the dispatch bench.\"\"\"\n\n\
         import json\nimport os\nfrom typing import Any, Optional\n\n\n",
    );
    let mut i = 0usize;
    while code.len() < target_bytes {
        code.push_str(&format!(
            "def process_record_{i}(record: dict) -> dict:\n    \
             \"\"\"Normalize record {i} and compute its derived fields.\"\"\"\n    \
             result = dict(record)\n    \
             result[\"index\"] = {i}\n    \
             result[\"doubled\"] = record.get(\"value\", 0) * 2\n    \
             result[\"source\"] = \"batch\"\n    \
             if result[\"doubled\"] > 100:\n        \
             result[\"flag\"] = \"high\"\n    \
             else:\n        \
             result[\"flag\"] = \"low\"\n    \
             return result\n\n\n"
        ));
        i += 1;
    }
    code
}

/// Varied natural-language prose that detects as `PlainText`, sized to
/// at least `target_bytes`. Sentence shape varies so this is prose to
/// the detector, not a single repeated token run.
fn prose(target_bytes: usize) -> String {
    const WORDS: &[&str] = &[
        "the",
        "release",
        "shipped",
        "after",
        "review",
        "and",
        "every",
        "service",
        "reported",
        "healthy",
        "metrics",
        "while",
        "operators",
        "watched",
        "dashboards",
        "during",
        "rollout",
        "windows",
        "before",
        "traffic",
        "returned",
        "to",
        "baseline",
        "levels",
        "overnight",
    ];
    let mut text = String::with_capacity(target_bytes + 64);
    let mut in_sentence = 0u32;
    for (n, word) in WORDS.iter().cycle().enumerate() {
        if text.len() >= target_bytes {
            break;
        }
        text.push_str(word);
        in_sentence += 1;
        // Vary sentence length between 7 and 12 words.
        if in_sentence >= 7 + (n as u32 % 6) {
            text.push_str(".\n");
            in_sentence = 0;
        } else {
            text.push(' ');
        }
    }
    text
}

fn dispatch(body: &[u8]) {
    let outcome = compress_anthropic_live_zone(body, 0, AuthMode::Payg, DEFAULT_MODEL)
        .expect("dispatcher returns Ok on valid bodies");
    black_box(outcome);
}

fn bench_dispatch(c: &mut Criterion) {
    let cases: &[(&str, Vec<u8>)] = &[
        (
            "source_code_2kb",
            body_with_tool_result(&python_source(2_200)),
        ),
        (
            "source_code_20kb",
            body_with_tool_result(&python_source(20_000)),
        ),
        ("plain_text_8kb", body_with_tool_result(&prose(8_192))),
        ("below_threshold_prose", body_with_tool_result(&prose(400))),
    ];

    // One warm-up dispatch per payload BEFORE any timed group, so
    // process-latched one-time costs (compressor singletons, the
    // Kompress `OnceLock` init and — cache-warm — its model load) land
    // here rather than skewing the first timed case.
    for (_, body) in cases {
        dispatch(body);
    }

    let mut group = c.benchmark_group("live_zone/dispatch");
    // The Kompress arm runs a real ONNX inference per iteration when
    // the model is cache-resident; keep sampling time bounded.
    group.sample_size(30);
    for (name, body) in cases {
        group.throughput(Throughput::Bytes(body.len() as u64));
        group.bench_function(*name, |b| b.iter(|| dispatch(black_box(body))));
    }
    group.finish();
}

/// Refuse to run on Windows without `ORT_DYLIB_PATH` — but only when the
/// hang it guards against is actually reachable. The `ort`-init deadlock
/// needs a warm Kompress model cache: cache-cold, `Kompress::from_cache`
/// resolves `Ok(None)` before any ONNX session exists, no `ort` code
/// runs, and the cache-cold measurement mode the module doc advertises
/// is perfectly safe — refusing it would block a legitimate
/// configuration to guard against a hang it cannot have. A hung
/// benchmark process gives no diagnostic; this message does.
#[cfg(windows)]
fn check_ort_dylib_path() {
    match std::env::var("ORT_DYLIB_PATH") {
        Ok(v) if !v.trim().is_empty() => {}
        _ if !kompress_model_cached() => {
            eprintln!(
                "live_zone_dispatch bench: ORT_DYLIB_PATH is not set, but the \
                 Kompress model cache is cold — no ONNX session will be \
                 created, so the Windows ort-init hang cannot occur. \
                 Proceeding in cache-cold mode (the plain_text case measures \
                 the deterministic no-op path)."
            );
        }
        _ => {
            eprintln!(
                "live_zone_dispatch bench: ORT_DYLIB_PATH is not set.\n\n\
                 On Windows with a warm Kompress model cache, ONNX Runtime's\n\
                 shared-library resolution can deadlock during `ort` init (see\n\
                 docs/content/docs/troubleshooting.mdx), which would hang the\n\
                 plain_text bench case indefinitely. Set ORT_DYLIB_PATH to the\n\
                 onnxruntime shared library and re-run, e.g.:\n\n  \
                 ORT_DYLIB_PATH=C:\\path\\to\\onnxruntime.dll cargo bench \
                 -p headroom-core --bench live_zone_dispatch\n\n\
                 Refusing to start rather than risk a silent hang."
            );
            std::process::exit(2);
        }
    }
}

/// Whether the Kompress model is cache-resident, asked of the loader
/// itself. A hand-rolled probe would have to mirror
/// `Kompress::from_cache`'s root and artifact resolution and would go
/// stale silently; this cannot.
#[cfg(all(windows, feature = "ml"))]
fn kompress_model_cached() -> bool {
    use headroom_core::transforms::kompress::{Kompress, KompressConfig};

    matches!(Kompress::from_cache(KompressConfig::default()), Ok(Some(_)))
}

/// Without the `ml` feature there is no ONNX session to deadlock on.
#[cfg(all(windows, not(feature = "ml")))]
fn kompress_model_cached() -> bool {
    false
}

criterion_group!(benches, bench_dispatch);

fn main() {
    #[cfg(windows)]
    check_ort_dylib_path();
    benches();
    Criterion::default().configure_from_args().final_summary();
}
