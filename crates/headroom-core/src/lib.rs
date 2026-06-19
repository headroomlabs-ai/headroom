//! headroom-core: foundation crate for the Rust port of Headroom.

pub mod auth_mode;
pub mod cache_control;
pub mod ccr;
pub mod compression_policy;
pub mod relevance;
pub mod signals;
pub mod tokenizer;
pub mod transforms;

// Re-exports for the live-zone dispatcher (Phase B PR-B2 consumes this).
// Hoisted to the crate root so the proxy crate gets one stable import
// path: `use headroom_core::compute_frozen_count;`. Keeping the
// `cache_control` module public too means downstream code can reach
// the helper types directly when needed.
pub use cache_control::compute_frozen_count;

/// Identity stub used by downstream crates and the Python binding to verify
/// linkage end-to-end.
pub fn hello() -> &'static str {
    "headroom-core"
}

/// Initialise the ONNX Runtime execution provider from `HEADROOM_ORT_EP`.
///
/// Must be called once at process startup, before any ONNX session is
/// created (fastembed `EmbeddingScorer` or magika `Session`). Both share
/// the same ORT singleton so a single `ort::init()` commit covers both.
///
/// ort 2.x selects EPs at runtime, but a non-CPU EP only works if the loaded
/// `onnxruntime` actually contains it. The OpenVINO EP in particular is NOT in
/// a stock `onnxruntime` — it requires an OpenVINO-enabled build (e.g. the
/// `onnxruntime-openvino` distribution) plus a matching `openvino` runtime on
/// the library path. On Windows, `ort-load-dynamic` will otherwise pick up
/// `C:\Windows\System32\onnxruntime.dll` (no OpenVINO EP) and this call logs a
/// misleading success while every session silently runs on CPU. Point
/// `ORT_DYLIB_PATH` at an OpenVINO-enabled `onnxruntime.dll`, and put the
/// matching `openvino` libs (version-matched to the provider) on `PATH`.
///
/// Valid values for `HEADROOM_ORT_EP`:
/// - unset / `cpu` — no-op; ORT defaults to CPU (always available)
/// - `openvino`    — Intel CPU, GPU, or NPU via OpenVINO runtime
/// - `cuda`        — NVIDIA GPU via CUDA
///
/// OpenVINO tuning env vars (only read when `HEADROOM_ORT_EP=openvino`):
/// - `HEADROOM_ORT_OPENVINO_DEVICE` — device string passed to OpenVINO
///   (default: `NPU`; also accepts `CPU`, `GPU`, `GPU.0`, `HETERO:NPU,GPU`)
/// - `HEADROOM_ORT_OPENVINO_CACHE` — directory for compiled NPU/GPU blobs;
///   first run compiles and saves, subsequent runs load instantly
pub fn init_ort_ep() {
    use ort::execution_providers::{OpenVINO, CUDA};

    let ep = std::env::var("HEADROOM_ORT_EP").unwrap_or_default();
    let ep = ep.trim().to_lowercase();

    match ep.as_str() {
        "" | "cpu" => {
            tracing::debug!(ep = "cpu", "ORT execution provider: CPU (default)");
        }
        "openvino" => {
            let device =
                std::env::var("HEADROOM_ORT_OPENVINO_DEVICE").unwrap_or_else(|_| "NPU".to_string());
            // `with_dynamic_shapes(false)` => OVEP `disable_dynamic_shapes=true`:
            // compile for a fixed input shape. REQUIRED for the NPU device — the
            // OVEP defaults to dynamic-shape compilation, which makes the NPU
            // graph compiler hang for minutes (vs ~13s for a static shape).
            // Callers feeding this EP must use fixed-shape inputs (e.g. the
            // Kompress static-shape model pads each chunk to a fixed length).
            let mut builder = OpenVINO::default()
                .with_device_type(&device)
                .with_dynamic_shapes(false);
            if let Ok(cache) = std::env::var("HEADROOM_ORT_OPENVINO_CACHE") {
                builder = builder.with_cache_dir(&cache);
            }
            if ort::init()
                .with_execution_providers([builder.build()])
                .commit()
            {
                tracing::info!(
                    ep = "openvino",
                    device = %device,
                    "ORT execution provider: OpenVINO"
                );
            } else {
                tracing::warn!(
                    ep = "openvino",
                    device = %device,
                    "ORT OpenVINO EP unavailable — falling back to CPU"
                );
            }
        }
        "cuda" => {
            if ort::init()
                .with_execution_providers([CUDA::default().build()])
                .commit()
            {
                tracing::info!(ep = "cuda", "ORT execution provider: CUDA");
            } else {
                tracing::warn!(ep = "cuda", "ORT CUDA EP unavailable — falling back to CPU");
            }
        }
        other => tracing::warn!(
            ep = other,
            "Unknown HEADROOM_ORT_EP — valid: cpu, openvino, cuda. Falling back to CPU"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hello_returns_crate_name() {
        assert_eq!(hello(), "headroom-core");
    }
}
