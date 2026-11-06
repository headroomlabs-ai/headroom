//! Shared CPU-capability guard for the precompiled ONNX Runtime binary.
//!
//! All ONNX entry points in this crate — Magika content detection
//! ([`crate::transforms::magika_detector`]), the embedding relevance
//! scorer ([`crate::relevance::EmbeddingScorer`]) and the Kompress
//! compressor ([`crate::transforms::kompress`]) — load the same
//! precompiled ONNX Runtime binary at run time (`ort-load-dynamic`).
//!
//! On x86/x86_64 that binary contains AVX2-family instructions. Executing
//! it on a CPU without AVX2 (common inside Docker / QEMU / older cloud VMs)
//! traps with `SIGILL` — a hardware fault that native code cannot turn into
//! a catchable exception, so the whole host process dies (issue #1723).
//!
//! Call this up front and skip the ONNX path when it returns `false`, so
//! callers fall back to non-ONNX behavior instead of crashing.

/// `true` if this CPU can run the precompiled ONNX Runtime binary.
///
/// On x86/x86_64 this requires AVX2. On non-x86 targets the AVX2 gate does
/// not apply and this always returns `true`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) fn onnx_runtime_supported_by_cpu() -> bool {
    std::is_x86_feature_detected!("avx2")
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub(crate) fn onnx_runtime_supported_by_cpu() -> bool {
    true
}
