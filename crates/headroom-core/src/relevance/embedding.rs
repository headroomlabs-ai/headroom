//! Embedding-based relevance scorer using `fastembed-rs`.
//!
//! Uses BAAI/bge-small-en-v1.5 (33M params, 384 dims) by default —
//! same model the Python side runs via the `fastembed` package, giving
//! byte-equal embeddings on identical inputs. fastembed wraps ONNX
//! Runtime under the hood, with the runtime binary auto-downloaded
//! once at build time and the model weights auto-downloaded from
//! Hugging Face Hub on first use (~30 MB int8-quantized ONNX).
//!
//! # Caching
//!
//! Loading a sentence-transformer model takes ~1-2 seconds (HF Hub
//! call + ONNX session init). Construct the scorer once per process
//! and reuse — `try_new` returns a `Result` because the first
//! construction may need network access to fetch the model.
//!
//! When constructed, `is_available()` returns `true` and `HybridScorer`
//! switches off the BM25-fallback path automatically. If construction
//! fails (e.g. offline + model not cached), callers should fall back
//! to `HybridScorer::default()` which uses the stub-fallback scorer.
//!
//! # Output stability vs Python
//!
//! Both languages call into the same ONNX file via ONNX Runtime (`ort`
//! crate in Rust, `onnxruntime` package in Python's fastembed). Same
//! kernels, same weights — embeddings agree to floating-point
//! representation. Cosine similarity agrees to ~1e-6.

#[cfg(feature = "ml")]
use std::sync::Mutex;

#[cfg(feature = "ml")]
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

use super::base::{RelevanceScore, RelevanceScorer};

/// fastembed-backed semantic relevance scorer.
///
/// Construct via `EmbeddingScorer::try_new()` to handle the model-load
/// fallible step explicitly. `EmbeddingScorer::default()` is provided
/// for backwards compatibility but `is_available()` returns `false`
/// when the inner model failed to load (mimicking Python's
/// "sentence-transformers not installed" branch).
#[cfg(feature = "ml")]
pub struct EmbeddingScorer {
    pub model_name: String,
    /// `None` when model load failed — `is_available()` returns false
    /// and `score`/`score_batch` return empty scores. This lets
    /// `HybridScorer::default()` work even when the model can't be
    /// loaded (e.g. offline, no model cache).
    ///
    /// Wrapped in a `Mutex` because `TextEmbedding::embed` requires
    /// `&mut self` (the underlying ONNX session is single-threaded).
    /// Concurrent callers serialize on the inner lock, which is fine
    /// for the SmartCrusher hot path — embedding inference is the
    /// dominant cost so contention is bounded by inference latency,
    /// not lock latency.
    model: Option<Mutex<TextEmbedding>>,
}

#[cfg(feature = "ml")]
impl Default for EmbeddingScorer {
    /// Returns an unloaded scorer (model = None, is_available = false).
    ///
    /// Mirrors Python's "sentence-transformers not installed" branch:
    /// `HybridScorer::default()` constructs an EmbeddingScorer via
    /// `default()`, finds it unavailable, and uses BM25-fallback.
    ///
    /// To get a real, model-backed scorer call `try_new()` explicitly
    /// and pass it via `HybridScorer::with_scorers`. This separation
    /// keeps `Default` cheap (no I/O) and predictable in tests —
    /// otherwise model availability would depend on whether the user
    /// has previously cached the weights.
    fn default() -> Self {
        EmbeddingScorer {
            model_name: "BAAI/bge-small-en-v1.5".to_string(),
            model: None,
        }
    }
}

#[cfg(feature = "ml")]
impl EmbeddingScorer {
    /// Construct the scorer with the default model
    /// (BAAI/bge-small-en-v1.5). May trigger a one-time HF Hub
    /// download if the model isn't cached locally; subsequent calls
    /// are fast.
    ///
    /// Returns an error from fastembed if model initialization fails
    /// (network failure during download, missing ONNX runtime
    /// binaries, etc.).
    pub fn try_new() -> Result<Self, String> {
        Self::try_new_with_model(EmbeddingModel::BGESmallENV15)
    }

    /// Construct with an explicit model. See `fastembed::EmbeddingModel`
    /// for the catalog. The default `BGESmallENV15` is the best
    /// quality/speed tradeoff for compression-relevance scoring on
    /// short snippets.
    pub fn try_new_with_model(model_kind: EmbeddingModel) -> Result<Self, String> {
        // fastembed links the precompiled ONNX Runtime binary, which contains
        // AVX2 instructions on x86. Loading/running it on a non-AVX2 CPU traps
        // with SIGILL (issue #1723) — an uncatchable native fault. Bail early so
        // callers fall back to the BM25/stub path instead of killing the process.
        if !crate::onnx_cpu::onnx_runtime_supported_by_cpu() {
            return Err("EmbeddingScorer: ONNX Runtime backend requires AVX2 on \
                 this x86 CPU; embedding relevance disabled (falling back to BM25)"
                .to_string());
        }
        // The crate loads ONNX Runtime dynamically (`ort-load-dynamic`);
        // resolve and commit the dylib before fastembed touches ort — a
        // failed in-ort load deadlocks instead of erroring (see
        // `dynamic_ort_loader_ready`).
        crate::transforms::magika_detector::dynamic_ort_loader_ready()
            .map_err(|e| format!("EmbeddingScorer: ONNX Runtime unavailable: {e}"))?;
        let name = format!("{:?}", model_kind);
        let model = TextEmbedding::try_new(InitOptions::new(model_kind))
            .map_err(|e| format!("EmbeddingScorer model load failed: {}", e))?;
        Ok(EmbeddingScorer {
            model_name: name,
            model: Some(Mutex::new(model)),
        })
    }
}

#[cfg(feature = "ml")]
impl RelevanceScorer for EmbeddingScorer {
    fn score(&self, item: &str, context: &str) -> RelevanceScore {
        if item.is_empty() || context.is_empty() {
            return RelevanceScore::empty("Embedding: empty input");
        }
        let Some(model) = &self.model else {
            return RelevanceScore::empty("Embedding: model not available");
        };
        let mut guard = match model.lock() {
            Ok(g) => g,
            Err(_) => return RelevanceScore::empty("Embedding: lock poisoned"),
        };
        let embeddings = match guard.embed(vec![item.to_string(), context.to_string()], None) {
            Ok(e) => e,
            Err(e) => return RelevanceScore::empty(format!("Embedding: inference failed: {}", e)),
        };
        if embeddings.len() != 2 {
            return RelevanceScore::empty("Embedding: unexpected embedding count");
        }
        let sim = cosine_similarity(&embeddings[0], &embeddings[1]);
        RelevanceScore::new(
            sim,
            format!("Embedding: semantic similarity {:.2}", sim),
            Vec::new(),
        )
    }

    fn score_batch(&self, items: &[&str], context: &str) -> Vec<RelevanceScore> {
        if items.is_empty() {
            return Vec::new();
        }
        if context.is_empty() {
            return items
                .iter()
                .map(|_| RelevanceScore::empty("Embedding: empty context"))
                .collect();
        }
        let Some(model) = &self.model else {
            return items
                .iter()
                .map(|_| RelevanceScore::empty("Embedding: model not available"))
                .collect();
        };
        let mut guard = match model.lock() {
            Ok(g) => g,
            Err(_) => {
                return items
                    .iter()
                    .map(|_| RelevanceScore::empty("Embedding: lock poisoned"))
                    .collect();
            }
        };

        // Encode items + context in one batch — saves model dispatch
        // overhead. Mirrors Python fastembed batch encoding.
        let mut all_texts: Vec<String> = items.iter().map(|s| s.to_string()).collect();
        all_texts.push(context.to_string());
        let embeddings = match guard.embed(all_texts, None) {
            Ok(e) => e,
            Err(e) => {
                return items
                    .iter()
                    .map(|_| RelevanceScore::empty(format!("Embedding: inference failed: {}", e)))
                    .collect();
            }
        };
        if embeddings.len() != items.len() + 1 {
            return items
                .iter()
                .map(|_| RelevanceScore::empty("Embedding: unexpected embedding count"))
                .collect();
        }

        let context_emb = embeddings.last().unwrap().clone();
        embeddings
            .iter()
            .take(items.len())
            .map(|emb| {
                let sim = cosine_similarity(emb, &context_emb);
                RelevanceScore::new(sim, format!("Embedding: {:.2}", sim), Vec::new())
            })
            .collect()
    }

    fn is_available(&self) -> bool {
        self.model.is_some()
    }
}

/// Lexical-only build stub.
///
/// Without the `ml` feature the fastembed/ONNX backend is compiled out
/// entirely. `EmbeddingScorer` still exists so `HybridScorer` and
/// `create_scorer` compile unchanged, but it carries no model and is
/// permanently unavailable: `is_available()` is always `false` and the
/// scoring methods return the same empty scores the ml build produces
/// when its model failed to load. `HybridScorer` therefore takes its
/// BM25 fallback path exactly as it does when embeddings are stubbed.
#[cfg(not(feature = "ml"))]
pub struct EmbeddingScorer {
    pub model_name: String,
}

#[cfg(not(feature = "ml"))]
impl Default for EmbeddingScorer {
    fn default() -> Self {
        EmbeddingScorer {
            model_name: "BAAI/bge-small-en-v1.5".to_string(),
        }
    }
}

#[cfg(not(feature = "ml"))]
impl RelevanceScorer for EmbeddingScorer {
    fn score(&self, item: &str, context: &str) -> RelevanceScore {
        if item.is_empty() || context.is_empty() {
            return RelevanceScore::empty("Embedding: empty input");
        }
        RelevanceScore::empty("Embedding: model not available")
    }

    fn score_batch(&self, items: &[&str], context: &str) -> Vec<RelevanceScore> {
        if items.is_empty() {
            return Vec::new();
        }
        if context.is_empty() {
            return items
                .iter()
                .map(|_| RelevanceScore::empty("Embedding: empty context"))
                .collect();
        }
        items
            .iter()
            .map(|_| RelevanceScore::empty("Embedding: model not available"))
            .collect()
    }

    fn is_available(&self) -> bool {
        false
    }
}

/// Cosine similarity for two vectors. Clamped to `[0, 1]` since we
/// only care about positive similarity (mirrors Python `_cosine_similarity`).
///
/// Only the `ml` build calls this at runtime (from the fastembed-backed
/// scorer); the lexical-only build keeps it solely for the unit tests
/// that pin its numeric behavior.
#[cfg(any(feature = "ml", test))]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let mut dot: f64 = 0.0;
    let mut norm_a: f64 = 0.0;
    let mut norm_b: f64 = 0.0;
    for i in 0..a.len() {
        let av = a[i] as f64;
        let bv = b[i] as f64;
        dot += av * bv;
        norm_a += av * av;
        norm_b += bv * bv;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    let sim = dot / (norm_a.sqrt() * norm_b.sqrt());
    sim.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // The real-model tests are gated behind RUN_FASTEMBED_TESTS=1
    // since they require network access on first run (~30 MB model
    // download). Without the env var, only the offline-safe stub
    // path is exercised.

    #[cfg(feature = "ml")]
    fn fastembed_enabled() -> bool {
        std::env::var("RUN_FASTEMBED_TESTS").is_ok()
    }

    /// Construct a stub scorer with `model = None` for offline-safe
    /// tests of the unavailable-path behavior.
    #[cfg(feature = "ml")]
    fn unavailable_scorer() -> EmbeddingScorer {
        EmbeddingScorer {
            model_name: "test".to_string(),
            model: None,
        }
    }

    /// In the lexical-only build the scorer is always unavailable, so
    /// `default()` already gives the stub we want to exercise.
    #[cfg(not(feature = "ml"))]
    fn unavailable_scorer() -> EmbeddingScorer {
        EmbeddingScorer::default()
    }

    #[test]
    fn cosine_similarity_orthogonal_vectors() {
        let a = vec![1.0_f32, 0.0, 0.0, 0.0];
        let b = vec![0.0_f32, 1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn cosine_similarity_identical_vectors() {
        let v = vec![1.0_f32, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-9, "got {}", sim);
    }

    #[test]
    fn cosine_similarity_opposite_clamped_to_zero() {
        let a = vec![1.0_f32, 1.0];
        let b = vec![-1.0_f32, -1.0];
        // Raw cosine = -1.0; clamp to 0.0 since we only care about
        // positive similarity for relevance scoring.
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn cosine_similarity_zero_vector_returns_zero() {
        let zero = vec![0.0_f32; 4];
        let v = vec![1.0_f32, 2.0, 3.0, 4.0];
        assert_eq!(cosine_similarity(&zero, &v), 0.0);
        assert_eq!(cosine_similarity(&v, &zero), 0.0);
    }

    #[test]
    fn cosine_similarity_mismatched_dim_returns_zero() {
        let a = vec![1.0_f32, 2.0];
        let b = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    // ---------- offline-safe scorer behavior (no model needed) ----------

    #[test]
    fn unavailable_scorer_returns_empty_scores() {
        // Construct a scorer with model=None to simulate the offline
        // path. Default uses try_new which would download — bypass for
        // unit tests.
        let s = unavailable_scorer();
        assert!(!s.is_available());

        let r = s.score("item", "query");
        assert_eq!(r.score, 0.0);

        let batch = s.score_batch(&["a", "b", "c"], "query");
        assert_eq!(batch.len(), 3);
        for sc in batch {
            assert_eq!(sc.score, 0.0);
        }
    }

    #[test]
    fn unavailable_scorer_empty_inputs_short_circuit() {
        let s = unavailable_scorer();
        let r = s.score("", "query");
        assert_eq!(r.score, 0.0);
        assert!(r.reason.contains("empty"));
    }

    #[test]
    fn batch_with_empty_items_returns_empty_vec() {
        let s = unavailable_scorer();
        let r = s.score_batch(&[], "anything");
        assert!(r.is_empty());
    }

    // ---------- AVX2 CPU guard (issue #1723) ----------

    #[cfg(feature = "ml")]
    #[test]
    fn onnx_guard_matches_cpu_features() {
        let supported = crate::onnx_cpu::onnx_runtime_supported_by_cpu();
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        assert_eq!(supported, std::is_x86_feature_detected!("avx2"));
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        assert!(supported);
    }

    #[cfg(feature = "ml")]
    #[test]
    fn try_new_errors_on_unsupported_cpu_instead_of_sigill() {
        // On a no-AVX2 host the guard must turn the SIGILL into a plain Err
        // so callers fall back to BM25. On AVX2 CI runners the guard passes and
        // there is nothing to assert (loading the model would need network).
        if crate::onnx_cpu::onnx_runtime_supported_by_cpu() {
            return;
        }
        match EmbeddingScorer::try_new() {
            Err(err) => assert!(err.contains("AVX2"), "unexpected error: {err}"),
            Ok(_) => panic!("ONNX backend must not load on a no-AVX2 CPU"),
        }
    }

    // ---------- model-backed tests (gated on RUN_FASTEMBED_TESTS) ----------

    #[cfg(feature = "ml")]
    #[test]
    fn fastembed_loads_default_model() {
        if !fastembed_enabled() {
            return;
        }
        let s = EmbeddingScorer::try_new().expect("model loads");
        assert!(s.is_available());
        assert_eq!(s.model_name, "BGESmallENV15");
    }

    #[cfg(feature = "ml")]
    #[test]
    fn fastembed_semantic_match_outranks_unrelated() {
        if !fastembed_enabled() {
            return;
        }
        let s = EmbeddingScorer::try_new().expect("model loads");
        let related = s.score("authentication failed for user", "login error");
        let unrelated = s.score("the weather is nice today", "login error");
        assert!(
            related.score > unrelated.score,
            "semantically-related text should score higher: related={}, unrelated={}",
            related.score,
            unrelated.score
        );
    }

    #[cfg(feature = "ml")]
    #[test]
    fn fastembed_batch_returns_one_score_per_item() {
        if !fastembed_enabled() {
            return;
        }
        let s = EmbeddingScorer::try_new().expect("model loads");
        let items = ["foo", "bar", "baz"];
        let scores = s.score_batch(&items, "query text");
        assert_eq!(scores.len(), 3);
        for sc in scores {
            assert!((0.0..=1.0).contains(&sc.score));
        }
    }
}
