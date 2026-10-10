//! Embedding-based relevance scorer using `fastembed-rs`.
//!
//! Uses BAAI/bge-small-en-v1.5 (33M params, 384 dims) by default, loaded
//! from the same pinned `Qdrant/bge-small-en-v1.5-onnx-Q` snapshot the
//! Python side's `fastembed` package runs (~67 MB quantized ONNX). fastembed
//! wraps ONNX Runtime under the hood, with the runtime library loaded
//! dynamically and the model files downloaded from Hugging Face Hub on
//! first use.
//!
//! # Caching
//!
//! Loading a sentence-transformer model takes ~1-2 seconds (HF Hub
//! call + ONNX session init). Construct the scorer once per process
//! and reuse — `try_new` returns a `Result` because the first
//! construction may need network access to fetch the model. The files
//! live in the standard HuggingFace hub cache (`HF_HUB_CACHE`,
//! `HF_HOME/hub`, `~/.cache/huggingface/hub`), shared with Python.
//!
//! When constructed, `is_available()` returns `true` and `HybridScorer`
//! switches off the BM25-fallback path automatically. If construction
//! fails (e.g. offline + model not cached), callers should fall back
//! to `HybridScorer::default()` which uses the stub-fallback scorer.
//!
//! # Output stability vs Python
//!
//! Both languages run the same ONNX file and tokenizer through ONNX
//! Runtime (`ort` crate in Rust, `onnxruntime` package in Python's
//! fastembed), with the same CLS pooling and normalization. On the same
//! ONNX Runtime build they agree within 1e-6 per component; across ORT
//! versions they drift by up to ~4e-4, so they are close, not byte-equal.
//! The gated parity fixture in `tests/parity/fixtures/embedding/` bounds
//! that drift on both sides (see `tests/parity/record_embedding.py`).

#[cfg(feature = "ml")]
use std::path::{Path, PathBuf};
#[cfg(feature = "ml")]
use std::sync::Mutex;

#[cfg(feature = "ml")]
use fastembed::{
    EmbeddingModel, InitOptions, InitOptionsUserDefined, Pooling, QuantizationMode, TextEmbedding,
    TokenizerFiles, UserDefinedEmbeddingModel,
};

use super::base::{RelevanceScore, RelevanceScorer};

/// Name the default model reports, the same string Python uses.
#[cfg(feature = "ml")]
const DEFAULT_MODEL_NAME: &str = "BAAI/bge-small-en-v1.5";

/// HF repo the default model loads from: the one Python's fastembed
/// resolves `BAAI/bge-small-en-v1.5` to. fastembed-rs maps its own
/// `BGESmallENV15` to `Xenova/bge-small-en-v1.5`, a different (fp32) export,
/// and always on `main`, so the default is loaded as a user-defined model
/// from this repo instead. Keep equal to `DEFAULT_MODEL_REPO` in
/// `headroom/relevance/embedding.py`.
#[cfg(feature = "ml")]
const DEFAULT_MODEL_REPO: &str = "Qdrant/bge-small-en-v1.5-onnx-Q";

/// Pinned commit of [`DEFAULT_MODEL_REPO`]. Keep equal to its entry in
/// `_PINNED_REVISIONS` (`headroom/onnx_runtime.py`);
/// `tests/test_fastembed_pinned_snapshot.py` enforces both equalities.
#[cfg(feature = "ml")]
const DEFAULT_MODEL_REVISION: &str = "52398278842ec682c6f32300af41344b1c0b0bb2";

#[cfg(feature = "ml")]
const DEFAULT_MODEL_FILE: &str = "model_optimized.onnx";

/// Every file the load reads: the ONNX graph, then what `TokenizerFiles`
/// needs. Same list as Python's `DEFAULT_MODEL_FILES`.
#[cfg(feature = "ml")]
const DEFAULT_MODEL_FILES: [&str; 5] = [
    DEFAULT_MODEL_FILE,
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
];

/// The revision to load: the pin, or `main` when `HEADROOM_HF_PIN` turns
/// pinning off (same values as Python's `onnx_runtime._resolve_revision`).
#[cfg(feature = "ml")]
fn default_model_revision() -> &'static str {
    let pin = std::env::var("HEADROOM_HF_PIN").unwrap_or_default();
    match pin.trim().to_ascii_lowercase().as_str() {
        "off" | "0" | "false" | "no" => "main",
        _ => DEFAULT_MODEL_REVISION,
    }
}

/// Find a cached snapshot of [`DEFAULT_MODEL_REPO`] at `revision` that holds
/// every file in [`DEFAULT_MODEL_FILES`]. Never touches the network.
///
/// Looks the snapshot up by directory rather than through `hf_hub::Cache`:
/// that resolves every revision through `refs/<revision>`, which Python's
/// `huggingface_hub` does not write for a commit SHA, so a snapshot Python
/// downloaded would be missed. A branch name still goes through its ref.
#[cfg(feature = "ml")]
fn cached_snapshot(roots: &[PathBuf], revision: &str) -> Option<PathBuf> {
    let repo_dir = format!("models--{}", DEFAULT_MODEL_REPO.replace('/', "--"));
    roots.iter().find_map(|root| {
        let repo = root.join(&repo_dir);
        let commit = std::fs::read_to_string(repo.join("refs").join(revision))
            .map(|c| c.trim().to_string())
            .unwrap_or_else(|_| revision.to_string());
        let snapshot = repo.join("snapshots").join(commit);
        // `is_file` follows the snapshot's symlinks, so blobs deleted from
        // under it count as a miss rather than a load failure.
        DEFAULT_MODEL_FILES
            .iter()
            .all(|name| snapshot.join(name).is_file())
            .then_some(snapshot)
    })
}

/// The Hub handle for [`DEFAULT_MODEL_REPO`] at `revision`, caching under
/// `cache_root`. Honors `HF_ENDPOINT` like the Python side does.
#[cfg(feature = "ml")]
fn default_model_repo(
    cache_root: &Path,
    revision: &str,
) -> Result<hf_hub::api::sync::ApiRepo, String> {
    // Air-gap chokepoint, ahead of the client: building it resolves the Hub
    // endpoint and the `ureq` agent, and `hf-hub` downloads on a cache miss
    // even with `HF_HUB_OFFLINE` set. Callers reach this only on a cache
    // miss, so a pre-seeded air-gapped host still loads.
    //
    // The refusal is soft by design: every caller degrades to the BM25
    // scorer on `Err`, so an air-gapped box loses embedding relevance instead
    // of failing a request. The message names the switch so the operator can
    // tell the causes apart in the log.
    crate::offline::guard_egress("fastembed embedding model download", DEFAULT_MODEL_REPO)
        .map_err(|e| format!("EmbeddingScorer model load refused: {e}"))?;
    let mut builder =
        hf_hub::api::sync::ApiBuilder::from_cache(hf_hub::Cache::new(cache_root.to_path_buf()))
            .with_progress(false);
    if let Ok(endpoint) = std::env::var("HF_ENDPOINT") {
        builder = builder.with_endpoint(endpoint);
    }
    let api = builder
        .build()
        .map_err(|e| format!("EmbeddingScorer: HuggingFace client init failed: {e}"))?;
    Ok(api.repo(hf_hub::Repo::with_revision(
        DEFAULT_MODEL_REPO.to_string(),
        hf_hub::RepoType::Model,
        revision.to_string(),
    )))
}

/// Download every file of the snapshot at `revision` through `repo` (from
/// [`default_model_repo`]) and return the snapshot directory.
#[cfg(feature = "ml")]
fn download_snapshot(repo: &hf_hub::api::sync::ApiRepo, revision: &str) -> Result<PathBuf, String> {
    let mut snapshot: Option<PathBuf> = None;
    for name in DEFAULT_MODEL_FILES {
        let path = repo.get(name).map_err(|e| {
            format!(
                "EmbeddingScorer: downloading {DEFAULT_MODEL_REPO}/{name}@{revision} failed: {e}"
            )
        })?;
        let dir = path
            .parent()
            .map(Path::to_path_buf)
            .ok_or_else(|| format!("EmbeddingScorer: {} has no parent", path.display()))?;
        match &snapshot {
            None => snapshot = Some(dir),
            Some(first) if *first == dir => {}
            Some(first) => {
                return Err(format!(
                    "EmbeddingScorer: {DEFAULT_MODEL_REPO} files resolved to more than one \
                     snapshot: {} and {}",
                    first.display(),
                    dir.display()
                ))
            }
        }
    }
    snapshot.ok_or_else(|| "EmbeddingScorer: no model files configured".to_string())
}

/// Load the snapshot in `dir` with the settings fastembed uses for this
/// model (`BGESmallENV15Q` in fastembed-rs, the Python default): CLS
/// pooling, static quantization, 512-token truncation.
#[cfg(feature = "ml")]
fn load_snapshot(dir: &Path) -> Result<TextEmbedding, String> {
    let read = |name: &str| {
        std::fs::read(dir.join(name))
            .map_err(|e| format!("EmbeddingScorer: reading {}: {e}", dir.join(name).display()))
    };
    let tokenizer_files = TokenizerFiles {
        tokenizer_file: read("tokenizer.json")?,
        config_file: read("config.json")?,
        special_tokens_map_file: read("special_tokens_map.json")?,
        tokenizer_config_file: read("tokenizer_config.json")?,
    };
    let model = UserDefinedEmbeddingModel::new(read(DEFAULT_MODEL_FILE)?, tokenizer_files)
        .with_pooling(Pooling::Cls)
        .with_quantization(QuantizationMode::Static);
    TextEmbedding::try_new_from_user_defined(model, InitOptionsUserDefined::new())
        .map_err(|e| format!("EmbeddingScorer model load failed: {e}"))
}

/// Refuse before ONNX Runtime is touched on a host that cannot run it.
#[cfg(feature = "ml")]
fn onnx_runtime_ready() -> Result<(), String> {
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
        .map_err(|e| format!("EmbeddingScorer: ONNX Runtime unavailable: {e}"))
}

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
    /// (BAAI/bge-small-en-v1.5) from its pinned snapshot. May trigger a
    /// one-time HF Hub download if the snapshot isn't cached locally;
    /// subsequent calls are fast.
    ///
    /// Returns an error if model initialization fails (refused while
    /// offline, network failure during download, missing ONNX runtime
    /// binaries, etc.).
    pub fn try_new() -> Result<Self, String> {
        Self::try_new_pinned(&crate::transforms::kompress::hf_hub_roots())
    }

    /// [`Self::try_new`] against explicit HF hub cache roots, searched in
    /// order; a download lands in the first. Split out so tests can point it
    /// at a temporary cache.
    fn try_new_pinned(roots: &[PathBuf]) -> Result<Self, String> {
        let revision = default_model_revision();
        let snapshot = match cached_snapshot(roots, revision) {
            Some(dir) => {
                onnx_runtime_ready()?;
                dir
            }
            None => {
                let root = roots.first().ok_or_else(|| {
                    "EmbeddingScorer: no HuggingFace cache directory (set HF_HOME)".to_string()
                })?;
                // Built (and so guarded) ahead of the AVX2 and ort-loader
                // probes, so an air-gapped box reports the policy refusal
                // rather than whichever local prerequisite was missing too.
                let repo = default_model_repo(root, revision)?;
                onnx_runtime_ready()?;
                download_snapshot(&repo, revision)?
            }
        };
        let model = load_snapshot(&snapshot)?;
        Ok(EmbeddingScorer {
            model_name: DEFAULT_MODEL_NAME.to_string(),
            model: Some(Mutex::new(model)),
        })
    }

    /// Construct with an explicit model from fastembed-rs's catalog (see
    /// `fastembed::EmbeddingModel`). Unlike [`Self::try_new`], these are not
    /// pinned: fastembed resolves them on the repo's `main`. Note that
    /// `BGESmallENV15` here is the `Xenova` fp32 export, not the snapshot
    /// Python runs; use `try_new` for the default model.
    pub fn try_new_with_model(model_kind: EmbeddingModel) -> Result<Self, String> {
        let name = format!("{:?}", model_kind);
        // Air-gap chokepoint, FIRST thing in the function. `TextEmbedding::
        // try_new` resolves the model's ONNX weights through `hf-hub`, which
        // downloads from huggingface.co on a cache miss even with
        // `HF_HUB_OFFLINE` set, and offers no cache-only attempt to put the
        // guard behind, so this refuses even a cached model.
        crate::offline::guard_egress("fastembed embedding model download", &name)
            .map_err(|e| format!("EmbeddingScorer model load refused: {e}"))?;
        onnx_runtime_ready()?;
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
    // since they require network access on first run (~67 MB model
    // download). Without the env var, only the offline-safe stub
    // path is exercised.

    #[cfg(feature = "ml")]
    fn fastembed_enabled() -> bool {
        std::env::var("RUN_FASTEMBED_TESTS").is_ok()
    }

    /// Lay out a fake HF cache snapshot of the default repo at `commit`,
    /// holding `files` (dummy bytes), and return the snapshot directory.
    #[cfg(feature = "ml")]
    fn fake_snapshot(root: &Path, commit: &str, files: &[&str]) -> PathBuf {
        let snapshot = root
            .join(format!("models--{}", DEFAULT_MODEL_REPO.replace('/', "--")))
            .join("snapshots")
            .join(commit);
        std::fs::create_dir_all(&snapshot).unwrap();
        for name in files {
            std::fs::write(snapshot.join(name), b"not a model").unwrap();
        }
        snapshot
    }

    /// A cold cache must refuse before anything resolves the model, and say
    /// so in the returned message rather than looking like an ordinary load
    /// failure — the caller only ever sees the `String`.
    ///
    /// Not gated on `RUN_FASTEMBED_TESTS`: the whole point is that no network
    /// call happens, so this is safe to run in CI. If the guard were removed
    /// this test would either download ~67 MB or fail with a load error, and
    /// neither says "refused".
    #[cfg(feature = "ml")]
    #[test]
    fn try_new_refuses_while_offline() {
        let cache = tempfile::tempdir().unwrap();
        let _guard = crate::test_support::env_lock();
        std::env::set_var(crate::offline::OFFLINE_ENV, "1");
        let result = EmbeddingScorer::try_new_pinned(&[cache.path().to_path_buf()]);
        std::env::remove_var(crate::offline::OFFLINE_ENV);

        let err = result.err().expect("guard must refuse while offline");
        assert!(err.contains("refused"), "{err}");
        assert!(err.contains(crate::offline::OFFLINE_ENV), "{err}");
        assert!(err.contains("fastembed embedding model download"), "{err}");
    }

    /// A pre-seeded snapshot is the air-gapped setup: the guard sits on the
    /// cache miss only, so a cached snapshot gets past it. The dummy ONNX
    /// bytes then fail to load (or ORT is missing), which is fine; what
    /// matters is that the failure is not the refusal.
    #[cfg(feature = "ml")]
    #[test]
    fn a_cached_snapshot_is_not_refused_offline() {
        let cache = tempfile::tempdir().unwrap();
        fake_snapshot(cache.path(), DEFAULT_MODEL_REVISION, &DEFAULT_MODEL_FILES);
        let _guard = crate::test_support::env_lock();
        let pin = std::env::var_os("HEADROOM_HF_PIN");
        std::env::remove_var("HEADROOM_HF_PIN");
        std::env::set_var(crate::offline::OFFLINE_ENV, "1");
        let result = EmbeddingScorer::try_new_pinned(&[cache.path().to_path_buf()]);
        std::env::remove_var(crate::offline::OFFLINE_ENV);
        if let Some(pin) = pin {
            std::env::set_var("HEADROOM_HF_PIN", pin);
        }

        let err = result
            .err()
            .expect("dummy model bytes can never load into a session");
        assert!(!err.contains("refused"), "{err}");
    }

    #[cfg(feature = "ml")]
    #[test]
    fn cached_snapshot_finds_the_pinned_commit() {
        let cache = tempfile::tempdir().unwrap();
        let snapshot = fake_snapshot(cache.path(), DEFAULT_MODEL_REVISION, &DEFAULT_MODEL_FILES);
        let roots = [cache.path().join("empty-root"), cache.path().to_path_buf()];
        assert_eq!(
            cached_snapshot(&roots, DEFAULT_MODEL_REVISION),
            Some(snapshot)
        );
    }

    /// The regression this pin exists for: a snapshot of whatever `main`
    /// pointed to must not satisfy the pinned lookup.
    #[cfg(feature = "ml")]
    #[test]
    fn cached_snapshot_ignores_other_commits() {
        let cache = tempfile::tempdir().unwrap();
        let main_commit = "aa8f8b060edb00e03bfdd08813a2949946c8ba55";
        let snapshot = fake_snapshot(cache.path(), main_commit, &DEFAULT_MODEL_FILES);
        let refs = snapshot.parent().unwrap().parent().unwrap().join("refs");
        std::fs::create_dir_all(&refs).unwrap();
        std::fs::write(refs.join("main"), main_commit).unwrap();
        let roots = [cache.path().to_path_buf()];

        assert_eq!(cached_snapshot(&roots, DEFAULT_MODEL_REVISION), None);
        // `main` still resolves through its ref, for HEADROOM_HF_PIN=off.
        assert_eq!(cached_snapshot(&roots, "main"), Some(snapshot));
    }

    #[cfg(feature = "ml")]
    #[test]
    fn cached_snapshot_needs_every_file() {
        let cache = tempfile::tempdir().unwrap();
        fake_snapshot(
            cache.path(),
            DEFAULT_MODEL_REVISION,
            &DEFAULT_MODEL_FILES[..DEFAULT_MODEL_FILES.len() - 1],
        );
        assert_eq!(
            cached_snapshot(&[cache.path().to_path_buf()], DEFAULT_MODEL_REVISION),
            None
        );
    }

    /// The download asks the Hub for the pinned commit. This is what Python's
    /// fastembed silently dropped; building the URL makes no network call.
    #[cfg(feature = "ml")]
    #[test]
    fn download_requests_the_pinned_revision() {
        let cache = tempfile::tempdir().unwrap();
        let _guard = crate::test_support::env_lock();
        std::env::remove_var(crate::offline::OFFLINE_ENV);
        let repo = default_model_repo(cache.path(), DEFAULT_MODEL_REVISION).unwrap();
        let url = repo.url(DEFAULT_MODEL_FILE);
        assert!(
            url.ends_with(&format!(
                "/{DEFAULT_MODEL_REPO}/resolve/{DEFAULT_MODEL_REVISION}/{DEFAULT_MODEL_FILE}"
            )),
            "{url}"
        );
    }

    #[cfg(feature = "ml")]
    #[test]
    fn pin_can_be_disabled() {
        let _guard = crate::test_support::env_lock();
        let pin = std::env::var_os("HEADROOM_HF_PIN");
        std::env::set_var("HEADROOM_HF_PIN", " OFF ");
        let disabled = default_model_revision();
        std::env::remove_var("HEADROOM_HF_PIN");
        let pinned = default_model_revision();
        if let Some(pin) = pin {
            std::env::set_var("HEADROOM_HF_PIN", pin);
        }
        assert_eq!(disabled, "main");
        assert_eq!(pinned, DEFAULT_MODEL_REVISION);
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
        assert_eq!(s.model_name, "BAAI/bge-small-en-v1.5");
    }

    /// Rust half of the cross-language parity check: the pinned model must
    /// reproduce the embeddings Python recorded (tests/parity/record_embedding.py)
    /// within the fixture's tolerance. The Python half is
    /// tests/test_fastembed_pinned_snapshot.py; both comparing against one
    /// fixture bounds the drift between the two languages.
    #[cfg(feature = "ml")]
    #[test]
    fn fastembed_matches_parity_fixture() {
        if !fastembed_enabled() {
            return;
        }
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/parity/fixtures/embedding/bge_small_en_v15_pinned.json");
        let fixture: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).expect("fixture readable"))
                .expect("fixture is JSON");
        assert_eq!(fixture["repo"], DEFAULT_MODEL_REPO);
        assert_eq!(fixture["revision"], DEFAULT_MODEL_REVISION);
        let max_abs_diff = fixture["max_abs_diff"].as_f64().expect("max_abs_diff");
        let texts: Vec<String> = fixture["texts"]
            .as_array()
            .expect("texts")
            .iter()
            .map(|t| t.as_str().expect("text").to_string())
            .collect();
        let expected: Vec<Vec<f64>> = fixture["embeddings"]
            .as_array()
            .expect("embeddings")
            .iter()
            .map(|row| {
                row.as_array()
                    .expect("row")
                    .iter()
                    .map(|v| v.as_f64().expect("float"))
                    .collect()
            })
            .collect();

        let s = EmbeddingScorer::try_new().expect("model loads");
        let got = s
            .model
            .as_ref()
            .unwrap()
            .lock()
            .unwrap()
            .embed(texts.clone(), None)
            .expect("embed");
        assert_eq!(got.len(), expected.len());
        for ((text, got), expected) in texts.iter().zip(&got).zip(&expected) {
            assert_eq!(got.len(), expected.len(), "{text}");
            let worst = got
                .iter()
                .zip(expected)
                .map(|(g, e)| (*g as f64 - e).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                worst <= max_abs_diff,
                "{text:?}: max |rust - fixture| = {worst:e} > {max_abs_diff:e}"
            );
        }
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
