//! Kompress — Rust port of `headroom.transforms.kompress_compressor`.
//!
//! A ModernBERT token compressor for prose / plain-text tool outputs.
//! Where SmartCrusher/Log/Search/Diff are deterministic structural
//! compressors, Kompress is an **ML** compressor: it runs the trained
//! `chopratejas/kompress-v2-base` model (a fine-tune of
//! `answerdotai/ModernBERT-base` with a token keep/discard head + a
//! span-importance CNN head, exported to ONNX) and keeps only the words
//! the model scores as salient.
//!
//! # Model layering
//!
//! - **Inference weights:** `chopratejas/kompress-v2-base` — the ONNX
//!   artifact (`onnx/kompress-int8-wo.onnx`, weight-only int8 via the
//!   `com.microsoft` `MatMulNBits` contrib op; falls through to
//!   `onnx/kompress-fp32.onnx` then `onnx/kompress-int8.onnx`). This is
//!   *the model behind text compression*.
//! - **Tokenizer:** `answerdotai/ModernBERT-base`'s `tokenizer.json`.
//!   Kompress is a fine-tune of ModernBERT and reuses its exact vocab,
//!   so the kompress repo ships no tokenizer of its own.
//!
//! # ONNX contract
//!
//! Inputs `input_ids` + `attention_mask` (both `int64`, shape
//! `[batch, seq]`); output `final_scores` (`f32`, shape `[batch, seq]`)
//! — per-token salience in `[0, 1]` with the dual-head logic baked into
//! the graph. Keep decision is `score > 0.5`.
//!
//! # Compression path (mirrors the Python ONNX/proxy path exactly)
//!
//! 1. `words = content.split_whitespace()`. If `< 10` words → passthrough.
//! 2. For each `chunk_words`-sized (default 350) window of words:
//!    tokenize with the word list as **pre-tokenized** input
//!    (`is_split_into_words=True` in `transformers`), truncating to 512
//!    tokens; recover `input_ids` / `attention_mask` / `word_ids`.
//! 3. Run ONNX → `final_scores`. Reduce to **max score per word**.
//! 4. Keep word `w` (global index `w + chunk_start`) when its max score
//!    exceeds the threshold (default 0.5), or, when `target_ratio` is
//!    set, when it is in the top-`ratio` fraction by score.
//! 5. Emit the kept words, in original order, joined by single spaces.
//!
//! # Parity
//!
//! Byte-exact against the Python reference on the ONNX path: tokenizer
//! `input_ids`/`word_ids` reproduce `transformers` exactly, ONNX scores
//! match to ~1e-6 (far below the 0.5 threshold), and the kept-word set +
//! joined output match byte-for-byte. See
//! `tests/parity/fixtures/kompress/` and `KompressComparator` in
//! `crates/headroom-parity`.
//!
//! # CCR
//!
//! This engine returns the compressed string only. CCR offload of the
//! dropped words (so the model can retrieve the original on demand) is
//! handled by the live-zone dispatcher via [`crate::ccr::CcrStore`],
//! exactly as for the Search/Log/Diff compressors — not inside this
//! engine. The Python reference's inline `[N items compressed... hash=]`
//! marker is intentionally **not** reproduced; the Rust side uses the
//! canonical `<<ccr:HASH>>` marker convention.

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ort::session::Session;
use ort::value::Tensor;
use thiserror::Error;
use tokenizers::tokenizer::TruncationParams;
use tokenizers::{EncodeInput, InputSequence, Tokenizer};

// ─── Tunable defaults (parity-pinned to kompress-v2-base) ───────────────

/// HuggingFace repo holding the trained ONNX weights.
pub const DEFAULT_MODEL_ID: &str = "chopratejas/kompress-v2-base";
/// HuggingFace repo holding the tokenizer Kompress reuses.
pub const DEFAULT_TOKENIZER_REPO: &str = "answerdotai/ModernBERT-base";
/// Pinned commits of [`DEFAULT_MODEL_ID`] and [`DEFAULT_TOKENIZER_REPO`], the
/// snapshots the cache-only loader reads. Keep each equal to its entry in
/// `_PINNED_REVISIONS` (`headroom/onnx_runtime.py`), which is what the Python
/// side downloads; `tests/test_kompress_rust_pins.py` enforces it.
///
/// The same two commits live in [`crate::hf_pins`], which every download and
/// cache path resolves through; `hf_pins::tests::kompress_constants_agree_with
/// _the_pin_table` keeps the two from drifting. They are literals here because
/// the Python parity test reads this file's source text.
pub const DEFAULT_MODEL_REVISION: &str = "b1563631b35bfdcee37587ad530147497d820d4c";
pub const DEFAULT_TOKENIZER_REVISION: &str = "8949b909ec900327062f0ebf497f51aef5e6f0c8";
/// Words per inference chunk. Coupled to the model's training window;
/// kompress-v2-base was trained for 350.
pub const DEFAULT_CHUNK_WORDS: usize = 350;
/// Keep a word when its max per-token score exceeds this. Matches the
/// ONNX `get_keep_mask` hard-coded `> 0.5`.
pub const DEFAULT_SCORE_THRESHOLD: f32 = 0.5;
/// Inputs shorter than this many words pass through untouched — too
/// little signal for the model and the per-call cost dominates.
pub const MIN_WORDS: usize = 10;
/// Max ModernBERT sequence length per chunk (truncation bound).
pub const MAX_SEQ_LEN: usize = 512;

/// ONNX artifact candidates, tried in order. The first is a fp32 model whose
/// input shape is frozen to a static `[1, MAX_SEQ_LEN]` — required by the
/// OpenVINO **NPU** EP, which cannot compile dynamic `seq` (it hangs during
/// graph compilation on the dynamic-shape variants). When a static model is
/// loaded, `score_chunk` right-pads each chunk to its fixed length (detected
/// via [`detect_static_seq`]); dynamic models take the chunk's natural length
/// and pay no padding cost. The static model is absent from a vanilla install
/// (it is generated separately for NPU deployments), so this entry is simply
/// skipped on CPU/GPU. The remaining variants are the dynamic fall-throughs:
/// weight-only int8 (smallest; `MatMulNBits`, unsupported on NPU), then dynamic
/// fp32 (lossless reference), then the v1-era dynamic int8. A candidate is
/// skipped on download miss or on session-load failure.
pub const ONNX_CANDIDATES: &[&str] = &[
    "onnx/kompress-fp32-static512.onnx",
    "onnx/kompress-int8-wo.onnx",
    "onnx/kompress-fp32.onnx",
    "onnx/kompress-int8.onnx",
];

// ─── Types ──────────────────────────────────────────────────────────────

/// Configuration for [`Kompress`]. Field defaults match kompress-v2-base;
/// domain-specific models override `model_id` + `chunk_words` together.
#[derive(Debug, Clone)]
pub struct KompressConfig {
    pub model_id: String,
    pub tokenizer_repo: String,
    pub chunk_words: usize,
    pub score_threshold: f32,
    pub min_words: usize,
    /// Commit SHA to fetch `model_id` at. `None` uses the pinned revision from
    /// [`crate::hf_pins`] (or the default branch for an unpinned/custom repo).
    pub model_revision: Option<String>,
    /// Commit SHA to fetch `tokenizer_repo` at, same rules as
    /// [`Self::model_revision`].
    pub tokenizer_revision: Option<String>,
}

impl Default for KompressConfig {
    fn default() -> Self {
        Self {
            model_id: DEFAULT_MODEL_ID.to_string(),
            tokenizer_repo: DEFAULT_TOKENIZER_REPO.to_string(),
            chunk_words: DEFAULT_CHUNK_WORDS,
            score_threshold: DEFAULT_SCORE_THRESHOLD,
            min_words: MIN_WORDS,
            model_revision: None,
            tokenizer_revision: None,
        }
    }
}

/// Result of a Kompress compression. Mirrors the Python `KompressResult`
/// fields that the proxy path populates (CCR `cache_key` is owned by the
/// dispatcher, not this engine).
#[derive(Debug, Clone, PartialEq)]
pub struct KompressResult {
    pub compressed: String,
    pub original: String,
    /// Whitespace-split word count of the input.
    pub original_tokens: usize,
    /// Word count of the output.
    pub compressed_tokens: usize,
    /// `compressed_tokens / original_tokens`, computed in f64 to match the
    /// Python reference's `float` division bit-for-bit.
    pub compression_ratio: f64,
    pub model_used: String,
}

impl KompressResult {
    /// Words dropped (never negative).
    pub fn tokens_saved(&self) -> usize {
        self.original_tokens.saturating_sub(self.compressed_tokens)
    }

    /// True when nothing was compressed (output == input word stream).
    pub fn is_passthrough(&self) -> bool {
        self.compressed_tokens == self.original_tokens
    }
}

#[derive(Debug, Error)]
pub enum KompressError {
    #[error("failed to load tokenizer for `{repo}`: {source}")]
    Tokenizer {
        repo: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("failed to download `{repo}` from HuggingFace Hub: {source}")]
    Hub {
        repo: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("no loadable ONNX artifact in `{model_id}` (tried {tried:?}): {source}")]
    Onnx {
        model_id: String,
        tried: Vec<String>,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// `HEADROOM_OFFLINE` is set, so the Hub fetch was refused before any
    /// socket was opened. A separate variant from [`Self::Hub`] for the same
    /// reason as [`crate::tokenizer::HfTokenizerError::Offline`]: retrying or
    /// falling back on a policy refusal is just a slower refusal.
    #[error(transparent)]
    Offline(#[from] crate::offline::OfflineEgressBlocked),
}

// ─── Compressor ─────────────────────────────────────────────────────────

/// A loaded Kompress model + tokenizer. Construct once (model load is
/// expensive) and share; `compress` takes `&self`.
///
/// ONNX inference is serialized behind a `Mutex` — matching the Python
/// reference, which caps ONNX execution to one concurrent call (the CPU
/// provider does not parallelize the batch dimension for this model).
pub struct Kompress {
    config: KompressConfig,
    tokenizer: Tokenizer,
    session: Mutex<Session>,
    /// `Some(n)` when the loaded ONNX has a **fixed** sequence dimension (a
    /// static `[1, n]` input), in which case `score_chunk` right-pads every
    /// chunk to `n`. `None` for the usual dynamic-`seq` models, which take the
    /// chunk's natural length. Detected from the session's `input_ids` shape
    /// (see [`detect_static_seq`]). The static path exists for execution
    /// providers that cannot compile dynamic shapes (OpenVINO NPU); masked
    /// padding leaves the real-token scores unchanged, so output is identical.
    static_seq: Option<usize>,
}

impl std::fmt::Debug for Kompress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kompress")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

/// Inspect a built session's `input_ids` input: return `Some(n)` if its
/// sequence dimension is a fixed `n > 0` (a static-shape model), else `None`
/// (dynamic `seq`). ONNX inputs are `[batch, seq]`; a dynamic dim is reported
/// as `-1` by ONNX Runtime.
fn detect_static_seq(session: &Session) -> Option<usize> {
    let outlet = session.inputs().iter().find(|o| o.name() == "input_ids")?;
    let seq = *outlet.dtype().tensor_shape()?.get(1)?;
    (seq > 0).then_some(seq as usize)
}

impl Kompress {
    /// Wrap built artifacts into a `Kompress`, detecting whether the loaded
    /// model has a static sequence length (so `score_chunk` knows to pad).
    fn assemble(config: KompressConfig, tokenizer: Tokenizer, session: Session) -> Self {
        let static_seq = detect_static_seq(&session);
        Self {
            config,
            tokenizer,
            session: Mutex::new(session),
            static_seq,
        }
    }

    /// Build from local artifact paths — no network. Used by tests and
    /// the parity harness against the on-disk HuggingFace cache.
    pub fn from_files(
        tokenizer_path: impl AsRef<Path>,
        onnx_path: impl AsRef<Path>,
        config: KompressConfig,
    ) -> Result<Self, KompressError> {
        let tokenizer = load_tokenizer(tokenizer_path.as_ref(), &config.tokenizer_repo)?;
        let session = build_session(onnx_path.as_ref()).map_err(|e| KompressError::Onnx {
            model_id: config.model_id.clone(),
            tried: vec![onnx_path.as_ref().display().to_string()],
            source: e,
        })?;
        Ok(Self::assemble(config, tokenizer, session))
    }

    /// Build by resolving artifacts from the HuggingFace Hub (cache-first,
    /// downloading on miss). Blocking - call off the hot path. Tries the
    /// [`ONNX_CANDIDATES`] in order.
    ///
    /// Both the model repo and the tokenizer repo are fetched at their pinned
    /// commit (`crate::hf_pins`), not `main`. The ONNX weights and the
    /// tokenizer they were exported against are one artifact pair: floating
    /// either changes compression output with no change here (GH #4939).
    /// [`KompressConfig`] carries optional `model_revision`/`tokenizer_revision`
    /// overrides for a caller with its own snapshot.
    pub fn from_pretrained(config: KompressConfig) -> Result<Self, KompressError> {
        // Pinned commits, resolved first (GH #4939): `Api::model` hard-codes
        // `main`, so every fetch below names its revision.
        let model_revision =
            crate::hf_pins::resolve(&config.model_id, config.model_revision.as_deref());
        let tokenizer_revision =
            crate::hf_pins::resolve(&config.tokenizer_repo, config.tokenizer_revision.as_deref());

        // Warm-cache path, network-free, mirroring `HfTokenizer::from_pretrained_at`.
        // `hf-hub` resolves every revision through `refs/<revision>`, but
        // Python's `huggingface_hub` writes `snapshots/<sha>/` with no
        // `refs/<sha>`, so a Python-seeded pin was a miss here and re-downloaded.
        // Reading the snapshot directly first makes that a hit — and lets an
        // air-gapped host serve a pre-seeded pin instead of refusing.
        //
        // The tokenizer is resolved separately from the weights and carried
        // into the fallback: when it is cached but the weights are not, asking
        // `hf-hub` for the tokenizer again would turn a cache hit into a
        // download request, and a failing tokenizer endpoint would abort before
        // the weights were ever requested. A cached tokenizer is reused as-is.
        let cached = Self::cached_tokenizer(&config);
        if let Some(kompress) = Self::from_cached_pair(&config, cached.as_ref(), model_revision) {
            return Ok(kompress);
        }

        // Air-gap chokepoint, shared with the tokenizer and embedder:
        // `hf_cache::hub_api` runs the guard before it builds the client (the
        // `ureq` agent and endpoint resolution happen in that constructor), and
        // it reads `HF_ENDPOINT`, `HF_TOKEN` and the one Hub cache directory
        // Python uses. Plain `Api::new()` honours none of those, so it would
        // write into `~/.cache` while the cache-first read above looks
        // elsewhere — and ignore a mirror or a gated-repo token.
        //
        // A caller that has the files outside the Hub cache should use
        // [`Self::from_files`], which is network-free by construction.
        let Some(cache) = crate::hf_cache::hub_cache_dir() else {
            // Offline still wins: the caller treats `Offline` as final, and a
            // missing cache directory is no reason to report a Hub failure.
            crate::offline::guard_egress("Kompress model download", &config.model_id)?;
            return Err(KompressError::Hub {
                repo: config.model_id.clone(),
                source: "no HuggingFace cache directory (set HF_HUB_CACHE or HF_HOME)".into(),
            });
        };
        let api = crate::hf_cache::hub_api(&cache, "Kompress model download", &config.model_id)
            .map_err(|e| match e {
                crate::hf_cache::HubApiError::Offline(blocked) => KompressError::Offline(blocked),
                crate::hf_cache::HubApiError::Build(source) => KompressError::Hub {
                    repo: config.model_id.clone(),
                    source: Box::new(source),
                },
            })?;

        // Only when nothing usable came from the cache: a `hf-hub` fetch here
        // would re-download the tokenizer we already have on disk.
        let tokenizer = match cached {
            Some(tokenizer) => tokenizer,
            None => {
                let tokenizer_api = match tokenizer_revision {
                    Some(rev) => api.repo(hf_hub::Repo::with_revision(
                        config.tokenizer_repo.clone(),
                        hf_hub::RepoType::Model,
                        rev.to_string(),
                    )),
                    None => api.model(config.tokenizer_repo.clone()),
                };
                let tok_path =
                    tokenizer_api
                        .get("tokenizer.json")
                        .map_err(|e| KompressError::Hub {
                            repo: config.tokenizer_repo.clone(),
                            source: Box::new(e),
                        })?;
                load_tokenizer(&tok_path, &config.tokenizer_repo)?
            }
        };

        let model_api = match model_revision {
            Some(rev) => api.repo(hf_hub::Repo::with_revision(
                config.model_id.clone(),
                hf_hub::RepoType::Model,
                rev.to_string(),
            )),
            None => api.model(config.model_id.clone()),
        };
        let mut last_err: Option<Box<dyn std::error::Error + Send + Sync>> = None;
        let mut tried: Vec<String> = Vec::new();
        for candidate in ONNX_CANDIDATES {
            tried.push((*candidate).to_string());
            let onnx_path: PathBuf = match model_api.get(candidate) {
                Ok(p) => p,
                Err(e) => {
                    last_err = Some(Box::new(e));
                    continue;
                }
            };
            match build_session(&onnx_path) {
                Ok(session) => {
                    return Ok(Self::assemble(config, tokenizer, session));
                }
                Err(e) => {
                    last_err = Some(e);
                    continue;
                }
            }
        }
        Err(KompressError::Onnx {
            model_id: config.model_id.clone(),
            tried,
            source: last_err.unwrap_or_else(|| "no ONNX candidates configured".to_string().into()),
        })
    }

    /// Build from the Hub cache alone when **both** pinned artifacts are
    /// present: the tokenizer and one of the [`ONNX_CANDIDATES`]. Returns
    /// `Ok(None)`-shaped `None` otherwise, so the caller continues to the
    /// network path.
    ///
    /// Exactly one snapshot per repo — [`hf_cache_file_at`] — so a candidate
    /// that exists only under a different cached revision is a miss, and the
    /// tokenizer and weights always come from the same commit.
    ///
    /// `cached_tokenizer` is the tokenizer already loaded from that snapshot,
    /// loaded here so the caller can reuse it when the weights still have to be
    /// fetched.
    fn from_cached_pair(
        config: &KompressConfig,
        cached_tokenizer: Option<&Tokenizer>,
        model_revision: Option<&str>,
    ) -> Option<Self> {
        let tokenizer = cached_tokenizer?;
        for candidate in ONNX_CANDIDATES {
            let rel: Vec<&str> = candidate.split('/').collect();
            let Some(onnx_path) = hf_cache_file_at(&config.model_id, &rel, model_revision) else {
                continue;
            };
            if let Ok(session) = build_session(&onnx_path) {
                return Some(Self::assemble(config.clone(), tokenizer.clone(), session));
            }
        }
        None
    }

    /// The tokenizer for `config` from the Hub cache at its pin, if present.
    ///
    /// Loaded before the pair check so the fallback can reuse it: a cached
    /// tokenizer must never be re-fetched just because the weights were not
    /// cached, or a hit becomes a request that can fail first.
    fn cached_tokenizer(config: &KompressConfig) -> Option<Tokenizer> {
        let revision =
            crate::hf_pins::resolve(&config.tokenizer_repo, config.tokenizer_revision.as_deref());
        let path = hf_cache_file_at(&config.tokenizer_repo, &["tokenizer.json"], revision)?;
        load_tokenizer(&path, &config.tokenizer_repo).ok()
    }
    /// Cache-only construction: resolve the tokenizer + ONNX artifact from
    /// the local HuggingFace cache **without ever hitting the network**, and
    /// return `Ok(None)` when they are not present (or no candidate loads).
    ///
    /// This is the Rust mirror of the Python reference's
    /// `allow_download=False` path (`KompressModelNotCached` → defer): it lets
    /// the live-zone dispatcher attempt a load on a hot path without risking a
    /// blocking 261 MB download. When the model isn't cached the caller passes
    /// plain text through untouched, exactly as Python does when Kompress is
    /// unavailable.
    pub fn from_cache(config: KompressConfig) -> Result<Option<Self>, KompressError> {
        let Some(tok_path) = hf_cache_file_at(
            &config.tokenizer_repo,
            &["tokenizer.json"],
            config.tokenizer_revision.as_deref(),
        ) else {
            // Diagnostic: a `None` here is the #1 cause of a silent
            // `kompress_ready=false`. Name the repo, the cache and the revision
            // looked in so operators don't have to guess between "not
            // downloaded" and
            // "present but unreadable" (e.g. HF symlinks over `\\wsl$`, which
            // native Windows can't follow — `is_file()` returns false on
            // the unresolved symlink). Cache-only, so this is a defer, not an
            // error: the caller passes plain text through.
            tracing::warn!(
                event = "kompress_cache_miss",
                stage = "tokenizer",
                tokenizer_repo = %config.tokenizer_repo,
                revision = %cache_revision(&config.tokenizer_repo),
                hub_cache = ?crate::hf_cache::hub_cache_dir(),
                "Kompress deferred: tokenizer.json not found in the HF cache at the \
                 pinned revision (not downloaded yet — the Python side fetches the \
                 pin on its next Kompress load — or present but unreadable, e.g. HF \
                 symlinks over \\\\wsl$ which native Windows cannot follow)"
            );
            return Ok(None);
        };
        let tokenizer = load_tokenizer(&tok_path, &config.tokenizer_repo)?;
        let mut found_onnx = false;
        for candidate in ONNX_CANDIDATES {
            let rel: Vec<&str> = candidate.split('/').collect();
            let Some(onnx_path) =
                hf_cache_file_at(&config.model_id, &rel, config.model_revision.as_deref())
            else {
                continue;
            };
            found_onnx = true;
            match build_session(&onnx_path) {
                Ok(session) => {
                    return Ok(Some(Self::assemble(config, tokenizer, session)));
                }
                Err(e) => {
                    // The ONNX file is present but the session would not
                    // build — e.g. the active ORT execution provider rejects
                    // the graph (OpenVINO/NPU cannot compile the int8
                    // weight-only `MatMulNBits` op). Loudly surface it and try
                    // the next candidate (fp32) rather than die silently.
                    tracing::warn!(
                        event = "kompress_session_build_failed",
                        candidate = %candidate,
                        onnx_path = %onnx_path.display(),
                        error = %e,
                        "Kompress: ONNX found but session build failed; trying next candidate"
                    );
                }
            }
        }
        tracing::warn!(
            event = "kompress_cache_miss",
            stage = "onnx",
            model_id = %config.model_id,
            candidates = ?ONNX_CANDIDATES,
            any_onnx_found = found_onnx,
            revision = %cache_revision(&config.model_id),
            hub_cache = ?crate::hf_cache::hub_cache_dir(),
            "Kompress deferred: no usable ONNX session \
             (no candidate file in cache, or every candidate failed to build)"
        );
        Ok(None)
    }

    /// Model-decides compression (the proxy path): keep words scoring
    /// above `config.score_threshold`.
    pub fn compress(&self, content: &str) -> KompressResult {
        self.compress_inner(content, None)
    }

    /// Forced-ratio compression: keep the top `target_ratio` fraction of
    /// words by score (at least one). `None` defers to the threshold path.
    /// The proxy never sets this — only the user-facing API does.
    pub fn compress_with_ratio(&self, content: &str, target_ratio: Option<f64>) -> KompressResult {
        self.compress_inner(content, target_ratio)
    }

    fn compress_inner(&self, content: &str, target_ratio: Option<f64>) -> KompressResult {
        let words: Vec<&str> = content.split_whitespace().collect();
        let n_words = words.len();
        if n_words < self.config.min_words {
            return self.passthrough(content, n_words);
        }

        let mut kept_ids: BTreeSet<usize> = BTreeSet::new();
        let mut chunk_start = 0usize;
        while chunk_start < n_words {
            let end = (chunk_start + self.config.chunk_words).min(n_words);
            match self.score_chunk(&words[chunk_start..end]) {
                Ok(word_scores) => {
                    self.select_words(&word_scores, chunk_start, target_ratio, &mut kept_ids);
                }
                Err(_) => {
                    // A chunk that fails inference is treated as
                    // "nothing salient here" — matches the Python
                    // reference's per-call passthrough-on-error.
                    return self.passthrough(content, n_words);
                }
            }
            chunk_start += self.config.chunk_words;
        }

        if kept_ids.is_empty() {
            return self.passthrough(content, n_words);
        }

        let compressed_words: Vec<&str> = kept_ids
            .iter()
            .filter(|&&w| w < n_words)
            .map(|&w| words[w])
            .collect();
        let compressed_tokens = compressed_words.len();
        let compressed = compressed_words.join(" ");
        let compression_ratio = if n_words == 0 {
            1.0
        } else {
            compressed_tokens as f64 / n_words as f64
        };
        KompressResult {
            compressed,
            original: content.to_string(),
            original_tokens: n_words,
            compressed_tokens,
            compression_ratio,
            model_used: self.config.model_id.clone(),
        }
    }

    /// Tokenize one chunk of words and return the **max score per word**
    /// (`word_index -> score`). `word_index` is local to the chunk.
    fn score_chunk(
        &self,
        chunk_words: &[&str],
    ) -> Result<HashMap<usize, f32>, Box<dyn std::error::Error + Send + Sync>> {
        let seq_in: Vec<&str> = chunk_words.to_vec();
        let encoding = self
            .tokenizer
            .encode(EncodeInput::Single(InputSequence::from(seq_in)), true)?;
        let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let attn: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&x| x as i64)
            .collect();
        let word_ids = encoding.get_word_ids();
        let mut ids = ids;
        let mut attn = attn;

        // Static-shape models (e.g. the OpenVINO NPU build, which cannot
        // compile a dynamic `seq`) require a fixed `[1, static_seq]` input, so
        // right-pad every chunk to that length. Real tokens occupy
        // `0..real_seq`; the tail is padding with `attention_mask = 0`, which
        // masks those positions out of self-attention — the scores at real
        // positions are identical to an unpadded run, so keep/discard decisions
        // (hence parity) are unchanged. The tokenizer truncates to
        // `MAX_SEQ_LEN`, so the chunk never exceeds a `static_seq` of that size.
        // Dynamic models (`static_seq == None`) take the chunk's natural length
        // and pay no padding cost — the default for CPU/GPU.
        let seq = match self.static_seq {
            Some(n) => {
                debug_assert!(ids.len() <= n);
                ids.resize(n, 0);
                attn.resize(n, 0);
                n
            }
            None => ids.len(),
        };

        let input_ids = Tensor::from_array(([1usize, seq], ids))?;
        let attention_mask = Tensor::from_array(([1usize, seq], attn))?;

        let scores: Vec<f32> = {
            let mut session = self
                .session
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let outputs = session.run(ort::inputs![
                "input_ids" => input_ids,
                "attention_mask" => attention_mask
            ])?;
            let (_shape, data) = outputs["final_scores"].try_extract_tensor::<f32>()?;
            data.to_vec()
        };

        let mut word_scores: HashMap<usize, f32> = HashMap::new();
        for (idx, wid) in word_ids.iter().enumerate() {
            let Some(w) = wid else { continue };
            let Some(&s) = scores.get(idx) else { continue };
            let entry = word_scores.entry(*w as usize).or_insert(f32::MIN);
            if s > *entry {
                *entry = s;
            }
        }
        Ok(word_scores)
    }

    /// Apply the threshold or top-k rule to one chunk's per-word scores,
    /// inserting kept **global** word indices into `kept_ids`.
    fn select_words(
        &self,
        word_scores: &HashMap<usize, f32>,
        chunk_start: usize,
        target_ratio: Option<f64>,
        kept_ids: &mut BTreeSet<usize>,
    ) {
        if word_scores.is_empty() {
            return;
        }
        match target_ratio {
            Some(ratio) => {
                // Stable top-k: iterate words in ascending index order so
                // equal scores break toward the lower word index — this
                // matches CPython's stable `sorted()` over the
                // insertion-ordered score dict (tokens emitted in word
                // order).
                let mut ordered: Vec<(usize, f32)> =
                    word_scores.iter().map(|(&w, &s)| (w, s)).collect();
                ordered.sort_by_key(|&(w, _)| w);
                ordered.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let num_keep = ((ordered.len() as f64 * ratio) as usize).max(1);
                for &(w, _) in ordered.iter().take(num_keep) {
                    kept_ids.insert(w + chunk_start);
                }
            }
            None => {
                for (&w, &s) in word_scores {
                    if s > self.config.score_threshold {
                        kept_ids.insert(w + chunk_start);
                    }
                }
            }
        }
    }

    fn passthrough(&self, content: &str, n_words: usize) -> KompressResult {
        KompressResult {
            compressed: content.to_string(),
            original: content.to_string(),
            original_tokens: n_words,
            compressed_tokens: n_words,
            compression_ratio: 1.0,
            model_used: self.config.model_id.clone(),
        }
    }

    /// Expose the active config (read-only).
    pub fn config(&self) -> &KompressConfig {
        &self.config
    }
}

// ─── Loading helpers ────────────────────────────────────────────────────

fn load_tokenizer(path: &Path, repo: &str) -> Result<Tokenizer, KompressError> {
    let mut tokenizer = Tokenizer::from_file(path).map_err(|e| KompressError::Tokenizer {
        repo: repo.to_string(),
        source: e,
    })?;
    // Match the Python reference: truncation=True, max_length=512.
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: MAX_SEQ_LEN,
            ..Default::default()
        }))
        .map_err(|e| KompressError::Tokenizer {
            repo: repo.to_string(),
            source: e,
        })?;
    Ok(tokenizer)
}

/// Decision behind [`ort_session_preconditions`]: an ONNX Runtime session may
/// only be built when the CPU can run the precompiled binary (else `SIGILL`,
/// #1723) and the dynamic loader is pinned (else `ort` deadlocks, #1715).
/// The CPU check comes first and `loader` is only invoked when it passes:
/// the loader initializes the native runtime, which itself can `SIGILL` on
/// a CPU without AVX2 (same order as the magika guard). `Err` carries a
/// stable `reason` tag plus detail for logging.
pub(crate) fn ort_session_gate(
    cpu_ok: bool,
    loader: impl FnOnce() -> Result<(), String>,
) -> Result<(), (&'static str, String)> {
    if !cpu_ok {
        return Err((
            "cpu_unsupported",
            "ONNX Runtime backend requires AVX2 on this x86 CPU".to_string(),
        ));
    }
    loader().map_err(|e| ("ort_loader_unavailable", e))
}

/// Run the two guards the crate requires before any `ort` session is built
/// (same pair, same order, as magika and the embedding scorer).
pub(crate) fn ort_session_preconditions() -> Result<(), (&'static str, String)> {
    ort_session_gate(
        crate::onnx_cpu::onnx_runtime_supported_by_cpu(),
        crate::transforms::magika_detector::dynamic_ort_loader_ready,
    )
}

fn build_session(path: &Path) -> Result<Session, Box<dyn std::error::Error + Send + Sync>> {
    // Every Kompress constructor (from_files / from_pretrained / from_cache)
    // funnels through here, so guarding here covers all of them.
    ort_session_preconditions().map_err(|(reason, e)| format!("{reason}: {e}"))?;
    let session = Session::builder()?.commit_from_file(path)?;
    Ok(session)
}

/// Whether this process can build Kompress sessions: the same CPU-then-loader
/// preconditions every `Kompress` constructor checks, in the same order. The
/// CPU must run the precompiled ONNX Runtime (AVX2 on x86), and only then is
/// the library resolved (`ORT_DYLIB_PATH`, or the pip `onnxruntime` under
/// `VIRTUAL_ENV`, `CONDA_PREFIX` or `./.venv`) and committed, once per
/// process. `Err` is `"<reason>: <detail>"`. For harnesses that should skip,
/// not fail, on a machine that can't run Kompress.
pub fn onnx_runtime_ready() -> Result<(), String> {
    onnx_runtime_ready_with(
        crate::onnx_cpu::onnx_runtime_supported_by_cpu(),
        crate::transforms::magika_detector::dynamic_ort_loader_ready,
    )
}

/// [`onnx_runtime_ready`] with the CPU verdict and loader passed in, so tests
/// can check the loader never runs on an unsupported CPU.
fn onnx_runtime_ready_with(
    cpu_ok: bool,
    loader: impl FnOnce() -> Result<(), String>,
) -> Result<(), String> {
    ort_session_gate(cpu_ok, loader).map_err(|(reason, e)| format!("{reason}: {e}"))
}

/// The tokenizer and the ONNX artifact `onnx` (e.g.
/// `"onnx/kompress-int8-wo.onnx"`) that the cache-only loader reads for
/// `config`: the pinned snapshots in Python's hub cache, as
/// [`Kompress::from_cache`] finds them. `None` when either is missing. For
/// parity harnesses that load one specific artifact through
/// [`Kompress::from_files`], so they test the files production would load.
pub fn cached_artifacts(config: &KompressConfig, onnx: &str) -> Option<(PathBuf, PathBuf)> {
    let tokenizer = hf_cache_file_at(
        &config.tokenizer_repo,
        &["tokenizer.json"],
        config.tokenizer_revision.as_deref(),
    )?;
    let rel: Vec<&str> = onnx.split('/').collect();
    let model = hf_cache_file_at(&config.model_id, &rel, config.model_revision.as_deref())?;
    Some((tokenizer, model))
}

/// The revision of `repo` to load from the cache: its pin ([`crate::hf_pins`],
/// the same table the Python side and the download path use), otherwise `main`,
/// which is also what Python's `_resolve_revision` gives an unpinned repo.
/// `HEADROOM_HF_PIN=off` floats the pinned ones on `main` too.
fn cache_revision(repo: &str) -> &str {
    crate::hf_pins::revision(repo).unwrap_or("main")
}

/// [`cache_revision`] with a [`KompressConfig`] override honoured first, so a
/// caller that pointed the download at its own snapshot reads that snapshot
/// back instead of the default pin.
fn cache_revision_or<'a>(repo: &str, explicit: Option<&'a str>) -> &'a str {
    crate::hf_pins::resolve(repo, explicit).unwrap_or("main")
}

/// Resolve `rel` (e.g. `["tokenizer.json"]` or `["onnx", "kompress-int8-wo.onnx"]`)
/// in `repo`'s snapshot at [`cache_revision`], inside the one hub cache
/// directory Python's `huggingface_hub` would use
/// ([`crate::hf_cache::hub_cache_dir`]). Returns `None` if not present —
/// never touches the network.
///
/// One snapshot, not the first file found in any: with several revisions in
/// the cache (every pin bump leaves the old one behind), which one a
/// directory scan returned depended on filesystem order, and the tokenizer
/// and the ONNX file could come from different snapshots. And one cache, not
/// every root that exists: Python does not fall back to `~/.cache` when
/// `HF_HUB_CACHE` names another directory, so neither does this. `is_file`
/// follows the cache's symlinks, so a blob missing under a snapshot (or a
/// symlink native Windows cannot follow over `\\wsl$`) is a miss.
///
/// `explicit` is a [`KompressConfig`] revision override; `None` uses the pin.
fn hf_cache_file_at(repo: &str, rel: &[&str], explicit: Option<&str>) -> Option<PathBuf> {
    let cache = crate::hf_cache::hub_cache_dir()?;
    let mut path = crate::hf_cache::snapshot_path(&cache, repo, cache_revision_or(repo, explicit));
    for part in rel {
        path.push(part);
    }
    path.is_file().then_some(path)
}

/// [`hf_cache_file_at`] with no revision override. Kept for the tests, which
/// assert the pinned snapshot wins over a directory scan; production paths pass
/// a [`KompressConfig`] override so a caller's own snapshot is read back.
#[cfg_attr(not(test), allow(dead_code))]
fn hf_cache_file(repo: &str, rel: &[&str]) -> Option<PathBuf> {
    hf_cache_file_at(repo, rel, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hf_cache::tests::CacheEnv;

    /// Minimal valid `tokenizer.json`: WordLevel model, Whitespace
    /// pre-tokenizer, three-token vocabulary. Enough for `load_tokenizer` to
    /// produce a real `Tokenizer` without a multi-MB fixture.
    const TINY_TOKENIZER_JSON: &str = r#"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": {"hello": 0, "world": 1, "[UNK]": 2},
            "unk_token": "[UNK]"
        }
    }"#;

    /// Lay out `rel` under `repo`'s snapshot `commit` in the hub cache `cache`.
    fn cached_file(cache: &Path, repo: &str, commit: &str, rel: &str) -> PathBuf {
        let path = cache
            .join(format!("models--{}", repo.replace('/', "--")))
            .join("snapshots")
            .join(commit)
            .join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, commit).unwrap();
        path
    }

    /// [`cached_file`] with real `tokenizer.json` bytes, for a test that has to
    /// actually load the tokenizer rather than just resolve its path.
    fn cached_tokenizer_json(cache: &Path, repo: &str, commit: &str) -> PathBuf {
        let path = cache
            .join(format!("models--{}", repo.replace('/', "--")))
            .join("snapshots")
            .join(commit)
            .join("tokenizer.json");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, TINY_TOKENIZER_JSON).unwrap();
        path
    }

    /// The cache after a pin bump: the pinned snapshot sits between an older
    /// one and a newer `main`, and both other snapshots hold the same files.
    fn cache_with_three_snapshots(cache: &Path) {
        for repo in [DEFAULT_MODEL_ID, DEFAULT_TOKENIZER_REPO] {
            let pin = cache_revision(repo).to_string();
            for commit in [
                "0000000000000000000000000000000000000000",
                &pin,
                "ffffffffffffffffffffffffffffffffffffffff",
            ] {
                cached_file(cache, repo, commit, "tokenizer.json");
                cached_file(cache, repo, commit, ONNX_CANDIDATES[1]);
            }
            let refs = cache
                .join(format!("models--{}", repo.replace('/', "--")))
                .join("refs");
            std::fs::create_dir_all(&refs).unwrap();
            std::fs::write(
                refs.join("main"),
                "ffffffffffffffffffffffffffffffffffffffff",
            )
            .unwrap();
        }
    }

    /// Python loads the pinned snapshot of both repos; so must the cache-only
    /// loader, whatever order the filesystem lists the snapshots in.
    #[test]
    fn the_cache_lookup_reads_the_pinned_snapshot() {
        let _guard = crate::test_support::env_lock();
        let _env = CacheEnv::cleared_with(["HEADROOM_HF_PIN".to_string()]);
        let cache = tempfile::tempdir().unwrap();
        std::env::set_var("HF_HUB_CACHE", cache.path());
        cache_with_three_snapshots(cache.path());

        let tokenizer = hf_cache_file(DEFAULT_TOKENIZER_REPO, &["tokenizer.json"]).unwrap();
        assert!(tokenizer.ends_with(Path::new(DEFAULT_TOKENIZER_REVISION).join("tokenizer.json")));
        let rel: Vec<&str> = ONNX_CANDIDATES[1].split('/').collect();
        let onnx = hf_cache_file(DEFAULT_MODEL_ID, &rel).unwrap();
        assert!(onnx.ends_with(Path::new(DEFAULT_MODEL_REVISION).join(ONNX_CANDIDATES[1])));

        // A candidate the pinned snapshot lacks is a miss, even when another
        // snapshot has it.
        cached_file(
            cache.path(),
            DEFAULT_MODEL_ID,
            "ffffffffffffffffffffffffffffffffffffffff",
            ONNX_CANDIDATES[2],
        );
        let rel: Vec<&str> = ONNX_CANDIDATES[2].split('/').collect();
        assert_eq!(hf_cache_file(DEFAULT_MODEL_ID, &rel), None);
    }

    /// A Python-seeded pinned snapshot — `snapshots/<sha>/` with **no**
    /// `refs/<sha>` — is found by the pair lookup, so `from_pretrained` never
    /// reaches for the Hub.
    ///
    /// This covers the *selection* half of the warm-cache path: which snapshot
    /// `from_cached_pair` reads. It stops at the tokenizer lookup because a
    /// session needs the real 261 MB ONNX artifact; the tokenizer is the file
    /// `hf-hub`'s `refs/<sha>`-only resolution used to miss, so asserting it is
    /// found is the regression signal. The network half (that a hit returns
    /// without building a client) is covered for the tokenizer loader by
    /// `hf_impl::tests::from_pretrained_reads_a_python_style_sha_snapshot_without_a_ref`,
    /// which uses the same `hf_cache::snapshot_path` resolution.
    #[test]
    fn cached_pair_finds_a_python_style_snapshot_without_a_ref() {
        let _guard = crate::test_support::env_lock();
        let _env = CacheEnv::cleared_with(["HEADROOM_HF_PIN".to_string()]);
        let cache = tempfile::tempdir().unwrap();
        std::env::set_var("HF_HUB_CACHE", cache.path());

        let config = KompressConfig::default();
        // Both artifacts at their pins, Python layout, no refs/ entries at all.
        cached_file(
            cache.path(),
            DEFAULT_TOKENIZER_REPO,
            DEFAULT_TOKENIZER_REVISION,
            "tokenizer.json",
        );
        cached_file(
            cache.path(),
            DEFAULT_MODEL_ID,
            DEFAULT_MODEL_REVISION,
            ONNX_CANDIDATES[1],
        );
        let repo_dir = cache.path().join(format!(
            "models--{}",
            DEFAULT_TOKENIZER_REPO.replace('/', "--")
        ));
        assert!(!repo_dir
            .join("refs")
            .join(DEFAULT_TOKENIZER_REVISION)
            .exists());

        let tokenizer_revision = crate::hf_pins::resolve(&config.tokenizer_repo, None);
        let model_revision = crate::hf_pins::resolve(&config.model_id, None);
        let tok = hf_cache_file_at(
            &config.tokenizer_repo,
            &["tokenizer.json"],
            tokenizer_revision,
        )
        .expect("a Python-style pinned snapshot is a cache hit");
        assert!(tok.ends_with(Path::new(DEFAULT_TOKENIZER_REVISION).join("tokenizer.json")));
        let rel: Vec<&str> = ONNX_CANDIDATES[1].split('/').collect();
        let onnx = hf_cache_file_at(&config.model_id, &rel, model_revision)
            .expect("the weights at the pin are a cache hit");
        assert!(onnx.ends_with(Path::new(DEFAULT_MODEL_REVISION).join(ONNX_CANDIDATES[1])));
    }

    /// A tokenizer at the pin with the weights only under a *different*
    /// revision is not a usable pair: the weights must not be borrowed from
    /// another snapshot, so the warm-cache path declines and the network path
    /// is what would run.
    #[test]
    fn cached_pair_rejects_weights_from_a_different_revision() {
        let _guard = crate::test_support::env_lock();
        let _env = CacheEnv::cleared_with(["HEADROOM_HF_PIN".to_string()]);
        let cache = tempfile::tempdir().unwrap();
        std::env::set_var("HF_HUB_CACHE", cache.path());

        let config = KompressConfig::default();
        cached_file(
            cache.path(),
            DEFAULT_TOKENIZER_REPO,
            DEFAULT_TOKENIZER_REVISION,
            "tokenizer.json",
        );
        cached_file(
            cache.path(),
            DEFAULT_MODEL_ID,
            "ffffffffffffffffffffffffffffffffffffffff",
            ONNX_CANDIDATES[1],
        );

        let tokenizer_revision = crate::hf_pins::resolve(&config.tokenizer_repo, None);
        let model_revision = crate::hf_pins::resolve(&config.model_id, None);
        assert!(hf_cache_file_at(
            &config.tokenizer_repo,
            &["tokenizer.json"],
            tokenizer_revision
        )
        .is_some());
        let rel: Vec<&str> = ONNX_CANDIDATES[1].split('/').collect();
        assert_eq!(
            hf_cache_file_at(&config.model_id, &rel, model_revision),
            None,
            "weights at another revision must not satisfy the pair"
        );
    }

    /// Cached tokenizer + absent weights, downloads enabled: the fallback must
    /// **reuse** the cached tokenizer instead of asking the Hub for it.
    ///
    /// This is the case the review flagged. Seeding only the tokenizer makes
    /// `from_cached_pair` decline (no weights), so the loader takes the network
    /// path for the weights. If that path re-fetched the tokenizer, a Hub that
    /// cannot serve `tokenizer.json` would abort the load *before* the weights
    /// were ever requested — turning a partial cache into a hard failure.
    ///
    /// The stand-in endpoint answers tokenizer requests with 500 and records
    /// every path it is asked for. The assertions are about which paths were
    /// requested. Building the session needs the real 261 MB artifact, so the
    /// load is expected to fail — after the weight request, which is the
    /// ordering under test.
    #[test]
    fn cached_tokenizer_is_reused_when_only_the_weights_need_fetching() {
        use std::io::{Read, Write};
        use std::sync::{Arc, Mutex};

        let _guard = crate::test_support::env_lock();
        let _env = CacheEnv::cleared_with(
            [
                "HEADROOM_HF_PIN",
                "HF_TOKEN",
                "HUGGING_FACE_HUB_TOKEN",
                "HF_TOKEN_PATH",
                "HF_ENDPOINT",
            ]
            .map(String::from),
        );
        let cache = tempfile::tempdir().unwrap();
        std::env::set_var("HF_HUB_CACHE", cache.path());
        std::env::set_var("HOME", cache.path().join("home"));
        std::env::set_var("USERPROFILE", cache.path().join("home"));

        // Only the tokenizer is cached, at its pin, in Python's layout (no refs).
        cached_tokenizer_json(
            cache.path(),
            DEFAULT_TOKENIZER_REPO,
            DEFAULT_TOKENIZER_REVISION,
        );

        // Stand-in Hub: refuse tokenizer paths, 404 everything else, and record
        // every path asked for.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let port = listener.local_addr().unwrap().port();
        std::env::set_var("HF_ENDPOINT", format!("http://127.0.0.1:{port}"));
        let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let seen_in_server = Arc::clone(&seen);
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let done_in_server = Arc::clone(&done);
        let server = std::thread::spawn(move || {
            // Serve until the client side says it is finished, so the test takes
            // as long as the load, not as long as this window. The deadline is
            // only a safety net against a hang.
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while !done_in_server.load(std::sync::atomic::Ordering::Relaxed)
                && std::time::Instant::now() < deadline
            {
                let Ok((mut conn, _)) = listener.accept() else {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                    continue;
                };
                let mut buf = [0u8; 8192];
                let Ok(n) = conn.read(&mut buf) else { continue };
                let request = String::from_utf8_lossy(&buf[..n]).to_string();
                let path = request.split_whitespace().nth(1).unwrap_or("").to_string();
                seen_in_server.lock().unwrap().push(path.clone());
                let status = if path.contains("tokenizer.json") {
                    // A Hub that cannot serve the tokenizer must not be asked.
                    "500 Internal Server Error"
                } else {
                    "404 Not Found"
                };
                let _ = write!(
                    conn,
                    "HTTP/1.1 {status}\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
                );
            }
        });

        let outcome = Kompress::from_pretrained(KompressConfig::default());
        done.store(true, std::sync::atomic::Ordering::Relaxed);
        let paths = seen.lock().unwrap().clone();
        let _ = server.join();

        assert!(
            !paths.iter().any(|p| p.contains("tokenizer.json")),
            "the cached tokenizer was re-fetched instead of reused: {paths:?}"
        );
        assert!(
            paths.iter().any(|p| p.contains("kompress")),
            "the weight request never happened: {paths:?}"
        );
        assert!(
            outcome.is_err(),
            "no ONNX artifact is on disk, so the load must report an error"
        );
    }

    /// `HEADROOM_HF_PIN=off` floats on `main` through `refs/main`, and a repo
    /// Kompress does not pin (a domain model) always does, as in Python.
    #[test]
    fn unpinned_lookups_follow_refs_main() {
        let _guard = crate::test_support::env_lock();
        let _env = CacheEnv::cleared_with(["HEADROOM_HF_PIN".to_string()]);
        let cache = tempfile::tempdir().unwrap();
        std::env::set_var("HF_HUB_CACHE", cache.path());
        cache_with_three_snapshots(cache.path());
        let newest = Path::new("ffffffffffffffffffffffffffffffffffffffff").join("tokenizer.json");

        std::env::set_var("HEADROOM_HF_PIN", "off");
        let tokenizer = hf_cache_file(DEFAULT_TOKENIZER_REPO, &["tokenizer.json"]).unwrap();
        assert!(tokenizer.ends_with(&newest));

        std::env::remove_var("HEADROOM_HF_PIN");
        let domain = "acme/kompress-legal";
        cached_file(
            cache.path(),
            domain,
            "ffffffffffffffffffffffffffffffffffffffff",
            "tokenizer.json",
        );
        let refs = cache
            .path()
            .join("models--acme--kompress-legal")
            .join("refs");
        std::fs::create_dir_all(&refs).unwrap();
        std::fs::write(
            refs.join("main"),
            "ffffffffffffffffffffffffffffffffffffffff",
        )
        .unwrap();
        assert!(hf_cache_file(domain, &["tokenizer.json"])
            .unwrap()
            .ends_with(&newest));
    }

    /// The Kompress lookup now names the directory Python would, including
    /// the cases the old root list missed: the legacy `HUGGINGFACE_HUB_CACHE`,
    /// `XDG_CACHE_HOME` and a `${VAR}` reference. And it reads only that one
    /// directory: a copy under `HOME` is not used when `HF_HUB_CACHE` points
    /// elsewhere.
    #[test]
    fn the_cache_lookup_uses_the_one_directory_python_uses() {
        let _guard = crate::test_support::env_lock();
        let _env = CacheEnv::cleared_with([
            "HEADROOM_HF_PIN".to_string(),
            "KOMPRESS_TEST_ROOT".to_string(),
        ]);
        let t = tempfile::tempdir().unwrap();
        let pin = cache_revision(DEFAULT_TOKENIZER_REPO).to_string();
        let lookup = || hf_cache_file(DEFAULT_TOKENIZER_REPO, &["tokenizer.json"]);
        std::env::set_var("HOME", t.path().join("home"));
        std::env::set_var("USERPROFILE", t.path().join("home"));

        let legacy = t.path().join("legacy");
        let want = cached_file(&legacy, DEFAULT_TOKENIZER_REPO, &pin, "tokenizer.json");
        std::env::set_var("HUGGINGFACE_HUB_CACHE", &legacy);
        assert_eq!(lookup(), Some(want));
        std::env::remove_var("HUGGINGFACE_HUB_CACHE");

        let xdg = t.path().join("xdg");
        let want = cached_file(
            &xdg.join("huggingface").join("hub"),
            DEFAULT_TOKENIZER_REPO,
            &pin,
            "tokenizer.json",
        );
        std::env::set_var("XDG_CACHE_HOME", &xdg);
        assert_eq!(lookup(), Some(want));
        std::env::remove_var("XDG_CACHE_HOME");

        let root = t.path().join("root");
        let want = cached_file(
            &root.join("hub"),
            DEFAULT_TOKENIZER_REPO,
            &pin,
            "tokenizer.json",
        );
        std::env::set_var("KOMPRESS_TEST_ROOT", &root);
        std::env::set_var("HF_HUB_CACHE", "${KOMPRESS_TEST_ROOT}/hub");
        assert_eq!(lookup(), Some(want));

        cached_file(
            &t.path()
                .join("home")
                .join(".cache")
                .join("huggingface")
                .join("hub"),
            DEFAULT_TOKENIZER_REPO,
            &pin,
            "tokenizer.json",
        );
        std::env::set_var("HF_HUB_CACHE", t.path().join("empty"));
        assert_eq!(lookup(), None);
    }

    #[test]
    fn ort_session_gate_rejects_unsupported_cpu_and_loader_failure() {
        assert_eq!(ort_session_gate(true, || Ok(())), Ok(()));
        let (reason, _) = ort_session_gate(false, || Ok(())).unwrap_err();
        assert_eq!(reason, "cpu_unsupported");
        let (reason, e) = ort_session_gate(true, || Err("no dylib".into())).unwrap_err();
        assert_eq!((reason, e.as_str()), ("ort_loader_unavailable", "no dylib"));
    }

    #[test]
    fn ort_session_gate_never_invokes_the_loader_on_an_unsupported_cpu() {
        // The loader initializes the native runtime, which can SIGILL on a CPU
        // without AVX2, so it must not run once the CPU check has failed.
        let called = std::cell::Cell::new(false);
        let out = ort_session_gate(false, || {
            called.set(true);
            Ok(())
        });
        assert_eq!(out.unwrap_err().0, "cpu_unsupported");
        assert!(!called.get(), "loader ran although the CPU check failed");
        // ...and it is invoked exactly when the CPU is fine.
        let called = std::cell::Cell::new(false);
        let _ = ort_session_gate(true, || {
            called.set(true);
            Ok(())
        });
        assert!(called.get());
    }

    #[test]
    fn onnx_runtime_ready_checks_the_cpu_before_the_loader() {
        // The public readiness check (used by harnesses to skip) must keep
        // the constructors' order: on an unsupported CPU it reports the CPU
        // reason without initializing the native runtime.
        let called = std::cell::Cell::new(false);
        let out = onnx_runtime_ready_with(false, || {
            called.set(true);
            Ok(())
        });
        let message = out.unwrap_err();
        assert!(message.starts_with("cpu_unsupported: "), "{message}");
        assert!(!called.get(), "loader ran although the CPU check failed");

        assert_eq!(
            onnx_runtime_ready_with(true, || Err("no dylib".into())),
            Err("ort_loader_unavailable: no dylib".to_string())
        );
        assert_eq!(onnx_runtime_ready_with(true, || Ok(())), Ok(()));
    }

    /// `from_pretrained` must refuse before a Hub client exists when the
    /// operator air-gapped the box.
    ///
    /// The model id cannot exist, which is what makes this a real assertion
    /// rather than a tautology: without the guard the call reaches `hf-hub`,
    /// dials `huggingface.co` and comes back a `Hub` error. Only the guard can
    /// produce `Offline`, and only by returning before `hf_cache::hub_api`
    /// builds the client — so `Offline` *is* the "no socket was opened"
    /// assertion here.
    #[test]
    fn from_pretrained_refuses_while_offline() {
        let _guard = crate::test_support::env_lock();
        std::env::set_var(crate::offline::OFFLINE_ENV, "1");
        let config = KompressConfig {
            model_id: "headroomlabs-ai/this-model-does-not-exist-a2".to_string(),
            ..KompressConfig::default()
        };
        let result = Kompress::from_pretrained(config);
        std::env::remove_var(crate::offline::OFFLINE_ENV);

        match result {
            Err(KompressError::Offline(blocked)) => {
                assert_eq!(blocked.purpose, "Kompress model download");
                assert_eq!(
                    blocked.destination,
                    "headroomlabs-ai/this-model-does-not-exist-a2"
                );
            }
            Err(other) => panic!("expected an offline refusal, got a network error: {other}"),
            Ok(_) => panic!("expected an offline refusal, got a loaded model"),
        }
    }

    #[test]
    fn config_defaults_match_kompress_v2_base() {
        let c = KompressConfig::default();
        assert_eq!(c.model_id, "chopratejas/kompress-v2-base");
        assert_eq!(c.tokenizer_repo, "answerdotai/ModernBERT-base");
        assert_eq!(c.chunk_words, 350);
        assert_eq!(c.score_threshold, 0.5);
        assert_eq!(c.min_words, 10);
    }

    #[test]
    fn result_helpers() {
        let r = KompressResult {
            compressed: "a b".into(),
            original: "a b c d".into(),
            original_tokens: 4,
            compressed_tokens: 2,
            compression_ratio: 0.5,
            model_used: DEFAULT_MODEL_ID.into(),
        };
        assert_eq!(r.tokens_saved(), 2);
        assert!(!r.is_passthrough());

        let p = KompressResult {
            compressed: "a b".into(),
            original: "a b".into(),
            original_tokens: 2,
            compressed_tokens: 2,
            compression_ratio: 1.0,
            model_used: DEFAULT_MODEL_ID.into(),
        };
        assert_eq!(p.tokens_saved(), 0);
        assert!(p.is_passthrough());
    }
}
