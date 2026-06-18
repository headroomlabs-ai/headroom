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

/// ONNX artifact candidates, tried in order: weight-only int8 (smallest,
/// fp32-equivalent quality), then fp32 (lossless reference), then the
/// v1-era dynamic int8. A candidate is skipped on download miss or on
/// session-load failure (e.g. an onnxruntime build without the 8-bit
/// `MatMulNBits` kernel falls through to fp32).
pub const ONNX_CANDIDATES: &[&str] = &[
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
}

impl Default for KompressConfig {
    fn default() -> Self {
        Self {
            model_id: DEFAULT_MODEL_ID.to_string(),
            tokenizer_repo: DEFAULT_TOKENIZER_REPO.to_string(),
            chunk_words: DEFAULT_CHUNK_WORDS,
            score_threshold: DEFAULT_SCORE_THRESHOLD,
            min_words: MIN_WORDS,
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
}

impl std::fmt::Debug for Kompress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kompress")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl Kompress {
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
        Ok(Self {
            config,
            tokenizer,
            session: Mutex::new(session),
        })
    }

    /// Build by resolving artifacts from the HuggingFace Hub (cache-first,
    /// downloading on miss). Blocking — call off the hot path. Tries the
    /// [`ONNX_CANDIDATES`] in order.
    pub fn from_pretrained(config: KompressConfig) -> Result<Self, KompressError> {
        let api = hf_hub::api::sync::Api::new().map_err(|e| KompressError::Hub {
            repo: config.model_id.clone(),
            source: Box::new(e),
        })?;

        let tok_path = api
            .model(config.tokenizer_repo.clone())
            .get("tokenizer.json")
            .map_err(|e| KompressError::Hub {
                repo: config.tokenizer_repo.clone(),
                source: Box::new(e),
            })?;
        let tokenizer = load_tokenizer(&tok_path, &config.tokenizer_repo)?;

        let model_api = api.model(config.model_id.clone());
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
                    return Ok(Self {
                        config,
                        tokenizer,
                        session: Mutex::new(session),
                    });
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
            source: last_err
                .unwrap_or_else(|| "no ONNX candidates configured".to_string().into()),
        })
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
        let seq = ids.len();

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
                ordered.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
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

fn build_session(path: &Path) -> Result<Session, Box<dyn std::error::Error + Send + Sync>> {
    let session = Session::builder()?.commit_from_file(path)?;
    Ok(session)
}

#[cfg(test)]
mod tests {
    use super::*;

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
