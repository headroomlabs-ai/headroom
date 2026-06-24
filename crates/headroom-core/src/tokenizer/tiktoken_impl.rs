//! `tiktoken-rs` adapter implementing [`Tokenizer`].
//!
//! `tiktoken-rs` and Python `tiktoken` use the same BPE merge tables; for the
//! same model and same input, this returns byte-identical token IDs and
//! therefore byte-identical token *counts*. This is what makes the parity
//! tests "byte-equal" rather than "approximate".
//!
//! Initialization (loading the BPE table) is non-trivial. Each encoding is
//! built lazily on first use and shared via `LazyLock<Arc<CoreBPE>>`, so the
//! first `for_model` call pays the cost and every subsequent call is cheap.

use std::sync::{Arc, LazyLock};

use thiserror::Error;
use tiktoken_rs::CoreBPE;

use super::{Backend, Tokenizer};

#[derive(Debug, Error)]
pub enum TiktokenError {
    /// We don't know which encoding `model` should use. The caller can fall
    /// back to estimation; the registry handles that automatically.
    #[error("unknown encoding for model `{0}`")]
    UnknownEncoding(String),
}

/// Lazy-built shared BPE for the four named encodings. Init failure here would
/// indicate `tiktoken-rs` itself is broken; we treat that as a programmer error
/// and panic.
static O200K: LazyLock<Arc<CoreBPE>> =
    LazyLock::new(|| Arc::new(tiktoken_rs::o200k_base().expect("o200k_base init")));
static CL100K: LazyLock<Arc<CoreBPE>> =
    LazyLock::new(|| Arc::new(tiktoken_rs::cl100k_base().expect("cl100k_base init")));
static P50K: LazyLock<Arc<CoreBPE>> =
    LazyLock::new(|| Arc::new(tiktoken_rs::p50k_base().expect("p50k_base init")));
static R50K: LazyLock<Arc<CoreBPE>> =
    LazyLock::new(|| Arc::new(tiktoken_rs::r50k_base().expect("r50k_base init")));

/// BPE token counter for OpenAI / o-series models.
pub struct TiktokenCounter {
    model: String,
    encoding_name: &'static str,
    bpe: Arc<CoreBPE>,
}

impl std::fmt::Debug for TiktokenCounter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TiktokenCounter")
            .field("model", &self.model)
            .field("encoding", &self.encoding_name)
            .finish()
    }
}

impl TiktokenCounter {
    /// Build a counter for `model`. Returns `UnknownEncoding` if the model
    /// doesn't fall into any of the supported BPE families.
    pub fn for_model(model: &str) -> Result<Self, TiktokenError> {
        let encoding_name = encoding_for(model)?;
        let bpe = match encoding_name {
            "o200k_base" => O200K.clone(),
            "cl100k_base" => CL100K.clone(),
            "p50k_base" => P50K.clone(),
            "r50k_base" => R50K.clone(),
            // unreachable: encoding_for only returns the four names above.
            _ => return Err(TiktokenError::UnknownEncoding(model.to_string())),
        };
        Ok(Self {
            model: model.to_string(),
            encoding_name,
            bpe,
        })
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn encoding_name(&self) -> &'static str {
        self.encoding_name
    }
}

impl Tokenizer for TiktokenCounter {
    fn count_text(&self, text: &str) -> usize {
        if text.is_empty() {
            // Match Python `TiktokenCounter.count_text`: short-circuit empty.
            return 0;
        }
        // For ORDINARY input (no literal special-token strings) `encode_ordinary`
        // here and `encoding.encode(text)` in Python yield identical token IDs
        // and counts — that's the byte-equality the parity harness verifies.
        //
        // Divergence (rare in practice): if `text` contains a literal
        // `<|endoftext|>` (or any other special-token string), Python's default
        // `encode` raises (because `disallowed_special="all"`) while we treat
        // it as ordinary text. We chose tolerance over panic since proxy users
        // can legitimately send those substrings; document for future readers.
        self.bpe.encode_ordinary(text).len()
    }

    fn backend(&self) -> Backend {
        Backend::Tiktoken
    }
}

/// Map model → encoding name. Mirrors `MODEL_ENCODINGS` and the prefix
/// fallbacks in `headroom/tokenizers/tiktoken_counter.py`.
fn encoding_for(model: &str) -> Result<&'static str, TiktokenError> {
    let m = model.to_ascii_lowercase();

    // o200k_base: GPT-4o + o1/o3 reasoning families.
    if m.starts_with("gpt-4o") || m.starts_with("o1") || m.starts_with("o3") {
        return Ok("o200k_base");
    }

    // cl100k_base: GPT-4, GPT-3.5-turbo, embeddings.
    if m.starts_with("gpt-4") || m.starts_with("gpt-3.5") || m.starts_with("text-embedding") {
        return Ok("cl100k_base");
    }

    // p50k_base: code-* and the davinci-002/003 text-completion line.
    if m.starts_with("code-")
        || m.starts_with("text-davinci-002")
        || m.starts_with("text-davinci-003")
    {
        return Ok("p50k_base");
    }

    // r50k_base: legacy davinci-001 and earlier completion families.
    if m.starts_with("text-davinci")
        || m.starts_with("davinci")
        || m.starts_with("curie")
        || m.starts_with("babbage")
        || m.starts_with("ada")
    {
        return Ok("r50k_base");
    }

    Err(TiktokenError::UnknownEncoding(model.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_string_is_zero() {
        let t = TiktokenCounter::for_model("gpt-4o-mini").unwrap();
        assert_eq!(t.count_text(""), 0);
    }

    #[test]
    fn nonempty_text_is_at_least_one_token() {
        let t = TiktokenCounter::for_model("gpt-4o-mini").unwrap();
        assert!(t.count_text("a") >= 1);
    }

    #[test]
    fn known_token_counts_for_o200k() {
        // These constants are the o200k_base BPE token counts produced by
        // both Python `tiktoken.encoding_for_model("gpt-4o-mini")` and
        // `tiktoken-rs::o200k_base()` for the given strings. They lock in
        // byte-equal parity so a future tiktoken-rs upgrade that subtly
        // changes BPE behavior would fail this test.
        let t = TiktokenCounter::for_model("gpt-4o-mini").unwrap();
        assert_eq!(t.count_text("hello"), 1);
        assert_eq!(t.count_text("Hello, world!"), 4);
        assert_eq!(
            t.count_text("the quick brown fox jumps over the lazy dog"),
            9
        );
    }

    #[test]
    fn determinism() {
        let t = TiktokenCounter::for_model("gpt-4o").unwrap();
        let s = "Determinism check across many calls.";
        let first = t.count_text(s);
        for _ in 0..1000 {
            assert_eq!(t.count_text(s), first);
        }
    }

    #[test]
    fn unicode_input_does_not_panic() {
        let t = TiktokenCounter::for_model("gpt-4o-mini").unwrap();
        // Each call should produce a reasonable count (>=1 for non-empty),
        // not panic, and not return absurd values.
        for s in [
            "héllo wörld",        // accented
            "你好世界",           // CJK
            "مرحبا بالعالم",      // Arabic (RTL)
            "🦀 ferris the crab", // emoji
            "\n\t\r\x07",         // control chars
        ] {
            let n = t.count_text(s);
            assert!(n >= 1, "{s:?}");
            assert!(n < s.len() * 4 + 10, "absurd count {n} for {s:?}");
        }
    }

    #[test]
    fn very_long_input() {
        let t = TiktokenCounter::for_model("gpt-4o-mini").unwrap();
        let s = "the quick brown fox ".repeat(50_000); // ~1MB
        let n = t.count_text(&s);
        // 50k repeats * ~5 tokens per repeat = ~250k tokens, sanity bound.
        assert!(n > 100_000 && n < 1_000_000, "n={n}");
    }

    #[test]
    fn encoding_dispatch() {
        for (model, expected) in [
            ("gpt-4o", "o200k_base"),
            ("gpt-4o-mini", "o200k_base"),
            ("gpt-4o-2024-08-06", "o200k_base"),
            ("o1-preview", "o200k_base"),
            ("o3-mini", "o200k_base"),
            ("gpt-4", "cl100k_base"),
            ("gpt-4-turbo", "cl100k_base"),
            ("gpt-3.5-turbo", "cl100k_base"),
            ("text-embedding-3-small", "cl100k_base"),
            ("code-davinci-002", "p50k_base"),
            ("text-davinci-002", "p50k_base"),
            ("text-davinci-003", "p50k_base"),
            ("text-davinci-001", "r50k_base"),
            ("davinci", "r50k_base"),
            ("curie", "r50k_base"),
            ("babbage", "r50k_base"),
            ("ada", "r50k_base"),
        ] {
            let t = TiktokenCounter::for_model(model)
                .unwrap_or_else(|e| panic!("for_model({model}) failed: {e}"));
            assert_eq!(t.encoding_name(), expected, "{model}");
        }
    }

    #[test]
    fn unknown_model_returns_error() {
        let r = TiktokenCounter::for_model("claude-3-opus");
        assert!(matches!(r, Err(TiktokenError::UnknownEncoding(_))));
    }

    #[test]
    fn case_insensitive_dispatch() {
        let t = TiktokenCounter::for_model("GPT-4o-Mini").unwrap();
        assert_eq!(t.encoding_name(), "o200k_base");
    }

    #[test]
    fn shared_bpe_instances() {
        // Two counters for the same encoding should share the underlying BPE
        // (same Arc), proving the LazyLock cache works.
        let a = TiktokenCounter::for_model("gpt-4o").unwrap();
        let b = TiktokenCounter::for_model("gpt-4o-mini").unwrap();
        assert!(Arc::ptr_eq(&a.bpe, &b.bpe));
    }

    #[test]
    fn backend_is_tiktoken() {
        let t = TiktokenCounter::for_model("gpt-4o-mini").unwrap();
        assert_eq!(t.backend(), Backend::Tiktoken);
    }
}
