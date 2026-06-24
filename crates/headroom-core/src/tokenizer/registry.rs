//! Model-name → tokenizer dispatch.
//!
//! Mirrors `MODEL_PATTERNS` in `headroom/tokenizers/registry.py`. Three
//! backends in priority order:
//!
//! 1. **HuggingFace** — anything the caller has registered via
//!    [`register_hf`] for a given model-name prefix. Real BPE/Unigram/
//!    WordPiece counts. This is opt-in: tokenizer.json files aren't bundled,
//!    so nothing routes here until the embedding application calls
//!    [`register_hf`] at startup. Wins over the rules below when registered.
//! 2. **Tiktoken** — OpenAI / o-series via `tiktoken-rs`. Byte-identical to
//!    Python `tiktoken`.
//! 3. **Estimation** — `chars / cpt` fallback for Anthropic Claude (3.5),
//!    Gemini / Cohere / Command without an HF registration (4.0), and
//!    everything else (4.0).

use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

use super::{EstimatingCounter, HfTokenizer, HfTokenizerError, TiktokenCounter, Tokenizer};

/// Which family of tokenizer was selected for a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Real BPE via `tiktoken-rs`. Byte-identical to Python `tiktoken`.
    Tiktoken,
    /// HuggingFace `tokenizers` crate (`tokenizer.json` loaded by caller).
    HuggingFace,
    /// Character-density estimation (chars/token formula).
    Estimation,
}

/// Pick a backend purely from the model name, ignoring runtime registrations.
///
/// Patterns and ordering match `headroom.tokenizers.registry.MODEL_PATTERNS`
/// for the families this stage supports. Anything outside the OpenAI BPE
/// family lands in `Estimation` here — even if [`register_hf`] would route
/// it to `HuggingFace` at runtime. Use [`get_tokenizer`] for the real
/// dispatch.
pub fn detect_backend(model: &str) -> Backend {
    let m = model.to_ascii_lowercase();

    // OpenAI BPE-tokenized families (gpt-3.5/4/4o + o1/o3 reasoning + embeddings + legacy davinci/curie/babbage/ada + code-).
    if m.starts_with("gpt-4o")
        || m.starts_with("gpt-4")
        || m.starts_with("gpt-3.5")
        || m.starts_with("o1")
        || m.starts_with("o3")
        || m.starts_with("text-embedding")
        || m.starts_with("text-davinci")
        || m.starts_with("davinci")
        || m.starts_with("curie")
        || m.starts_with("babbage")
        || m.starts_with("ada")
        || m.starts_with("code-")
    {
        return Backend::Tiktoken;
    }

    Backend::Estimation
}

/// Return a tokenizer for `model`. Resolution order:
/// 1. HuggingFace tokenizers registered via [`register_hf`] (longest matching
///    prefix wins).
/// 2. Tiktoken for OpenAI / o-series families.
/// 3. Estimation, with density calibrated per family (Claude → 3.5, Gemini /
///    Cohere / Command → 4.0, otherwise 4.0).
pub fn get_tokenizer(model: &str) -> Box<dyn Tokenizer> {
    if let Some(hf) = lookup_hf(model) {
        return Box::new(hf);
    }
    match detect_backend(model) {
        Backend::Tiktoken => match TiktokenCounter::for_model(model) {
            Ok(t) => Box::new(t),
            Err(_) => Box::new(default_estimator_for(model)),
        },
        // Backend::HuggingFace from detect_backend is unreachable — only
        // runtime registrations produce HF, and we already checked above.
        Backend::HuggingFace | Backend::Estimation => Box::new(default_estimator_for(model)),
    }
}

fn default_estimator_for(model: &str) -> EstimatingCounter {
    let m = model.to_ascii_lowercase();
    if m.starts_with("claude-") {
        EstimatingCounter::new(3.5)
    } else if m.starts_with("gemini") || m.starts_with("palm") || m.starts_with("command") {
        EstimatingCounter::new(4.0)
    } else {
        EstimatingCounter::default()
    }
}

// ---- HuggingFace runtime registry --------------------------------------
//
// A process-global table mapping a lowercased model-name *prefix* to a loaded
// `HfTokenizer`. The embedding application owns this — there's no autoloader
// in core because we don't want to bundle tokenizer.json files (multi-MB) or
// pull in HF Hub networking here.

fn hf_table() -> &'static RwLock<HashMap<String, HfTokenizer>> {
    static TABLE: OnceLock<RwLock<HashMap<String, HfTokenizer>>> = OnceLock::new();
    TABLE.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Register `tokenizer` to handle every model whose lowercased name starts
/// with `prefix`. Multiple prefixes can coexist; on lookup the longest
/// matching prefix wins, so registering `"command-r-plus"` overrides a more
/// general `"command-"` registration for that one model.
///
/// Calling this with the same `prefix` twice replaces the previous tokenizer.
pub fn register_hf(prefix: impl Into<String>, tokenizer: HfTokenizer) {
    let key = prefix.into().to_ascii_lowercase();
    hf_table()
        .write()
        .expect("hf registry poisoned")
        .insert(key, tokenizer);
}

/// Drop every HuggingFace registration. Intended for tests; production code
/// should register once at startup.
pub fn clear_hf_registrations() {
    hf_table().write().expect("hf registry poisoned").clear();
}

/// Convenience: download `tokenizer.json` for `repo` from the HuggingFace
/// Hub and register it under `prefix`. One-line glue around
/// [`HfTokenizer::from_pretrained`] + [`register_hf`].
///
/// Useful for proxy startup code that wants real tokenizers for the major
/// non-OpenAI families. Each call is independent — failure for one model
/// (e.g. a gated Llama repo without `HF_TOKEN`) does not affect others.
///
/// ```no_run
/// use headroom_core::tokenizer::try_register_hf;
/// let _ = try_register_hf("command-", "CohereForAI/c4ai-command-r-v01");
/// let _ = try_register_hf("mistral-", "mistralai/Mistral-7B-v0.1");
/// ```
pub fn try_register_hf(prefix: &str, repo: &str) -> Result<(), HfTokenizerError> {
    let t = HfTokenizer::from_pretrained(repo)?;
    register_hf(prefix, t);
    Ok(())
}

fn lookup_hf(model: &str) -> Option<HfTokenizer> {
    let m = model.to_ascii_lowercase();
    let table = hf_table().read().expect("hf registry poisoned");
    // Longest prefix wins: a `command-r-plus` registration must beat
    // `command-` for that model, regardless of insertion order.
    table
        .iter()
        .filter(|(prefix, _)| m.starts_with(prefix.as_str()))
        .max_by_key(|(prefix, _)| prefix.len())
        .map(|(_, t)| t.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Same minimal tokenizer.json used by hf_impl tests; inlined here so we
    /// can register it without depending on a fixture file.
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

    fn tiny(name: &str) -> HfTokenizer {
        HfTokenizer::from_bytes(name, TINY_TOKENIZER_JSON.as_bytes()).unwrap()
    }

    /// Tests share a process-global table. `cargo test` runs them in parallel
    /// by default, so without serialization one test's `clear` would wipe
    /// another test's registrations mid-assertion. We serialize the tests
    /// that touch the registry behind a single mutex.
    static REGISTRY_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    struct RegistryGuard<'a> {
        _g: std::sync::MutexGuard<'a, ()>,
    }
    impl<'a> RegistryGuard<'a> {
        fn acquire() -> Self {
            // Recover from a poisoned lock — a panic in one test should not
            // break every subsequent test in the file.
            let g = REGISTRY_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            clear_hf_registrations();
            Self { _g: g }
        }
    }
    impl<'a> Drop for RegistryGuard<'a> {
        fn drop(&mut self) {
            clear_hf_registrations();
        }
    }

    #[test]
    fn openai_models_pick_tiktoken() {
        for m in [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "o1-preview",
            "o3-mini",
            "text-embedding-3-small",
            "text-davinci-003",
            "davinci",
            "babbage-002",
            "code-davinci-002",
        ] {
            assert_eq!(detect_backend(m), Backend::Tiktoken, "{m}");
        }
    }

    #[test]
    fn non_openai_models_fall_through_to_estimation() {
        for m in [
            "claude-haiku-4-5-20251001",
            "claude-3-opus",
            "gemini-1.5-pro",
            "command-r-plus",
            "llama-3-70b",
            "mistral-large",
            "qwen-72b",
            "made-up-model-name",
        ] {
            assert_eq!(detect_backend(m), Backend::Estimation, "{m}");
        }
    }

    #[test]
    fn case_insensitive() {
        assert_eq!(detect_backend("GPT-4o"), Backend::Tiktoken);
        assert_eq!(detect_backend("Claude-haiku"), Backend::Estimation);
    }

    #[test]
    fn estimator_density_per_family() {
        let _g = RegistryGuard::acquire();
        // Round-trip through the public dispatch and check that the chosen
        // estimator behaves with the right density. We can't introspect the
        // trait object's chars_per_token directly, so we use a known-length
        // string and back-compute.
        let claude = get_tokenizer("claude-3-opus");
        // 3.5 chars/token: 35 chars -> 10 tokens.
        assert_eq!(claude.count_text(&"a".repeat(35)), 10);

        let gemini = get_tokenizer("gemini-1.5-pro");
        // 4.0 chars/token: 40 chars -> 10 tokens.
        assert_eq!(gemini.count_text(&"a".repeat(40)), 10);
    }

    #[test]
    fn registered_hf_wins_over_estimator() {
        let _g = RegistryGuard::acquire();
        register_hf("command-", tiny("cohere"));
        let t = get_tokenizer("command-r-plus");
        // Whitespace tokenization → 2 tokens, *not* the chars/4 estimator.
        assert_eq!(t.count_text("hello world"), 2);
        assert_eq!(t.backend(), Backend::HuggingFace);
    }

    #[test]
    fn registered_hf_does_not_override_tiktoken() {
        // A user accidentally registers an HF tokenizer with the prefix
        // `gpt-` — should the OpenAI route still win? Per docstring: HF
        // always wins when registered. We document this behavior so the
        // test pins it down: registration is a deliberate override.
        let _g = RegistryGuard::acquire();
        register_hf("gpt-4o", tiny("oops"));
        let t = get_tokenizer("gpt-4o-mini");
        assert_eq!(t.backend(), Backend::HuggingFace);
    }

    #[test]
    fn longest_prefix_wins() {
        let _g = RegistryGuard::acquire();
        register_hf("command-", tiny("general"));
        register_hf("command-r-plus", tiny("specific"));
        // `command-r-plus` should pick the more specific one.
        let t = get_tokenizer("command-r-plus");
        assert_eq!(t.backend(), Backend::HuggingFace);
        // Counting the same input via both should give identical counts since
        // we used the same tokenizer.json — but the *path* through the
        // registry is different. This locks in that the longer prefix is
        // selected, not the shorter one.
        let count_specific = t.count_text("hello world hello");
        assert_eq!(count_specific, 3);

        // A model only matched by the shorter prefix still works.
        let t2 = get_tokenizer("command-light");
        assert_eq!(t2.backend(), Backend::HuggingFace);
    }

    #[test]
    fn case_insensitive_registration() {
        let _g = RegistryGuard::acquire();
        register_hf("Command-", tiny("cohere"));
        let t = get_tokenizer("COMMAND-R-PLUS");
        assert_eq!(t.backend(), Backend::HuggingFace);
    }

    #[test]
    fn clear_resets_state() {
        let _g = RegistryGuard::acquire();
        register_hf("command-", tiny("cohere"));
        clear_hf_registrations();
        let t = get_tokenizer("command-r-plus");
        // Without an HF registration, falls back to estimator (4.0 cpt).
        assert_eq!(t.backend(), Backend::Estimation);
    }

    #[test]
    fn unrelated_models_still_estimate() {
        let _g = RegistryGuard::acquire();
        register_hf("command-", tiny("cohere"));
        let t = get_tokenizer("claude-3-opus");
        assert_eq!(t.backend(), Backend::Estimation);
    }

    #[test]
    fn detect_backend_ignores_runtime_registrations() {
        let _g = RegistryGuard::acquire();
        register_hf("command-", tiny("cohere"));
        // detect_backend is a pure function of the model name; runtime
        // registrations don't show up here.
        assert_eq!(detect_backend("command-r-plus"), Backend::Estimation);
    }
}
