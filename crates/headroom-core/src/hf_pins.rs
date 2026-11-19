//! Immutable HuggingFace revisions for the artifacts Headroom ships against.
//!
//! GH #4939. `hf-hub` resolves a repo id against its default branch (`main`)
//! unless a revision is supplied, so the same id can hand back different
//! `tokenizer.json` or ONNX bytes a month later. Tokenizer bytes decide token
//! accounting and Kompress bytes decide compression behaviour, so a silent
//! upstream push changes what the proxy reports with no change on our side.
//!
//! Every download path therefore resolves its revision through [`revision`]
//! (or [`resolve`] for an explicit override) instead of letting `hf-hub`
//! default. The table is the Rust twin of the Python pins in
//! `headroom.onnx_runtime._PINNED_REVISIONS` and
//! `headroom.tokenizers.huggingface._TOKENIZER_REVISIONS`, and is a **subset**
//! of them: the Python side also owns torch-path loaders (sentence-transformers,
//! SigLIP, the technique router) the Rust core never loads.
//! `tests/test_hf_pin_parity.py` fails if the tables disagree on a shared
//! repository, or if this one pins something Python does not.
//!
//! Refresh deliberately: bump the SHA, run the tests, ship it. `HEADROOM_HF_PIN=off`
//! floats every repository again — the same switch the Python side honours — for
//! evaluating a newer upstream revision before adopting it.
//!
//! Offline deploys do not need this table at all: [`crate::offline::guard_egress`]
//! refuses these paths before a socket opens, and `from_file`/`from_files` are
//! network-free by construction.

use std::collections::HashMap;
use std::sync::OnceLock;

/// The switch that floats every pin. Off-values match `onnx_runtime._resolve_revision`.
pub const PIN_ENV: &str = "HEADROOM_HF_PIN";

/// Pinned commit SHAs, keyed by Hub repo id.
///
/// Kompress (model + tokenizer), the technique router and the SigLIP encoder are
/// the ML artifacts the live-zone dispatcher loads; the two Qdrant repos are the
/// embedding models behind relevance scoring and the memory adapters. Everything
/// else is a tokenizer vocabulary from `headroom.tokenizers.huggingface`.
///
/// Python pins more than this (`sentence-transformers/all-MiniLM-L6-v2`,
/// `google/siglip-base-patch16-224`, `chopratejas/technique-router`) because those
/// load through torch in the Python package; the Rust core has no equivalent
/// loader, so listing them here would be a pin with no download to protect.
pub static PINNED_REVISIONS: &[(&str, &str)] = &[
    // ─── Kompress (Rust live-zone ML compressor) ────────────────────────
    // ONNX weights + the tokenizer the export was produced against. Loaded by
    // `transforms::kompress`, so a floating revision here changes compression
    // output, not just a count.
    (
        "chopratejas/kompress-v2-base",
        "b1563631b35bfdcee37587ad530147497d820d4c",
    ),
    (
        "answerdotai/ModernBERT-base",
        "8949b909ec900327062f0ebf497f51aef5e6f0c8",
    ),
    // ─── Other shipped ONNX artifacts ──────────────────────────────────
    (
        "chopratejas/technique-router-onnx",
        "27b0b4bfa510a1cff66d888072c0b807082721a8",
    ),
    (
        "chopratejas/siglip-image-encoder-onnx",
        "d0a9fbd66d4bd8c761bff592d44831f7c2ae184e",
    ),
    // Third-party repos — pinning matters most here.
    (
        "Qdrant/all-MiniLM-L6-v2-onnx",
        "5f1b8cd78bc4fb444dd171e59b18f3a3af89a079",
    ),
    (
        "Qdrant/bge-small-en-v1.5-onnx-Q",
        "52398278842ec682c6f32300af41344b1c0b0bb2",
    ),
    // ─── Tokenizer vocabularies (mirrors _TOKENIZER_REVISIONS) ─────────
    (
        "meta-llama/Meta-Llama-3-8B",
        "8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920",
    ),
    (
        "meta-llama/Meta-Llama-3-70B",
        "c82494877ce7f6d7d317c56ec081328e382c72fe",
    ),
    (
        "meta-llama/Llama-3.1-8B",
        "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
    ),
    (
        "meta-llama/Llama-3.1-70B",
        "349b2ddb53ce8f2849a6c168a81980ab25258dac",
    ),
    (
        "meta-llama/Llama-3.1-405B",
        "b906e4dc842aa489c962f9db26554dcfdde901fe",
    ),
    (
        "meta-llama/Llama-3.2-1B",
        "4e20de362430cd3b72f300e6b0f18e50e7166e08",
    ),
    (
        "meta-llama/Llama-3.2-3B",
        "13afe5124825b4f3751f836b40dafda64c1ed062",
    ),
    (
        "meta-llama/Llama-3.3-70B-Instruct",
        "6f6073b423013f6a7d4d9f39144961bfbfbc386b",
    ),
    (
        "meta-llama/Llama-2-7b-hf",
        "01c7f73d771dfac7d292323805ebc428287df4f9",
    ),
    (
        "meta-llama/Llama-2-13b-hf",
        "5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1",
    ),
    (
        "meta-llama/Llama-2-70b-hf",
        "3aba440b59558f995867ba6e1f58f21d0336b5bb",
    ),
    (
        "codellama/CodeLlama-7b-hf",
        "6c284d1468fe6c413cf56183e69b194dcfa27fe6",
    ),
    (
        "codellama/CodeLlama-13b-hf",
        "8da65ff4ee20f74ecd107ca9d54f9f121b279860",
    ),
    (
        "codellama/CodeLlama-34b-hf",
        "6008b9656730b71c7d19a15370c7ff6d2902f4ef",
    ),
    (
        "mistralai/Mistral-7B-v0.1",
        "27d67f1b5f57dc0953326b2601d68371d40ea8da",
    ),
    (
        "mistralai/Mistral-7B-Instruct-v0.2",
        "63a8b081895390a26e140280378bc85ec8bce07a",
    ),
    (
        "mistralai/Mistral-7B-Instruct-v0.3",
        "c170c708c41dac9275d15a8fff4eca08d52bab71",
    ),
    (
        "mistralai/Mistral-Nemo-Base-2407",
        "a4477a2f977929a969745b69bbd62e03043551a5",
    ),
    (
        "mistralai/Mistral-Small-Instruct-2409",
        "4600506f6b13c7ef89e61a54263f4c9bf483de30",
    ),
    (
        "mistralai/Mistral-Large-Instruct-2407",
        "a286006d554cb37a61d13c7ae61bc90cc1d372fc",
    ),
    (
        "mistralai/Mixtral-8x7B-v0.1",
        "fc7ac94680e38d7348cfa806e51218e6273104b0",
    ),
    (
        "mistralai/Mixtral-8x22B-v0.1",
        "e1cd34ff1747406fb2277635ed25242803009bc2",
    ),
    ("Qwen/Qwen-7B", "ef3c5c9c57b252f3149c1408daf4d649ec8b6c85"),
    ("Qwen/Qwen-14B", "c4051215126d906ac22bb67fe5edb39a921cd831"),
    ("Qwen/Qwen-72B", "b8e18ac61df64d35308695769ff46b976b6a00f4"),
    ("Qwen/Qwen2-7B", "453ed1575b739b5b03ce3758b23befdb0967f40e"),
    ("Qwen/Qwen2-72B", "5f149cae83c11c408a953dec64b12e7e662e8cb9"),
    (
        "Qwen/Qwen2.5-7B",
        "d149729398750b98c0af14eb82c78cfe92750796",
    ),
    (
        "Qwen/Qwen2.5-72B",
        "efba10c8e54e91e0d9570ab5f7b51a958474d4cb",
    ),
    (
        "deepseek-ai/deepseek-llm-7b-base",
        "7683fea62db869066ddaff6a41d032262c490d4f",
    ),
    (
        "deepseek-ai/deepseek-llm-67b-base",
        "c3f813a1121c95488a20132d3a4da89f4a46452f",
    ),
    (
        "deepseek-ai/deepseek-coder-6.7b-base",
        "ce2207a8bfef3ee92bd7dd4cc31c52cfa0046912",
    ),
    (
        "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        "e434a23f91ba5b4923cf6c9d9a238eb4a08e3a11",
    ),
    (
        "deepseek-ai/DeepSeek-V2",
        "4461458f186c35188585855f28f77af5661ad489",
    ),
    (
        "deepseek-ai/DeepSeek-V2-Lite",
        "604d5664dddd88a0433dbae533b7fe9472482de0",
    ),
    (
        "deepseek-ai/DeepSeek-V3",
        "e815299b0bcbac849fa540c768ef21845365c9eb",
    ),
    (
        "deepseek-ai/DeepSeek-V3-0324",
        "e9b33add76883f293d6bf61f6bd89b497e80e335",
    ),
    (
        "deepseek-ai/DeepSeek-V3.2",
        "a7e62ac04ecb2c0a54d736dc46601c5606cf10a6",
    ),
    (
        "deepseek-ai/DeepSeek-R1",
        "56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad",
    ),
    (
        "deepseek-ai/DeepSeek-R1-0528",
        "4236a6af538feda4548eca9ab308586007567f52",
    ),
    (
        "deepseek-ai/DeepSeek-V4.1-Flash",
        "314113f8a7dadb388215396ba5fbee2c610a04d6",
    ),
    (
        "deepseek-ai/DeepSeek-V4-Pro",
        "b5968e9190ef611bbf34a7229255be88a0e937c1",
    ),
    (
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "916b56a44061fd5cd7d6a8fb632557ed4f724f60",
    ),
    (
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "6a6f4aa4197940add57724a7707d069478df56b1",
    ),
    ("01-ai/Yi-6B", "80080be87ec5a0103f643195f2d9003b8068941b"),
    ("01-ai/Yi-34B", "e1fa7c83283f2e1d5a12cb4adea37c3f829e28f8"),
    (
        "01-ai/Yi-1.5-6B",
        "157a3d77ab5a8a8f079510eaa25169b547481f06",
    ),
    (
        "microsoft/phi-2",
        "810d367871c1d460086d9f82db8696f2e0a0fcd0",
    ),
    (
        "microsoft/Phi-3-mini-4k-instruct",
        "f39ac1d28e925b323eae81227eaba4464caced4e",
    ),
    (
        "microsoft/Phi-3-small-8k-instruct",
        "188b876e997ee4055df01afe46274c14c400fe40",
    ),
    (
        "microsoft/Phi-3-medium-4k-instruct",
        "48d87cd5e0523b430d77e93becd3655cd6897230",
    ),
    (
        "tiiuae/falcon-7b",
        "ec89142b67d748a1865ea4451372db8313ada0d8",
    ),
    (
        "tiiuae/falcon-40b",
        "05ab2ee8d6b593bdbab17d728de5c028a7a94d83",
    ),
    (
        "tiiuae/falcon-180B",
        "d2ea5531862d4fe907280234990e6380d2befd97",
    ),
    (
        "bigcode/starcoder",
        "827cd7721231872c152e3faef5cae1b22964ed58",
    ),
    (
        "bigcode/starcoder2-15b",
        "46d44742909c03ac8cee08eb03fdebce02e193ec",
    ),
    (
        "bigcode/starcoder2-3b",
        "733247c55e3f73af49ce8e9c7949bf14af205928",
    ),
    (
        "bigcode/starcoder2-7b",
        "bb9afde76d7945da5745592525db122d4d729eb1",
    ),
    (
        "google/gemma-7b",
        "ff6768d9368919a1f025a54f9f5aa0ee591730bb",
    ),
    (
        "google/gemma-2b",
        "9cf48e52b224239de00d483ec8eb84fb8d0f3a3a",
    ),
    (
        "google/gemma-2-9b",
        "33c193028431c2fde6c6e51f29e6f17b60cbfac6",
    ),
    (
        "google/gemma-2-27b",
        "938270f5272feb02779b55c2bb2fffdd0f53ff0c",
    ),
];

fn table() -> &'static HashMap<&'static str, &'static str> {
    static TABLE: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
    TABLE.get_or_init(|| PINNED_REVISIONS.iter().copied().collect())
}

/// True when the operator asked for floating revisions (`HEADROOM_HF_PIN=off`).
fn pinning_disabled() -> bool {
    match std::env::var(PIN_ENV) {
        Ok(raw) => matches!(
            raw.trim().to_ascii_lowercase().as_str(),
            "off" | "0" | "false" | "no"
        ),
        Err(_) => false,
    }
}

/// The revision to fetch `repo` at, or `None` for the floating default branch.
///
/// `None` is only returned for a repository we do not pin, or when the operator
/// turned pinning off. Both are deliberate: an unknown repo has no committed
/// snapshot we could name, and pretending `main` is one would be worse than
/// saying so.
pub fn revision(repo: &str) -> Option<&'static str> {
    if pinning_disabled() {
        return None;
    }
    table().get(repo).copied()
}

/// `explicit` when the caller supplied one, else the pin for `repo`.
///
/// Every `from_pretrained`-style entry point goes through this rather than
/// [`revision`] so an override stays possible for a caller that pins its own
/// artifacts (a vendored export, a mirrored repo).
pub fn resolve<'a>(repo: &str, explicit: Option<&'a str>) -> Option<&'a str> {
    explicit.or_else(|| revision(repo))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_pin_env<T>(value: Option<&str>, f: impl FnOnce() -> T) -> T {
        let _guard = crate::test_support::env_lock();
        let previous = std::env::var(PIN_ENV).ok();
        match value {
            Some(v) => std::env::set_var(PIN_ENV, v),
            None => std::env::remove_var(PIN_ENV),
        }
        let out = f();
        match previous {
            Some(v) => std::env::set_var(PIN_ENV, v),
            None => std::env::remove_var(PIN_ENV),
        }
        out
    }

    #[test]
    fn every_pin_is_a_full_commit_sha() {
        assert!(!PINNED_REVISIONS.is_empty());
        for (repo, sha) in PINNED_REVISIONS {
            assert_eq!(sha.len(), 40, "{repo} is not a full commit SHA: {sha:?}");
            assert!(
                sha.chars()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
                "{repo} is not a lowercase hex SHA: {sha:?}"
            );
        }
    }

    #[test]
    fn no_repo_is_pinned_twice() {
        let mut seen = std::collections::HashSet::new();
        for (repo, _) in PINNED_REVISIONS {
            assert!(
                seen.insert(*repo),
                "{repo} appears twice in PINNED_REVISIONS"
            );
        }
    }

    #[test]
    fn the_shipped_ml_artifacts_are_pinned() {
        // These are the three paths GH #4939 names, plus the tokenizer the
        // Kompress export was produced against.
        for repo in [
            "chopratejas/kompress-v2-base",
            "answerdotai/ModernBERT-base",
            "chopratejas/technique-router-onnx",
            "chopratejas/siglip-image-encoder-onnx",
        ] {
            assert!(table().contains_key(repo), "{repo} lost its pin");
        }
    }

    #[test]
    fn known_repo_resolves_to_its_pin() {
        with_pin_env(None, || {
            assert_eq!(
                revision("chopratejas/kompress-v2-base"),
                Some("b1563631b35bfdcee37587ad530147497d820d4c")
            );
        });
    }

    #[test]
    fn unknown_repo_floats() {
        with_pin_env(None, || {
            assert_eq!(revision("acme/not-a-shipped-model"), None);
        });
    }

    #[test]
    fn explicit_revision_wins_over_the_pin() {
        with_pin_env(None, || {
            assert_eq!(
                resolve("chopratejas/kompress-v2-base", Some("deadbeef")),
                Some("deadbeef")
            );
        });
    }

    #[test]
    fn pin_env_floats_every_repo() {
        for value in ["off", "0", "false", "no", "OFF"] {
            with_pin_env(Some(value), || {
                assert_eq!(revision("chopratejas/kompress-v2-base"), None, "{value}");
            });
        }
    }

    #[test]
    fn pin_env_floats_but_still_honours_an_explicit_revision() {
        with_pin_env(Some("off"), || {
            assert_eq!(
                resolve("chopratejas/kompress-v2-base", Some("abc123")),
                Some("abc123")
            );
        });
    }

    #[test]
    fn an_unrecognised_pin_env_value_keeps_pinning() {
        // Fail closed: a typo must not silently float the artifacts.
        with_pin_env(Some("maybe"), || {
            assert!(revision("chopratejas/kompress-v2-base").is_some());
        });
    }

    /// `transforms::kompress` keeps named constants for parity harnesses, and
    /// this table is what every download and cache path resolves through. If a
    /// pin bump lands in one and not the other, the model download and the
    /// cache read would disagree about which snapshot is current.
    ///
    /// Kompress is `ml`-gated, so this is too — `cargo test -p headroom-core
    /// --no-default-features --lib` must compile and pass.
    #[cfg(feature = "ml")]
    #[test]
    fn kompress_constants_agree_with_the_pin_table() {
        use crate::transforms::kompress::{
            DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, DEFAULT_TOKENIZER_REPO,
            DEFAULT_TOKENIZER_REVISION,
        };
        with_pin_env(None, || {
            assert_eq!(revision(DEFAULT_MODEL_ID), Some(DEFAULT_MODEL_REVISION));
            assert_eq!(
                revision(DEFAULT_TOKENIZER_REPO),
                Some(DEFAULT_TOKENIZER_REVISION)
            );
        });
    }
}
