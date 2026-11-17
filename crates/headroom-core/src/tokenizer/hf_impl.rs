//! HuggingFace `tokenizers`-crate adapter implementing [`Tokenizer`].
//!
//! Loads a `tokenizer.json` (HuggingFace's serialization format) and counts
//! tokens via real BPE / Unigram / WordPiece — whatever the file describes.
//! This closes the gap between OpenAI (tiktoken, byte-equal) and the
//! Anthropic/Google chars-per-token fallback: every other major family
//! (Cohere `command-*`, Llama-3.x, Mistral, Qwen, BERT, T5, …) publishes a
//! `tokenizer.json` on the HuggingFace Hub and the `tokenizers` crate is a
//! pure-Rust loader, so we don't have to estimate.
//!
//! # Sources
//! - [`HfTokenizer::from_bytes`] — for tokenizers embedded via `include_bytes!`.
//! - [`HfTokenizer::from_file`] — for a local `tokenizer.json`.
//! - [`HfTokenizer::from_pretrained`] — pulls `tokenizer.json` from the
//!   HuggingFace Hub via `hf-hub`. First call downloads into the hub cache
//!   Python's `huggingface_hub` uses (`HF_HUB_CACHE`, `HF_HOME/hub`,
//!   `~/.cache/huggingface/hub`, …), subsequent calls hit it. Blocking;
//!   call from `main()` or `tokio::task::spawn_blocking`. Gated repos
//!   (Llama, Mistral) need a token: `HF_TOKEN`, or the file
//!   `hf auth login` writes (`HF_HOME/token`, or `HF_TOKEN_PATH`).
//!
//! # What's NOT here
//! - **No tokenizer.json bundled in the binary.** Bundling Llama / Cohere
//!   tokenizers would add several MB of binary bloat for code paths most users
//!   don't hit. `from_pretrained` lazily downloads instead.

use std::path::Path;
use std::sync::Arc;

use thiserror::Error;
use tokenizers::Tokenizer as HfInner;

use super::{Backend, Tokenizer};

#[derive(Debug, Error)]
pub enum HfTokenizerError {
    /// The bytes / file did not parse as a valid HuggingFace `tokenizer.json`,
    /// or the model component referenced an unsupported algorithm.
    #[error("failed to load tokenizer for `{name}`: {source}")]
    Load {
        name: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// The HuggingFace Hub fetch failed: network error, 404 on the repo, or
    /// 401 on a gated model without an `HF_TOKEN`.
    #[error("failed to download `{repo}` from HuggingFace Hub: {source}")]
    Hub {
        repo: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// `HEADROOM_OFFLINE` is set, so the Hub fetch was refused before any
    /// socket was opened. Deliberately a separate variant from [`Self::Hub`]:
    /// a caller that retries or falls back on a transient download failure
    /// must not do either here — the operator asked for no egress, and
    /// retrying an air-gap refusal is just a slower refusal.
    #[error(transparent)]
    Offline(#[from] crate::offline::OfflineEgressBlocked),
}

/// Token counter backed by a HuggingFace `tokenizer.json`.
///
/// Cheap to clone — internally an `Arc<tokenizers::Tokenizer>`. Construct once
/// at startup, share across handlers.
#[derive(Clone)]
pub struct HfTokenizer {
    name: String,
    inner: Arc<HfInner>,
}

impl std::fmt::Debug for HfTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HfTokenizer")
            .field("name", &self.name)
            .finish()
    }
}

impl HfTokenizer {
    /// Build from raw `tokenizer.json` bytes. Use this when the tokenizer is
    /// embedded via `include_bytes!` or fetched from a non-HF source.
    pub fn from_bytes(name: impl Into<String>, bytes: &[u8]) -> Result<Self, HfTokenizerError> {
        let name = name.into();
        let inner = HfInner::from_bytes(bytes).map_err(|e| HfTokenizerError::Load {
            name: name.clone(),
            source: e,
        })?;
        Ok(Self {
            name,
            inner: Arc::new(inner),
        })
    }

    /// Build from a `tokenizer.json` on disk.
    pub fn from_file(
        name: impl Into<String>,
        path: impl AsRef<Path>,
    ) -> Result<Self, HfTokenizerError> {
        let name = name.into();
        let inner = HfInner::from_file(path.as_ref()).map_err(|e| HfTokenizerError::Load {
            name: name.clone(),
            source: e,
        })?;
        Ok(Self {
            name,
            inner: Arc::new(inner),
        })
    }

    /// Download (or fetch from cache) `tokenizer.json` for `repo` from the
    /// HuggingFace Hub and load it. `repo` is the canonical Hub identifier,
    /// e.g. `"CohereForAI/c4ai-command-r-v01"` or `"meta-llama/Meta-Llama-3-8B"`.
    ///
    /// Uses the `main` revision. Blocking — calls into `ureq` synchronously.
    /// The file lands in the hub cache directory Python's `huggingface_hub`
    /// would use ([`crate::hf_cache::hub_cache_dir`]: `HF_HUB_CACHE`, then
    /// `HUGGINGFACE_HUB_CACHE`, `HF_HOME/hub`, `XDG_CACHE_HOME/huggingface/hub`,
    /// `~/.cache/huggingface/hub`), so later calls, and the Python side, hit
    /// the same copy. The token and `HF_ENDPOINT` are read as Python reads
    /// them too ([`crate::hf_cache::hf_token`]); `hf-hub`'s own defaults
    /// ignore every variable but `HF_HOME`, and with `Api::new` even that.
    ///
    /// Errors:
    /// - [`HfTokenizerError::Offline`] when `HEADROOM_OFFLINE` is set. Refused
    ///   before any socket is opened; see below.
    /// - [`HfTokenizerError::Hub`] for download failures (no network, 404,
    ///   401 on a gated model without `HF_TOKEN`).
    /// - [`HfTokenizerError::Load`] if the downloaded bytes don't parse as
    ///   a valid `tokenizer.json`. Should not happen for healthy HF repos.
    pub fn from_pretrained(repo: &str) -> Result<Self, HfTokenizerError> {
        // Air-gap chokepoint, matching `headroom/offline.py`'s `guard_egress`.
        // This is the Rust half of the same switch, so an operator who sets
        // HEADROOM_OFFLINE=1 gets the same refusal whichever runtime reaches
        // for the Hub.
        //
        // `crate::hf_cache::hub_api` runs the guard before it builds the
        // client, rather than around the `get` below: building it creates the
        // `ureq` agent and resolves the endpoint, and hf-hub's own
        // HF_HUB_OFFLINE handling only covers the cache lookup — a cache miss
        // still dials out. So the air-gapped case never constructs a client.
        //
        // The refusal is hard on purpose. Silently returning a cached or
        // estimating tokenizer would leave token counts subtly wrong with no
        // signal; the caller (`tokenizer::registry::try_register_hf`) already
        // propagates the error so the operator sees which repo was wanted.
        //
        // Note it refuses even on a warm cache: `hf-hub` 0.5 exposes no
        // cache-hit/cache-miss split at this layer, so "guard before the
        // client exists" and "serve from the cache" cannot both hold here. An
        // air-gapped deployment with pre-seeded artifacts should point at them
        // with `from_file`, which is network-free by construction.
        let api = hub_api(repo)?;
        let path = api
            .model(repo.to_string())
            .get("tokenizer.json")
            .map_err(|e| HfTokenizerError::Hub {
                repo: repo.to_string(),
                source: Box::new(e),
            })?;
        // `get` returns the on-disk path; reuse `from_file` to keep the load
        // path identical to user-supplied tokenizer.json files.
        Self::from_file(repo, path)
    }

    /// The logical name this tokenizer was registered under (e.g.
    /// `"command-r-plus"`). Used in logs and metrics.
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// The Hub client for [`HfTokenizer::from_pretrained`] fetching `repo`:
/// caching in Python's hub cache directory, with Python's token and endpoint.
/// Refuses with [`HfTokenizerError::Offline`] under `HEADROOM_OFFLINE`.
fn hub_api(repo: &str) -> Result<hf_hub::api::sync::Api, HfTokenizerError> {
    use crate::hf_cache::HubApiError;
    const PURPOSE: &str = "HuggingFace tokenizer download";

    let hub_error = |source: Box<dyn std::error::Error + Send + Sync>| HfTokenizerError::Hub {
        repo: repo.to_string(),
        source,
    };
    let Some(cache) = crate::hf_cache::hub_cache_dir() else {
        // Offline still wins: the caller treats `Offline` as final, and a
        // missing cache directory is no reason to report a Hub failure.
        crate::offline::guard_egress(PURPOSE, repo)?;
        return Err(hub_error(
            "no HuggingFace cache directory (set HF_HUB_CACHE or HF_HOME)".into(),
        ));
    };
    crate::hf_cache::hub_api(&cache, PURPOSE, repo).map_err(|e| match e {
        HubApiError::Offline(blocked) => HfTokenizerError::Offline(blocked),
        HubApiError::Build(source) => hub_error(Box::new(source)),
    })
}

impl Tokenizer for HfTokenizer {
    fn count_text(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }
        // `add_special_tokens=false` matches the spirit of
        // `tiktoken.encode_ordinary`: count *content* tokens, leaving
        // BOS/EOS/CLS/SEP padding to be added (or not) by the upstream API.
        // Different providers add different specials, so counting them here
        // would systematically over-charge. Documented for future readers.
        match self.inner.encode(text, false) {
            Ok(enc) => enc.len(),
            // `encode` only fails for malformed inputs that pass UTF-8 but
            // violate the tokenizer's constraints (e.g. a normalizer that
            // rejects certain code points). We degrade to "0 known tokens"
            // rather than panic — the proxy must keep flowing requests.
            Err(_) => 0,
        }
    }

    fn backend(&self) -> Backend {
        Backend::HuggingFace
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal valid `tokenizer.json`: WordLevel model, Whitespace
    /// pre-tokenizer, three-token vocabulary. Lets us test the API surface
    /// without committing a multi-MB tokenizer fixture or hitting the
    /// HuggingFace Hub.
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

    fn tiny() -> HfTokenizer {
        HfTokenizer::from_bytes("tiny-test", TINY_TOKENIZER_JSON.as_bytes())
            .expect("tiny tokenizer.json parses")
    }

    #[test]
    fn empty_is_zero() {
        assert_eq!(tiny().count_text(""), 0);
    }

    #[test]
    fn known_vocab_matches_count() {
        let t = tiny();
        // Each word in the vocab is one token; whitespace splits them.
        assert_eq!(t.count_text("hello"), 1);
        assert_eq!(t.count_text("hello world"), 2);
        assert_eq!(t.count_text("hello world hello"), 3);
    }

    #[test]
    fn unknown_words_become_unk() {
        // OOV tokens collapse to [UNK] — still 1 per whitespace-split chunk.
        let t = tiny();
        assert_eq!(t.count_text("supercalifragilistic"), 1);
        assert_eq!(t.count_text("foo bar baz"), 3);
    }

    #[test]
    fn deterministic() {
        let t = tiny();
        let s = "hello world hello world";
        let first = t.count_text(s);
        for _ in 0..100 {
            assert_eq!(t.count_text(s), first);
        }
    }

    #[test]
    fn unicode_does_not_panic() {
        let t = tiny();
        for s in ["héllo wörld", "你好世界", "🦀 ferris", "\n\t\r"] {
            // We only assert non-panic and a reasonable upper bound; the exact
            // count depends on Whitespace pre-tokenizer behavior, which is
            // tokenizers-crate internal.
            let n = t.count_text(s);
            assert!(n < s.len() * 4 + 10, "absurd count {n} for {s:?}");
        }
    }

    #[test]
    fn invalid_bytes_returns_error() {
        let r = HfTokenizer::from_bytes("bad", b"not a tokenizer.json");
        assert!(matches!(r, Err(HfTokenizerError::Load { .. })));
    }

    #[test]
    fn name_round_trips() {
        let t = tiny();
        assert_eq!(t.name(), "tiny-test");
    }

    #[test]
    fn backend_is_huggingface() {
        assert_eq!(tiny().backend(), Backend::HuggingFace);
    }

    #[test]
    fn clone_shares_inner() {
        let a = tiny();
        let b = a.clone();
        assert!(Arc::ptr_eq(&a.inner, &b.inner));
    }

    /// A-2: `HEADROOM_OFFLINE=1` must stop the Hub fetch before a socket exists.
    ///
    /// The repo name is deliberately one that cannot exist, which is what makes
    /// this a real assertion rather than a tautology: without the guard the call
    /// reaches `hf-hub`, dials `huggingface.co`, gets a 401/404 and returns
    /// `HfTokenizerError::Hub`. Only the guard can produce `Offline`, and only
    /// by returning before `hf_cache::hub_api` builds the client — so `Offline` *is* the "no socket was
    /// opened" assertion for this path. (A packet-level assertion is not
    /// available to a `cargo test` unit test the way `socket.socket` patching is
    /// in Python; this typed-error distinction is the equivalent signal.)
    #[test]
    fn from_pretrained_refuses_while_offline() {
        let _guard = crate::test_support::env_lock();
        std::env::set_var(crate::offline::OFFLINE_ENV, "1");
        let result = HfTokenizer::from_pretrained("headroomlabs-ai/this-repo-does-not-exist-a2");
        std::env::remove_var(crate::offline::OFFLINE_ENV);

        match result {
            Err(HfTokenizerError::Offline(blocked)) => {
                assert_eq!(blocked.purpose, "HuggingFace tokenizer download");
                assert_eq!(
                    blocked.destination,
                    "headroomlabs-ai/this-repo-does-not-exist-a2"
                );
            }
            Err(other) => panic!("expected an offline refusal, got a network error: {other}"),
            Ok(_) => panic!("expected an offline refusal, got a tokenizer"),
        }
    }

    // The "switch off ⇒ guard is a no-op" half is covered in `crate::offline`'s
    // own tests rather than here: asserting it through `from_pretrained` would
    // mean actually dialling huggingface.co, which is exactly the thing a unit
    // test must not do.

    #[test]
    fn from_file_loads_a_real_file() {
        // Round-trip via a temp file to cover the on-disk constructor.
        use std::io::Write;
        let dir = std::env::temp_dir().join(format!(
            "headroom-hf-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tokenizer.json");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(TINY_TOKENIZER_JSON.as_bytes()).unwrap();
        drop(f);

        let t = HfTokenizer::from_file("from-file", &path).expect("loads");
        assert_eq!(t.count_text("hello world"), 2);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Lay out `tokenizer.json` for `repo` in hub cache `cache`, the way
    /// Python's `huggingface_hub` leaves a `main` download: a snapshot plus
    /// `refs/main` naming it.
    fn cache_main_tokenizer(cache: &std::path::Path, repo: &str) {
        let repo_dir = cache.join(format!("models--{}", repo.replace('/', "--")));
        let snapshot = repo_dir.join("snapshots").join("abc123");
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join("tokenizer.json"), TINY_TOKENIZER_JSON).unwrap();
        std::fs::create_dir_all(repo_dir.join("refs")).unwrap();
        std::fs::write(repo_dir.join("refs").join("main"), "abc123").unwrap();
    }

    /// Variables `from_pretrained` reads, saved and restored around a test.
    fn hub_env() -> crate::hf_cache::tests::CacheEnv {
        crate::hf_cache::tests::CacheEnv::cleared_with(
            [
                "HF_TOKEN",
                "HUGGING_FACE_HUB_TOKEN",
                "HF_TOKEN_PATH",
                "HF_ENDPOINT",
                "HF_TEST_ROOT",
            ]
            .map(String::from),
        )
    }

    /// The tokenizer is read from (and would be downloaded into) the hub cache
    /// Python uses, here an `HF_HUB_CACHE` given through `${VAR}`. Before, it
    /// was always `~/.cache/huggingface/hub`, so this was a cache miss and a
    /// download. `HF_ENDPOINT` points at a closed loopback port, so a miss
    /// fails fast instead of reaching the internet.
    #[test]
    fn from_pretrained_reads_the_hub_cache_python_uses() {
        let _guard = crate::test_support::env_lock();
        let _env = hub_env();
        let t = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", t.path().join("home"));
        std::env::set_var("USERPROFILE", t.path().join("home"));
        std::env::set_var("HF_TEST_ROOT", t.path());
        std::env::set_var("HF_HUB_CACHE", "${HF_TEST_ROOT}/hub");
        std::env::set_var("HF_ENDPOINT", "http://127.0.0.1:9");
        cache_main_tokenizer(&t.path().join("hub"), "acme/tiny");

        let tokenizer = HfTokenizer::from_pretrained("acme/tiny").expect("served from the cache");
        assert_eq!(tokenizer.count_text("hello world"), 2);
    }

    /// A cache miss asks the Hub at `HF_ENDPOINT` with the token from
    /// `HF_TOKEN`. `hf-hub`'s `Api::new` read neither, so a gated repo
    /// (Llama, Mistral) failed with 401 even with `HF_TOKEN` set. A loopback
    /// stand-in for the Hub records the request and answers 404.
    #[test]
    fn from_pretrained_sends_hf_token_to_hf_endpoint() {
        use std::io::{Read, Write};
        let _guard = crate::test_support::env_lock();
        let _env = hub_env();
        let t = tempfile::tempdir().unwrap();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        // Waits at most 10 s for the request, so a regression that never
        // reaches the endpoint fails instead of hanging the suite.
        listener.set_nonblocking(true).unwrap();
        let hub = std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            let mut conn = loop {
                match listener.accept() {
                    Ok((conn, _)) => break conn,
                    Err(_) if std::time::Instant::now() < deadline => {
                        std::thread::sleep(std::time::Duration::from_millis(20))
                    }
                    Err(_) => return String::new(),
                }
            };
            conn.set_nonblocking(false).unwrap();
            let mut request = Vec::new();
            let mut buf = [0u8; 1024];
            while !request.windows(4).any(|w| w == b"\r\n\r\n") {
                let n = conn.read(&mut buf).unwrap();
                if n == 0 {
                    break;
                }
                request.extend_from_slice(&buf[..n]);
            }
            conn.write_all(
                b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            )
            .unwrap();
            String::from_utf8_lossy(&request).to_ascii_lowercase()
        });
        std::env::set_var("HOME", t.path().join("home"));
        std::env::set_var("USERPROFILE", t.path().join("home"));
        std::env::set_var("HF_HUB_CACHE", t.path().join("hub"));
        // Trailing slash on purpose: Python strips it, and a kept one
        // would request `//acme/...`.
        std::env::set_var("HF_ENDPOINT", format!("http://127.0.0.1:{port}/"));
        std::env::set_var("HF_TOKEN", "hf_test_token");

        let result = HfTokenizer::from_pretrained("acme/gated");
        let request = hub.join().unwrap();
        assert!(
            matches!(result, Err(HfTokenizerError::Hub { .. })),
            "{result:?}"
        );
        assert!(
            request.contains(" /acme/gated/resolve/main/tokenizer.json "),
            "{request}"
        );
        assert!(
            request.contains("authorization: bearer hf_test_token"),
            "{request}"
        );
    }

    /// Network-dependent: hits HuggingFace Hub. Run with
    /// `cargo test -p headroom-core -- --ignored from_pretrained_downloads_real_tokenizer`.
    /// `gpt2` is a small public unauthenticated repo (~1.4 MB tokenizer.json).
    #[test]
    #[ignore = "network-dependent: hits HuggingFace Hub"]
    fn from_pretrained_downloads_real_tokenizer() {
        let t = HfTokenizer::from_pretrained("gpt2").expect("download succeeds");
        // GPT-2 BPE: "hello world" is 2 tokens. Locks in that we got a real
        // BPE tokenizer (not a WhitespaceSplit fixture) from HF.
        assert_eq!(t.count_text("hello world"), 2);
        assert_eq!(t.name(), "gpt2");
        assert_eq!(t.backend(), Backend::HuggingFace);
    }

    /// Negative path: a malformed repo name fails before any network call
    /// (the Hub URL builder rejects it). No network required to run.
    #[test]
    fn from_pretrained_invalid_repo_returns_hub_error() {
        // Empty repo name — hf-hub rejects this without making a network
        // request. Locks in that we propagate Hub errors as `Hub`, not
        // `Load`, so callers can distinguish "couldn't fetch" from
        // "fetched but malformed".
        //
        // Runs online: hold the env lock so a parallel test's
        // `HEADROOM_OFFLINE=1` can't turn this into `Offline`, and clear an
        // inherited value for the same reason.
        let _guard = crate::test_support::env_lock();
        let inherited = std::env::var_os(crate::offline::OFFLINE_ENV);
        std::env::remove_var(crate::offline::OFFLINE_ENV);
        let r = HfTokenizer::from_pretrained("");
        if let Some(value) = inherited {
            std::env::set_var(crate::offline::OFFLINE_ENV, value);
        }
        assert!(
            matches!(r, Err(HfTokenizerError::Hub { .. })),
            "expected HfTokenizerError::Hub, got {r:?}"
        );
    }
}
