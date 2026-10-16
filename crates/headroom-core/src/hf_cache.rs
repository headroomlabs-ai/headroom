//! The HuggingFace hub cache, located and read the way Python's
//! `huggingface_hub` does, so the Rust loaders (the relevance embedder,
//! Kompress and the HuggingFace tokenizer) find the files the Python side
//! downloaded and pinned, and authenticate with the same token.
//!
//! All network-free: [`hub_cache_dir`] picks the one cache directory Python
//! would use, [`snapshot_path`] resolves a repo at a revision inside it,
//! [`hf_token`] finds the access token and [`hub_api`] builds a client from
//! them. One known divergence: an empty variable counts as unset here, where
//! Python would take it as a path relative to the working directory (see
//! [`hub_cache_dir`]). The pinned revisions themselves live
//! next to each loader and are kept equal to `_PINNED_REVISIONS` in
//! `headroom/onnx_runtime.py` by tests on the Python side.
//!
//! Not behind the `ml` feature: the tokenizer, which is not either, uses the
//! cache and token lookups.

use std::path::{Path, PathBuf};

/// True when `HEADROOM_HF_PIN` turns revision pinning off, so pinned repos
/// float on `main` (same values as Python's `onnx_runtime._resolve_revision`).
#[cfg_attr(not(feature = "ml"), allow(dead_code))] // the pinned loaders are `ml`-only
pub(crate) fn pin_disabled() -> bool {
    let pin = std::env::var("HEADROOM_HF_PIN").unwrap_or_default();
    matches!(
        pin.trim().to_ascii_lowercase().as_str(),
        "off" | "0" | "false" | "no"
    )
}

/// The revision to load for a repo pinned at `pinned`: the pin, or `main`
/// when `HEADROOM_HF_PIN` turns pinning off.
#[cfg_attr(not(feature = "ml"), allow(dead_code))]
pub(crate) fn pinned_or_main(pinned: &str) -> &str {
    if pin_disabled() {
        "main"
    } else {
        pinned
    }
}

/// The snapshot directory of `repo` (`"owner/name"`) at `revision` in the hub
/// cache `cache`, whether or not it exists. Never touches the network.
///
/// A branch name such as `main` resolves through `refs/<revision>`; a commit
/// SHA is used as the directory name directly. `hf_hub::Cache` resolves every
/// revision through `refs/`, which Python's `huggingface_hub` does not write
/// for a commit SHA, so a snapshot Python downloaded at a pin would be missed.
#[cfg_attr(not(feature = "ml"), allow(dead_code))]
pub(crate) fn snapshot_path(cache: &Path, repo: &str, revision: &str) -> PathBuf {
    let repo_dir = cache.join(format!("models--{}", repo.replace('/', "--")));
    let commit = std::fs::read_to_string(repo_dir.join("refs").join(revision))
        .map(|c| c.trim().to_string())
        .unwrap_or_else(|_| revision.to_string());
    repo_dir.join("snapshots").join(commit)
}

/// The HuggingFace hub cache directory, resolved the way Python's
/// `huggingface_hub` resolves `constants.HF_HUB_CACHE`, so both languages look
/// up and download into the same place: `HF_HUB_CACHE`, then
/// `HUGGINGFACE_HUB_CACHE` (its legacy name), then `HF_HOME/hub`, then
/// `XDG_CACHE_HOME/huggingface/hub`, then `~/.cache/huggingface/hub`.
///
/// It is exactly one directory. A snapshot in a lower-priority cache is not
/// used when a higher one is configured, which is also what Python does: with
/// `HF_HUB_CACHE` set, a model under `~/.cache` is neither found nor reused.
///
/// Both expansions happen as in `huggingface_hub.constants`:
/// `expandvars(expanduser(...))`, first on `HF_HOME` (or its
/// `XDG_CACHE_HOME` default) and then on the hub path itself. So
/// `HF_HUB_CACHE=${ROOT}/hub` names the same directory in both languages, and
/// a `~` that only appears once a variable is expanded stays literal, as it
/// does in Python. The home directory comes from `std::env::home_dir`, which
/// falls back to the user database when `HOME` is unset, as `expanduser`
/// does. An empty variable counts as unset; Python would treat it as a path
/// relative to the working directory. `tests/parity/fixtures/hf_hub_cache/`
/// holds the cases both languages are checked against.
pub(crate) fn hub_cache_dir() -> Option<PathBuf> {
    let hub = match env_path("HF_HUB_CACHE").or_else(|| env_path("HUGGINGFACE_HUB_CACHE")) {
        Some(dir) => dir,
        None => hf_home_dir()?.join("hub"),
    };
    Some(expand(hub))
}

/// `huggingface_hub.constants.HF_HOME`: `HF_HOME`, else
/// `XDG_CACHE_HOME/huggingface`, else `~/.cache/huggingface`, expanded as
/// Python does. The hub cache defaults to its `hub` folder and the login
/// token to its `token` file.
fn hf_home_dir() -> Option<PathBuf> {
    let dir = match env_path("HF_HOME") {
        Some(dir) => dir,
        None => env_path("XDG_CACHE_HOME")
            .or_else(|| std::env::home_dir().map(|home| home.join(".cache")))?
            .join("huggingface"),
    };
    Some(expand(dir))
}

/// The HuggingFace access token, found the way `huggingface_hub.get_token()`
/// finds it: `HF_TOKEN`, then its legacy name `HUGGING_FACE_HUB_TOKEN`, then
/// the file at `HF_TOKEN_PATH` (default `HF_HOME/token`, where
/// `hf auth login` writes it). Each value is cleaned like Python's
/// `_clean_token`: line breaks removed, outer whitespace trimmed, and an
/// empty result counts as no token.
///
/// `hf-hub`'s own lookup reads only a `token` file beside the hub cache, so
/// `HF_TOKEN`, an explicit `HF_TOKEN_PATH`, or a hub cache moved away from
/// `HF_HOME` would leave a gated download unauthenticated.
pub(crate) fn hf_token() -> Option<String> {
    let clean = |token: String| {
        let token = token.replace(['\r', '\n'], "");
        let token = token.trim();
        (!token.is_empty()).then(|| token.to_string())
    };
    // `HF_TOKEN or HUGGING_FACE_HUB_TOKEN`, then clean: a whitespace-only
    // `HF_TOKEN` is still the one picked, and cleans to no token, so the
    // legacy name is not consulted (the login file is).
    ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"]
        .into_iter()
        .find_map(|name| std::env::var(name).ok().filter(|v| !v.is_empty()))
        .and_then(clean)
        .or_else(|| {
            let path = match env_path("HF_TOKEN_PATH") {
                Some(path) => expand(path),
                None => hf_home_dir()?.join("token"),
            };
            std::fs::read_to_string(path).ok().and_then(clean)
        })
}

/// The Hub endpoint: `HF_ENDPOINT` without trailing slashes (Python's
/// `constants.ENDPOINT` strips them, and `hf-hub` appends `/{repo}/…`, so a
/// kept slash would request `//repo`), else `https://huggingface.co`.
pub(crate) fn hf_endpoint() -> String {
    match std::env::var("HF_ENDPOINT") {
        Ok(endpoint) => endpoint.trim_end_matches('/').to_string(),
        Err(_) => "https://huggingface.co".to_string(),
    }
}

/// An `hf-hub` client that caches in `cache` and talks to [`hf_endpoint`]
/// with [`hf_token`], as Python's `huggingface_hub` would. Building it opens
/// no connection; callers put their `guard_egress` in front of it.
///
/// The token is not sent over plain `http://` to a host other than loopback:
/// it would cross the network in clear text on every request, so the client
/// goes without it and says why. Python would send it; a gated repo behind
/// such a mirror then answers 401 rather than leak the credential.
pub(crate) fn hub_api(cache: &Path) -> Result<hf_hub::api::sync::Api, hf_hub::api::sync::ApiError> {
    let endpoint = hf_endpoint();
    let token = hf_token().filter(|_| {
        let exposed = sends_in_clear_text(&endpoint);
        if exposed {
            tracing::warn!(
                event = "hf_token_withheld",
                endpoint = %endpoint,
                "HF_ENDPOINT is plain http to a non-loopback host; not sending the \
                 HuggingFace token to it. Use an https endpoint for gated repos."
            );
        }
        !exposed
    });
    hf_hub::api::sync::ApiBuilder::from_cache(hf_hub::Cache::new(cache.to_path_buf()))
        .with_progress(false)
        .with_endpoint(endpoint)
        .with_token(token)
        .build()
}

/// Whether a request to `endpoint` would carry its headers unencrypted off
/// this machine: `http://` to anything but a loopback host.
fn sends_in_clear_text(endpoint: &str) -> bool {
    let Some(rest) = endpoint
        .get(..7)
        .filter(|scheme| scheme.eq_ignore_ascii_case("http://"))
        .map(|_| &endpoint[7..])
    else {
        return false;
    };
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    let host_port = authority.rsplit('@').next().unwrap_or("");
    let host = match host_port.strip_prefix('[') {
        Some(v6) => v6.split(']').next().unwrap_or(""),
        None => host_port.split(':').next().unwrap_or(""),
    };
    let loopback = host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback());
    !loopback
}

/// A path-valued variable; an empty one counts as unset.
fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

/// `expandvars(expanduser(path))`, the order `huggingface_hub.constants` uses.
fn expand(path: PathBuf) -> PathBuf {
    expand_vars(&expand_user(path, std::env::home_dir().as_deref()))
}

/// Python's `os.path.expanduser` for a leading bare `~` component. `~user`
/// stays as written, which is also what `expanduser` does when it cannot
/// resolve the user.
fn expand_user(path: PathBuf, home: Option<&Path>) -> PathBuf {
    match (path.strip_prefix("~"), home) {
        (Ok(rest), Some(home)) => home.join(rest),
        _ => path,
    }
}

/// Python's `os.path.expandvars`: `posixpath` rules off Windows, `ntpath`
/// rules on it. A path that is not valid Unicode is returned unchanged.
fn expand_vars(path: &Path) -> PathBuf {
    let Some(text) = path.to_str() else {
        return path.to_path_buf();
    };
    #[cfg(windows)]
    let expanded = expand_vars_nt(text, env_lookup);
    #[cfg(not(windows))]
    let expanded = expand_vars_posix(text, env_lookup);
    PathBuf::from(expanded)
}

/// An environment value for `expand_vars`. Names the OS cannot hold (empty,
/// or containing `=` or NUL) are unset, which is how Python's `os.environ`
/// lookup fails for them too.
fn env_lookup(name: &str) -> Option<std::ffi::OsString> {
    if name.is_empty() || name.contains(['=', '\0']) {
        return None;
    }
    std::env::var_os(name)
}

/// `posixpath.expandvars`: `$name` (ASCII letters, digits, `_`) or
/// `${anything but a brace}`. An unset name, and a `$` that starts neither
/// form, stay as written. Expanded values are not scanned again.
#[cfg_attr(windows, allow(dead_code))] // tests run both rule sets everywhere
fn expand_vars_posix(
    text: &str,
    lookup: impl Fn(&str) -> Option<std::ffi::OsString>,
) -> std::ffi::OsString {
    let mut out = std::ffi::OsString::new();
    let mut rest = text;
    while let Some(at) = rest.find('$') {
        out.push(&rest[..at]);
        let after = &rest[at + 1..];
        let word = after
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(after.len());
        // (name, length of the reference after the `$`)
        let reference = if word > 0 {
            Some((&after[..word], word))
        } else {
            after
                .strip_prefix('{')
                .and_then(|braced| braced.find('}').map(|end| (&braced[..end], end + 2)))
        };
        match reference {
            Some((name, len)) => {
                match lookup(name) {
                    Some(value) => out.push(value),
                    None => out.push(&rest[at..at + 1 + len]),
                }
                rest = &after[len..];
            }
            None => {
                out.push("$");
                rest = after;
            }
        }
    }
    out.push(rest);
    out
}

/// `ntpath.expandvars`: `%name%`, `${name}` and `$name` (ASCII letters,
/// digits, `_`, `-`); `%%` and `$$` are a literal `%` and `$`; nothing inside
/// single quotes expands. An unset or unterminated reference stays as
/// written.
#[cfg_attr(not(windows), allow(dead_code))] // tests run both rule sets everywhere
fn expand_vars_nt(
    text: &str,
    lookup: impl Fn(&str) -> Option<std::ffi::OsString>,
) -> std::ffi::OsString {
    let mut out = std::ffi::OsString::new();
    let mut rest = text;
    while let Some(c) = rest.chars().next() {
        let after = &rest[c.len_utf8()..];
        match c {
            '\'' => match after.find('\'') {
                Some(end) => {
                    out.push(&rest[..end + 2]);
                    rest = &after[end + 1..];
                }
                None => {
                    out.push(rest);
                    rest = "";
                }
            },
            '%' if after.starts_with('%') => {
                out.push("%");
                rest = &after[1..];
            }
            '%' => match after.find('%') {
                Some(end) => {
                    let name = &after[..end];
                    match lookup(name) {
                        Some(value) => out.push(value),
                        None => out.push(&rest[..end + 2]),
                    }
                    rest = &after[end + 1..];
                }
                None => {
                    out.push(rest);
                    rest = "";
                }
            },
            '$' if after.starts_with('$') => {
                out.push("$");
                rest = &after[1..];
            }
            '$' if after.starts_with('{') => match after[1..].find('}') {
                Some(end) => {
                    let name = &after[1..1 + end];
                    match lookup(name) {
                        Some(value) => out.push(value),
                        None => out.push(&rest[..end + 3]),
                    }
                    rest = &after[end + 2..];
                }
                None => {
                    out.push(rest);
                    rest = "";
                }
            },
            '$' => {
                let end = after
                    .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '-'))
                    .unwrap_or(after.len());
                let name = &after[..end];
                match lookup(name) {
                    Some(value) => out.push(value),
                    None => out.push(&rest[..end + 1]),
                }
                rest = &after[end..];
            }
            _ => {
                out.push(&rest[..c.len_utf8()]);
                rest = after;
            }
        }
    }
    out
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Every variable `hub_cache_dir` reads.
    const CACHE_ENV: [&str; 6] = [
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_HOME",
        "XDG_CACHE_HOME",
        "HOME",
        "USERPROFILE",
    ];

    /// Saves [`CACHE_ENV`], clears it, and puts it back on drop, so a failing
    /// assertion cannot leak a fake cache into later tests. Take
    /// `env_lock()` first.
    pub(crate) struct CacheEnv(Vec<(String, Option<std::ffi::OsString>)>);

    impl CacheEnv {
        pub(crate) fn cleared() -> Self {
            Self::cleared_with(std::iter::empty())
        }

        /// [`Self::cleared`], plus `extra` variables (the ones a test's
        /// paths reference).
        pub(crate) fn cleared_with(extra: impl IntoIterator<Item = String>) -> Self {
            let names: Vec<String> = CACHE_ENV
                .iter()
                .map(|name| name.to_string())
                .chain(extra)
                .collect();
            let saved = names
                .into_iter()
                .map(|name| {
                    let value = std::env::var_os(&name);
                    std::env::remove_var(&name);
                    (name, value)
                })
                .collect();
            CacheEnv(saved)
        }
    }

    impl Drop for CacheEnv {
        fn drop(&mut self) {
            for (name, value) in &self.0 {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
        }
    }

    /// Same order as `huggingface_hub.constants.HF_HUB_CACHE`: each variable
    /// wins over every one after it, and only one directory comes back.
    #[test]
    fn hub_cache_dir_follows_huggingface_hub_precedence() {
        let _guard = crate::test_support::env_lock();
        let _env = CacheEnv::cleared();
        let t = tempfile::tempdir().unwrap();
        let p = |name: &str| t.path().join(name);
        std::env::set_var("HOME", p("home"));
        std::env::set_var("USERPROFILE", p("home"));
        std::env::set_var("XDG_CACHE_HOME", p("xdg"));
        std::env::set_var("HF_HOME", p("hf-home"));
        std::env::set_var("HUGGINGFACE_HUB_CACHE", p("legacy"));
        std::env::set_var("HF_HUB_CACHE", p("explicit"));

        assert_eq!(hub_cache_dir(), Some(p("explicit")));
        std::env::remove_var("HF_HUB_CACHE");
        assert_eq!(hub_cache_dir(), Some(p("legacy")));
        std::env::remove_var("HUGGINGFACE_HUB_CACHE");
        assert_eq!(hub_cache_dir(), Some(p("hf-home").join("hub")));
        std::env::remove_var("HF_HOME");
        assert_eq!(
            hub_cache_dir(),
            Some(p("xdg").join("huggingface").join("hub"))
        );
        std::env::remove_var("XDG_CACHE_HOME");
        assert_eq!(
            hub_cache_dir(),
            Some(p("home").join(".cache").join("huggingface").join("hub"))
        );

        // An empty variable counts as unset, and a leading `~` expands.
        std::env::set_var("HF_HUB_CACHE", "");
        std::env::set_var("HF_HOME", "~/models");
        assert_eq!(hub_cache_dir(), Some(p("home").join("models").join("hub")));
    }

    /// Cases checked against Python itself (`posixpath`, `ntpath` and
    /// `huggingface_hub.constants`) by `tests/test_fastembed_pinned_snapshot.py`.
    const HF_CACHE_FIXTURE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/parity/fixtures/hf_hub_cache/cases.json"
    ));

    /// Both rule sets on every platform, so the Windows port is exercised by
    /// the macOS and Linux runs too.
    #[test]
    fn expand_vars_matches_python_posixpath_and_ntpath() {
        let fixture: serde_json::Value = serde_json::from_str(HF_CACHE_FIXTURE).unwrap();
        let section = &fixture["expandvars"];
        let env: std::collections::HashMap<String, String> =
            serde_json::from_value(section["env"].clone()).unwrap();
        let lookup = |name: &str| env.get(name).map(std::ffi::OsString::from);
        for case in section["cases"].as_array().unwrap() {
            let input = case["input"].as_str().unwrap();
            let want = |key: &str| std::ffi::OsString::from(case[key].as_str().unwrap());
            assert_eq!(
                expand_vars_posix(input, lookup),
                want("posix"),
                "posix {input:?}"
            );
            assert_eq!(expand_vars_nt(input, lookup), want("nt"), "nt {input:?}");
        }
    }

    /// The review case among them: `HF_HUB_CACHE=${ROOT}/hub` is
    /// `<ROOT>/hub` in Python, so Rust must not look up or download into a
    /// literal `${ROOT}` directory instead.
    #[test]
    fn hub_cache_dir_matches_the_shared_fixture() {
        let fixture: serde_json::Value = serde_json::from_str(HF_CACHE_FIXTURE).unwrap();
        let section = &fixture["cache_dir"];
        let probe_vars: Vec<String> =
            serde_json::from_value(section["probe_vars"].clone()).unwrap();
        let _guard = crate::test_support::env_lock();
        let _env = CacheEnv::cleared_with(probe_vars.clone());
        let t = tempfile::tempdir().unwrap();
        let (root, home) = (t.path().join("root"), t.path().join("home"));
        let fill = |text: &str| {
            text.replace("{root}", root.to_str().unwrap())
                .replace("{home}", home.to_str().unwrap())
        };
        for case in section["cases"].as_array().unwrap() {
            for name in CACHE_ENV
                .iter()
                .copied()
                .chain(probe_vars.iter().map(String::as_str))
            {
                std::env::remove_var(name);
            }
            std::env::set_var("HOME", &home);
            std::env::set_var("USERPROFILE", &home);
            for (name, value) in case["env"].as_object().unwrap() {
                std::env::set_var(name, fill(value.as_str().unwrap()));
            }
            let expected = match &case["expected"] {
                serde_json::Value::Object(by_os) => {
                    &by_os[if cfg!(windows) { "nt" } else { "posix" }]
                }
                expected => expected,
            };
            assert_eq!(
                hub_cache_dir(),
                Some(PathBuf::from(fill(expected.as_str().unwrap()))),
                "{}",
                case["name"]
            );
        }
    }

    /// With no `HOME` (a service started with a bare environment) the home
    /// directory still comes from the user database, as Python's
    /// `expanduser` does, instead of leaving the scorer with no cache.
    #[cfg(all(feature = "ml", unix))]
    #[test]
    fn hub_cache_dir_without_home_uses_the_user_database() {
        let _guard = crate::test_support::env_lock();
        let _env = CacheEnv::cleared();
        let dir = hub_cache_dir().expect("a cache directory without HOME");
        assert!(dir.is_absolute(), "{}", dir.display());
        assert!(
            dir.ends_with(Path::new(".cache").join("huggingface").join("hub")),
            "{}",
            dir.display()
        );
    }

    /// The fixture's `token` cases, which the Python half checks against
    /// `huggingface_hub.get_token()`.
    #[test]
    fn hf_token_matches_the_shared_fixture() {
        let fixture: serde_json::Value = serde_json::from_str(HF_CACHE_FIXTURE).unwrap();
        let probe_vars: Vec<String> =
            serde_json::from_value(fixture["cache_dir"]["probe_vars"].clone()).unwrap();
        let _guard = crate::test_support::env_lock();
        let _env = CacheEnv::cleared_with(probe_vars.clone());
        for case in fixture["token"]["cases"].as_array().unwrap() {
            let t = tempfile::tempdir().unwrap();
            let (root, home) = (t.path().join("root"), t.path().join("home"));
            let fill = |text: &str| {
                text.replace("{root}", root.to_str().unwrap())
                    .replace("{home}", home.to_str().unwrap())
            };
            for name in CACHE_ENV
                .iter()
                .copied()
                .chain(probe_vars.iter().map(String::as_str))
            {
                std::env::remove_var(name);
            }
            std::env::set_var("HOME", &home);
            std::env::set_var("USERPROFILE", &home);
            for (name, value) in case["env"].as_object().unwrap() {
                std::env::set_var(name, fill(value.as_str().unwrap()));
            }
            for (path, content) in case["files"].as_object().unwrap() {
                let path = PathBuf::from(fill(path));
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                std::fs::write(path, content.as_str().unwrap()).unwrap();
            }
            assert_eq!(
                hf_token().as_deref(),
                case["expected"].as_str(),
                "{}",
                case["name"]
            );
        }
    }

    #[test]
    fn hf_endpoint_drops_trailing_slashes_like_python() {
        let _guard = crate::test_support::env_lock();
        let saved = std::env::var_os("HF_ENDPOINT");
        std::env::remove_var("HF_ENDPOINT");
        assert_eq!(hf_endpoint(), "https://huggingface.co");
        std::env::set_var("HF_ENDPOINT", "https://mirror.example.com//");
        assert_eq!(hf_endpoint(), "https://mirror.example.com");
        match saved {
            Some(v) => std::env::set_var("HF_ENDPOINT", v),
            None => std::env::remove_var("HF_ENDPOINT"),
        }
    }

    /// The token is withheld only where it would cross the network unencrypted.
    #[test]
    fn plain_http_off_loopback_is_the_only_clear_text_endpoint() {
        for endpoint in [
            "http://mirror.example.com",
            "HTTP://mirror.example.com:8080/hub",
            "http://10.0.0.5",
            "http://user@hub.internal",
            "http://localhost.example.com",
        ] {
            assert!(sends_in_clear_text(endpoint), "{endpoint}");
        }
        for endpoint in [
            "https://huggingface.co",
            "https://mirror.example.com",
            "http://localhost:8000",
            "http://127.0.0.1:9",
            "http://127.8.0.1",
            "http://[::1]:8080/",
        ] {
            assert!(!sends_in_clear_text(endpoint), "{endpoint}");
        }
    }

    /// A commit SHA names the snapshot directly; a branch goes through its
    /// ref, as in Python's cache layout.
    #[test]
    fn snapshot_path_resolves_commits_directly_and_branches_through_refs() {
        let cache = tempfile::tempdir().unwrap();
        let repo = cache.path().join("models--owner--name");
        std::fs::create_dir_all(repo.join("refs")).unwrap();
        std::fs::write(repo.join("refs").join("main"), "abc123\n").unwrap();
        let snapshots = repo.join("snapshots");
        assert_eq!(
            snapshot_path(cache.path(), "owner/name", "def456"),
            snapshots.join("def456")
        );
        assert_eq!(
            snapshot_path(cache.path(), "owner/name", "main"),
            snapshots.join("abc123")
        );
    }

    #[test]
    fn pin_can_be_turned_off() {
        let _guard = crate::test_support::env_lock();
        let saved = std::env::var_os("HEADROOM_HF_PIN");
        for (value, disabled) in [("", false), ("on", false), ("off", true), (" No ", true)] {
            std::env::set_var("HEADROOM_HF_PIN", value);
            assert_eq!(pin_disabled(), disabled, "{value:?}");
            assert_eq!(pinned_or_main("sha"), if disabled { "main" } else { "sha" });
        }
        match saved {
            Some(v) => std::env::set_var("HEADROOM_HF_PIN", v),
            None => std::env::remove_var("HEADROOM_HF_PIN"),
        }
    }
}
