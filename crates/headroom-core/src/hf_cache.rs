//! The HuggingFace hub cache, located and read the way Python's
//! `huggingface_hub` does, so the Rust loaders (the relevance embedder and
//! Kompress) find the files the Python side downloaded and pinned.
//!
//! Two halves, both network-free: [`hub_cache_dir`] picks the one cache
//! directory Python would use, and [`snapshot_path`] resolves a repo at a
//! revision inside it. The pinned revisions themselves live next to each
//! loader and are kept equal to `_PINNED_REVISIONS` in
//! `headroom/onnx_runtime.py` by tests on the Python side.

use std::path::{Path, PathBuf};

/// True when `HEADROOM_HF_PIN` turns revision pinning off, so pinned repos
/// float on `main` (same values as Python's `onnx_runtime._resolve_revision`).
pub(crate) fn pin_disabled() -> bool {
    let pin = std::env::var("HEADROOM_HF_PIN").unwrap_or_default();
    matches!(
        pin.trim().to_ascii_lowercase().as_str(),
        "off" | "0" | "false" | "no"
    )
}

/// The revision to load for a repo pinned at `pinned`: the pin, or `main`
/// when `HEADROOM_HF_PIN` turns pinning off.
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
    let var = |name: &str| {
        std::env::var_os(name)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    let home = std::env::home_dir();
    let expand = |path: PathBuf| expand_vars(&expand_user(path, home.as_deref()));
    let hub = match var("HF_HUB_CACHE").or_else(|| var("HUGGINGFACE_HUB_CACHE")) {
        Some(dir) => dir,
        None => {
            let hf_home = match var("HF_HOME") {
                Some(dir) => dir,
                None => var("XDG_CACHE_HOME")
                    .or_else(|| home.as_ref().map(|home| home.join(".cache")))?
                    .join("huggingface"),
            };
            expand(hf_home).join("hub")
        }
    };
    Some(expand(hub))
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
