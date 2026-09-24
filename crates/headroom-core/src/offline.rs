//! Air-gap / no-egress master switch (`HEADROOM_OFFLINE`) — Rust half.
//!
//! The Python side lives in `headroom/offline.py`. This is the same switch
//! read by the same environment variable with the same truthiness rules, so a
//! deployment that sets `HEADROOM_OFFLINE=1` gets identical behaviour whether
//! the egress attempt originates in the Python proxy or in a Rust core path.
//!
//! Two functions, mirroring the Python module deliberately:
//!
//! - [`is_offline`] — the predicate, for code that wants to take a different
//!   route (use a cached artifact, skip an optional refresh).
//! - [`guard_egress`] — the chokepoint, for code that is about to open a
//!   socket. It returns [`OfflineEgressBlocked`], a distinct error type, so a
//!   caller that otherwise degrades gracefully on network failure can still
//!   tell "the operator air-gapped this box" from "the network was flaky" and
//!   refuse loudly instead of silently falling back.
//!
//! # Parity contract
//!
//! `HEADROOM_OFFLINE` is true for `1`, `true`, `yes`, `on` — trimmed and
//! ASCII-case-insensitive — and false for anything else, including unset and
//! empty. Any change here must be mirrored in `headroom/offline.py`'s
//! `_TRUE_VALUES`, and vice versa: the two implementations are a pair, and a
//! deployment that reads as offline to Python but online to Rust is precisely
//! the failure this module exists to prevent.
//!
//! The parity covers **normalisation**, not just the accepted values. This
//! used to call `str::trim()` while Python called `str.strip()`, and those are
//! not the same set: Python's strips U+001C-U+001F (the ASCII file/group/
//! record/unit separators) because `str.isspace()` includes them, Rust's does
//! not because the Unicode `White_Space` property does not. `HEADROOM_OFFLINE`
//! set to `"\x1c1"` read as offline to Python and online to Rust — one
//! environment variable, one process air-gapped and the other not. Both sides
//! now trim exactly [`TRIM_CHARS`].

use std::env;

use thiserror::Error;

/// The environment variable that selects fully-offline operation.
pub const OFFLINE_ENV: &str = "HEADROOM_OFFLINE";

/// Values that read as "yes, offline". Kept byte-identical to the Python
/// side's `_TRUE_VALUES` (see the parity contract in the module docs).
const TRUE_VALUES: [&str; 4] = ["1", "true", "yes", "on"];

/// Whitespace trimmed off the raw value before matching. Enumerated rather
/// than left to `str::trim()`, which is the Unicode `White_Space` property and
/// does not agree with Python's `str.strip()`. Kept byte-identical to the
/// Python side's `_TRIM_CHARS` (see the parity contract in the module docs).
const TRIM_CHARS: [char; 6] = [' ', '\t', '\n', '\r', '\u{b}', '\u{c}'];

/// An egress attempt was refused because `HEADROOM_OFFLINE` is in force.
///
/// Carries the human-readable `purpose` and `destination` handed to
/// [`guard_egress`] so the operator learns which feature to turn off rather
/// than just that "something" was blocked.
#[derive(Debug, Clone, Error)]
#[error(
    "{OFFLINE_ENV} is set: refusing outbound network access for {purpose} to \
     {destination}. Unset {OFFLINE_ENV}, or turn off the feature that needs \
     this connection."
)]
pub struct OfflineEgressBlocked {
    pub purpose: String,
    pub destination: String,
}

/// Return `true` when `HEADROOM_OFFLINE` selects fully-offline operation.
pub fn is_offline() -> bool {
    match env::var(OFFLINE_ENV) {
        Ok(raw) => {
            let normalized = raw.trim_matches(TRIM_CHARS.as_slice()).to_ascii_lowercase();
            TRUE_VALUES.contains(&normalized.as_str())
        }
        // Unset, or not valid UTF-8 — neither is an opt-in to offline mode.
        Err(_) => false,
    }
}

/// The chokepoint every Rust egress path must call before opening a socket.
///
/// `Ok(())` when online; `Err(OfflineEgressBlocked)` when `HEADROOM_OFFLINE`
/// is set. Call it *before* constructing the HTTP client, not merely before
/// the request — several clients (`ureq`, `reqwest`) resolve DNS or warm a
/// connection pool during setup, so a guard placed at the request would leak
/// the very packets the air-gap switch promises not to send.
pub fn guard_egress(purpose: &str, destination: &str) -> Result<(), OfflineEgressBlocked> {
    if is_offline() {
        return Err(OfflineEgressBlocked {
            purpose: purpose.to_string(),
            destination: destination.to_string(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::env_lock;

    #[test]
    fn unset_is_online() {
        let _guard = env_lock();
        env::remove_var(OFFLINE_ENV);
        assert!(!is_offline());
        assert!(guard_egress("anything", "https://example.invalid").is_ok());
    }

    #[test]
    fn truthy_values_match_python() {
        let _guard = env_lock();
        // Same set, same trimming, same case-insensitivity as
        // headroom/offline.py::_TRUE_VALUES. If this list drifts, the two
        // halves of the switch disagree and the air-gap has a hole.
        for raw in ["1", "true", "TRUE", "Yes", " on ", "\ton\n"] {
            env::set_var(OFFLINE_ENV, raw);
            assert!(is_offline(), "{raw:?} should read as offline");
        }
        env::remove_var(OFFLINE_ENV);
    }

    #[test]
    fn falsy_values_stay_online() {
        let _guard = env_lock();
        for raw in ["", "0", "off", "no", "false", "maybe", " "] {
            env::set_var(OFFLINE_ENV, raw);
            assert!(!is_offline(), "{raw:?} should read as online");
        }
        env::remove_var(OFFLINE_ENV);
    }

    /// Normalisation parity, not just value parity. Every case here is
    /// duplicated verbatim in `tests/test_offline_egress_chokepoint.py`'s
    /// `test_normalisation_matches_the_rust_side`, and the two must agree.
    ///
    /// The `\x1c` cases are the regression: they are whitespace to Python's
    /// `str.strip()` and not to Rust's `str::trim()`, so before `TRIM_CHARS`
    /// existed `"\x1c1"` air-gapped the Python proxy and left the Rust core
    /// dialling out.
    #[test]
    fn normalisation_matches_python() {
        let _guard = env_lock();
        for (raw, expected) in [
            ("1", true),
            (" 1 ", true),
            ("\t1\n", true),
            ("\r\n TRUE \r\n", true),
            ("\u{b}yes\u{c}", true),
            ("\u{1c}1", false),
            ("1\u{1f}", false),
            ("\u{a0}1", false),
            ("\u{2007}on", false),
            ("", false),
            (" ", false),
        ] {
            env::set_var(OFFLINE_ENV, raw);
            assert_eq!(is_offline(), expected, "{raw:?}");
        }
        env::remove_var(OFFLINE_ENV);
    }

    #[test]
    fn guard_reports_purpose_and_destination() {
        let _guard = env_lock();
        env::set_var(OFFLINE_ENV, "1");
        let err = guard_egress("HuggingFace tokenizer download", "hf.co/acme/model")
            .expect_err("guard must refuse while offline");
        assert_eq!(err.purpose, "HuggingFace tokenizer download");
        assert_eq!(err.destination, "hf.co/acme/model");
        let rendered = err.to_string();
        assert!(rendered.contains(OFFLINE_ENV), "{rendered}");
        assert!(
            rendered.contains("HuggingFace tokenizer download"),
            "{rendered}"
        );
        assert!(rendered.contains("hf.co/acme/model"), "{rendered}");
        env::remove_var(OFFLINE_ENV);
    }
}
