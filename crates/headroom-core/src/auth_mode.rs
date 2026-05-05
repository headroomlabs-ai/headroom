//! Auth-mode classifier — Phase F PR-F1.
//!
//! A single pure function turns the inbound request `HeaderMap` into one
//! of three classes that drive every downstream compression / cache /
//! header policy decision:
//!
//! - [`AuthMode::Payg`]: caller pays per token (Anthropic API key,
//!   OpenAI `sk-...`, Gemini `x-goog-api-key`, x-api-key). Aggressive
//!   compression saves them money — turn it all on.
//! - [`AuthMode::OAuth`]: caller is on a fixed-cost subscription / IAM
//!   plan (Claude Pro OAuth, Codex Enterprise, Cursor Pro, Bedrock
//!   IAM-signed downstream, Vertex ADC). Per-token cost is opaque to
//!   them; cache-safety is paramount because OAuth scopes pin to
//!   `(account, model, session)` and beta-header drift voids them.
//! - [`AuthMode::Subscription`]: caller is a UX-bound CLI / IDE
//!   (Claude Code, ChatGPT Plus, Cursor, Copilot, Antigravity).
//!   Provider rate-limits by request count; programmatic-fingerprint
//!   detection means we MUST look like the upstream agent (preserve
//!   `User-Agent`, never inject `X-Headroom-*`, never strip
//!   `accept-encoding`).
//!
//! See `~/.claude/projects/.../memory/project_auth_mode_compression_nuances.md`
//! for the user-decision rationale (2026-05-01).
//!
//! The classifier is **pure** (no I/O, no allocation beyond a single
//! lowercase copy of the User-Agent), runs in <10us per call, and
//! NEVER panics on malformed headers — non-UTF-8 values fall through to
//! the safe default [`AuthMode::Payg`] with a `tracing::warn!` event so
//! operators can spot bad clients in the log stream without taking the
//! proxy down.

use http::HeaderMap;

/// Three auth-mode classes Headroom routes compression policy through.
///
/// `Copy` because the value is passed through dozens of compression
/// decisions per request; cloning a 1-byte enum is cheaper than holding
/// a reference across `await` points. `Hash + Eq` so it can key the
/// per-tenant TOIN aggregation map (Phase F PR-F3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AuthMode {
    /// Pay-as-you-go API key. Aggressive live-zone compression OK.
    Payg,
    /// OAuth bearer / Bedrock IAM / Vertex ADC. Passthrough-prefer:
    /// no auto-`cache_control`, no auto-`prompt_cache_key`, no lossy
    /// compressors. Lossless-only path.
    OAuth,
    /// Subscription-bound CLI / IDE. Stealth: same as OAuth +
    /// preserve `accept-encoding`, never strip; never inject
    /// `X-Headroom-*`; never mutate `User-Agent`.
    Subscription,
}

impl AuthMode {
    /// Lower-snake-case label suitable for structured-log fields and
    /// metric labels. Stable wire format — Python parity tests check
    /// this exact string.
    pub fn as_str(self) -> &'static str {
        match self {
            AuthMode::Payg => "payg",
            AuthMode::OAuth => "oauth",
            AuthMode::Subscription => "subscription",
        }
    }
}

/// User-Agent prefixes that identify a UX-bound CLI / IDE.
///
/// Lives at module scope (not inside `classify`) so:
/// 1. The compiler can constant-fold the slice into the rodata segment.
/// 2. A future PR can swap this for a configurable list (Phase F
///    follow-up) without touching the function body.
/// 3. Adding a new client = one-line edit here, no logic change.
///
/// Match is `str::contains` against a lowercased copy of the UA — so
/// the prefix can appear anywhere in the value (Anthropic CLIs prefix
/// their own UA with the parent agent's UA in some cases).
const SUBSCRIPTION_UA_PREFIXES: &[&str] = &[
    "claude-cli/",
    "claude-code/",
    "codex-cli/",
    "cursor/",
    "claude-vscode/",
    "github-copilot/",
    "anthropic-cli/",
    "antigravity/",
];

/// Classify the auth mode of an inbound request from its headers.
///
/// Decision order (most-specific signal wins):
///
/// 1. **Subscription UA prefix** → [`AuthMode::Subscription`].
///    The CLI's own auth-mode wins over the bearer token shape it
///    happens to be carrying — a Claude Code session uses a
///    `sk-ant-oat-*` token but is a subscription client, not OAuth.
/// 2. **`Authorization: Bearer sk-ant-oat-*`** → [`AuthMode::OAuth`]
///    (Claude Pro / Max OAuth). Checked before the broader `sk-` PAYG
///    rule because `sk-ant-oat-` shares the `sk-` prefix.
/// 3. **`Authorization: Bearer sk-ant-api*` or `Bearer sk-*`** →
///    [`AuthMode::Payg`] (Anthropic / OpenAI API key).
/// 4. **`Authorization: Bearer <jwt>`** (3 dot-separated segments) →
///    [`AuthMode::OAuth`] (Codex / Cursor / Copilot OAuth).
/// 5. **`Authorization` present but not `Bearer ...`** →
///    [`AuthMode::OAuth`] (AWS SigV4 `AWS4-HMAC-SHA256 ...` →
///    Bedrock; any other non-Bearer scheme is presumed
///    passthrough-prefer too).
/// 6. **`x-api-key` present** → [`AuthMode::Payg`] (Anthropic API key
///    style).
/// 7. **`x-goog-api-key` present** → [`AuthMode::Payg`] (Gemini key).
/// 8. **Default** → [`AuthMode::Payg`] (safest default; aggressive
///    compression on a misclassified request just costs us a re-run,
///    not a revoked subscription).
///
/// # Performance
///
/// One owned `String` allocation for the lowercase UA copy. All other
/// matches are zero-allocation `str::starts_with` / `str::contains` /
/// `str::split('.').count()`. Bench at
/// `crates/headroom-core/benches/auth_mode.rs` asserts <10us / call.
pub fn classify(headers: &HeaderMap) -> AuthMode {
    // ── User-Agent ───────────────────────────────────────────────
    // Subscription clients identify by UA prefix; this is the most
    // specific signal because the same OAuth token shape appears in
    // both Claude Pro (web) and Claude Code (CLI), and only the UA
    // tells them apart. Read once, lowercase once.
    let ua_owned = match headers.get("user-agent") {
        Some(value) => match value.to_str() {
            Ok(s) => s.to_ascii_lowercase(),
            Err(_) => {
                tracing::warn!(
                    event = "auth_mode_classify_unparseable_user_agent",
                    "non-UTF-8 user-agent header; falling through to bearer-token classification"
                );
                String::new()
            }
        },
        None => String::new(),
    };
    if SUBSCRIPTION_UA_PREFIXES
        .iter()
        .any(|prefix| ua_owned.contains(prefix))
    {
        return AuthMode::Subscription;
    }

    // ── Authorization header ─────────────────────────────────────
    // We must NOT log the value. `to_str` returns an `&str` with the
    // same lifetime as the `HeaderMap`, so no copy here.
    let auth = match headers.get("authorization") {
        Some(value) => match value.to_str() {
            Ok(s) => s,
            Err(_) => {
                tracing::warn!(
                    event = "auth_mode_classify_unparseable_authorization",
                    "non-UTF-8 authorization header; falling back to default Payg"
                );
                ""
            }
        },
        None => "",
    };

    if let Some(token) = auth.strip_prefix("Bearer ") {
        // Order matters: the OAuth shape `sk-ant-oat-*` shares a
        // prefix with `sk-ant-api*` only at `sk-ant-`, so we check
        // the OAuth shape FIRST. Then the broad PAYG shapes.
        if token.starts_with("sk-ant-oat-") {
            return AuthMode::OAuth;
        }
        if token.starts_with("sk-ant-api") || token.starts_with("sk-") {
            return AuthMode::Payg;
        }
        // JWT: classic three-segment `header.payload.signature`.
        // We don't validate the JWT — just count dot-separated
        // segments. This catches Codex / Cursor / Copilot OAuth.
        if token.split('.').count() >= 3 {
            return AuthMode::OAuth;
        }
        // Unknown bearer shape — fall through to header-based
        // detection below; ultimately defaults to Payg.
    } else if !auth.is_empty() {
        // Authorization is present but NOT `Bearer ...` — most
        // commonly AWS SigV4 (`AWS4-HMAC-SHA256 ...`) on a Bedrock
        // request, or a `Basic ...` from a custom proxy chain. We
        // treat all such non-Bearer schemes as passthrough-prefer.
        // The IAM / signed flow is opaque to us; we never strip or
        // mutate the value — just classify the policy.
        return AuthMode::OAuth;
    }

    // ── Vendor-specific API-key headers ──────────────────────────
    // Anthropic API-key style. Direct PAYG; same compression policy
    // as a `Bearer sk-ant-api...`.
    if headers.contains_key("x-api-key") {
        return AuthMode::Payg;
    }
    // Gemini API key. Same PAYG semantics.
    if headers.contains_key("x-goog-api-key") {
        return AuthMode::Payg;
    }

    // ── Default ──────────────────────────────────────────────────
    // Anything else: assume PAYG. Misclassifying a non-PAYG client
    // as PAYG only over-compresses; under-compressing a PAYG client
    // would leave money on the table, which is worse for the
    // OSS-default user.
    AuthMode::Payg
}

#[cfg(test)]
mod inline_tests {
    //! Smoke tests inlined alongside the function so `cargo test -p
    //! headroom-core --lib` exercises the helper without pulling in
    //! the integration-test binary. The exhaustive test matrix lives
    //! in `crates/headroom-core/tests/auth_mode.rs`.

    use super::*;
    use http::HeaderValue;

    #[test]
    fn enum_as_str_is_stable() {
        // Python parity tests assert these exact strings.
        assert_eq!(AuthMode::Payg.as_str(), "payg");
        assert_eq!(AuthMode::OAuth.as_str(), "oauth");
        assert_eq!(AuthMode::Subscription.as_str(), "subscription");
    }

    #[test]
    fn empty_headers_default_to_payg() {
        // No Authorization, no x-api-key, no x-goog-api-key, no UA →
        // safest default is PAYG. The bedrock OAuth branch fires only
        // when there's a positive non-Bearer Authorization signal.
        let headers = HeaderMap::new();
        assert_eq!(classify(&headers), AuthMode::Payg);
    }

    #[test]
    fn unparseable_auth_falls_back_to_default() {
        // Non-UTF-8 Authorization header — the warn! fires but we
        // do NOT panic. With no other distinguishing headers, we
        // fall through to the default → Payg.
        let mut headers = HeaderMap::new();
        headers.insert(
            "authorization",
            HeaderValue::from_bytes(b"\xFFnope").unwrap(),
        );
        assert_eq!(classify(&headers), AuthMode::Payg);
    }
}
