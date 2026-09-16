//! Session-sticky provider beta headers — Rust port of the Python
//! proxy's `SessionBetaTracker` (PR-A6, `headroom/proxy/helpers.py`).
//!
//! ## Why
//!
//! Provider beta headers (`anthropic-beta`, `openai-beta`) are part of
//! the request bytes that determine the upstream prefix-cache key.
//! Interactive clients (Claude Code, Codex CLI) MAY drop a beta token
//! between turn N and turn N+1 of the same conversation; the cache hot
//! zone is positional, so the next turn's prefix hashes differently and
//! the prefix-cache read misses — the customer silently pays for a full
//! prompt re-write. The Python proxy defeats this with a bounded LRU
//! tracker that unions the client's tokens with every token previously
//! seen for the same `(provider, session)` and forwards the union.
//!
//! The Rust proxy replaces the Python request path in Phase H, which
//! deletes `SessionBetaTracker` with the rest of
//! `headroom/proxy/helpers.py`. Without this port the protection —
//! and its documented operator contract
//! (`docs/content/docs/configuration.mdx`, "Session Beta Header
//! Tracking") — would silently not survive the migration.
//!
//! ## Behaviour contract (parity with Python)
//!
//! - Union client tokens with previously-seen tokens for the session,
//!   preserving first-seen order; case-insensitive dedup where the
//!   first-seen casing wins.
//! - Keyed by `(provider, session)` so the same session id against
//!   Anthropic and OpenAI upstreams keeps independent token sets.
//! - Bounded LRU (`BETA_TRACKER_CAPACITY` sessions): lookups touch
//!   recency, overflow evicts the oldest session.
//! - The tracker only ever records tokens the client itself sent.
//!   Headroom-added tokens (e.g. memory-tool betas on the Python
//!   path) are NOT recorded — the forwarded union is always a subset
//!   of values this client already put on the wire, which is what
//!   keeps the mechanism consistent with the subscription-stealth
//!   invariant (REALIGNMENT invariant #10: "no beta drift").
//!
//! The operator opt-out lives at the call site: when
//! `Config::beta_header_sticky` is `disabled` the proxy skips the
//! tracker entirely and forwards the client header verbatim (the
//! Python proxy's `HEADROOM_BETA_HEADER_STICKY=disabled` diagnostic
//! mode). That gate is per REALIGNMENT build constraint #4 an explicit
//! loud opt-in, not a silent fallback.
//!
//! Session identity comes from
//! [`super::drift_detector::derive_session_key`] — the same
//! conversation-aware key the drift detector uses (explicit
//! `x-headroom-session-id` when the client declares it, otherwise
//! credential/IP arms folded with a first-message conversation
//! discriminator).
//!
//! ## Divergence from Python: per-conversation, not per-(model, system)
//!
//! The Python tracker keys on the store session id — explicit header,
//! else a hash of `(model, leading system prompt)` — so all parallel
//! conversations sharing a model + system prompt (a Claude Code
//! session and every one of its subagents) share ONE token union and
//! cross-inherit each other's tokens. This port keys on the drift
//! detector's conversation-aware key instead, so each conversation
//! keeps its own union; the integration test
//! `separate_conversations_do_not_leak_tokens` pins that. Deliberate:
//! the `(model, system)` bucket conflating parallel agentic
//! conversations is the exact defect #2085 / #2193 / #2301 chased out
//! of the other session-sticky subsystems. The cost is losing
//! Python's accidental cross-conversation repair (conversation B
//! turn 1 inheriting a token only conversation A ever sent); each
//! conversation's stickiness now starts from its own first sighting,
//! which is also the only variant that can't leak one tenant-visible
//! experiment token into an unrelated conversation's request bytes.

use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

use http::header::{HeaderMap, HeaderValue};
use lru::LruCache;

use super::drift_detector::session_key_log_prefix;

/// Maximum number of `(provider, session)` entries tracked. Sessions
/// are keyed per conversation (see module docs), so the working set is
/// the number of concurrently active conversations — same sizing
/// rationale as the drift detector's capacity. Eviction cost is
/// re-learning a live session's dropped tokens from scratch (the next
/// turn forwards the client value verbatim), not a lost request.
pub const BETA_TRACKER_CAPACITY: usize = 1000;

/// Upstream namespace for a tracked beta-token set. Mirrors the
/// Python tracker's `provider` string key ("anthropic" / "openai").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BetaProvider {
    /// `/v1/messages` — `anthropic-beta` header.
    Anthropic,
    /// `/v1/chat/completions` and `/v1/responses` — `openai-beta`
    /// header. One namespace for both endpoints, matching the Python
    /// proxy's single `provider="openai"` key.
    OpenAi,
}

impl BetaProvider {
    /// Stable lower-case label for log fields; matches the Python
    /// tracker's provider strings.
    pub fn as_str(self) -> &'static str {
        match self {
            BetaProvider::Anthropic => "anthropic",
            BetaProvider::OpenAi => "openai",
        }
    }

    /// The request header this provider's beta tokens travel in.
    pub fn header_name(self) -> &'static str {
        match self {
            BetaProvider::Anthropic => "anthropic-beta",
            BetaProvider::OpenAi => "openai-beta",
        }
    }
}

/// Split a comma-separated beta-header value into trimmed, non-empty
/// tokens. Port of the Python `split_beta_tokens` helper.
pub fn split_beta_tokens(value: Option<&str>) -> Vec<String> {
    value
        .unwrap_or("")
        .split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(str::to_string)
        .collect()
}

/// Per-session ordered token lists, keyed by `(provider, session)`.
type SessionTokenCache = LruCache<(BetaProvider, String), Vec<String>>;

/// Bounded LRU of beta tokens observed per `(provider, session)`.
///
/// Cloning shares the underlying map (`Arc`), mirroring
/// [`super::drift_detector::DriftState`] so one instance lives in
/// `AppState` and clones freely into every handler path.
#[derive(Clone)]
pub struct BetaStickyState {
    sessions: Arc<Mutex<SessionTokenCache>>,
}

impl BetaStickyState {
    /// Create a tracker bounded to `capacity` sessions.
    ///
    /// # Panics
    ///
    /// Panics when `capacity == 0`, mirroring `DriftState::new` (the
    /// Python tracker raises `ValueError` on a non-positive bound).
    pub fn new(capacity: usize) -> Self {
        let cap = NonZeroUsize::new(capacity).expect("BetaStickyState capacity must be > 0");
        Self {
            sessions: Arc::new(Mutex::new(LruCache::new(cap))),
        }
    }

    /// Union `client_value`'s tokens with the session's previously
    /// seen tokens, update the session, and return the merged
    /// comma-separated value (possibly empty). Port of the Python
    /// `SessionBetaTracker.record_and_get_sticky_betas`.
    ///
    /// On a poisoned lock the tracker fails open: the client value is
    /// returned verbatim (trimmed) and state is left untouched —
    /// never drop or delay the request for a telemetry-adjacent
    /// protection.
    pub fn record_and_get_sticky_betas(
        &self,
        provider: BetaProvider,
        session_key: &str,
        client_value: Option<&str>,
    ) -> String {
        let client_tokens = split_beta_tokens(client_value);

        let mut sessions = match self.sessions.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                tracing::warn!(
                    event = "beta_sticky_lock_poisoned",
                    provider = provider.as_str(),
                    "beta tracker lock poisoned; forwarding client value verbatim"
                );
                drop(poisoned);
                return client_tokens.join(",");
            }
        };

        let key = (provider, session_key.to_string());
        // `get_mut` touches LRU recency on hit, mirroring the Python
        // tracker's move-to-end.
        if let Some(merged) = sessions.get_mut(&key) {
            // Dedup is case-insensitive with the first-seen casing
            // winning. Header values reaching this point are visible
            // ASCII (`HeaderValue::to_str` rejects anything else), so
            // ASCII lowercasing matches Python's `str.lower()` over
            // the reachable domain.
            let mut seen: HashSet<String> = merged.iter().map(|t| t.to_ascii_lowercase()).collect();
            for token in client_tokens {
                if seen.insert(token.to_ascii_lowercase()) {
                    merged.push(token);
                }
            }
            return merged.join(",");
        }

        let mut merged: Vec<String> = Vec::with_capacity(client_tokens.len());
        let mut seen: HashSet<String> = HashSet::with_capacity(client_tokens.len());
        for token in client_tokens {
            if seen.insert(token.to_ascii_lowercase()) {
                merged.push(token);
            }
        }
        let joined = merged.join(",");
        // `put` on a fresh key evicts the oldest entry once the cache
        // is at capacity — the Python tracker's bounded-LRU overflow
        // pop. Sessions that never sent a beta token still occupy a
        // slot (Python stores their empty list too); the cost is one
        // LRU entry, the benefit is identical recency behaviour.
        sessions.put(key, merged);
        joined
    }

    /// Number of tracked sessions (test observability).
    #[cfg(test)]
    fn active_sessions(&self) -> usize {
        self.sessions.lock().map(|c| c.len()).unwrap_or(0)
    }
}

/// Count tokens in a raw header value without allocating a `Vec`
/// (log-field helper; same tokenization as [`split_beta_tokens`]).
fn count_beta_tokens(value: Option<&str>) -> usize {
    value
        .unwrap_or("")
        .split(',')
        .filter(|t| !t.trim().is_empty())
        .count()
}

/// Record the client's beta header for this `(provider, session)` and
/// rewrite the upstream-bound header to the session union when they
/// differ. The full merge site: reads `provider.header_name()` from
/// `outgoing_headers`, unions via the tracker, mutates the map in
/// place. Mirrors the Python handler block (anthropic.py PR-A6):
/// rewrite only when the union is non-empty and differs from the
/// client value; an absent client header gains the union; a session
/// with no tokens anywhere stays header-less.
///
/// Fail-open contract: a client value that isn't visible ASCII is
/// forwarded verbatim and nothing is recorded (never rewrite what we
/// can't faithfully parse); an unencodable union (unreachable — every
/// token came from a parsed header value) logs and forwards verbatim.
///
/// Logging: counts only — beta tokens can carry experiment IDs the
/// user hasn't opted to share with Headroom logs (Python
/// `log_beta_header_merge` contract). Python logs every merge at
/// info; here the no-op case drops to debug, matching the drift
/// detector's silent-on-stable precedent, so an info-level
/// `beta_header_merge` always marks an actual cache-affecting
/// rewrite.
pub fn apply_sticky_betas(
    tracker: &BetaStickyState,
    provider: BetaProvider,
    session_key: &str,
    outgoing_headers: &mut HeaderMap,
    request_id: &str,
) {
    let header_name = provider.header_name();
    // Join repeated field lines with "," per RFC 9110 §5.3 list
    // semantics BEFORE recording, so a client sending two beta lines
    // has both recorded and a later rewrite (which `insert`s a single
    // line, dropping the others) can never shrink the upstream token
    // set mid-conversation.
    let mut parts: Vec<&str> = Vec::new();
    for raw in outgoing_headers.get_all(header_name) {
        match raw.to_str() {
            Ok(s) => parts.push(s),
            Err(_) => {
                tracing::debug!(
                    event = "beta_header_merge_skipped",
                    request_id = %request_id,
                    provider = provider.as_str(),
                    reason = "non_ascii_header_value",
                    "client beta header is not visible ASCII; forwarding verbatim"
                );
                return;
            }
        }
    }
    let client_value: Option<String> = if parts.is_empty() {
        None
    } else {
        Some(parts.join(","))
    };

    let sticky =
        tracker.record_and_get_sticky_betas(provider, session_key, client_value.as_deref());
    let rewritten = !sticky.is_empty() && sticky != client_value.as_deref().unwrap_or("");
    if rewritten {
        match HeaderValue::from_str(&sticky) {
            Ok(value) => {
                outgoing_headers.insert(header_name, value);
            }
            Err(error) => {
                tracing::warn!(
                    event = "beta_header_merge_skipped",
                    request_id = %request_id,
                    provider = provider.as_str(),
                    reason = "unencodable_union",
                    error = %error,
                    "sticky beta union not encodable as a header value"
                );
                return;
            }
        }
    }

    let client_betas = count_beta_tokens(client_value.as_deref());
    let sticky_betas = count_beta_tokens(Some(&sticky));
    if rewritten {
        tracing::info!(
            event = "beta_header_merge",
            request_id = %request_id,
            provider = provider.as_str(),
            session_key_hash = %session_key_log_prefix(session_key),
            client_betas,
            sticky_betas,
            "session-sticky beta merge rewrote the upstream header"
        );
    } else {
        tracing::debug!(
            event = "beta_header_merge",
            request_id = %request_id,
            provider = provider.as_str(),
            session_key_hash = %session_key_log_prefix(session_key),
            client_betas,
            sticky_betas,
            "session-sticky beta merge (no-op)"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // split_beta_tokens — port of the Python tokenizer contract.
    // -----------------------------------------------------------------

    #[test]
    fn split_none_and_empty_yield_no_tokens() {
        assert!(split_beta_tokens(None).is_empty());
        assert!(split_beta_tokens(Some("")).is_empty());
        assert!(split_beta_tokens(Some("   ")).is_empty());
        assert!(split_beta_tokens(Some(",, ,")).is_empty());
    }

    #[test]
    fn split_trims_and_drops_empty_segments() {
        assert_eq!(
            split_beta_tokens(Some(" a , ,b,  c-1 ")),
            vec!["a".to_string(), "b".to_string(), "c-1".to_string()]
        );
    }

    // -----------------------------------------------------------------
    // record_and_get_sticky_betas — tracker semantics ported from
    // tests/test_anthropic_beta_session_sticky.py.
    // -----------------------------------------------------------------

    fn tracker() -> BetaStickyState {
        BetaStickyState::new(BETA_TRACKER_CAPACITY)
    }

    #[test]
    fn first_request_returns_client_tokens() {
        let t = tracker();
        let got = t.record_and_get_sticky_betas(BetaProvider::Anthropic, "s1", Some("a,b"));
        assert_eq!(got, "a,b");
    }

    #[test]
    fn dropped_token_is_reinjected_on_next_turn() {
        // The cache-killer this module exists for: turn N sends
        // "a,b", turn N+1 drops "b" — the union must restore it.
        let t = tracker();
        t.record_and_get_sticky_betas(BetaProvider::Anthropic, "s1", Some("a,b"));
        let got = t.record_and_get_sticky_betas(BetaProvider::Anthropic, "s1", Some("a"));
        assert_eq!(got, "a,b");
    }

    #[test]
    fn union_preserves_first_seen_order() {
        let t = tracker();
        t.record_and_get_sticky_betas(BetaProvider::Anthropic, "s1", Some("b,a"));
        let got = t.record_and_get_sticky_betas(BetaProvider::Anthropic, "s1", Some("a,c"));
        assert_eq!(got, "b,a,c");
    }

    #[test]
    fn dedup_is_case_insensitive_first_casing_wins() {
        let t = tracker();
        t.record_and_get_sticky_betas(
            BetaProvider::Anthropic,
            "s1",
            Some("Context-Management-2025-06-27"),
        );
        let got = t.record_and_get_sticky_betas(
            BetaProvider::Anthropic,
            "s1",
            Some("context-management-2025-06-27"),
        );
        assert_eq!(got, "Context-Management-2025-06-27");
    }

    #[test]
    fn duplicate_client_tokens_are_deduped() {
        let t = tracker();
        let got = t.record_and_get_sticky_betas(BetaProvider::Anthropic, "s1", Some("a,a,b,A"));
        assert_eq!(got, "a,b");
    }

    #[test]
    fn client_whitespace_is_trimmed_in_union() {
        let t = tracker();
        t.record_and_get_sticky_betas(BetaProvider::Anthropic, "s1", Some(" a , b "));
        let got = t.record_and_get_sticky_betas(BetaProvider::Anthropic, "s1", Some("c "));
        assert_eq!(got, "a,b,c");
    }

    #[test]
    fn absent_client_value_returns_session_union() {
        let t = tracker();
        t.record_and_get_sticky_betas(BetaProvider::Anthropic, "s1", Some("a"));
        let got = t.record_and_get_sticky_betas(BetaProvider::Anthropic, "s1", None);
        assert_eq!(got, "a");
    }

    #[test]
    fn empty_session_and_client_yield_empty_string() {
        let t = tracker();
        assert_eq!(
            t.record_and_get_sticky_betas(BetaProvider::Anthropic, "s1", None),
            ""
        );
    }

    #[test]
    fn providers_keep_independent_namespaces() {
        // Same session id, different providers — token sets must not
        // leak across (Python: the (provider, session_id) tuple key).
        let t = tracker();
        t.record_and_get_sticky_betas(BetaProvider::Anthropic, "s1", Some("anth-only"));
        let got = t.record_and_get_sticky_betas(BetaProvider::OpenAi, "s1", Some("oai-only"));
        assert_eq!(got, "oai-only");
    }

    #[test]
    fn sessions_keep_independent_token_sets() {
        let t = tracker();
        t.record_and_get_sticky_betas(BetaProvider::Anthropic, "s1", Some("a"));
        let got = t.record_and_get_sticky_betas(BetaProvider::Anthropic, "s2", Some("b"));
        assert_eq!(got, "b");
    }

    #[test]
    fn lru_evicts_oldest_session_at_capacity() {
        let t = BetaStickyState::new(2);
        t.record_and_get_sticky_betas(BetaProvider::Anthropic, "s1", Some("a"));
        t.record_and_get_sticky_betas(BetaProvider::Anthropic, "s2", Some("b"));
        t.record_and_get_sticky_betas(BetaProvider::Anthropic, "s3", Some("c"));
        assert_eq!(t.active_sessions(), 2);
        // s1 was evicted: its history is gone, so a bare re-request
        // returns only the fresh client value.
        let got = t.record_and_get_sticky_betas(BetaProvider::Anthropic, "s1", Some("z"));
        assert_eq!(got, "z");
    }

    #[test]
    fn lru_hit_touches_recency() {
        let t = BetaStickyState::new(2);
        t.record_and_get_sticky_betas(BetaProvider::Anthropic, "s1", Some("a"));
        t.record_and_get_sticky_betas(BetaProvider::Anthropic, "s2", Some("b"));
        // Touch s1 so s2 becomes the eviction candidate.
        t.record_and_get_sticky_betas(BetaProvider::Anthropic, "s1", None);
        t.record_and_get_sticky_betas(BetaProvider::Anthropic, "s3", Some("c"));
        // s1 survived the s3 insert…
        assert_eq!(
            t.record_and_get_sticky_betas(BetaProvider::Anthropic, "s1", None),
            "a"
        );
        // …and s2 did not.
        assert_eq!(
            t.record_and_get_sticky_betas(BetaProvider::Anthropic, "s2", None),
            ""
        );
    }

    // -----------------------------------------------------------------
    // apply_sticky_betas — header-map plumbing.
    // -----------------------------------------------------------------

    fn header_map(values: &[&str]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for v in values {
            map.append("anthropic-beta", HeaderValue::from_str(v).unwrap());
        }
        map
    }

    fn beta_values(map: &HeaderMap) -> Vec<String> {
        map.get_all("anthropic-beta")
            .iter()
            .map(|v| v.to_str().unwrap().to_string())
            .collect()
    }

    #[test]
    fn apply_rewrites_dropped_token_to_union() {
        let t = tracker();
        let mut turn1 = header_map(&["a,b"]);
        apply_sticky_betas(&t, BetaProvider::Anthropic, "s1", &mut turn1, "req-1");
        assert_eq!(beta_values(&turn1), vec!["a,b"]);

        let mut turn2 = header_map(&["a"]);
        apply_sticky_betas(&t, BetaProvider::Anthropic, "s1", &mut turn2, "req-2");
        assert_eq!(beta_values(&turn2), vec!["a,b"]);
    }

    #[test]
    fn apply_reinserts_union_when_header_fully_omitted() {
        let t = tracker();
        let mut turn1 = header_map(&["a,b"]);
        apply_sticky_betas(&t, BetaProvider::Anthropic, "s1", &mut turn1, "req-1");

        let mut turn2 = HeaderMap::new();
        apply_sticky_betas(&t, BetaProvider::Anthropic, "s1", &mut turn2, "req-2");
        assert_eq!(beta_values(&turn2), vec!["a,b"]);
    }

    #[test]
    fn apply_never_invents_a_header() {
        let t = tracker();
        let mut map = HeaderMap::new();
        apply_sticky_betas(&t, BetaProvider::Anthropic, "s1", &mut map, "req-1");
        assert!(map.get("anthropic-beta").is_none());
    }

    #[test]
    fn apply_noop_leaves_header_lines_untouched() {
        let t = tracker();
        let mut map = header_map(&["a,b"]);
        apply_sticky_betas(&t, BetaProvider::Anthropic, "s1", &mut map, "req-1");
        let mut again = header_map(&["a,b"]);
        apply_sticky_betas(&t, BetaProvider::Anthropic, "s1", &mut again, "req-2");
        assert_eq!(beta_values(&again), vec!["a,b"]);
    }

    #[test]
    fn apply_records_all_repeated_header_lines() {
        // RFC 9110 list semantics: two field lines are one list. The
        // union must record BOTH lines, so a later rewrite (which
        // collapses to a single line) can never shrink the upstream
        // token set mid-conversation.
        let t = tracker();
        let mut turn1 = header_map(&["a,x", "b"]);
        apply_sticky_betas(&t, BetaProvider::Anthropic, "s1", &mut turn1, "req-1");
        // No rewrite on turn 1 (union == joined client list): both
        // lines pass through untouched.
        assert_eq!(beta_values(&turn1), vec!["a,x", "b"]);

        // Turn 2 drops "x" from the first line: the rewrite must
        // carry the full set from both turn-1 lines.
        let mut turn2 = header_map(&["a", "b"]);
        apply_sticky_betas(&t, BetaProvider::Anthropic, "s1", &mut turn2, "req-2");
        assert_eq!(beta_values(&turn2), vec!["a,x,b"]);
    }

    #[test]
    fn apply_skips_non_ascii_value_and_records_nothing() {
        let t = tracker();
        let mut map = HeaderMap::new();
        map.insert(
            "anthropic-beta",
            HeaderValue::from_bytes(&[0xfa, 0xfb]).unwrap(),
        );
        apply_sticky_betas(&t, BetaProvider::Anthropic, "s1", &mut map, "req-1");
        // Wire bytes untouched…
        assert_eq!(map.get("anthropic-beta").unwrap().as_bytes(), &[0xfa, 0xfb]);
        // …and nothing recorded: the next ASCII turn sees only its
        // own tokens.
        let got = t.record_and_get_sticky_betas(BetaProvider::Anthropic, "s1", Some("y"));
        assert_eq!(got, "y");
    }

    #[test]
    fn concurrent_unions_lose_no_tokens() {
        // Port of the Python thread-hammering test: concurrent turns
        // on one session must never drop a recorded token.
        let t = tracker();
        std::thread::scope(|s| {
            for i in 0..8 {
                let t = t.clone();
                s.spawn(move || {
                    let token = format!("tok-{i}");
                    for _ in 0..50 {
                        t.record_and_get_sticky_betas(BetaProvider::Anthropic, "s1", Some(&token));
                    }
                });
            }
        });
        let merged = t.record_and_get_sticky_betas(BetaProvider::Anthropic, "s1", None);
        let tokens: HashSet<&str> = merged.split(',').collect();
        for i in 0..8 {
            assert!(tokens.contains(format!("tok-{i}").as_str()));
        }
    }
}
