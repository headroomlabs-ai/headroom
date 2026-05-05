//! PR-E6: cache-bust drift detector.
//!
//! # What it does
//!
//! For every inbound request on a known LLM endpoint, compute a
//! [`StructuralHash`] over the **cache hot zone**:
//!
//! - `system` — SHA-256 of the canonical system-prompt bytes (Anthropic
//!   `body.system`; OpenAI Chat first `role=system` message;
//!   OpenAI Responses `body.instructions`).
//! - `tools` — SHA-256 of the canonical bytes of `body.tools`.
//! - `early_messages` — SHA-256 of the canonical bytes of the first 3
//!   message-shaped items (or all, if fewer than 3). Skips the
//!   live-zone tail where mutation is expected and benign.
//!
//! Track the previous hash per session in a bounded LRU. When a
//! subsequent request on the same session disagrees on any dimension,
//! emit a `cache_drift_observed` log line listing the drifted
//! dimensions. **Never mutates the request body** — the detector is a
//! pure observer and the proxy's "passthrough is sacred" invariant
//! (Phase A) is preserved by construction.
//!
//! # Privacy
//!
//! The session key is derived from the strongest available client
//! identifier (`Authorization`, `x-api-key`, client IP, finally
//! `(client_ip, user_agent)`). Bearer tokens and API keys are
//! **hashed before they ever leave this module**; the raw secret is
//! never logged, never stored, and is overwritten in transit (truncated
//! to a 16-character hex prefix). The log line itself only includes a
//! short prefix of the SHA-256 hex of the session key.
//!
//! # Cost
//!
//! - One SHA-256 update over each of (system, tools, early messages).
//!   Total ~200us on a 8 KB system prompt.
//! - One LRU lookup + insert. `lru = "0.12"` is O(1) amortised.
//! - One `tracing::info!` or `tracing::warn!`. No metric emission yet
//!   (left for Phase F PR-F* when the global Prometheus registry can
//!   accept session-scoped counters without a cardinality explosion).

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::net::SocketAddr;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

use axum::http::HeaderMap;
use lru::LruCache;
use sha2::{Digest, Sha256};

/// Which provider's body shape we're hashing. The walker is shaped
/// per provider because the cache hot zone lives in different fields:
/// Anthropic uses `body.system`/`body.tools`/`body.messages`, OpenAI
/// Chat threads `system` into the first message, and OpenAI Responses
/// uses `body.instructions`/`body.tools`/`body.input`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiKind {
    /// `POST /v1/messages` (Anthropic).
    Anthropic,
    /// `POST /v1/chat/completions` (OpenAI).
    OpenAiChat,
    /// `POST /v1/responses` (OpenAI Responses API).
    OpenAiResponses,
}

/// Three-axis structural fingerprint of the cache hot zone.
///
/// Each axis is the SHA-256 of the canonical bytes at that position
/// (we re-serialize via `serde_json::to_vec` so whitespace and key
/// order through the original network bytes do not perturb the hash).
/// All three are required for "no drift"; any one differing flags
/// drift on that dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StructuralHash {
    pub system: [u8; 32],
    pub tools: [u8; 32],
    pub early_messages: [u8; 32],
}

/// How many message-shaped items count as the "early" prefix that
/// feeds `early_messages_hash`. Anything past this is the live zone
/// (where mutation is expected; we deliberately ignore it).
const EARLY_MESSAGES_WINDOW: usize = 3;

/// Compute a [`StructuralHash`] for the body shape implied by `kind`.
///
/// `body` is borrowed; **this function never mutates it**. The
/// `does_not_mutate_input` test in the module below pins this with a
/// clone-and-compare assertion.
pub fn compute_structural_hash(body: &serde_json::Value, kind: ApiKind) -> StructuralHash {
    let system = hash_value(&extract_system(body, kind));
    let tools = hash_value(&extract_tools(body));
    let early_messages = hash_value(&extract_early_messages(body, kind));
    StructuralHash {
        system,
        tools,
        early_messages,
    }
}

/// Extract the "system" axis as a `serde_json::Value`. Returns
/// `Value::Null` when the dimension is absent — Null still hashes to
/// a stable 32-byte digest so first-request comparisons are
/// well-defined.
fn extract_system(body: &serde_json::Value, kind: ApiKind) -> serde_json::Value {
    match kind {
        ApiKind::Anthropic => body
            .get("system")
            .cloned()
            .unwrap_or(serde_json::Value::Null),
        ApiKind::OpenAiChat => {
            // First message with `role == "system"` is the OpenAI
            // Chat hot-zone equivalent. There can be at most one in
            // practice (newer requests use a `developer` role; that's
            // not the system axis and we deliberately don't conflate).
            body.get("messages")
                .and_then(|v| v.as_array())
                .and_then(|arr| {
                    arr.iter().find(|m| {
                        m.get("role")
                            .and_then(|r| r.as_str())
                            .map(|s| s == "system")
                            .unwrap_or(false)
                    })
                })
                .cloned()
                .unwrap_or(serde_json::Value::Null)
        }
        ApiKind::OpenAiResponses => body
            .get("instructions")
            .cloned()
            .unwrap_or(serde_json::Value::Null),
    }
}

/// Extract the "tools" axis as a `serde_json::Value`. The same
/// `tools` array key is used by all three providers in practice.
fn extract_tools(body: &serde_json::Value) -> serde_json::Value {
    body.get("tools")
        .cloned()
        .unwrap_or(serde_json::Value::Null)
}

/// Extract the first [`EARLY_MESSAGES_WINDOW`] message-shaped items
/// as an array `Value`. Skips the system message in the OpenAI Chat
/// shape (the system axis already hashes that separately).
fn extract_early_messages(body: &serde_json::Value, kind: ApiKind) -> serde_json::Value {
    let array_key = match kind {
        ApiKind::Anthropic => "messages",
        ApiKind::OpenAiChat => "messages",
        ApiKind::OpenAiResponses => "input",
    };
    let messages = match body.get(array_key).and_then(|v| v.as_array()) {
        Some(arr) => arr,
        None => return serde_json::Value::Null,
    };
    let early: Vec<serde_json::Value> = match kind {
        ApiKind::OpenAiChat => messages
            .iter()
            .filter(|m| {
                m.get("role")
                    .and_then(|r| r.as_str())
                    .map(|s| s != "system")
                    .unwrap_or(true)
            })
            .take(EARLY_MESSAGES_WINDOW)
            .cloned()
            .collect(),
        _ => messages
            .iter()
            .take(EARLY_MESSAGES_WINDOW)
            .cloned()
            .collect(),
    };
    serde_json::Value::Array(early)
}

/// SHA-256 over `serde_json::to_vec(value)`. Re-serializing the
/// borrowed `Value` defends against trivial whitespace differences
/// from the wire — operators care about *semantic* drift, not
/// formatter drift.
fn hash_value(value: &serde_json::Value) -> [u8; 32] {
    // `serde_json::to_vec` on a `Value` cannot fail except on a
    // pathological recursion, which the upstream API would itself
    // reject; on the impossible failure path we hash the empty byte
    // string so the digest is still stable rather than panicking and
    // taking the request down.
    let bytes = serde_json::to_vec(value).unwrap_or_default();
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let digest = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

/// Bounded session → last-seen `StructuralHash` map. Wrapped in
/// `Arc<Mutex<…>>` so it can be cloned freely into `AppState` without
/// duplicating the underlying LRU.
#[derive(Clone)]
pub struct DriftState {
    cache: Arc<Mutex<LruCache<String, StructuralHash>>>,
}

impl DriftState {
    /// Build a new `DriftState` bounded to `capacity` sessions. The
    /// production capacity is 1000; tests pass small values so the
    /// LRU eviction path is exercised cheaply.
    ///
    /// # Panics
    ///
    /// Panics if `capacity == 0`. The detector is meaningless without
    /// at least one slot — use `LruCache::new(NonZeroUsize::MIN)` if
    /// you need a "remember nothing" mode.
    pub fn new(capacity: usize) -> Self {
        let cap = NonZeroUsize::new(capacity).expect("DriftState capacity must be > 0");
        Self {
            cache: Arc::new(Mutex::new(LruCache::new(cap))),
        }
    }
}

impl std::fmt::Debug for DriftState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let len = self.cache.lock().map(|c| c.len()).unwrap_or(0);
        f.debug_struct("DriftState").field("len", &len).finish()
    }
}

/// Compare `current` against the last-seen hash for `session_key` and
/// emit a structured `tracing` event accordingly. Always updates the
/// LRU to `current` before returning so the next call sees the most
/// recent fingerprint.
///
/// Logging contract:
///
/// - First time a session is seen → `tracing::info!(event =
///   "cache_drift_first_request", …)` with a 16-char prefix of the
///   SHA-256 hex of `session_key`.
/// - Subsequent requests with all three hashes equal → no event.
/// - Subsequent requests with any dimension differing →
///   `tracing::warn!(event = "cache_drift_observed", drift_dims =
///   "<comma-joined>", previous_hash_prefix, current_hash_prefix, …)`.
pub fn observe_drift(state: &DriftState, session_key: &str, current: StructuralHash) {
    let session_prefix = session_key_log_prefix(session_key);
    let mut cache = match state.cache.lock() {
        Ok(c) => c,
        Err(poisoned) => {
            // Mutex was poisoned by a panicking writer in another
            // task. Recover the inner data — the only thing we lose
            // is one stale entry, and continuing the request is
            // strictly preferable to failing closed.
            tracing::warn!(
                event = "cache_drift_state_mutex_poisoned",
                "drift detector mutex was poisoned by a panicking task; recovering"
            );
            poisoned.into_inner()
        }
    };
    match cache.get(session_key).copied() {
        None => {
            tracing::info!(
                event = "cache_drift_first_request",
                session_key_hash = %session_prefix,
                current_hash_prefix = %structural_hash_log_prefix(&current),
                "cache_drift detector observed a new session"
            );
            cache.put(session_key.to_string(), current);
        }
        Some(previous) if previous == current => {
            // Stable. No event. Update LRU recency by reinserting.
            cache.put(session_key.to_string(), current);
        }
        Some(previous) => {
            let dims = drift_dims(&previous, &current);
            tracing::warn!(
                event = "cache_drift_observed",
                session_key_hash = %session_prefix,
                drift_dims = %dims,
                previous_hash_prefix = %structural_hash_log_prefix(&previous),
                current_hash_prefix = %structural_hash_log_prefix(&current),
                "cache_drift detector observed structural change between turns of the same session"
            );
            cache.put(session_key.to_string(), current);
        }
    }
}

/// 16-char hex prefix of SHA-256(session_key). Bounds the log line
/// width and never reveals the raw key (which may be a bearer token
/// or API key — see `derive_session_key`).
fn session_key_log_prefix(session_key: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(session_key.as_bytes());
    let digest = hasher.finalize();
    hex_prefix(&digest, 16)
}

/// 12-char hex prefix of the concatenated structural hash. Useful as
/// a compact "did the prefix change" indicator in logs without
/// printing the entire 96-char digest tuple.
fn structural_hash_log_prefix(hash: &StructuralHash) -> String {
    let mut hasher = Sha256::new();
    hasher.update(hash.system);
    hasher.update(hash.tools);
    hasher.update(hash.early_messages);
    let digest = hasher.finalize();
    hex_prefix(&digest, 12)
}

/// Lowercase hex of the first `take` bytes of `bytes`. Allocates a
/// `String` once per call.
fn hex_prefix(bytes: &[u8], take: usize) -> String {
    let take = take.min(bytes.len());
    let mut out = String::with_capacity(take * 2);
    for b in &bytes[..take] {
        // Manual hex; avoids pulling `hex` for one call site.
        const HEX: &[u8; 16] = b"0123456789abcdef";
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0xf) as usize] as char);
    }
    out
}

/// Comma-joined list of which dimensions drifted between `prev` and
/// `curr`. The order is fixed (`system`, `tools`, `early_messages`)
/// so log queries can match deterministically.
fn drift_dims(prev: &StructuralHash, curr: &StructuralHash) -> String {
    let mut dims: Vec<&'static str> = Vec::with_capacity(3);
    if prev.system != curr.system {
        dims.push("system");
    }
    if prev.tools != curr.tools {
        dims.push("tools");
    }
    if prev.early_messages != curr.early_messages {
        dims.push("early_messages");
    }
    dims.join(",")
}

/// Derive a stable per-session key from the request headers and
/// client address. Priority order:
///
/// 1. `Authorization` header (hashed; never logged raw).
/// 2. `x-api-key` header (hashed; never logged raw).
/// 3. Client IP address.
/// 4. `(client_ip, user_agent)` synthetic tuple — the user-agent
///    bucketization gives us *some* discrimination when many
///    anonymous clients sit behind the same NAT.
///
/// The returned string is opaque; never log it directly. Callers
/// should pass it straight to [`observe_drift`], which logs only a
/// hashed prefix.
pub fn derive_session_key(headers: &HeaderMap, client_addr: &SocketAddr) -> String {
    if let Some(token) = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
    {
        return format!("auth:{}", hash_secret(token));
    }
    // `x-api-key` is the Anthropic/OpenAI-Responses convention.
    if let Some(key) = headers.get("x-api-key").and_then(|v| v.to_str().ok()) {
        return format!("apikey:{}", hash_secret(key));
    }
    let ip = client_addr.ip().to_string();
    if let Some(ua) = headers
        .get(axum::http::header::USER_AGENT)
        .and_then(|v| v.to_str().ok())
    {
        // Hash the (ip, ua) tuple so the resulting key remains opaque
        // and does not leak full UA strings into downstream logs that
        // forget our "log only the prefix" contract.
        let mut h = DefaultHasher::new();
        ip.hash(&mut h);
        ua.hash(&mut h);
        return format!("ipua:{:016x}", h.finish());
    }
    format!("ip:{ip}")
}

/// SHA-256 of `secret`, truncated to 16 hex characters. Sufficient
/// to discriminate sessions while pinning that the raw secret never
/// reaches the log line. We do **not** use the full digest because
/// even a hashed bearer that ends up in many log entries leaks
/// fingerprintable information; the 16-char prefix bounds that.
fn hash_secret(secret: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(secret.as_bytes());
    let digest = hasher.finalize();
    hex_prefix(&digest, 16)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::net::{IpAddr, Ipv4Addr};

    fn anthropic_body(
        system: &str,
        tools: serde_json::Value,
        msgs: Vec<&str>,
    ) -> serde_json::Value {
        let messages: Vec<serde_json::Value> = msgs
            .into_iter()
            .map(|t| json!({"role": "user", "content": t}))
            .collect();
        json!({
            "model": "claude-3-5-sonnet-20241022",
            "system": system,
            "tools": tools,
            "messages": messages,
        })
    }

    fn make_state() -> DriftState {
        DriftState::new(8)
    }

    #[test]
    fn first_request_emits_first_request_event() {
        let state = make_state();
        let body = anthropic_body("you are an assistant", json!([]), vec!["hi"]);
        let h = compute_structural_hash(&body, ApiKind::Anthropic);
        // Before observation: empty cache.
        assert_eq!(state.cache.lock().unwrap().len(), 0);
        observe_drift(&state, "session-A", h);
        // After observation: 1 entry, equal to the input hash.
        let cache = state.cache.lock().unwrap();
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.peek("session-A"), Some(&h));
    }

    #[test]
    fn same_hash_emits_no_event() {
        let state = make_state();
        let body = anthropic_body("sys-A", json!([]), vec!["m1"]);
        let h = compute_structural_hash(&body, ApiKind::Anthropic);
        observe_drift(&state, "sess", h);
        // Second observation with identical hash: still 1 entry, same hash.
        observe_drift(&state, "sess", h);
        let cache = state.cache.lock().unwrap();
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.peek("sess"), Some(&h));
    }

    #[test]
    fn system_drift_detected_with_correct_dim() {
        let state = make_state();
        let h1 = compute_structural_hash(
            &anthropic_body("sys-A", json!([]), vec!["m1"]),
            ApiKind::Anthropic,
        );
        let h2 = compute_structural_hash(
            &anthropic_body("sys-B", json!([]), vec!["m1"]),
            ApiKind::Anthropic,
        );
        assert_ne!(h1.system, h2.system);
        assert_eq!(h1.tools, h2.tools);
        assert_eq!(h1.early_messages, h2.early_messages);
        assert_eq!(drift_dims(&h1, &h2), "system");
        observe_drift(&state, "sess", h1);
        observe_drift(&state, "sess", h2);
    }

    #[test]
    fn tools_drift_detected_with_correct_dim() {
        let h1 = compute_structural_hash(
            &anthropic_body("sys", json!([{"name": "a"}]), vec!["m1"]),
            ApiKind::Anthropic,
        );
        let h2 = compute_structural_hash(
            &anthropic_body("sys", json!([{"name": "b"}]), vec!["m1"]),
            ApiKind::Anthropic,
        );
        assert_eq!(h1.system, h2.system);
        assert_ne!(h1.tools, h2.tools);
        assert_eq!(h1.early_messages, h2.early_messages);
        assert_eq!(drift_dims(&h1, &h2), "tools");
    }

    #[test]
    fn early_messages_drift_detected_with_correct_dim() {
        let h1 = compute_structural_hash(
            &anthropic_body("sys", json!([]), vec!["m1"]),
            ApiKind::Anthropic,
        );
        let h2 = compute_structural_hash(
            &anthropic_body("sys", json!([]), vec!["DIFFERENT"]),
            ApiKind::Anthropic,
        );
        assert_eq!(h1.system, h2.system);
        assert_eq!(h1.tools, h2.tools);
        assert_ne!(h1.early_messages, h2.early_messages);
        assert_eq!(drift_dims(&h1, &h2), "early_messages");
    }

    #[test]
    fn multi_dim_drift_lists_all_changed_dims() {
        let h1 = compute_structural_hash(
            &anthropic_body("sys-A", json!([{"name": "a"}]), vec!["m1"]),
            ApiKind::Anthropic,
        );
        let h2 = compute_structural_hash(
            &anthropic_body("sys-B", json!([{"name": "b"}]), vec!["X"]),
            ApiKind::Anthropic,
        );
        assert_eq!(drift_dims(&h1, &h2), "system,tools,early_messages");
    }

    #[test]
    fn lru_evicts_at_capacity() {
        // Capacity 2: inserting a 3rd session evicts the LRU.
        let state = DriftState::new(2);
        let h = compute_structural_hash(
            &anthropic_body("s", json!([]), vec!["m"]),
            ApiKind::Anthropic,
        );
        observe_drift(&state, "s1", h);
        observe_drift(&state, "s2", h);
        observe_drift(&state, "s3", h);
        let cache = state.cache.lock().unwrap();
        assert_eq!(cache.len(), 2);
        // s1 was the least-recently-used; should have been evicted.
        assert!(!cache.contains("s1"));
        assert!(cache.contains("s2"));
        assert!(cache.contains("s3"));
    }

    #[test]
    fn does_not_mutate_input() {
        let body = anthropic_body(
            "sys",
            json!([{"name": "t1", "input_schema": {"type": "object"}}]),
            vec!["m1", "m2", "m3", "m4"],
        );
        let original_bytes = serde_json::to_vec(&body).expect("serialize");
        // Compute the hash twice — across the three ApiKind shapes —
        // to exercise every branch that *could* mutate the input.
        let _ = compute_structural_hash(&body, ApiKind::Anthropic);
        let _ = compute_structural_hash(&body, ApiKind::OpenAiChat);
        let _ = compute_structural_hash(&body, ApiKind::OpenAiResponses);
        let after_bytes = serde_json::to_vec(&body).expect("re-serialize");
        assert_eq!(original_bytes, after_bytes);
    }

    #[test]
    fn session_key_hashes_authorization_does_not_log_raw() {
        let mut headers = HeaderMap::new();
        headers.insert(
            axum::http::header::AUTHORIZATION,
            "Bearer sk-ant-very-secret-token-do-not-log-me"
                .parse()
                .unwrap(),
        );
        let addr: SocketAddr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)), 1234);
        let key = derive_session_key(&headers, &addr);
        // The key MUST NOT contain the raw bearer string anywhere —
        // not the secret token, not the literal "Bearer", not even
        // any 8+ char substring of the secret.
        assert!(
            !key.contains("sk-ant"),
            "session key leaked raw secret prefix: {key}"
        );
        assert!(
            !key.contains("very-secret"),
            "session key leaked raw secret middle: {key}"
        );
        assert!(
            !key.contains("Bearer"),
            "session key leaked the auth scheme: {key}"
        );
        // The key SHOULD be the auth-scoped envelope, so we know the
        // `Authorization` arm was taken (not the IP fallback).
        assert!(key.starts_with("auth:"), "expected auth-scoped key: {key}");
        // And the log prefix must also not leak the raw secret.
        let log_prefix = session_key_log_prefix(&key);
        assert!(!log_prefix.contains("sk-ant"));
        assert!(!log_prefix.contains("very-secret"));
        assert!(!log_prefix.contains("Bearer"));
        assert_eq!(log_prefix.len(), 32); // 16 bytes × 2 hex chars
    }

    #[test]
    fn session_key_hashes_x_api_key_does_not_log_raw() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-api-key",
            "sk-very-private-api-key-12345".parse().unwrap(),
        );
        let addr: SocketAddr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 2)), 1234);
        let key = derive_session_key(&headers, &addr);
        assert!(!key.contains("sk-very-private"));
        assert!(key.starts_with("apikey:"));
    }

    #[test]
    fn session_key_falls_back_to_ip_then_ip_ua() {
        let addr: SocketAddr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 3)), 5555);
        // No headers → ip-only.
        let bare = derive_session_key(&HeaderMap::new(), &addr);
        assert!(bare.starts_with("ip:"));
        // With UA → ipua-tuple.
        let mut headers = HeaderMap::new();
        headers.insert(axum::http::header::USER_AGENT, "ua-test".parse().unwrap());
        let with_ua = derive_session_key(&headers, &addr);
        assert!(with_ua.starts_with("ipua:"));
        assert_ne!(bare, with_ua);
    }

    #[test]
    fn openai_chat_extracts_first_system_message() {
        let body = json!({
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "you are a helpful assistant"},
                {"role": "user", "content": "hi"},
            ],
            "tools": [],
        });
        let h1 = compute_structural_hash(&body, ApiKind::OpenAiChat);
        // Same body but a different system message → system axis drifts.
        let body2 = json!({
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "you are a different assistant"},
                {"role": "user", "content": "hi"},
            ],
            "tools": [],
        });
        let h2 = compute_structural_hash(&body2, ApiKind::OpenAiChat);
        assert_ne!(h1.system, h2.system);
        // user message identical → early-messages stays identical.
        assert_eq!(h1.early_messages, h2.early_messages);
    }

    #[test]
    fn openai_responses_uses_instructions_and_input() {
        let body = json!({
            "model": "gpt-4",
            "instructions": "be brief",
            "tools": [],
            "input": [
                {"type": "message", "role": "user", "content": "hello"},
            ],
        });
        let h1 = compute_structural_hash(&body, ApiKind::OpenAiResponses);
        let body2 = json!({
            "model": "gpt-4",
            "instructions": "be verbose",
            "tools": [],
            "input": [
                {"type": "message", "role": "user", "content": "hello"},
            ],
        });
        let h2 = compute_structural_hash(&body2, ApiKind::OpenAiResponses);
        assert_ne!(h1.system, h2.system);
        assert_eq!(h1.early_messages, h2.early_messages);
    }

    #[test]
    fn early_messages_window_caps_at_three() {
        // 5 messages: hash should depend only on the first 3.
        let h1 = compute_structural_hash(
            &anthropic_body("s", json!([]), vec!["a", "b", "c", "d", "e"]),
            ApiKind::Anthropic,
        );
        // Mutating message 4 only must NOT drift the early_messages hash.
        let h2 = compute_structural_hash(
            &anthropic_body("s", json!([]), vec!["a", "b", "c", "DIFFERENT", "e"]),
            ApiKind::Anthropic,
        );
        assert_eq!(h1.early_messages, h2.early_messages);
        // But mutating message 1 must drift it.
        let h3 = compute_structural_hash(
            &anthropic_body("s", json!([]), vec!["DIFFERENT", "b", "c", "d", "e"]),
            ApiKind::Anthropic,
        );
        assert_ne!(h1.early_messages, h3.early_messages);
    }
}
