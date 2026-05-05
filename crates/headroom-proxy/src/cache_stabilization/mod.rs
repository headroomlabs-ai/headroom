//! Phase E cache-stabilization surface.
//!
//! The realignment plan (`REALIGNMENT/07-phase-E-cache-stabilization.md`)
//! groups every cache-stabilization mechanism behind one module so
//! operators searching for "what does Headroom do to keep prompt
//! caches warm" land in one place. Phase E PRs in this module sit
//! *next to* the request path — never on it. They observe inbound
//! bodies and emit structured logs so customers can see why their
//! prompt-cache hit rate is degrading. Nothing in here mutates
//! request bytes; the cache-safety invariant from Phase A still holds.
//!
//! Currently shipped:
//!
//! - [`volatile_detector`] — PR-E5: scans inbound bodies for patterns
//!   that bust prompt-cache hits (ISO 8601 timestamps, UUID v4s,
//!   ID-named fields) and emits one structured WARN log per finding
//!   so customers know what to move out of the cached prefix.
//! - [`drift_detector`] — PR-E6: per-session SHA-256 fingerprint of
//!   the cache hot zone (system / tools / early messages). Emits
//!   `cache_drift_first_request` on first sight and
//!   `cache_drift_observed` when consecutive requests on the same
//!   session disagree on any of the three dimensions.
//! - [`anthropic_cache_control`] — PR-E3: on PAYG-classified
//!   requests where the customer hasn't placed any `cache_control`
//!   marker, auto-inserts one ephemeral marker on the last tool
//!   definition so unsophisticated callers (hand-rolled SDK code,
//!   smaller agents, plain `curl`) get prompt-cache hits without
//!   learning Anthropic's marker API. **Mutates request bytes**;
//!   gated on auth_mode == PAYG and the absence of any pre-existing
//!   marker.
//!
//! Future PRs (E1 — tool-array sort, E2 — JSON Schema key sort, E3 —
//! `cache_control` auto-placement, E4 — `prompt_cache_key` injection)
//! hang sibling submodules off this same `mod.rs`. Conflict
//! resolution between parallel Phase E PRs is intentionally trivial:
//! each detector lives in its own file, the only shared surface is
//! this `mod.rs`'s `pub mod` list.

pub mod anthropic_cache_control;
pub mod drift_detector;
pub mod volatile_detector;
