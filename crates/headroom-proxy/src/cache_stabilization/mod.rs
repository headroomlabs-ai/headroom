//! Phase E cache-stabilization surface.
//!
//! The realignment plan (`REALIGNMENT/07-phase-E-cache-stabilization.md`)
//! groups every cache-stabilization mechanism behind one module so
//! operators searching for "what does Headroom do to keep prompt
//! caches warm" land in one place. Phase E PRs in this module either:
//!
//! - **Observe** inbound bodies and emit structured warnings so
//!   customers can see why their prompt-cache hit rate is degrading
//!   ([`volatile_detector`], PR-E5; [`drift_detector`], PR-E6). These
//!   never mutate request bytes.
//! - **Normalize** request bytes to make cache hits deterministic
//!   under PAYG mode ([`tool_def_normalize`], PR-E1 / PR-E2;
//!   [`anthropic_cache_control`], PR-E3; [`openai_cache_key`], PR-E4).
//!   These mutate bytes only when the auth-mode gate and per-policy
//!   preconditions (e.g. no customer `cache_control` marker) all clear;
//!   OAuth and Subscription always passthrough.
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
//! - [`tool_def_normalize`] — PR-E1 / PR-E2: sorts `tools[]`
//!   alphabetically by name (PR-E1) and recursively sorts JSON
//!   Schema object keys inside each tool's `input_schema` /
//!   `function.parameters` (PR-E2). PAYG-only. PR-E1 additionally
//!   skips when any tool already carries a top-level
//!   `cache_control` marker; PR-E2 has no marker check because
//!   sorting schema keys never moves the marker (which lives on
//!   the tool object, not inside the schema).
//! - [`anthropic_cache_control`] — PR-E3: on PAYG-classified
//!   requests where the customer hasn't placed any `cache_control`
//!   marker, auto-inserts one ephemeral marker on the last tool
//!   definition so unsophisticated callers (hand-rolled SDK code,
//!   smaller agents, plain `curl`) get prompt-cache hits without
//!   learning Anthropic's marker API. **Mutates request bytes**;
//!   gated on auth_mode == PAYG and the absence of any pre-existing
//!   marker.
//! - [`openai_cache_key`] — PR-E4: on PAYG OpenAI requests where the
//!   customer has not set `prompt_cache_key`, derive a stable key from
//!   `(model, system, tools)` and inject it so the upstream pins
//!   cache lookup to a tenant-stable identity. **Mutates the body**
//!   (only on PAYG) — see its docs for the gating contract.
//!
//! Sibling PRs hang additional submodules off this `mod.rs`. Conflict
//! resolution between parallel Phase E PRs is intentionally trivial:
//! each lives in its own file, the only shared surface is this
//! `mod.rs`'s `pub mod` list.

pub mod anthropic_cache_control;
pub mod drift_detector;
pub mod openai_cache_key;
pub mod tool_def_normalize;
pub mod volatile_detector;
