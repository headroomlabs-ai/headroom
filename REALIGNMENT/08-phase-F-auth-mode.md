# Phase F — Auth-Mode Policy Gates

**Goal:** Make auth mode a first-class policy axis. Detect (PAYG / OAuth / subscription) at request entry; gate compression behavior, header injection, and TOIN aggregation per mode. Stealth mode for subscription CLIs.

**Calendar:** 1 week.

**Shape:** 4 PRs. F1 first; F2 + F3 + F4 parallel after.

Reference: `~/.claude/projects/-Users-tchopra-claude-projects-headroom/memory/project_auth_mode_compression_nuances.md`.

---

## PR-F1 — `classify_auth_mode` helper

**Branch:** `realign-F1-classify-auth-mode`
**Worktree:** `~/claude-projects/headroom-worktrees/realign-F1-classify-auth-mode`
**Risk:** **LOW**
**LOC:** +500

### Scope
Single helper called at request entry returning `AuthMode = Payg | OAuth | Subscription`. Pure-function classification from headers (Authorization shape, OpenAI-Beta, anthropic-beta) and User-Agent prefix.

### Files

**Add:**
- `crates/headroom-core/src/auth_mode.rs`:
  ```rust
  #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
  pub enum AuthMode { Payg, OAuth, Subscription }

  pub fn classify(headers: &http::HeaderMap) -> AuthMode {
      let ua = headers.get("user-agent").and_then(|h| h.to_str().ok()).unwrap_or("").to_lowercase();
      const SUBSCRIPTION_UA_PREFIXES: &[&str] = &[
          "claude-cli/", "claude-code/", "codex-cli/", "cursor/",
          "claude-vscode/", "github-copilot/", "anthropic-cli/", "antigravity/",
      ];
      if SUBSCRIPTION_UA_PREFIXES.iter().any(|p| ua.contains(p)) {
          return AuthMode::Subscription;
      }

      let auth = headers.get("authorization").and_then(|h| h.to_str().ok()).unwrap_or("");
      if auth.starts_with("Bearer ") {
          let token = &auth[7..];
          if token.starts_with("sk-ant-api") || token.starts_with("sk-") {
              return AuthMode::Payg;
          }
          if token.starts_with("sk-ant-oat-") || token.split('.').count() >= 3 {
              // sk-ant-oat-* (Claude Pro OAuth) or JWT (Codex/Cursor OAuth)
              return AuthMode::OAuth;
          }
      }

      // Bedrock / Vertex (no Authorization header from client; signed downstream)
      if !auth.is_empty() == false
         && headers.get("x-api-key").is_none()
         && headers.get("x-goog-api-key").is_none()
      {
          return AuthMode::OAuth;
      }

      // x-api-key present (Anthropic API key style)
      if headers.contains_key("x-api-key") {
          return AuthMode::Payg;
      }

      AuthMode::Payg  // default
  }
  ```

**Add (Python):**
- `headroom/proxy/auth_mode.py` — Python port of the same logic. Used in Python paths until Phase H deletes them.

**Modify:**
- `crates/headroom-core/src/lib.rs` — `pub mod auth_mode;`.
- `crates/headroom-proxy/src/proxy.rs` — call `classify` at request entry; store in request extensions for downstream handlers.
- `headroom/proxy/handlers/anthropic.py` — call Python `classify_auth_mode(headers)` at request entry.
- `headroom/proxy/handlers/openai.py` — same.

**Tests added:**
- `crates/headroom-core/tests/auth_mode.rs::api_key_classified_payg`
- `crates/headroom-core/tests/auth_mode.rs::oauth_jwt_classified_oauth`
- `crates/headroom-core/tests/auth_mode.rs::oauth_sk_ant_oat_classified_oauth`
- `crates/headroom-core/tests/auth_mode.rs::claude_code_ua_classified_subscription`
- `crates/headroom-core/tests/auth_mode.rs::cursor_ua_classified_subscription`
- `crates/headroom-core/tests/auth_mode.rs::no_auth_no_user_agent_default_payg`
- `crates/headroom-core/tests/auth_mode.rs::bedrock_no_auth_classified_oauth`
- Python equivalents in `tests/test_auth_mode.py`.

### Acceptance criteria

- Tests pass.
- Detection runs in <10us per call.
- Documented in `docs/auth-modes.md` with the detection rules and how to extend.

### Blocked by

None.

### Blocks

PR-F2, PR-F3, PR-F4.

### Rollback

`git revert`. All requests treated as PAYG (current behavior).

---

## PR-F2 — Per-mode compression policy gates

**Branch:** `realign-F2-per-mode-policy`
**Worktree:** `~/claude-projects/headroom-worktrees/realign-F2-per-mode-policy`
**Risk:** **MEDIUM-HIGH** (changes compression behavior per request)
**LOC:** +600

### Scope
Wire `AuthMode` into every compression decision per the policy matrix in `02-architecture.md §2.4`. PAYG = aggressive (current default). OAuth = passthrough-prefer (no auto-`cache_control`, no auto-`prompt_cache_key`, no lossy compressors). Subscription = stealth (everything OAuth does + preserve `accept-encoding`, never strip; never inject `X-Headroom-*`; never mutate User-Agent).

### Files

**Modify:**
- `crates/headroom-proxy/src/compression/live_zone_anthropic.rs` — gate `auto_place_breakpoints` on `auth_mode == Payg`.
- `crates/headroom-proxy/src/compression/live_zone_openai.rs` — gate `inject_prompt_cache_key` on `auth_mode == Payg`.
- `crates/headroom-proxy/src/compression/live_zone.rs` — gate lossy compressors (Kompress text) on `auth_mode == Payg`. OAuth and Subscription get lossless-only compression.
- `crates/headroom-proxy/src/headers.rs:103-117` — `add_x_forwarded_headers` becomes `add_x_forwarded_headers_if(auth_mode)`. Skip on Subscription.
- `crates/headroom-proxy/src/proxy.rs` — `accept-encoding` strip becomes conditional on `auth_mode != Subscription`.
- `headroom/proxy/handlers/anthropic.py` — gate Python compression decisions identically.
- `headroom/proxy/handlers/openai.py` — same.

**Tests added:**
- `crates/headroom-proxy/tests/integration_authmode_policy.rs::payg_aggressive_compression`
- `crates/headroom-proxy/tests/integration_authmode_policy.rs::oauth_no_auto_cache_control`
- `crates/headroom-proxy/tests/integration_authmode_policy.rs::oauth_no_auto_prompt_cache_key`
- `crates/headroom-proxy/tests/integration_authmode_policy.rs::oauth_lossless_only`
- `crates/headroom-proxy/tests/integration_authmode_policy.rs::subscription_no_x_forwarded`
- `crates/headroom-proxy/tests/integration_authmode_policy.rs::subscription_preserves_accept_encoding`
- `crates/headroom-proxy/tests/integration_authmode_policy.rs::subscription_lossless_only`

### Acceptance criteria

- Tests pass.
- Manual smoke test: a real Claude Code session through the proxy produces no `X-Forwarded-*` upstream and preserves `accept-encoding`.

### Blocked by

PR-F1, PR-E3, PR-E4.

### Blocks

None.

### Rollback

`git revert`. All requests treated as PAYG. No functional regression for PAYG users; OAuth/Subscription users may see scope-rejection or revocation increase.

---

## PR-F3 — TOIN per-tenant aggregation key

**Branch:** `realign-F3-toin-per-tenant`
**Worktree:** `~/claude-projects/headroom-worktrees/realign-F3-toin-per-tenant`
**Risk:** **MEDIUM**
**LOC:** -200 / +400

### Scope
Eliminate P5-56. Extend TOIN's aggregation key from `structure_hash` to `(auth_mode, model_family, structure_hash)`. Storage key prefix updated; in-memory dicts re-keyed. Existing observations from before this PR are preserved under a `legacy/` prefix and deprecated over the next 30 days.

### Files

**Modify:**
- `headroom/telemetry/toin.py:103` — `Pattern` adds `auth_mode: str = "unknown"`, `model_family: str = "unknown"`. (Already covered by Phase B PR-B5; this PR ensures the wiring lands.)
- `headroom/telemetry/toin.py:477, 496, 727, 729, 1248, 1256` — change aggregation key to tuple.
- `headroom/telemetry/toin.py` — migration helper that walks the legacy `structure_hash`-only store and copies entries under `("unknown", "unknown", structure_hash)` for graceful degrade.
- `headroom/telemetry/toin.py` — bumping aggregation key invalidates earlier recommendations; re-publish via the deploy CLI.
- `headroom/subscription/tracker.py:166` — replace `_current_token: str` (raw OAuth bearer storage) with `_current_token_id: str` (a one-way hash + last-4 chars for debugging). Polling code adapts to use the actual `Authorization` header per request rather than the stored copy.

**Tests added:**
- `tests/test_toin_per_tenant.py::test_aggregation_key_includes_auth_mode_model`
- `tests/test_toin_per_tenant.py::test_legacy_observations_preserved_under_unknown`
- `tests/test_toin_per_tenant.py::test_publish_per_auth_mode_writes_separate_recommendations`
- `tests/test_subscription_tracker_token_hardening.py::test_raw_token_not_stored_in_memory`

### Acceptance criteria

- Tests pass.
- Recommendations file becomes structured `recommendations.toml` with sections per `(auth_mode, model_family)`.
- The subscription tracker token-leak risk closed.

### Blocked by

PR-B5, PR-F1.

### Blocks

None.

### Rollback

`git revert`. TOIN reverts to global aggregation. No functional break.

---

## PR-F4 — `X-Forwarded-*` conditional in Rust path

**Branch:** `realign-F4-x-forwarded-conditional`
**Worktree:** `~/claude-projects/headroom-worktrees/realign-F4-x-forwarded-conditional`
**Risk:** **LOW**
**LOC:** +100

### Scope
Eliminate P5-53. The Rust proxy currently always adds `X-Forwarded-For`, `X-Forwarded-Proto`, `X-Forwarded-Host`, `X-Request-Id` to upstream-bound requests. Make this conditional: PAYG → add; OAuth → add; Subscription → skip (fingerprint risk).

### Files

**Modify:**
- `crates/headroom-proxy/src/headers.rs:103-117` — `add_x_forwarded_headers` takes an `AuthMode` parameter; no-ops on Subscription.

**Tests added:**
- `crates/headroom-proxy/tests/integration_x_forwarded_authmode.rs::payg_adds_xfwd`
- `crates/headroom-proxy/tests/integration_x_forwarded_authmode.rs::oauth_adds_xfwd`
- `crates/headroom-proxy/tests/integration_x_forwarded_authmode.rs::subscription_no_xfwd`

### Acceptance criteria

- Tests pass.

### Blocked by

PR-F1.

### Blocks

None.

### Rollback

`git revert`. Headers always added. Mild fingerprint regression for Subscription users.

---

## Phase F acceptance summary

After all 4 PRs land:

- ✅ `classify_auth_mode` helper detects PAYG / OAuth / Subscription
- ✅ Per-mode compression policy gates (auto-cache_control, prompt_cache_key, lossy compressors)
- ✅ TOIN aggregation key per `(auth_mode, model_family, structure_hash)`
- ✅ Subscription tracker doesn't store raw OAuth bearer
- ✅ `X-Forwarded-*` skipped on Subscription mode
- ✅ `accept-encoding` preserved on Subscription mode

**Phase F retires P5-52, P5-53, P5-54, P5-55, P5-56.**

After Phase F, fingerprint risk for Subscription CLI users is dramatically reduced.
