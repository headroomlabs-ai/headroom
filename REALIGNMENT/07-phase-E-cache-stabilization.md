# Phase E — Phase 3 Cache Stabilization

**Goal:** Add the cache-stabilization surface that today is **completely missing**: tool array deterministic sort, recursive JSON Schema key sort, auto `cache_control` placement (Anthropic), `prompt_cache_key` auto-injection (OpenAI), volatile-content detector with customer warning (no rewrite), cache-bust drift telemetry. These are guide §8.5, §9.11, §6.2, §4.17 implementations and the "Phase 3" of the guide's implementation checklist.

**Calendar:** 1 week.

**Shape:** 6 PRs. Mostly parallel; E3 + E4 should land paired (per-mode policy).

---

## PR-E1 — Tool array deterministic sort (Rust)

**Branch:** `realign-E1-tool-array-sort`
**Worktree:** `~/claude-projects/headroom-worktrees/realign-E1-tool-array-sort`
**Risk:** **LOW**
**LOC:** +200

### Scope
Eliminate P3-28. Sort `tools[]` alphabetically by name on the way out. Idempotent: re-sorting an already-sorted array is a no-op. Implementation matches Python's existing `_sort_tools_deterministically` (`headroom/proxy/handlers/anthropic.py:34-58`) — same sort key, same output bytes (modulo serialization).

### Files

**Add:**
- `crates/headroom-proxy/src/compression/tool_def_normalize.rs`:
  ```rust
  pub fn sort_tools_deterministically(tools: &mut Vec<&RawValue>) -> Result<()> {
      // Sort key: tool["name"] string, fallback to MD5(serialized) for unnamed tools.
      tools.sort_by_key(|t| {
          let parsed: serde_json::Value = serde_json::from_str(t.get()).unwrap_or_default();
          parsed.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string()
      });
      Ok(())
  }
  ```

**Modify:**
- `crates/headroom-proxy/src/compression/live_zone_anthropic.rs` — call `sort_tools_deterministically` on the request body's `tools` array before forwarding (only on PAYG; gated by Phase F PR-F2).
- `crates/headroom-proxy/src/compression/live_zone_openai.rs` — same.
- `crates/headroom-proxy/src/compression/live_zone_responses.rs` — same.

**Tests added:**
- `crates/headroom-proxy/tests/integration_tool_sort.rs::sort_alphabetic_by_name`
- `crates/headroom-proxy/tests/integration_tool_sort.rs::idempotent_resort_no_change`
- `crates/headroom-proxy/tests/integration_tool_sort.rs::byte_stable_across_runs`

### Acceptance criteria

- Tests pass.
- Output `tools[]` byte-equal between Rust and Python sort implementations.

### Blocked by

PR-B2.

### Blocks

PR-E2.

### Rollback

`git revert`. Rust path matches client's tool order (subject to the cache-bust risk).

---

## PR-E2 — Recursive JSON Schema key sort

**Branch:** `realign-E2-schema-key-sort`
**Worktree:** `~/claude-projects/headroom-worktrees/realign-E2-schema-key-sort`
**Risk:** **MEDIUM** (recursive sort over tool schemas; risk of breaking schema semantics if there's an `if`/`then`/`else` or `oneOf` ordering invariant — there isn't, but verify)
**LOC:** +300

### Scope
Eliminate P3-29. Recursively sort JSON Schema object keys in every tool's `input_schema`. This includes nested `properties`, `definitions`, `oneOf`, `anyOf`, `allOf`, `if/then/else`, `additionalProperties`, etc.

### Files

**Modify:**
- `crates/headroom-proxy/src/compression/tool_def_normalize.rs` — add `sort_schema_keys_recursive`. Walks every Object node; replaces with `IndexMap` rebuilt in alphabetic key order. Preserves Array order (JSON Schema arrays are ordered: `prefixItems`, `oneOf` alternatives, etc.).

**Tests added:**
- `crates/headroom-proxy/tests/integration_schema_sort.rs::flat_schema_keys_sorted`
- `crates/headroom-proxy/tests/integration_schema_sort.rs::nested_properties_sorted`
- `crates/headroom-proxy/tests/integration_schema_sort.rs::oneof_array_order_preserved`
- `crates/headroom-proxy/tests/integration_schema_sort.rs::definitions_keys_sorted`
- `crates/headroom-proxy/tests/integration_schema_sort.rs::idempotent_resort`

### Acceptance criteria

- Tests pass.
- Snapshot test on a real production tool schema (e.g., Claude Code's `Read` tool) — pin the sorted bytes.

### Blocked by

PR-E1.

### Blocks

PR-E5.

### Rollback

`git revert`. Schema keys reflect customer ordering.

---

## PR-E3 — Auto `cache_control` breakpoint placement (Anthropic)

**Branch:** `realign-E3-cache-control-auto-place`
**Worktree:** `~/claude-projects/headroom-worktrees/realign-E3-cache-control-auto-place`
**Risk:** **MEDIUM-HIGH** (auto-adds bytes to client request — only on PAYG)
**LOC:** +400

### Scope
Eliminate P3-31. When PAYG mode is detected (Phase F PR-F1) and the customer has not set any `cache_control` markers, auto-place up to 4 ephemeral markers at:
1. End of system prompt (1 marker)
2. End of `tools[]` (1 marker)
3. After the last stable conversation history boundary (1 marker; configurable threshold for "stable")
4. Before the latest user message (1 marker)

OAuth and subscription modes: never auto-place (could void scope).

### Files

**Add:**
- `crates/headroom-proxy/src/compression/cache_control.rs` — `pub fn auto_place_breakpoints(body: &mut serde_json::Value, auth_mode: AuthMode)`. Walks the structure; appends `cache_control: {type: "ephemeral"}` to the trailing block of system, tools, history, and current user message.

**Modify:**
- `crates/headroom-proxy/src/compression/live_zone_anthropic.rs` — call `auto_place_breakpoints` on PAYG only.

**Tests added:**
- `crates/headroom-proxy/tests/integration_cache_control_auto.rs::payg_auto_places_4_markers`
- `crates/headroom-proxy/tests/integration_cache_control_auto.rs::oauth_no_auto_placement`
- `crates/headroom-proxy/tests/integration_cache_control_auto.rs::subscription_no_auto_placement`
- `crates/headroom-proxy/tests/integration_cache_control_auto.rs::customer_set_markers_respected_no_addition`
- `crates/headroom-proxy/tests/integration_cache_control_auto.rs::ttl_ordering_correct_1h_before_5m`

### Acceptance criteria

- Tests pass.
- Customer requests with existing markers are unmodified.
- A representative PAYG request gets 4 markers in the right positions.

### Blocked by

PR-A4, PR-F1, PR-F2.

### Blocks

None.

### Rollback

`git revert`. Customers without their own `cache_control` markers don't get auto-placement; cache benefit smaller but correct.

---

## PR-E4 — `prompt_cache_key` auto-injection (OpenAI)

**Branch:** `realign-E4-prompt-cache-key-inject`
**Worktree:** `~/claude-projects/headroom-worktrees/realign-E4-prompt-cache-key-inject`
**Risk:** **MEDIUM**
**LOC:** +250

### Scope
Eliminate P3-30. For OpenAI Chat Completions and Responses requests on PAYG mode where the customer has not set `prompt_cache_key`, auto-inject one derived from a stable session hash. OAuth/subscription modes: never inject (the CLI may already populate this and Headroom must preserve byte-for-byte).

### Files

**Modify:**
- `crates/headroom-proxy/src/compression/live_zone_openai.rs` — add `inject_prompt_cache_key` step.
- `crates/headroom-proxy/src/compression/live_zone_responses.rs` — same.

**Add:**
- `crates/headroom-proxy/src/session.rs` — `pub fn derive_prompt_cache_key(session_id: &str, model: &str) -> String` — returns `{session_id}_{model_family}` (deterministic per session+model).

**Tests added:**
- `crates/headroom-proxy/tests/integration_prompt_cache_key.rs::payg_auto_injects_when_absent`
- `crates/headroom-proxy/tests/integration_prompt_cache_key.rs::customer_value_preserved`
- `crates/headroom-proxy/tests/integration_prompt_cache_key.rs::oauth_no_injection`
- `crates/headroom-proxy/tests/integration_prompt_cache_key.rs::subscription_no_injection`
- `crates/headroom-proxy/tests/integration_prompt_cache_key.rs::same_session_same_key_deterministic`

### Acceptance criteria

- Tests pass.

### Blocked by

PR-F1, PR-F2.

### Blocks

None.

### Rollback

`git revert`. OpenAI cache routing less sticky on PAYG; functional.

---

## PR-E5 — Volatile-content detector with customer warning (no rewrite)

**Branch:** `realign-E5-volatile-detector`
**Worktree:** `~/claude-projects/headroom-worktrees/realign-E5-volatile-detector`
**Risk:** **LOW**
**LOC:** +400

### Scope
Eliminate P3-32. Detect dynamic content in the early prompt (timestamps, UUIDs, JWT tokens, build hashes, randomized IDs) and surface to the customer via a log line and a Prometheus metric. **Do not rewrite.** This is what Python's `cache_aligner.py` was *trying* to do correctly (the rewrite path is gone in Phase A PR-A2).

### Files

**Add:**
- `crates/headroom-core/src/transforms/volatile_detector.rs`:
  ```rust
  pub struct VolatileDetector { /* compiled regex set */ }

  impl VolatileDetector {
      pub fn scan(&self, content: &str) -> Vec<VolatileFinding>;
  }

  pub struct VolatileFinding {
      pub kind: VolatileKind,  // Timestamp | Uuid | Jwt | BuildHash | ...
      pub byte_offset: usize,
      pub matched: String,
      pub recommendation: String,  // "Move to metadata"
  }
  ```
  Patterns:
  - ISO 8601 timestamp regex
  - UUID v4 regex
  - JWT shape (`eyJ...`)
  - Long hex strings >=32 chars (build hashes)
  - Unix epoch timestamps within an order of magnitude of `now`

**Modify:**
- `crates/headroom-proxy/src/compression/live_zone_anthropic.rs` — run scanner on system prompt; if findings, log warning and increment metric.
- `crates/headroom-proxy/src/compression/live_zone_openai.rs` — same on `instructions` field.
- `crates/headroom-proxy/src/compression/live_zone_responses.rs` — same.

**Tests added:**
- `crates/headroom-core/tests/volatile_detector.rs::detects_iso_8601_timestamp`
- `crates/headroom-core/tests/volatile_detector.rs::detects_uuid_v4`
- `crates/headroom-core/tests/volatile_detector.rs::detects_jwt_shape`
- `crates/headroom-core/tests/volatile_detector.rs::detects_build_hash`
- `crates/headroom-core/tests/volatile_detector.rs::no_false_positives_on_normal_prose`
- `crates/headroom-proxy/tests/integration_volatile.rs::warning_logged_on_volatile_system_prompt`
- `crates/headroom-proxy/tests/integration_volatile.rs::system_prompt_bytes_unchanged`

### Acceptance criteria

- Tests pass.
- Detector runs in <1ms on a 4KB system prompt (don't slow the request path).

### Blocked by

PR-E2.

### Blocks

None.

### Rollback

`git revert`. Detector loses; no functional regression.

---

## PR-E6 — Cache-bust drift detector telemetry

**Branch:** `realign-E6-cache-bust-detector`
**Worktree:** `~/claude-projects/headroom-worktrees/realign-E6-cache-bust-detector`
**Risk:** **LOW**
**LOC:** +500

### Scope
Eliminate P3-35. Hash the prefix (system + tools + first N stable messages) of every request. Track per-session prefix hashes. When the prefix hash changes between turns, increment a counter and log a `prefix_drift` event with which subsystem mutated (likely none after Phase A — but if it happens, we want to know).

### Files

**Add:**
- `crates/headroom-proxy/src/observability/prefix_drift.rs`:
  ```rust
  pub struct PrefixDriftDetector {
      // Keyed by session_id; stores last-seen prefix hash + timestamp.
      cache: Cache<SessionId, (PrefixHash, Instant)>,
  }

  impl PrefixDriftDetector {
      pub fn check(&self, session_id: &str, body: &serde_json::Value) -> DriftCheck;
  }

  pub enum DriftCheck {
      FirstSeen,
      Stable,
      Drifted { previous_hash: PrefixHash, new_hash: PrefixHash, age: Duration },
  }
  ```

**Modify:**
- `crates/headroom-proxy/src/observability/prometheus.rs` — add `prefix_drift_detected_total{provider, model}` counter.

**Tests added:**
- `crates/headroom-proxy/tests/integration_prefix_drift.rs::stable_prefix_no_drift`
- `crates/headroom-proxy/tests/integration_prefix_drift.rs::system_change_detected_as_drift`
- `crates/headroom-proxy/tests/integration_prefix_drift.rs::tools_reorder_detected`

### Acceptance criteria

- Tests pass.
- A canary request that mutates the system prompt mid-session triggers the counter.

### Blocked by

PR-B2.

### Blocks

None.

### Rollback

`git revert`. Loses observability; no functional regression.

---

## Phase E acceptance summary

After all 6 PRs land:

- ✅ Tools alphabetically sorted (deterministic, idempotent)
- ✅ JSON Schema keys recursively sorted
- ✅ `cache_control` auto-placement on PAYG (4 markers)
- ✅ `prompt_cache_key` auto-injection on PAYG (OpenAI)
- ✅ Volatile-content detector + customer warning (no rewrite)
- ✅ Cache-bust drift telemetry per session

**Phase E retires P3-28 through P3-32, P3-35.**
