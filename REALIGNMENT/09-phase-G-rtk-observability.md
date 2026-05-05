# Phase G — RTK Breadth + Observability

**Goal:** Extend RTK coverage to more wrap-CLI agents; close the dead `tokens_saved_rtk` data plane; add per-invocation RTK metrics; add the cache-hit-rate, compression-ratio, token-validation observability surface that's missing today.

**Calendar:** 1 week.

**Shape:** 3 PRs.

**Decision context:** Per Agent F audit and 2026-05-01 user direction, **RTK stays on the wrap-CLI side, NOT the proxy side**. Proxy-side invocation is rejected because (a) cache hot zone risk on tool_result content compression, (b) parallel implementation with `crates/headroom-core/src/transforms/log_compressor.rs`, (c) RTK rewrites *commands* not *outputs* — different value proposition. "Integrate RTK with everything" reads as "extend wrap-CLI breadth + close the data plane + observability."

---

## PR-G1 — Wrap CLI breadth: cline, continue, goose, openhands

**Branch:** `realign-G1-wrap-more-agents`
**Worktree:** `~/claude-projects/headroom-worktrees/realign-G1-wrap-more-agents`
**Risk:** **LOW**
**LOC:** +800

### Scope
Eliminate P5-62. Add `headroom wrap cline`, `headroom wrap continue`, `headroom wrap goose`, `headroom wrap openhands` (extending the existing pattern from `wrap claude` / `wrap codex` / `wrap aider` / `wrap copilot` / `wrap cursor`). Each wrap subcommand:
1. Ensures the RTK binary is installed (`_ensure_rtk_binary()`).
2. Injects the `<!-- headroom:rtk-instructions -->` block into the agent's instruction file (AGENTS.md / .cursorrules / etc.).
3. Spawns the proxy (or attaches to a running one).
4. Launches the agent CLI with proxy env-var overrides.

### Files

**Add:**
- `headroom/cli/wrap/cline.py` — wrap implementation for Cline (agent that lives in VS Code; instruction file is `.clinerules`).
- `headroom/cli/wrap/continue_dev.py` — Continue agent (`.continue/config.json` configuration; system message injection).
- `headroom/cli/wrap/goose.py` — Goose agent (Block's CLI; `.goose/config.yaml`).
- `headroom/cli/wrap/openhands.py` — OpenHands (instruction injection via `OPENHANDS_INSTRUCTIONS` env var).

**Modify:**
- `headroom/cli/wrap/__init__.py` — register new subcommands.
- `headroom/cli/main.py` — `headroom wrap --help` lists new agents.
- `e2e/wrap/run.py` — extend the e2e runner to exercise the new wrappers (each wrapper has a smoke test that asserts: binary installed, instruction injected, proxy started, dummy LLM call works).

**Tests added:**
- `tests/test_cli/test_wrap_cline.py::test_wrap_cline_smoke`
- `tests/test_cli/test_wrap_continue.py::test_wrap_continue_smoke`
- `tests/test_cli/test_wrap_goose.py::test_wrap_goose_smoke`
- `tests/test_cli/test_wrap_openhands.py::test_wrap_openhands_smoke`
- `tests/test_cli/test_wrap_idempotent_inject.py::test_double_injection_no_duplicate_block` (for each new wrapper)

### Acceptance criteria

- Tests pass.
- Manual test: `headroom wrap cline -- claude-3-7-sonnet` launches a Cline session with the proxy in-front and RTK instructions in `.clinerules`.

### Blocked by

None.

### Blocks

None.

### Rollback

`git revert`. Existing wrappers continue working; new ones absent.

### Notes

- **Future agents to add later (not in this PR):** Roo Code, Devin-style CLIs, raw `gh copilot` standalone, gpt-engineer, sweep, smol-developer. Add as separate PRs as adoption justifies.

---

## PR-G2 — Wire `tokens_saved_rtk` data plane

**Branch:** `realign-G2-tokens-saved-rtk`
**Worktree:** `~/claude-projects/headroom-worktrees/realign-G2-tokens-saved-rtk`
**Risk:** **LOW**
**LOC:** +200

### Scope
Eliminate P5-60. The `tokens_saved_rtk` field on `SubscriptionContribution` (`headroom/subscription/models.py:260`) exists but is never populated. Wire it: poll `rtk gain --format json` periodically (already done by `_get_rtk_stats` in `helpers.py:132`), diff the cumulative `tokens_saved` since last snapshot, and feed into `tracker.update_session_savings(tokens_saved_rtk=delta)`.

### Files

**Modify:**
- `headroom/subscription/tracker.py` — add `_last_rtk_tokens_saved: int = 0` state; on every `update_session_savings` call, fetch `_get_rtk_stats()`, compute `delta = current.tokens_saved - self._last_rtk_tokens_saved`, set `tokens_saved_rtk=delta`, update state.
- `headroom/proxy/helpers.py:132` — `_get_rtk_stats` returns `RtkStats { invocations: int, tokens_saved: int, last_run_at: datetime }`. Memoization stays at 5s.

**Tests added:**
- `tests/test_subscription_tracker_rtk_wired.py::test_tokens_saved_rtk_populated_from_rtk_stats`
- `tests/test_subscription_tracker_rtk_wired.py::test_delta_computed_correctly_across_polls`
- `tests/test_subscription_tracker_rtk_wired.py::test_rtk_failure_zero_delta_no_throw`

### Acceptance criteria

- Tests pass.
- A wrap session with RTK invocations produces `tokens_saved_rtk > 0` after the session ends.

### Blocked by

None.

### Blocks

None.

### Rollback

`git revert`. `tokens_saved_rtk` returns to silent zero.

---

## PR-G3 — Per-invocation RTK metrics + observability gaps

**Branch:** `realign-G3-rtk-metrics-and-obs`
**Worktree:** `~/claude-projects/headroom-worktrees/realign-G3-rtk-metrics-and-obs`
**Risk:** **LOW**
**LOC:** +600

### Scope
Eliminate P6-68, P6-69, P5-58, P4-41, P4-42, P4-45, and P5-61 (documentation). Add Prometheus metrics:
- `wrap_rtk_invocations_total{tool}` — derived from `rtk gain --format json` polling (`tool` label is the `git`, `ls`, `cargo`, etc. command).
- `wrap_rtk_tokens_saved_per_session` — histogram, populated at session end.
- `proxy_cache_hit_rate_per_session` — histogram, computed from `usage.cache_read_input_tokens / total_input_tokens` per session.
- `proxy_compression_ratio_by_strategy{strategy, content_type}` — histogram.
- `proxy_compression_rejected_by_token_check_total{strategy}` — counter (already in PR-B4; ensure it's exported here).
- `proxy_passthrough_bytes_modified_total{path}` — gauge that **must stay 0** outside compression-on path. Alarm if non-zero.
- `proxy_rate_limit_remaining_*` — extracted from upstream response headers.
- `proxy_service_tier_count_total{tier}` — counter for `service_tier` distribution.
- `proxy_response_status_count_total{status}` — `incomplete | failed | cancelled | completed | in_progress`.
- `proxy_image_generation_call_log_redacted_total` — counter for log redactions of multi-MB base64.

Plus image base64 log redaction (P4-45) lands here.

### Files

**Modify:**
- `crates/headroom-proxy/src/observability/prometheus.rs` — add all new metrics.
- `crates/headroom-proxy/src/sse/anthropic.rs` — emit `proxy_cache_hit_rate_per_session` from `usage.cache_read_input_tokens / total_input_tokens` on `message_delta`.
- `crates/headroom-proxy/src/sse/openai_responses.rs` — emit on `response.completed`.
- `crates/headroom-proxy/src/sse/openai_chat.rs` — emit on final usage chunk.
- `crates/headroom-proxy/src/handlers/responses.rs` — extract and log `service_tier`.
- `crates/headroom-proxy/src/handlers/responses.rs` — log `incomplete_details.reason` when `status == incomplete`.
- `headroom/proxy/request_logger.py` — redact base64 strings >1024 bytes; replace with `<base64 truncated, X bytes>`.
- `crates/headroom-proxy/src/observability/cache_hit_rate.rs` — new module.
- `crates/headroom-proxy/src/observability/compression_ratio.rs` — new module.

**Add:**
- `docs/observability.md` — documents every metric, what it means, what an operator should do when it drifts.
- `docs/rtk-architecture.md` — explicitly documents the decision: RTK is wrap-CLI-only; proxy-side invocation is rejected. Includes the rationale (cache hot zone, parallel-impl with log_compressor, command-rewrite-vs-output-rewrite). Future contributors hit this doc before considering a proxy-side RTK call.

**Tests added:**
- `crates/headroom-proxy/tests/integration_metrics.rs::cache_hit_rate_emitted_per_session`
- `crates/headroom-proxy/tests/integration_metrics.rs::compression_ratio_emitted_per_strategy`
- `crates/headroom-proxy/tests/integration_metrics.rs::passthrough_bytes_modified_zero_when_no_compression`
- `crates/headroom-proxy/tests/integration_metrics.rs::service_tier_logged`
- `crates/headroom-proxy/tests/integration_metrics.rs::incomplete_status_logged_with_reason`
- `tests/test_image_log_redaction.py::test_large_base64_truncated`

### Acceptance criteria

- All tests pass.
- Manual scrape of `/metrics` shows the new metric families.
- `docs/rtk-architecture.md` reviewed and approved.

### Blocked by

None.

### Blocks

None.

### Rollback

`git revert`. Loses observability; no functional regression.

---

## Phase G acceptance summary

After all 3 PRs land:

- ✅ Wrap CLI coverage extends to cline, continue, goose, openhands
- ✅ `tokens_saved_rtk` field populated end-to-end
- ✅ Per-invocation RTK Prometheus metrics
- ✅ Per-session cache-hit-rate metric
- ✅ Per-block compression-ratio histogram
- ✅ Token-validation rejection counter
- ✅ Passthrough-bytes-modified gauge (alarm-able)
- ✅ Rate-limit headers observed and exported
- ✅ `service_tier` distribution metric
- ✅ Response status (`incomplete | failed | cancelled`) logged with reason
- ✅ Image base64 log redaction
- ✅ `docs/rtk-architecture.md` documents the keep-RTK-on-wrap-side decision

**Phase G retires P4-41, P4-42, P4-45, P5-58, P5-60, P5-61, P5-62, P6-68, P6-69, P6-72.**
