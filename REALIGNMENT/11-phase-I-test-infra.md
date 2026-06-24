# Phase I — Test Infrastructure (Continuous, Parallel)

**Goal:** Build the test/CI surface that makes the realignment safe to land and stays safe afterward. This phase runs **in parallel** with all other phases — its PRs land alongside the corresponding feature work.

**Calendar:** Continuous. Each test PR pairs with the feature PR it gates.

**Shape:** ~10 PRs, mostly small, parallelizable.

---

## PR-I1 — SHA-256 byte-faithful round-trip test on recorded production payload

**Branch:** `realign-I1-sha256-round-trip`
**Worktree:** `~/claude-projects/headroom-worktrees/realign-I1-sha256-round-trip`
**Risk:** **LOW**
**LOC:** +400

### Scope
Eliminate P6-63. The single most important regression test for cache safety. Records a real Anthropic `/v1/messages` payload (sanitized of secrets), sends it through the proxy with compression off, asserts SHA-256 byte-equality at the upstream mock.

### Files

**Add:**
- `tests/fixtures/anthropic_messages_request_real.json` — sanitized real payload. Includes:
  - `system` as a list of blocks with `cache_control` markers
  - `tools[]` with non-trivial JSON Schema (nested properties, oneOf, definitions)
  - `messages[]` with mixed block types: text, image, thinking + signature, tool_use with non-trivial input, tool_result with array content + image
  - Non-ASCII characters (`🔥`, CJK)
  - Numeric values: `temperature: 1.0`, large integers, scientific notation
  - `cache_control` markers on `messages[*].content[*]`
  - `null` and absent fields side-by-side
- `tests/fixtures/openai_chat_completions_real.json` — same shape for OpenAI Chat.
- `tests/fixtures/openai_responses_real.json` — same shape for Responses, includes V4A patch, local_shell_call, reasoning, compaction items.
- `crates/headroom-proxy/tests/integration_byte_faithful.rs::sha256_round_trip_anthropic_messages_passthrough`
- `crates/headroom-proxy/tests/integration_byte_faithful.rs::sha256_round_trip_anthropic_messages_compression_off_via_auth_mode`
- `crates/headroom-proxy/tests/integration_byte_faithful.rs::sha256_round_trip_openai_chat`
- `crates/headroom-proxy/tests/integration_byte_faithful.rs::sha256_round_trip_openai_responses`
- `tests/test_python_byte_faithful.py::test_sha256_round_trip_anthropic_passthrough` — Python side, gates Phase H readiness.

**Modify:**
- `Makefile` — `make test-byte-faithful` target that runs all of the above.
- `.github/workflows/rust.yml` — make `make test-byte-faithful` a per-PR gate.

### Acceptance criteria

- All tests pass after Phase A PR-A3, PR-A4 land.
- Test runs in <5 seconds.

### Blocked by

PR-A1.

### Blocks

PR-H1 (Phase H gating).

---

## PR-I2 — SSE corner-case fixtures + fuzz tests

**Branch:** `realign-I2-sse-corner-cases`
**Worktree:** `~/claude-projects/headroom-worktrees/realign-I2-sse-corner-cases`
**Risk:** **LOW**
**LOC:** +800

### Scope
Eliminate P6-66, P6-71. Record fixtures for every SSE corner case the audit identified.

### Files

**Add:**
- `crates/headroom-proxy/tests/fixtures/sse/anthropic_thinking_with_signature.sse`
- `crates/headroom-proxy/tests/fixtures/sse/anthropic_interleaved_blocks.sse` (synthetic; locks the index-keyed model)
- `crates/headroom-proxy/tests/fixtures/sse/anthropic_input_json_delta_split_utf8.sse` (4-byte emoji split across chunks)
- `crates/headroom-proxy/tests/fixtures/sse/anthropic_ping_mid_stream.sse`
- `crates/headroom-proxy/tests/fixtures/sse/anthropic_error_mid_stream.sse`
- `crates/headroom-proxy/tests/fixtures/sse/openai_chat_tool_call_split.sse`
- `crates/headroom-proxy/tests/fixtures/sse/openai_chat_done_with_trailing_whitespace.sse`
- `crates/headroom-proxy/tests/fixtures/sse/openai_responses_out_of_order_done.sse`
- `crates/headroom-proxy/tests/fixtures/sse/openai_429_as_application_json.http` (HTTP error, not SSE)
- `crates/headroom-proxy/tests/fixtures/sse/anthropic_tcp_drop_before_message_stop.sse`
- `crates/headroom-proxy/tests/integration_sse_fixtures.rs` — runs every fixture against the parser; asserts expected state.
- Property test in `crates/headroom-proxy/tests/proptest_sse.rs::sse_parser_no_panic_on_arbitrary_bytes`.

### Acceptance criteria

- All fixtures parse correctly.
- Property test runs 10K random byte sequences without panic.

### Blocked by

PR-C1.

### Blocks

None.

---

## PR-I3 — Property tests for compression invariants

**Branch:** `realign-I3-compression-proptest`
**Worktree:** `~/claude-projects/headroom-worktrees/realign-I3-compression-proptest`
**Risk:** **LOW**
**LOC:** +500

### Scope
Property tests that exercise the realigned compressor invariants:
1. **Determinism**: `compress(input) == compress(input)` for any valid input.
2. **Idempotence**: `compress(compress(input).output) == compress(input).output` (compressing already-compressed content is a no-op).
3. **Token-non-increasing**: `tokens(output) <= tokens(input)` for any valid input — fallback ensures this.
4. **Position preservation**: For all valid block arrays, `len(compressed) == len(original)`; block types match per index; `tool_use_id` / `call_id` preserved.
5. **Frozen-prefix integrity**: For any `frozen_count`, messages `0..frozen_count` are byte-equal in input and output.

### Files

**Add:**
- `crates/headroom-core/tests/proptest_compression.rs` — proptest strategies for `Block`, `Message`, `RequestBody`. Five property tests above.
- `crates/headroom-core/tests/proptest_ccr.rs` — round-trip property test: `decompress(compress(content)) == content` for any content.

### Acceptance criteria

- All property tests pass with `cases = 1000`.

### Blocked by

PR-B4.

### Blocks

None.

---

## PR-I4 — Real-traffic shadow test (Python vs Rust)

**Branch:** `realign-I4-shadow-test`
**Worktree:** `~/claude-projects/headroom-worktrees/realign-I4-shadow-test`
**Risk:** **MEDIUM**
**LOC:** +1000

### Scope
Eliminate P6-67. A canary deployment runs the Python proxy and the Rust proxy side-by-side; for every request, both produce upstream-bound bytes; a comparator hashes both and reports SHA-256 mismatch percentage. Goal: 99.9% byte-equality before Phase H deletes Python.

### Files

**Add:**
- `e2e/shadow/runner.py` — splits incoming requests into "primary" (Python, response goes to client) and "shadow" (Rust, response discarded). Hashes upstream-bound bytes from both; reports per-request, per-endpoint, per-auth-mode mismatch rates.
- `e2e/shadow/dashboard.py` — Grafana dashboard JSON that visualizes the shadow comparison.
- `docs/operations/shadow-deploy.md` — operator guide for running the shadow test.

### Acceptance criteria

- Shadow test runs against a non-trivial corpus (10K requests) and reports.
- Mismatch rate <0.1% before declaring Phase H ready.

### Blocked by

PR-A1 through PR-G3.

### Blocks

PR-H1.

---

## PR-I5 — Promote stub parity comparators to real

**Branch:** `realign-I5-parity-stubs-to-real`
**Worktree:** `~/claude-projects/headroom-worktrees/realign-I5-parity-stubs-to-real`
**Risk:** **MEDIUM**
**LOC:** +800

### Scope
Eliminate P6-64. `crates/headroom-parity/src/lib.rs:172-174` stubs three comparators with `bail!()`:
- `ccr` (25 fixtures recorded; comparator is `Skipped`)
- `log_compressor` (20 fixtures recorded; comparator is `Skipped`)
- `cache_aligner` (20 fixtures recorded; comparator is `Skipped`)

Build real comparators that exercise the Rust port against the recorded Python fixtures.

### Files

**Modify:**
- `crates/headroom-parity/src/lib.rs` — replace `stub_comparator!(CCRComparator, ...)` etc. with real impls.
- Add `CcrComparator`, `LogCompressorComparator`, `CacheAlignerComparator` modules.

**Tests added:**
- Each comparator has a `harness_reports_match_for_real_fixture` test.

### Acceptance criteria

- All three comparators run against their recorded fixtures.
- Mismatch rate is 0% (parity locked).

### Blocked by

PR-B3 (LogCompressor live in proxy); PR-B7 (CCR hardening); PR-A2 (CacheAligner detector).

### Blocks

PR-I6.

---

## PR-I6 — Make `make test-parity` a per-PR CI gate

**Branch:** `realign-I6-parity-per-pr-gate`
**Worktree:** `~/claude-projects/headroom-worktrees/realign-I6-parity-per-pr-gate`
**Risk:** **LOW**
**LOC:** +50

### Scope
Eliminate P6-65. Today parity is a soft nightly with `continue-on-error: true`. Move it to per-PR with `Diff` failures blocking merge; `Skipped` allowed (so still-stubbed comparators don't block).

### Files

**Modify:**
- `.github/workflows/rust.yml:125-149` — move parity job from `cron` schedule to `pull_request` trigger. Remove `continue-on-error`. Set parity-run flags so `Skipped` is acceptable but `Diff` fails the build.
- `Makefile` — `test-parity` already exists; ensure it's invokable in CI.

### Acceptance criteria

- A purposely-broken Rust port that diverges from a recorded fixture fails CI on the next PR.

### Blocked by

PR-I5.

### Blocks

None.

---

## PR-I7 — Cache hot zone non-mutation tests

**Branch:** `realign-I7-cache-hot-zone-tests`
**Worktree:** `~/claude-projects/headroom-worktrees/realign-I7-cache-hot-zone-tests`
**Risk:** **LOW**
**LOC:** +600

### Scope
Test that nothing — compression, memory injection, tool registration — mutates the cache hot zone.

### Files

**Add:**
- `crates/headroom-proxy/tests/integration_cache_hot_zone.rs::system_byte_equal_under_compression`
- `crates/headroom-proxy/tests/integration_cache_hot_zone.rs::tools_byte_equal_under_compression` (modulo Phase E sort + schema-key sort, which is deterministic — assert post-sort byte-equal)
- `crates/headroom-proxy/tests/integration_cache_hot_zone.rs::frozen_messages_byte_equal_under_compression`
- `crates/headroom-proxy/tests/integration_cache_hot_zone.rs::reasoning_encrypted_content_byte_equal`
- `crates/headroom-proxy/tests/integration_cache_hot_zone.rs::thinking_signature_byte_equal`
- `crates/headroom-proxy/tests/integration_cache_hot_zone.rs::redacted_thinking_data_byte_equal`
- `crates/headroom-proxy/tests/integration_cache_hot_zone.rs::compaction_encrypted_content_byte_equal`
- `crates/headroom-proxy/tests/integration_cache_hot_zone.rs::v4a_patch_diff_byte_equal`
- `crates/headroom-proxy/tests/integration_cache_hot_zone.rs::local_shell_call_argv_array_preserved`

### Acceptance criteria

- All tests pass.

### Blocked by

PR-B2.

### Blocks

None.

---

## PR-I8 — Tool-definition byte-stability snapshot tests

**Branch:** `realign-I8-tool-def-snapshot`
**Worktree:** `~/claude-projects/headroom-worktrees/realign-I8-tool-def-snapshot`
**Risk:** **LOW**
**LOC:** +300

### Scope
For every tool definition Headroom auto-injects (`ccr_retrieve`, `memory_*`), pin the bytes via golden-file snapshot. Any change to a definition fails CI; a deliberate change requires updating the snapshot. This prevents accidental cache busts on Headroom deploys.

### Files

**Add:**
- `crates/headroom-core/tests/tool_def_byte_stability.rs::ccr_retrieve_definition_anthropic_byte_stable`
- `crates/headroom-core/tests/tool_def_byte_stability.rs::ccr_retrieve_definition_openai_byte_stable`
- `crates/headroom-core/tests/tool_def_byte_stability.rs::memory_save_definition_byte_stable`
- `crates/headroom-core/tests/tool_def_byte_stability.rs::memory_search_definition_byte_stable`
- Golden files under `crates/headroom-core/tests/golden/tool_defs/`.

### Acceptance criteria

- Tests pass.
- Renaming a field in a tool definition fails CI; updating the golden file fixes it.

### Blocked by

PR-B7.

### Blocks

None.

---

## PR-I9 — Continuous cache-hit-rate alarm

**Branch:** `realign-I9-cache-hit-rate-alarm`
**Worktree:** `~/claude-projects/headroom-worktrees/realign-I9-cache-hit-rate-alarm`
**Risk:** **LOW**
**LOC:** +200

### Scope
A Prometheus alarm rule that fires when the per-session cache hit rate (`proxy_cache_hit_rate_per_session`) drops below a baseline (90% of yesterday's rolling p50) for >15 minutes. Catches drift in production.

### Files

**Add:**
- `docs/operations/prometheus_rules.yaml` — alarm rule definition.
- `docs/operations/runbook.md` — what to do when the alarm fires.

### Acceptance criteria

- Rule passes `promtool check rules`.
- Runbook reviewed.

### Blocked by

PR-G3.

### Blocks

None.

---

## PR-I10 — Replace fake RTK shim with real RTK in wrap E2E

**Branch:** `realign-I10-real-rtk-in-e2e`
**Worktree:** `~/claude-projects/headroom-worktrees/realign-I10-real-rtk-in-e2e`
**Risk:** **LOW**
**LOC:** +200

### Scope
Eliminate P6-72. `e2e/wrap/run.py:250-267` has an `rtk` shim that just prints "rtk shim" and exits 0. Replace with real RTK invocation in CI, OR keep the shim but add an explicit assertion that the shim was used (so it doesn't silently mask a missing RTK install).

### Files

**Modify:**
- `e2e/wrap/run.py:250-267` — switch to real RTK download in CI; cache the binary.
- `.github/workflows/wrap-e2e.yml` — pin RTK version.

### Acceptance criteria

- E2E tests download real RTK and exercise its rewrite behavior.

### Blocked by

None.

### Blocks

None.

---

## Phase I acceptance summary

After all 10 PRs land:

- ✅ SHA-256 byte-faithful round-trip test gates CI
- ✅ SSE corner-case fixtures + fuzz tests
- ✅ Property tests for compression invariants
- ✅ Real-traffic shadow test comparing Python vs Rust
- ✅ Stub parity comparators promoted to real
- ✅ `make test-parity` is a per-PR gate
- ✅ Cache hot zone non-mutation tests
- ✅ Tool-definition byte-stability snapshot tests
- ✅ Cache-hit-rate Prometheus alarm
- ✅ Real RTK in wrap E2E

**Phase I retires P6-63 through P6-72.**

After Phase I, regressing the realignment requires actively breaking tests — the cache safety properties become continuously enforced.
