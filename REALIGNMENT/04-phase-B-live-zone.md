# Phase B — Live-Zone-Only Compression Engine

**Goal:** Delete ~10 K LOC of architectural over-build (ICM, scoring, relevance, rolling-window, progressive-summarizer, tool-crusher); build the correct architecture: per-block compression on the live zone only, with type-aware dispatch, token validation, and CCR hardening.

**Calendar:** 2 weeks. PR-B1 is the big delete (high-LOC, lower-risk-than-it-looks because the code was unreachable after Phase A). PR-B2..B7 build the replacement.

**Shape:** 7 PRs. B1 is independent; B2..B5 depend on B1; B6 + B7 layer on B2.

---

## PR-B1 — The big delete: retire ICM and its dependencies

**Branch:** `realign-B1-delete-icm-and-deps`
**Worktree:** `~/claude-projects/headroom-worktrees/realign-B1-delete-icm-and-deps`
**Risk:** **MEDIUM** (large diff, but most code became unreachable after Phase A PR-A1)
**LOC:** **-10,000 / +50** (the big retirement)

### Scope
Delete the wrong-mental-model machinery wholesale. After Phase A PR-A1 made the proxy a passthrough on `/v1/messages`, none of this code is reached at runtime; this PR removes the source so future contributors can't re-wire it.

### Files

**Delete (Python):**
- `headroom/transforms/intelligent_context.py` (1077 LOC)
- `headroom/transforms/rolling_window.py` (395 LOC)
- `headroom/transforms/progressive_summarizer.py` (508 LOC)
- `headroom/transforms/scoring.py` (459 LOC)
- `headroom/transforms/tool_crusher.py` (338 LOC)

**Delete (Rust):**
- `crates/headroom-core/src/context/manager.rs`
- `crates/headroom-core/src/context/config.rs`
- `crates/headroom-core/src/context/workspace.rs`
- `crates/headroom-core/src/context/candidate.rs`
- `crates/headroom-core/src/context/ccr_drop.rs`
- `crates/headroom-core/src/context/strategy/mod.rs`
- `crates/headroom-core/src/context/strategy/drop_by_score.rs`
- All of `crates/headroom-core/src/scoring/*.rs` (~1500 LOC)
- All of `crates/headroom-core/src/relevance/*.rs` (~1600 LOC)
- `crates/headroom-core/.fastembed_cache/` directory and its `bge-small-en-v1.5` ONNX artifacts (~50 MB)

**Move:**
- `crates/headroom-core/src/context/safety.rs` → `crates/headroom-core/src/transforms/safety.rs`. Update all callers' `use` paths. The tool-pair atomicity logic is preserved verbatim (it's correct and live-zone code needs it).

**Modify:**
- `crates/headroom-core/src/lib.rs` — remove `pub mod context;`, `pub mod scoring;`, `pub mod relevance;`. Add `pub use transforms::safety;`.
- `crates/headroom-core/src/context/mod.rs` — delete (empty after move).
- `crates/headroom-proxy/src/lib.rs` — no changes (already doesn't reach into deleted modules after PR-A1).
- `crates/headroom-py/src/lib.rs` — remove any PyO3 exports of `MessageScorer`, `IntelligentContextManager`, etc. (per agent reports, MessageScorer was exposed in PR #338/#343).
- `headroom/transforms/__init__.py` — remove imports of deleted modules.
- `headroom/proxy/handlers/anthropic.py` — remove all imports / call sites of `IntelligentContextManager`. (Agent C found these at multiple locations; track via `grep -n IntelligentContextManager headroom/`.)
- `headroom/proxy/server.py` — remove ICM import and instantiation.
- `Cargo.toml` workspace dependencies — drop `fastembed`, `tantivy`, `ort` (if only used by relevance), and any other deps that become orphaned.

**Tests deleted:**
- All `tests/test_intelligent_context*.py`
- All `tests/test_rolling_window*.py`
- All `tests/test_progressive_summarizer*.py`
- All `tests/test_scoring*.py`
- All `tests/test_tool_crusher*.py`
- `crates/headroom-core/tests/scoring_*.rs`, `relevance_*.rs`, `context_*.rs` (other than safety)
- Parity comparator for `message_scorer` (PR #338/#343 work) — delete the comparator and the fixtures it consumed.
- Parity fixtures `tests/parity/fixtures/message_scorer/` (13 fixtures per Agent F report).

### Acceptance criteria

- `cargo build --workspace` green.
- `cargo test --workspace` green (after test deletions).
- `make ci-precheck` green.
- `pytest -x` green.
- Workspace builds **without** the fastembed cache dir.
- `git grep -i "IntelligentContextManager\|MessageScorer\|RollingWindow\|ProgressiveSummarizer\|ToolCrusher\|DropByScoreStrategy"` returns nothing in `crates/`, `headroom/`, `tests/` except comments referencing the deletion.

### Blocked by

PR-A1 (ICM call site must be removed first).

### Blocks

PR-B2 (live-zone block dispatcher fills the void).

### Rollback

`git revert`. ~10K LOC returns. Cache-killer bugs DO NOT return because Phase A PR-A1 already removed the call site (the deleted code is unreachable). Safe to revert.

### Notes

- **MessageScorer Rust port retirement:** PR #338 and #343 (April 2026) ported MessageScorer to Rust. That work becomes deletable here. Sunk cost stays sunk. The fixtures and the parity-harness scaffolding learnings carry forward to live-zone work.
- **`bge-small-en-v1.5` ONNX cache:** ~50 MB. Removing it is reversible (re-fetched on next fastembed init if anyone re-adds the dep). Document in CHANGELOG.
- **`anchor_selector.py`** — Agent G suspected it might be ICM-only. Check: `git grep AnchorSelector headroom/ crates/`. If only consumed by ICM/scoring/SmartCrusher, delete; if consumed by SmartCrusher's anchor logic, keep. Current Rust has `crates/headroom-core/src/transforms/anchor_selector.rs` which is consumed by SmartCrusher — keep that one, delete the Python one if ICM was its only caller.

---

## PR-B2 — Live-zone block dispatcher in Rust

**Branch:** `realign-B2-live-zone-dispatcher`
**Worktree:** `~/claude-projects/headroom-worktrees/realign-B2-live-zone-dispatcher`
**Risk:** **MEDIUM-HIGH** (new architecture; the central piece)
**LOC:** +800

### Scope
Build the new compressor: a function that takes an Anthropic `/v1/messages` body, identifies the live-zone blocks (latest user message tool_results, latest user message text, latest assistant tool_use is hot zone — exclude), and dispatches each to a type-aware compressor. Does NOT yet wire the type-aware compressors (PR-B3); does NOT yet validate tokens (PR-B4); does NOT yet inject CCR (PR-B7). PR-B2 lays the dispatching skeleton with no-op compressors; subsequent PRs fill them in.

### Files

**Add:**
- `crates/headroom-core/src/transforms/live_zone.rs` — the dispatcher. Public API:
  ```rust
  pub fn compress_live_zone(
      body_raw: &serde_json::value::RawValue,
      frozen_message_count: usize,
      auth_mode: AuthMode,
  ) -> Result<LiveZoneOutcome>;

  pub enum LiveZoneOutcome {
      NoChange,
      Modified { new_body: Box<serde_json::value::RawValue>, manifest: CompressionManifest },
  }
  ```
  Implementation skeleton:
  1. Parse `body` minimally (only `messages` field; leave the rest as `RawValue`).
  2. For each message at index `>= frozen_message_count`:
     - Identify if it's the latest user message (live zone candidate).
     - For each block in its content:
       - If block type is `tool_result`, dispatch to a no-op compressor (filled in PR-B3).
       - If block type is `text`, dispatch to text compressor (no-op for now).
       - Otherwise (image, etc.), no-op.
  3. Reassemble the modified `messages` array, preserving unmodified messages as `RawValue` byte-copies.
  4. Reassemble the body, preserving the original envelope as `RawValue` byte-copies; the only modified bytes are within the messages array.
  5. Return `Modified` only if any block was actually mutated; otherwise `NoChange` and the caller forwards original bytes.

**Modify:**
- `crates/headroom-proxy/src/compression/mod.rs` — add `pub mod live_zone_anthropic;` and route `/v1/messages` to it (replacing the no-op stub from PR-A1).
- `crates/headroom-proxy/src/compression/live_zone_anthropic.rs` (new) — calls `compress_live_zone` with `frozen_count = compute_frozen_count(parsed)` (from PR-A4).
- `crates/headroom-proxy/src/compression/anthropic.rs` — delete (replaced by `live_zone_anthropic.rs`).

**Tests added:**
- `crates/headroom-core/tests/live_zone_skeleton.rs::dispatches_only_to_latest_user_message`
- `crates/headroom-core/tests/live_zone_skeleton.rs::respects_frozen_message_count`
- `crates/headroom-core/tests/live_zone_skeleton.rs::no_change_when_no_block_mutated_returns_original`
- `crates/headroom-core/tests/live_zone_skeleton.rs::modified_messages_byte_equal_outside_block`
- `crates/headroom-core/tests/live_zone_skeleton.rs::system_and_tools_byte_equal_always`
- `crates/headroom-proxy/tests/integration_live_zone.rs::end_to_end_live_zone_passthrough`

### Acceptance criteria

- `cargo test -p headroom-core` green.
- `cargo test -p headroom-proxy` green.
- All Phase A SHA-256 tests still pass (live-zone with no-op compressors is byte-identical).

### Blocked by

PR-B1 (deletion); PR-A4 (cache_control / RawValue features).

### Blocks

PR-B3, PR-B4, PR-B7.

### Rollback

`git revert`. Compression returns to passthrough (the Phase A state); no functional regression.

### Notes

- The `RawValue`-based approach is the correctness mechanism: bytes outside modified blocks are byte-copies, not parse-then-reserialize.
- `AuthMode` parameter is unused in B2 (always `Payg` from B2's perspective); Phase F PR-F2 wires the gate.

---

## PR-B3 — Wire type-aware compressors into live-zone dispatcher

**Branch:** `realign-B3-wire-type-aware-compressors`
**Worktree:** `~/claude-projects/headroom-worktrees/realign-B3-wire-type-aware-compressors`
**Risk:** **MEDIUM** (existing compressors are battle-tested)
**LOC:** +600

### Scope
Wire `SmartCrusher`, `LogCompressor`, `SearchCompressor`, `DiffCompressor`, `CodeCompressor` into the dispatcher. Per-block content-type detection drives dispatch. No token validation yet (PR-B4); no CCR hardening yet (PR-B7).

### Files

**Modify:**
- `crates/headroom-core/src/transforms/live_zone.rs` — replace no-op compressors with real dispatch:
  ```rust
  fn compress_block(block: &mut Block, content_type: ContentType) -> Result<Option<CompressionResult>> {
      match content_type {
          ContentType::JsonArrayOfDicts => smart_crusher::crush(block),
          ContentType::Logs => log_compressor::compress(block),
          ContentType::SearchResults => search_compressor::compress(block),
          ContentType::Diff => diff_compressor::compress(block),
          ContentType::SourceCode => code_compressor::compress(block),
          ContentType::PlainText => Ok(None), // PR-B4 adds Kompress; for now, leave untouched
          ContentType::Image | ContentType::Unknown => Ok(None),
      }
  }
  ```
- `crates/headroom-core/src/transforms/content_detector.rs` — extend `ContentType` enum with the variants above. Use existing `Magika` + `unidiff-rs` + heuristic detectors.

**Tests added:**
- `crates/headroom-core/tests/live_zone_dispatch.rs::json_tool_result_routes_to_smart_crusher`
- `crates/headroom-core/tests/live_zone_dispatch.rs::log_tool_result_routes_to_log_compressor`
- `crates/headroom-core/tests/live_zone_dispatch.rs::diff_tool_result_routes_to_diff_compressor`
- `crates/headroom-core/tests/live_zone_dispatch.rs::source_code_tool_result_routes_to_code_compressor`
- `crates/headroom-core/tests/live_zone_dispatch.rs::unknown_content_type_no_op`

### Acceptance criteria

- All new tests pass.
- Existing SmartCrusher / LogCompressor / DiffCompressor / SearchCompressor tests still pass (their code is unchanged; only the caller is new).
- A representative `/v1/messages` request with a 50KB JSON tool_result through the proxy results in measurable compression (>2× size reduction) and SHA-256-equal envelope outside the compressed block.

### Blocked by

PR-B2.

### Blocks

PR-B4 (token validation gate), PR-B7 (CCR injection).

### Rollback

`git revert`. Live-zone goes back to no-op compressors.

---

## PR-B4 — Token validation gate with fallback; per-content-type byte thresholds

**Branch:** `realign-B4-token-validation-gate`
**Worktree:** `~/claude-projects/headroom-worktrees/realign-B4-token-validation-gate`
**Risk:** **LOW**
**LOC:** +250

### Scope
Eliminate P3-33 and P3-34. After every per-block compression, run the tokenizer over `original` and `compressed`. If `compressed.tokens >= original.tokens`, fall back to original. Add per-content-type byte thresholds: code>2KB, JSON>1KB, logs>500B, plain text>5KB. Below threshold → no compression attempted.

### Files

**Modify:**
- `crates/headroom-core/src/transforms/live_zone.rs` — wrap each compressor call with:
  ```rust
  let original_tokens = tokenizer.count(&original_bytes)?;
  let compressed_tokens = tokenizer.count(&compressed_bytes)?;
  if compressed_tokens >= original_tokens {
      metrics::compression_rejected_by_token_check(compressor_name);
      return Ok(None); // fall back to original
  }
  ```
- `crates/headroom-core/src/transforms/live_zone.rs::compress_block` — gate on byte threshold per content type:
  ```rust
  const THRESHOLDS: &[(ContentType, usize)] = &[
      (ContentType::SourceCode, 2048),
      (ContentType::JsonArrayOfDicts, 1024),
      (ContentType::Logs, 512),
      (ContentType::PlainText, 5120),
      (ContentType::Diff, 1024),
      (ContentType::SearchResults, 1024),
  ];
  if block.bytes_len() < threshold_for(content_type) {
      return Ok(None);
  }
  ```

**Tests added:**
- `crates/headroom-core/tests/live_zone_thresholds.rs::below_threshold_no_compression_attempted`
- `crates/headroom-core/tests/live_zone_thresholds.rs::above_threshold_compression_attempted`
- `crates/headroom-core/tests/live_zone_token_validation.rs::compressed_more_tokens_falls_back`
- `crates/headroom-core/tests/live_zone_token_validation.rs::compressed_fewer_tokens_accepted`
- Property test: `proptest! { fn live_zone_compression_token_count_non_increasing(blocks in arb_blocks_strategy()) { ... } }`

### Acceptance criteria

- All tests pass.
- A pathological input (already-minified JSON, dense base64) falls back to original instead of inflating tokens.
- Prometheus emits `compression_rejected_by_token_check_total{strategy=...}` counter.

### Blocked by

PR-B3.

### Blocks

PR-B6, PR-B7.

### Rollback

`git revert`. Token validation removed; bytes-only gate returns. Slight regression risk on pathological inputs.

---

## PR-B5 — TOIN observation-only refactor

**Branch:** `realign-B5-toin-observation-only`
**Worktree:** `~/claude-projects/headroom-worktrees/realign-B5-toin-observation-only`
**Risk:** **MEDIUM** (TOIN is a preserved primitive; refactor must maintain its learning value)
**LOC:** -300 / +400

### Scope
Eliminate P2-27 and P5-56. Strip TOIN's request-time hint API; keep the recording API. Recommendations published between deploys via a CLI tool that aggregates and writes a TOML file the compressor loads at startup. Per-tenant aggregation key extended to `(auth_mode, model_family, structure_hash)`.

### Files

**Modify:**
- `headroom/telemetry/toin.py:853-927` — remove `get_recommendation()` and `CompressionHint`. Replace with a no-op stub that returns `None`; deprecation warning in docstring.
- `headroom/telemetry/toin.py:103` — `Pattern` adds `auth_mode: str`, `model_family: str` fields.
- `headroom/telemetry/toin.py:477, 496, 727, 729, 1248, 1256` — change aggregation key from `sig_hash` to `(auth_mode, model_family, sig_hash)` tuple. Update all dict-key uses.
- `headroom/telemetry/toin.py:1596` — keep `tenant_prefix` for storage but document it's now redundant with the aggregation key.
- `headroom/transforms/smart_crusher.py:446` — remove the `get_recommendation()` call site. SmartCrusher is now deterministic; TOIN observes outcomes only.
- New CLI: `headroom/cli/toin_publish.py` — aggregates the on-disk TOIN store and produces `recommendations.toml`. Run as part of the deploy pipeline.
- New: `crates/headroom-core/src/transforms/recommendations.rs` — loads `recommendations.toml` at startup. Provides API like `recommendations::get(auth_mode, model, structure_hash) -> Option<Recommendation>`. Used to bias which compressor variants to try first (deterministic; no per-request mutation).

**Tests added:**
- `tests/test_toin_observation_only.py::test_no_request_time_hint_api_exposed`
- `tests/test_toin_observation_only.py::test_aggregation_key_includes_auth_mode_and_model`
- `tests/test_toin_observation_only.py::test_record_does_not_alter_compression_decision`
- `tests/test_toin_publish.py::test_publish_command_writes_toml`
- Determinism property test: `proptest! { fn compressor_deterministic_under_toin(input in arb_input()) { let r1 = compress(input); let r2 = compress(input); assert_eq!(r1, r2); } }`

### Acceptance criteria

- All new tests pass.
- TOIN's `record_compression` call sites still work (recording is kept).
- Removing TOIN's recommendations.toml at startup makes compression behave as if TOIN had never observed anything (graceful degrade).

### Blocked by

PR-B4.

### Blocks

PR-F3 (auth-mode aggregation key requires the TOIN refactor).

### Rollback

`git revert`. Per-request hint API returns; non-determinism returns.

### Notes

- This PR preserves TOIN per user direction: the learning value is intact; the dangerous request-time mutation is gone.

---

## PR-B6 — Memory subsystem refactor: live-zone tail injection only

**Branch:** `realign-B6-memory-live-zone-tail`
**Worktree:** `~/claude-projects/headroom-worktrees/realign-B6-memory-live-zone-tail`
**Risk:** **MEDIUM-HIGH** (touches memory feature semantics)
**LOC:** -400 / +300

### Scope
Eliminate P2-24. Memory retrieval moves out of the request lifecycle "auto-prepend" position. Two modes:
1. **Auto-tail mode** (default for now): retrieval runs at request entry; results appended to the latest user message tail (live zone). Same content always positions at the same place. Deterministic results for the same query.
2. **Tool mode** (preferred long-term): the model calls `memory_search` explicitly; retrieval runs in the tool execution path, not in the prompt-construction path. Memory is opt-in, not invisible.

This PR ships auto-tail-mode as default; tool-mode is wired but off-by-default.

### Files

**Modify:**
- `headroom/proxy/memory_handler.py:498-510` — delete `_inject_to_system_or_instructions`. Replace with `_append_to_latest_user_tail`.
- `headroom/proxy/handlers/openai.py:535-540` — same.
- `headroom/proxy/handlers/anthropic.py:1117-1135` — promote the existing `_append_context_to_latest_non_frozen_user_turn` to be the default path.
- `headroom/proxy/server.py:1050-1058` — already deleted in PR-A2; verify nothing reintroduces it.
- `headroom/proxy/memory_handler.py` — add `MemoryMode` enum: `AutoTail | Tool`. Default `AutoTail`. `Tool` mode skips auto-injection entirely.

**Tests added:**
- `tests/test_memory_auto_tail.py::test_memory_appears_in_latest_user_message_tail`
- `tests/test_memory_auto_tail.py::test_memory_does_not_modify_system_or_tools`
- `tests/test_memory_auto_tail.py::test_same_query_byte_identical_across_runs`
- `tests/test_memory_tool_mode.py::test_tool_mode_skips_auto_injection`

### Acceptance criteria

- All new tests pass.
- Existing memory feature tests pass (semantics preserved; position changes from system to user-tail).
- The bytes inserted are deterministic for the same query (no randomness in vector search results — verify or seed).

### Blocked by

PR-A2, PR-B4.

### Blocks

None (memory tool injection session-stickiness from PR-A7 stays).

### Rollback

`git revert`. Memory returns to auto-prepend.

---

## PR-B7 — CCR hardening: persistent backend + always-on tool registration

**Branch:** `realign-B7-ccr-hardening`
**Worktree:** `~/claude-projects/headroom-worktrees/realign-B7-ccr-hardening`
**Risk:** **MEDIUM**
**LOC:** -150 / +600

### Scope
Eliminate P2-25, P2-26. Two changes:
1. **Persistent CCR backend.** `CcrStore` trait gets a `SqliteCcrStore` impl (default) and a `RedisCcrStore` impl (opt-in for multi-worker). The in-memory store stays for tests. RUST_DEV.md "Multi-worker deployment — CCR fragmentation" section gets updated.
2. **`ccr_retrieve` tool always-on.** Once a session has performed any CCR compression, the tool is registered in `body["tools"]` for every subsequent request. The session ID derives from the existing `session_tracker_store` plumbing.

In Rust, the live-zone dispatcher writes `<<ccr:HASH>>` markers into the compressed block content (side-channel) and stores the original bytes in the configured backend.

### Files

**Add:**
- `crates/headroom-core/src/ccr/backends/sqlite.rs` — SQLite-backed `CcrStore`. Schema: `ccr_entries(hash TEXT PRIMARY KEY, original BLOB, created_at INTEGER, ttl_seconds INTEGER)`. Auto-purge on read (`WHERE created_at + ttl_seconds > now`).
- `crates/headroom-core/src/ccr/backends/redis.rs` — Redis-backed `CcrStore`. `SETEX hash ttl_seconds original`.
- `crates/headroom-core/src/ccr/backends/mod.rs` — `pub trait CcrStore` (already exists at `ccr.rs`); `pub fn from_config(config: &CcrConfig) -> Box<dyn CcrStore>`.

**Modify:**
- `crates/headroom-core/src/ccr.rs` — extract `InMemoryCcrStore` to its own file; rest stays.
- `crates/headroom-core/src/transforms/live_zone.rs` — when a compressor returns a `CompressionResult` with original bytes, store original bytes in CCR backend keyed by `BLAKE3(original_bytes)`. Append `<<ccr:HASH>>` marker to compressed block content.
- `headroom/proxy/handlers/anthropic.py` — `inject_ccr_retrieve_tool`: always add the tool when `session.has_done_ccr` is true; never toggle off.
- `headroom/proxy/handlers/openai.py` — same for OpenAI Chat / Responses.
- `headroom/ccr/tool_injection.py:302-328` — change `if has_compressed_content:` to `if session.has_done_ccr:`.
- `RUST_DEV.md` — update "Multi-worker deployment — CCR fragmentation" section: with `SqliteCcrStore` + sticky-session not required; with `RedisCcrStore` no stickiness needed at all.

**Tests added:**
- `crates/headroom-core/tests/ccr_backends.rs::sqlite_round_trip`
- `crates/headroom-core/tests/ccr_backends.rs::sqlite_ttl_purge`
- `crates/headroom-core/tests/ccr_backends.rs::redis_round_trip` (gated behind `cfg(feature = "redis")`)
- `crates/headroom-core/tests/ccr_backends.rs::backend_swap_byte_equal_keys`
- `tests/test_ccr_tool_always_on.py::test_tool_registered_on_every_request_after_first_ccr`
- `tests/test_ccr_tool_always_on.py::test_tool_not_registered_if_session_never_did_ccr`
- `tests/test_ccr_tool_always_on.py::test_tool_definition_byte_stable`

### Acceptance criteria

- All new tests pass.
- `RUST_DEV.md` reflects the new multi-worker story.
- A simulated proxy restart (kill + restart with `SqliteCcrStore`) can still resolve CCR markers from before the restart.
- Tool definition bytes are byte-stable (snapshot test pins them).

### Blocked by

PR-B2, PR-B3, PR-B4.

### Blocks

None.

### Rollback

`git revert`. In-memory-only CCR returns; tool-list flip returns. Operations stays — just less safe.

### Notes

- The `<<ccr:HASH>>` marker format is unchanged — existing markers from before this PR (in any cached prefix) still work.
- The session ID for "has done CCR" is the existing `session_id` from `session_tracker_store`; no new persistence needed.

---

## Phase B acceptance summary

After all 7 PRs land:

- ✅ ICM + RollingWindow + ProgressiveSummarizer + scoring + relevance + ToolCrusher deleted (~10K LOC retired)
- ✅ Live-zone block dispatcher operational
- ✅ Type-aware compressors wired (SmartCrusher, LogCompressor, SearchCompressor, DiffCompressor, CodeCompressor)
- ✅ Token validation gate with per-type byte thresholds and fallback
- ✅ TOIN observation-only with per-tenant aggregation key
- ✅ Memory routes to live-zone tail (no system mutation)
- ✅ CCR persistent backend + always-on tool registration
- ✅ MessageScorer Rust port (PR #338, #343) retired

**Phase B retires P0-4, P1-13, P2-18 through P2-27, P3-33, P3-34, P5-56, P6-70.**

After Phase B, Headroom's compression value is **back online** — and now it's correct.
