# 01 — Comprehensive Bug & Gap List

Ranked P0 (cache-killer) → P5 (long tail). Every entry has: title, file:line, evidence, guide §, fix, ROI estimate.

Sources: 10 parallel deep-audit subagents (Rust proxy passthrough; Rust compression correctness; Python proxy + bridges; prefix cache safety; streaming + wire-format; RTK + tests/parity; over-engineering; OpenAI long-tail + Bedrock; Headroom-side injections; auth-mode handling).

---

## P0 — Cache-killer smoking guns (every customer affected)

These bugs collapse Anthropic prompt-cache hit rate toward 0% for any traffic that triggers them. Fix in Phase A.

### P0-1. System prompt mutated by `.strip()` and memory-context append
- **File:** `headroom/proxy/server.py:1050-1058`; `headroom/proxy/handlers/openai.py:1212`
- **Evidence:** `body["system"] = (existing_system + "\n\n" + context).strip()` — strips whitespace and appends dynamic memory context to the cache hot zone on every memory-enabled call.
- **Guide:** §1.11 (whitespace fidelity), §6.3 #10 (tiny system prompt edits invalidate cache), §10.1 (system = always cache hot).
- **Fix:** Remove `_inject_system_context` path; route memory context to the **first block of the latest user message** (live zone). The existing `_append_context_to_latest_non_frozen_user_turn` already does this — make it the only path.
- **ROI:** Restores cache hits for ~all memory-enabled traffic.
- **Phase A → PR-A2.**

### P0-2. Every Python forwarder re-serializes JSON via `httpx ... json=body`
- **File:** `headroom/proxy/server.py:1088, 1090`; `headroom/proxy/handlers/streaming.py:651`; `headroom/proxy/handlers/openai.py:2392-2397`; `headroom/proxy/handlers/batch.py:344`
- **Evidence:** httpx default encoder calls `json.dumps(body, separators=(", ", ": "), ensure_ascii=True)`. Inbound bytes use `,`/`:` and raw UTF-8 in user content; outbound bytes use `, `/`: ` and `\uXXXX` escapes. Bytes never reach upstream byte-equal to bytes that arrived.
- **Guide:** §1.9 (the single most expensive proxy mistake), §1.10 (numeric precision), §1.11 (whitespace fidelity).
- **Fix:** Switch every forwarder to `httpx ... content=raw_bytes_modified_in_place`. Keep the original `await request.body()` bytes; if a transform mutated the body, re-serialize with `separators=(",", ":")` + `ensure_ascii=False`. Better: surgical byte-fragment replacement on `messages` only, leaving the envelope's bytes untouched.
- **ROI:** Restores cache hits for **all** Python-forwarded traffic.
- **Phase A → PR-A3.**

### P0-3. Rust proxy ignores customer `cache_control` markers
- **File:** `crates/headroom-proxy/src/compression/anthropic.rs:151-156`
- **Evidence:** `frozen_message_count: 0` hardcoded with `TODO: detect provider prefix-cached messages from the request. Until we wire that detection, we treat the whole list as droppable.` Combined with ICM, every compression event drops messages from index 0.
- **Guide:** §2.19 (up to 4 cache_control markers), §6.2 (cache breakpoints define the prefix).
- **Fix:** Walk `messages[*].content[*].cache_control`, `system[*].cache_control`, `tools[*].cache_control`; set `frozen_message_count` to the highest message index that contains a cache_control marker.
- **ROI:** Restores cache hits for **all** clients using Anthropic prompt caching (which is virtually all production Anthropic traffic).
- **Phase A → PR-A4.**

### P0-4. ICM compresses by dropping messages from cache hot zone (wrong scope)
- **File:** `crates/headroom-proxy/src/compression/anthropic.rs:146-157`; `crates/headroom-core/src/context/strategy/drop_by_score.rs:64-80`; `crates/headroom-core/src/context/manager.rs`; `headroom/transforms/intelligent_context.py:354-450`
- **Evidence:** ICM with default `keep_last_turns: 2` is allowed to drop any message older than the last two turns. Combined with P0-3, this is a 100%-likely cache-buster on any conversation with ≥3 user turns.
- **Guide:** §6.5 (live zone vs hot zone), §10.1 ("Old conversation turns ... never compress"), §6.3 #11 ("Truncation/summarization at the head"), §7.2 (append-only compression).
- **Fix:** Delete ICM. Replace with live-zone-only block-level compression. Phase B builds the replacement.
- **ROI:** Eliminates the largest single class of cache-bust events.
- **Phase A → PR-A1 (stop calling ICM); Phase B → PR-B1 (delete ICM).**

### P0-5. Numeric precision lost via `serde_json::Value` round-trip
- **File:** `crates/headroom-proxy/src/compression/anthropic.rs:91, 172`
- **Evidence:** Body parsed into `serde_json::Value` and re-serialized via `serde_json::to_vec(&parsed)`. `Value::Number` is `i64|u64|f64` so any `1.0` round-trips to `1`; large integers above 2^53 lose precision. `Cargo.toml:34` enables `preserve_order` only — no `arbitrary_precision`, no `RawValue`.
- **Guide:** §1.10.
- **Fix:** Add `arbitrary_precision` and `raw_value` features to `serde_json`. Use `&RawValue` for `messages[*]` so individual messages forward as exact byte copies. Strategy outputs only need to be "drop this index" or "replace this block's content."
- **ROI:** Closes the second-largest re-serialization byte-drift class.
- **Phase A → PR-A4 (jointly with P0-3).**

### P0-6. Memory tool injection toggles tools list and mutates `anthropic-beta`
- **File:** `headroom/proxy/memory_handler.py:389-398`; `headroom/proxy/handlers/anthropic.py:1147-1171`
- **Evidence:** Memory adds `memory_save`, `memory_search` tools to `body["tools"]` only when memory is enabled for the request. Mid-session config flicker → tool set changes → cache busts (§6.3 #2). Same code mutates `anthropic-beta` adding `context-management-2025-06-27` when injection happens (§6.3 #6).
- **Fix:** Make memory tool injection **session-sticky**: once injected, always inject for the lifetime of the session. Pin `anthropic-beta` order; never reorder tokens within the comma-list.
- **ROI:** Eliminates mid-session cache busts.
- **Phase A → PR-A6, PR-A7.**

### P0-7. `responses_converter.py` drops Codex `phase` field and corrupts multi-text-part rebuild
- **File:** `headroom/proxy/responses_converter.py:94, 221-256`
- **Evidence:** `phase` field is dropped on the Chat-Completions trip (line 94 maps `role` only); only `copy.copy(original)` accidentally retains it on the rebuild path. Multi-text-part input messages get corrupted: `_extract_text_from_parts` joins with `\n`, `_reconstruct_item:254-256` puts the concatenated text into the first part only and leaves parts 1..N as-is, doubling content.
- **Guide:** §4.5 (preserve `phase` exactly), §7.9 (position preservation).
- **Fix:** Stash `phase` and restore in `_reconstruct_item`. Rebuild text parts by index, replacing each part's text in place. Better: in Phase C, port `/v1/responses` to Rust and never decompose item structure for compression.
- **Phase A → PR-A8 (Python hotfix), Phase C → PR-C5 (full rebuild).**

---

## P1 — Wire-format / streaming corruption

### P1-8. SSE buffers decoded with `errors="ignore"` / `errors="replace"`
- **File:** `headroom/proxy/handlers/streaming.py:58, 772`; `headroom/ccr/response_handler.py:672`
- **Evidence:** `chunk.decode("utf-8", errors="ignore")` silently drops emoji/CJK bytes split across TCP reads. The wire passthrough at `streaming.py:788` is bytes (correct), but every `_parse_sse_usage_from_buffer` and `_parse_sse_to_response` decision is made on a string that may have lost bytes.
- **Guide:** §1.4 (UTF-8 multi-byte split across chunks).
- **Fix:** Bytes-level buffer; find `\n\n` boundary in bytes; decode each complete event after split.
- **Phase C → PR-C1 (Rust SSE parser); Phase A → PR-A8 includes a Python hotfix.**

### P1-9. SSE parser misses `thinking_delta`, `signature_delta`, `citations_delta`
- **File:** `headroom/proxy/handlers/streaming.py:213-298`
- **Evidence:** Only `text_delta` and `input_json_delta` are switched on (lines 268-271). Thinking blocks reconstructed without text or signature; signature-protected blocks rejected on replay.
- **Guide:** §2.5, §2.7, §5.1 transitions table.
- **Fix:** Add all delta-type arms. In Rust SSE parser (Phase C), implement guide §5.1 fully.
- **Phase A → PR-A8 (Python); Phase C → PR-C1 (Rust).**

### P1-10. Memory continuation re-emitter emits whole `partial_json` in one delta
- **File:** `headroom/proxy/handlers/streaming.py:300-391` (`_response_to_sse`)
- **Evidence:** `"partial_json": json.dumps(block["input"])` (line 366-374) emits the entire input JSON as a single delta — clients accumulating per-spec receive one giant fragment instead of an incremental stream. Tool IDs are fabricated as `f"toolu_{idx}"` (line 345). Thinking blocks dropped entirely.
- **Guide:** §2.6.
- **Fix:** Either delete this function (do memory continuation as non-streaming retry) or rewrite to spec.
- **Phase B → PR-B6 (memory injection refactor likely deletes it).**

### P1-11. LiteLLM bridge fabricates `toolu_<uuid>` when upstream `tc.id` missing
- **File:** `headroom/backends/litellm.py:860`
- **Evidence:** `tool_id = tc.id or f"toolu_{uuid.uuid4().hex[:24]}"`. If upstream omits `id` on chunk 1, fake ID is generated and the upstream tool_call_id is lost forever; next turn `tool_result` references the fake ID and pairing breaks.
- **Guide:** §3.5, §2.10.
- **Fix:** Drop the fallback; surface an error if `tc.id` is None on first appearance.
- **Phase D → PR-D1 deletes this whole file.**

### P1-12. OpenAI WS→HTTP fallback uses single-`\n` SSE split
- **File:** `headroom/proxy/handlers/openai.py:2422-2447`
- **Evidence:** `aiter_text()` decodes UTF-8 chunk-by-chunk → `buffer.split("\n", 1)` instead of `\n\n`. Multi-line `data:` payloads get wrong-split.
- **Fix:** Switch to `aiter_bytes()` + bytes-level `\n\n` boundary.
- **Phase C → PR-C3 ports this surface to Rust.**

### P1-13. Re-serialization in Rust path even when no body fields mutated
- **File:** `crates/headroom-proxy/src/compression/anthropic.rs:90-188`
- **Evidence:** When `should_apply` is true and ICM doesn't drop anything (`anthropic.rs:162-168`), the function correctly returns `NoCompression` and forwards original bytes. But the `Compressed` path always re-serializes via `serde_json::to_vec(&parsed)` — even if only one message changed, every retained message gets re-encoded through `Value`.
- **Guide:** §1.12.
- **Fix:** Use `RawValue` for retained `messages[*]` entries; only the modified message gets re-encoded.
- **Phase A → PR-A4 / Phase B → PR-B2 (live-zone replacement).**

### P1-14. Mid-stream `error` events not handled (Anthropic + OpenAI)
- **File:** `headroom/proxy/handlers/streaming.py:160-211, 213-298`
- **Evidence:** No `event_type == "error"` arm. The wire passthrough is byte-faithful (good) but Headroom's bookkeeping (`stream_state.input_tokens` etc.) silently doesn't reflect the failure; `_finalize_stream_response` reports a clean PERF line for an errored stream.
- **Guide:** §1.7, §2.21.
- **Fix:** Add `error` handling to telemetry.
- **Phase C → PR-C1 (Rust SSE).**

### P1-15. Connection drop without `message_stop`/`[DONE]` not surfaced
- **File:** `headroom/proxy/handlers/streaming.py:899` (`finally: await self._finalize_stream_response`)
- **Evidence:** The `finally` runs but there's no flag indicating the stream was truncated; logs report a PERF line as if it succeeded.
- **Guide:** §1.8.
- **Fix:** Track terminator-seen flag; emit truncation telemetry when missing.
- **Phase C → PR-C1.**

### P1-16. OpenAI `refusal` field on Chat assistant message not handled
- **File:** None (zero references)
- **Evidence:** Memory and tool-call extraction look only at `message.content` / `tool_calls`; refusal turns silently look like content==null with output_tokens=0.
- **Guide:** §3.7.
- **Fix:** Inspect `refusal` field; surface in telemetry.
- **Phase C → PR-C2 (Rust /v1/chat/completions).**

### P1-17. `current_block: Optional[dict]` instead of `blocks: HashMap<usize, BlockState>`
- **File:** `headroom/proxy/handlers/streaming.py:227, 700`
- **Evidence:** Anthropic emits one block at a time today, but the guide explicitly says "track blocks by `index`" — current code captures `index` and never uses it as a key.
- **Guide:** §2.4, §5.1.
- **Fix:** Index-keyed map.
- **Phase C → PR-C1.**

---

## P2 — Architectural over-build

### P2-18. ICM-as-history-dropper (the structural mismatch)
- **Files:** `headroom/transforms/intelligent_context.py`; `crates/headroom-core/src/context/manager.rs`; `crates/headroom-proxy/src/compression/icm.rs`
- **Status:** Delete in Phase B (PR-B1).

### P2-19. `RollingWindow`, `ProgressiveSummarizer` (head-truncation strategies)
- **Files:** `headroom/transforms/rolling_window.py` (395 LOC); `headroom/transforms/progressive_summarizer.py` (508 LOC)
- **Guide:** §6.3 #11, §6.4 (compaction is the explicit exception, intended to break cache once).
- **Status:** Delete in Phase B (PR-B1).

### P2-20. `MessageScorer`, `scoring/`, `relevance/` machinery
- **Files:** `crates/headroom-core/src/scoring/{scorer,score,weights,traits,mod}.rs` (~1500 LOC); `crates/headroom-core/src/relevance/{embedding,bm25,hybrid,base,mod}.rs` (~1600 LOC); `headroom/transforms/scoring.py` (459 LOC)
- **Evidence:** Sole consumer is `DropByScoreStrategy::try_fit`. Without ICM, no consumer.
- **Status:** Delete in Phase B (PR-B1). MessageScorer Rust port (PR #338, #343) becomes wasted work.

### P2-21. `crates/headroom-core/src/context/` — except `safety.rs`
- **Files:** `crates/headroom-core/src/context/{config,workspace,candidate,ccr_drop,manager,strategy/}.rs` (~1500 LOC)
- **Status:** Delete in Phase B (PR-B1). `safety.rs` (tool-pair atomicity) is moved to `crates/headroom-core/src/transforms/safety.rs` and kept.

### P2-22. `ToolCrusher` operates without `frozen_message_count`
- **File:** `headroom/transforms/tool_crusher.py:106`
- **Evidence:** Iterates all result_messages, no frozen check. Crushes any tool message above token threshold regardless of position.
- **Guide:** §10.1 (old tool results are cache-hot).
- **Status:** Delete in Phase B (PR-B1). ContentRouter covers the use case correctly.

### P2-23. `CacheAligner` rewrite path violates the very thing it claims to stabilize
- **File:** `headroom/transforms/cache_aligner.py:160-262`
- **Evidence:** Strips dynamic content from system prompt and re-inserts as a context block — mutates the cache hot zone. Currently `enabled=False` in `server.py:299`.
- **Guide:** §9.3.
- **Fix:** Delete the rewrite path (~400 LOC); keep detector + customer warning (~140 LOC).
- **Phase A → PR-A2 includes the deletion.**

### P2-24. Memory-handler injection at request lifecycle entry
- **File:** `headroom/proxy/memory_handler.py:498-510`; `headroom/proxy/handlers/openai.py:535-540`
- **Evidence:** Prepends a system message with retrieved memories on every turn. Retrieval is non-deterministic (vector store grows turn-to-turn).
- **Fix:** Move retrieval out of the request lifecycle; treat as an explicit customer-invoked tool.
- **Phase B → PR-B6 (memory refactor).**

### P2-25. CCR `ccr_retrieve` tool injected only when content was compressed
- **File:** `headroom/ccr/tool_injection.py:302-328`
- **Evidence:** `inject_tool_definition()` only adds the tool when `has_compressed_content` is true. Tool list size flips between requests.
- **Guide:** §6.3 #2 (tool list reordering).
- **Fix:** Inject `ccr_retrieve` on **every** request once a session has ever done CCR; or always inject for sessions that have CCR enabled.
- **Phase B → PR-B7.**

### P2-26. CCR markers computed but never injected into outgoing body in Rust path
- **File:** `crates/headroom-core/src/context/manager.rs:172-185`; `crates/headroom-proxy/src/proxy.rs:285`
- **Evidence:** `markers_inserted` is logged but never written into the body. The model is never told about dropped messages or about `ccr_retrieve`.
- **Guide:** §7.3 (reversibility).
- **Fix:** Once Phase B replaces ICM, CCR-on-live-zone-content writes the marker into the block content as a side-channel. Phase B PR-B7.

### P2-27. TOIN influences per-request decisions
- **File:** `headroom/telemetry/toin.py:853-927`
- **Evidence:** `get_recommendation()` consults pattern stats and returns hints that bias compression decisions; `pattern.observations += 1` mutates state during the call.
- **Guide:** §7.1, §11.17, §11.18.
- **Fix:** Strict observation-only. Recommendations published at deploy time, never altered request-time.
- **Phase B → PR-B5.**

---

## P3 — Missing infrastructure (Phase 3 cache stabilization)

### P3-28. No tool-array deterministic sort in Rust path
- **File:** Missing entirely in `crates/headroom-proxy/`
- **Evidence:** Python sorts at `handlers/anthropic.py:1198, 1217, 2041, 2118`; Rust does not.
- **Guide:** §8.5, §9.11.
- **Phase E → PR-E1.**

### P3-29. JSON Schema keys never sorted recursively
- **File:** None — `_sort_tools_deterministically` only sorts the tools array, not their `input_schema` contents.
- **Guide:** §8.5.
- **Phase E → PR-E2.**

### P3-30. No `prompt_cache_key` auto-injection
- **Evidence:** Zero references in the codebase.
- **Guide:** §4.17.
- **Phase E → PR-E4.**

### P3-31. No `cache_control` auto-placement (Anthropic)
- **Evidence:** `cache_control` only mentioned in stripping for hashing (`helpers.py:295-304`) and pass-through (`server.py:1053`).
- **Guide:** §2.19, §6.2.
- **Phase E → PR-E3.**

### P3-32. No volatile-content detector + warning
- **Evidence:** `cache_aligner` has detection but rewrites instead of warning.
- **Guide:** §9.3.
- **Phase E → PR-E5.**

### P3-33. No per-block token validation with fallback
- **Evidence:** Compression acceptance is `bytes_saved > 0` (`crates/headroom-core/src/transforms/pipeline/orchestrator.rs:158-165`); ICM aggregate-checks tokens (`anthropic.rs:162-168`) but per-block transforms don't.
- **Guide:** §7.5, §11.15, §11.20.
- **Phase B → PR-B4.**

### P3-34. No per-content-type byte thresholds
- **Evidence:** Threshold gating is by ratio (`bloat_threshold=0.5`) not by bytes (code>2KB, JSON>1KB, logs>500B, plain text>5KB per guide §7.6).
- **Phase B → PR-B4.**

### P3-35. No cache-bust drift detector telemetry
- **Evidence:** No prefix-hash drift detection across requests.
- **Phase E → PR-E6.**

### P3-36. No shared content-hash cache across customers (Phase 4)
- **Evidence:** `CompressionCache` is per-session, per-worker.
- **Status:** Out of scope for this realignment; queued for Phase 4 of the guide.

---

## P4 — OpenAI long-tail + Bedrock/Vertex

### P4-37. **Bedrock support is fake — lossy LiteLLM converter**
- **File:** `headroom/backends/litellm.py:486-628`
- **Evidence:** `_convert_messages_for_litellm` switch covers only `text` / `tool_use` / `tool_result`; drops `thinking`, `redacted_thinking`, `document`, `search_result`, `image`, `server_tool_use`, `mcp_tool_use`. Response converter hardcodes `"stop_sequence": None` (line 626) — §11.1 violation. Function-call arguments parsed and rewrapped (line 600) — string fidelity broken (§4.4).
- **Phase D → PR-D1, D2, D3 rebuild natively.**

### P4-38. Vertex same lossy converter
- **File:** Same — `headroom/backends/litellm.py`
- **Phase D → PR-D4 builds native Vertex.**

### P4-39. No native Bedrock/Vertex paths in Rust
- **File:** `crates/headroom-proxy/src/compression/mod.rs:50` only matches `/v1/messages`.
- **Phase D → PR-D1-D4.**

### P4-40. `/v1/conversations` blind spot (§4.14)
- **Evidence:** Zero references. Server-side prepended items invisible to Headroom; tokenizer count over-reports.
- **Phase C → PR-C4.**

### P4-41. `service_tier` never logged or surfaced
- **Guide:** §4.12.
- **Phase G → PR-G3 (observability).**

### P4-42. `incomplete`, `failed`, `cancelled` statuses never surfaced
- **Guide:** §4.10.
- **Phase C → PR-C3 / C4.**

### P4-43. `function_call.arguments` parsed-and-rewrapped in 2 places
- **File:** `headroom/backends/litellm.py:600`; `headroom/learn/plugins/codex.py:283`
- **Guide:** §4.4.
- **Phase D → PR-D1 deletes litellm.py; learn plugin moved to read-only.**

### P4-44. `phase` field "accidentally preserved" via `copy.copy(original)`
- **File:** `headroom/proxy/responses_converter.py:94, 235`
- **Status:** Already covered by P0-7. **Phase A → PR-A8** (hotfix), **Phase C → PR-C5** (full rebuild).

### P4-45. `image_generation_call` no log redaction
- **File:** `headroom/proxy/request_logger.py` — no base64/image redaction
- **Guide:** §11.6.
- **Phase G → PR-G3 includes a redaction step.**

### P4-46. `Cargo.toml` missing `arbitrary_precision` + `raw_value` features on `serde_json`
- **File:** `Cargo.toml:34`
- **Phase A → PR-A4 enables them.**

### P4-47. Apply patch V4A, local_shell_call argv, MCP items, compaction items only "accidentally" preserved
- **File:** `headroom/proxy/responses_converter.py:99` — "Unknown item type: preserve"
- **Evidence:** Survives only because the catch-all is conservative. No log line, no test. One refactor away from silent data loss.
- **Phase A → PR-A8 adds a warning log; Phase C → PR-C5 makes it explicit.**

### P4-48. No SSE parser in Rust at all
- **Status:** Phase 1 of the Rust proxy was passthrough; Phase C builds the parser.
- **Phase C → PR-C1.**

---

## P5 — Auth-mode + observability + fingerprinting

### P5-49. `X-Headroom-*` request headers leak upstream
- **File:** `headroom/proxy/handlers/anthropic.py:526` — `dict(request.headers.items())` captured unmodified, no strip step before `httpx.post(headers=headers)`.
- **Risk:** Subscription-revocation fingerprint.
- **Phase A → PR-A5.**

### P5-50. `anthropic-beta` mutated when memory enabled, not session-sticky
- **File:** `headroom/proxy/handlers/anthropic.py:1162-1168`
- **Status:** Already covered by P0-6. **Phase A → PR-A6, A7.**

### P5-51. `OpenAI-Beta` auto-injection on WS path
- **File:** `headroom/proxy/handlers/openai.py:1566-1567`
- **Risk:** OAuth scope rejection if scope doesn't grant the auto-injected beta.
- **Phase F → PR-F2 (gate by mode).**

### P5-52. `accept-encoding` stripped — fingerprint signal
- **File:** `handlers/anthropic.py:533`, `handlers/openai.py:264`
- **Risk:** Real Claude Code negotiates compression; stripping reveals the proxy.
- **Phase F → PR-F2 (preserve when subscription mode).**

### P5-53. `X-Forwarded-*` always added by Rust proxy
- **File:** `crates/headroom-proxy/src/headers.rs:103-117`
- **Phase F → PR-F4 (conditional on auth mode).**

### P5-54. Subscription tracker stores raw OAuth bearer token in process memory
- **File:** `headroom/subscription/tracker.py:166`
- **Risk:** Core dump or debugger attach exposes the token.
- **Phase F → PR-F3 hardens (hash + only the ID, not the token).**

### P5-55. Auth-mode never drives compression policy
- **Evidence:** Single policy applied to all three modes today.
- **Phase F → PR-F1 (`classify_auth_mode`), PR-F2 (gates).**

### P5-56. TOIN aggregates globally by `structure_hash` only
- **File:** `headroom/telemetry/toin.py:477, 496`
- **Risk:** Cross-tenant pattern leakage.
- **Phase F → PR-F3 changes key to `(auth_mode, model_family, structure_hash)`.**

### P5-57. Upstream `request-id` not captured in logs
- **File:** `crates/headroom-proxy/src/proxy.rs:355-358, 377-383`
- **Guide:** §11.10.
- **Phase A → PR-A8 (telemetry capture in Python); Phase C carries forward to Rust.**

### P5-58. Rate-limit headers forwarded but never observed
- **File:** `crates/headroom-proxy/src/headers.rs:126-139`
- **Guide:** §11.9.
- **Phase G → PR-G3 (Prometheus metric).**

### P5-59. Body size cap returns wrong status code (400 instead of 413)
- **File:** `crates/headroom-proxy/src/proxy.rs:243-263`
- **Phase A → PR-A8 fix, low priority.**

### P5-60. `tokens_saved_rtk` field is dead (allocated, never populated)
- **File:** `headroom/subscription/models.py:260`; `headroom/subscription/tracker.py:173`
- **Phase G → PR-G2.**

### P5-61. RTK never invoked from proxy (correct posture; document explicitly)
- **Status:** Per audit recommendation (Agent F): proxy-side invocation is wrong; cache hot zone risk + parallel impl with `log_compressor.rs`. Document explicitly so future contributors don't add it.
- **Phase G → PR-G1, G3.**

### P5-62. Wrap CLIs missing for cline, continue, goose, openhands, devin-style CLIs
- **Files:** `headroom/cli/wrap.py` — only Claude/Codex/Aider/Copilot/Cursor today.
- **Phase G → PR-G1.**

---

## P6 — Test-infra & parity

### P6-63. No SHA-256 byte-faithful round-trip test on recorded production payload
- **Phase A → PR-A8.**

### P6-64. `ccr`, `log_compressor`, `cache_aligner` parity comparators are `Skipped` stubs
- **File:** `crates/headroom-parity/src/lib.rs:172-174`
- **Phase I (parallel) — promote stubs to real comparators.**

### P6-65. `make test-parity` not a per-PR gate
- **File:** `.github/workflows/rust.yml:125-149` — nightly only, `continue-on-error: true`
- **Phase I — make per-PR; `Diff` fails build, `Skipped` allowed.**

### P6-66. No SSE corner-case fixtures (UTF-8 split, ping, all delta types, [DONE], mid-stream error)
- **Phase I — record fixtures during Phase C work.**

### P6-67. No real-traffic shadow test comparing Python vs Rust output byte-for-byte
- **Phase I — implement during Phase C.**

### P6-68. No per-session cache-hit-rate metric
- **File:** `headroom/proxy/prometheus_metrics.py` — only aggregate by provider
- **Phase G → PR-G3.**

### P6-69. No per-block compression-ratio histogram (only invocation count)
- **Phase G → PR-G3.**

### P6-70. No token-validation rejection counter
- **Phase B → PR-B4 emits the metric.**

### P6-71. WS-handshake `OpenAI-Beta` injection un-tested for OAuth-scope rejection paths
- **Phase I — record a fixture.**

### P6-72. Wrap E2E uses an `rtk` shim that just exits 0 (`e2e/wrap/run.py:250-267`) — doesn't exercise real RTK
- **Phase I — replace shim with a containerized real RTK or assert-on-shim-only-in-CI flag.**

---

## Summary table

| Priority | Count | Location |
|---|---:|---|
| P0 (cache-killer) | 7 | Phase A |
| P1 (wire-format) | 10 | Phase A + Phase C |
| P2 (over-build) | 10 | Phase B |
| P3 (missing Phase 3) | 9 | Phase E |
| P4 (long-tail + Bedrock) | 12 | Phase C + Phase D |
| P5 (auth + obs + fingerprint) | 14 | Phase F + Phase G |
| P6 (test infra) | 10 | Phase I (parallel) |
| **Total** | **72** | — |
