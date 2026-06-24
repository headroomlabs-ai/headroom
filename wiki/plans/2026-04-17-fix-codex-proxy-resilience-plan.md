---
title: "fix: Codex proxy resilience under reconnect storms"
type: fix
status: active
date: 2026-04-17
origin: wiki/plans/2026-04-17-codex-proxy-runtime-analysis.md
---

# fix: Codex proxy resilience under reconnect storms

## Overview

Harden the shared Headroom proxy so it can survive real multi-agent Codex traffic — especially the **large Anthropic `/v1/messages?beta=true` reconnect/retry storm** that hits the proxy immediately after a restart — without appearing dead (`/livez` timing out, new `/v1/responses` websocket handshakes hanging) and without bypassing compression.

This is the follow-on work to the runtime analysis captured in the origin document. The previous branch (`fix/responses-retries-keep-compression`) fixed the upstream WS handshake/fallback issues and kept compression enabled. This plan addresses the **remaining long-lived runtime degradation** described in §4 of the origin ("Long-lived service degradation on `8787`").

The plan deliberately focuses on **observability + lifecycle hygiene + cold-start backpressure**, not more blind patching. Compression stays enabled throughout.

## Problem Frame

Aged `8787` processes enter a state where:

- new `GET /livez` requests time out
- new `/v1/responses` opening handshakes time out
- existing established streams continue working
- the process is alive, listens on the port, and sampling shows heavy ONNX thread activity

Controlled reproductions ruled out the obvious single-factor causes (port, launchd, memory alone, idle socket count, compression-preserving changes, cold Kompress load). The surviving hypotheses (§"What Is Still Plausible" in origin) converge on:

1. **Long-lived real traffic leaves stuck websocket relay tasks / lifecycle bookkeeping leaks** (H1)
2. **Real Codex traffic, not synthetic traffic, triggers slow hidden work** (H2)
3. **ONNX/memory amplifies but does not solely cause the failure** (H3)
4. **Shared-proxy reconnect/retry storms after restart drive the proxy into this state** (H4 — confirmed by the "Latest Correction" in origin)

Without stage timings, active-task introspection, or session bookkeeping, the next iteration of debugging will again rely on `sample`, `lsof`, and guesswork. This plan fixes that first, then layers cold-start backpressure on top — in that order — so each subsequent bug hunt converges faster.

(see origin: `wiki/plans/2026-04-17-codex-proxy-runtime-analysis.md`)

## Requirements Trace

- **R1.** `/livez` remains responsive during and immediately after a restart that triggers large Anthropic replay traffic from active agent sessions. (origin §"Latest Correction", §"Updated upstream patch focus" item 4)
- **R2.** Cold-start heavy assets (Kompress ONNX, memory embedder, tokenizers, tree-sitter parsers) are loaded once at startup and *shared* between all provider pipelines; concurrent first-use callers wait on a single future, not N parallel loads. (origin §"Updated upstream patch focus" item 1)
- **R3.** The proxy provides enough runtime observability to prove — with data, not guesses — whether a future degradation is WS lifecycle starvation, memory/embedder contention, replay-storm amplification, or something else. (origin §"Priority 1: add instrumentation, not more blind patching")
- **R4.** Codex websocket relay tasks are explicitly tracked and deterministically cancelled when either side of the relay exits; a leaked relay task cannot hold the process alive past the client's disconnect. (origin §"Code Paths Most Relevant To The Remaining Bug" item 3; H1)
- **R5.** Cold-start compression + memory-context work on the Anthropic path is **bounded in concurrency** so that N simultaneous large replay requests cannot monopolize the event loop and thread pool. Compression stays enabled. (origin §"Updated upstream patch focus" item 2)
- **R6.** A reproducible harness exists for the real-agent reconnect/retry scenario so regressions can be caught locally instead of only in production. (origin §"Priority 2: reproduce degradation with real Codex traffic on a fresh process")
- **R7.** Existing fork behavior is preserved: compression is not bypassed on WS/streaming; upstream WS retry/open-timeout hardening is kept; WS→HTTP fallback normalization is kept; memory-context fail-open timeout is kept. (origin §"What Was Changed In The Fork", §"Kept locally")

## Scope Boundaries

- **Not** re-introducing any "skip compression" fast paths. Compression-preserving direction is non-negotiable.
- **Not** re-litigating the launchd setup bug (already fixed upstream in dotfiles; out of this repo).
- **Not** redesigning the memory stack, embedder choice, or Kompress model. Those are upstream concerns.
- **Not** building a full distributed tracing system. The instrumentation added here is structured logs + in-process counters; OpenTelemetry hookup is a separate plan.
- **Not** changing the WS→HTTP fallback semantics beyond what's needed to plumb through the new request-id/session-id logging fields.
- **Not** modifying `headroom-ai[ml]` dependencies, HuggingFace model IDs, or the embedder's own backend.

### Deferred to Separate Tasks

- **OpenTelemetry / metrics exporter wiring**: the counters added here expose `prometheus_metrics` entries; OTLP export belongs in a follow-up.
- **External watchdog in the LaunchAgent**: a plist-level `WatchPaths`/`ThrottleInterval` revision belongs in the `.dotfiles` repo (see origin §"Important External Files"), not here. This plan only adds the in-process signal the watchdog would consume (§Unit 5).
- **Quantifying multi-agent reconnect budget**: tuning the `Unit 4` semaphore default via load testing is work for after merge.
- **Dedicated Codex WS-side cap**: `/v1/responses` remains intentionally ungated by the Anthropic HTTP semaphore. The current plan surfaces `active_relay_tasks` on `/readyz` so operators can see WS pressure early; add a separate WS-side cap only if those counters show the threat model has shifted.

## Context & Research

### Relevant Code and Patterns

- `headroom/proxy/handlers/openai.py:1206` — `handle_openai_responses_ws` (Codex WS entry point, 600+ lines)
- `headroom/proxy/handlers/openai.py:1559-1767` — upstream WS connect retry loop, `open_timeout` handling
- `headroom/proxy/handlers/openai.py:1815` — `_ws_http_fallback` (preserved as-is)
- `headroom/proxy/handlers/openai.py:1439-1452` — memory context timeout fail-open (preserved as-is)
- `headroom/proxy/handlers/anthropic.py:293` — `handle_anthropic_messages` (HTTP entry point, also covers the `?beta=true` replay traffic)
- `headroom/proxy/server.py:586-733` — `HeadroomProxy.startup` (where eager preload runs)
- `headroom/proxy/server.py:634-637` — current eager preload iterates **only** `anthropic_pipeline.transforms` and breaks on first match
- `headroom/proxy/server.py:301-308` — both pipelines currently share the same `transforms` list (so the module-level `_kompress_cache` is de facto shared, but this is fragile)
- `headroom/proxy/server.py:1190-1212` — `/livez`, `/readyz`, `/health` handlers (trivial JSON, no I/O)
- `headroom/proxy/memory_handler.py:134-207` — `MemoryHandler._ensure_initialized` (lazy, no `asyncio.Lock`)
- `headroom/transforms/content_router.py:1221-1297` — `eager_load_compressors` (Kompress + Magika + Code-Aware + SmartCrusher)
- `headroom/transforms/kompress_compressor.py:163-221` — `_load_kompress_onnx` / `_load_kompress` with module-level `_kompress_cache` + `threading.Lock`
- `headroom/proxy/request_logger.py` — existing structured request log sink (extend, don't replace)
- `headroom/proxy/prometheus_metrics.py` — existing `PrometheusMetrics` class; add counters/gauges here
- `headroom/proxy/helpers.py:138-207` — `_read_request_json` (pre-upstream work on Anthropic path)

### Institutional Learnings

- `docs/solutions/` does not exist in this repo. The `wiki/plans/` folder holds design-style documents; no rolling solutions log to mine.
- **Prior fork learning** (origin §"What Was Changed In The Fork"): keep compression on, retry upstream WS, normalize fallback body, wrap memory-context lookup in a timeout. These are invariants — Unit 2 and Unit 4 must not regress them.
- **Prior rollback learning** (origin §"Rolled back locally"): latency-first skips of compression / memory injection were rolled back. Any new "fast path" must not recreate that shape.

### External References

None used for this plan. Python `asyncio` lock, `asyncio.Semaphore`, `asyncio.Task`, and `asyncio.all_tasks()` semantics are sufficient; no framework-specific research needed. External research was intentionally skipped (§1.2: strong local patterns, team knows the area).

## Key Technical Decisions

- **Observe before mitigating.** Units 2 and 3 (instrumentation + lifecycle accounting) land before Unit 4 (backpressure) so the backpressure defaults can be tuned from real data instead of guessed. The origin explicitly calls this out as Priority 1.
  - *Rationale:* the last round of patches was driven by symptoms; the next one should be driven by timings.
- **Share cold-start state across pipelines explicitly, not by accident.** Currently both `TransformPipeline` instances share the same `transforms` list by coincidence of construction (`server.py:301-308`). Unit 1 moves the preload out of "first matching transform on the Anthropic pipeline" into a startup-level orchestration step that holds references it can reuse across both pipelines, and adds a single `asyncio.Lock` around first-use paths so concurrent requests land on the same future.
  - *Rationale:* origin §"Updated upstream patch focus" item 1: "make eager preload and request-time use share the same in-process singleton/cache".
- **Track WS sessions in a registry, not via ad-hoc `logger.info`.** A `WebSocketSessionRegistry` makes active-count a first-class observable and lets `/debug/ws-sessions` return something useful.
- **Explicit relay-task cancellation** replaces `asyncio.gather(..., return_exceptions=True)` for the two relay halves. When one side exits, the other is cancelled deterministically — no wait on TCP timeout, no task leak (addresses H1).
- **Bounded pre-upstream concurrency on the Anthropic path, not on all paths.** The Codex WS path already serializes naturally (one client → one upstream WS). The Anthropic HTTP path is where replay storms arrive. Limiting concurrency only where the problem actually exists keeps the blast radius tight.
  - *Rationale:* origin §"Hypothesis 4" + §"Updated upstream patch focus" item 2.
- **Debug endpoints are loopback-only, always.** No config flag, no auth header — a remote IP gets a 404, period. This sidesteps "did someone accidentally expose task state?" as a concern.
- **Frontmatter, not freeform.** Unlike the existing `wiki/plans/` files this plan uses YAML frontmatter (`status: active`, `origin:`, etc.) to participate in the `ce:plan` deepening + search flow.

## Open Questions

### Resolved During Planning

- *Q: Target upstream or the fork?* Resolved: target the fork (`fix/responses-retries-keep-compression`), structured so each unit is cherry-pickable into a PR against `chopratejas/headroom#172`.
- *Q: Should Unit 5's debug endpoints require an explicit flag to enable?* Resolved: **no** — loopback-only gating is sufficient and the debug data is useless if it isn't always available when the process is struggling.
- *Q: Does `MemoryHandler._ensure_initialized` already have a concurrency guard?* Resolved: **no**. It relies on `self._initialized = True` flip, which is not atomic across `await` points. Unit 1 adds an `asyncio.Lock`.
- *Q: Are the WS relay tasks already cancelled on partner exit?* Resolved: **no**. `asyncio.gather(return_exceptions=True)` waits for both; the survivor only exits when its own loop raises. Unit 3 fixes this.
- *Q: Is the existing Kompress singleflight sufficient?* Resolved: partially. The `threading.Lock` serializes same-model-id loads, but holds during `hf_hub_download` (network I/O). Unit 1 supplements with an `asyncio.Lock` at the request-handler layer so async callers don't each spawn the thread-pool job.

### Deferred to Implementation

- **Exact default for the Anthropic pre-upstream semaphore.** Start at `max(2, min(8, os.cpu_count()))` and expose via `--anthropic-pre-upstream-concurrency` / `HEADROOM_ANTHROPIC_PRE_UPSTREAM_CONCURRENCY`. Tune after Unit 2's stage timings land.
- **Which asyncio task names are "long-lived" thresholds** for Unit 5's dump. Likely `> 2× median lifetime`, but the median is only observable after Unit 2 runs in production.
- **Whether the repro harness (Unit 6) needs to speak realistic Codex subprotocol framing or can synthesize enough with recorded frames.** Depends on how faithfully `tests/test_openai_codex_routing.py` fixtures already capture the handshake.
- **Final placement of `WebSocketSessionRegistry`**: `headroom/proxy/ws_session_registry.py` as a new module, or folded into `headroom/proxy/server.py`. Likely new module; confirm during implementation once imports are real.

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not implementation specification. The implementing agent should treat it as context, not code to reproduce.*

```
                    ┌───────────────────────────────────────────────────────┐
                    │                   HeadroomProxy.startup                │
                    │                                                        │
                    │   Unit 1: shared_warmup()                              │
                    │     • eager-load on both pipelines (not just first)    │
                    │     • Kompress, Magika, Code-Aware, SmartCrusher        │
                    │     • memory backend + embedder                         │
                    │     • populates WarmupRegistry singletons               │
                    └───────────────┬────────────────────┬──────────────────┘
                                    │                    │
                                    ▼                    ▼
              ┌─────────────────────────────┐   ┌─────────────────────────────┐
              │   Codex WS path             │   │   Anthropic HTTP path       │
              │   /v1/responses             │   │   /v1/messages?beta=true    │
              │                             │   │                             │
              │ Unit 3: session registry    │   │ Unit 4: pre-upstream        │
              │   • register on accept      │   │         Semaphore(N)        │
              │   • cancel partner on exit  │   │   • guards _read_request    │
              │   • deregister in finally   │   │   • deep-copy                │
              │                             │   │   • first compression stage │
              │ Unit 2: stage timings       │   │   • memory-context lookup   │
              │   accept → first_frame      │   │                             │
              │   → upstream_connect        │   │ Unit 2: stage timings       │
              │   → upstream_first_event    │   │   read_json → deep_copy →   │
              │   → total_session_ms        │   │   compress → memory →       │
              └─────────────────────────────┘   │   upstream_first_byte       │
                                                └─────────────────────────────┘
                                    │                    │
                                    └────────┬───────────┘
                                             ▼
                      ┌─────────────────────────────────────────┐
                      │   Unit 5: /debug/* (loopback-only)       │
                      │     • /debug/tasks                        │
                      │     • /debug/ws-sessions                  │
                      │     • /debug/warmup                       │
                      └─────────────────────────────────────────┘
                                             │
                                             ▼
                      ┌─────────────────────────────────────────┐
                      │   Unit 6: scripts/repro_codex_replay.py  │
                      │     • spawn N concurrent Codex WS         │
                      │     • fire Anthropic replay-shaped POSTs  │
                      │     • assert /livez stays < 100ms          │
                      └─────────────────────────────────────────┘
```

Decision matrix for "where does this request wait?":

| Path                          | Pre-upstream gate    | Relay lifecycle gate | Memory lookup timeout |
|-------------------------------|----------------------|----------------------|------------------------|
| `/v1/responses` (Codex WS)    | *(none — natural 1:1)* | `WSSessionRegistry`  | existing `wait_for`   |
| `/v1/messages` (Anthropic)    | `Semaphore(N)` (new) | *(HTTP-single-shot)* | existing `wait_for`   |
| `/v1/chat/completions` (OpenAI) | *(unchanged)*        | *(unchanged)*        | *(unchanged)*         |
| `/livez`, `/readyz`           | *(none — must be free)* | *(n/a)*             | *(n/a)*                |

## Implementation Units

- [ ] **Unit 1: Shared cold-start warmup + async singleflight for memory init**

**Goal:** Make preload truthful and shared across all provider pipelines; ensure the first wave of concurrent requests after restart never kick duplicate expensive loads.

**Requirements:** R2, R7

**Dependencies:** None (pure startup change; no other unit depends on this landing first, but landing it first reduces noise in Unit 2's timings)

**Files:**
- Modify: `headroom/proxy/server.py` (startup orchestration around lines 627-675)
- Modify: `headroom/proxy/memory_handler.py` (add `asyncio.Lock` in `_ensure_initialized`)
- Modify: `headroom/transforms/content_router.py` (no behavior change — ensure `eager_load_compressors` is idempotent when called via both pipelines)
- Create: `headroom/proxy/warmup.py` (new `WarmupRegistry` holding preloaded handles)
- Test: `tests/test_proxy_warmup.py` (new)
- Test: `tests/test_memory_handler_concurrent_init.py` (new)

**Approach:**
- Introduce `WarmupRegistry` with typed slots for kompress, magika, code-aware, smart-crusher, memory-embedder, memory-backend. Populate during `HeadroomProxy.startup`, expose via `proxy.warmup` for `/debug/warmup` (Unit 5) and `/readyz`.
- Iterate **both** `self.anthropic_pipeline.transforms` and `self.openai_pipeline.transforms` when calling `eager_load_compressors`; dedupe by `id(transform)` so shared transforms don't double-load.
- Preload the memory embedder explicitly — today `ensure_initialized` initializes the backend but the embedder's first-use cost may still be deferred until the first `search_and_format_context`. Force one warm-up encode (a single short string) to pre-compile the ONNX graph.
- Add `asyncio.Lock` in `MemoryHandler._ensure_initialized` so concurrent first callers await one load, not N.
- Replace the "break after first matching transform" loop with explicit orchestration. Log whether the warmup was a no-op (already loaded) vs. a fresh load.

**Patterns to follow:**
- Existing `eager_status` dict shape in `server.py:630-666`.
- `threading.Lock` module-level pattern in `kompress_compressor.py` (for the sync side); supplement with `asyncio.Lock` in memory_handler (async side).
- `request_logger`'s structured-log pattern for startup events.

**Test scenarios:**
- Happy path: starting the proxy with `optimize=True` logs one preload event per component and `WarmupRegistry` reports all slots loaded.
- Happy path: starting with `optimize=False` populates `WarmupRegistry` on first request instead (lazy path still works).
- Edge case: `optimize=True` with `enable_kompress=False` — `WarmupRegistry.kompress` is `None`, no error logged.
- Edge case: both pipelines share the same router transform — `eager_load_compressors` runs once, not twice.
- Integration: `MemoryHandler._ensure_initialized` called 10 concurrent times from different tasks — only one backend init runs (assert via counter on `LocalBackend.__init__` hit count).
- Integration: embedder warm-up encode is issued at startup — verify via mock that `embed_text("warmup")` (or equivalent) was called once during startup, not lazily on first request.
- Error path: memory backend init raises — startup still completes, `WarmupRegistry.memory_backend` is `None`, health reports degraded memory.

**Verification:**
- Startup logs show each component preloaded exactly once, with timings.
- `/readyz` reports all configured subsystems as `initialized=true` before accepting traffic.
- Concurrent first-request simulation (`asyncio.gather` 20 requests) triggers only one Kompress load in logs.

---

- [ ] **Unit 2: Stage-timing instrumentation on Codex WS and Anthropic HTTP paths**

**Goal:** Emit structured per-request timings for every stage that plausibly contributes to cold-start or degradation latency, so the next debugging session starts from data.

**Requirements:** R3

**Dependencies:** None (independent), but landing alongside Unit 1 gives early visibility into whether warmup actually helped.

**Files:**
- Modify: `headroom/proxy/handlers/openai.py` (WS path at lines 1206+; HTTP path at 798+)
- Modify: `headroom/proxy/handlers/anthropic.py` (`handle_anthropic_messages` around line 293 and downstream)
- Modify: `headroom/proxy/request_logger.py` (extend schema)
- Modify: `headroom/proxy/prometheus_metrics.py` (add histograms per stage)
- Modify: `headroom/proxy/helpers.py` (thread `stage_timer` through `_read_request_json` / compression helpers)
- Create: `headroom/proxy/stage_timer.py` (small context-manager util)
- Test: `tests/test_stage_timer.py` (new)
- Test: `tests/test_openai_codex_ws_timings.py` (new)
- Test: `tests/test_anthropic_stage_timings.py` (new)

**Approach:**
- Add a `StageTimer` context manager with `stage_timer.measure("memory_context")` support; emits structured log on exit. Unified across handlers.
- For Codex WS: instrument `accept`, `first_client_frame`, `upstream_connect`, `upstream_first_event`, `memory_context`, `compression`, `total_session`. Log on session close with all fields.
- For Anthropic HTTP: instrument `read_request_json`, `deep_copy`, `compression_first_stage`, `memory_context`, `upstream_connect`, `upstream_first_byte`, `total_pre_upstream`. Log after first upstream byte arrives.
- Also export each stage as a Prometheus histogram — the existing `PrometheusMetrics` class has the right pattern; mirror it.
- `request_id` is already threaded through both handlers. Add a `session_id` (UUID generated at WS accept / HTTP request start) so multi-turn sessions are correlatable.

**Execution note:** Land the util + test (`StageTimer`) first; only then plumb it through the two handlers. Keeps the diff reviewable.

**Patterns to follow:**
- `headroom/proxy/request_logger.py` structured-field schema.
- `PrometheusMetrics.record_request` histogram pattern.
- Existing `request_id = await self._next_request_id()` at `openai.py:1232`.

**Test scenarios:**
- Happy path: a full Codex WS session (accept → response.completed) emits one structured log line with all 7 stage fields populated and a positive `total_session` > 0.
- Happy path: a full Anthropic HTTP request emits one log line with all pre-upstream stage fields populated.
- Edge case: a session that exits during upstream connect (no `first_event`) logs `upstream_first_event=null` without raising.
- Edge case: `request_id` and `session_id` appear together on every log line.
- Error path: a timeout in `memory_context` logs `memory_context` duration = the timeout value, not `null` (prove the timer captures the failure window).
- Integration: Prometheus histograms emit non-zero observations after a real request round-trip.

**Verification:**
- Every `/v1/responses` WS session and every `/v1/messages` request produces exactly one timing log line.
- `/metrics` endpoint shows the new histogram series.
- No existing tests regress; the request_logger schema additions are backward-compatible (new fields only).

---

- [x] **Unit 3: WebSocket session registry + deterministic relay-task cancellation**

**Goal:** Eliminate the "aged process has leaked relay tasks" hypothesis by making every WS session explicitly tracked and both relay tasks deterministically cancelled when either exits.

**Requirements:** R4

**Dependencies:** Unit 2 (uses the same `session_id`)

**Files:**
- Create: `headroom/proxy/ws_session_registry.py` (new)
- Modify: `headroom/proxy/handlers/openai.py` (`handle_openai_responses_ws` at 1206; relay task construction around 1559-1767; `_upstream_to_client` at 1565)
- Modify: `headroom/proxy/prometheus_metrics.py` (add `active_ws_sessions` gauge, `active_relay_tasks` gauge, `ws_session_duration` histogram)
- Test: `tests/test_ws_session_registry.py` (new)
- Test: `tests/test_openai_codex_ws_lifecycle.py` (new)

**Approach:**
- `WebSocketSessionRegistry` holds `dict[session_id, WSSessionHandle]` where handle tracks: `started_at`, `client_addr`, `upstream_url`, `relay_tasks: list[asyncio.Task]`, `last_activity_at`.
- Register on `websocket.accept()` success; deregister in a `try/finally` around the whole handler body.
- Replace `asyncio.gather(_client_to_upstream(), _upstream_to_client(), return_exceptions=True)` with explicit `asyncio.create_task(...)` for each, then `asyncio.wait(..., return_when=FIRST_COMPLETED)`. Cancel the other task explicitly, then `await` the cancelled task's `.cancelled()` settlement (suppress `CancelledError`).
- Emit one structured log line per session termination with cause (`client_disconnect` / `upstream_disconnect` / `upstream_error` / `client_error`).

**Execution note:** Start from a failing integration test that asserts "after a client disconnects mid-stream, the upstream relay task is `done()` within 100ms and the registry is empty". Then make it pass.

**Patterns to follow:**
- Existing `_client_to_upstream()` and `_upstream_to_client()` structure in `openai.py:1559-1767`.
- `contextlib.suppress(asyncio.CancelledError)` pattern.
- `PrometheusMetrics` gauge update pattern.

**Test scenarios:**
- Happy path: a normal session completes; registry is empty after `response.completed`; both relay tasks `.done()` is True.
- Edge case: client disconnects mid-stream; upstream relay task is cancelled within 100ms; registry no longer contains the session.
- Edge case: upstream closes first; client-side relay task is cancelled within 100ms; client receives a clean close frame.
- Edge case: 50 concurrent WS sessions open and close; `active_ws_sessions` gauge rises to 50 then returns to 0; no tasks remain in `asyncio.all_tasks()` with the relay task name pattern.
- Error path: upstream connect fails before relay tasks are created; registry is deregistered cleanly (no leak from the handshake phase).
- Error path: `_upstream_to_client` raises mid-stream; client task is cancelled; registry reports `termination_cause=upstream_error`.
- Integration: the `/debug/ws-sessions` endpoint (Unit 5) returns live data consistent with the registry under load.

**Verification:**
- After a torture-test of 100 rapid connect/disconnect cycles, `asyncio.all_tasks()` count returns to baseline within 1s.
- `active_ws_sessions` gauge returns to 0 after all sessions close.
- No `RuntimeWarning: coroutine was never awaited` in logs.

---

- [ ] **Unit 4: Bounded pre-upstream concurrency for Anthropic replay storms**

**Goal:** Prevent the cold-start replay storm from occupying every event-loop slot and thread-pool worker with deep-copy / compression / memory-context work before upstream receives the request.

**Requirements:** R1, R5, R7

**Dependencies:** Unit 2 (must have stage timings landed so the default semaphore size can be tuned from real numbers; the unit itself can ship with a conservative default and be retuned later)

**Files:**
- Modify: `headroom/proxy/handlers/anthropic.py` (wrap the pre-upstream phases in `handle_anthropic_messages` at ~293 through first upstream call)
- Modify: `headroom/proxy/models.py` or config module (add `anthropic_pre_upstream_concurrency: int` config field)
- Modify: `headroom/proxy/server.py` (construct the `asyncio.Semaphore` on `HeadroomProxy.__init__` so it's per-process, not per-request)
- Modify: CLI surface (add `--anthropic-pre-upstream-concurrency` flag + env var)
- Test: `tests/test_anthropic_pre_upstream_backpressure.py` (new)

**Approach:**
- Add `self.anthropic_pre_upstream_sem = asyncio.Semaphore(config.anthropic_pre_upstream_concurrency or max(2, min(8, os.cpu_count())))` in `HeadroomProxy.__init__`.
- In `handle_anthropic_messages`, wrap the region from `_read_request_json` through the first `self.http_client.send()` / `stream()` call in `async with self.anthropic_pre_upstream_sem:`.
- **Critically:** `/livez` and `/readyz` do not go through this semaphore. They remain free even under replay storm.
- Emit a log (not a warning — expected under load) when a request waits > 100ms for the semaphore, so we can see queueing in the Unit 2 stage timings.
- Default config: `HEADROOM_ANTHROPIC_PRE_UPSTREAM_CONCURRENCY` env var honored; CLI flag overrides; unset defaults to the computed floor.

**Patterns to follow:**
- Existing config field addition pattern in `headroom/proxy/models.py`.
- Existing CLI flag plumbing in `headroom/cli/`.
- `request_logger.info(..., stage="pre_upstream_wait_ms", ...)` field extension from Unit 2.

**Test scenarios:**
- Happy path: a single request passes through with negligible `pre_upstream_wait_ms`.
- Edge case: N+1 concurrent requests (where N = configured concurrency) — exactly one waits; all complete; `pre_upstream_wait_ms > 0` only on the waiter.
- Edge case: `anthropic_pre_upstream_concurrency=1` serializes two concurrent requests deterministically (useful for tests).
- Error path: a request that raises inside the critical section releases the semaphore (verify via counter that `_value` returns to baseline).
- Integration: under a synthetic storm of 20 concurrent large POSTs (simulating the retry replay), `/livez` response time stays under 100ms (p99 from Unit 2 timings).
- Integration: compression is *not* bypassed — assert a known large body still produces a compressed upstream request (no regression of R7).

**Verification:**
- Under `scripts/repro_codex_replay.py` (Unit 6), `/livez` p99 stays under 100ms during the storm phase.
- Counter-factual: setting concurrency to `10000` (effectively unbounded) reproduces the original starvation in the harness.

---

- [ ] **Unit 5: Loopback-only debug introspection endpoints**

**Goal:** Make "what is this process doing right now?" a single `curl` away when degradation happens, instead of `sample` + `lsof` + guesswork.

**Requirements:** R3

**Dependencies:** Unit 3 (ws session data), Unit 1 (warmup registry data)

**Files:**
- Modify: `headroom/proxy/server.py` (add `/debug/tasks`, `/debug/ws-sessions`, `/debug/warmup` routes near the existing `/livez` at 1190)
- Create: `headroom/proxy/debug_introspection.py` (pure functions that serialize state)
- Create: `headroom/proxy/loopback_guard.py` (middleware / dep that 404s non-loopback requests)
- Test: `tests/test_proxy_debug_endpoints.py` (new)

**Approach:**
- `loopback_guard`: inspect `request.client.host`; if not in `{"127.0.0.1", "::1", "localhost"}`, return 404 with no body. A 404 (not 403) keeps debug endpoints invisible to external scanners.
- `/debug/tasks` returns `asyncio.all_tasks()` with: name, coro name, age, current-line (via `get_coro().cr_frame.f_code.co_qualname`), stack depth. Sort by age desc.
- `/debug/ws-sessions` returns the registry dump from Unit 3.
- `/debug/warmup` returns `WarmupRegistry` state from Unit 1 + whether each slot is `loaded` / `loading` / `null`.
- All three return JSON. None mutate state. None block.

**Patterns to follow:**
- Existing `/readyz` handler shape in `server.py:1204-1207`.
- Loopback detection: `request.client.host` is already FastAPI's standard; mirror the guard style used elsewhere in the repo if any; otherwise a small dependency function.

**Test scenarios:**
- Happy path: `curl 127.0.0.1:8787/debug/tasks` returns 200 with a JSON array; each entry has `name`, `age_seconds`, `coro`.
- Edge case: `/debug/tasks` called during a live WS session lists the relay tasks with non-zero age.
- Edge case: loopback guard returns 404 (not 403) for a simulated non-loopback client.
- Edge case: `/debug/warmup` reports `memory_backend=loaded` after Unit 1's startup completes; reports `loading` if called during startup (race window).
- Error path: `asyncio.all_tasks()` raising (hypothetical) — handler returns 500 with structured error, doesn't crash the server.
- Integration: `/debug/ws-sessions` output is consistent with the session count gauge from Unit 3 during a load test.

**Verification:**
- Manual: from a remote IP, all three endpoints return 404.
- Manual: during the Unit 6 harness, `/debug/tasks` shows relay tasks matching the expected count.
- Documented in `wiki/proxy.md` under a new "Debug endpoints" subsection.

---

- [ ] **Unit 6: Repro harness for multi-agent reconnect storm**

**Goal:** Produce a single script that reproducibly exercises the failure class from origin §"Latest Correction" (active agent reconnects + large replay requests) against a fresh local proxy, so the fix is provable and regressions are catchable.

**Requirements:** R6

**Dependencies:** Unit 2 (so the harness can assert on stage timings), Unit 4 (so the harness can verify backpressure helps)

**Files:**
- Create: `scripts/repro_codex_replay.py` (new)
- Create: `scripts/fixtures/anthropic_replay_body.json` (recorded shape of a large replay request body, sanitized)
- Create: `scripts/fixtures/codex_response_create_frame.json` (recorded first-frame shape)
- Modify: `scripts/README.md` (add harness section)

**Approach:**
- CLI: `python scripts/repro_codex_replay.py --url http://127.0.0.1:8787 --ws-clients 8 --anthropic-clients 4 --duration 30s`
- Phase 1 (warmup): open 1 WS, send one `response.create`, drain to `response.completed`. Confirms proxy is live.
- Phase 2 (storm): simultaneously:
  - Open N Codex WS connections, send one `response.create` each, keep the session open for the duration.
  - Fire M concurrent large Anthropic POSTs shaped like agent-reconnect replays (from the fixture). Each retries on connection error for up to 60s, mimicking real agent behavior.
- Throughout: probe `/livez` every 250ms, record p50/p95/p99.
- Exit code: non-zero if `/livez` p99 exceeds 500ms during the storm (soft assertion).
- Print a summary: per-phase timings, livez stats, count of successful Codex `response.completed`, count of Anthropic successes.

**Execution note:** Harness-first is fine here — the fixture JSONs can be hand-crafted initially and swapped for captured ones later.

**Patterns to follow:**
- Existing benchmark harness in `benchmarks/` for run-loop structure.
- `websockets` library usage already in `tests/test_proxy_codex_route_aliases.py`.

**Test scenarios:**
- Test expectation: light — the harness is a script, not a library. A smoke test in `tests/test_scripts/test_repro_codex_replay_smoke.py` launches it against a mock server and verifies it exits 0 with the expected summary shape.
- Happy path: harness runs against a healthy proxy; exit code 0; summary shows livez p99 < 500ms.
- Error path: harness invoked with `--url` pointing at a closed port; exits with clear "connection refused" message, non-zero code, within 5 seconds.
- Integration: after Unit 4 lands, harness with default Anthropic concurrency limits shows livez p99 < 100ms; harness with `HEADROOM_ANTHROPIC_PRE_UPSTREAM_CONCURRENCY=10000` (unbounded) reproduces the original starvation (p99 > 5s).

**Verification:**
- CI smoke test runs the script against a mock server on every PR.
- `wiki/proxy.md` gets a "Reproducing the reconnect storm" subsection referencing the script.

## System-Wide Impact

- **Interaction graph:** the new `WarmupRegistry` (Unit 1) and `WebSocketSessionRegistry` (Unit 3) are accessed from `HeadroomProxy` during request handling and by `/debug/*` routes. Both are single-writer-per-event-loop; no locks needed beyond the `asyncio.Lock` in `MemoryHandler._ensure_initialized` (Unit 1).
- **Error propagation:** Unit 3 changes `asyncio.gather(..., return_exceptions=True)` to explicit `asyncio.wait(FIRST_COMPLETED)` + cancel. This means partner-side exceptions surface earlier — previously, a client-side error would still let upstream-side drain to completion. Verify in test scenarios that this does not drop in-flight frames that the client had already queued.
- **State lifecycle risks:** Unit 3's session registry must be deregistered in the outermost `finally` — a leak here re-creates the exact bug this plan fixes. Unit 1's `asyncio.Lock` must be released on exception paths (use `async with`, not manual acquire/release).
- **API surface parity:** no change to `/v1/responses` or `/v1/messages` request/response contract. The only new routes are `/debug/*`, loopback-gated.
- **Integration coverage:** Unit 3 + Unit 6 together are the main integration story — unit-test-only coverage of relay cancellation is insufficient; the repro harness is how we prove it in aggregate.
- **Unchanged invariants:** Compression stays enabled on all paths. Upstream WS retry/open-timeout handling (`openai.py:1559-1767`) is untouched. WS→HTTP fallback normalization (`openai.py:1815`) is untouched except for threading through the new `session_id` / stage-timer context. Memory-context fail-open `asyncio.wait_for` (`openai.py:1439-1452`, and the equivalent on the Anthropic path) is untouched. `/livez` logic stays trivial and IO-free.

## Risks & Dependencies

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Unit 3's new cancellation semantics drop in-flight frames the client had already queued on the upstream side. | Med | High (user-visible stream corruption) | Explicit test scenarios for "client disconnect mid-stream" verify no frames silently lost; harness (Unit 6) measures successful `response.completed` count. |
| Unit 4's semaphore default is too conservative and adds visible latency under normal multi-agent use. | Med | Med | Default computed from `cpu_count()` with a floor of 2 and ceiling of 8; tunable via CLI + env; Unit 2 timings will surface if the wait is user-visible. |
| Unit 4's semaphore default is too permissive and doesn't actually stop starvation under real replay storms. | Med | High (plan fails R1) | Unit 6 harness tests both bounded and unbounded configs; ship with conservative default, relax based on harness data post-merge. |
| Unit 5's `/debug/tasks` exposes sensitive state (e.g., in-flight request bodies via coro locals). | Low | Med (info-leak if endpoint ever becomes non-loopback) | Serializer strips coro locals; only task *metadata* (name, age, qualname) is exposed, never arguments. Loopback guard is enforced by middleware before the handler runs. |
| Unit 1's embedder warm-up encode triggers a HuggingFace download in CI and slows test runs. | Med | Low | Warm-up encode is skipped when `optimize=False`; test environment uses a stub embedder; document in `tests/conftest.py`. |
| The real degradation is *not* caused by WS task leaks or replay storms (our top hypotheses), and this plan's instrumentation exposes it but the mitigations don't fix it. | Low | Med | This is fine. The plan is observe-first; Unit 2 + Unit 5 give the data for the next iteration. The plan is valuable even if Units 3 and 4 turn out to be insufficient on their own. |
| `asyncio.Lock` added in Unit 1 is held across the memory backend init, which itself calls `await HierarchicalMemory.create()` — if that hangs, first-request + `ensure_initialized` both hang. | Low | High (deadlock at startup) | Wrap `_ensure_initialized` in a `wait_for(..., timeout=STARTUP_INIT_TIMEOUT_SECONDS)` (configurable, default 30s). On timeout, log error and leave `_initialized=False` so subsequent requests retry. |
| Tests that monkey-patch `MemoryHandler._initialized` directly break because of the new lock ordering. | Low | Low | Audit `tests/test_proxy_memory_integration.py` and fixtures; use `ensure_initialized` via the public entry point only. |

## Documentation / Operational Notes

- Add a "Debug endpoints" subsection to `wiki/proxy.md` covering the three new loopback-only routes, their output shape, and the loopback-only guarantee.
- Add a "Reproducing the reconnect storm" subsection to `wiki/proxy.md` referencing `scripts/repro_codex_replay.py`.
- Update `wiki/metrics.md` with the new histogram series names from Unit 2 and the new gauges from Unit 3.
- Update `CHANGELOG.md` under an "Unreleased" heading with: (a) compression preserved — no behavioral change for existing Codex users; (b) new stage timings visible in logs; (c) new `HEADROOM_ANTHROPIC_PRE_UPSTREAM_CONCURRENCY` env var; (d) new `/debug/*` loopback endpoints.
- **Rollout note:** deploy behind the existing launchd flow; monitor `/metrics` for the new histograms; monitor logs for `pre_upstream_wait_ms` — sustained non-zero values across many requests mean the Unit 4 default is too tight for this machine's load profile.
- **Rollback note:** each unit is cherry-pickable. If Unit 4 causes regression, revert only Unit 4 (semaphore removal); Units 1–3 and 5–6 have no user-visible behavior change.

## Sources & References

- **Origin document:** `wiki/plans/2026-04-17-codex-proxy-runtime-analysis.md`
- Related code:
  - `headroom/proxy/handlers/openai.py` — `handle_openai_responses_ws`, `_client_to_upstream`, `_upstream_to_client`, `_ws_http_fallback`
  - `headroom/proxy/handlers/anthropic.py` — `handle_anthropic_messages`
  - `headroom/proxy/server.py` — `HeadroomProxy.__init__`, `.startup`, health routes
  - `headroom/proxy/memory_handler.py` — `MemoryHandler._ensure_initialized`
  - `headroom/transforms/content_router.py` — `eager_load_compressors`
  - `headroom/transforms/kompress_compressor.py` — `_load_kompress_onnx`, `_kompress_cache`
- Related PRs/issues:
  - Upstream: `https://github.com/chopratejas/headroom/issues/172`
  - Fork branch: `fix/responses-retries-keep-compression` (commit `0b11637`)
- External docs: none used for this plan.
