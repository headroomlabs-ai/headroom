# context-mode → Headroom: enterprise plugin & variant analysis

Analysis date: 2026-07-29. Sources: `/Users/tcms/demo/context-mode` @ v1.0.169, `/Users/tcms/demo/headroom` @ main.

---

## 1. Bottom line

context-mode and Headroom attack the same cost problem at **two different layers**, and they do not
overlap where it matters:

| | context-mode | Headroom |
|---|---|---|
| Interception point | agent **tool-call boundary** (host hooks + MCP) | model **API boundary** (proxy / SDK / MCP) |
| Position relative to context | **pre-context** — data never enters | **in-context** — data already entered, gets squeezed |
| Mechanism | admission control: block, redirect, sandbox, externalize | compression: crush, cache, retrieve |
| Touches the wire request | never | always |
| Loss | lossless (full content in FTS5, queryable) | lossy squeeze + hash rehydrate |

Headroom's own realignment doc identifies its correct compression target as the **live zone**:
"latest user message content + latest `tool_result` + latest `function_call_output` + latest
`local_shell_call_output`" (`REALIGNMENT/00-overview.md`, Phase B).

**That is precisely the payload context-mode intercepts one layer earlier.** Headroom Phase B is
building a Rust engine to compress the latest tool result *after* it hits the wire. context-mode
stops that tool result from being produced at all. These are complements, not competitors — and the
upstream position is strictly cheaper: nothing to compress, nothing to cache-invalidate, no
token-validation fallback needed.

Three strategic unlocks, in order of value:

1. **Cache safety.** Headroom's #1 identified bug class is prompt-cache busting from request
   mutation (5 top-tier cache-killer bugs, `REALIGNMENT/00-overview.md`). context-mode has
   *structurally zero* cache-bust risk because it never touches the request body.
2. **Subscription safety.** The realignment flags "fingerprint-class subscription-revocation
   risks" from `X-Headroom-*` header leakage, `anthropic-beta` mutation and re-serialization on
   OAuth/subscription CLIs. A hook-layer product carries none of this — it is invisible to the
   upstream. This is a *deployable-where-the-proxy-can't-go* capability.
3. **Proxy-free deployment.** Headroom's value today requires being in the API path
   (`127.0.0.1:8787`). Verified live this session: with the proxy down, `headroom_stats` returns all
   zeros and `headroom_compress` no-ops. Enterprises that cannot reroute model traffic (TLS trust,
   egress policy, subscription auth) currently get nothing. context-mode's hook+MCP model needs no
   interposition.

Zero references to context-mode exist in the Headroom tree today — clean slate.

---

## 2. context-mode: portable IP inventory

41,617 lines of TypeScript, 11 MCP tools, 18 host adapters, npm-distributed
(`context-mode@1.0.169`, 8 runtime deps, esbuild-bundled).

Ranked by *how hard it would be for Headroom to rebuild*:

### Tier 1 — genuinely hard, no Headroom equivalent

**1. Cross-host hook adapter layer** — `src/adapters/**` (~10K LOC), `src/adapters/types.ts`,
`src/adapters/detect.ts` (737 lines), `configs/` (18 hosts).
Normalizes three incompatible paradigms — `json-stdio` (Claude Code, Gemini/Qwen, Copilot, Codex,
Kimi, Cursor, Kiro, Antigravity), `ts-plugin` (OpenCode, KiloCode, OpenClaw), `mcp-only` (Zed, Pi,
OMP) — behind one contract: normalized `PreToolUse` / `PostToolUse` / `PreCompact` /
`SessionStart` events, a `PlatformCapabilities` matrix, and a 5-way decision
(`allow | deny | modify | context | ask`). Per-host install, config-format, and self-heal machinery
included (`hooks/heal-partial-install.mjs`, `scripts/plugin-cache-integrity.mjs`).
*Why hard to rebuild:* the value is entirely in the accumulated per-host quirks. There is no spec to
implement against.

**2. Tool-boundary policy engine** — `src/security.ts` (889 lines).
A real policy decision point, not a regex list: glob→regex compilation, chained-command splitting
(`&&`/`;`/`|` with escape awareness), subshell extraction, deny/ask pattern ingestion from host
settings files, project-boundary containment (`evaluateProjectContainment` — Issue #852: an approved
`ctx_execute_file` cannot escape the repo via a path the user couldn't see), and a
**shell-escape scanner** (`SHELL_ESCAPE_PATTERNS`, `extractShellCommands`) that detects
`execSync`/`subprocess`/etc. embedded inside sandboxed *non-shell* code and re-evaluates the escaped
command against policy.
*Why hard to rebuild:* this is the sandbox-escape prevention layer. Getting it wrong is a CVE.

**3. Multi-language sandbox executor** — `src/executor.ts` (785), `src/runPool.ts`,
`src/exit-classify.ts`, `src/truncate.ts`.
12 languages, stdout-only egress, timeouts, background detach, output caps, exit classification.
Enforces the "Think in Code" contract: the agent programs the analysis, only the answer enters
context.

**4. Lossless externalization store** — `src/store.ts` (2,071 lines).
Dual SQLite FTS5 index — a tokenized `chunks` table *plus* a `chunks_trigram` table for
substring/identifier search where BM25 tokenization fails on code — with a `vocabulary` table and
schema migration path. Auto-externalizes any output >100 KB into FTS5 and returns a pointer.
Nothing is discarded; the model queries on demand.

### Tier 2 — valuable, but partially duplicated in Headroom

**5. Counterfactual savings accounting** — `src/session/analytics.ts` (3,085 lines),
`src/session/project-attribution.ts`, `src/session/db.ts` (1,726).
`ContextSavings`, `ThinkInCodeComparison`, `RealBytesStats`, `MultiAdapterLifetimeStats`,
`enumerateAdapterDirs()`. Measures *what would have entered context but didn't* — a different and
harder quantity than Headroom's `savings_ledger.py`, which records actual compression deltas.
Session event ledger + `tool_calls` + resume + per-project attribution.

**6. Multi-vendor pricing catalog** — `src/session/pricing.ts` + `model-prices.json`.
61 curated models × 4 rate buckets (input / output / cache-read / cache-write), refreshed from
litellm, unknown model → `null` rather than a silently wrong Claude rate.
**Overlaps `headroom/pricing/*` heavily. Do not port.**

### Tier 3 — do not port

Compression heuristics, memory/graph/relevance, telemetry transport, dashboard, install UX,
update-check. Headroom has all of these, more mature, and Phase B/H is actively consolidating them.

---

## 3. Headroom's actual extension seams

Verified entry-point groups (all `importlib.metadata`-discovered, all opt-in):

| Seam | Group | Contract | Source |
|---|---|---|---|
| Proxy extension | `headroom.proxy_extension` | `install(app: FastAPI, config: ProxyConfig) -> None` | `headroom/proxy/extensions.py:52` |
| Pipeline extension | `headroom.pipeline_extension` | `on_pipeline_event(PipelineEvent) -> PipelineEvent \| None` over 11 stages | `headroom/pipeline.py:13,68` |
| Learn plugin | `headroom.learn_plugin` | — | `headroom/learn/registry.py:44` |
| Memory text store | `headroom.memory_text` | — | `headroom/memory/config.py:41`, `factory.py:57` |
| Memory vector store | `headroom.memory_vector` | — | `headroom/memory/config.py:34` |
| Memory store | `headroom.memory_store` | — | `headroom/memory/config.py:25` |
| CCR backend | `headroom.ccr_backend` | — | `headroom/cache/compression_store.py:981` |
| Compression hooks | (subclass, not entry point) | `pre_compress` / `compute_biases` / `post_compress` | `headroom/hooks.py:1-31` |

Two things worth noting:

- `headroom/proxy/extensions.py:32` states an explicit **stability contract**: changing
  `install(app, config)` or the group name requires a deprecation cycle. This is a supported public
  seam, not an accident.
- `headroom/hooks.py:16` says outright: *"Headroom SaaS implements position-aware compression and
  cross-turn deduplication via these hooks."* The open-core split is already designed in.

**The exemplar to copy:** `plugins/headroom-oauth2/` — own `pyproject.toml`, own `LICENSE`, own
`SPEC.md`, registers on `headroom.proxy_extension`, dormant until `--proxy-extension oauth2`,
all config via env, "zero core changes." That is the enterprise plugin template.

**The precedent to copy:** `headroom/lean_ctx/installer.py` and `headroom/rtk/installer.py` —
Headroom already ships thin installers that adopt sibling products. `plugins/headroom-agent-hooks`
already installs startup hooks into Claude Code and Copilot CLI. The socket exists.

**The gap:** Headroom has *no tool-boundary interception anywhere*. It sees `tool_use`/`tool_result`
only as message content after the fact (`headroom/parser.py`, `headroom/tokenizers/*`). Its
`PipelineStage` enum has no tool-result stage. Everything context-mode does is upstream of
Headroom's earliest hook.

---

## 4. Proposed plugins & variants

Ranked by value ÷ effort.

### P1 — `headroom-recall`: FTS5+trigram lossless store as `headroom.memory_text`

**What:** port `src/store.ts` behind the existing `headroom.memory_text` seam.

**Why this first:** it is the smallest diff onto an *already-existing* contract, and it fixes a real
product limitation. Today `headroom_retrieve(hash)` requires you to *know the hash* — the tool
description literally says "hash comes from compression markers like `[N items compressed... hash=abc123]`".
With an FTS5-backed store you get `retrieve-by-query`: "what did that build log say about OOM"
instead of "paste hash abc123". The trigram index matters specifically because BM25 tokenization
loses identifiers and stack frames.

Composes rather than replaces: `compress` → return squeezed text + hash → store the *original* in
FTS5 → rehydrate by hash **or** by query. Also a natural `headroom.ccr_backend` implementation —
the realignment wants "CCR hardens: persistent backend" (Phase B), and this is one.

**Enterprise variant:** shared team store, retention/TTL policy, per-project scoping (context-mode
already has `project-attribution.ts`), audit of every retrieval.

**Effort:** medium. Reimplement in Python/Rust against Headroom's memory interface, or ship the
node store as a sidecar. Do not port the MCP tool surface — only the store.

### P2 — `headroom-admission`: tool-boundary admission control across 18 hosts

**What:** context-mode's adapter + hook layer, distributed the way `plugins/openclaw` and
`plugins/opencode` already are (TS package under `plugins/`), reporting savings into Headroom's
`savings_ledger.py` JSONL and emitting Headroom pipeline events.

**Why:** this is the strategic piece. It gives Headroom:
- a **pre-wire** enforcement point, upstream of Phase B's live-zone engine, with no cache-bust and
  no token-validation fallback required;
- coverage of **18 agent hosts** — the realignment's Phase G wants to "extend wrap CLIs (cline,
  continue, goose, openhands)"; this is that work already done, and then some;
- a deployment mode that works under **subscription auth**, where the proxy is a revocation risk.

**Enterprise value — this is the DLP story Headroom cannot currently tell.** A `curl` inside a Bash
tool call never touches the proxy, so Headroom is blind to it. context-mode blocks
`curl`/`wget`/`WebFetch`/inline `fetch()`/`requests.get` at the tool boundary and forces network
egress through `ctx_fetch_and_index`. That converts a token-savings feature into an
**egress-control** feature — a different budget line and a different buyer.

**Effort:** high, but it's mostly packaging + a reporting bridge, not a rewrite. Keep it TypeScript;
Phase H retires Python *proxy* code but explicitly preserves "CLI wrappers, RTK installer" — the
installer layer is the surviving Python, and it can shell out.

### P3 — `headroom-policy` (Enterprise, license-gated): the PDP

**What:** `src/security.ts` as a policy decision point, plus centrally-managed org rulesets.

Two attach points: the hook layer from P2 (tool-level `allow/deny/ask`), and
`headroom.pipeline_extension` at `PRE_SEND` (prompt-level policy). Feeds `headroom/audit/`.

**Enterprise features that only make sense paid:** central policy service, org-wide allow/deny
rulesets, project-boundary containment enforcement, shell-escape detection inside sandboxed code,
tamper-evident audit trail, per-team reporting. Gate it with the ELv2 license key (see §6).

**Effort:** medium. The engine exists and is tested (`tests/security/`, `src/security.ts` 889 lines);
the work is the control plane.

### P4 — `headroom-sandbox`: Think-in-Code execution

**What:** `executor.ts` exposed as a Headroom MCP tool (`headroom_execute`), 12 languages,
stdout-only.

**Why:** this is the mechanism behind context-mode's largest measured savings —
`ctx_execute_file` returns 98% savings across 315 KB of real fixtures (`BENCHMARK.md` Part 1),
versus 82% for index+search (Part 2). Programming the analysis beats compressing the output.

Must ship *with* P3: the shell-escape scanner is what stops the sandbox being an escape hatch.

**Effort:** medium-high. Runtime isolation is the hard part; `headroom` already has a `sandbox` extra
in `pyproject.toml` to build on.

### P5 — `headroom-attribution`: counterfactual savings + per-project cost

**What:** port the *methodology* from `session/analytics.ts` — `RealBytesStats`,
`ThinkInCodeComparison`, `enumerateAdapterDirs`, `project-attribution.ts` — into Headroom's
`savings_ledger` / `reporting` / `dashboard`.

**Why:** Headroom measures compression deltas (what it squeezed). context-mode measures the
counterfactual (what never entered). Enterprise buyers want the second number, sliced by team and
repo. Do **not** port `pricing.ts` — `headroom/pricing/*` already does this with litellm resolution.

**Merge, don't port.** `headroom/audit/reads.py` is already a counterfactual measurement tool over
the same Claude Code transcript corpus (see §8). It has the better mechanism taxonomy — identical
repeat, subset containment, write-readback, stale, line-number scaffolding, context residency,
cache-death windows. `analytics.ts` has the multi-host coverage and per-project attribution it
lacks. Combine the two rather than adding a third implementation.

**Effort:** low-medium, mostly a metrics-definition merge.

### Variants (packaging, not code)

- **Headroom No-Proxy Edition** — P1+P2 only, zero API interposition. Sells to buyers who cannot
  reroute model traffic and to every subscription-auth user. Removes the single biggest deployment
  blocker Headroom has.
- **Headroom Admission Control (Enterprise)** — P2+P3+P4 with a central policy plane and fleet
  enrollment across 18 hosts. Positioned as AI-agent DLP/governance, not token savings.
- **Headroom Fleet** — P5 + `enumerateAdapterDirs` for org-wide rollout state and cost reporting.

---

## 5. Evidence base

context-mode's `BENCHMARK.md`: 21 scenarios, 376 KB raw → 16.5 KB context, **96% overall**, all
fixtures captured from real tool invocations (Context7, Playwright, `gh`, vitest, tsc, nginx logs,
`git log`, analytics CSV) rather than synthetic. Honest about its weak cases — 13% on a 0.4 KB
Playwright network dump, and Part 2 openly explains why index+search only reaches 50-93% (it returns
exact code blocks rather than summaries, by design).

Test suite: 125 tests across executor/store/MCP-integration/ecosystem, plus 45 test dirs in `tests/`
covering adapters, security, session, hooks, analytics.

That's a defensible enough evidence base to reuse in Headroom's own materials, and the fixture corpus
itself is reusable for Headroom's `benchmarks/`.

---

## 6. Blockers — resolve these before writing code

**1. License incompatibility (hard blocker).**
context-mode is **Elastic License 2.0**, "Copyright 2026 Mert Koseoglu". Headroom is
**Apache-2.0**, "Copyright 2025 Headroom Contributors".

- ELv2 code **cannot** be merged into the Apache-2.0 core. Not a technicality — it would relicense
  Headroom's core.
- ELv2 forbids providing the software "to third parties as a hosted or managed service." That
  directly constrains `headroom-managed/`.
- Different copyright holders means this needs an **IP arrangement between entities**, not an
  engineering decision.

The good news: Headroom's plugin architecture is exactly the boundary that makes this tractable.
A separate package with its own `pyproject.toml` and its own `LICENSE`, registered on an entry
point — the `plugins/headroom-oauth2/` shape — can carry ELv2 while core stays Apache-2.0. ELv2 is
also the *right* license for a license-key-gated enterprise tier; it explicitly contemplates one.

Recommendation: any context-mode-derived code ships as separately-licensed plugin packages under
`plugins/`, never vendored into `headroom/`. Get the IP arrangement in writing first.

**2. Realignment collision.**
Phases A–I are ~40 PRs / 8–13 weeks and include deleting ~25K LOC. Do not open a new integration
front mid-Phase-B. P1 (`headroom.memory_text` / `ccr_backend`) is the exception — it *serves* Phase
B's "CCR hardens: persistent backend" goal rather than competing with it.

**3. Phase H direction.**
Python proxy code is being retired. Write nothing new in `headroom/proxy/`. Target the surviving
layers: installers, memory writers, CLI wrappers, and Rust.

---

## 7. Sequencing

| Order | Item | Gate |
|---|---|---|
| 0 | IP/licensing arrangement | before any code |
| 1 | P1 `headroom-recall` — FTS5 store on `memory_text`/`ccr_backend` | lands inside Phase B, serves it |
| 2 | P2 `headroom-admission` — 18-host hook layer under `plugins/` | after Phase A stabilizes |
| 3 | Variant: **No-Proxy Edition** = P1+P2 | as soon as P2 works on 3+ hosts |
| 4 | P3 `headroom-policy` (Enterprise, ELv2, key-gated) | after P2 |
| 5 | P4 `headroom-sandbox` | with P3, never before |
| 6 | P5 `headroom-attribution` | opportunistic |

---

## 8. Follow-up verification

All four items flagged as open in the first pass are now resolved.

**`headroom-managed/` is the SaaS arm, and it is unlicensed.**
`headroom-managed/pyproject.toml`: `name = "headroom-managed"`, `description = "Headroom SaaS
Platform - Managed context window optimization"`, `version = 0.1.0`. It has `app/auth.py`,
`app/middleware/`, `app/routes/`, `app/services/`, `app/models.py`, alembic migrations, and a
`pilot/`. There is **no `license` field and no LICENSE file** — i.e. proprietary by default.

This *sharpens* the §6 blocker rather than easing it. ELv2 forbids providing the software "to third
parties as a hosted or managed service." The product whose name is literally *Managed* is the one
place context-mode-derived code cannot go without an explicit commercial grant from the copyright
holder. Plan the plugin boundary so that `headroom-managed` consumes only Apache-2.0 core
interfaces, never ELv2 implementations.

**`headroom/audit/reads.py` does not overlap P3 — and it independently validates the whole thesis.**
It is a *measurement* tool, not an audit trail: it streams Claude Code `*.jsonl` transcripts to size
"the addressable bytes for each Read compression mechanism... so defaults are set from traffic, not
theory." No policy, no tamper-evidence. P3's audit trail remains a gap.

Two lines in its docstring are the most useful corroboration in either repo:

- *"context residency — how many assistant turns each Read stays in context (the multiplier on its
  prefix-cache read cost; **the case for compress-before-cache-entry**)"* — Headroom is already
  arguing, from its own traffic, for moving earlier in the pipeline. context-mode is the terminus of
  that argument: compress before **context** entry, not merely before cache entry.
- *"identical repeat — a dedup mechanism for this was prototyped and removed: it measured 0.1% of
  Read bytes on real traffic."* — Headroom has already empirically established that
  message-history-level dedup is worthless. The addressable bytes are at the tool boundary, not in
  history. That is the same conclusion the realignment reached from the cache side, arrived at
  independently from the traffic side.

It *does* overlap **P5** — `audit/reads.py` and context-mode's `session/analytics.ts` are two
independent implementations of counterfactual measurement over the same transcript corpus. Merge
them rather than porting; `audit/reads.py` has the better mechanism taxonomy, `analytics.ts` has
multi-host coverage and per-project attribution.

**No plugin-authoring docs exist.** `docs/` is a Next.js site (`app/`, `content/`, `components/`);
`wiki/` has nothing on extension authoring (only `macos-deployment.md` matched). `plugins/headroom-oauth2/SPEC.md`
remains the de-facto authoring reference — which means whichever plugin lands first sets the house
style. Worth writing the authoring doc as part of P1.

**Headroom publishes no benchmark results.** `benchmarks/` is 29 runner scripts with no committed
results artifacts, so no like-for-like number exists to compare against context-mode's 96%. The
comparison has to be run. The harness is there and is unusually strong on exactly the axis that
matters: `prefix_cache_benchmark.py`, `cache_bust_trace_report.py`, `cache_validation_bundle.py`,
`synthetic_token_cache_bust_report.py`, `proxy_mode_benchmark.py`, `agent_cost_benchmark.py`,
`real_world_agent_benchmark.py`. Use it to *prove* the §1 cache-safety claim empirically rather than
asserting it — a measured "zero cache-bust events" result is the strongest possible artifact for the
No-Proxy Edition.

**Bonus finding — the platform axes are orthogonal.**
`docs/platform-feature-matrix.json` (schema v1, updated 2026-07-06) tracks coverage across
`["linux", "macos", "windows"]` — Headroom's platform axis is **operating system**. context-mode's
platform axis is **agent host** (18 of them). Headroom tracks no host-coverage matrix at all. P2
therefore fills a dimension that does not currently exist in Headroom's own feature accounting,
which also means it needs a second matrix rather than new rows in this one.

*Process note:* six subagents were dispatched across this analysis and all six stalled at the
600-second watchdog; one reported "Bash is temporarily unavailable" before dying, so the failures
were tool-layer, not analytical. Every finding in this document was verified directly.
