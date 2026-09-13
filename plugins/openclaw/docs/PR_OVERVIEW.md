# OpenClaw plugin PR — overview vs upstream `main`

This document describes **why** this PR exists, **what** each area changes compared to stock
`headroomlabs-ai/headroom:main`, and **where** to look in the tree. It contains no operator-specific
paths or credentials.

## Problem statement (stock `main` gaps)

| # | Stock behavior | After this PR |
|---|----------------|---------------|
| 1 | OpenClaw 2026.9.x bypasses the Headroom engine every turn when `transcriptSemantics` / durable `commitTurn()` are missing → `assemble()` never runs | Engine declares semantics; durable turn advancement persists accepted messages |
| 2 | One proxy env var → one upstream; multi-provider deployments 404 at the proxy | `providerUpstreams` + `x-headroom-base-url` per provider |
| 3 | Providers with non-`/v1` first-party URLs fail when rewritten to the proxy | Protocol-aware `resolveProxyPathPrefix()` (`/v1` vs `/v1beta` for Gemini) |
| 4 | Session-gated APIs (e.g. opencode-go) return 400 without a session header | `providerSessionHeaders` injects a stable per-process UUID |
| 5 | `compact()` / `maintain()` were no-ops despite `ownsCompaction` | Real durable compaction via Headroom `/v1/compress` + SQLite rewrite |
| 6 | `assemble()` always hit the proxy, even far under token budget | Budget short-circuit (~85% of `tokenBudget`) |
| 7 | Tool payloads mangled in `convert.ts` (images, tool names, assistant blocks) | Non-text blocks stay local (placeholder on the wire); restored from originals by `tool_call_id` / position — no dependency on proxy echoing metadata |
| 8 | Misleading `headroom_retrieve` hint without CCR hashes | Hint gated on `ccrHashes.length > 0` |
| 9 | Double compression when gateway providers route through proxy | Opt-in, provider-aware `skipAssembleWhenGatewayRouted` |
| 11 | Image-heavy sessions over-estimated (`base64.length / 4`) and tripped the budget skip every turn | Fixed ~1.5k token estimate per image |
| 10 | Durable hygiene used `protect_recent: 0` | Default `protect_recent: 2`; skip rewrite of protected tool/image payloads |
| 12 | Turn-advancement lock used `open(wx)` then wrote the PID; a second writer could read the empty lock, treat it as stale, unlink it and commit concurrently | `store-lock.ts`: owner record published atomically (hard-link publish, `O_EXCL` fallback); a lock is stale only when *old* **and** its owner is provably gone (EPERM = alive); recovery claims via atomic `rename` |
| 13 | Forced/compress-result truncate kept a raw message suffix — the durable tail could start at an orphan `toolResult` or an assistant continuation | `truncate-boundary.ts`: cut re-aligned to a turn start (OpenClaw's own `isTurnStartMessage` rule), orphan tool results dropped, whole tool groups kept; verified against the real SQLite `SessionManager` |

## File-level diff map (vs `main`)

### Core runtime

| File | Role |
|------|------|
| `src/engine.ts` | `assemble()`, `compact()`, `maintain()`, `commitTurn()`; budget skip; CCR hint gating; gateway-routed assemble skip; hybrid/openclaw compaction modes |
| `src/convert.ts` | AgentMessage ↔ OpenAI conversion; tool `name`; `openAIToAgent(..., { originals })` restore; image token estimate |
| `src/content-blocks.ts` | Block type guards, wire placeholders, `mergeCompressedTextIntoBlocks`, `messageHasProtectedToolPayload` |
| `src/original-lookup.ts` | Match compressed messages to originals (`tool_call_id`, `hrIndex` hint, position) |
| `src/assemble-skip.ts` | Provider-aware `skipAssembleWhenGatewayRouted` decision |
| `src/compaction.ts` | Durable `/v1/compress` planning + SQLite apply; protected-payload skip on replace |
| `src/compaction-mode.ts` | `persistentCompaction`: `openclaw` (default), `hybrid`, `headroom` |
| `src/compress-request-config.ts` | Shared assemble + durable compress config defaults (`protect_recent: 2`) |
| `src/turn-advancement-store.ts` | Idempotent on-disk turn advancement keyed by `advancementKey`; digest-only records (no message bodies), 14-day / 5 000-record retention |
| `src/store-lock.ts` | Cross-process lock: atomic ownership publish, age + liveness staleness, rename-claimed recovery |
| `src/truncate-boundary.ts` | Turn-boundary selection + orphan `toolResult` removal for durable truncation |
| `src/gateway-config.ts` | In-memory provider rewrite; `providerUpstreams`; `providerSessionHeaders` |
| `src/proxy-routing.ts` | `/v1` vs `/v1beta` pathname normalization |
| `src/session-headers.ts` | Per-provider session UUID generation |
| `src/transcript-hygiene.ts` | Turn-end replace-only hygiene (hybrid mode) |
| `src/hygiene-debounce.ts` | Per-session debounce for hygiene rewrites |
| `src/transcript-projection.ts` | Wait for OpenClaw transcript projection after rewrites |
| `src/openclaw-compaction.ts` | Delegate durable compaction to OpenClaw native when configured |
| `src/plugin/index.ts` | Plugin registration; reads new config keys |

### Config & packaging

| File | Role |
|------|------|
| `openclaw.plugin.json` | Schema for routing, compaction modes, hygiene, `assembleCompressConfig`, `assembleSkipBudgetRatio`, `assembleReserveTokens`, `protectToolResults`, `skipAssembleWhenGatewayRouted` |
| `package.json` | Aligns `headroom-ai` with monorepo release (`^0.37.0`); stress test scripts |
| `package-lock.json` | Locked deps (merged from upstream #3531) |

### Documentation

| File | Role |
|------|------|
| `README.md` | Operator-facing config + proxy env mitigations |
| `docs/tool-call-preservation.md` | Tool-call bug analysis, fix phases, staging checklist |
| `docs/PR_OVERVIEW.md` | This file |
| `docs/TEST_MATRIX.md` | Test coverage map for the full PR |
| `PR_BODY.md` | GitHub PR description (sanitized) |

### Tests

See [TEST_MATRIX.md](./TEST_MATRIX.md). Summary: **320 vitest cases** across 22 files, plus optional live proxy stress scripts.

## Merge with upstream `main` (2026-09-11) — done

Branch `pr-prep` is **current with `origin/main`** (0 commits behind). Integrated upstream:

| Commit | Change |
|--------|--------|
| `#3521` | Dependency security remediation (monorepo-wide) |
| `#3516` | Node 24 for npm release packaging |
| `#3531` | Consolidated dependency updates; `plugins/openclaw/package-lock.json` |

Merge commit: `fc7650a`. No plugin source conflicts — auto-merge on lockfiles only.

## Default behavior vs opt-in features

**Unchanged for stock installs:** `persistentCompaction: "openclaw"`, hygiene off, `skipAssembleWhenGatewayRouted: false`.

**Always-on fixes (no config required):** transcript semantics, `commitTurn` contract, conversion preservation, assemble budget short-circuit, safer durable compress defaults when compaction modes are enabled.

**Opt-in:** multi-upstream routing maps, session headers, hybrid/headroom compaction, gateway assemble skip, custom `assembleCompressConfig`, custom `assembleSkipBudgetRatio` / `assembleReserveTokens`, `protectToolResults`.
