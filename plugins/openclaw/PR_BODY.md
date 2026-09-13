## Description

<!-- Briefly explain the change and why it is needed. -->

**Goal:** make the Headroom OpenClaw plugin work correctly on OpenClaw 2026.9.x and be safe for tool-heavy agent sessions, **without changing any default that upstream `main` has chosen**. Everything in this PR is either (a) a fix for behavior that is currently broken on the 2026.9.x runtime, or (b) opt-in and off by default.

The plugin on `main` is a good design; this PR is the result of running it in production against a multi-provider OpenClaw gateway and fixing what broke, then hardening those fixes through four review rounds. I have tried to keep each change small, tested, and traceable to a concrete failure.

### What differs from `main` today, and why

| Area | `main` today | This PR | Why |
|------|--------------|---------|-----|
| **Turn contract** | Engine does not declare `transcriptSemantics` or implement `commitTurn()` | Declares `currentTurnFence` + `turnAdvancementIdempotency`; `commitTurn()` durably records the accepted turn (key + SHA-256 digest of `messages`, no message bodies) keyed by `advancementKey`, returns `duplicate` on retry / after restart | On 2026.9.x the runtime logs `Context engine "headroom" degraded to "legacy" for this logical turn` every turn and never calls `assemble()`. Without this, the plugin does nothing. |
| **Turn-store locking** | n/a | `store-lock.ts`: ownership published atomically (hard-link publish, `O_EXCL` fallback); a lock is stale only when *old* **and** its owner is provably gone (EPERM counts as alive); recovery via atomic `rename`; release never unlinks someone else's lock | Round-4 review reproduced a live lock being unlinked during the `open(wx)` → PID-write window. |
| **Tool payloads through the proxy** | `extractText()` flattens tool content; images dropped; `toolName` falls back to `"headroom"` | Non-text blocks never leave the process — a short placeholder goes on the wire — and are restored from the local originals by `tool_call_id` / position. OpenAI `tool.name` set from `toolName`. | Vision/browser/`view_image` results were silently lost; the proxy's protect list keys on tool names. An intermediate design that sent blocks as base64 text inflated a 60 KB image to ~50k tokens and depended on the proxy echoing `_headroomMeta`, which it does not do reliably — replaced. |
| **Deferred tool wrappers** | OpenClaw Tool Search sends every MCP call as `tool_call`; the proxy's protect/exclude lists key on that name | `agentToOpenAI` resolves `{id: "mcp:<server>:<server>__<tool>"}` / Hermes `{name}` to `mcp__<server>__<tool>` on the wire (wrapper restored from originals); opt-in `protectToolResults` restores named tools' results verbatim after every compress round trip | Per-tool protection was impossible for MCP tools; prose outputs (video/vision analysis) came back paraphrased |
| **Durable compaction** | `ownsCompaction: false`, `compact()` → `delegateCompactionToRuntime()` (#2304) | **Same default** (`persistentCompaction: "openclaw"`). Opt-in `"hybrid"` (Headroom replace-only pre-pass + turn-end hygiene, OpenClaw still summarizes) and `"headroom"` (zero-LLM `/v1/compress` rewrite, `ownsCompaction: true`) | Upstream's delegation is correct and stays the default. The opt-in modes exist for very large tool-heavy transcripts where LLM `/compact` is slow or fails. |
| **Truncation boundary** | n/a | `truncate-boundary.ts`: the durable tail always starts at a turn start (OpenClaw's own `isTurnStartMessage` rule), never at a `toolResult` or assistant continuation; orphan tool results dropped; `resetLeaf()` used so the prefix is actually removed | Round-3/4 review: `branch(parentId)` retained the prefix, and a raw suffix cut could start at an orphan `toolResult`. |
| **Multi-upstream routing** | One `*_TARGET_API_URL` per proxy | Opt-in `providerUpstreams` → `x-headroom-base-url` (the proxy's own documented mechanism); protocol-aware path prefix (`/v1` vs `/v1beta` for Gemini) | Real deployments route several OpenAI-compatible APIs through one proxy; Gemini must keep `/v1beta` to reach `handle_gemini_generate_content` (round-1 review). |
| **Session-gated providers** | n/a | Opt-in `providerSessionHeaders` (per-process UUID under an operator-chosen header) | e.g. opencode-go returns `MissingSessionID` without one. No provider names hardcoded. |
| **Assemble cost** | Proxy called every turn | Skip when rough estimate < `(tokenBudget − assembleReserveTokens) × assembleSkipBudgetRatio` (defaults 20000 / 0.7). OpenClaw hands `assemble()` the full window as `tokenBudget` while its overflow precheck also counts the system prompt and a ≥ 20k reserve, so a flat 85 % rule never fired before native compaction on ~200k windows; opt-in, provider-aware `skipAssembleWhenGatewayRouted`; images estimated at a fixed ~1.5k tokens (not `base64.length / 4`) | Sub-budget turns on 1M-window models paid multi-minute compress for 0 savings; base64 length was defeating the budget check. |
| **Durable compress defaults** | `protect_recent: 0` when hygiene runs | `protect_recent: 2`; replace-mode skips messages with protected tool/image payloads | Avoid rewriting the turn the model is about to answer, and never lossy-rewrite multimodal payloads on disk. |

### Review history

| Round | Finding | Resolution |
|-------|---------|------------|
| 1 | `commitTurn` was a no-op; Gemini `/v1beta` collapsed to `/v1` | `TurnAdvancementStore`; `resolveProxyPathPrefix()` |
| 2 | Failed persist returned `duplicate` on retry; cross-instance clobbering; truncate via `branch()` kept the prefix | Reload-under-lock + atomic write; exclusive lock; `resetLeaf()` |
| 3 | Upstream #2304 flipped to delegation | Adopted as the default; Headroom modes made opt-in |
| 4 | Empty new lock treated as stale; raw suffix cut at orphan `toolResult`; Windows test failures; stale PR body | `store-lock.ts`; `truncate-boundary.ts`; portable paths/URLs in tests; this rewrite |

Closes # (no upstream issue; surfaced from production use of the plugin on OpenClaw 2026.9.x)

## Type of Change

- [x] Bug fix (non-breaking change that fixes an issue)
- [x] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [x] Documentation update
- [x] Performance improvement
- [ ] Code refactoring (no functional changes)

## Changes Made

All changes are confined to `plugins/openclaw/`. No Python runtime or proxy changes.

**Always-on fixes (no config needed)**
- `engine.ts` — declare `transcriptSemantics`; `commitTurn()` → `TurnAdvancementStore`; budget short-circuit in `assemble()`; CCR retrieve hint only when `ccrHashes` is non-empty; pass local originals to `openAIToAgent`.
- `turn-advancement-store.ts` + `store-lock.ts` — durable, idempotent advancement keyed by `advancementKey`; records hold only a digest of the accepted messages (OpenClaw owns the transcript) and are pruned after 14 days / 5 000 records so the file stays small; v1 files that embedded message bodies are migrated on first commit; reload-under-lock; atomic temp+rename persist; safe cross-process lock protocol (see table).
- `convert.ts`, `content-blocks.ts`, `original-lookup.ts` — placeholder wire format for non-text blocks; originals-based restore (`tool_call_id` → `hrIndex` hint → position); OpenAI `tool.name`; `isError` preserved; fixed per-image token estimate; assistant text no longer duplicated across blocks.
- `compaction.ts` + `truncate-boundary.ts` — turn-aligned truncation, orphan `toolResult` removal, `resetLeaf()`; replace-mode skips protected payloads; `protect_recent: 2` (`compress-request-config.ts`).

**Opt-in (default off / matches upstream)**
- `compaction-mode.ts`, `openclaw-compaction.ts`, `transcript-hygiene.ts`, `hygiene-debounce.ts`, `transcript-projection.ts` — `persistentCompaction: "openclaw" | "hybrid" | "headroom"` (default `"openclaw"` = upstream delegation); `transcriptHygiene` (on only in `hybrid`).
- `gateway-config.ts`, `proxy-routing.ts`, `session-headers.ts` — `providerUpstreams`, `/v1` vs `/v1beta`, `providerSessionHeaders`.
- `assemble-skip.ts` — provider-aware `skipAssembleWhenGatewayRouted` (reads `runtimeSettings.model.provider`).
- `openclaw.plugin.json` — schema for the new keys; `README.md`, `docs/PR_OVERVIEW.md`, `docs/tool-call-preservation.md`, `docs/TEST_MATRIX.md`, `test/README.md`.

## Testing

<!-- Check what you actually ran, then paste the real command output below. -->

The plugin is TypeScript, so the Python commands in the template do not apply; the equivalents are:

- [x] Unit tests pass (`npm test` — vitest; equivalent of `pytest`)
- [x] Linting passes (`npm run typecheck` — `tsc --noEmit`; the plugin has no separate lint config)
- [x] Type checking passes (`npm run typecheck`)
- [x] New tests added for new functionality
- [x] Manual testing performed

### Test Output

```text
$ cd plugins/openclaw && npm test
 Test Files  21 passed (21)
      Tests  320 passed (320)
$ npm run typecheck        # clean
$ npm run build            # dist/index.js 81.75 KB
$ node test/live-compress-smoke.mjs   # against a running Headroom proxy
Request tool.name: view_image
Response has image block: true
PASS: live compress smoke test — image payload preserved
```

New/updated tests for this round:
- `test/store-lock.test.ts` — fresh empty lock is never stolen (**deterministic two-process test**: a child creates the lock with `open(wx)` and holds it before writing its PID; the parent waits ≥ hold time, child observes `STILL_HELD`, both commits land); stale recovery requires age **and** dead owner; EPERM = alive; hard ceiling for PID reuse; release never unlinks a foreign lock.
- `test/truncate-boundary.test.ts` + `test/compaction.test.ts` — the 41-message fixture (assistant `toolCall` + `toolResult` + 39 users) plans and applies to 39 messages starting at `user-0`; multi-call turns are kept whole; a proxy cut landing on a `toolResult` is re-aligned; a transcript with no turn boundary is not truncated.
- Windows portability: `tmpdir()`/`join()` instead of `/tmp` literals, `file://` URL for the child-process dist import, `npm` via shell on win32, `homedir()` fallback for the state dir. I don't have a Windows host — would appreciate a re-run there.

Full coverage map: `docs/TEST_MATRIX.md`.

## Real Behavior Proof

### Environment.

- OpenClaw 2026.9.x gateway, `plugins.slots.contextEngine: "headroom"`, plugin built from this branch and loaded from `dist`
- Headroom proxy 0.37.x on `127.0.0.1:8787`, `HEADROOM_MODE=token`
- Providers routed through the proxy: an Anthropic-shape portal API, OpenRouter, an OpenCode-compatible API (`HEADROOM_UPSTREAM_ALLOWED_HOSTS` set accordingly)
- Linux, Node 26

### Exact command / steps.

1. `cd plugins/openclaw && npm install && npm test && npm run typecheck && npm run build`
2. Truncation against the **real** OpenClaw `SessionManager` (SQLite store, not the test double): open a temp session, append the 41-message fixture, `planHeadroomCompaction({ force: true })`, `applyCompactionPlan()`, reopen the store and read the branch.
3. Lock boundary: `npx vitest run test/store-lock.test.ts` (spawns the second process).
4. Tool payloads: `node test/live-compress-smoke.mjs` and `node test/live-compress-stress.mjs` against the live proxy; a raw `/v1/compress` probe with a 60 KB PNG `view_image` result before/after the wire-format change.
5. Gateway: load the built plugin, restart, run tool-heavy sessions (browser / vision / file tools), watch proxy and gateway logs.

### Observed result.

- Real `SessionManager`: persisted branch 41 → **39** after apply (`bytesFreed: 507`), first kept message `user-0`, last `user-38`, no `toolResult` / assistant in the reloaded transcript.
- Lock boundary test: parent `commit()` blocked ~400 ms until the child released; child printed `STILL_HELD`; lock file gone afterwards; store contains the parent key. Old protocol would have unlinked the child's lock immediately.
- Wire format: for the same 60 KB image transcript, `tokens_before` dropped from 49,780 (base64 sent twice) to 9,765; `view_image` content on the wire 80,110 → 62 chars; image block restored locally. Live stress (80-message mixed-tool transcript): 23,518 → 5,025 tokens, `isError` and image blocks preserved with and without `_headroomMeta` echo.
- Gateway: the per-turn `degraded to "legacy"` log line disappears once `transcriptSemantics` is declared; `assemble()` fires each turn. Earlier production measurement with this branch: ~7.7 % overall input-token savings, ~9 % per compressed turn, best single turn ~50k tokens saved. With opt-in `hybrid` compaction, a ~232k-token session compacted to ~24k.

### Not tested.

- **Windows.** The three failures from the last review were addressed by reading the tests (hardcoded `/tmp`, non-URL ESM import path, `npm` spawn); not executed on a Windows host.
- OpenClaw < 2026.9.x — older runtimes ignore `transcriptSemantics` and never call `commitTurn()`, so behavior should be unchanged, but I only run 2026.9.x.
- `persistentCompaction: "headroom"` as the *sole* durable compactor in production. Under sustained load I saw `Pending input ownership ended` mid-turn with it, which is why the default stays `"openclaw"` and my own deployment uses `"hybrid"`.
- Remote (TLS) proxy and multi-proxy fleets — single local proxy per gateway only.

## Runtime Rollout Safety

- Rollout-managed feature(s): `providerUpstreams`, `providerSessionHeaders`, `persistentCompaction` (`hybrid`/`headroom`), `transcriptHygiene`, `skipAssembleWhenGatewayRouted`, `assembleCompressConfig` — all opt-in, default off / upstream-equivalent.
- Minimum rollout channel: any OpenClaw 2026.9.x install; no canary needed since defaults match `main` and the always-on parts are fixes for a path that currently does not run.
- Stable/default behavior changed: **No** for compaction (`"openclaw"` = current `main` delegation). **Yes, intentionally**, for: `assemble()` now actually runs on 2026.9.x; tool/image payloads are preserved instead of flattened; sub-budget turns skip the proxy. For operators who already set `gatewayProviderIds` with non-`/v1` providers, the rewritten URL path now normalizes (`/v1`, or `/v1beta` for Gemini) instead of 404-ing.
- Kill switch / disable path: unset the opt-in keys to return to upstream behavior for that feature; `plugins.slots.contextEngine` back to `legacy` disables the engine entirely; reverting the branch restores `main`.
- Unsafe override required: No.
- Qualification impact: compression metrics (CCR, content_router) for 2026.9.x operators become real rather than zero, since `assemble()` now fires.
- Rollback path: `git revert` the merge; the turn-advancement store file (`headroom-turn-advancements.json` next to the session store) is additive and can be deleted safely.

## Review Readiness

- [x] I have performed a self-review
- [x] This PR is ready for human review

## Checklist

- [x] My code follows the project's style guidelines
- [x] I have performed a self-review of my code
- [x] I have commented my code, particularly in hard-to-understand areas
- [x] I have made corresponding changes to the documentation (`README.md`, `docs/*`, `openclaw.plugin.json` schema)
- [x] My changes generate no new warnings (`npm run typecheck`, `npm run build` clean)
- [x] I have added tests that prove my fix is effective or that my feature works (320 vitest cases; reviewer fixtures locked as regressions)
- [x] New and existing unit tests pass locally with my changes
- [x] I did **not** edit `CHANGELOG.md` — it is generated by release-please from my Conventional Commit PR title (a CI guard enforces this)

## Additional Notes

- **On the compaction fork in the road:** I initially took `ownsCompaction: true` with a real implementation; upstream landed #2304 (delegate to the runtime) in the meantime. I think upstream's default is the right one for stock installs and adopted it. The Headroom-owned modes are kept only as explicit operator opt-ins for cases where LLM `/compact` is impractical. If maintainers would rather not carry those modes, they are isolated (`compaction-mode.ts`, `transcript-hygiene.ts`, `hygiene-debounce.ts`, `transcript-projection.ts`) and can be dropped without touching the always-on fixes.
- **Tradeoffs:** per-process (not per-request) session UUIDs; `/v1` path normalization drops deeper upstream path segments (proxy matches those by query string); the lock waits up to 5 s and then throws rather than stealing — a crashed holder is recovered after 30 s (dead PID) or 10 min (hard ceiling).
- **Scope:** only `plugins/openclaw/`. Nothing in `headroom/` (Python) or the OpenClaw runtime is modified.
- Operator configuration examples for the opt-in features are in `plugins/openclaw/README.md`.
