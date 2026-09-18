# Test matrix — full PR coverage

Run everything:

```bash
cd plugins/openclaw
npm test                    # 320 vitest cases
npm run typecheck
npm run build
npm run test:stress         # native-tool mock stress
npm run test:live-stress    # optional; requires Headroom proxy on :8787
```

## Pillar 1 — Durable turn contract (OpenClaw 2026.9.x)

| Behavior | Why stock `main` fails | Test file |
|----------|------------------------|-----------|
| `transcriptSemantics` declared | Engine degraded to legacy without it | `test/engine.test.ts`, `test/pr-regression.test.ts` |
| `commitTurn()` returns `{ status }` | Wrong shape leaves outbox stuck | `test/engine.test.ts` |
| Turn advancement idempotent | Retry/restart must not duplicate | `test/turn-advancement-store.test.ts` |
| Failed persist → retry returns `committed` | Key must not be cached on failure | `test/turn-advancement-store.test.ts` |
| Two instances / two processes preserve all keys | Last-write-wins clobbering | `test/turn-advancement-store.test.ts` |
| Store keeps digests only; v1 message bodies dropped on migration; retention prunes by age and cap | Unbounded store growth, transcript duplication on disk | `test/turn-advancement-store.test.ts` |
| Fresh empty lock is never stolen (two-process boundary) | `open(wx)` window let a second writer unlink a live lock | `test/store-lock.test.ts` |
| Stale recovery needs age **and** dead owner; EPERM = alive; hard ceiling for PID reuse | Any unreadable lock was treated as stale | `test/store-lock.test.ts` |

## Pillar 2 — Multi-upstream gateway routing

| Behavior | Why stock `main` fails | Test file |
|----------|------------------------|-----------|
| `providerUpstreams` → `x-headroom-base-url` | Single upstream env only | `test/gateway-config.test.ts` |
| Pathname `/v1` normalization | Non-`/v1` providers 404 | `test/gateway-config.test.ts` |
| Gemini `/v1beta` preserved | Google routes need `/v1beta` | `test/gateway-config.test.ts`, `test/proxy-routing` via gateway tests |
| `providerSessionHeaders` UUID | Session-gated APIs 400 | `test/gateway-config.test.ts` |
| Plugin runtime wiring | Config not applied at register | `test/plugin-runtime-routing.test.ts` |

## Pillar 3 — Durable compaction & hygiene

| Behavior | Why stock `main` fails | Test file |
|----------|------------------------|-----------|
| `compact()` rewrites SQLite | Was instant no-op | `test/compaction.test.ts` |
| Truncate fallback on noop | Huge sessions stuck | `test/compaction.test.ts` |
| Truncate starts at a turn boundary; no orphan `toolResult`; whole tool groups | Raw suffix cut mid-turn (41-message reviewer fixture) | `test/compaction.test.ts`, `test/truncate-boundary.test.ts` |
| Truncate uses `resetLeaf()` and reloaded branch is shorter | `branch(parentId)` retained the prefix | `test/compaction.test.ts` |
| `persistentCompaction` modes | Only one behavior before | `test/compaction-mode.test.ts` |
| Hygiene replace-only | N/A on stock | `test/transcript-hygiene.test.ts` |
| Hygiene debounce | Stacked rewrites | `test/hygiene-debounce.test.ts` |
| Projection wait | Race after rewrite | `test/transcript-projection.test.ts` |
| `protect_recent: 2` on durable compress | Was `0` | `test/compaction.test.ts`, `test/compress-request-config.test.ts` |
| Skip replace for image/tool payloads | Permanent multimodal loss | `test/compaction.test.ts`, `test/stress/openclaw-tools.stress.test.ts` |

## Pillar 4 — Assemble performance & routing

| Behavior | Why stock `main` fails | Test file |
|----------|------------------------|-----------|
| Budget short-circuit (~85%) | Every turn hit proxy | `test/engine.test.ts` |
| `assembleCompressConfig` passed to SDK | No per-turn protect_recent | `test/engine.test.ts`, `test/stress/...` |
| Budget skip threshold `(budget − reserve) × ratio`: 130k/200k compresses by default; reserve models a large system prompt; invalid ratio/reserve clamp to defaults; oversized reserve never disables compression | Headroom never ran before OpenClaw's native compaction on ~200k windows | `test/engine.test.ts` |
| `skipAssembleWhenGatewayRouted` (provider-aware) | Double compression; blanket skip would disable compression for direct providers | `test/assemble-skip.test.ts`, `test/engine.test.ts`, `test/tool-call-preservation.integration.test.ts` |
| CCR hint only with hashes | False retrieve spirals | `test/engine.test.ts`, `test/tool-call-preservation.integration.test.ts` |

## Pillar 5 — Tool-call preservation

| Behavior | Why stock `main` fails | Test file |
|----------|------------------------|-----------|
| Image bytes never sent to proxy | Stock dropped them; earlier iteration sent base64 twice | `test/convert.test.ts`, `test/stress/...` (mega no-base64) |
| Image blocks restored with meta echoed **or** stripped | Stripped by `extractText()`; proxy echo unreliable | `test/convert.test.ts`, `test/tool-call-preservation.integration.test.ts` |
| Wire placeholders + text merge | Structure lost on lossy rewrite | `test/content-blocks.test.ts` |
| Original lookup by `tool_call_id` / hint / position | N/A on stock | `test/original-lookup.test.ts` |
| OpenAI `tool.name` from `toolName` | Proxy protect-list miss | `test/convert.test.ts`, `test/stress/...` |
| Deferred `tool_call` wrappers (OpenClaw `{id: "mcp:<server>:<server>__<tool>"}`, Hermes `{name}`) resolved to the real tool name on the wire; wrapper restored from originals | Every MCP tool reached the proxy as `tool_call`, so no per-tool protect/exclude entry could ever match | `test/tool-names.test.ts`, `test/convert.test.ts` |
| `protectToolResults`: matching results restored verbatim after compress; other tools stay compressed; bare / server-prefixed / canonical / glob spellings | Vision and other prose tool outputs were paraphrased by lossy compression | `test/tool-names.test.ts`, `test/convert.test.ts` |
| Assistant thinking/toolCall blocks; no text duplication | Unknown blocks dropped | `test/convert.test.ts` |
| User embedded `tool_result` | Flattened to text | `test/convert.test.ts` |
| `isError` after meta strip | Lost on proxy round-trip | `test/convert.test.ts`, `test/stress/...` |
| Image token estimate | Budget skip defeated by base64 length | `test/convert.test.ts` |
| End-to-end assemble + mock crush | Real failure mode | `test/tool-call-preservation.integration.test.ts` |
| All native OpenClaw tools | Regression breadth | `test/stress/openclaw-tools.stress.test.ts` |
| Live proxy `/v1/compress` | Production validation | `test/live-compress-smoke.mjs`, `test/live-compress-stress.mjs` |

## Pillar 6 — Infrastructure

| Behavior | Test file |
|----------|-----------|
| Proxy manager startup | `test/proxy-manager.test.ts` |
| Message normalization passthrough | `test/engine-normalization.test.ts` |
| Compress config defaults | `test/compress-request-config.test.ts` |

## `test/pr-regression.test.ts`

Single entry point that **asserts stock-main regressions stay fixed** — imports public APIs and runs
smoke checks for each pillar. Failures indicate a reversion of PR value vs upstream.
