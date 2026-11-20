# OpenClaw plugin tests

## Quick commands

| Command | Purpose |
|---------|---------|
| `npm test` | Full vitest suite (unit + integration + PR regression + stress mocks) |
| `npm run test:stress` | Native-tool mock stress only |
| `npm run test:live-stress` | Live Headroom proxy batch (no LLM); requires proxy on `:8787` |
| `npm run typecheck` | TypeScript validation |
| `npm run build` | Production `dist/` build |

## Layout

| Path | Scope |
|------|-------|
| `pr-regression.test.ts` | **Start here** — one block per PR pillar vs stock `main` |
| `convert.test.ts` | Message conversion & tool-call preservation (wire format, originals-based restore) |
| `content-blocks.test.ts` | Wire placeholders, text/block merge, protected-payload detection |
| `original-lookup.test.ts` | Matching compressed messages back to originals |
| `assemble-skip.test.ts` | Provider-aware `skipAssembleWhenGatewayRouted` |
| `engine.test.ts` | Assemble, compact, circuit breaker, CCR hints |
| `compaction*.test.ts` | Durable SQLite compaction modes |
| `gateway-config.test.ts` | Multi-upstream routing |
| `turn-advancement-store.test.ts` | Durable `commitTurn` persistence |
| `store-lock.test.ts` | Lock protocol: atomic publish, staleness rules, two-process acquire boundary |
| `truncate-boundary.test.ts` | Turn-start selection and orphan `toolResult` removal |
| `tool-call-preservation.integration.test.ts` | Mock proxy + assemble round-trip |
| `stress/openclaw-tools.stress.test.ts` | All native tools under aggressive crush |
| `fixtures/openclaw-native-tools.ts` | Shared transcript builders |
| `live-compress-*.mjs` | Optional live proxy smoke/stress (not vitest) |

Full PR-to-test mapping: [../docs/TEST_MATRIX.md](../docs/TEST_MATRIX.md).
