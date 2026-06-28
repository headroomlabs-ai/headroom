# Headroom Realignment — Index

**Status:** Drafted 2026-05-01 from a 10-agent deep audit against `~/Downloads/llm-proxy-compression-guide.md`.
**Owner:** chopratejas
**Goal:** Move the entire codebase to Rust, preserve prefix cache, retain compression value, integrate RTK end-to-end, and gate compression policy by auth mode (PAYG / OAuth / subscription).

## Read in this order

1. [00-overview.md](./00-overview.md) — executive summary; the wrong mental model; what changes
2. [01-bug-list.md](./01-bug-list.md) — comprehensive ranked bug list with file:line and guide §
3. [02-architecture.md](./02-architecture.md) — the realigned target architecture
4. Phase docs (PR-by-PR, executable):
   - [03-phase-A-lockdown.md](./03-phase-A-lockdown.md) — **start here**: stop-the-bleeding (8 PRs, ~1 week)
   - [04-phase-B-live-zone.md](./04-phase-B-live-zone.md) — live-zone-only compression (7 PRs, ~2 weeks)
   - [05-phase-C-rust-proxy.md](./05-phase-C-rust-proxy.md) — port handlers to Rust (5 PRs, ~3 weeks)
   - [06-phase-D-bedrock-vertex.md](./06-phase-D-bedrock-vertex.md) — native envelopes (4 PRs, ~2 weeks)
   - [07-phase-E-cache-stabilization.md](./07-phase-E-cache-stabilization.md) — Phase 3 stabilization (6 PRs, ~1 week)
   - [08-phase-F-auth-mode.md](./08-phase-F-auth-mode.md) — auth-mode policy gates (4 PRs, ~1 week)
   - [09-phase-G-rtk-observability.md](./09-phase-G-rtk-observability.md) — RTK breadth + metrics (3 PRs, ~1 week)
   - [10-phase-H-python-retirement.md](./10-phase-H-python-retirement.md) — delete Python proxy (3 PRs, ~2 weeks)
   - [11-phase-I-test-infra.md](./11-phase-I-test-infra.md) — test/CI gates (parallel)
5. [12-decisions-needed.md](./12-decisions-needed.md) — open questions

## Conventions

- **Branch name:** `realign-<phase-letter><pr-num>-<slug>`. Example: `realign-A1-icm-passthrough`.
- **Worktree path:** `~/claude-projects/headroom-worktrees/realign-<phase><num>-<slug>`. Use `git worktree add` so each PR is an isolated checkout.
- **Commit prefix:** `fix:` for Rust-migration phase commits (per project memory — `feat:` would inflate semantic-release version).
- **No `Co-Authored-By: Claude` trailer** (per project memory).
- **Pre-push gate:** `make ci-precheck` per project memory; never push without it.

## Phase totals

| Phase | PRs | LOC delta (est.) | Calendar (sequential) |
|---|---:|---:|---:|
| A — Lockdown | 8 | -200 / +400 | 1 week |
| B — Live-zone engine | 7 | **-10,000 / +1,500** | 2 weeks |
| C — Rust proxy paths | 5 | -2,000 / +5,000 | 3 weeks |
| D — Bedrock/Vertex native | 4 | -800 / +2,500 | 2 weeks |
| E — Cache stabilization | 6 | -100 / +900 | 1 week |
| F — Auth-mode policy | 4 | -50 / +600 | 1 week |
| G — RTK + observability | 3 | -50 / +400 | 1 week |
| H — Python retirement | 3 | **-15,000 / +200** | 2 weeks |
| I — Test infra | parallel | +2,000 | continuous |
| **Total** | **40** | **~-28,000 / +13,500** | **~13 weeks** sequential, **~8 weeks** parallel |

## Cross-cutting invariants

These never get violated by any PR:

1. Bytes that the proxy doesn't intend to modify must arrive at upstream **byte-equal** (SHA-256) to bytes that arrived at the proxy. (§1.9)
2. The cache hot zone — system, tools, old turns, reasoning/thinking/redacted/compaction items — is never modified. (§10)
3. Compression is **append-only**: only the live zone (latest user message, latest tool/function/shell/patch outputs) is ever rewritten. (§6.4 + §10.3)
4. Compression is deterministic: same input bytes → same output bytes. (§7.1)
5. Tool definitions are **normalized** (sorted), never compressed. (§8.5)
6. `signature`, `encrypted_content`, `redacted_thinking.data`, `compaction.encrypted_content` are passthrough-only. (§2.7, §2.8, §4.3, §4.8, §10.1)
7. TOIN never alters request-time decisions; it observes and publishes recommendations between deploys. (§7.1, §11.17)
8. CCR markers and the `ccr_retrieve` tool are present **on every request** for a session that ever did CCR — never toggled. (§6.3 #2)
9. `Authorization` header is forwarded byte-faithfully and never logged or persisted unredacted.
10. Auth mode (PAYG / OAuth / subscription) gates compression policy; subscription mode runs in stealth (no `X-Headroom-*` upstream, no beta drift, no UA mutation, no `accept-encoding` strip).

## Preserved primitives (per user direction)

- **TOIN** — refactored to strict observation-only; per-tenant aggregation key.
- **CCR** — hardened with persistent backend + always-on tool registration.
- **Kompress-base** — stays as plain-text compressor (§8.6); Rust port via `ort` later.
- **ContentRouter** — the architecturally correct piece (~2150 LOC); ported to Rust as the live-zone block dispatcher.
- **Type-aware compressors** — SmartCrusher, Code, Log, Search, Diff (already in Rust); kept.
- **`signals/` Rust trait module** — keeps; drives live-zone consumers.
- **`tokenizer/` Rust** — keeps.
- **`safety.rs`** — tool-pair atomicity logic; moved to `transforms/safety.rs` after Phase B.

## Retired (~25K LOC)

- ICM (Python `intelligent_context.py`, Rust `context/manager.rs`)
- `RollingWindow`, `ProgressiveSummarizer`, `scoring.py`, `tool_crusher.py` (Python)
- `crates/headroom-core/src/scoring/`, `relevance/`, most of `context/` (Rust)
- `crates/headroom-proxy/src/compression/icm.rs`
- `headroom/transforms/cache_aligner.py` rewrite path (keep detector + warning)
- `headroom/proxy/server.py`, `handlers/anthropic.py`, `handlers/openai.py`, `handlers/streaming.py`, `handlers/gemini.py`, `responses_converter.py`, `memory_handler.py`, `memory_tool_adapter.py`, `semantic_cache.py`, `batch.py` — once Rust hits parity (Phase H)
- `headroom/backends/litellm.py` Bedrock/Vertex converter — replaced by native envelopes (Phase D)
- MessageScorer Rust port (PR #338, #343) — wasted work; deleted in Phase B
