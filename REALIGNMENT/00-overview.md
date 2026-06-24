# 00 — Overview & Wrong Mental Model

## Executive summary

Headroom is built on the wrong mental model: **"compression means choosing what to drop from conversation history."** The flagship `IntelligentContextManager` (ICM) tokenizes the entire `messages` array, scores each message for importance, and removes old messages until the budget is hit. It has been wired into the Rust proxy on `/v1/messages` with `frozen_message_count: 0` hardcoded — so every compression event drops messages from index 0, busting the Anthropic prompt cache for every customer that triggers it.

The correct mental model — confirmed by an authoritative engineering guide and ten parallel deep-audit subagents — is the opposite: **"passthrough is sacred; compress only the live zone, type-aware, hash-keyed, position-preserving, with side-channel metadata."** The cache hot zone (system prompt, tools, old turns, reasoning/thinking/redacted/compaction items) is **never** touched.

The audit found:

- **5 top-tier cache-killer bugs** all stemming from the wrong model
- **~10 K LOC of architectural over-build** (ICM + scoring + relevance + rolling-window + progressive-summarizer + tool-crusher + cache-aligner rewrite path + most of `crates/headroom-core/src/{context,scoring,relevance}/`)
- **Wire-format gaps** in the streaming SSE parser (missing `thinking_delta`, `signature_delta`, `citations_delta`; UTF-8-split corruption; single-`\n` SSE split bugs in fallback paths)
- **Bedrock/Vertex parity is fake** — a lossy LiteLLM Anthropic-to-OpenAI conversion drops `thinking`, `redacted_thinking`, `document`, `search_result`, `image`, `server_tool_use`, `mcp_tool_use` blocks
- **No tool-definition normalization** anywhere
- **No auth-mode awareness** — PAYG, OAuth, and subscription CLIs all get the same policy and the same fingerprint-leaking re-serialization
- **`X-Headroom-*` request headers leak upstream**, plus `anthropic-beta` mutation and `OpenAI-Beta` auto-injection — fingerprint-class subscription-revocation risks
- **CCR markers** are computed but never injected into the outgoing request body in the Rust path; the `ccr_retrieve` tool flips on/off per request — busts the tools array on every state change

## What changes

The realignment is structured in 9 phases, 40 PRs, ~13 weeks sequential or ~8 weeks with parallel work:

- **Phase A — Lockdown (1 week):** stop the cache bleeding immediately. Make `/v1/messages` compression a passthrough; stop mutating the system prompt; switch Python forwarders from `httpx ... json=body` (re-serializes) to `httpx ... content=raw_bytes`; honor customer-set `cache_control` markers in Rust; strip `x-headroom-*` from upstream-bound headers; pin `anthropic-beta` order and make it session-sticky; add a SHA-256 byte-faithful round-trip test.
- **Phase B — Live-zone engine (2 weeks):** delete ICM, scoring, relevance, rolling-window, progressive-summarizer, tool-crusher (~10 K LOC). Build a live-zone-only block dispatcher in Rust that runs SmartCrusher / LogCompressor / DiffCompressor / SearchCompressor / KompressCompressor on the latest user message content + latest tool_result + latest function_call_output + latest local_shell_call_output. Token-validate every compression with fallback. CCR hardens: persistent backend + always-on `ccr_retrieve` tool registration.
- **Phase C — Rust proxy paths (3 weeks):** byte-level SSE parser with full state machine; `/v1/chat/completions`, `/v1/responses` (HTTP and streaming) handlers; per-item-type passthrough preservation (V4A patches, `local_shell_call.action.command` argv, Codex `phase` field, MCP items, `compaction`).
- **Phase D — Bedrock/Vertex native (2 weeks):** delete the LiteLLM lossy converter; build native `/model/.../invoke` (AWS) and `/v1beta1/projects/.../publishers/anthropic/.../streamRawPredict` (GCP) routes with SigV4 + ADC signing. Cache fidelity restored on Bedrock/Vertex traffic.
- **Phase E — Phase 3 cache stabilization (1 week):** sort tool array deterministically; sort JSON Schema keys recursively; auto-place up to 4 `cache_control` breakpoints (Anthropic); auto-inject `prompt_cache_key` (OpenAI); volatile-content detector with customer warning (no rewrite); cache-bust drift telemetry.
- **Phase F — Auth-mode policy (1 week):** `classify_auth_mode(headers)` helper returning `payg | oauth | subscription`; per-mode compression policy gates; TOIN aggregation key extended to `(auth_mode, model_family, structure_hash)`; conditional `X-Forwarded-*` headers in Rust.
- **Phase G — RTK + observability (1 week):** extend wrap CLIs (cline, continue, goose, openhands); wire the dead `tokens_saved_rtk` field; per-invocation RTK Prometheus metrics.
- **Phase H — Python retirement (2 weeks):** delete `headroom/proxy/server.py`, all handlers, `responses_converter.py`, `memory_handler.py`, `memory_tool_adapter.py`, `batch.py`, `semantic_cache.py`, all of `headroom/transforms/*` Python (per Phase B); keep CLI wrappers, RTK installer, evals, learn, memory writers, tokenizers, TOIN.
- **Phase I — Test infra (continuous, parallel):** SHA-256 round-trip tests; SSE corner-case fixtures (UTF-8 split, ping, all delta types, `[DONE]`, mid-stream error); property tests (no-panic SSE parser, tokens-non-increasing compression); cache-hit-rate continuous metric; promote `ccr` / `log_compressor` / `cache_aligner` parity comparators from `Skipped` stubs to real; make `make test-parity` a per-PR gate.

## Top 5 wrong assumptions

1. **"Compression means choosing what to drop from history."** Implemented as ICM + DropByScoreStrategy + MessageScorer + relevance + scoring + rolling-window + progressive-summarizer. Fix: retire entirely; compress live-zone content only.
2. **"TOIN can influence per-request compression decisions."** `headroom/telemetry/toin.py:853-927` mutates pattern state during a call and returns hints that bias the same-input-bytes decision. Fix: strict observation-only; recommendations published between deploys.
3. **"CCR can mutate the cache hot zone (tools array, system prompt) on demand."** `headroom/ccr/tool_injection.py:302-328` only adds `ccr_retrieve` when content was compressed — tools list flips between requests. `cache_aligner.py:160-262` and `headroom/proxy/server.py:1051` rewrite the system prompt. Fix: register `ccr_retrieve` on every request; route memory injection to the live zone tail; delete the cache_aligner rewrite path.
4. **"Summarizing past turns is a strategy."** `intelligent_context.py:316-353` SUMMARIZE replaces messages with a single summary at the same position — head modification. Fix: delete; offer compaction only as an explicit customer-initiated action.
5. **"ToolCrusher operates on every tool message in history without a frozen check."** `headroom/transforms/tool_crusher.py:106` iterates all tool messages. Fix: delete; ContentRouter covers the use case correctly.

## What's preserved

Per your direction:
- **TOIN** (Tool Output Intelligence Network) — observation-only refactor; per-tenant key
- **CCR** (Compress-Cache-Retrieve) — persistent backend + always-on tool
- **Kompress-base** — plain-text §8.6 compressor; stays in Python now, Rust port via `ort` crate later
- **ContentRouter** — Python ~2150 LOC, the architecturally correct piece (NOTE: earlier project memory said 53 K lines — that was wrong by 25×; the file is fine)
- All per-type compressors: SmartCrusher (Rust 25 files), CodeCompressor, LogCompressor, SearchCompressor, DiffCompressor

## What's deleted

~25 K LOC across two languages. See [01-bug-list.md](./01-bug-list.md) §6 for the full retirement list with file:line evidence.
