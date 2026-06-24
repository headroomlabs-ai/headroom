# 022. Rust Migration

**Status:** in-progress (Stage 0 complete)

Headroom is evolving from a pure-Python proxy into a Rust engine with a Python SDK layer. This document describes the motivation, target architecture, and phased execution plan. It is the source of truth for the migration; the Discord announcement and contributor docs are derived from it.

## Why Rust

Three forces push us toward Rust:

1. **Latency.** Every LLM request flows through compression transforms on the hot path. Rust runs that math 2–10× faster than Python, with negligible interpreter overhead and near-instant cold starts.
2. **Deployment.** A Rust proxy is a single static binary (~10 MB). No Python interpreter, no pip, no wheel matrix, no model downloads at startup. It drops cleanly into containers, serverless runtimes, or bare hosts.
3. **Non-Python consumers.** Integrations like the TypeScript OpenClaw plugin talk to Headroom over HTTP. A faster proxy makes every downstream client faster without changing any client code.

The Python code is not being thrown away. Rust lives alongside Python in the same repository, the HTTP contract stays stable, and existing users see no breaking changes.

## Target architecture

Two new artifacts, built over time:

- **`headroom-proxy`** — a standalone Rust binary that speaks the existing Headroom HTTP API and contains native implementations of the proxy's hot-path logic (routing, streaming, compression, telemetry). This is the deployable artifact.
- **`headroom-core`** — a Rust library with the compression transforms (CCR, log compressor, diff compressor, tokenizer, code compressor, content router, etc.). Consumed by `headroom-proxy` directly; optionally exposed to Python via a PyO3 binding if/when embedded SDK use warrants it.

Both live in a Cargo workspace under `crates/` at the repo root. They are built, tested, and released together with the Python package.

## Migration strategy

**Trunk-based, no long-lived branch.** Every Rust change ships as a small PR to `main`. Rust and Python live side by side. CI enforces parity on every PR — if the Rust port diverges from the Python reference, the build fails.

**Proxy-first, not transforms-first.** We build the Rust proxy binary as the primary deliverable, starting from a pure HTTP passthrough that forwards upstream to the existing Python proxy. Native Rust implementations of individual routes then replace passthroughs one at a time, gated by feature flags. This lets us ship a deployable Rust binary on day one and iterate without modifying Python internals.

**Feature-flagged cutover.** Each native Rust route is toggled by config. Default is passthrough-to-Python until a route has been shadow-tested and validated. Rollback is a flag flip, not a redeploy.

**Parity by shadow traffic.** Recorded input/output fixtures give us a unit-test-level parity check. Shadow mode — running both proxies on live traffic and diffing outputs — gives us the real validation gate before any cutover.

## Stages

### Stage 0 — Foundation ✅

Cargo workspace with four crates (`headroom-core`, `headroom-proxy`, `headroom-py`, `headroom-parity`), CI, build tooling (Makefile, GitHub Actions), and a parity test harness seeded with 125 recorded fixtures across 5 leaf transforms. No production behavior changed.

### Stage 1 — Rust proxy as passthrough

`headroom-proxy` accepts requests on the same HTTP contract as the Python proxy and forwards everything upstream to Python. No transforms yet, no intelligence. The point is a deployable binary that can run in front of the existing stack with zero risk.

### Stage 2 — First native route

Replace the passthrough for `/v1/chat/completions` (OpenAI) with a native Rust implementation: Rust transforms, direct provider call, streamed response. Feature-flagged. Python proxy handles everything else unchanged.

### Stage 3 — Shadow mode validation

Run both proxies on real traffic. Diff outputs with tolerance for chunk timing (SSE). Gate: one week of ≤ 0.1% content divergence before flipping the flag for real.

### Stage 4 — Provider expansion

Anthropic, Google, Cohere, Mistral, Bedrock, others. Each provider goes through its own shadow period before cutover.

### Stage 5 — Code-aware transforms

Port `code_compressor` (tree-sitter, already Rust-native upstream), `content_router`, `content_detector`. These are larger transforms but have no ML dependencies.

### Stage 6 — Storage layer

SQLite (`rusqlite` + `sqlite-vec`), HNSW vector index (`instant-distance`), graph store (`neo4rs`). Feature-flagged backend for the memory subsystem.

### Stage 7 — ONNX migration

Remove the torch-dependent LLMLingua compressor. Convert remaining ML models (SmartCrusher, IntelligentContext, memory embedders) to ONNX with fixed opset. Run via the `ort` crate. Remove `torch`, `transformers`, `sentence-transformers`, `llmlingua` from runtime dependencies.

### Stage 8 — Retire the Python proxy

Delete `headroom/proxy/server.py` and the Python HTTP routes. The Python package (`import headroom`) continues to exist for SDK users and integration adapters (LangChain, Agno, MCP, Strands), which talk HTTP and don't care which proxy implementation is behind the endpoint.

## What does not change

- `pip install headroom` continues to work.
- The HTTP API contract is preserved.
- LangChain, Agno, MCP, Strands integrations keep working unchanged.
- The CLI, dashboard, eval framework, examples, and documentation site are all unaffected.
- Python contributions remain welcome for integrations, tooling, examples, and any code paths not yet ported.

## Contributing

- **Python contributors** do not need to learn Rust. CI will flag a PR if a Python change diverges from a Rust-ported counterpart; in that case either update both implementations or disable the feature flag for the affected route.
- **Rust contributors** should read `RUST_DEV.md` for workspace setup, then pick a transform or proxy route from the open issues.

## Risks and open questions

- **ONNX export parity** for embedding models: numerical reproducibility must be validated per model before cutover. Some models may resist clean export and require keeping a Python inference path behind PyO3 as a fallback.
- **LLMLingua removal** is a feature removal visible to users relying on it; deprecation timing will be announced on Discord before Stage 7 begins.
- **PyO3 binding scope**: we may ultimately not need a PyO3-exposed `headroom._core` at all, if existing Python SDK users are happy with the HTTP contract. Decision deferred to Stage 8.

## References

- `RUST_DEV.md` — developer setup and workspace reference
- `crates/` — Rust sources
- `tests/parity/` — fixtures and parity harness
- `Makefile` — `make test`, `make test-parity`, `make build-proxy`, `make build-wheel`, `make fmt`, `make lint`
