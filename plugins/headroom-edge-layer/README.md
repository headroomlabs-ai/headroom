# headroom-edge-layer

Implements the "Compression SaaS — Edge Plan & Test Protocol": a
`headroom.pipeline_extension` that compresses content Headroom's engine skips
by default (file reads, grep, web pages), using the agent's own latest
reasoning as the relevance signal, plus the census/replay tooling used to
prove or kill the idea in dollars per solved task before building a hosted
product around it.

**Status: work in progress, built against commit
`d13e1966f820220b482a33c30bde1e926743939a`.** See `BASELINE.md` for the
pinned-commit setup, and the plan doc for the full protocol.

## What's here

- `src/headroom_edge_layer/` — the pipeline extension itself:
  - `intent_query.py` — edge 1, no model.
  - `grep_edge.py` — edge 3, with the enumeration guard.
  - `file_read_edge.py` — edges 4a/4b (repeat reads, post-edit diffs). Edge 4c
    (outlines for 800+ line files) is deliberately not built — gated behind a
    live A/B this sandbox can't run.
  - `rerank_edge.py` — edge 2. Ships with a deterministic token-overlap
    `Scorer` as a wiring stand-in; real next-step-recall numbers need the
    `HttpRerankerScorer` pointed at a real reranker (e.g. a 0.6B model on a
    local GPU), not this fallback.
  - `cache_safety.py` — the shared rules every edge is built on: marker
    formats the engine already recognizes, config-scoped hashing, TTL
    refresh, CPU-side size caps in place of an unenforceable timeout.
  - `pipeline_extension.py` — wires the edges together as
    `EdgeLayerExtension`, the `headroom.pipeline_extension` entry point.
- `scripts/census.py` — Phase 0's token census over your own Claude Code
  session logs. **You run this yourself** against `~/.claude/projects/*/*.jsonl`
  — this sandbox has no real session history to test against.
- `replay/` — Phase 1's offline replay harness, and Phase 3's *unexecuted*
  agent-runner scaffold (no live API calls are made by anything in this repo).

## Enabling it

Installing this package does **not** turn it on — pipeline extensions are
opt-in:

```bash
pip install -e plugins/headroom-edge-layer
HEADROOM_PIPELINE_EXTENSIONS=edge-layer headroom proxy --port 8787 --mode cache
```

Per-edge flags (`HEADROOM_EDGE_INTENT_QUERY`, `HEADROOM_EDGE_GREP`,
`HEADROOM_EDGE_FILE_READ`, `HEADROOM_EDGE_RERANK`) default to on; set any to
`0`/`false`/`off` to isolate one edge at a time in the replay harness, per the
protocol's "one flag at a time, then all together" scoring.

## Known limitations (see the plan doc's corrections section for detail)

- Only covers Anthropic and OpenAI chat-completions traffic — OpenAI
  Responses API (Codex) and Gemini don't fire `INPUT_RECEIVED` in this
  checkout, so that traffic bypasses the edge layer entirely.
- Never touches `thinking`/`redacted_thinking` blocks, by construction — a
  request with signed thinking blocks that doesn't survive mutation gets
  forwarded byte-for-byte by `headroom/proxy/body_forwarding.py`, silently
  discarding any edit, so the edges are scoped to only ever read/write
  `tool_result` and read-only `tool_use`/text content.
- The rerank edge's real quality depends entirely on which `Scorer` is
  wired in; the shipped fallback is for plumbing tests only.
