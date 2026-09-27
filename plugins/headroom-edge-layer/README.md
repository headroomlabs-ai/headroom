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
- `scripts/census_copilot.py` — the same Phase-0 census, but for GitHub
  Copilot Chat (VS Code) sessions, which store data completely differently
  (SQLite-adjacent JSONL under `<workspaceStorage>/<hash>/chatSessions/`, not
  Claude Code's own transcript format). See "Copilot Chat census" below —
  its bucket-level numbers carry a real, documented double-counting risk
  that the Claude Code census doesn't have; the total-cost number does not.
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

## Copilot Chat census

GitHub Copilot Chat (in VS Code) doesn't keep session logs anywhere near
Claude Code's format. Each session is one `.jsonl` file at
`<workspaceStorage>/<hash>/chatSessions/<uuid>.jsonl` (on Windows:
`%APPDATA%\Code\User\workspaceStorage\<hash>\chatSessions\`), but despite the
extension, every *line* is a full re-saved snapshot of the whole session
state, not an appended delta — `census_copilot.py` reads every line and
keeps whichever one has the most requests.

```bash
python3 scripts/census_copilot.py --root "/mnt/c/Users/<you>/AppData/Roaming/Code/User/workspaceStorage"
# native Windows path (not via WSL): %APPDATA%\Code\User\workspaceStorage
```

This was reverse-engineered from one real user's actual session history
(Copilot Chat extension v0.59.0 through v0.65.0), and the schema visibly
moved between versions — `census_copilot.py`'s module docstring has the
full list of what's checked in more than one place because of that. Two
concrete asymmetries versus `census.py`:

- **The total cost figure is solid.** Each request carries real
  `promptTokens`/`outputTokens` (checked in both schema locations that were
  observed), priced from your own account's model catalog, embedded right
  in the session file itself (GitHub's "AI Credits" — 1 AIC = $0.01, per
  GitHub's billing docs) — not a third-party pricing database. Local Ollama
  models (`ollama-models/*`) price at real $0; anything else missing from
  every scanned catalog is flagged as unpriced, never silently $0.
- **The bucket-level split is approximate, not solid.** Tool-call output
  lives in a VS Code-internal rich-text AST
  (`toolCallResults[id].content`), extracted here by recursively collecting
  every `"text"` value found in the tree, and terminal output specifically
  can double-count across schema versions (sometimes a standalone
  `"terminal"` response part, sometimes folded into a tool-invocation
  wrapper's `toolSpecificData`, with no reliable shared ID to dedupe on
  between the two forms). `census_copilot.py` prints this limitation in its
  own output every time — treat the bucket percentages as directional, and
  the total-cost number as the trustworthy one.

Never reads `toolCallRounds[].thinking` — that's Claude's own raw thinking
block content, passed through Copilot's metadata when the underlying model
is Claude, and this script has no legitimate reason to touch it.

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
