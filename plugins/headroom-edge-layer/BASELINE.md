# Phase 0 — baseline setup

Verified against this checkout: commit `d13e1966f820220b482a33c30bde1e926743939a`
(`chore: release 0.39.1 (#3807)`). **Every arm in every test in this protocol must run
against this same commit** — write it down again wherever you record results, since a
Headroom upgrade between arms invalidates the comparison.

Run the baseline on Linux, not Windows: on Windows Headroom falls back to regex content
detection (see `headroom/transforms/content_detector.py`'s watchdog-guarded native-detector
fallback), which would handicap the baseline relative to what production traffic actually
sees.

```bash
git clone https://github.com/burhankhanlodhy/tiktoken headroom
cd headroom
git checkout d13e1966f820220b482a33c30bde1e926743939a

pip install -e ".[all]"
# or: uv sync --extra dev

headroom doctor

HEADROOM_BEACON=off headroom proxy --port 8787 --mode cache
```

`--mode cache` is what freezes the prior-turn prefix and diffs only the newest turn
(`headroom/proxy/handlers/anthropic.py`, `_extract_cache_stable_delta`) — this is the
baseline the edge layer has to beat, not a generic default.

To capture full request bodies (system prompt + tool definitions), which the Claude Code
session logs alone don't carry, run 2-3 sessions through:

```bash
headroom proxy --port 8787 --mode cache --no-optimize --log-messages --log-file /path/to/log.jsonl
```

## What this baseline does NOT need

No GPU, no live spend, no session history from you yet — this step only proves the harness
launches against the pinned commit. The token census (`scripts/census.py`) is the first
thing that needs your own `~/.claude/projects/*/*.jsonl` files, and it's a separate step you
run yourself; see the plugin README.
